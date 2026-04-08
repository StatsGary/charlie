import pandas as pd
from typing import Optional, Dict, Any, Callable
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from charlie.charlie_foresight.time_index import infer_seasonal_period, make_future_index
from charlie.charlie_foresight.preprocessors import TimePreprocessor
from charlie.charlie_foresight.transforms import SeriesTransformer
from charlie.charlie_foresight.cv import CrossValidator
from charlie.charlie_foresight.ensemble import Ensembler
from charlie.charlie_foresight.metrics import mase
from charlie.charlie_foresight.forecast_models.sarimax import SeasonalARIMAModel
from charlie.charlie_foresight.forecast_models.ets import ETSModel
import optuna
import logging
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from charlie_foresight.forecast_models.prophet import ProphetModel
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False


def _get_tqdm():
    """
    Return a tqdm implementation that NEVER crashes:
    - Uses notebook tqdm only if ipywidgets is available
    - Falls back cleanly otherwise
    """
    # --- Try notebook tqdm safely ---
    try:
        from IPython import get_ipython
        ip = get_ipython()

        if ip is not None and "IPKernelApp" in ip.config:
            try:
                from tqdm.notebook import tqdm
                # Touch tqdm to force widget init
                _ = tqdm(total=0)
                return tqdm
            except Exception:
                pass
    except Exception:
        pass

    # --- Try standard tqdm ---
    try:
        from tqdm import tqdm
        return tqdm
    except Exception:
        pass

    # --- Absolute fallback (no-op) ---
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

    return tqdm


def select_models_by_relative_error(
    cv_results: Dict[str, list],
    *,
    max_relative_error: float = 1.25,
    metric_fn: Callable = mase,
) -> Dict[str, list]:
    """Selects models based on relative error compared to the best performing model.

    Keeps only models whose mean cross-validation error is within a specified
    multiple of the best model's error. This helps filter out poorly performing models.

    Args:
        cv_results (Dict[str, list]): Cross-validation results from CrossValidator.evaluate.
        max_relative_error (float, optional): Maximum allowed error as multiple of best error. Defaults to 1.25.
        metric_fn (Callable, optional): Metric function for evaluating performance. Defaults to mase.

    Returns:
        Dict[str, list]: Filtered cross-validation results containing only selected models.

    Examples:
        >>> # cv_results = cv.evaluate(data, models)
        >>> selected = select_models_by_relative_error(cv_results, max_relative_error=1.5)
        >>> len(selected) <= len(cv_results)
        True
    """
    mean_errors = {}

    for model, folds in cv_results.items():
        scores = [
            metric_fn(f["actual"], f["pred"])
            for f in folds
            if f.get("pred") is not None
        ]
        if scores:
            mean_errors[model] = sum(scores) / len(scores)

    if not mean_errors:
        return cv_results

    best_error = min(mean_errors.values())

    selected = {
        model: cv_results[model]
        for model, err in mean_errors.items()
        if err <= best_error * max_relative_error
    }

    if not selected:
        best_model = min(mean_errors, key=mean_errors.get)
        selected = {best_model: cv_results[best_model]}

    return selected


def make_forecast_plotter(
    *,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    forecast,
    lower=None,
    upper=None,
):
    """Creates a plotting function for visualizing forecast results.

    Returns a callable function that generates a seaborn-based plot showing
    historical data, forecast, and confidence intervals.

    Args:
        history (pd.Series): Historical time series data.
        future_index (pd.DatetimeIndex): Datetime index for forecast period.
        forecast: Array-like forecast values.
        lower: Array-like lower confidence interval values (optional).
        upper: Array-like upper confidence interval values (optional).

    Returns:
        Callable: A plotting function with parameters for customization.

    Examples:
        >>> import pandas as pd
        >>> history = pd.Series([1, 2, 3], index=pd.date_range('2020-01-01', periods=3))
        >>> future_index = pd.date_range('2020-01-04', periods=2)
        >>> forecast = [4, 5]
        >>> plot_fn = make_forecast_plotter(history=history, future_index=future_index, forecast=forecast)
        >>> # plot_fn()  # Displays the forecast plot
    """

    def plot(
        *,
        title: str = "Charlie Foresight – Forecast",
        figsize=(12, 5),
        show_confidence: bool = True,
        history_label: str = "History",
        forecast_label: str = "Forecast (Ensemble)",
    ):
        

        sns.set_theme(style="whitegrid")

        # --- SAFE history dataframe (NO reset_index) ---
        history_df = pd.DataFrame({
            "date": history.index,
            "value": history.to_numpy(),
        })

        # --- Build forecast continuation ---
        last_date = history.index[-1]
        last_value = history.iloc[-1]

        forecast_dates = [last_date] + list(future_index)
        forecast_values = [last_value] + list(np.asarray(forecast, dtype=float))

        forecast_df = pd.DataFrame({
            "date": pd.to_datetime(forecast_dates),
            "value": forecast_values,
        })

        # --- Confidence intervals (optional) ---
        lower_plot = upper_plot = None
        if show_confidence and lower is not None and upper is not None:
            lower_plot = [last_value] + list(np.asarray(lower, dtype=float))
            upper_plot = [last_value] + list(np.asarray(upper, dtype=float))

        # --- Defensive validation (prevents seaborn cryptic errors) ---
        for name, df in {
            "history_df": history_df,
            "forecast_df": forecast_df,
        }.items():
            if not {"date", "value"}.issubset(df.columns):
                raise RuntimeError(
                    f"{name} malformed — columns found: {df.columns.tolist()}"
                )

        # --- Plot ---
        plt.figure(figsize=figsize)

        sns.lineplot(
            data=history_df,
            x="date",
            y="value",
            label=history_label,
            linewidth=2,
        )

        sns.lineplot(
            data=forecast_df,
            x="date",
            y="value",
            label=forecast_label,
            linewidth=2,
        )

        if lower_plot is not None:
            plt.fill_between(
                forecast_df["date"],
                lower_plot,
                upper_plot,
                alpha=0.25,
                label="Uncertainty",
            )

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return plot


def _suppress_optuna_logging():
    """
    Suppress Optuna logging and progress output.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

def _suppress_statsmodels_warnings():
    """
    Suppress common, non-fatal statsmodels warnings
    that are expected during SARIMAX fitting.
    """
    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="statsmodels",
    )

def _suppress_prophet_logging():
    """
    Suppress Prophet INFO-level logs.
    """
    logging.getLogger("prophet").setLevel(logging.WARNING)

def run_forecast(
    *,
    series: pd.Series,
    horizon: int,
    seasonal_period: Optional[int] = None,
    transformer_kwargs: Optional[dict] = None,
    preprocessor_kwargs: Optional[dict] = None,
    sarimax_kwargs: Optional[dict] = None,
    sarimax_fit_kwargs: Optional[dict] = None,
    sarimax_optuna_kwargs: Optional[dict] = None,
    ets_kwargs: Optional[dict] = None,
    ets_fit_kwargs: Optional[dict] = None,
    prophet_kwargs: Optional[dict] = None,
    prophet_fit_kwargs: Optional[dict] = None,
    cv_kwargs: Optional[dict] = None,
    ensemble_kwargs: Optional[dict] = None,
    model_selector: Optional[Callable[[Dict[str, list]], Dict[str, list]]] = None,
    model_factories: Optional[Dict[str, Callable[[], Any]]] = None,
    return_models: bool = True,
    forecast_warn_suppress: bool = True
) -> Dict[str, Any]:
    """
    End-to-end forecasting pipeline with automatic model selection and ensembling.
    
    Executes a complete forecasting workflow including data preprocessing, 
    cross-validation across multiple models (SARIMAX, ETS, Prophet), adaptive 
    ensemble weighting, and forecast generation with confidence intervals.
    
    Args:
        series: Time series data with DatetimeIndex to forecast.
        horizon: Number of periods to forecast ahead.
        seasonal_period: Seasonality period for models. If None, automatically inferred
            from the series index frequency.
        transformer_kwargs: Keyword arguments passed to SeriesTransformer for 
            scaling/normalization. Defaults to {}.
        preprocessor_kwargs: Keyword arguments passed to TimePreprocessor. Defaults to {}.
        sarimax_kwargs: Keyword arguments passed to SeasonalARIMAModel constructor.
        sarimax_fit_kwargs: Keyword arguments passed to SARIMAX model.fit().
        sarimax_optuna_kwargs: Keyword arguments for Optuna hyperparameter optimization
            in SARIMAX.
        ets_kwargs: Keyword arguments passed to ETSModel constructor.
        ets_fit_kwargs: Keyword arguments passed to ETS model.fit().
        prophet_kwargs: Keyword arguments passed to ProphetModel constructor.
        prophet_fit_kwargs: Keyword arguments passed to Prophet model.fit().
        cv_kwargs: Keyword arguments passed to CrossValidator. Defaults to 
            {"n_splits": 3}.
        ensemble_kwargs: Keyword arguments passed to Ensembler. Defaults to {}.
        model_selector: Optional callable to filter models after cross-validation.
            Receives cv_results dict and returns filtered dict. If None, all models
            are used in the ensemble.
        model_factories: Optional dict mapping model names to callable factories that 
            return model instances. If None, default models (SARIMAX, ETS, Prophet 
            if available) are used.
        return_models: If True, fitted models are included in the return dict.
        forecast_warn_suppress: If True, suppresses logs/warnings from Optuna, 
            statsmodels, and Prophet during fitting.
    
    Returns:
        Dictionary containing:
            - 'forecast': Array of ensemble forecast values (original scale).
            - 'lower': Lower confidence interval (original scale, None if unavailable).
            - 'upper': Upper confidence interval (original scale, None if unavailable).
            - 'future_index': DatetimeIndex for forecast periods.
            - 'ensemble_weights': Dict mapping model names to ensemble weights.
            - 'cv_results': Cross-validation results from model evaluation.
            - 'plot': Callable plotting function (no args required).
            - 'preprocessor': TimePreprocessor instance for inverse transformations.
            - 'models': Dict of fitted models (only if return_models=True).
    
    Raises:
        ValueError: If series does not have a DatetimeIndex.
    
    Example:
        >>> import pandas as pd
        >>> # Create a sample time series
        >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
        >>> values = 10 + 2 * (dates.dayofyear % 30) + pd.Series(range(100))
        >>> series = pd.Series(values, index=dates)
        >>> 
        >>> # Run full forecasting pipeline
        >>> result = run_forecast(
        ...     series=series,
        ...     horizon=30,
        ...     cv_kwargs={"n_splits": 3},
        ...     sarimax_optuna_kwargs={"n_trials": 10},
        ...     forecast_warn_suppress=True
        ... )
        >>> 
        >>> # Access results
        >>> print(result['forecast'])  # 30-period forecast
        >>> print(result['ensemble_weights'])  # Model weights
        >>> result['plot']()  # Display forecast visualization
        >>> 
        >>> # Use preprocessor for custom transformations
        >>> scaled_data = result['preprocessor'].transform(series.to_frame('target'))
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")


    tqdm = _get_tqdm()

    if seasonal_period is None:
        seasonal_period = infer_seasonal_period(series.index)

    transformer = SeriesTransformer(**(transformer_kwargs or {}))
    preprocessor = TimePreprocessor(
        transformer=transformer,
        **(preprocessor_kwargs or {})
    )

    df = series.to_frame("target")
    df_scaled = preprocessor.fit_transform(df)

    if model_factories is None:
        model_factories = {
            "sarimax": lambda: SeasonalARIMAModel(
                seasonal_period=seasonal_period,
                sarimax_kwargs=sarimax_kwargs or {},
                fit_kwargs=sarimax_fit_kwargs or {},
                optuna_kwargs=sarimax_optuna_kwargs or {},
            ),
            "ets": lambda: ETSModel(
                seasonal_periods=seasonal_period,
                ets_kwargs=ets_kwargs or {},
                fit_kwargs=ets_fit_kwargs or {},
            ),
        }

        if _HAS_PROPHET:
            model_factories["prophet"] = lambda: ProphetModel(
                prophet_kwargs=prophet_kwargs or {},
                fit_kwargs=prophet_fit_kwargs or {},
            )

    models = {}

    for name, factory in tqdm(
        model_factories.items(),
        desc="Initialising models",
        leave=False):
        models[name] = factory()


    if forecast_warn_suppress:
        _suppress_optuna_logging()
        _suppress_statsmodels_warnings()
        _suppress_prophet_logging()

    
    cv = CrossValidator(**(cv_kwargs or {"n_splits": 3}))
    cv_results = cv.evaluate(df_scaled, models)

    if model_selector is not None:
        cv_results = model_selector(cv_results)
        models = {name: models[name] for name in cv_results}

    ensembler = Ensembler(**(ensemble_kwargs or {}))
    ensembler.fit(cv_results)

    future_index = make_future_index(series.index, horizon)
    future_df = pd.DataFrame(index=future_index)

    preds, lowers, uppers = {}, {}, {}

    for name, model in tqdm(
      models.items(),
      desc="Fitting & forecasting models",
      leave=False,):
      model.fit(df_scaled)
      p_s, lo_s, hi_s = model.predict(future_df)

      preds[name] = preprocessor.inverse_transform(p_s)
      lowers[name] = preprocessor.inverse_transform(lo_s) if lo_s is not None else None
      uppers[name] = preprocessor.inverse_transform(hi_s) if hi_s is not None else None

    forecast, lower, upper = ensembler.combine(preds, lowers, uppers)

    plot_fn = make_forecast_plotter(
        history=series,
        future_index=future_index,
        forecast=forecast,
        lower=lower,
        upper=upper,
    )

    result = {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "future_index": future_index,
        "ensemble_weights": ensembler.weights,
        "cv_results": cv_results,
        "plot": plot_fn,
        "preprocessor": preprocessor,
    }

    if return_models:
        result["models"] = models

    return result
