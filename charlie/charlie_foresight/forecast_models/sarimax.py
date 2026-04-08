import numpy as np
import pandas as pd
from typing import Optional, Dict
import optuna
import logging
import statsmodels.api as sm
from charlie.charlie_foresight.forecast_models.base import BaseModel


class SeasonalARIMAModel(BaseModel):
    """A forecasting model using Seasonal ARIMA (SARIMA).

    This class implements SARIMA (Seasonal ARIMA) forecasting with automatic hyperparameter
    tuning using Optuna. It can detect seasonality and optimize both non-seasonal and
    seasonal ARIMA orders.

    Attributes:
        seasonal_period (Optional[int]): The seasonal period for the model.
        order_ranges (dict): Ranges for ARIMA order parameters (p, d, q, P, D, Q).
        sarimax_kwargs (dict): Additional keyword arguments for SARIMAX model.
        fit_kwargs (dict): Keyword arguments for model fitting.
        optuna_kwargs (dict): Keyword arguments for Optuna optimization.
        best_params (Optional[Dict]): Best parameters found by hyperparameter search.
        model: The fitted SARIMAX model instance.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> train_data = pd.DataFrame({'y': np.random.randn(100)}, index=pd.date_range('2020-01-01', periods=100))
        >>> model = SeasonalARIMAModel(seasonal_period=12)
        >>> model.fit(train_data)
        >>> future = pd.DataFrame(index=pd.date_range('2020-04-11', periods=10))
        >>> preds, lower, upper = model.predict(future)
        >>> preds.shape
        (10,)
    """

    def __init__(
        self,
        *,
        seasonal_period: Optional[int] = None,
        order_ranges: Optional[dict] = None,
        sarimax_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
        optuna_kwargs: Optional[dict] = None,
    ):
        """Initializes the SeasonalARIMAModel.

        Args:
            seasonal_period (Optional[int], optional): Seasonal period for the model. Defaults to None.
            order_ranges (Optional[dict], optional): Ranges for ARIMA parameters. Defaults to standard ranges.
            sarimax_kwargs (Optional[dict], optional): Additional SARIMAX kwargs. Defaults to None.
            fit_kwargs (Optional[dict], optional): Fitting kwargs. Defaults to None.
            optuna_kwargs (Optional[dict], optional): Optuna optimization kwargs. Defaults to None.
        """
        self.seasonal_period = seasonal_period
        self.order_ranges = order_ranges or {
            "p": (0, 2), "d": (0, 1), "q": (0, 2),
            "P": (0, 2), "D": (0, 1), "Q": (0, 2),
        }
        self.sarimax_kwargs = sarimax_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.optuna_kwargs = optuna_kwargs or {}

        self.best_params: Optional[Dict] = None
        self.model = None

    def _objective(self, trial, train_df, val_df):
        y = train_df.iloc[:, 0]
        s = self.seasonal_period
        n = len(y)

        seasonal_allowed = s is not None and n >= 2 * s

        p = trial.suggest_int("p", *self.order_ranges["p"])
        d = trial.suggest_int("d", *self.order_ranges["d"])
        q = trial.suggest_int("q", *self.order_ranges["q"])

        if seasonal_allowed:
            P = trial.suggest_int("P", *self.order_ranges["P"])
            D = trial.suggest_int("D", *self.order_ranges["D"])
            Q = trial.suggest_int("Q", *self.order_ranges["Q"])
            seasonal_order = (P, D, Q, s)
        else:
            seasonal_order = (0, 0, 0, 0)

        try:
            m = sm.tsa.SARIMAX(
                y,
                order=(p, d, q),
                seasonal_order=seasonal_order,
                **self.sarimax_kwargs,
            )
            res = m.fit(disp=False, **self.fit_kwargs)
            pred = res.get_forecast(steps=len(val_df)).predicted_mean.values
            return float(np.mean((val_df.iloc[:, 0].values - pred) ** 2))
        except Exception:
            return float("inf")

    def hyperparameter_search(self, train_df, splits):
        """Performs hyperparameter search using Optuna with cross-validation.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.
            splits: Cross-validation splits (e.g., from sklearn.model_selection.TimeSeriesSplit).
        """
        n_trials = int(self.optuna_kwargs.get("n_trials", 30))
        study = optuna.create_study(direction="minimize")

        for tr_idx, va_idx in splits:
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            study.optimize(lambda t: self._objective(t, tr, va), n_trials=n_trials)

        self.best_params = study.best_params

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits the SARIMA model to the training data.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.
        """
        if self.best_params is None:
            self.best_params = {"p": 1, "d": 0, "q": 1}

        p = self.best_params["p"]
        d = self.best_params["d"]
        q = self.best_params["q"]

        s = self.seasonal_period
        seasonal_order = (0, 0, 0, 0)

        if s and any(k in self.best_params for k in ("P", "D", "Q")):
            seasonal_order = (
                self.best_params.get("P", 0),
                self.best_params.get("D", 0),
                self.best_params.get("Q", 0),
                s,
            )

        m = sm.tsa.SARIMAX(
            train_df.iloc[:, 0],
            order=(p, d, q),
            seasonal_order=seasonal_order,
            **self.sarimax_kwargs,
        )
        self.model = m.fit(disp=False, **self.fit_kwargs)

    def predict(self, future_df: pd.DataFrame):
        """Makes predictions using the fitted SARIMA model.

        Args:
            future_df (pd.DataFrame): Future data for prediction with datetime index.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of predictions, lower confidence intervals,
                and upper confidence intervals.
        """
        fc = self.model.get_forecast(steps=len(future_df))
        ci = fc.conf_int(alpha=0.05)

        def _as_array(x):
            return x.values if hasattr(x, "values") else np.asarray(x)

        return (
            _as_array(fc.predicted_mean),
            _as_array(ci.iloc[:, 0]),
            _as_array(ci.iloc[:, 1]),
        )