from __future__ import annotations
import pandas as pd
import pytest
import charlie.charlie_foresight.forecast_runner as m


class FakePreprocessor:
    """Mimics TimePreprocessor enough for run_forecast."""
    def __init__(self, transformer=None, **kwargs):
        self.transformer = transformer
        self.kwargs = kwargs
        self.fit_transform_called = False
        self.inverse_transform_calls = 0

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit_transform_called = True
        return df.copy()

    def inverse_transform(self, obj):
        """
        run_forecast passes:
          - p_s, lo_s, hi_s from model.predict(...)
        We'll just return the underlying data unchanged.
        """
        self.inverse_transform_calls += 1
        return obj


class FakeCV:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.evaluate_called = False
        self.evaluate_args = None

    def evaluate(self, df_scaled: pd.DataFrame, models: dict):
        self.evaluate_called = True
        self.evaluate_args = (df_scaled, models)

        out = {}
        for name in models:
            out[name] = [{"actual": [1, 2, 3], "pred": [1, 2, 3]}]
        return out


class FakeEnsembler:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_called = False
        self.combine_called = False
        self.weights = {}

    def fit(self, cv_results: dict):
        self.fit_called = True
        keys = list(cv_results.keys())
        self.weights = {k: 1.0 / len(keys) for k in keys} if keys else {}

    def combine(self, preds: dict, lowers: dict, uppers: dict):
        self.combine_called = True
        first = next(iter(preds.keys()))
        forecast = preds[first]
        lower = lowers[first]
        upper = uppers[first]
        return forecast, lower, upper


class FakeModel:
    """Mimics model interface used by run_forecast."""
    def __init__(self, *, with_intervals: bool):
        self.with_intervals = with_intervals
        self.fit_called = False
        self.predict_called = False

    def fit(self, df_scaled: pd.DataFrame):
        self.fit_called = True

    def predict(self, future_df: pd.DataFrame):
        self.predict_called = True
        n = len(future_df.index)
        pred = [10] * n
        if self.with_intervals:
            lo = [9] * n
            hi = [11] * n
        else:
            lo = None
            hi = None
        return pred, lo, hi


@pytest.fixture
def series_daily():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.Series(range(10), index=idx, name="y")


def _patch_pipeline(monkeypatch, *, future_index: pd.DatetimeIndex):
    """
    Patch run_forecast dependencies inside module m.
    Keeps tests fast and deterministic.
    """
    monkeypatch.setattr(m, "_get_tqdm", lambda: (lambda it=None, **kwargs: it if it is not None else []))
    monkeypatch.setattr(m, "infer_seasonal_period", lambda idx: 7)
    monkeypatch.setattr(m, "make_future_index", lambda idx, horizon: future_index)
    monkeypatch.setattr(m, "make_forecast_plotter", lambda **kwargs: (lambda **k: None))
    monkeypatch.setattr(m, "TimePreprocessor", FakePreprocessor)
    monkeypatch.setattr(m, "CrossValidator", FakeCV)
    monkeypatch.setattr(m, "Ensembler", FakeEnsembler)



def test_run_forecast_requires_datetimeindex():
    s = pd.Series([1, 2, 3], index=[0, 1, 2])  
    with pytest.raises(ValueError, match="DatetimeIndex"):
        m.run_forecast(series=s, horizon=3)


def test_run_forecast_happy_path_two_models_intervals_and_no_intervals(monkeypatch, series_daily):
    horizon = 5
    future_index = pd.date_range(series_daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    _patch_pipeline(monkeypatch, future_index=future_index)

    model_factories = {
        "with_ci": lambda: FakeModel(with_intervals=True),
        "no_ci": lambda: FakeModel(with_intervals=False),
    }

    out = m.run_forecast(
        series=series_daily,
        horizon=horizon,
        seasonal_period=None,         
        model_factories=model_factories,
        return_models=True,
        forecast_warn_suppress=False,  
    )

    # Core keys present
    assert set(out.keys()) >= {
        "forecast", "lower", "upper", "future_index",
        "ensemble_weights", "cv_results", "plot", "preprocessor", "models"
    }

    assert out["future_index"].equals(future_index)
    assert len(out["forecast"]) == horizon

    assert out["forecast"] == [10] * horizon
    assert out["lower"] == [9] * horizon
    assert out["upper"] == [11] * horizon
    assert set(out["ensemble_weights"].keys()) == {"with_ci", "no_ci"}

    assert set(out["models"].keys()) == {"with_ci", "no_ci"}
    assert out["models"]["with_ci"].fit_called is True
    assert out["models"]["with_ci"].predict_called is True
    assert out["models"]["no_ci"].fit_called is True
    assert out["models"]["no_ci"].predict_called is True

    # Plot is callable
    assert callable(out["plot"])


def test_run_forecast_model_selector_filters_models(monkeypatch, series_daily):
    horizon = 3
    future_index = pd.date_range(series_daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    _patch_pipeline(monkeypatch, future_index=future_index)

    model_factories = {
        "keep": lambda: FakeModel(with_intervals=True),
        "drop": lambda: FakeModel(with_intervals=True),
    }

    def selector(cv_results: dict) -> dict:
        # Keep only one model
        return {"keep": cv_results["keep"]}

    out = m.run_forecast(
        series=series_daily,
        horizon=horizon,
        model_factories=model_factories,
        model_selector=selector,
        return_models=True,
        forecast_warn_suppress=False,
    )

    assert set(out["cv_results"].keys()) == {"keep"}
    assert set(out["ensemble_weights"].keys()) == {"keep"}
    assert set(out["models"].keys()) == {"keep"}  


def test_run_forecast_when_all_preds_have_no_intervals_returns_none_bounds(monkeypatch, series_daily):
    horizon = 4
    future_index = pd.date_range(series_daily.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    _patch_pipeline(monkeypatch, future_index=future_index)

    model_factories = {
        "m1": lambda: FakeModel(with_intervals=False),
        "m2": lambda: FakeModel(with_intervals=False),
    }

    out = m.run_forecast(
        series=series_daily,
        horizon=horizon,
        model_factories=model_factories,
        return_models=False,         
        forecast_warn_suppress=False,
    )

    assert "models" not in out
    assert len(out["forecast"]) == horizon
    assert out["lower"] is None
    assert out["upper"] is None
