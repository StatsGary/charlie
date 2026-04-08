import numpy as np
import pandas as pd
import pytest
from charlie.charlie_foresight.inference import forecast_unseen

class FakePreprocessor:
    def __init__(self):
        self.transform_called = False
        self.inverse_transform_called = False

    def transform(self, df):
        self.transform_called = True
        return df.copy()

    def inverse_transform(self, arr):
        self.inverse_transform_called = True
        return np.asarray(arr)


class FakeModel:
    def __init__(self, value):
        self.value = value
        self.predict_called = False

    def predict(self, future_df):
        self.predict_called = True
        n = len(future_df)
        return np.full(n, self.value), None, None


def test_forecast_unseen_requires_datetimeindex():
    s = pd.Series([1, 2, 3], index=[0, 1, 2])

    with pytest.raises(ValueError, match="DatetimeIndex"):
        forecast_unseen(
            series=s,
            horizon=3,
            models={},
            weights={},
            preprocessor=FakePreprocessor(),
        )

def test_forecast_unseen_single_model():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    series = pd.Series(range(5), index=idx)

    model = FakeModel(value=10)
    pre = FakePreprocessor()

    out = forecast_unseen(
        series=series,
        horizon=3,
        models={"m": model},
        weights={"m": 1.0},
        preprocessor=pre,
    )

    assert np.all(out["forecast"] == np.array([10, 10, 10]))
    assert len(out["future_index"]) == 3
    assert "m" in out["individual_forecasts"]

    assert model.predict_called is True
    assert pre.transform_called is True
    assert pre.inverse_transform_called is True


def test_forecast_unseen_weighted_ensemble():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    series = pd.Series(range(5), index=idx)

    m1 = FakeModel(value=10)
    m2 = FakeModel(value=20)

    out = forecast_unseen(
        series=series,
        horizon=2,
        models={"a": m1, "b": m2},
        weights={"a": 0.25, "b": 0.75},
        preprocessor=FakePreprocessor(),
    )

    expected = 0.25 * 10 + 0.75 * 20
    assert np.all(out["forecast"] == expected)


def test_forecast_unseen_missing_weight_defaults_zero():
    idx = pd.date_range("2025-01-01", periods=5, freq="D")
    series = pd.Series(range(5), index=idx)

    m1 = FakeModel(value=10)
    m2 = FakeModel(value=999)

    out = forecast_unseen(
        series=series,
        horizon=1,
        models={"a": m1, "b": m2},
        weights={"a": 1.0},
        preprocessor=FakePreprocessor(),
    )

    assert out["forecast"][0] == 10
