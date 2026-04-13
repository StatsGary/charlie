import pytest
import pandas as pd
import numpy as np
import builtins
from charlie.charlie_foresight.forecast_models.prophet import ProphetModel

class DummyProphet:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted = False

    def fit(self, df, **kwargs):
        self.fitted = True

    def predict(self, fut):
        n = len(fut)
        return pd.DataFrame({
            "yhat": np.arange(n),
            "yhat_lower": np.arange(n) - 1,
            "yhat_upper": np.arange(n) + 1,
        })

@pytest.fixture
def sample_train_data():
    dates = pd.date_range(start="2020-01-01", periods=10)
    data = pd.DataFrame({'y': np.arange(10)}, index=dates)
    return data

@pytest.fixture
def sample_future_data():
    dates = pd.date_range(start="2020-01-11", periods=5)
    return pd.DataFrame(index=dates)

def test_fit_predict(monkeypatch, sample_train_data, sample_future_data):
    monkeypatch.setattr("charlie.charlie_foresight.forecast_models.prophet.Prophet", DummyProphet)
    monkeypatch.setattr("charlie.charlie_foresight.forecast_models.prophet._HAS_PROPHET", True)
    model = ProphetModel()
    model.fit(sample_train_data)
    preds, lower, upper = model.predict(sample_future_data)
    assert len(preds) == len(sample_future_data)
    assert len(lower) == len(sample_future_data)
    assert len(upper) == len(sample_future_data)

def test_fit_raises_import_error(monkeypatch, sample_train_data):
    monkeypatch.setattr("charlie.charlie_foresight.forecast_models.prophet._HAS_PROPHET", False)
    model = ProphetModel()
    with pytest.raises(ImportError):
        model.fit(sample_train_data)
