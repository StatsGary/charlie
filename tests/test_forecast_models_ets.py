import pytest
import pandas as pd
import numpy as np
from charlie.charlie_foresight.forecast_models.ets import ETSModel

class DummySeasonalPolicy:
    def __call__(self, n, seasonal_periods):
        return "add" if n > 10 else None

@pytest.fixture
def sample_train_data():
    dates = pd.date_range(start="2020-01-01", periods=50)
    data = pd.DataFrame({'y': np.random.randn(50)}, index=dates)
    return data

@pytest.fixture
def sample_future_data():
    dates = pd.date_range(start="2020-02-20", periods=10)
    return pd.DataFrame(index=dates)

def test_ets_fit_predict(sample_train_data, sample_future_data):
    model = ETSModel(seasonal_periods=12)
    model.fit(sample_train_data)
    preds, lower, upper = model.predict(sample_future_data)
    assert isinstance(preds, np.ndarray)
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)
    assert preds.shape[0] == len(sample_future_data)

def test_ets_seasonal_policy(sample_train_data, sample_future_data):
    policy = DummySeasonalPolicy()
    model = ETSModel(seasonal_periods=12, seasonal_policy=policy)
    model.fit(sample_train_data)
    preds, lower, upper = model.predict(sample_future_data)
    assert preds.shape[0] == len(sample_future_data)

def test_ets_no_seasonality(sample_train_data, sample_future_data):
    model = ETSModel()
    model.fit(sample_train_data)
    preds, lower, upper = model.predict(sample_future_data)
    assert preds.shape[0] == len(sample_future_data)
