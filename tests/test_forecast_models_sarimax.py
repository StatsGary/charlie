import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from charlie.charlie_foresight.forecast_models.sarimax import SeasonalARIMAModel

@pytest.fixture
def sample_train_data():
    dates = pd.date_range(start="2020-01-01", periods=50)
    data = pd.DataFrame({'y': np.random.randn(50)}, index=dates)
    return data

@pytest.fixture
def sample_future_data():
    dates = pd.date_range(start="2020-02-20", periods=10)
    return pd.DataFrame(index=dates)

def test_fit_predict_basic(sample_train_data, sample_future_data):
    model = SeasonalARIMAModel(seasonal_period=12)
    # Mock the SARIMAX and fit to avoid heavy computation
    mock_fit = MagicMock()
    mock_fit.get_forecast.return_value = MagicMock(
        predicted_mean=np.arange(len(sample_future_data)),
        conf_int=MagicMock(return_value=pd.DataFrame({0: np.arange(len(sample_future_data)), 1: np.arange(len(sample_future_data)) + 1}))
    )
    model.model = mock_fit
    # We test predict directly with mocked model
    preds, lower, upper = model.predict(sample_future_data)
    assert len(preds) == len(sample_future_data)
    assert len(lower) == len(sample_future_data)
    assert len(upper) == len(sample_future_data)

def test_hyperparameter_search_runs(sample_train_data):
    model = SeasonalARIMAModel(seasonal_period=12, optuna_kwargs={'n_trials': 1})
    # Create dummy splits
    splits = [(range(30), range(30, 50))]
    # Patch _objective to a dummy function that returns a fixed value
    model._objective = lambda trial, tr, va: 1.0
    model.hyperparameter_search(sample_train_data, splits)
    assert model.best_params is not None

def test_fit_sets_model(sample_train_data):
    model = SeasonalARIMAModel(seasonal_period=12)
    model.best_params = {'p': 1, 'd': 0, 'q': 1, 'P': 0, 'D': 0, 'Q': 0}
    model.fit(sample_train_data)
    assert model.model is not None
