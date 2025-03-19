""""Tests for the CHARLIE model."""

import pytest
import numpy as np
import torch
from models.ensemble import CHARLIE

@pytest.fixture
def regression_data():
    np.random.seed(0)
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    return X, y

@pytest.fixture
def classification_data():
    np.random.seed(0)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y

def test_initialization_regression():
    model = CHARLIE(input_dim=10, classification=False)
    assert isinstance(model.rf, type(model.rf))  # Should be RandomForestRegressor
    assert model.alpha.requires_grad
    assert model.selected_features <= 10

def test_initialization_classification():
    model = CHARLIE(input_dim=10, classification=True)
    assert isinstance(model.rf, type(model.rf))  # Should be RandomForestClassifier
    assert model.alpha.requires_grad
    assert model.selected_features <= 10

def test_train_regression(regression_data):
    X, y = regression_data
    model = CHARLIE(input_dim=10, classification=False, selected_features=5)
    model.train_model(X, y, epochs=2, lr=0.01)  # Small epochs for speed
    assert model.nn_model is not None
    assert model.top_features.shape[0] == 5

def test_train_classification(classification_data):
    X, y = classification_data
    model = CHARLIE(input_dim=10, classification=True, selected_features=4)
    model.train_model(X, y, epochs=2, lr=0.01)
    assert model.nn_model is not None
    assert model.top_features.shape[0] == 4

def test_predict_regression_shape(regression_data):
    X, y = regression_data
    model = CHARLIE(input_dim=10, classification=False)
    model.train_model(X, y, epochs=2)
    preds = model.predict(X)
    assert preds.shape == (100, 1)

def test_predict_classification_shape(classification_data):
    X, y = classification_data
    model = CHARLIE(input_dim=10, classification=True)
    model.train_model(X, y, epochs=2)
    preds = model.predict(X)
    assert preds.shape[0] == 100  # Should match number of samples

def test_forward_raises_before_training(regression_data):
    X, _ = regression_data
    model = CHARLIE(input_dim=10, classification=False)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    with pytest.raises(ValueError):
        model.forward(X_tensor)
