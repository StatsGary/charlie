import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from charlie.models.cross_validation import MultiModelCrossValidator  

@pytest.fixture
def sample_data():
    """Load a small dataset for testing."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

@pytest.fixture
def sample_models():
    """Provide a simple model list for testing."""
    return [
        ("RandomForest", RandomForestClassifier, {"n_estimators": 10}),
    ]

def test_cross_validate_and_predict(sample_data, sample_models):
    X, y = sample_data
    validator = MultiModelCrossValidator(
        models=sample_models,
        score_fn=accuracy_score,
        higher_is_better=True,
        cv_splits=3
    )

    results = validator.cross_validate(X, y)
    assert isinstance(results, dict)
    assert "RandomForest" in results
    assert validator.fitted_best_model is not None

    preds = validator.predict(X[:5])
    assert len(preds) == 5

    probs = validator.predict_proba(X[:5])
    assert probs.shape[1] == 2  # Probabilities for two classes

def test_predict_without_cross_validate(sample_data, sample_models):
    X, _ = sample_data
    validator = MultiModelCrossValidator(
        models=sample_models,
        score_fn=accuracy_score
    )
    with pytest.raises(ValueError):
        validator.predict(X)

def test_invalid_models(sample_data):
    X, y = sample_data
    invalid_models = [("BadModel", None, {})]  # Missing class
    validator = MultiModelCrossValidator(
        models=invalid_models,
        score_fn=accuracy_score
    )
    with pytest.raises(Exception):
        validator.cross_validate(X, y)

def test_mismatched_X_y(sample_models):
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, size=5)  # Wrong length
    validator = MultiModelCrossValidator(
        models=sample_models,
        score_fn=accuracy_score
    )
    with pytest.raises(ValueError):
        validator.cross_validate(X, y)

def test_tqdm_option(sample_data, sample_models):
    X, y = sample_data
    # Ensure tqdm option doesn't break anything
    validator = MultiModelCrossValidator(
        models=sample_models,
        score_fn=accuracy_score,
        use_tqdm=True,
        progress_mode="model"
    )
    results = validator.cross_validate(X, y)
    assert "RandomForest" in results
