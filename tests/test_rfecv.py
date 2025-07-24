import numpy as np
import pytest
import sys
import os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from charlie.feature_selection.rfecv import CharlieRFECV

# --- Additional imports for new tests ---
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold

@pytest.fixture
def small_dataset():
    """
    Fixture to generate a small synthetic classification dataset.

    Returns:
        X (ndarray): Feature matrix.
        y (ndarray): Target vector.
    """
    # Generate a toy dataset with 50 samples, 6 features (3 informative)
    X, y = make_classification(
        n_samples=50, n_features=6, n_informative=3, random_state=42
    )
    return X, y

@pytest.fixture
def charlie_rfecv_instance():
    """
    Fixture to create a CharlieRFECV instance with a RandomForestClassifier.

    Returns:
        selector (CharlieRFECV): Configured RFECV selector.
    """
    # Create a random forest classifier as the base estimator
    estimator = RandomForestClassifier(n_estimators=10, random_state=42)
    # Initialize CharlieRFECV with specific parameters
    selector = CharlieRFECV(
        estimator=estimator,
        step=2,                   # Number of features to remove at each step
        min_features_to_select=2, # Minimum number of features to keep
        verbose=0
    )
    return selector

def test_fit_sets_attributes(small_dataset, charlie_rfecv_instance):
    """
    Test that fitting sets the required attributes on the selector.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)  # Fit selector to data

    # Check that all necessary attributes are set after fit
    assert hasattr(selector, "best_support_")
    assert hasattr(selector, "best_score_")
    assert hasattr(selector, "best_model_")
    assert hasattr(selector, "history_")

def test_min_features_selected(small_dataset, charlie_rfecv_instance):
    """
    Test that at least the minimum number of features are selected.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)

    # The sum of True values in best_support_ should be >= min_features_to_select
    assert sum(selector.best_support_) >= selector.min_features_to_select

def test_transform_shape(small_dataset, charlie_rfecv_instance):
    """
    Test that the transformed feature matrix has the correct shape.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)

    # Transform the input features using selected features
    X_trans = selector.transform(X)

    # Check that the number of columns matches the number of selected features
    assert X_trans.shape[1] == sum(selector.best_support_)
    # The number of rows should remain unchanged
    assert X_trans.shape[0] == X.shape[0]

def test_metric_history_feature_counts_decrease(small_dataset, charlie_rfecv_instance):
    """
    Test that the feature counts in the metric history do not increase over iterations.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)

    # Get the metric history (feature count, score)
    metric_history = selector.get_metric_history()
    feature_counts = [n for n, _ in metric_history]

    # Assert that the number of features never increases between steps
    assert all(feature_counts[i] >= feature_counts[i+1] for i in range(len(feature_counts)-1))

def test_all_models_have_predict(small_dataset, charlie_rfecv_instance):
    """
    Test that all models stored in the selector's history implement 'predict'.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)

    # Retrieve all models from the history
    models = selector.get_all_models()

    # Assert every model has a 'predict' method and the count matches history length
    assert all(hasattr(m, "predict") for m in models)
    assert len(models) == len(selector.history_)

def test_best_score_is_finite(small_dataset, charlie_rfecv_instance):
    """
    Test that the best score found is a finite number (not inf or NaN).
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)

    # Check that the best score is a finite value
    assert np.isfinite(selector.best_score_)

def test_fit_transform_matches_transform(small_dataset, charlie_rfecv_instance):
    """
    Test that fit_transform produces the same output as calling fit then transform.
    """
    X, y = small_dataset
    selector = charlie_rfecv_instance
    selector.fit(X, y)
    # Transform after fit
    X_trans = selector.transform(X)
    # Fit and transform in one step
    X_fit_trans = selector.fit_transform(X, y)

    # Assert that both methods give the same result
    assert np.allclose(X_fit_trans, X_trans)

# =============== ADDITIONAL COVERAGE TESTS ================

def test_fit_with_linear_model(small_dataset):
    """
    Test CharlieRFECV with a linear model that uses 'coef_' attribute.
    """
    X, y = small_dataset
    selector = CharlieRFECV(
        estimator=LogisticRegression(solver='liblinear', random_state=42),
        step=1, min_features_to_select=2, verbose=0
    )
    selector.fit(X, y)
    assert hasattr(selector, "best_support_")
    assert hasattr(selector, "best_score_")

def test_fit_with_dummy_model(small_dataset):
    """
    Test CharlieRFECV with a model that lacks 'feature_importances_' and 'coef_'.
    Should use random importances fallback.
    """
    X, y = small_dataset
    selector = CharlieRFECV(
        estimator=DummyClassifier(strategy='uniform', random_state=42),
        step=1, min_features_to_select=2, verbose=0
    )
    selector.fit(X, y)
    assert hasattr(selector, "best_support_")
    assert hasattr(selector, "best_score_")

def test_custom_scoring_function(small_dataset):
    """
    Test CharlieRFECV with a custom scoring function (accuracy).
    """
    X, y = small_dataset
    def accuracy(y_true, y_pred):
        return np.mean((y_pred > 0.5) == y_true)
    selector = CharlieRFECV(
        estimator=RandomForestClassifier(n_estimators=5, random_state=42),
        step=1, min_features_to_select=2, scoring=accuracy, verbose=0
    )
    selector.fit(X, y)
    assert hasattr(selector, "best_support_")
    assert hasattr(selector, "best_score_")

def test_min_features_to_select_greater_than_n_features(small_dataset):
    """
    Test error raised if min_features_to_select > number of features.
    """
    X, y = small_dataset
    selector = CharlieRFECV(
        estimator=RandomForestClassifier(n_estimators=5, random_state=42),
        step=1, min_features_to_select=X.shape[1]+1, verbose=0
    )
    with pytest.raises(Exception):
        selector.fit(X, y)

def test_custom_cv_object(small_dataset):
    """
    Test using a custom cross-validation object (KFold instead of StratifiedKFold).
    """
    X, y = small_dataset
    selector = CharlieRFECV(
        estimator=RandomForestClassifier(n_estimators=5, random_state=42),
        step=1, min_features_to_select=2, cv=KFold(n_splits=3), verbose=0
    )
    selector.fit(X, y)
    assert hasattr(selector, "best_support_")

def test_verbose_prints_score(small_dataset, capsys):
    """
    Test that verbose mode prints score progress to stdout.
    """
    X, y = small_dataset
    selector = CharlieRFECV(
        estimator=RandomForestClassifier(n_estimators=5, random_state=42),
        step=1, min_features_to_select=2, verbose=1
    )
    selector.fit(X, y)
    captured = capsys.readouterr()
    assert "Features:" in captured.out
