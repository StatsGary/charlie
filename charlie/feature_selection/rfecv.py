import numpy as np
import copy
from typing import Optional, Callable, Any, List, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

class CharlieRFECV:
    """
    Recursive Feature Elimination with Cross-Validation (RFECV) wrapper.

    This class iteratively removes the least important features based on model
    feature importances or coefficients and evaluates performance using
    cross-validation. It selects the subset of features that yields the
    highest cross-validated score and refits the final model using those
    selected features.

    Features:
    ----------
    - Works with any scikit-learn estimator supporting `fit` and `predict`/`predict_proba`.
    - Supports custom scoring functions (default is `roc_auc_score`).
    - Allows step-wise feature elimination and tracks score history.
    - Uses Stratified K-Fold cross-validation by default.
    - Can handle models with `feature_importances_` or `coef_`; random fallback if unavailable.

    Parameters:
    -----------
    estimator : BaseEstimator
        The model to evaluate and use for feature elimination (e.g., RandomForestClassifier).

    step : int, default=1
        Number of features to remove at each iteration.

    cv : int or cross-validation generator, default=5
        Number of cross-validation folds or an object with `split()` method.

    scoring : callable, default=roc_auc_score
        Function with signature `score(y_true, y_pred)` returning a float score.

    min_features_to_select : int, default=1
        Minimum number of features to keep during elimination.

    verbose : int, default=1
        Verbosity level (prints score at each iteration if >0).

    random_state : int, default=42
        Random seed for reproducibility (affects shuffling in CV).

    shuffle : bool, default=True
        Whether to shuffle data before cross-validation split.

    Attributes:
    -----------
    best_model_ : BaseEstimator
        The final fitted model on the selected features.

    best_support_ : np.ndarray
        Boolean mask of selected features.

    best_score_ : float
        Best cross-validated score achieved.

    history_ : list of dict
        Details of each iteration: features, support mask, score, and model.

    Methods:
    --------
    fit(X, y):
        Perform recursive feature elimination with cross-validation.

    transform(X):
        Reduce input X to the selected features.

    fit_transform(X, y):
        Fit and transform in one step.

    get_metric_history():
        Returns a list of (num_features, score) for each iteration.

    get_all_models():
        Returns a list of all fitted models from each iteration.

    Usage Example:
    --------------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from charlie.models.rfecv import CharlieRFECV

    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target

    >>> selector = CharlieRFECV(
    ...     estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    ...     step=2,
    ...     cv=5,
    ...     scoring=roc_auc_score,
    ...     min_features_to_select=5,
    ...     verbose=1
    ... )

    >>> selector.fit(X, y)
    >>> print("Best score:", selector.best_score_)
    >>> print("Selected features:", selector.best_support_)

    >>> # Transform data to selected features
    >>> X_reduced = selector.transform(X)
    """
    def __init__(
        self,
        estimator: BaseEstimator,
        step: int = 1,
        cv: Union[int, Any] = 5,
        scoring: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        min_features_to_select: int = 1,
        verbose: int = 1,
        random_state: int = 42,
        shuffle: bool = True
    ) -> None:
        self.estimator = estimator
        self.step = step
        self.cv = cv if hasattr(cv, "split") else StratifiedKFold(
            n_splits=cv, shuffle=shuffle, random_state=random_state
        )
        self.scoring = scoring if scoring is not None else roc_auc_score
        self.min_features_to_select = min_features_to_select
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CharlieRFECV':
        X = np.asarray(X)
        n_features = X.shape[1]
        support = np.ones(n_features, dtype=bool)
        self.history_: List[dict] = []
        feature_indices = np.arange(n_features)
        best_score = -np.inf
        best_model: Optional[BaseEstimator] = None
        best_support: Optional[np.ndarray] = None

        while support.sum() >= self.min_features_to_select:
            cols = feature_indices[support]
            X_sub = X[:, cols]

            cv_scores: List[float] = []
            for train_idx, val_idx in self.cv.split(X_sub, y):
                X_train, X_val = X_sub[train_idx], X_sub[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = copy.deepcopy(self.estimator)
                model.fit(X_train, y_train)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_val)[:, 1]
                else:
                    y_prob = model.predict(X_val)
                score = self.scoring(y_val, y_prob)
                cv_scores.append(score)
            mean_score = np.mean(cv_scores)

            final_model = copy.deepcopy(self.estimator)
            final_model.fit(X_sub, y)
            self.history_.append({
                "features": cols,
                "support": support.copy(),
                "score": mean_score,
                "model": copy.deepcopy(final_model)
            })

            if self.verbose:
                print(f"Features: {support.sum()}, Mean CV Score: {mean_score:.5f}")

            if mean_score > best_score:
                best_score = mean_score
                best_model = copy.deepcopy(final_model)
                best_support = support.copy()

            if support.sum() == self.min_features_to_select:
                break

            importances: Optional[np.ndarray] = None
            if hasattr(final_model, "feature_importances_"):
                importances = final_model.feature_importances_
            elif hasattr(final_model, "coef_"):
                importances = np.abs(final_model.coef_).flatten()
            else:
                importances = np.random.rand(support.sum())

            n_remove = min(self.step, support.sum() - self.min_features_to_select)
            least_important_idx = np.argsort(importances)[:n_remove]
            support[np.where(support)[0][least_important_idx]] = False

        self.best_model_: Optional[BaseEstimator] = best_model
        self.best_support_: Optional[np.ndarray] = best_support
        self.best_score_: float = best_score
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.best_support_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_metric_history(self) -> List[Tuple[int, float]]:
        """Return a list of (number of features, score) tuples."""
        return [(np.sum(h["support"]), h["score"]) for h in self.history_]

    def get_all_models(self) -> List[BaseEstimator]:
        """Return a list of all models fitted at each step."""
        return [h["model"] for h in self.history_]