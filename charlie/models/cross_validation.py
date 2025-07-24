from sklearn.model_selection import StratifiedKFold
import numpy as np

class MultiModelCrossValidator:
    """
    Multi-model cross-validation utility with flexible tqdm progress modes,
    robust error handling, and support for custom probability thresholds.

    Features:
    ---------
    - Accepts multiple models with custom hyperparameters.
    - Validates input data, models, and scoring function.
    - Supports any metric (AUC, accuracy, F1, etc.).
    - Automatically refits the best model on the full dataset after CV.
    - Flexible progress bar modes: 'model', 'nested', 'aggregate'.
    - Custom threshold for converting probabilities to labels (default 0.5).
    - Safe fallback for models missing `predict_proba` (uses `predict`).

    Parameters:
    -----------
    models : list of tuples
        List of (model_name, model_class, model_params_dict).
    score_fn : callable
        Scoring function with signature `score_fn(y_true, y_pred)`.
    higher_is_better : bool, default=True
        Whether higher scores indicate better performance.
    cv_splits : int, default=5
        Number of stratified k-fold splits.
    random_state : int, default=42
        Random seed for reproducibility.
    use_tqdm : bool, default=False
        Whether to show progress bars using tqdm.
    progress_mode : {'model', 'nested', 'aggregate'}, default='model'
        Type of progress bar to display if `use_tqdm=True`.
    threshold : float, default=0.5
        Probability cutoff for converting probabilities to labels
        (used only when score_fn expects labels, e.g., accuracy or F1).

    Usage Example:
    --------------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    >>> from sklearn.metrics import roc_auc_score

    >>> data = load_breast_cancer()
    >>> X, y = data.data, data.target

    >>> models = [
    ...     ("RandomForest", RandomForestClassifier, {"n_estimators": 100}),
    ...     ("GradientBoosting", GradientBoostingClassifier, {"n_estimators": 200})
    ... ]

    >>> validator = MultiModelCrossValidator(
    ...     models=models,
    ...     score_fn=roc_auc_score,
    ...     higher_is_better=True,
    ...     cv_splits=5,
    ...     use_tqdm=True,
    ...     progress_mode="model",
    ...     threshold=0.4   # custom threshold for label-based metrics
    ... )

    >>> results = validator.cross_validate(X, y)
    >>> print("Best model:", validator.best_model_info)
    >>> predictions = validator.predict_proba(X[:5])
    """

    def __init__(self, models, score_fn, higher_is_better=True, cv_splits=5,
                 random_state=42, use_tqdm=False, progress_mode="model",
                 threshold=0.5, **kwargs):
        self.models = models
        self.score_fn = score_fn
        self.higher_is_better = higher_is_better
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.results = {}
        self.best_model_info = None
        self.fitted_best_model = None
        self.use_tqdm = use_tqdm
        self.progress_mode = progress_mode
        self.threshold = threshold

    def _validate_inputs(self, X, y):
        """Check input data and configurations."""
        if not self.models or not isinstance(self.models, list):
            raise ValueError("Models must be a non-empty list of (name, class, params) tuples.")

        for model in self.models:
            if not isinstance(model, tuple) or len(model) != 3:
                raise ValueError("Each model must be a tuple: (name, model_class, params_dict)")

        if not callable(self.score_fn):
            raise TypeError("score_fn must be a callable, e.g., accuracy_score or roc_auc_score.")

        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have the same number of rows. Got {X.shape[0]} and {len(y)}.")

        if self.cv_splits > len(y):
            raise ValueError("Number of CV splits cannot exceed number of samples.")

    def cross_validate(self, X, y):
        # Validate inputs
        self._validate_inputs(X, y)

        # Handle tqdm import
        if self.use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                print("tqdm not installed, running without progress bar.")
                self.use_tqdm = False

        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)

        # Aggregate mode: one bar for total tasks
        total_tasks = len(self.models) * self.cv_splits if self.progress_mode == "aggregate" else None
        if self.use_tqdm and self.progress_mode == "aggregate":
            progress_bar = tqdm(total=total_tasks, desc="Cross-Validation Progress")

        for name, model_class, params in self.models:
            scores = []

            # Model-level progress
            if self.use_tqdm and self.progress_mode == "model":
                from tqdm import tqdm
                fold_iterator = tqdm(range(self.cv_splits), desc=f"{name} folds")
            else:
                fold_iterator = range(self.cv_splits)

            for fold_idx in fold_iterator:
                try:
                    train_idx, val_idx = list(skf.split(X, y))[fold_idx]

                    if hasattr(X, "iloc"):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    else:
                        X_train, X_val = X[train_idx], X[val_idx]

                    if hasattr(y, "iloc"):
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    else: 
                        y_train, y_val = y[train_idx], y[val_idx]

                    model = model_class(random_state=self.random_state, **params)
                    model.fit(X_train, y_train)

                    # Predict probabilities if available
                    if hasattr(model, "predict_proba"):
                        y_pred = model.predict_proba(X_val)[:, 1]
                    else:
                        y_pred = model.predict(X_val)

                    # Convert to labels if metric is label-based
                    if "accuracy" in self.score_fn.__name__ or "f1" in self.score_fn.__name__:
                        y_pred = (y_pred >= self.threshold).astype(int)

                    score = self.score_fn(y_val, y_pred)
                    scores.append(score)

                except Exception as e:
                    print(f"Warning: Fold {fold_idx+1} failed for model {name} with error: {e}")
                    continue

                if self.use_tqdm and self.progress_mode == "aggregate":
                    progress_bar.update(1)

            if scores:
                self.results[name] = {
                    "mean_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "all_scores": scores,
                    "model_class": model_class,
                    "params": params
                }
            else:
                print(f"Warning: Model {name} failed on all folds and will be excluded.")

        if self.use_tqdm and self.progress_mode == "aggregate":
            progress_bar.close()

        if not self.results:
            raise RuntimeError("No valid models were successfully evaluated.")

        # Determine best model
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]["mean_score"],
            reverse=self.higher_is_better
        )
        self.best_model_info = sorted_models[0]

        # Refit best model on full dataset
        name, info = self.best_model_info
        best_model = info["model_class"](random_state=self.random_state, **info["params"])
        best_model.fit(X, y)
        self.fitted_best_model = best_model

        return self.results

    def predict(self, X):
        if self.fitted_best_model is None:
            raise ValueError("Run cross_validate() before predicting.")
        return self.fitted_best_model.predict(X)

    def predict_proba(self, X):
        if self.fitted_best_model is None:
            raise ValueError("Run cross_validate() before predicting.")
        if hasattr(self.fitted_best_model, "predict_proba"):
            return self.fitted_best_model.predict_proba(X)
        # Fallback: Convert labels to probability format
        preds = self.fitted_best_model.predict(X)
        return np.vstack([1 - preds, preds]).T