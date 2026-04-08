import numpy as np
from typing import Dict, Callable
from charlie.charlie_foresight.metrics import mase

class Ensembler:
    """An ensemble combiner for forecasting model predictions.

    This class combines predictions from multiple forecasting models by weighting
    them based on their cross-validation performance using a specified metric.
    Models with lower error get higher weights.

    Attributes:
        weight_metric (Callable): The metric function used to evaluate model performance.
        min_weight (float): Minimum weight assigned to any model (currently unused).
        eps (float): Small epsilon added to avoid division by zero in weight calculation.
        weights (Dict[str, float]): Computed weights for each model based on CV performance.

    Examples:
        >>> from charlie_foresight.cv import CrossValidator
        >>> from charlie_foresight.forecast_models.prophet import ProphetModel
        >>> # cv_results = cv.evaluate(data, {'model1': ProphetModel(), 'model2': ETSModel()})
        >>> ensembler = Ensembler()
        >>> # ensembler.fit(cv_results)
        >>> # combined = ensembler.combine({'model1': preds1, 'model2': preds2}, ...)
    """

    def __init__(
        self,
        *,
        weight_metric: Callable = mase,
        min_weight: float = 0.0,
        eps: float = 1e-9,
    ):
        """Initializes the Ensembler.

        Args:
            weight_metric (Callable, optional): Metric for weighting models. Defaults to mase.
            min_weight (float, optional): Minimum weight threshold. Defaults to 0.0.
            eps (float, optional): Epsilon for numerical stability. Defaults to 1e-9.
        """
        self.weight_metric = weight_metric
        self.min_weight = min_weight
        self.eps = eps
        self.weights: Dict[str, float] = {}

    def fit(self, cv_results: Dict[str, list]) -> None:
        """Fits the ensembler by computing model weights from cross-validation results.

        Weights are computed as the inverse of the mean metric score across folds,
        normalized to sum to 1.

        Args:
            cv_results (Dict[str, list]): Cross-validation results from CrossValidator.evaluate,
                with model names as keys and lists of fold results as values.
        """
        inv_err = {}

        for model, folds in cv_results.items():
            scores = [
                self.weight_metric(f["actual"], f["pred"])
                for f in folds
                if np.isfinite(self.weight_metric(f["actual"], f["pred"]))
            ]
            if scores:
                inv_err[model] = 1 / (np.mean(scores) + self.eps)

        total = sum(inv_err.values())
        self.weights = {k: v / total for k, v in inv_err.items()}

    def combine(self, preds, lowers, uppers):
        """Combines predictions from multiple models using learned weights.

        Args:
            preds: Dictionary mapping model names to prediction arrays.
            lowers: Dictionary mapping model names to lower confidence interval arrays.
            uppers: Dictionary mapping model names to upper confidence interval arrays.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: Combined prediction,
                lower confidence interval, and upper confidence interval. CIs are None if
                any model has None CIs.
        """
        pt = sum(self.weights[m] * preds[m] for m in self.weights)
        if all(lowers[m] is not None for m in self.weights):
            lo = sum(self.weights[m] * lowers[m] for m in self.weights)
            hi = sum(self.weights[m] * uppers[m] for m in self.weights)
        else:
            lo = hi = None
        return pt, lo, hi
