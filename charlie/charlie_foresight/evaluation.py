import numpy as np
import pandas as pd
from charlie.charlie_foresight.metrics import mse, rmse, mae, mase


def summarise_cv_metrics(cv_results: dict) -> pd.DataFrame:
    """
    Aggregate cross-validation results into per-model performance metrics.
    
    Combines predictions and actuals across all folds for each model and 
    computes aggregated error metrics (MAE, RMSE, MSE, MASE).
    
    Args:
        cv_results: Dictionary mapping model names to lists of fold results.
            Each fold is a dict with keys 'pred' (list of predictions, can be None),
            and 'actual' (list of true values).
    
    Returns:
        DataFrame with models as index and metrics (MAE, RMSE, MSE, MASE) as columns,
        sorted by MASE (ascending).
    
    Example:
        >>> cv_results = {
        ...     'model_a': [
        ...         {'pred': [1.2, 2.1, 3.3], 'actual': [1.0, 2.0, 3.0]},
        ...         {'pred': [1.1, 2.2, 3.1], 'actual': [1.0, 2.0, 3.0]}
        ...     ],
        ...     'model_b': [
        ...         {'pred': [1.3, 2.0, 3.2], 'actual': [1.0, 2.0, 3.0]},
        ...         {'pred': None, 'actual': [1.0, 2.0, 3.0]}
        ...     ]
        ... }
        >>> results = summarise_cv_metrics(cv_results)
        >>> results
                  MAE      RMSE       MSE      MASE
        model   
        model_a  0.133333  0.155556  0.024242  ...
        model_b  0.155556  0.195050  0.038050  ...
    """
    rows = []

    for model, folds in cv_results.items():
        y_true_all = []
        y_pred_all = []

        for f in folds:
            if f["pred"] is None:
                continue
            y_true_all.extend(f["actual"])
            y_pred_all.extend(f["pred"])

        if not y_true_all:
            continue

        rows.append({
            "model": model,
            "MAE": mae(y_true_all, y_pred_all),
            "RMSE": rmse(y_true_all, y_pred_all),
            "MSE": mse(y_true_all, y_pred_all),
            "MASE": mase(y_true_all, y_pred_all),
        })

    return pd.DataFrame(rows).set_index("model").sort_values("MASE")


def compute_ensemble_cv_metrics(cv_results, weights):
    """
    Compute performance metrics for a weighted ensemble across CV folds.
    
    Creates an ensemble model by combining predictions from multiple models 
    using specified weights, then calculates aggregated error metrics across 
    all folds.
    
    Args:
        cv_results: Dictionary mapping model names to lists of fold results.
            Each fold is a dict with keys 'pred' (list of predictions, can be None),
            and 'actual' (list of true values).
        weights: Dictionary mapping model names to their ensemble weights.
            Only models present in weights are included in the ensemble.
    
    Returns:
        Dictionary with ensemble metrics containing keys:
            - 'MAE': Mean Absolute Error
            - 'RMSE': Root Mean Squared Error
            - 'MSE': Mean Squared Error
            - 'MASE': Mean Absolute Scaled Error
    
    Example:
        >>> cv_results = {
        ...     'model_a': [
        ...         {'pred': [1.2, 2.1, 3.3], 'actual': [1.0, 2.0, 3.0]},
        ...         {'pred': [1.1, 2.2, 3.1], 'actual': [1.0, 2.0, 3.0]}
        ...     ],
        ...     'model_b': [
        ...         {'pred': [1.3, 2.0, 3.2], 'actual': [1.0, 2.0, 3.0]},
        ...         {'pred': [0.9, 2.1, 3.0], 'actual': [1.0, 2.0, 3.0]}
        ...     ]
        ... }
        >>> weights = {'model_a': 0.6, 'model_b': 0.4}
        >>> metrics = compute_ensemble_cv_metrics(cv_results, weights)
        >>> metrics
        {'MAE': 0.116667, 'RMSE': 0.135401, 'MSE': 0.018333, 'MASE': ...}
    """
    y_true_all = []
    y_pred_all = []

    n_folds = len(next(iter(cv_results.values())))

    for i in range(n_folds):
        fold_preds = []
        fold_weights = []

        for model, folds in cv_results.items():
            if model in weights and folds[i]["pred"] is not None:
                fold_preds.append(folds[i]["pred"])
                fold_weights.append(weights[model])

        if not fold_preds:
            continue

        ensemble_pred = sum(w * p for w, p in zip(fold_weights, fold_preds))
        y_true_all.extend(cv_results[next(iter(cv_results))][i]["actual"])
        y_pred_all.extend(ensemble_pred)

    return {
        "MAE": mae(y_true_all, y_pred_all),
        "RMSE": rmse(y_true_all, y_pred_all),
        "MSE": mse(y_true_all, y_pred_all),
        "MASE": mase(y_true_all, y_pred_all),
    }