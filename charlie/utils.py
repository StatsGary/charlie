import numpy as np
from typing import List, Optional, Sequence

def weighted_ensemble(
    probs_list: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None
) -> np.ndarray:
    """
    Computes a weighted ensemble of probability arrays.

    Each probability array in `probs_list` should have the same shape, typically
    (n_samples, n_classes) for classification model outputs.
    If `weights` is not provided, all arrays are averaged equally.

    Parameters
    ----------
    probs_list : Sequence[np.ndarray]
        List or sequence of probability arrays to ensemble.
    weights : Optional[Sequence[float]], optional
        Sequence of weights for each probability array. If None, uniform weighting is used.

    Returns
    -------
    np.ndarray
        The weighted ensemble probability array.

    Examples
    --------
    >>> probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    >>> weighted_ensemble([probs1, probs2], weights=[0.8, 0.2])
    array([[0.68, 0.32],
           [0.42, 0.58]])
    """

    probs_arr = np.array(probs_list)
    n_models = len(probs_arr)
    if weights is None:
        weights_arr = np.ones(n_models) / n_models
    else:
        if len(weights) != n_models:
            raise ValueError(
                f"Number of weights ({len(weights)}) does not match "
                f"number of probability arrays ({n_models})"
            )
        weights_arr = np.array(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()
    combined = np.zeros_like(probs_arr[0])
    for w, p in zip(weights_arr, probs_arr):
        combined += w * p
    return combined



def mean_ensemble(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Computes the mean ensemble of probability arrays.

    Parameters
    ----------
    probs_list : Sequence[np.ndarray]
        List or sequence of probability arrays to ensemble.

    Returns
    -------
    np.ndarray
        The mean ensemble probability array.

    Examples
    --------
    >>> probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    >>> mean_ensemble([probs1, probs2])
    array([[0.65, 0.35],
           [0.45, 0.55]])
    """
    return np.mean(probs_list, axis=0)

def median_ensemble(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Computes the median ensemble of probability arrays.

    Parameters
    ----------
    probs_list : Sequence[np.ndarray]
        List or sequence of probability arrays to ensemble.

    Returns
    -------
    np.ndarray
        The median ensemble probability array.

    Examples
    --------
    >>> probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    >>> probs3 = np.array([[0.9, 0.1], [0.2, 0.8]])
    >>> median_ensemble([probs1, probs2, probs3])
    array([[0.7, 0.3],
           [0.4, 0.6]])
    """
    return np.median(probs_list, axis=0)

def max_ensemble(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Computes the max ensemble of probability arrays (elementwise maximum).

    Parameters
    ----------
    probs_list : Sequence[np.ndarray]
        List or sequence of probability arrays to ensemble.

    Returns
    -------
    np.ndarray
        The maximum ensemble probability array.

    Examples
    --------
    >>> probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    >>> max_ensemble([probs1, probs2])
    array([[0.7, 0.4],
           [0.5, 0.6]])
    """
    return np.max(probs_list, axis=0)


def rank_ensemble(probs_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Computes the rank ensemble of probability arrays.
    
    For each class probability, the models' probabilities are replaced by their rank
    (1 for lowest, N for highest, ties resolved by average rank). The mean rank
    is computed across models and returned as the ensemble score.

    Parameters
    ----------
    probs_list : Sequence[np.ndarray]
        List or sequence of probability arrays to ensemble. Each array should have
        the same shape.

    Returns
    -------
    np.ndarray
        The mean rank array, same shape as an individual entry of `probs_list`.

    Examples
    --------
    >>> import numpy as np
    >>> from charlie.ensemble import rank_ensemble
    >>> probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    >>> rank_ensemble([probs1, probs2])
    array([[1.5, 1.5],
           [1. , 2. ]])
    """
    probs_arr = np.array(probs_list)
    # Rank across the 0th axis (models), shape: (n_models, n_samples, n_classes)
    ranks = np.argsort(np.argsort(probs_arr, axis=0), axis=0) + 1
    # Mean rank
    return np.mean(ranks, axis=0)

