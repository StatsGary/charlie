import numpy as np

def mse(y_true, y_pred) -> float:
    """Computes the mean squared error (MSE) between true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean squared error.

    Examples:
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1.1, 2.1, 2.9]
        >>> mse(y_true, y_pred)
        0.01
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred) -> float:
    """Computes the root mean squared error (RMSE) between true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The root mean squared error.

    Examples:
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1.1, 2.1, 2.9]
        >>> rmse(y_true, y_pred)
        0.1
    """
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    """Computes the mean absolute error (MAE) between true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean absolute error.

    Examples:
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1.1, 2.1, 2.9]
        >>> mae(y_true, y_pred)
        0.1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred) -> float:
    """Computes the mean absolute percentage error (MAPE) between true and predicted values.

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean absolute percentage error as a percentage. Returns NaN if all true values are zero.

    Examples:
        >>> y_true = [1, 2]
        >>> y_pred = [1.1, 2.1]
        >>> mape(y_true, y_pred)
        7.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def mase(y_true, y_pred) -> float:
    """Computes the mean absolute scaled error (MASE) between true and predicted values.

    MASE is the mean absolute error (MAE) of the predictions divided by the MAE of the naive
    forecast method (using the previous value as the prediction).

    Args:
        y_true (array-like): The true values.
        y_pred (array-like): The predicted values.

    Returns:
        float: The mean absolute scaled error. Returns NaN if there are fewer than 2 true values
            or if the naive forecast MAE is zero.

    Examples:
        >>> y_true = [1, 2, 3]
        >>> y_pred = [1.1, 2.1, 2.9]
        >>> mase(y_true, y_pred)
        0.1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return float("nan")
    naive = np.abs(np.diff(y_true))
    denom = np.mean(naive)
    if denom == 0:
        return float("nan")
    return float(mae(y_true, y_pred) / denom)