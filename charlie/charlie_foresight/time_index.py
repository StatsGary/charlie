import pandas as pd
from typing import Optional


def infer_seasonal_period(index: pd.DatetimeIndex) -> Optional[int]:
    """Infers the seasonal period from a pandas DatetimeIndex based on its frequency.

    This function attempts to infer the frequency of the index and returns a common
    seasonal period for that frequency (e.g., 7 for daily data assuming weekly seasonality).

    Args:
        index (pd.DatetimeIndex): The datetime index from which to infer the seasonal period.

    Returns:
        Optional[int]: The inferred seasonal period. Returns 7 for daily ('D'), 52 for weekly ('W'),
            12 for monthly ('M'), 4 for quarterly ('Q'), or None if frequency cannot be inferred
            or is not supported.

    Examples:
        >>> import pandas as pd
        >>> index = pd.date_range('2020-01-01', periods=10, freq='D')
        >>> infer_seasonal_period(index)
        7
    """
    freq = pd.infer_freq(index)
    if freq is None:
        return None
    if freq.startswith("D"):
        return 7
    if freq.startswith("W"):
        return 52
    if freq.startswith("M"):
        return 12
    if freq.startswith("Q"):
        return 4
    return None


def make_future_index(history_index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    """Creates a future DatetimeIndex extending from the end of the history index for a specified horizon.

    This function infers the frequency from the provided history index and generates a new
    DatetimeIndex with the specified number of future periods.

    Args:
        history_index (pd.DatetimeIndex): The historical datetime index to extend from.
        horizon (int): The number of future periods to generate.

    Returns:
        pd.DatetimeIndex: A new DatetimeIndex containing the future dates.

    Raises:
        ValueError: If the frequency cannot be inferred from the history index.

    Examples:
        >>> import pandas as pd
        >>> history = pd.date_range('2020-01-01', periods=5, freq='D')
        >>> make_future_index(history, 3)
        DatetimeIndex(['2020-01-06', '2020-01-07', '2020-01-08'], dtype='datetime64[ns]', freq='D')
    """
    freq = pd.infer_freq(history_index)
    if freq is None:
        raise ValueError("Cannot infer frequency from history index")

    return pd.date_range(
        start=history_index[-1],
        periods=horizon + 1,
        freq=freq
    )[1:]