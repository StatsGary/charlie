import pandas as pd

def to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Converts a pandas DataFrame to the format expected by Facebook Prophet.

    The input DataFrame should have a DatetimeIndex and exactly one column representing
    the target variable. The function renames the index to "ds" and the column to "y"
    as required by Prophet.

    Args:
        df (pd.DataFrame): The input DataFrame with a DatetimeIndex and a single target column.

    Returns:
        pd.DataFrame: A new DataFrame with columns "ds" (datetime) and "y" (target value).

    Raises:
        ValueError: If the DataFrame's index is not a DatetimeIndex or if it does not have exactly one column.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3, freq='D'))
        >>> to_prophet_df(df)
           ds  y
        0 2020-01-01  1
        1 2020-01-02  2
        2 2020-01-03  3
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")

    if df.shape[1] != 1:
        raise ValueError("Expected single target column")

    return (
        df.reset_index()
          .rename(columns={
              df.index.name or "index": "ds",
              df.columns[0]: "y"
          })
    )