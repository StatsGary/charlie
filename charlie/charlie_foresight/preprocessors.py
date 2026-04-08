import numpy as np
import pandas as pd
from typing import Optional, Type
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from charlie.charlie_foresight.transforms import SeriesTransformer


class TimePreprocessor:
    """A preprocessor for time series data that handles imputation, scaling, and transformation.

    This class provides a complete preprocessing pipeline for time series data, including
    interpolation of missing values, imputation, scaling, and optional series transformation
    for handling seasonality and variance stabilization.

    Attributes:
        transformer (Optional[SeriesTransformer]): Optional series transformer for seasonality.
        interpolate_method (str): Interpolation method used by pandas.
        imputer: The fitted imputer instance.
        scaler: The fitted scaler instance.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({'y': [1, np.nan, 3, 4, 5]})
        >>> preprocessor = TimePreprocessor()
        >>> processed = preprocessor.fit_transform(data)
        >>> original = preprocessor.inverse_transform(processed)
        >>> original.shape
        (5,)
    """

    def __init__(
        self,
        *,
        scaler_type: str = "robust",
        scaler_cls: Optional[Type] = None,
        scaler_kwargs: Optional[dict] = None,
        imputer_cls: Type = KNNImputer,
        imputer_kwargs: Optional[dict] = None,
        transformer: Optional[SeriesTransformer] = None,
        interpolate_method: str = "time",
    ):
        """Initializes the TimePreprocessor.

        Args:
            scaler_type (str, optional): Type of scaler ('standard' or 'robust'). Defaults to "robust".
            scaler_cls (Optional[Type], optional): Custom scaler class. Defaults to None.
            scaler_kwargs (Optional[dict], optional): Keyword arguments for the scaler. Defaults to None.
            imputer_cls (Type, optional): Imputer class. Defaults to KNNImputer.
            imputer_kwargs (Optional[dict], optional): Keyword arguments for the imputer. Defaults to None.
            transformer (Optional[SeriesTransformer], optional): Series transformer instance. Defaults to None.
            interpolate_method (str, optional): Interpolation method for pandas. Defaults to "time".
        """
        self.transformer = transformer
        self.interpolate_method = interpolate_method

        self.imputer = imputer_cls(**(imputer_kwargs or {}))

        if scaler_cls:
            self.scaler = scaler_cls(**(scaler_kwargs or {}))
        else:
            scaler_kwargs = scaler_kwargs or {}
            if scaler_type == "standard":
                self.scaler = StandardScaler(**scaler_kwargs)
            elif scaler_type == "robust":
                self.scaler = RobustScaler(**scaler_kwargs)
            else:
                raise ValueError("scaler_type must be 'standard' or 'robust'")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the preprocessor and transforms the data.

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = df.copy().sort_index().interpolate(self.interpolate_method)
        df = df.astype(float)

        if self.transformer:
            df = self.transformer.fit_transform(df)

        df[:] = self.imputer.fit_transform(df)
        df[:] = self.scaler.fit_transform(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted preprocessor.

        Args:
            df (pd.DataFrame): Input DataFrame to preprocess.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = df.copy().sort_index().interpolate(self.interpolate_method)

        if self.transformer:
            df = self.transformer.fit_transform(df)

        df[:] = self.imputer.transform(df)
        df[:] = self.scaler.transform(df)
        return df

    def inverse_transform(self, arr) -> np.ndarray:
        """Applies inverse transformations to the data.

        Args:
            arr: Input array, DataFrame, or Series to inverse transform.

        Returns:
            np.ndarray: Data in original scale.
        """
        if isinstance(arr, pd.DataFrame):
            values = arr.values
        elif isinstance(arr, pd.Series):
            values = arr.values.reshape(-1, 1)
        else:
            values = np.asarray(arr).reshape(-1, 1)

        values = self.scaler.inverse_transform(values).ravel()

        if self.transformer:
            values = self.transformer.inverse_transform(values)

        return values