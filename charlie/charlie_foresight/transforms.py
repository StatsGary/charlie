import numpy as np
import pandas as pd
from typing import Optional
from scipy.stats import boxcox
from scipy.special import inv_boxcox


class SeriesTransformer:
    """A transformer for time series data that detects seasonality type and applies transformations.

    This class automatically detects whether a time series exhibits multiplicative or additive seasonality
    based on the correlation between rolling mean and rolling standard deviation. For multiplicative series,
    it can apply log or Box-Cox transformations to stabilize variance.

    Attributes:
        detection_window (int): Window size for rolling statistics in seasonality detection.
        corr_threshold (float): Correlation threshold to classify as multiplicative seasonality.
        transform (Optional[str]): Transformation to apply ('log', 'boxcox', or None).
        eps (float): Small value added to avoid log of zero.
        force_series_type (Optional[str]): Force a specific series type ('additive' or 'multiplicative').
        series_type (Optional[str]): Detected or forced series type.
        lambda_ (Optional[float]): Lambda parameter for Box-Cox transformation.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({'y': [1, 2, 3, 4, 5]})
        >>> transformer = SeriesTransformer()
        >>> transformed = transformer.fit_transform(data)
        >>> original = transformer.inverse_transform(transformed.values)
        >>> original.shape
        (5,)
    """

    def __init__(
        self,
        *,
        detection_window: int = 12,
        corr_threshold: float = 0.5,
        transform: Optional[str] = "log",  
        eps: float = 1e-6,
        force_series_type: Optional[str] = None,  
    ):
        """Initializes the SeriesTransformer.

        Args:
            detection_window (int, optional): Window size for rolling statistics. Defaults to 12.
            corr_threshold (float, optional): Correlation threshold for multiplicative detection. Defaults to 0.5.
            transform (Optional[str], optional): Transformation type ('log', 'boxcox', or None). Defaults to "log".
            eps (float, optional): Epsilon for log transformation. Defaults to 1e-6.
            force_series_type (Optional[str], optional): Force series type ('additive' or 'multiplicative'). Defaults to None.
        """
        self.detection_window = detection_window
        self.corr_threshold = corr_threshold
        self.transform = transform
        self.eps = eps
        self.force_series_type = force_series_type

        self.series_type: Optional[str] = None
        self.lambda_: Optional[float] = None

    def detect_series_type(self, y: pd.Series) -> str:
        """Detects the seasonality type of the series.

        Args:
            y (pd.Series): The time series data.

        Returns:
            str: "multiplicative" if correlation > threshold, otherwise "additive".
        """
        if self.force_series_type:
            return self.force_series_type

        roll_mean = y.rolling(self.detection_window).mean()
        roll_std = y.rolling(self.detection_window).std()
        corr = roll_mean.corr(roll_std)

        return "multiplicative" if corr and corr > self.corr_threshold else "additive"

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the transformer and applies transformation to the data.

        Args:
            df (pd.DataFrame): Input DataFrame with time series in first column.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        out = df.copy()
        out.iloc[:, 0] = out.iloc[:, 0].astype(float)  
        y = out.iloc[:, 0]

        self.series_type = self.detect_series_type(y)

        if self.series_type == "multiplicative" and self.transform:
            if self.transform == "log":
                out.iloc[:, 0] = np.log(out.iloc[:, 0] + self.eps)
            elif self.transform == "boxcox":
                out.iloc[:, 0], self.lambda_ = boxcox(out.iloc[:, 0] + self.eps)
            else:
                raise ValueError("transform must be 'log', 'boxcox', or None")

        return out

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Applies inverse transformation to the data.

        Args:
            arr (np.ndarray): Transformed data array.

        Returns:
            np.ndarray: Data in original scale.
        """
        arr = np.asarray(arr)

        if self.series_type == "multiplicative" and self.transform:
            if self.transform == "log":
                return np.exp(arr)
            if self.transform == "boxcox":
                return inv_boxcox(arr, self.lambda_)

        return arr