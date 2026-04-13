import numpy as np
import pandas as pd
from typing import Optional
from charlie.charlie_foresight.forecast_models.base import BaseModel
from charlie.charlie_foresight.utils import to_prophet_df

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    Prophet = None
    _HAS_PROPHET = False


class ProphetModel(BaseModel):
    """A forecasting model using Facebook Prophet.

    This class provides an interface to Facebook Prophet for time series forecasting,
    wrapping it to be compatible with the BaseModel interface.

    Attributes:
        prophet_kwargs (dict): Keyword arguments passed to Prophet initialization.
        fit_kwargs (dict): Keyword arguments passed to Prophet fit method.
        model: The fitted Prophet model instance (None until fit is called).

    Examples:
        >>> import pandas as pd
        >>> train_data = pd.DataFrame({'y': [1, 2, 3, 4, 5]}, index=pd.date_range('2020-01-01', periods=5))
        >>> model = ProphetModel()
        >>> model.fit(train_data)
        >>> future = pd.DataFrame(index=pd.date_range('2020-01-06', periods=3))
        >>> preds, lower, upper = model.predict(future)
        >>> preds.shape
        (3,)
    """

    def __init__(
        self,
        *,
        prophet_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
    ):
        """Initializes the ProphetModel.

        Args:
            prophet_kwargs (Optional[dict], optional): Keyword arguments for Prophet initialization. Defaults to None.
            fit_kwargs (Optional[dict], optional): Keyword arguments for Prophet fit method. Defaults to None.
        """
        self.prophet_kwargs = prophet_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits the Prophet model to the training data.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.
        """
        if not _HAS_PROPHET:
            raise ImportError("prophet is not installed")

        df = to_prophet_df(train_df)
        self.model = Prophet(**self.prophet_kwargs)
        self.model.fit(df, **self.fit_kwargs)

    def predict(self, future_df: pd.DataFrame):
        """Makes predictions using the fitted Prophet model.

        Args:
            future_df (pd.DataFrame): Future data for prediction with datetime index.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of predictions, lower confidence intervals,
                and upper confidence intervals.
        """
        fut = to_prophet_df(future_df.assign(_dummy=0).iloc[:, :1])
        fc = self.model.predict(fut)
        return (
            fc["yhat"].values,
            fc["yhat_lower"].values,
            fc["yhat_upper"].values,
        )