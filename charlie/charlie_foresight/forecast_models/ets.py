import numpy as np
import pandas as pd
from typing import Optional, Callable
import statsmodels.api as sm
from charlie.charlie_foresight.forecast_models.base import BaseModel

class ETSModel(BaseModel):
    """A forecasting model using Exponential Smoothing (ETS).

    This class implements ETS (Error, Trend, Seasonal) forecasting using statsmodels'
    ExponentialSmoothing. It can automatically determine seasonality based on data length
    and seasonal periods.

    Attributes:
        seasonal_periods (Optional[int]): Number of periods in a seasonal cycle.
        seasonal_policy (Optional[Callable]): Function to determine seasonal component type.
        ets_kwargs (dict): Additional keyword arguments for ExponentialSmoothing.
        fit_kwargs (dict): Keyword arguments for model fitting.
        model: The fitted ExponentialSmoothing model instance.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> train_data = pd.DataFrame({'y': np.random.randn(100)}, index=pd.date_range('2020-01-01', periods=100))
        >>> model = ETSModel(seasonal_periods=12)
        >>> model.fit(train_data)
        >>> future = pd.DataFrame(index=pd.date_range('2020-04-11', periods=10))
        >>> preds, lower, upper = model.predict(future)
        >>> preds.shape
        (10,)
    """

    def __init__(
        self,
        *,
        seasonal_periods: Optional[int] = None,
        seasonal_policy: Optional[Callable[[int, Optional[int]], Optional[str]]] = None,
        ets_kwargs: Optional[dict] = None,
        fit_kwargs: Optional[dict] = None,
    ):
        """Initializes the ETSModel.

        Args:
            seasonal_periods (Optional[int], optional): Number of seasonal periods. Defaults to None.
            seasonal_policy (Optional[Callable], optional): Function determining seasonal type. Defaults to None.
            ets_kwargs (Optional[dict], optional): ExponentialSmoothing kwargs. Defaults to None.
            fit_kwargs (Optional[dict], optional): Fitting kwargs. Defaults to None.
        """
        self.seasonal_periods = seasonal_periods
        self.seasonal_policy = seasonal_policy
        self.ets_kwargs = ets_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits the ETS model to the training data.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.
        """
        y = train_df.iloc[:, 0]
        n = len(y)

        seasonal = None
        if self.seasonal_policy:
            seasonal = self.seasonal_policy(n, self.seasonal_periods)
        elif self.seasonal_periods and n >= 2 * self.seasonal_periods:
            seasonal = "add"

        self.model = sm.tsa.ExponentialSmoothing(
            y,
            seasonal=seasonal,
            seasonal_periods=self.seasonal_periods if seasonal else None,
            **self.ets_kwargs,
        ).fit(**self.fit_kwargs)

    def predict(self, future_df: pd.DataFrame):
        """Makes predictions using the fitted ETS model.

        Args:
            future_df (pd.DataFrame): Future data for prediction with datetime index.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of predictions, lower confidence intervals,
                and upper confidence intervals.
        """
        steps = len(future_df)
        preds = self.model.forecast(steps)
        resid_std = np.std(self.model.resid) if hasattr(self.model, "resid") else np.std(preds) * 0.1
        lower = preds - 1.96 * resid_std
        upper = preds + 1.96 * resid_std
        return preds.values, lower.values, upper.values