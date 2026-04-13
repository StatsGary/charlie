import pandas as pd
import numpy as np
from typing import Optional, Tuple


class BaseModel:
    """Base class for forecasting models.

    This abstract base class defines the interface for all forecasting models.
    Subclasses must implement the fit, predict, and hyperparameter_search methods.

    Attributes:
        None

    Examples:
        >>> from charlie_foresight.forecast_models.base import BaseModel
        >>> import numpy as np
        >>> class MyModel(BaseModel):
        ...     def fit(self, train_df):
        ...         pass
        ...     def predict(self, future_df):
        ...         return (np.array([1, 2, 3]), None, None)
        ...     def hyperparameter_search(self, train_df, splits):
        ...         pass
        >>> model = MyModel()
        >>> # model.fit(train_data)  # Would implement actual fitting
    """
    def fit(self, train_df: pd.DataFrame) -> None:
        """Fits the model to the training data.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.

        Examples:
            >>> model = MyModel()
            >>> # model.fit(train_df)  # Fits the model to training data
        """
        raise NotImplementedError

    def predict(
        self,
        future_df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError

    def hyperparameter_search(self, train_df: pd.DataFrame, splits) -> None:
        """Performs hyperparameter search using cross-validation splits.

        Args:
            train_df (pd.DataFrame): Training data with datetime index and target column.
            splits: Cross-validation splits for hyperparameter tuning.
        """
        return None