import copy
import pandas as pd
from typing import Dict
from sklearn.model_selection import TimeSeriesSplit
from charlie.charlie_foresight.forecast_models.base import BaseModel


class CrossValidator:
    """A cross-validator for time series forecasting models.

    This class provides time series cross-validation using sklearn's TimeSeriesSplit,
    ensuring that validation sets come after training sets chronologically. It can
    evaluate multiple models and perform hyperparameter search if available.

    Attributes:
        cv_kwargs (dict): Keyword arguments passed to TimeSeriesSplit.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from charlie_foresight.forecast_models.prophet import ProphetModel
        >>> data = pd.DataFrame({'y': np.random.randn(100)}, index=pd.date_range('2020-01-01', periods=100))
        >>> cv = CrossValidator(n_splits=3)
        >>> models = {'prophet': ProphetModel()}
        >>> results = cv.evaluate(data, models)
        >>> len(results['prophet'])
        3
    """

    def __init__(self, *, n_splits: int = 3, cv_kwargs: dict | None = None):
        """Initializes the CrossValidator.

        Args:
            n_splits (int, optional): Number of cross-validation splits. Defaults to 3.
            cv_kwargs (dict | None, optional): Additional kwargs for TimeSeriesSplit. Defaults to None.
        """
        self.cv_kwargs = cv_kwargs or {"n_splits": n_splits}

    def split(self, df: pd.DataFrame):
        """Generates cross-validation splits for time series data.

        Args:
            df (pd.DataFrame): Input DataFrame with datetime index.

        Returns:
            list: List of tuples (train_indices, validation_indices).
        """
        cv = TimeSeriesSplit(**self.cv_kwargs)
        return list(cv.split(df))

    def evaluate(self, df: pd.DataFrame, models: Dict[str, BaseModel]):
        """Evaluates multiple forecasting models using cross-validation.

        For each model, performs hyperparameter search if available, then fits and
        predicts on each cross-validation fold.

        Args:
            df (pd.DataFrame): Full dataset with datetime index and target column.
            models (Dict[str, BaseModel]): Dictionary mapping model names to BaseModel instances.

        Returns:
            dict: Dictionary with model names as keys and lists of fold results as values.
                Each fold result is a dict with 'pred', 'lower', 'upper', 'actual' arrays.
        """
        splits = self.split(df)
        results = {name: [] for name in models}

        for name, model in models.items():
            if hasattr(model, "hyperparameter_search"):
                model.hyperparameter_search(df, splits)

            for tr_idx, va_idx in splits:
                tr = df.iloc[tr_idx]
                va = df.iloc[va_idx]

                m = copy.deepcopy(model)
                m.fit(tr)
                p, lo, hi = m.predict(va)

                results[name].append({
                    "pred": p,
                    "lower": lo,
                    "upper": hi,
                    "actual": va.iloc[:, 0].values,
                })

        return results
