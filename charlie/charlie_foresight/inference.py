import numpy as np
import pandas as pd
from typing import Dict, Any

from charlie.charlie_foresight.time_index import make_future_index


def forecast_unseen(
    *,
    series: pd.Series,
    horizon: int,
    models: Dict[str, Any],
    weights: Dict[str, float],
    preprocessor,
):
    """
    Forecast future values using pre-trained models and ensemble weights.

    This function performs NO training, NO tuning, and NO model selection.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have a DatetimeIndex")

    # Apply identical preprocessing
    df = series.to_frame("target")
    df_scaled = preprocessor.transform(df)

    # Build future index
    future_index = make_future_index(series.index, horizon)
    future_df = pd.DataFrame(index=future_index)

    # Weighted ensemble
    ensemble = np.zeros(horizon)
    individual = {}

    for name, model in models.items():
        p_s, _, _ = model.predict(future_df)
        p = preprocessor.inverse_transform(p_s)

        w = weights.get(name, 0.0)
        ensemble += w * p
        individual[name] = p

    return {
        "forecast": ensemble,
        "future_index": future_index,
        "individual_forecasts": individual,
        "weights": weights,
    }

def forecast_recursive(
    *,
    series: pd.Series,
    steps: int,
    block: int,
    models,
    weights,
    preprocessor,
):
    current = series.copy()
    forecasts = []

    for _ in range(steps // block):
        out = forecast_unseen(
            series=current,
            horizon=block,
            models=models,
            weights=weights,
            preprocessor=preprocessor,
        )

        f = pd.Series(out["forecast"], index=out["future_index"])
        forecasts.append(f)
        current = pd.concat([current, f])

    return pd.concat(forecasts)
