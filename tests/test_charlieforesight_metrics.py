import math
import numpy as np
import pytest

from charlie.charlie_foresight.metrics import (
    mse,
    rmse,
    mae,
    mape,
    mase,
)

def test_mse_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    # errors: 0, 0, 1 → mean = 1/3
    assert mse(y_true, y_pred) == pytest.approx(1 / 3)

def test_rmse_basic():
    y_true = [1, 2, 3]
    y_pred = [1, 2, 4]
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(1 / 3))

def test_mae_basic():
    y_true = [1, 2, 3]
    y_pred = [2, 2, 4]
    # errors: 1, 0, 1 → mean = 2/3
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)

def test_mape_basic():
    y_true = [10, 20]
    y_pred = [11, 18]
    # errors: 10%, 10% → mean = 10
    assert mape(y_true, y_pred) == pytest.approx(10.0)

def test_mape_all_zero_true_returns_nan():
    y_true = [0, 0, 0]
    y_pred = [1, 2, 3]
    assert math.isnan(mape(y_true, y_pred))


def test_mase_basic():
    y_true = [1, 2, 3, 4]
    y_pred = [1, 2, 4, 4]

    # naive errors: |2-1|, |3-2|, |4-3| → [1,1,1] → mean = 1
    # MAE = (0 + 0 + 1 + 0) / 4 = 0.25
    assert mase(y_true, y_pred) == pytest.approx(0.25)

def test_mase_too_short_returns_nan():
    assert math.isnan(mase([1], [1]))

def test_mase_zero_naive_denom_returns_nan():
    y_true = [2, 2, 2, 2]
    y_pred = [2, 2, 2, 2]
    assert math.isnan(mase(y_true, y_pred))
