import numpy as np
import pytest

from charlie.utils import mean_ensemble, weighted_ensemble, median_ensemble, max_ensemble, rank_ensemble


@pytest.fixture
def sample_probs():
    probs1 = np.array([[0.7, 0.3], [0.4, 0.6]])
    probs2 = np.array([[0.6, 0.4], [0.5, 0.5]])
    probs3 = np.array([[0.9, 0.1], [0.2, 0.8]])
    return [probs1, probs2, probs3]

def test_mean_ensemble(sample_probs):
    out = mean_ensemble(sample_probs[:2])
    expected = np.array([[0.65, 0.35], [0.45, 0.55]])
    np.testing.assert_allclose(out, expected)

def test_weighted_ensemble_uniform(sample_probs):
    out = weighted_ensemble(sample_probs[:2])
    expected = np.array([[0.65, 0.35], [0.45, 0.55]])  # same as mean
    np.testing.assert_allclose(out, expected)

def test_weighted_ensemble_custom_weights(sample_probs):
    out = weighted_ensemble(sample_probs[:2], weights=[0.8, 0.2])
    expected = np.array([[0.68, 0.32], [0.42, 0.58]])
    np.testing.assert_allclose(out, expected)

def test_weighted_ensemble_weights_sum_to_one(sample_probs):
    out = weighted_ensemble(sample_probs[:2], weights=[2, 3])
    # Should auto-normalize to [0.4, 0.6]
    expected = sample_probs[0]*0.4 + sample_probs[1]*0.6
    np.testing.assert_allclose(out, expected)

def test_median_ensemble(sample_probs):
    out = median_ensemble(sample_probs)
    expected = np.array([[0.7, 0.3], [0.4, 0.6]])
    np.testing.assert_allclose(out, expected)

def test_max_ensemble(sample_probs):
    out = max_ensemble(sample_probs[:2])
    expected = np.array([[0.7, 0.4], [0.5, 0.6]])
    np.testing.assert_allclose(out, expected)

def test_rank_ensemble(sample_probs):
    out = rank_ensemble(sample_probs[:2])
    # Ranks: probs1 [[2,1],[1,2]], probs2 [[1,2],[2,1]], mean [[1.5,1.5],[1.5,1.5]]
    expected = np.array([[1.5, 1.5], [1.5, 1.5]])
    np.testing.assert_allclose(out, expected)

def test_ensemble_shape_consistency(sample_probs):
    for func in [mean_ensemble, median_ensemble, max_ensemble, rank_ensemble]:
        out = func(sample_probs)
        assert out.shape == sample_probs[0].shape
    out = weighted_ensemble(sample_probs)
    assert out.shape == sample_probs[0].shape

def test_weighted_ensemble_errors_on_bad_weights(sample_probs):
    with pytest.raises(ValueError):
        # Mismatched weights and arrays
        weighted_ensemble(sample_probs[:2], weights=[1, 2, 3])
