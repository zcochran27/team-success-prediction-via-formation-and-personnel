"""Tests for :mod:`evaluation.metrics`."""

from __future__ import annotations


def test_perfect_predictions_yield_zero_error_and_r2_one() -> None:
    """When ``y_pred == y_true``, MAE and RMSE are 0 and R^2 is 1."""
    # TODO: assert mae/rmse == 0 and r2 == 1 on identical arrays.
    raise NotImplementedError


def test_compute_metrics_returns_expected_keys() -> None:
    """``compute_metrics`` returns a dict with keys ``mae``, ``rmse``, ``r2``."""
    # TODO: call compute_metrics on a small array and assert key set.
    raise NotImplementedError


def test_build_comparison_table_shape() -> None:
    """Comparison table has one row per model and one column per (target, metric)."""
    # TODO: feed a small results dict, assert resulting DataFrame shape.
    raise NotImplementedError
