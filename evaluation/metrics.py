"""Regression metrics and cross-model comparison-table formatting."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    # TODO: ``np.mean(np.abs(y_true - y_pred))``.
    raise NotImplementedError


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    # TODO: ``np.sqrt(np.mean((y_true - y_pred) ** 2))``.
    raise NotImplementedError


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    # TODO: 1 - SS_res / SS_tot.
    raise NotImplementedError


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return ``{"mae": ..., "rmse": ..., "r2": ...}``."""
    # TODO: call the three metric functions and bundle results.
    raise NotImplementedError


def build_comparison_table(
    results: dict[tuple[str, str], dict[str, float]],
    model_order: Iterable[str] | None = None,
    target_order: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Produce a formatted comparison table across all models and both targets.

    Parameters
    ----------
    results
        Mapping ``(model_name, target_name) -> metric dict``.
    model_order
        Optional explicit row ordering.
    target_order
        Optional explicit column-group ordering.

    Returns
    -------
    pandas.DataFrame
        Rows = models, columns = (target, metric) MultiIndex.
    """
    # TODO: pivot the results dict into a MultiIndex DataFrame.
    raise NotImplementedError
