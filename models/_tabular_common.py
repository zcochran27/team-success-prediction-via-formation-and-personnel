"""Shared training + cross-validation utilities for the tabular models.

All four tabular variants (Models 1-4) share the same XGBoost regressor and
the same blocked-by-match k-fold CV harness. The variants differ only in
which columns each one's ``build_features`` selects from the snapshot table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


# Hyperparameters mirror the ``baseline`` block in ``configs/config.yaml``.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 42,
    "n_jobs": -1,
}


def fit_xgb(X: pd.DataFrame, y: pd.Series, **overrides: Any) -> xgb.XGBRegressor:
    """Fit a single XGBoost regressor on the prepared features."""
    params = {**DEFAULT_PARAMS, **overrides}
    model = xgb.XGBRegressor(**params)
    model.fit(X, y)
    return model


def _sign_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of nonzero-target rows where predicted and true sign match.

    Rows with ``y_true == 0`` have no correct direction (no chance creation
    by either team) and are excluded -- they otherwise distort the metric
    because any nonzero prediction is automatically "wrong".
    """
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))


def cross_validate_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    **xgb_overrides: Any,
) -> dict[str, np.ndarray]:
    """Blocked k-fold CV with ``GroupKFold`` on ``groups`` (typically ``match_id``).

    Both focal-perspective rows of any given (match, segment) share the same
    ``match_id`` and therefore land in the same fold -- no train/test leakage
    between perspectives of the same match.

    Returns a dict mapping metric name -> per-fold array.
    """
    gkf = GroupKFold(n_splits=n_splits)
    metrics: dict[str, list[float]] = {"mae": [], "rmse": [], "r2": [], "sign_acc": []}
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = fit_xgb(X.iloc[train_idx], y.iloc[train_idx], **xgb_overrides)
        preds = model.predict(X.iloc[test_idx])
        y_test = y.iloc[test_idx].to_numpy()
        metrics["mae"].append(mean_absolute_error(y_test, preds))
        metrics["rmse"].append(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["r2"].append(r2_score(y_test, preds))
        metrics["sign_acc"].append(_sign_accuracy(y_test, preds))
    return {k: np.asarray(v) for k, v in metrics.items()}


def coerce_categoricals(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast string-typed columns to ``category`` dtype so XGBoost's native
    categorical handling kicks in. Fills NaN with a sentinel string first
    because the categorical dtype's NaN-in-categories behavior is sharp
    enough to surprise downstream code.
    """
    X = X.copy()
    for c in columns:
        X[c] = X[c].astype("object").fillna("MISSING").astype("category")
    return X


TARGET_COL = "xg_team1_minus_team2_per_30"
GROUP_COL = "match_id"
