"""Model 1 -- Tabular, raw positions, ego (focal team only).

The lower-bound baseline. Each training row is one
``(match, joint-stable segment, focal-team perspective)`` snapshot from
``data/processed/lineup_snapshots.parquet``. The feature vector encodes
*only* the focal team's setup -- 12 columns total:

  * ``team1_formation``      -- categorical (e.g. ``"4-2-3-1"``)
  * ``team1_position_1..11`` -- 11 categorical raw Wyscout position labels
    for the slot occupants, slotted 1=GK, 9=ST, 7/11=wingers, etc.
  * ``team1_is_home``        -- binary home / away flag

The opponent's setup is **not** observable. This isolates the predictive
signal available from a unilateral view of the focal team's tactical
shape and personnel.

The target is ``xg_team1_minus_team2_per_30`` -- the focal team's xG
advantage during the segment, per-30 normalized.

Why XGBoost: gradient-boosted trees handle the categorical position
labels natively (via ``enable_categorical=True``), need no embedding
table for the discrete formation strings, and produce a competitive
non-linear baseline against which the matchup / archetype / graph
variants are compared.
"""

from __future__ import annotations

import pandas as pd

from ._tabular_common import (
    GROUP_COL, TARGET_COL,
    coerce_categoricals, cross_validate_xgb, fit_xgb,
)


NAME = "tab_pos_ego"

# Feature contract.
FORMATION_COLS = ["team1_formation"]
POSITION_COLS = [f"team1_position_{n}" for n in range(1, 12)]
CATEGORICAL_COLS = FORMATION_COLS + POSITION_COLS
NUMERIC_COLS = ["team1_is_home"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def build_features(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Project the snapshot table to ``(X, y, groups)`` for this model.

    Returns
    -------
    X : DataFrame
        Feature matrix with categorical dtypes already set.
    y : Series
        Target ``xg_team1_minus_team2_per_30`` values.
    groups : Series
        ``match_id`` for each row -- pass to ``GroupKFold`` to prevent
        train/test leakage between rows from the same match.
    """
    X = snapshots[FEATURE_COLS]
    X = coerce_categoricals(X, CATEGORICAL_COLS)
    y = snapshots[TARGET_COL]
    groups = snapshots[GROUP_COL]
    return X, y, groups


def cross_validate(snapshots: pd.DataFrame, n_splits: int = 5):
    """Run the model's blocked k-fold CV. Returns the metric dict."""
    X, y, groups = build_features(snapshots)
    return cross_validate_xgb(X, y, groups, n_splits=n_splits)


def fit(snapshots: pd.DataFrame):
    """Fit the model on all available rows; returns the trained estimator."""
    X, y, _ = build_features(snapshots)
    return fit_xgb(X, y)
