"""Model 1+stats -- Tabular, raw positions, ego, with per-season stats.

Same column contract as :mod:`models.tab_pos_ego` (formation + 11 raw
position labels + ``team1_is_home``) plus the focal team's 11 × 10
per-season behavioral stat vector flattened across slot × stat. The
stat block adds 110 numeric columns; see
:mod:`models._tabular_stats.STAT_COLS` for the per-slot fields.

This isolates the marginal value of continuous per-player behavior
on top of the categorical position-token baseline, holding ego /
matchup and position-vs-archetype constant.
"""

from __future__ import annotations

import pandas as pd

from ._tabular_common import (
    GROUP_COL, TARGET_COL,
    coerce_categoricals, cross_validate_xgb, fit_xgb,
)
from ._tabular_stats import attach_stats, stat_columns_for


NAME = "tab_pos_ego_stats"

FORMATION_COLS = ["team1_formation"]
POSITION_COLS = [f"team1_position_{n}" for n in range(1, 12)]
CATEGORICAL_COLS = FORMATION_COLS + POSITION_COLS
NUMERIC_COLS = ["team1_is_home"]
STATS_COLS = stat_columns_for(("team1",))
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS + STATS_COLS


def build_features(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Project the snapshot table to ``(X, y, groups)`` for this model."""
    aug = attach_stats(snapshots, sides=("team1",))
    X = aug[FEATURE_COLS]
    X = coerce_categoricals(X, CATEGORICAL_COLS)
    y = aug[TARGET_COL]
    groups = aug[GROUP_COL]
    return X, y, groups


def cross_validate(snapshots: pd.DataFrame, n_splits: int = 5):
    X, y, groups = build_features(snapshots)
    return cross_validate_xgb(X, y, groups, n_splits=n_splits)


def fit(snapshots: pd.DataFrame):
    X, y, _ = build_features(snapshots)
    return fit_xgb(X, y)
