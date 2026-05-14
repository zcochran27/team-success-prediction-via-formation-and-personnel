"""Model 2 -- Tabular, raw positions, matchup (focal team + opponent).

Same structure as Model 1 but the feature vector also sees the opponent's
formation and 11-man lineup. Each training row carries 24 features:

  * ``team1_formation``, ``team2_formation``     -- both teams' shape strings
  * ``team1_position_1..11``                     -- focal team slots
  * ``team2_position_1..11``                     -- opponent slots
  * ``team1_is_home``                            -- home / away flag

Conceptually, this asks: *given that you know who I'm playing and how
they're set up, can you predict the xG flow during this segment better
than a unilateral view of just my team?*

Comparison with Model 1 isolates the marginal value of opponent
conditioning under a raw-position personnel encoding. Same XGBoost
hyperparameters, same target, same CV harness (blocked by ``match_id``)
as Model 1.
"""

from __future__ import annotations

import pandas as pd

from ._tabular_common import (
    GROUP_COL, TARGET_COL,
    coerce_categoricals, cross_validate_xgb, fit_xgb,
)


NAME = "tab_pos_matchup"

FORMATION_COLS = ["team1_formation", "team2_formation"]
POSITION_COLS = (
    [f"team1_position_{n}" for n in range(1, 12)]
    + [f"team2_position_{n}" for n in range(1, 12)]
)
CATEGORICAL_COLS = FORMATION_COLS + POSITION_COLS
NUMERIC_COLS = ["team1_is_home"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS


def build_features(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Project the snapshot table to ``(X, y, groups)`` for this model."""
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
