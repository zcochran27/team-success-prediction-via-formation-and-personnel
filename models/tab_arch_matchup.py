"""Model 4 -- Tabular, archetypes, matchup (focal team + opponent).

The fully-enriched tabular variant. Each row sees both teams' formations
and both teams' 11-slot archetype encodings -- 24 features total:

  * ``team1_formation``, ``team2_formation``      -- both shape strings
  * ``team1_archetype_1..11``                     -- focal team archetypes
  * ``team2_archetype_1..11``                     -- opponent archetypes
  * ``team1_is_home``                             -- home / away flag

Comparison with the other three tabular models isolates the *joint*
contribution of archetype enrichment and opponent conditioning:

  * vs. Model 3 (matchup gain under archetypes)
  * vs. Model 2 (archetype gain under matchup)
  * vs. Model 1 (combined gain over the bare baseline)

This is the tabular ceiling against which the graph variants (Models
5-8) are subsequently benchmarked.
"""

from __future__ import annotations

import pandas as pd

from ._tabular_common import (
    GROUP_COL, TARGET_COL,
    coerce_categoricals, cross_validate_xgb, fit_xgb,
)


NAME = "tab_arch_matchup"

FORMATION_COLS = ["team1_formation", "team2_formation"]
ARCHETYPE_COLS = (
    [f"team1_archetype_{n}" for n in range(1, 12)]
    + [f"team2_archetype_{n}" for n in range(1, 12)]
)
CATEGORICAL_COLS = FORMATION_COLS + ARCHETYPE_COLS
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
