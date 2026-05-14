"""Model 3 -- Tabular, archetypes, ego (focal team only).

Same shape as Model 1, but each slot's personnel is encoded by the
player's *per-season archetype label* (e.g. ``"CD-2"``, ``"CM-1"``)
instead of their raw position. The archetype label comes from the
per-season clustering pipeline in ``archetypes/`` and represents a
behavioral profile rather than a slot label.

Feature columns (12 total):

  * ``team1_formation``       -- categorical formation string
  * ``team1_archetype_1..11`` -- 11 categorical archetype labels for the
    focal team's slot occupants. Slots whose player had too few events
    in the season to clear the archetype-pipeline threshold get the
    string ``"MISSING"`` (set by ``coerce_categoricals``); this is
    informative -- it flags low-data players, not bugs.
  * ``team1_is_home``         -- binary home / away flag

Comparison with Model 1 isolates the marginal value of archetype-based
personnel encoding over raw positional labels, holding the ego /
matchup axis constant.
"""

from __future__ import annotations

import pandas as pd

from ._tabular_common import (
    GROUP_COL, TARGET_COL,
    coerce_categoricals, cross_validate_xgb, fit_xgb,
)


NAME = "tab_arch_ego"

FORMATION_COLS = ["team1_formation"]
ARCHETYPE_COLS = [f"team1_archetype_{n}" for n in range(1, 12)]
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
