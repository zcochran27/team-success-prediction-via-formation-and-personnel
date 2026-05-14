"""Assign each player their archetype label using the fit cluster models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def assign_archetypes(
    player_features: pd.DataFrame,
    archetype_models: dict[str, Any],
) -> pd.DataFrame:
    """Add an ``archetype`` column to the player feature table.

    Parameters
    ----------
    player_features
        One row per player with a ``position_group`` column and the feature
        columns the cluster models were trained on.
    archetype_models
        Mapping from position group to fitted archetype clustering model,
        as produced by
        :func:`archetypes.archetype_clustering.load_archetype_clusters`.

    Returns
    -------
    pandas.DataFrame
        Same frame with an added ``archetype`` column (string label of the form
        ``"<position_group>-<cluster_index>"``).
    """
    # TODO: groupby position_group, predict via each group's model, format label.
    raise NotImplementedError


def save_player_archetype_map(
    assignments: pd.DataFrame,
    output_path: Path,
) -> None:
    """Persist the player→archetype mapping (one row per player) to ``output_path``."""
    # TODO: write CSV/parquet with ``player_id, position_group, archetype``.
    raise NotImplementedError


def load_player_archetype_map(input_path: Path) -> pd.DataFrame:
    """Load a previously saved player→archetype mapping."""
    # TODO: read the artifact written by ``save_player_archetype_map``.
    raise NotImplementedError
