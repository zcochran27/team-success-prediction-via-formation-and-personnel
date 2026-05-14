"""Cluster the per-player feature vectors into archetypes, per position group.

Archetypes are position-group-specific: clustering happens separately within
each of the 5 position groups, so an archetype label is only meaningful in the
context of its group (e.g. ``CD-0`` vs. ``CM-0`` are unrelated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def fit_archetype_clusters(
    player_features: pd.DataFrame,
    position_group: str,
    n_clusters: int,
    random_state: int = 0,
) -> Any:
    """Fit a clustering model on the player feature vectors of a single position group.

    Parameters
    ----------
    player_features
        Output of :func:`archetypes.player_aggregation.build_player_feature_table`,
        filtered to a single position group.
    position_group
        The position group being clustered (for logging / model metadata).
    n_clusters
        Number of archetypes to fit for this group.
    random_state
        Seed for reproducibility.

    Returns
    -------
    Any
        A fitted clustering model exposing ``.predict``.
    """
    # TODO: scale features, fit clustering model.
    raise NotImplementedError


def fit_all_archetype_clusters(
    player_features: pd.DataFrame,
    n_archetypes_per_group: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    """Fit one archetype clustering model per position group and persist them.

    Parameters
    ----------
    player_features
        Player-level feature table (all groups).
    n_archetypes_per_group
        Mapping from position group to desired number of archetypes.
    output_dir
        Directory to serialize fitted models to.

    Returns
    -------
    dict
        Mapping from position group to fitted clustering model.
    """
    # TODO: loop over groups, call fit_archetype_clusters, save to disk.
    raise NotImplementedError


def load_archetype_clusters(input_dir: Path) -> dict[str, Any]:
    """Load previously fit per-position-group archetype models from disk."""
    # TODO: deserialize models written by ``fit_all_archetype_clusters``.
    raise NotImplementedError
