"""Aggregate event-instance cluster labels into per-player feature vectors.

For each player, this module:
  1. Looks up the cluster label for each of their events using the models
     fit by :mod:`archetypes.event_clustering`.
  2. Aggregates those labels into a feature vector encoding behavioral
     tendencies: for each event type, ``(% of total actions, % in cluster 1,
     % in cluster 2, ...)``.

The output (one row per player, plus position group label) is the input to
:mod:`archetypes.archetype_clustering`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def assign_event_clusters(
    events: pd.DataFrame,
    cluster_models: dict[tuple[str, str], Any],
) -> pd.DataFrame:
    """Append a ``cluster`` column to ``events`` using the fit cluster models.

    Parameters
    ----------
    events
        Events dataframe with at least ``player_id``, ``position_group``,
        ``event_type``, and the feature columns expected by each model.
    cluster_models
        Mapping ``(position_group, event_type) -> fitted clustering model``
        as returned by :func:`archetypes.event_clustering.load_event_clusters`.

    Returns
    -------
    pandas.DataFrame
        ``events`` with an added ``cluster`` column.
    """
    # TODO: groupby (position_group, event_type) and predict cluster labels.
    raise NotImplementedError


def aggregate_player_vector(
    player_events: pd.DataFrame,
    event_types: list[str],
    n_clusters_per_event: dict[str, int],
) -> pd.Series:
    """Build the feature vector for a single player.

    Parameters
    ----------
    player_events
        All events produced by a single player (with cluster labels).
    event_types
        Canonical ordering of event types for feature layout.
    n_clusters_per_event
        Per-event-type cluster counts, used to enumerate cluster columns.

    Returns
    -------
    pandas.Series
        Concatenation, per event type, of ``[pct_of_total, pct_in_cluster_0,
        pct_in_cluster_1, ...]``.
    """
    # TODO: compute per-event-type and per-cluster proportions.
    raise NotImplementedError


def build_player_feature_table(
    events: pd.DataFrame,
    cluster_models: dict[tuple[str, str], Any],
    event_types: list[str],
    n_clusters_per_event: dict[str, int],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build a player-level feature table for archetype clustering.

    Parameters
    ----------
    events
        Full event dataframe.
    cluster_models
        Output of :func:`archetypes.event_clustering.load_event_clusters`.
    event_types
        Canonical event types.
    n_clusters_per_event
        Per-event-type cluster counts.
    output_path
        Optional path to write the resulting table.

    Returns
    -------
    pandas.DataFrame
        One row per player with ``player_id``, ``position_group``, and feature
        columns.
    """
    # TODO: orchestrate cluster assignment + per-player aggregation.
    raise NotImplementedError
