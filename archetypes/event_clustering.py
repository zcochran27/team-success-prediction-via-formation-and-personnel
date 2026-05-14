"""Intra-event clustering.

For each (position group, event type) pair, fit a clustering model over
individual event instances to recover qualitatively distinct event subtypes
(e.g. short vs. long passes, progressive vs. lateral dribbles).

The fitted cluster models are persisted to disk so that
:mod:`archetypes.player_aggregation` can reuse them when assigning cluster
labels to events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_event_feature_matrix(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Extract a numeric feature matrix for clustering events of ``event_type``.

    Parameters
    ----------
    events
        Filtered event dataframe (already restricted to a position group).
    event_type
        Event type to extract features for (e.g. ``"pass"``).

    Returns
    -------
    pandas.DataFrame
        Rows = event instances, columns = numeric features (e.g. start/end x/y,
        length, angle, outcome flags).
    """
    # TODO: per-event-type feature engineering.
    raise NotImplementedError


def fit_event_clusters(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 0,
) -> Any:
    """Fit a clustering model (default: KMeans) on event-instance features.

    Parameters
    ----------
    features
        Output of :func:`build_event_feature_matrix`.
    n_clusters
        Number of clusters to fit.
    random_state
        Seed for reproducibility.

    Returns
    -------
    Any
        A fitted clustering model exposing ``.predict``.
    """
    # TODO: instantiate, scale, fit clustering model.
    raise NotImplementedError


def fit_all_event_clusters(
    events: pd.DataFrame,
    position_groups: list[str],
    event_types: list[str],
    n_clusters_per_event: dict[str, int],
    output_dir: Path,
) -> dict[tuple[str, str], Any]:
    """Fit one clustering model per (position group, event type) and persist them.

    Parameters
    ----------
    events
        Full event dataframe (will be filtered internally).
    position_groups
        Position groups to iterate over.
    event_types
        Event types to cluster within each position group.
    n_clusters_per_event
        Mapping from event type to desired number of clusters.
    output_dir
        Directory to serialize fitted models to.

    Returns
    -------
    dict
        Mapping from ``(position_group, event_type)`` to fitted model.
    """
    # TODO: loop over (group, event_type), call fit_event_clusters, save to disk.
    raise NotImplementedError


def load_event_clusters(input_dir: Path) -> dict[tuple[str, str], Any]:
    """Load previously fit (position group, event type) cluster models from disk."""
    # TODO: deserialize models written by ``fit_all_event_clusters``.
    raise NotImplementedError
