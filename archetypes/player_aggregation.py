"""Aggregate event cluster labels into per-player feature vectors.

For each player we look up the cluster label of each event (using the
models from event_clustering.py) and turn those labels into a feature
vector: per event type, the share of total actions and the share falling in
each cluster. The result, one row per player-season, feeds
archetype_clustering.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .event_clustering import predict_event_clusters
from .position_groups import filter_events_by_group


def assign_event_clusters(
    events: pd.DataFrame,
    cluster_models: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    """Add a cluster column to events using the fitted cluster models.

    events must already have position_group and logical_event_type columns.
    Rows with no model for their (group, event_type) get cluster = -1.
    """
    out = events.copy()
    out["cluster"] = -1
    for (group, et), model in cluster_models.items():
        mask = (out["position_group"] == group) & (out["logical_event_type"] == et)
        if not mask.any():
            continue
        labels = predict_event_clusters(out.loc[mask], model, et)
        out.loc[mask, "cluster"] = labels
    return out


def aggregate_player_vector(
    player_events: pd.DataFrame,
    event_types: list[str],
    n_clusters_per_event: dict[str, int],
) -> pd.Series:
    """Build the feature vector for a single player.

    In event-type order: [pct_total_<et>, pct_cluster_<et>_0,
    pct_cluster_<et>_1, ...] for each event type. If the player has no events
    of a type, that block is all zeros.
    """
    total = max(len(player_events), 1)
    parts: dict[str, float] = {}
    for et in event_types:
        et_events = player_events[player_events["logical_event_type"] == et]
        n_et = len(et_events)
        parts[f"pct_total_{et}"] = n_et / total
        k = n_clusters_per_event[et]
        if n_et == 0:
            for i in range(k):
                parts[f"pct_cluster_{et}_{i}"] = 0.0
        else:
            counts = et_events["cluster"].value_counts(normalize=True)
            for i in range(k):
                parts[f"pct_cluster_{et}_{i}"] = float(counts.get(i, 0.0))
    return pd.Series(parts)


def build_player_feature_table(
    events: pd.DataFrame,
    event_types_per_group: dict[str, list[str]],
    n_clusters_per_event: dict[str, dict[str, int]],
    min_events_per_player: int = 100,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build a player-season feature table for archetype clustering.

    events must already have position_group, logical_event_type, season, and
    cluster columns. n_clusters_per_event is nested as {group: {event_type: k}}.

    One row per (player_id, season). Each player-season is assigned the
    position group it logged the most relevant events in that season, and the
    feature vector uses only that season's events. A player-season is dropped
    if it has fewer than min_events_per_player relevant events in its primary
    group, so a player can appear in some seasons and not others, and can land
    in different archetypes across seasons.
    """
    # Determine each (player, season)'s primary position group by event
    # volume on the event types relevant to that group, that season.
    relevant_rows = []
    for group, event_types in event_types_per_group.items():
        sub = filter_events_by_group(events, group, event_types)
        relevant_rows.append(
            sub.groupby(["player_id", "season"]).size()
               .rename("n").reset_index().assign(position_group=group)
        )
    counts = pd.concat(relevant_rows, ignore_index=True)
    primary = (
        counts.sort_values("n", ascending=False)
              .drop_duplicates(subset=["player_id", "season"], keep="first")
              .reset_index(drop=True)
    )
    primary = primary[primary["n"] >= min_events_per_player]

    # For each (player, season, primary_group) build the feature vector using
    # only the event types relevant to that group, restricted to events the
    # player produced while playing in that group that season.
    rows: list[dict[str, Any]] = []
    for group, gdf in primary.groupby("position_group"):
        event_types = event_types_per_group[group]
        keys = set(zip(gdf["player_id"], gdf["season"]))
        ev = events[
            (events["position_group"] == group)
            & (events["logical_event_type"].isin(event_types))
        ]
        group_k = n_clusters_per_event[group]
        for (pid, season), pdf in ev.groupby(["player_id", "season"]):
            if (pid, season) not in keys:
                continue
            vec = aggregate_player_vector(pdf, event_types, group_k)
            # Most common raw Wyscout position this season, among the events
            # that fed the feature vector. Ties broken by first occurrence.
            pos_mode = pdf["player_position"].mode()
            player_position = pos_mode.iloc[0] if not pos_mode.empty else pd.NA
            rows.append({
                "player_id": pid,
                "season": int(season),
                "position_group": group,
                "player_position": player_position,
                **vec.to_dict(),
            })

    table = pd.DataFrame(rows).fillna(0.0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(output_path, index=False)

    return table


def feature_columns_for_group(
    event_types: list[str],
    n_clusters_per_event: dict[str, int],
) -> list[str]:
    """Return the ordered feature-column names for a position group."""
    cols: list[str] = []
    for et in event_types:
        cols.append(f"pct_total_{et}")
        cols.extend(f"pct_cluster_{et}_{i}" for i in range(n_clusters_per_event[et]))
    return cols
