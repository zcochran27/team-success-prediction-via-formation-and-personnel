"""End-to-end archetype pipeline orchestrator.

Runs the full pipeline against ``data/raw/all_events.parquet``:
  1. Load events, attach ``position_group`` and ``logical_event_type``.
  2. Fit intra-event cluster models per (position group, event type).
  3. Assign event-level cluster labels.
  4. Aggregate to a per-player feature table.
  5. Fit archetype cluster models per position group.
  6. Assign each player an archetype label and save to disk.

Usage (from repo root):
    python -m archetypes.run_pipeline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .archetype_clustering import fit_all_archetype_clusters
from .assign_archetypes import assign_archetypes, save_player_archetype_map
from .event_clustering import fit_all_event_clusters
from .player_aggregation import assign_event_clusters, build_player_feature_table
from .position_groups import (
    assign_position_groups,
    derive_logical_event_type,
)


# Columns we actually need from the raw events parquet.
EVENT_COLUMNS = [
    "player_id", "player_position",
    "type_primary", "type_secondary",
    "start_x", "start_y", "end_x", "end_y",
    "pass_length", "pass_angle", "pass_accurate", "pass_height",
    "shot_xg", "shot_onTarget", "shot_bodyPart",
    "groundDuel_keptPossession", "groundDuel_progressedWithBall",
    "groundDuel_stoppedProgress", "groundDuel_recoveredPossession",
    "groundDuel_takeOn",
]


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_events(raw_path: Path) -> pd.DataFrame:
    print(f"[load] reading {raw_path}")
    df = pd.read_parquet(raw_path, columns=EVENT_COLUMNS)
    print(f"[load] {len(df):,} rows")
    return df


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    print("[prep] assigning position_group + logical_event_type")
    events = assign_position_groups(events, position_col="player_position")
    events["logical_event_type"] = derive_logical_event_type(events)

    # Drop passes whose end location is the origin -- these are unrecorded
    # endpoints in the Wyscout export (~0.37% of passes) and would otherwise
    # contaminate the spatial pass clusters.
    junk_pass = (
        (events["logical_event_type"] == "pass")
        & (events["end_x"] == 0)
        & (events["end_y"] == 0)
    )
    n_junk = int(junk_pass.sum())
    if n_junk:
        events.loc[junk_pass, "logical_event_type"] = pd.NA
        print(f"[prep] dropped {n_junk:,} passes with end at (0, 0)")

    # Keep only the rows that matter for any downstream step.
    keep = events["position_group"].notna() & events["logical_event_type"].notna()
    events = events.loc[keep].copy()
    print(f"[prep] retained {len(events):,} rows after group/type filter")
    print("       group counts:")
    print(events["position_group"].value_counts().to_string())
    print("       logical_event_type counts:")
    print(events["logical_event_type"].value_counts().to_string())
    return events


def run(config_path: Path = Path("configs/config.yaml")) -> None:
    cfg = load_config(config_path)

    raw_dir = Path(cfg["data_paths"]["raw"])
    artifacts_dir = Path(cfg["data_paths"]["archetype_artifacts"])
    event_models_dir = artifacts_dir / "event_clusters"
    archetype_models_dir = artifacts_dir / "archetype_clusters"

    event_types_per_group = cfg["event_clustering"]["event_types_per_group"]
    n_clusters_per_event = cfg["event_clustering"]["n_clusters_per_event"]
    n_archetypes_per_group = cfg["archetype_clustering"]["n_archetypes_per_group"]
    min_events = cfg["event_clustering"].get("min_events_per_player", 100)
    rs_event = cfg["event_clustering"].get("random_state", 0)
    rs_arch = cfg["archetype_clustering"].get("random_state", 0)

    events = load_events(raw_dir / "all_events.parquet")
    events = prepare_events(events)

    print("\n[step 1/3] fitting intra-event cluster models")
    event_models = fit_all_event_clusters(
        events,
        event_types_per_group=event_types_per_group,
        n_clusters_per_event=n_clusters_per_event,
        output_dir=event_models_dir,
        random_state=rs_event,
    )

    print("\n[step 2/3] assigning event clusters + building player feature table")
    events = assign_event_clusters(events, event_models)
    player_features = build_player_feature_table(
        events,
        event_types_per_group=event_types_per_group,
        n_clusters_per_event=n_clusters_per_event,
        min_events_per_player=min_events,
        output_path=artifacts_dir / "player_features.parquet",
    )
    print(f"       player feature table: {player_features.shape}")
    print(player_features["position_group"].value_counts().to_string())

    print("\n[step 3/3] fitting archetype cluster models + assigning labels")
    archetype_models = fit_all_archetype_clusters(
        player_features,
        event_types_per_group=event_types_per_group,
        n_clusters_per_event=n_clusters_per_event,
        n_archetypes_per_group=n_archetypes_per_group,
        output_dir=archetype_models_dir,
        random_state=rs_arch,
    )
    assignments = assign_archetypes(player_features, archetype_models)
    save_player_archetype_map(assignments, artifacts_dir / "player_archetype_map.parquet")

    print("\n[done] artifacts written to", artifacts_dir)
    print("       archetype distribution:")
    print(assignments["archetype"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    run()
