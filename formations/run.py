"""Build the lineup-snapshot table from ``data/raw/formations.parquet``.

Usage (from repo root):
    python -m formations.run
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .lineups import build_lineup_snapshots
from .match_state import add_match_state_features
from .player_archetypes import add_season_and_archetypes


def run(config_path: Path = Path("configs/config.yaml")) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    raw_dir = Path(cfg["data_paths"]["raw"])
    processed_dir = Path(cfg["data_paths"]["processed"])
    archetype_map = Path(cfg["data_paths"]["archetype_artifacts"]) / "player_archetype_map.parquet"
    out_path = processed_dir / "lineup_snapshots.parquet"

    print(f"[load] {raw_dir / 'formations.parquet'}")
    formations = pd.read_parquet(raw_dir / "formations.parquet")
    print(f"[load] {len(formations):,} rows across {formations['matchId'].nunique()} matches")

    print(f"[load] {raw_dir / 'matches.parquet'}")
    matches = pd.read_parquet(raw_dir / "matches.parquet", columns=["wyId", "home_team", "away_team", "duration"])

    print("[build] pivoting to (match, joint-window) snapshots")
    snaps = build_lineup_snapshots(formations, matches, output_path=None)
    print(f"       {len(snaps):,} snapshot rows")

    print("[feat] computing per-window xG / goals from events")
    snaps = add_match_state_features(snaps, raw_dir / "all_events.parquet")

    print("[feat] attaching season + per-slot archetypes")
    snaps = add_season_and_archetypes(
        snaps,
        matches_path=raw_dir / "matches.parquet",
        seasons_path=raw_dir / "seasons.parquet",
        archetype_map_path=archetype_map,
    )

    snaps.to_parquet(out_path, index=False)
    print(f"[done] -> {out_path}")
    print("       formation pair counts (top 10):")
    pair = snaps["home_formation"].astype(str) + " vs " + snaps["away_formation"].astype(str)
    print(pair.value_counts().head(10).to_string())
    print("       xG / goal totals across all snapshots:")
    print(snaps[["home_xg", "away_xg", "home_goals", "away_goals"]].sum().to_string())


if __name__ == "__main__":
    run()
