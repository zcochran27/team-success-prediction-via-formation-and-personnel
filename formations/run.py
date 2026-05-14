"""Build the final lineup-snapshot training table.

Pipeline (single end-to-end run; no intermediate artifacts are written):

  1. Pivot ``data/raw/formations.parquet`` into one row per
     ``(match, joint-stable window)`` with both teams' 11-player lineups
     slotted 1..11.
  2. Add per-window xG and goal aggregates from the events log.
  3. Attach the match's season and each player's per-season archetype.
  4. Reframe the home/away wide row into two focal-team rows per snapshot
     (``team1`` = the team being analyzed, ``team2`` = opponent), tagged
     with a ``team1_is_home`` binary.

Output: ``data/processed/lineup_snapshots.parquet``.

Usage (from repo root):
    python -m formations.run
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .cluster_collapse import collapse_sub_clusters
from .lineups import build_lineup_snapshots
from .match_state import add_match_state_features
from .player_archetypes import add_season_and_archetypes


_DIFF_SUBSTR = {
    "home_minus_away": "team1_minus_team2",
    "away_minus_home": "team2_minus_team1",
}
_DIFF_SUBSTR_FLIPPED = {
    "home_minus_away": "team2_minus_team1",
    "away_minus_home": "team1_minus_team2",
}


def _rename_one_perspective(df: pd.DataFrame, team1_is_home: bool) -> pd.DataFrame:
    """Relabel ``home_``/``away_`` columns into ``team1_``/``team2_``.

    Values are never modified -- the underlying ``home_xg`` *is* team1's xG
    when team1 is the home team, and is team2's xG otherwise -- so only
    column labels change. Differential substrings (``home_minus_away`` and
    ``away_minus_home``) are rewritten consistently with the prefix swap.
    """
    diff_map = _DIFF_SUBSTR if team1_is_home else _DIFF_SUBSTR_FLIPPED
    home_target = "team1_" if team1_is_home else "team2_"
    away_target = "team2_" if team1_is_home else "team1_"

    renames: dict[str, str] = {}
    for c in df.columns:
        new_c = c
        for src, dst in diff_map.items():
            if src in new_c:
                new_c = new_c.replace(src, dst)
        if new_c.startswith("home_"):
            new_c = home_target + new_c[len("home_"):]
        elif new_c.startswith("away_"):
            new_c = away_target + new_c[len("away_"):]
        renames[c] = new_c

    out = df.rename(columns=renames)
    out["team1_is_home"] = int(team1_is_home)
    return out


def _canonical_column_order(columns: list[str]) -> list[str]:
    """Group columns as match-metadata | team1 features | team2 features | targets."""
    meta = [
        "match_id", "season",
        "period_start_min", "period_end_min", "period_duration_min",
        "team1_is_home",
    ]

    def _split(side: str) -> list[str]:
        out: list[str] = []
        if f"{side}_team_id" in columns:
            out.append(f"{side}_team_id")
        if f"{side}_formation" in columns:
            out.append(f"{side}_formation")
        for kind in ("player", "position", "archetype"):
            for n in range(1, 12):
                col = f"{side}_{kind}_{n}"
                if col in columns:
                    out.append(col)
        for c in columns:
            if c.startswith(f"{side}_") and c not in out and c not in meta:
                out.append(c)
        return out

    team1 = _split("team1")
    team2 = _split("team2")
    placed = set(meta) | set(team1) | set(team2)
    remaining = [c for c in columns if c not in placed]
    ordered = meta + team1 + team2 + remaining
    return [c for c in ordered if c in columns]


def _to_focal_perspective(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Double the snapshot table into focal-team perspective rows."""
    home_view = _rename_one_perspective(snapshots, team1_is_home=True)
    away_view = _rename_one_perspective(snapshots, team1_is_home=False)
    doubled = pd.concat([home_view, away_view], ignore_index=True)
    return doubled[_canonical_column_order(list(doubled.columns))]


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

    T = float(cfg["formations"]["sub_cluster_window_min"])
    print(f"[merge] collapsing sub-clusters with window T={T} min")
    before = len(snaps)
    snaps = collapse_sub_clusters(snaps, T)
    print(f"       {before:,} -> {len(snaps):,} segments  ({1 - len(snaps)/before:.1%} reduction)")

    print("[focal] reframing to (team1=focal, team2=opponent) perspective")
    focal = _to_focal_perspective(snaps)
    focal.to_parquet(out_path, index=False)
    print(f"[done] {len(focal):,} focal rows -> {out_path}")


if __name__ == "__main__":
    run()
