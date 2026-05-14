"""Tag each snapshot with its season and per-slot player archetypes.

For each row of the lineup-snapshot table:
  - attach a ``season`` column (calendar year, ``int``) derived from
    ``matches.seasonId -> seasons.name``;
  - for each of the 22 player slots (``home_player_1..11`` /
    ``away_player_1..11``), look up that player's archetype label for
    the match's season from the per-season archetype map and append it
    as ``home_archetype_<n>`` / ``away_archetype_<n>``.

Slots whose player did not clear the archetype pipeline's per-season event
threshold (or whose player wasn't recorded in the events log at all) get
``NaN`` for the archetype -- this is normal and informative.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _match_to_season(matches_path: Path, seasons_path: Path) -> pd.Series:
    """Return a ``match_id -> season_year`` Series, where ``season_year`` is
    the 4-digit year extracted from ``seasons.name`` (e.g. ``"2024 Fall" -> 2024``).
    """
    matches = pd.read_parquet(matches_path, columns=["wyId", "seasonId"])
    seasons = pd.read_parquet(seasons_path, columns=["seasonId", "name"])
    seasons["season"] = seasons["name"].str.extract(r"(\d{4})")[0].astype(int)
    return (
        matches.merge(seasons[["seasonId", "season"]], on="seasonId", how="left")
               .set_index("wyId")["season"]
    )


def add_season_and_archetypes(
    snapshots: pd.DataFrame,
    matches_path: Path,
    seasons_path: Path,
    archetype_map_path: Path,
) -> pd.DataFrame:
    """Return ``snapshots`` with a ``season`` column and 22 archetype columns.

    Slot ``n`` for the home team gains ``home_archetype_<n>``; same on
    the away side. NaNs are preserved when a player has no archetype for
    that season.
    """
    out = snapshots.copy().reset_index(drop=True)
    out["_snap_id"] = out.index

    # 1. Attach season to each snapshot via match_id.
    season_by_match = _match_to_season(matches_path, seasons_path)
    out["season"] = out["match_id"].map(season_by_match).astype("Int64")

    # 2. Build a (player_id, season) -> archetype lookup and join it into
    #    each of the 22 player slots. We melt to long, join once, pivot back.
    amap = pd.read_parquet(archetype_map_path, columns=["player_id", "season", "archetype"])

    slot_frames: list[pd.DataFrame] = []
    for side in ("home", "away"):
        for n in range(1, 12):
            slot_frames.append(
                out[["_snap_id", "season", f"{side}_player_{n}"]]
                .rename(columns={f"{side}_player_{n}": "player_id"})
                .assign(side=side, slot=n)
            )
    long = pd.concat(slot_frames, ignore_index=True)
    # Cast to a common int dtype so the merge keys line up. NaN players stay
    # NaN; they'll simply not match and get NaN archetype.
    long["player_id"] = long["player_id"].astype("Int64")
    long["season"] = long["season"].astype("Int64")
    amap = amap.astype({"player_id": "Int64", "season": "Int64"})

    long = long.merge(amap, on=["player_id", "season"], how="left")

    # Pivot the archetype column back to wide, indexed by snapshot row.
    pivot = long.pivot(index="_snap_id", columns=["side", "slot"], values="archetype")
    pivot.columns = [f"{side}_archetype_{slot}" for side, slot in pivot.columns]
    # Restore the natural column order: home_archetype_1..11, then away.
    ordered = [f"home_archetype_{n}" for n in range(1, 12)] + \
              [f"away_archetype_{n}" for n in range(1, 12)]
    pivot = pivot.reindex(columns=ordered)

    out = out.merge(pivot, left_on="_snap_id", right_index=True, how="left")
    return out.drop(columns=["_snap_id"])
