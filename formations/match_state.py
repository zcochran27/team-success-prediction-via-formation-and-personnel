"""Augment the lineup-snapshot table with in-window match-state features.

For each snapshot row -- one ``(match, joint-stable window)`` -- this module
sums the shots taken during the window and attaches per-team xG and goal
counts, plus the home-minus-away / away-minus-home differentials.

Output columns added:
  home_xg, away_xg, home_goals, away_goals,
  xg_home_minus_away, xg_away_minus_home,
  goals_home_minus_away, goals_away_minus_home

Penalty-shootout events (``matchPeriod == "P"``) are excluded; their clock is
not continuous with the snapshot timeline. Extra time (``1E`` / ``2E``) is
included since its clock continues past 90'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Shot event columns we need from the raw events log.
_SHOT_EVENT_COLUMNS = [
    "matchId", "type_primary", "matchPeriod", "matchTimestamp",
    "team_id", "shot_xg", "shot_isGoal",
]


def _timestamp_to_minutes(ts: pd.Series) -> pd.Series:
    """Parse ``"HH:MM:SS.ms"`` strings into continuous match minutes (float).

    The Wyscout event clock starts at 0 at first-half kickoff and resumes at
    ~45' at second-half kickoff (halftime doesn't accumulate), so the
    resulting minute aligns with the formations table's ``period_start_min``.
    """
    parts = ts.str.split(":", expand=True)
    h = parts[0].astype(float)
    m = parts[1].astype(float)
    s = parts[2].astype(float)
    return h * 60.0 + m + s / 60.0


def _load_shot_log(events_path: Path, match_ids: set[int]) -> pd.DataFrame:
    """Read only the shot rows we care about and project to the columns we use."""
    ev = pd.read_parquet(events_path, columns=_SHOT_EVENT_COLUMNS)
    shots = ev[
        (ev["type_primary"] == "shot")
        & ev["matchId"].isin(match_ids)
        & (ev["matchPeriod"] != "P")
    ].copy()
    shots["time_min"] = _timestamp_to_minutes(shots["matchTimestamp"])
    shots["is_goal"] = (shots["shot_isGoal"] == True).astype(int)  # noqa: E712 (object dtype)
    return shots[["matchId", "team_id", "time_min", "shot_xg", "is_goal"]].rename(
        columns={"matchId": "match_id", "shot_xg": "xg"}
    )


def add_match_state_features(
    snapshots: pd.DataFrame,
    events_path: Path,
) -> pd.DataFrame:
    """Return ``snapshots`` with xG / goal columns appended per joint window.

    ``snapshots`` must have ``match_id``, ``period_start_min``,
    ``period_end_min``, ``home_team_id``, ``away_team_id``. The function
    reads shots straight from ``events_path`` (a parquet file in the schema
    of ``data/raw/all_events.parquet``).
    """
    snap = snapshots.copy().reset_index(drop=True)
    snap["_snap_id"] = snap.index

    shots = _load_shot_log(events_path, set(snap["match_id"].unique()))

    # Bucket every shot into the snapshot whose [start, end) interval covers
    # its time, within the same match. merge_asof finds the closest preceding
    # period_start_min; we then validate that the shot still lies before
    # period_end_min. merge_asof requires the asof key globally sorted, so
    # sort by it only (the `by` arg handles the per-match grouping).
    snap_sorted = snap.sort_values("period_start_min")
    shots = shots.sort_values("time_min")

    joined = pd.merge_asof(
        shots,
        snap_sorted[["match_id", "period_start_min", "period_end_min",
                     "home_team_id", "away_team_id", "_snap_id"]],
        left_on="time_min",
        right_on="period_start_min",
        by="match_id",
        direction="backward",
    )
    # Drop shots that fall outside any snapshot window (e.g., a shot at 95'
    # when the last period ends at 93' with NaN upper-bound -- those just
    # get filtered out here).
    joined = joined.dropna(subset=["_snap_id"])
    joined = joined[joined["time_min"] < joined["period_end_min"]]

    # Tag each shot as home / away based on which team shot it.
    side = np.where(
        joined["team_id"] == joined["home_team_id"], "home",
        np.where(joined["team_id"] == joined["away_team_id"], "away", None),
    )
    joined["side"] = side
    joined = joined[joined["side"].notna()]

    # Aggregate per (snapshot, side).
    agg = (
        joined.groupby(["_snap_id", "side"])
              .agg(xg=("xg", "sum"), goals=("is_goal", "sum"))
              .reset_index()
    )
    wide = agg.pivot(index="_snap_id", columns="side", values=["xg", "goals"])
    wide.columns = [f"{side}_{stat}" for stat, side in wide.columns]
    wide = wide.reindex(columns=["home_xg", "away_xg", "home_goals", "away_goals"])

    out = snap.merge(wide, left_on="_snap_id", right_index=True, how="left")
    fill = {"home_xg": 0.0, "away_xg": 0.0, "home_goals": 0, "away_goals": 0}
    out = out.fillna(fill)
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    out["xg_home_minus_away"]    = out["home_xg"]    - out["away_xg"]
    out["xg_away_minus_home"]    = out["away_xg"]    - out["home_xg"]
    out["goals_home_minus_away"] = out["home_goals"] - out["away_goals"]
    out["goals_away_minus_home"] = out["away_goals"] - out["home_goals"]

    # Rate-normalize each stat to a per-30-minute basis. Joint-interval
    # construction already guarantees duration > 0, so direct division is safe.
    out["period_duration_min"] = out["period_end_min"] - out["period_start_min"]
    scale = 30.0 / out["period_duration_min"]
    for col in [
        "home_xg", "away_xg", "home_goals", "away_goals",
        "xg_home_minus_away", "xg_away_minus_home",
        "goals_home_minus_away", "goals_away_minus_home",
    ]:
        out[f"{col}_per_30"] = out[col] * scale

    return out.drop(columns=["_snap_id"])
