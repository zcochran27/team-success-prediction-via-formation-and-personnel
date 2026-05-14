"""Collapse personnel-only sub-windows into sub-cluster segments.

The raw joint-stable-window pivot in ``lineups.py`` cuts the timeline every
time *either* team substitutes a player, which produces a huge tail of short
windows that differ from their neighbors only in personnel (~92% of all
boundaries are subs, not formation changes). This module merges those
sub-induced windows into longer "sub-cluster" segments.

Algorithm
---------
Within each match, walk snapshots in chronological order. A new segment
starts at any of:

  * match boundary,
  * formation change (hard barrier -- never merged across),
  * a sub event whose snapshot start time is at least ``T`` minutes after
    the *first sub* of the currently-open cluster (it's too late to belong
    to the current cluster, so it opens a new one).

All sub-induced snapshots that fall inside ``[first_sub_min, first_sub_min + T)``
belong to a single cluster and collapse into one segment whose:

  * ``period_start_min`` is the segment's first snapshot's start;
  * ``period_end_min`` is the segment's last snapshot's end;
  * personnel (player / position / archetype, both teams) comes from the
    **last** snapshot of the segment -- i.e., the post-last-sub lineup,
    representing what the coach committed to play with after the sub
    flurry resolved;
  * formation strings, team ids, match id, season are constants within the
    segment by construction;
  * ``home_xg`` / ``away_xg`` / ``home_goals`` / ``away_goals`` are summed
    across all underlying snapshots; differentials and per-30 rates are
    recomputed from the summed totals against the merged duration.

The trade-off vs. a "modal-player" merge: post-cluster players who came on
late in the cluster are credited with the segment's full duration -- the
"extra credit" is bounded above by ``T`` minutes per cluster, by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_PLAYER_COLS = (
    [f"home_player_{n}" for n in range(1, 12)]
    + [f"away_player_{n}" for n in range(1, 12)]
)
_POSITION_COLS = (
    [f"home_position_{n}" for n in range(1, 12)]
    + [f"away_position_{n}" for n in range(1, 12)]
)
_ARCHETYPE_COLS = (
    [f"home_archetype_{n}" for n in range(1, 12)]
    + [f"away_archetype_{n}" for n in range(1, 12)]
)
_FIRST_COLS = [
    "match_id", "season", "period_start_min",
    "home_team_id", "away_team_id",
    "home_formation", "away_formation",
]
_LAST_COLS = _PLAYER_COLS + _POSITION_COLS + _ARCHETYPE_COLS + ["period_end_min"]
_SUM_COLS = ["home_xg", "away_xg", "home_goals", "away_goals"]


def _assign_segment_ids(snaps: pd.DataFrame, T: float) -> np.ndarray:
    """One-pass segment-id assignment following the cluster-collapse rules."""
    n = len(snaps)
    seg_ids = np.empty(n, dtype=np.int64)

    cur_seg = -1
    cluster_first_sub_min: float | None = None
    prev_match = None
    prev_hf = prev_af = None
    prev_personnel: tuple | None = None

    # Pre-extract arrays for speed; iteration through itertuples on a 90k
    # frame is fine but tuple comparisons of 22 player ids each row are the
    # hot loop.
    match_arr = snaps["match_id"].to_numpy()
    start_arr = snaps["period_start_min"].to_numpy()
    hf_arr = snaps["home_formation"].to_numpy()
    af_arr = snaps["away_formation"].to_numpy()
    player_arrays = [snaps[c].fillna(-1).astype("int64").to_numpy() for c in _PLAYER_COLS]

    for i in range(n):
        personnel_now = tuple(arr[i] for arr in player_arrays)
        same_match = match_arr[i] == prev_match
        form_chg = same_match and (hf_arr[i] != prev_hf or af_arr[i] != prev_af)
        pers_chg = same_match and prev_personnel is not None and personnel_now != prev_personnel
        is_sub = pers_chg and not form_chg

        if not same_match or form_chg:
            cur_seg += 1
            cluster_first_sub_min = None
        elif is_sub:
            if cluster_first_sub_min is None or (start_arr[i] - cluster_first_sub_min) >= T:
                cur_seg += 1
                cluster_first_sub_min = float(start_arr[i])

        seg_ids[i] = cur_seg
        prev_match = match_arr[i]
        prev_hf = hf_arr[i]
        prev_af = af_arr[i]
        prev_personnel = personnel_now

    return seg_ids


def collapse_sub_clusters(snapshots: pd.DataFrame, T: float) -> pd.DataFrame:
    """Merge personnel-only sub-windows into sub-cluster segments.

    ``snapshots`` is the home/away wide table emitted by
    ``add_season_and_archetypes`` (one row per joint-stable window). Returns
    one row per merged segment, same schema, with derived per-30 / diff
    columns recomputed.
    """
    snaps = snapshots.sort_values(["match_id", "period_start_min"]).reset_index(drop=True)
    snaps["_seg"] = _assign_segment_ids(snaps, T)

    agg_spec: dict[str, str] = {}
    agg_spec.update({c: "first" for c in _FIRST_COLS if c in snaps.columns})
    agg_spec.update({c: "last" for c in _LAST_COLS if c in snaps.columns})
    agg_spec.update({c: "sum" for c in _SUM_COLS if c in snaps.columns})

    merged = snaps.groupby("_seg", sort=True, as_index=False).agg(agg_spec)
    merged = merged.drop(columns="_seg")

    merged["period_duration_min"] = merged["period_end_min"] - merged["period_start_min"]
    merged["xg_home_minus_away"]    = merged["home_xg"]    - merged["away_xg"]
    merged["xg_away_minus_home"]    = -merged["xg_home_minus_away"]
    merged["goals_home_minus_away"] = merged["home_goals"] - merged["away_goals"]
    merged["goals_away_minus_home"] = -merged["goals_home_minus_away"]

    scale = 30.0 / merged["period_duration_min"]
    for col in [
        "home_xg", "away_xg", "home_goals", "away_goals",
        "xg_home_minus_away", "xg_away_minus_home",
        "goals_home_minus_away", "goals_away_minus_home",
    ]:
        merged[f"{col}_per_30"] = merged[col] * scale

    # Restore nullable Int64 on player columns so empty slots round-trip as
    # NaN rather than -1 once the agg-induced cast is undone.
    for c in _PLAYER_COLS:
        if c in merged.columns:
            s = merged[c]
            merged[c] = s.where(s != -1).astype("Int64")
    merged["home_goals"] = merged["home_goals"].astype(int)
    merged["away_goals"] = merged["away_goals"].astype(int)

    return merged
