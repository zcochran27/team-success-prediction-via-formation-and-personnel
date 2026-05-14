"""Pivot the per-player formation log into one row per (match, joint-stable window).

The raw ``formations.parquet`` table is long: one row per (match, team, period,
player) where a "period" is an interval during which that team's formation
didn't change. Each team has its own period boundaries -- team A might switch
shape at 27', team B at 41' -- so the *joint* state of both teams' lineups is
stable only on the intersection of the two timelines.

This module turns that into a wide snapshot table:

  match_id, period_start_min, period_end_min,
  home_team_id, away_team_id, home_formation, away_formation,
  home_player_1 .. home_player_11, home_position_1 .. home_position_11,
  away_player_1 .. away_player_11, away_position_1 .. away_position_11

Slot 1..11 follows the modern English shirt-number convention:
  1=GK, 2=RB, 3=LB, 4=RCB, 5=LCB, 6=CDM, 7=RW, 8=CM, 9=ST, 10=CAM, 11=LW

Numbers are assigned from the raw position label; ambiguous labels (un-prefixed
``CB``, multiple ``CM``s, etc.) are disambiguated using ``position_x`` / rank.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Each raw Wyscout-ish position label maps to one of four lines.
# When a team's final period has ``period_end_min == NaN``, we replace it
# with the match's end-of-clock minute so joint-interval math stays finite.
# ``matches.duration`` only distinguishes regulation vs. ET, so we pick a
# generous stoppage allowance for each category. Penalties are excluded
# from the match clock (the shootout uses its own timeline), so a Penalties
# match's continuous clock still ends at the end of ET.
_DURATION_TO_END_MIN: dict[str, float] = {
    "Regular":   95.0,
    "ExtraTime": 125.0,
    "Penalties": 125.0,
}


_LABEL_TO_LINE: dict[str, str] = {
    "GK":  "GK",
    "CB":  "DEF", "LCB": "DEF", "RCB": "DEF",
    "LB":  "DEF", "RB":  "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM":  "MID", "CAM": "MID", "LAM": "MID", "RAM": "MID",
    "LW":  "FWD", "RW":  "FWD", "LWF": "FWD", "RWF": "FWD",
    "ST":  "FWD", "SS":  "FWD",
}

# Direct label -> shirt-number where the label uniquely identifies the slot.
_LABEL_TO_NUMBER: dict[str, int] = {
    "GK":  1,
    "RB":  2, "RWB": 2,
    "LB":  3, "LWB": 3,
    "RCB": 4,
    "LCB": 5,
    "CDM": 6,
    "RW":  7, "RWF": 7, "RAM": 7,
    "ST":  9,
    "CAM": 10, "SS": 10,
    "LW":  11, "LWF": 11, "LAM": 11,
}


def assign_slot_numbers(team_players: pd.DataFrame) -> dict:
    """Assign shirt numbers 1..11 to a team's players in one period.

    ``team_players`` has one row per player on the pitch with at least the
    columns ``position`` (raw label), ``position_x``, ``position_y``. The
    function returns a dict ``row_label -> slot_number`` covering every row
    in ``team_players`` (using its existing index, so callers can join back).

    Strategy:
      1. Each player's label proposes a preferred number from
         ``_LABEL_TO_NUMBER``.
      2. Walk players in order of *most constrained first* (single-option
         labels before ambiguous ones), assigning each the first free number
         from their preference list.
      3. Ambiguous labels (``CB``, ``CM``) have a multi-number preference
         list ordered by ``position_x`` rank so deeper players land on 4/6
         and more advanced players on 5/8/10.
      4. Anything still unassigned (rare: <11 players on the pitch or an
         unrecognized label) is dropped into the lowest free slot, sorted
         by ``position_x``.
    """
    df = team_players.copy()

    # Per-player preference list.
    prefs: dict = {}
    cb_idx = df.index[df["position"] == "CB"].tolist()
    # Multiple un-prefixed CBs: split by y-rank, deeper-left gets 5, deeper-right gets 4.
    cb_sorted = sorted(cb_idx, key=lambda i: df.loc[i, "position_y"])
    for rank, i in enumerate(cb_sorted):
        # alternate 5, 4, then fall through (3 CBs: 5, 4, then leftover handles 3rd)
        prefs[i] = ([5, 4] if rank == 0 else [4, 5])[: len([5, 4])]

    cm_idx = df.index[df["position"] == "CM"].tolist()
    # Sort CMs by depth (position_x ascending). Deeper -> 6, middle -> 8, advanced -> 10.
    cm_sorted = sorted(cm_idx, key=lambda i: df.loc[i, "position_x"])
    cm_pref_by_rank = {0: [6, 8, 10], 1: [8, 6, 10], 2: [10, 8, 6]}
    for rank, i in enumerate(cm_sorted):
        prefs[i] = cm_pref_by_rank.get(rank, [8, 6, 10])

    for i in df.index:
        if i in prefs:
            continue
        label = df.loc[i, "position"]
        n = _LABEL_TO_NUMBER.get(label)
        prefs[i] = [n] if n is not None else []

    # Most-constrained-first ordering: fewer preferences first; within ties,
    # deeper player (lower x) first so they grab the "back" number in their
    # candidate list before the advanced player consumes it.
    order = sorted(df.index, key=lambda i: (len(prefs[i]) or 99, df.loc[i, "position_x"]))

    slot: dict = {}
    used: set = set()
    for i in order:
        for n in prefs[i]:
            if n not in used:
                slot[i] = n
                used.add(n)
                break

    # Fill in anyone left over with the lowest free number, ordered by depth.
    leftover = [i for i in df.index if i not in slot]
    leftover.sort(key=lambda i: df.loc[i, "position_x"])
    free = [n for n in range(1, 12) if n not in used]
    for i, n in zip(leftover, free):
        slot[i] = n
        used.add(n)

    return slot


def _team_period_active_at(team_periods: pd.DataFrame, t: float) -> int | None:
    """Return the ``period_index`` of the team whose interval covers ``t``,
    or ``None`` if no period covers it. Intervals are treated as
    ``[start, end)``; the very last period is treated as closed on the right
    too, so a snapshot ending at full-time still resolves.
    """
    starts = team_periods["period_start_min"].values
    ends = team_periods["period_end_min"].values
    mask = (starts <= t) & (t < ends)
    if not mask.any():
        # tolerate the closed right edge of the last period
        mask = (starts <= t) & (t <= ends)
        if not mask.any():
            return None
    return int(team_periods.loc[mask, "period_index"].iloc[0])


def _joint_intervals(team_periods: dict[int, pd.DataFrame]) -> list[tuple[float, float, dict[int, int]]]:
    """Compute the intersection of two teams' period timelines.

    ``team_periods`` is ``{team_id: dataframe with period_index/start/end}``.
    Returns a list of ``(start, end, {team_id: period_index})`` covering every
    sub-interval over which *both* teams' formations are stable.
    """
    breakpoints: set[float] = set()
    for tdf in team_periods.values():
        breakpoints.update(tdf["period_start_min"].tolist())
        breakpoints.update(tdf["period_end_min"].tolist())
    bps = sorted(breakpoints)

    out: list[tuple[float, float, dict[int, int]]] = []
    for lo, hi in zip(bps[:-1], bps[1:]):
        if hi <= lo:
            continue
        # Evaluate "active period" at a point strictly inside the interval.
        mid = (lo + hi) / 2
        active = {tid: _team_period_active_at(tdf, mid) for tid, tdf in team_periods.items()}
        if any(p is None for p in active.values()):
            continue
        out.append((lo, hi, active))
    return out


def build_lineup_snapshots(
    formations: pd.DataFrame,
    matches: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the wide one-row-per-(match, joint-stable-window) snapshot table.

    ``formations`` must have the schema of ``data/raw/formations.parquet``
    (matchId, teamId, period_index, formation, period_start_min,
    period_end_min, playerId, position, position_x, position_y, ...).

    ``matches`` must have ``wyId``, ``home_team``, ``away_team``, and
    ``duration`` -- the first two label the two teams in each match, and
    ``duration`` (``Regular`` / ``ExtraTime`` / ``Penalties``) is used to
    cap the open-ended last period of each team at a realistic end-of-clock
    minute. Matches with only one team in ``formations`` are skipped.
    """
    home_by_match = matches.set_index("wyId")[["home_team", "away_team"]]
    end_min_by_match = (
        matches.set_index("wyId")["duration"].map(_DURATION_TO_END_MIN).fillna(95.0)
    )

    # Per-(match, team, period) we'll need quick access to: the formation
    # string + the 11 player rows.
    team_period_idx = formations.set_index(["matchId", "teamId", "period_index"])

    snapshot_rows: list[dict] = []

    for match_id, mdf in formations.groupby("matchId"):
        if match_id not in home_by_match.index:
            continue
        home_id = int(home_by_match.loc[match_id, "home_team"])
        away_id = int(home_by_match.loc[match_id, "away_team"])

        team_ids_in_data = set(mdf["teamId"].unique())
        if not {home_id, away_id}.issubset(team_ids_in_data):
            continue

        # Per-team period table (collapsed to one row per period). The very
        # last period of a team usually has ``period_end_min == NaN`` (open-
        # ended through the end of the match); replace it with the match's
        # actual end-of-clock minute so per-30 rates aren't deflated by an
        # inflated denominator, and NaN never leaks into the breakpoint sort
        # below (sorted() with NaN is non-deterministic).
        match_end = float(end_min_by_match.loc[match_id])
        team_periods: dict[int, pd.DataFrame] = {}
        for tid in (home_id, away_id):
            tp = (
                mdf[mdf["teamId"] == tid]
                .drop_duplicates(subset=["period_index"])
                [["period_index", "period_start_min", "period_end_min", "formation"]]
                .sort_values("period_start_min")
                .reset_index(drop=True)
            )
            tp["period_end_min"] = tp["period_end_min"].fillna(match_end)
            team_periods[tid] = tp

        for lo, hi, active in _joint_intervals(team_periods):
            row: dict = {
                "match_id": int(match_id),
                "period_start_min": float(lo),
                "period_end_min": float(hi),
                "home_team_id": home_id,
                "away_team_id": away_id,
            }
            for side, tid in (("home", home_id), ("away", away_id)):
                period_idx = active[tid]
                players = team_period_idx.loc[(match_id, tid, period_idx)]
                if isinstance(players, pd.Series):
                    # only 1 player row, shouldn't really happen, but be defensive
                    players = players.to_frame().T
                players = players.reset_index(drop=True)
                row[f"{side}_formation"] = players["formation"].iloc[0]

                slots = assign_slot_numbers(players[["position", "position_x", "position_y"]])
                for ridx, n in slots.items():
                    row[f"{side}_player_{n}"]   = int(players.loc[ridx, "playerId"])
                    row[f"{side}_position_{n}"] = players.loc[ridx, "position"]

            snapshot_rows.append(row)

    # Ensure all 22 player/position columns exist even if some matches are short-handed.
    base_cols = [
        "match_id", "period_start_min", "period_end_min",
        "home_team_id", "away_team_id",
        "home_formation", "away_formation",
    ]
    slot_cols: list[str] = []
    for side in ("home", "away"):
        for n in range(1, 12):
            slot_cols.append(f"{side}_player_{n}")
            slot_cols.append(f"{side}_position_{n}")

    snapshots = pd.DataFrame(snapshot_rows, columns=base_cols + slot_cols)
    # Player slots are nullable ints (some team-periods have <11 players).
    for c in (c for c in slot_cols if c.endswith(tuple(f"_player_{n}" for n in range(1, 12)))):
        snapshots[c] = snapshots[c].astype("Int64")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshots.to_parquet(output_path, index=False)

    return snapshots
