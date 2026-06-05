"""Assign players to position groups and filter the event log.

The eight position groups are center defenders (CD), left/right wide
defenders (LWD/RWD), center midfielders (CM), center forwards (CF),
left/right wide players (LWP/RWP), and goalkeepers (GK).

Most logical event types map straight from Wyscout's type_primary (pass,
shot, goalkeeper_exit). The two ground duels (offensive_duel,
defensive_duel) are type_primary == "duel" rows whose type_secondary list
carries the matching tag; aerial duels are not tagged offensive or
defensive in this export.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


POSITION_GROUPS = ("CD", "LWD", "RWD", "CM", "CF", "LWP", "RWP", "GK")

WYSCOUT_POSITION_TO_GROUP: dict[str, str] = {
    # Center Defenders
    "LCB": "CD", "RCB": "CD", "CB": "CD", "LCB3": "CD", "RCB3": "CD",
    # Wide Defenders (full-backs and wing-backs), split by side
    "LB": "LWD", "LWB": "LWD", "LB5": "LWD",
    "RB": "RWD", "RWB": "RWD", "RB5": "RWD",
    # Center Midfielders (defensive, central, attacking)
    "DMF": "CM", "LDMF": "CM", "RDMF": "CM",
    "LCMF": "CM", "RCMF": "CM", "LCMF3": "CM", "RCMF3": "CM",
    "AMF": "CM",
    # Center Forwards
    "CF": "CF", "SS": "CF",
    # Wide Players: wingers, wide attacking mids, wide forwards, split by side
    "LW": "LWP", "LAMF": "LWP", "LWF": "LWP",
    "RW": "RWP", "RAMF": "RWP", "RWF": "RWP",
    # Goalkeepers
    "GK": "GK",
}

EVENT_TYPES = (
    "pass",
    "offensive_duel",
    "defensive_duel",
    "aerial_duel",
    "shot",
    "goalkeeper_exit",
)


def wyscout_position_to_group(position: str) -> str:
    """Map a Wyscout position label (e.g. "LCB", "RW", "GK") to one of the
    eight position groups. Raises KeyError on an unknown label.
    """
    return WYSCOUT_POSITION_TO_GROUP[position]


def assign_position_groups(players: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """Add a position_group column to a players or events dataframe.

    Unknown or missing labels become NaN instead of raising, so this can be
    applied to the full event log (which has None rows for non-player events).
    """
    out = players.copy()
    out["position_group"] = out[position_col].map(WYSCOUT_POSITION_TO_GROUP)
    return out


def derive_logical_event_type(events: pd.DataFrame) -> pd.Series:
    """Compute the logical event type for each event row.

    Each row maps to one of EVENT_TYPES (or NaN). pass/shot/goalkeeper_exit
    follow type_primary; offensive_duel/defensive_duel are duels whose
    type_secondary list contains the matching tag.
    """
    tp = events["type_primary"]
    out = pd.Series(pd.NA, index=events.index, dtype="object")

    out[tp == "pass"] = "pass"
    out[tp == "shot"] = "shot"
    out[tp == "goalkeeper_exit"] = "goalkeeper_exit"

    duel_mask = tp == "duel"
    if duel_mask.any():
        sec = events.loc[duel_mask, "type_secondary"]
        # Each cell is a list[str] or None. A duel carries at most one of
        # offensive_duel, defensive_duel, aerial_duel.
        def _has(tags, want):
            return tags is not None and want in tags
        off = sec.map(lambda t: _has(t, "offensive_duel")).fillna(False).astype(bool)
        deff = sec.map(lambda t: _has(t, "defensive_duel")).fillna(False).astype(bool)
        aer = sec.map(lambda t: _has(t, "aerial_duel")).fillna(False).astype(bool)
        idx = sec.index
        out.loc[idx[off]] = "offensive_duel"
        out.loc[idx[deff]] = "defensive_duel"
        out.loc[idx[aer]] = "aerial_duel"

    return out


def filter_events_by_group(
    events: pd.DataFrame,
    position_group: str,
    event_types: Iterable[str],
) -> pd.DataFrame:
    """Return the events from players in position_group whose logical event
    type is in event_types.

    events must already have position_group and logical_event_type columns;
    these are computed once on the full event log upstream.
    """
    want = set(event_types)
    mask = (events["position_group"] == position_group) & events["logical_event_type"].isin(want)
    return events.loc[mask]
