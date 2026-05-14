"""Partition players into the 6 position groups and filter the event log.

Position groups
---------------
- Center Defenders (CD)
- Left Wide Defenders (LWD), Right Wide Defenders (RWD)
- Center Midfielders (CM)
- Center Forwards (CF)
- Left Wide Players (LWP), Right Wide Players (RWP)
  -- wingers, wide attacking mids, wide forwards
- Goalkeepers (GK)

The "logical event types" used by the archetype pipeline are not all primary
Wyscout types. ``pass``, ``shot``, and ``goalkeeper_exit`` are direct
``type_primary`` values, but ``offensive_duel`` / ``defensive_duel`` are
``type_primary == "duel"`` rows whose ``type_secondary`` list contains the
corresponding tag (ground duels only -- aerial duels are not tagged
offensive/defensive in Wyscout).
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
    # Wide Players: wingers, wide attacking mids, wide forwards -- split by side
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
    """Map a Wyscout position label to one of the 6 position groups.

    Parameters
    ----------
    position
        A Wyscout position label as it appears in the raw event/lineup data
        (e.g. ``"LCB"``, ``"RW"``, ``"GK"``).

    Returns
    -------
    str
        One of ``POSITION_GROUPS``.

    Raises
    ------
    KeyError
        If ``position`` is not a recognized Wyscout label.
    """
    return WYSCOUT_POSITION_TO_GROUP[position]


def assign_position_groups(players: pd.DataFrame, position_col: str = "position") -> pd.DataFrame:
    """Add a ``position_group`` column to a players or events dataframe.

    Unknown / missing position labels are mapped to ``NaN`` rather than raising
    so this can be applied to the full event log (which contains ``None``
    rows for non-player events).
    """
    out = players.copy()
    out["position_group"] = out[position_col].map(WYSCOUT_POSITION_TO_GROUP)
    return out


def derive_logical_event_type(events: pd.DataFrame) -> pd.Series:
    """Compute the project's "logical event type" for each event row.

    Maps each row to one of ``EVENT_TYPES`` (or ``NaN`` if it doesn't belong
    to any of them). ``pass`` / ``shot`` / ``goalkeeper_exit`` follow
    ``type_primary``; ``offensive_duel`` / ``defensive_duel`` are duels whose
    ``type_secondary`` list contains the corresponding tag.
    """
    tp = events["type_primary"]
    out = pd.Series(pd.NA, index=events.index, dtype="object")

    out[tp == "pass"] = "pass"
    out[tp == "shot"] = "shot"
    out[tp == "goalkeeper_exit"] = "goalkeeper_exit"

    duel_mask = tp == "duel"
    if duel_mask.any():
        sec = events.loc[duel_mask, "type_secondary"]
        # Each cell is a list[str] (or None). A duel can carry at most one
        # of {offensive_duel, defensive_duel, aerial_duel} in the project's
        # taxonomy -- aerial duels are not tagged offensive/defensive in this
        # Wyscout export.
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
    """Return the subset of ``events`` produced by players in ``position_group``
    whose logical event type is in ``event_types``.

    The frame is expected to already have a ``position_group`` column (from
    :func:`assign_position_groups`) and a ``logical_event_type`` column (from
    :func:`derive_logical_event_type`). The orchestrator computes both once
    on the full event log so callers don't pay for the lookup repeatedly.
    """
    want = set(event_types)
    mask = (events["position_group"] == position_group) & events["logical_event_type"].isin(want)
    return events.loc[mask]
