"""Partition players into the 5 position groups and filter the event log.

Position groups
---------------
- Center Defenders (CD)
- Wide Defenders (WD)
- Center Midfielders (CM)
- Center Forwards (CF)
- Wide Players (WP) — wide midfielders and wingers

This module exposes the canonical mapping from Wyscout position labels to the
project's 5 position groups, plus utilities for filtering the event dataframe
to a given position group and set of event types.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


POSITION_GROUPS = ("CD", "WD", "CM", "CF", "WP")


def wyscout_position_to_group(position: str) -> str:
    """Map a Wyscout position label (e.g. ``"LCB"``, ``"RW"``) to one of the 5 position groups.

    Parameters
    ----------
    position
        A Wyscout position label as it appears in the raw event/lineup data.

    Returns
    -------
    str
        One of ``POSITION_GROUPS``.

    Raises
    ------
    KeyError
        If ``position`` is not recognized.
    """
    # TODO: implement the Wyscout-label -> position-group lookup table.
    raise NotImplementedError


def assign_position_groups(players: pd.DataFrame) -> pd.DataFrame:
    """Add a ``position_group`` column to a players dataframe.

    Parameters
    ----------
    players
        DataFrame with at least a Wyscout ``position`` column (one row per player
        or per player-season).

    Returns
    -------
    pandas.DataFrame
        Same frame with an added ``position_group`` column.
    """
    # TODO: vectorized application of ``wyscout_position_to_group``.
    raise NotImplementedError


def filter_events_by_group(
    events: pd.DataFrame,
    position_group: str,
    event_types: Iterable[str],
) -> pd.DataFrame:
    """Return the subset of ``events`` produced by players in ``position_group``
    whose event type is in ``event_types``.

    Parameters
    ----------
    events
        Long-form event dataframe. Must include a player identifier that can be
        joined to a player→position-group mapping and an ``event_type`` column.
    position_group
        One of ``POSITION_GROUPS``.
    event_types
        Iterable of event-type strings to keep (e.g. ``["pass", "dribble"]``).

    Returns
    -------
    pandas.DataFrame
        Filtered events.
    """
    # TODO: join events to the player→group map and filter on both axes.
    raise NotImplementedError
