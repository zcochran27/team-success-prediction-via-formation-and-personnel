"""Tests for :mod:`archetypes.position_groups`."""

from __future__ import annotations


def test_wyscout_position_to_group_covers_all_labels() -> None:
    """Every Wyscout position label encountered in the data maps to a group."""
    # TODO: build a small set of known Wyscout labels and assert mapping.
    raise NotImplementedError


def test_assign_position_groups_adds_column() -> None:
    """``assign_position_groups`` adds a ``position_group`` column with valid values."""
    # TODO: build a tiny players DataFrame and assert output schema + values.
    raise NotImplementedError


def test_filter_events_by_group_filters_both_axes() -> None:
    """``filter_events_by_group`` filters by both position group and event type."""
    # TODO: fixture event DataFrame, run filter, assert subset is correct.
    raise NotImplementedError
