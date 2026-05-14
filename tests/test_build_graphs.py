"""Tests for :mod:`features.build_graphs`."""

from __future__ import annotations


def test_register_and_build_edge_index_for_4_3_3() -> None:
    """Edge index for a registered 4-3-3 formation is symmetric and matches the rule."""
    # TODO: register a 4-3-3 rule, build edges, assert undirected pairs.
    raise NotImplementedError


def test_build_node_features_position_vs_archetype_shapes() -> None:
    """Node feature tensor has 11 rows in both ``position`` and ``archetype`` modes."""
    # TODO: build features in both modes, assert .shape[0] == 11.
    raise NotImplementedError


def test_build_team_season_graph_has_expected_attributes() -> None:
    """Returned ``Data`` has ``x``, ``edge_index``, and (if provided) ``y``."""
    # TODO: build a single graph with a target and assert attribute presence.
    raise NotImplementedError
