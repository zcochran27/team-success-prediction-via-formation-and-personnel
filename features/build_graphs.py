"""Construct PyTorch Geometric ``Data`` objects representing team-seasons as graphs.

Graph design
------------
- Nodes: 11 players (modal starting lineup)
- Node features: raw positional encoding (Model 3) or archetype encoding (Model 4)
- Edges: formation-specific connectivity rules
    * Defenders connected to one another
    * Midfielders connected to one another
    * Attackers connected to one another
    * Cross-line edges where the formation implies direct interaction
      (e.g. CM-ST, FB-W)

The formation-to-edge-rule mapping is centralized so new formations can be
added by extending one table.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd


PlayerFeatureKind = Literal["position", "archetype"]


# Mapping from formation string to a callable / table describing edges.
# Populated by ``register_formation``.
FORMATION_EDGE_RULES: dict[str, Any] = {}


def register_formation(formation: str, edge_rule: Any) -> None:
    """Register a formation's connectivity rule in ``FORMATION_EDGE_RULES``.

    Parameters
    ----------
    formation
        Formation string, e.g. ``"4-3-3"``.
    edge_rule
        Description of which slot pairs are connected for this formation
        (concrete representation TBD: edge list, callable, etc.).
    """
    # TODO: store ``edge_rule`` under ``formation``.
    raise NotImplementedError


def build_edge_index(formation: str) -> Any:
    """Return the ``edge_index`` tensor for a formation.

    Parameters
    ----------
    formation
        Formation string registered in ``FORMATION_EDGE_RULES``.

    Returns
    -------
    Any
        A PyG-compatible ``edge_index`` (2 × num_edges LongTensor).
    """
    # TODO: look up rule, build undirected edge list, convert to tensor.
    raise NotImplementedError


def build_node_features(
    lineup: pd.DataFrame,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
) -> Any:
    """Build the (11, F) node feature tensor for a team-season.

    Parameters
    ----------
    lineup
        Modal starting lineup, ordered by formation slot.
    kind
        ``"position"`` for Model 3, ``"archetype"`` for Model 4.
    archetype_map
        Player→archetype mapping; required when ``kind == "archetype"``.
    """
    # TODO: encode each player, stack into a FloatTensor.
    raise NotImplementedError


def build_team_season_graph(
    lineup: pd.DataFrame,
    formation: str,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
    target: float | None = None,
) -> Any:
    """Build a PyG ``Data`` object for one team-season.

    Parameters
    ----------
    lineup
        Modal starting lineup.
    formation
        Formation string.
    kind
        ``"position"`` or ``"archetype"``.
    archetype_map
        Player→archetype mapping; required when ``kind == "archetype"``.
    target
        Optional regression target (win rate or points per game).

    Returns
    -------
    torch_geometric.data.Data
        Graph with ``x``, ``edge_index``, and (if provided) ``y``.
    """
    # TODO: combine ``build_node_features`` + ``build_edge_index`` into Data.
    raise NotImplementedError


def build_graph_dataset(
    events_or_lineups: pd.DataFrame,
    team_seasons: pd.DataFrame,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
) -> list[Any]:
    """Build the list of PyG ``Data`` objects, one per team-season."""
    # TODO: iterate team_seasons -> modal lineup -> build_team_season_graph.
    raise NotImplementedError
