"""Build per-snapshot PyTorch Geometric ``Data`` graphs for the GNN models.

A single public entry point, :func:`build_snapshot_graph`, produces either
the 11-node single-team graph (Models 1 & 2) or the 22-node paired graph
with inter-team matchup edges (Models 3 & 4). The ``kind`` argument
switches the per-slot vocabulary between raw position labels and archetype
labels, giving the 2 x 2 model matrix:

==============  =========================  =========================
                ``kind="position"``        ``kind="archetype"``
==============  =========================  =========================
``mode="single"``  Model 1                    Model 2
``mode="paired"``  Model 3                    Model 4
==============  =========================  =========================

Topology
--------
Intra-team edges come from the rule-based formation templates in
:mod:`graphs.templates`. In ``"paired"`` mode the team-2 intra edges are
shifted by 11 and inter-team matchup edges are added from
:func:`graphs.matchups.build_intermatch_edges`.

Node features
-------------
Stored as integer category indices on ``Data.x`` (dtype ``torch.long``)
so the GNN model can look them up with an ``nn.Embedding``.

Slot ordering
-------------
The caller is responsible for supplying ``slot_labels`` in template-slot
order (GK first at index 0, then the 10 outfield slots in the order
returned by :func:`graphs.templates.formation_template`). The lineup
dataset uses :func:`graphs.alignment.align_lineup_to_template` to permute
raw Wyscout slots into template order before calling this builder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch_geometric.data import Data

from graphs.templates import formation_edges


PlayerFeatureKind = Literal["position", "archetype"]
GraphMode = Literal["single", "paired"]


# Union of (a) positions observed in lineup_snapshots.parquet and (b) slot
# labels used in graphs.templates.FORMATION_TEMPLATES.
POSITION_VOCAB: tuple[str, ...] = (
    "CAM", "CB", "CDM", "CM", "GK",
    "LAM", "LB", "LCB", "LM", "LW", "LWB", "LWF",
    "RAM", "RB", "RCB", "RM", "RW", "RWB", "RWF",
    "SS", "ST",
)

# Archetypes are "<position_group>-<cluster_idx>" plus a "MISSING" sentinel.
# Many players don't have a season-level archetype assignment (sub minutes
# too low, missing season data, etc.) and the dataset deliberately keeps
# those rows -- the fact-of-missingness is itself a learnable signal, so
# NaN archetype labels are mapped to MISSING_ARCHETYPE before embedding.
MISSING_ARCHETYPE: str = "MISSING"
ARCHETYPE_VOCAB: tuple[str, ...] = (
    "CD-0", "CD-1", "CD-2",
    "CF-0", "CF-1", "CF-2",
    "CM-0", "CM-1", "CM-2", "CM-3",
    "GK-0", "GK-1",
    "LWD-0", "LWD-1", "LWD-2",
    "LWP-0", "LWP-1", "LWP-2",
    "RWD-0", "RWD-1", "RWD-2",
    "RWP-0", "RWP-1", "RWP-2",
    MISSING_ARCHETYPE,
)


_POSITION_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(POSITION_VOCAB)}
_ARCHETYPE_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(ARCHETYPE_VOCAB)}


def build_edge_index(formation: str) -> torch.Tensor:
    """Return the PyG ``edge_index`` for a single-team ``formation`` graph."""
    pairs = formation_edges(formation)
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    src: list[int] = []
    dst: list[int] = []
    for i, j in pairs:
        src.append(i); dst.append(j)
        src.append(j); dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def _to_indices(labels: Sequence[str], vocab_index: dict[str, int], kind_label: str) -> torch.Tensor:
    indices = torch.empty((len(labels),), dtype=torch.long)
    for i, lab in enumerate(labels):
        idx = vocab_index.get(lab)
        if idx is None:
            raise KeyError(f"{kind_label} label {lab!r} (slot {i}) not in vocabulary")
        indices[i] = idx
    return indices


def build_node_features(slot_labels: Sequence[str], kind: PlayerFeatureKind) -> torch.Tensor:
    """Return integer category indices for ``slot_labels`` under ``kind``."""
    if kind == "position":
        return _to_indices(slot_labels, _POSITION_INDEX, "position")
    if kind == "archetype":
        return _to_indices(slot_labels, _ARCHETYPE_INDEX, "archetype")
    raise ValueError(f"unknown kind {kind!r} (expected 'position' or 'archetype')")


def vocab_size(kind: PlayerFeatureKind) -> int:
    """Return ``num_embeddings`` for the matching ``nn.Embedding``."""
    if kind == "position":
        return len(POSITION_VOCAB)
    if kind == "archetype":
        return len(ARCHETYPE_VOCAB)
    raise ValueError(f"unknown kind {kind!r}")


def _build_single(
    formation: str,
    slot_labels: Sequence[str],
    kind: PlayerFeatureKind,
    target: float | None,
) -> Data:
    if len(slot_labels) != 11:
        raise ValueError(f"expected 11 slot labels, got {len(slot_labels)}")
    data = Data(
        x=build_node_features(slot_labels, kind),
        edge_index=build_edge_index(formation),
    )
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float)
    return data


def _build_paired(
    formation_team1: str,
    slot_labels_team1: Sequence[str],
    formation_team2: str,
    slot_labels_team2: Sequence[str],
    position_labels_team1: Sequence[str],
    position_labels_team2: Sequence[str],
    kind: PlayerFeatureKind,
    target: float | None,
) -> Data:
    if len(slot_labels_team1) != 11 or len(slot_labels_team2) != 11:
        raise ValueError(
            f"expected 11 labels per team, got "
            f"{len(slot_labels_team1)} / {len(slot_labels_team2)}"
        )
    if len(position_labels_team1) != 11 or len(position_labels_team2) != 11:
        raise ValueError(
            f"expected 11 position labels per team, got "
            f"{len(position_labels_team1)} / {len(position_labels_team2)}"
        )

    from graphs.matchups import build_intermatch_edges

    n_team1 = 11
    intra1 = formation_edges(formation_team1)
    intra2 = [(i + n_team1, j + n_team1) for i, j in formation_edges(formation_team2)]
    inter = build_intermatch_edges(
        team1_labels=list(position_labels_team1),
        team2_labels=list(position_labels_team2),
        team2_offset=n_team1,
        formation_team1=formation_team1,
        formation_team2=formation_team2,
    )

    src: list[int] = []
    dst: list[int] = []
    edge_types: list[float] = []
    for i, j in intra1 + intra2:
        src.append(i); dst.append(j); edge_types.append(0.0)
        src.append(j); dst.append(i); edge_types.append(0.0)
    for i, j in inter:
        src.append(i); dst.append(j); edge_types.append(1.0)
        src.append(j); dst.append(i); edge_types.append(1.0)

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(edge_types, dtype=torch.float).unsqueeze(1)
        if edge_types
        else torch.empty((0, 1), dtype=torch.float)
    )

    all_labels = list(slot_labels_team1) + list(slot_labels_team2)
    data = Data(
        x=build_node_features(all_labels, kind),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.team = torch.cat([
        torch.zeros(n_team1, dtype=torch.long),
        torch.ones(n_team1, dtype=torch.long),
    ])
    if target is not None:
        data.y = torch.tensor([float(target)], dtype=torch.float)
    return data


def build_snapshot_graph(
    mode: GraphMode,
    kind: PlayerFeatureKind,
    formation_team1: str,
    slot_labels_team1: Sequence[str],
    formation_team2: str | None = None,
    slot_labels_team2: Sequence[str] | None = None,
    position_labels_team1: Sequence[str] | None = None,
    position_labels_team2: Sequence[str] | None = None,
    target: float | None = None,
) -> Data:
    """Build the PyG ``Data`` graph for one lineup snapshot.

    Parameters
    ----------
    mode
        ``"single"`` for the 11-node team1 graph (Models 1 & 2) or
        ``"paired"`` for the 22-node graph with inter-team matchup edges
        and a same-team edge flag (Models 3 & 4).
    kind
        ``"position"`` for raw position labels or ``"archetype"`` for
        archetype labels; selects the embedding vocabulary.
    formation_team1, slot_labels_team1
        Team-1 formation string and 11 slot labels in template order.
    formation_team2, slot_labels_team2
        Required when ``mode="paired"``; ignored in ``"single"`` mode.
    position_labels_team1, position_labels_team2
        Position labels (not archetypes) used in ``"paired"`` mode to
        compute inter-team matchup edges via
        :func:`graphs.matchups.build_intermatch_edges` -- ``zone_of`` is
        defined only over position labels. Default to ``slot_labels_*``
        when ``kind="position"`` (they're the same thing); REQUIRED
        when ``kind="archetype"`` and ``mode="paired"``.
    target
        Optional scalar stored on ``data.y`` as shape ``(1,)``.

    Returns
    -------
    torch_geometric.data.Data
        For ``"single"``: 11 nodes, no ``edge_attr``.
        For ``"paired"``: 22 nodes, ``edge_attr`` of shape ``(2*E, 1)``
        carrying the same-team flag (0 intra-team, 1 inter-team), plus
        a ``data.team`` mask (``(22,)`` long; 0 for team-1 nodes, 1 for
        team-2 nodes).
    """
    if mode == "single":
        return _build_single(formation_team1, slot_labels_team1, kind, target)
    if mode == "paired":
        if formation_team2 is None or slot_labels_team2 is None:
            raise ValueError("paired mode requires formation_team2 and slot_labels_team2")
        if position_labels_team1 is None or position_labels_team2 is None:
            if kind == "position":
                position_labels_team1 = slot_labels_team1
                position_labels_team2 = slot_labels_team2
            else:
                raise ValueError(
                    "paired mode with kind='archetype' requires "
                    "position_labels_team1 and position_labels_team2 "
                    "(matchup edges are defined only over position labels)"
                )
        return _build_paired(
            formation_team1, slot_labels_team1,
            formation_team2, slot_labels_team2,
            position_labels_team1, position_labels_team2,
            kind, target,
        )
    raise ValueError(f"unknown mode {mode!r} (expected 'single' or 'paired')")
