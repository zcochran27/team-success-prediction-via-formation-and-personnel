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

from graphs.templates import formation_edges, formation_template


# Pitch dimensions used to normalize the template (x, y) coords stored on
# ``data.pos``. The graph templates were laid out on a 100 x 80 pitch
# (see :mod:`graphs.templates`); dividing keeps coordinates in [0, 1].
_PITCH_X = 100.0
_PITCH_Y = 80.0


def _template_coords(formation: str) -> torch.Tensor:
    """Normalized template (x, y) coords for ``formation`` in template-slot order.

    Returns a ``(11, 2)`` float tensor with coordinates in ``[0, 1]``.
    Slot order matches :func:`graphs.templates.formation_template` (GK at
    index 0).
    """
    tmpl = formation_template(formation)
    return torch.tensor(
        [(t[1] / _PITCH_X, t[2] / _PITCH_Y) for t in tmpl],
        dtype=torch.float,
    )


def _flip_coords(coords: torch.Tensor) -> torch.Tensor:
    """Rotate 180° about the (normalized) pitch center.

    Applied to team-2 coords in paired mode so opposing players who occupy
    the same physical channel (e.g. team-1 LW and team-2 RB) end up with
    nearby coordinates -- matching the lane-mirror semantics of the
    inter-team matchup edges.
    """
    return 1.0 - coords


def _edge_features(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    same_team_flags: torch.Tensor | None,
    stats: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build coord- and (optionally) stat-derived edge features.

    Column layout (per directed edge), prepending each block in order:

    1. ``same_team`` -- only when ``same_team_flags`` is supplied (paired mode).
    2. ``distance, dx, dy`` -- Euclidean distance and signed (``dst - src``)
       deltas in normalized pitch coords. Always present.
    3. Four stat-diff features -- only when ``stats`` is supplied:

       a. ``src.dribble_success_rate  - dst.def_duel_stop_rate`` -- src attacking dst.
       b. ``dst.dribble_success_rate  - src.def_duel_stop_rate`` -- dst attacking src
          ("two ways" dribble-vs-defense comparison surfaced on every edge,
          rather than relying on the back-edge to carry the reverse view).
       c. ``src.pass_comp_rate        - dst.pass_comp_rate`` -- passing matchup.
       d. ``src.aerial_duel_win_rate  - dst.aerial_duel_win_rate`` -- aerial matchup.

    Stat indices follow :data:`archetypes.season_stats.STAT_COLS` order.
    """
    base_cols = 3 + (1 if same_team_flags is not None else 0)
    stat_cols = 4 if stats is not None else 0
    if edge_index.numel() == 0:
        return torch.empty((0, base_cols + stat_cols), dtype=torch.float)

    delta = pos[edge_index[1]] - pos[edge_index[0]]  # (E, 2)
    dist = delta.norm(dim=1, keepdim=True)            # (E, 1)
    parts: list[torch.Tensor] = []
    if same_team_flags is not None:
        parts.append(same_team_flags.unsqueeze(1))
    parts.append(dist)
    parts.append(delta)
    if stats is not None:
        # STAT_COLS order: 6=pass_comp_rate, 7=dribble_success_rate,
        # 8=def_duel_stop_rate, 9=aerial_duel_win_rate.
        src = stats[edge_index[0]]
        dst = stats[edge_index[1]]
        dribble_ab = (src[:, 7] - dst[:, 8]).unsqueeze(1)
        dribble_ba = (dst[:, 7] - src[:, 8]).unsqueeze(1)
        pass_diff = (src[:, 6] - dst[:, 6]).unsqueeze(1)
        aerial_diff = (src[:, 9] - dst[:, 9]).unsqueeze(1)
        parts.extend([dribble_ab, dribble_ba, pass_diff, aerial_diff])
    return torch.cat(parts, dim=1)


PlayerFeatureKind = Literal["position", "archetype"]
GraphMode = Literal["single", "paired"]
IntraTopology = Literal["template", "full"]
InterTopology = Literal["rule", "radial", "full"]

# Default radial threshold for the dense matchup topology, in normalized pitch
# coords. ~0.35 gives 2-4 opposing edges per node on the formations we see.
DEFAULT_MATCHUP_RADIUS: float = 0.35


def _full_intra_edges(n: int = 11) -> list[tuple[int, int]]:
    """Every undirected (i, j) pair with i < j -- 55 edges for n=11.

    Returns undirected pairs; the caller is responsible for duplicating to
    ``(src, dst)`` + ``(dst, src)`` directed edges.
    """
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _full_bipartite_inter_edges(
    n_team1: int = 11,
    n_team2: int = 11,
    team2_offset: int = 11,
) -> list[tuple[int, int]]:
    """Every (i, team2_offset + j) pair -- 121 undirected edges for n=11.

    Used by the dense architecture so every player has an inter-team edge
    to every opponent; the attention pool + GAT edge weights are expected
    to learn which matchups carry signal.
    """
    return [
        (i, team2_offset + j)
        for i in range(n_team1)
        for j in range(n_team2)
    ]


def _radial_inter_edges(
    pos_team1: torch.Tensor,
    pos_team2: torch.Tensor,
    radius: float,
    team2_offset: int = 11,
) -> list[tuple[int, int]]:
    """Undirected (i, team2_offset + j) edges when ``||pos1[i] - pos2[j]|| < radius``.

    Distances are in normalized pitch coords. ``pos_team2`` is expected to
    be the *flipped* team-2 coords (the same array stored on ``data.pos``)
    so that opponents who occupy the same physical channel sit close to
    each other and therefore get connected.
    """
    if pos_team1.numel() == 0 or pos_team2.numel() == 0:
        return []
    diff = pos_team1.unsqueeze(1) - pos_team2.unsqueeze(0)  # (11, 11, 2)
    dist = diff.norm(dim=-1)                                # (11, 11)
    pairs = (dist < radius).nonzero(as_tuple=False)         # (E, 2)
    return [(int(i), team2_offset + int(j)) for i, j in pairs.tolist()]

# Per-player season-stat vector width (see archetypes.season_stats.STAT_COLS).
# Hard-coded so build_graphs and models.gnn don't have to import the stats
# module just to know how wide the projection layer is.
STAT_DIM = 10

# Width of the stat-derived edge feature block appended by ``_edge_features``
# when stats are supplied. See its docstring for the column meanings.
STAT_EDGE_DIM = 4


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
    stats: torch.Tensor | None = None,
    intra_topology: IntraTopology = "template",
) -> Data:
    if len(slot_labels) != 11:
        raise ValueError(f"expected 11 slot labels, got {len(slot_labels)}")
    if intra_topology == "template":
        edge_index = build_edge_index(formation)
    elif intra_topology == "full":
        pairs = _full_intra_edges(11)
        if pairs:
            src = [i for i, _ in pairs] + [j for _, j in pairs]
            dst = [j for _, j in pairs] + [i for i, _ in pairs]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        raise ValueError(f"unknown intra_topology {intra_topology!r}")
    pos = _template_coords(formation)
    data = Data(
        x=build_node_features(slot_labels, kind),
        edge_index=edge_index,
        pos=pos,
    )
    stats_f = stats.to(torch.float) if stats is not None else None
    if stats_f is not None and stats_f.shape[0] != 11:
        raise ValueError(f"expected 11 stat rows, got {stats_f.shape[0]}")
    # edge_attr = [dist, dx, dy] in normalized pitch coords; when stats
    # are supplied, 4 stat-diff columns are appended -- see _edge_features.
    data.edge_attr = _edge_features(pos, edge_index, same_team_flags=None, stats=stats_f)
    if stats_f is not None:
        data.stats = stats_f
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
    stats_team1: torch.Tensor | None = None,
    stats_team2: torch.Tensor | None = None,
    intra_topology: IntraTopology = "template",
    inter_topology: InterTopology = "rule",
    matchup_radius: float = DEFAULT_MATCHUP_RADIUS,
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
    if intra_topology == "template":
        intra1 = formation_edges(formation_team1)
        intra2 = [(i + n_team1, j + n_team1) for i, j in formation_edges(formation_team2)]
    elif intra_topology == "full":
        intra1 = _full_intra_edges(n_team1)
        intra2 = [(i + n_team1, j + n_team1) for i, j in _full_intra_edges(n_team1)]
    else:
        raise ValueError(f"unknown intra_topology {intra_topology!r}")

    # Team-2 coords get rotated 180 below; the radial inter-edge logic needs
    # the post-flip positions so opponents who occupy the same physical
    # channel sit close in the shared coordinate frame.
    pos_team1_unflipped = _template_coords(formation_team1)
    pos_team2_flipped = _flip_coords(_template_coords(formation_team2))

    if inter_topology == "rule":
        inter = build_intermatch_edges(
            team1_labels=list(position_labels_team1),
            team2_labels=list(position_labels_team2),
            team2_offset=n_team1,
            formation_team1=formation_team1,
            formation_team2=formation_team2,
        )
    elif inter_topology == "radial":
        inter = _radial_inter_edges(
            pos_team1=pos_team1_unflipped,
            pos_team2=pos_team2_flipped,
            radius=matchup_radius,
            team2_offset=n_team1,
        )
    elif inter_topology == "full":
        inter = _full_bipartite_inter_edges(
            n_team1=n_team1,
            n_team2=n_team1,
            team2_offset=n_team1,
        )
    else:
        raise ValueError(f"unknown inter_topology {inter_topology!r}")

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
    same_team_flags = (
        torch.tensor(edge_types, dtype=torch.float)
        if edge_types
        else torch.empty((0,), dtype=torch.float)
    )

    # Team-2 coords are rotated 180° about the pitch center so lane-mirror
    # opponents share normalized coordinates (matches inter-team matchup edge
    # geometry from :mod:`graphs.matchups`); reusing the tensors computed
    # above for the radial-edge case.
    pos = torch.cat([pos_team1_unflipped, pos_team2_flipped])

    all_labels = list(slot_labels_team1) + list(slot_labels_team2)
    data = Data(
        x=build_node_features(all_labels, kind),
        edge_index=edge_index,
        pos=pos,
    )
    stats_combined: torch.Tensor | None = None
    if stats_team1 is not None and stats_team2 is not None:
        if stats_team1.shape[0] != 11 or stats_team2.shape[0] != 11:
            raise ValueError(
                f"expected 11 stat rows per team, got "
                f"{stats_team1.shape[0]} / {stats_team2.shape[0]}"
            )
        stats_combined = torch.cat([stats_team1, stats_team2], dim=0).to(torch.float)
        data.stats = stats_combined
    # edge_attr = [same_team, dist, dx, dy] (+ 4 stat-diff cols when stats are
    # supplied). Non-coords paired models slice to [:, :1] at forward time.
    data.edge_attr = _edge_features(
        pos, edge_index, same_team_flags=same_team_flags, stats=stats_combined,
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
    stats_team1: torch.Tensor | None = None,
    stats_team2: torch.Tensor | None = None,
    target: float | None = None,
    intra_topology: IntraTopology = "template",
    inter_topology: InterTopology = "rule",
    matchup_radius: float = DEFAULT_MATCHUP_RADIUS,
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
        return _build_single(
            formation_team1, slot_labels_team1, kind, target,
            stats=stats_team1,
            intra_topology=intra_topology,
        )
    if mode == "paired":
        if formation_team2 is None or slot_labels_team2 is None:
            raise ValueError("paired mode requires formation_team2 and slot_labels_team2")
        if position_labels_team1 is None or position_labels_team2 is None:
            if kind == "position":
                position_labels_team1 = slot_labels_team1
                position_labels_team2 = slot_labels_team2
            elif inter_topology == "rule":
                raise ValueError(
                    "paired mode with kind='archetype' and inter_topology='rule' "
                    "requires position_labels_team1 and position_labels_team2 "
                    "(rule-based matchup edges are defined only over position labels)"
                )
            else:
                # Radial matchup edges don't need position labels; pass through
                # the slot labels as a no-op placeholder so the signature holds.
                position_labels_team1 = slot_labels_team1
                position_labels_team2 = slot_labels_team2
        if (stats_team1 is None) != (stats_team2 is None):
            raise ValueError("stats_team1 and stats_team2 must be both set or both None")
        return _build_paired(
            formation_team1, slot_labels_team1,
            formation_team2, slot_labels_team2,
            position_labels_team1, position_labels_team2,
            kind, target,
            stats_team1=stats_team1,
            stats_team2=stats_team2,
            intra_topology=intra_topology,
            inter_topology=inter_topology,
            matchup_radius=matchup_radius,
        )
    raise ValueError(f"unknown mode {mode!r} (expected 'single' or 'paired')")
