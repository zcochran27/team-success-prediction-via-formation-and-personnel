"""Graph builder for the half-with-subs GNN.

Each (match, half, team) becomes a variable-size graph whose nodes are
every player who appeared in the half (11 starters plus 0 to 11
substitutes). Three edge types share one edge_index:

  - Formation: the starter template's tactical adjacency, plus sub-inherited
    copies (a sub picks up the formation neighbors of whoever they replaced).
  - Sub: a bidirectional link between a substitute and the starter they
    replaced, carrying the substitution minute and a same-position flag.
  - Matchup (paired mode only): every cross-team pair whose on-pitch
    intervals overlap, weighted by overlap_fraction (shared minutes / half
    duration). This captures who was actually on the pitch together.

Edge type is exposed both implicitly (column-conditional features that are 0
outside the relevant type) and explicitly via a 3-d one-hot block. The
same_team flag and [dist, dx, dy] (normalized template coords) are always
present. The 4-d stat-diff edge features (src vs dst on dribble/pass/aerial
axes) are appended when use_stats is set.

The builder reads from a row dict with the column layout of
data/processed/lineup_snapshots_half_subs.parquet and returns a PyG Data
object.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from archetypes.season_stats import STAT_COLS
from features.vocab import (
    ARCHETYPE_VOCAB,
    GraphMode,
    MISSING_ARCHETYPE,
    POSITION_VOCAB,
    PlayerFeatureKind,
    STAT_DIM,
    STAT_EDGE_DIM,
)
from graphs.alignment import align_lineup_to_template
from graphs.templates import formation_edges, formation_template


Side = Literal["team1", "team2"]


N_STARTERS = 11
N_SUB_SLOTS = 11

# Edge type indices into the 3-d one-hot block on ``edge_attr``.
EDGE_TYPE_INTRA = 0
EDGE_TYPE_SUB = 1
EDGE_TYPE_MATCHUP = 2
N_EDGE_TYPES = 3

# Backwards-compatible alias for callers that still want the old name.
EDGE_TYPE_FORMATION = EDGE_TYPE_INTRA

# Edge-attr column layout. All edges share this schema; columns that don't
# apply to a given edge type are 0 there.
#
#   [0]   same_team             1 intra-team, 0 matchup
#   [1:4] dist, dx, dy          normalized template coords
#   [4:7] edge type one-hot     intra / sub / matchup
#   [7]   sub_minute            normalized [0, 1] within half (sub edges only)
#   [8]   same_position         1 iff sub took the same template slot (sub only)
#   [9]   overlap_fraction      shared on-pitch fraction (matchup only)
#   [10]  is_template_adjacent  1 iff the (template-slot of i, template-slot of j) pair
#                               appears in the formation's 17-rule template adjacency
#                               (subs inherit their replaced starter's template slot).
#                               0 for non-intra edges and for non-tactical intra pairs.
#   [11:15] stat-diff block     only present when use_stats=True (4 cols)
EDGE_BASE_DIM = 11
EDGE_STATS_DIM = STAT_EDGE_DIM  # 4
EDGE_TIMING_TOLERANCE_MIN = 1.0  # sub vs starter end-min matching window

# Node-feature numeric block layout.
#
#   [0] is_starter
#   [1] duration / half_duration
#   [2] start_min / half_duration
#   [3] end_min / half_duration
#   [4] was_substituted_off
NODE_NUM_DIM = 5


_POSITION_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(POSITION_VOCAB)}
_ARCHETYPE_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(ARCHETYPE_VOCAB)}

# Canonical (x, y) per position label, on the same 100 x 80 grid as
# graphs.templates. Used when a sub's recorded position isn't a slot in
# the starter formation (so we don't have a template coord for them).
# Same anchors as graphs.alignment._DEFAULT_POSITION_XY, kept independent
# so this module doesn't reach across private boundaries.
_POSITION_XY: dict[str, tuple[float, float]] = {
    "GK":  (6.0, 40.0),
    "LB":  (25.0, 68.0), "LWB": (30.0, 72.0),
    "LCB": (25.0, 56.0), "CB":  (25.0, 40.0), "RCB": (25.0, 24.0),
    "RB":  (25.0, 12.0), "RWB": (30.0, 8.0),
    "CDM": (45.0, 40.0), "CM":  (60.0, 40.0), "CAM": (75.0, 40.0),
    "LM":  (60.0, 68.0), "LAM": (75.0, 64.0),
    "RM":  (60.0, 12.0), "RAM": (75.0, 16.0),
    "LW":  (95.0, 68.0), "LWF": (95.0, 68.0),
    "RW":  (95.0, 12.0), "RWF": (95.0, 12.0),
    "ST":  (100.0, 40.0), "SS": (95.0, 40.0),
}
_PITCH_X, _PITCH_Y = 100.0, 80.0


def _category_index(label: Any, kind: PlayerFeatureKind) -> int:
    """Map a position/archetype label to its embedding index.

    NaN archetypes resolve to MISSING. Unknown positions raise (positions
    come from a fixed Wyscout vocabulary, so an unknown one indicates a
    data bug, not a missing value).
    """
    if kind == "position":
        if label is None or (isinstance(label, float) and np.isnan(label)):
            raise KeyError("position label is null; row should have been filtered")
        idx = _POSITION_INDEX.get(label)
        if idx is None:
            raise KeyError(f"unknown position label {label!r}")
        return idx
    # archetype
    if label is None or (isinstance(label, float) and np.isnan(label)) or pd.isna(label):
        label = MISSING_ARCHETYPE
    idx = _ARCHETYPE_INDEX.get(label)
    if idx is None:
        raise KeyError(f"unknown archetype label {label!r}")
    return idx


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)


def _template_coords_for_starter(formation: str, slot_label: str) -> tuple[float, float]:
    """Return the normalized template (x, y) for a starter's template slot.

    The starter has already been aligned to a template slot, so slot_label is
    the template's label there. We look up the canonical coord from the
    template entry, not the player's recorded position.
    """
    # formation_template returns 11 (label, x, y) tuples; look up by label.
    tmpl = formation_template(formation)
    for lab, x, y in tmpl:
        if lab == slot_label:
            return (x / _PITCH_X, y / _PITCH_Y)
    # Fall back to the canonical xy for the recorded position.
    px, py = _POSITION_XY.get(slot_label, (50.0, 40.0))
    return (px / _PITCH_X, py / _PITCH_Y)


def _template_coords_at_slot(formation: str, slot_idx: int) -> tuple[float, float]:
    """Return normalized (x, y) for the template's slot_idx in formation."""
    _, x, y = formation_template(formation)[slot_idx]
    return (x / _PITCH_X, y / _PITCH_Y)


def _position_coords(position_label: str) -> tuple[float, float]:
    """Normalized (x, y) for an arbitrary position label (used for sub nodes)."""
    x, y = _POSITION_XY.get(position_label, (50.0, 40.0))
    return (x / _PITCH_X, y / _PITCH_Y)


def _resolve_subs_to_starters(
    row: pd.Series, side: Side, half_duration: float, half_start_min: float,
) -> list[int]:
    """Return replaced_starter_idx per active sub slot (0..N_SUB_SLOTS-1).

    Pair sub k to the still-unmatched starter whose duration-in-half is
    closest to the sub's time-since-half-start (within
    EDGE_TIMING_TOLERANCE_MIN), preferring same-position matches on close
    ties. Subs are processed in chronological order so earlier subs reserve
    their starter first.

    sub_start_min is stored in match-clock minutes (e.g. 72.1 for an H2 sub
    in the 72nd minute), so we subtract half_start_min (0 for H1, 45 for H2)
    to bring it into the same "minutes since half kickoff" frame as
    starter_duration. The caller passes float(row["period_start_min"]).

    Returns -1 for sub slots with no recorded sub player, or that can't be
    matched to a starter.
    """
    matched: list[int] = [-1] * N_SUB_SLOTS

    # Pre-extract starter info we'll reuse a few times.
    starter_durations: list[float] = []
    starter_positions: list[str] = []
    for s in range(N_STARTERS):
        starter_durations.append(float(row[f"{side}_starter_duration_{s+1}"]))
        starter_positions.append(row[f"{side}_position_{s+1}"])

    # Active subs sorted by entry time (earliest first).
    sub_order = sorted(
        range(N_SUB_SLOTS),
        key=lambda k: row[f"{side}_sub_start_min_{k+1}"]
        if not _is_null(row[f"{side}_sub_start_min_{k+1}"])
        else float("inf"),
    )
    used: set[int] = set()
    for k in sub_order:
        start_min = row[f"{side}_sub_start_min_{k+1}"]
        if _is_null(start_min):
            continue
        # Express the sub's entry in "minutes since half kickoff" so it
        # lines up with starter_duration (which is half-relative).
        sub_in_half = float(start_min) - half_start_min
        sub_pos = row[f"{side}_sub_position_{k+1}"]

        # Candidate starters: not yet matched, duration < half_duration, and
        # their duration-in-half within tolerance of when the sub came on.
        best: tuple[float, int, bool] | None = None
        for s in range(N_STARTERS):
            if s in used:
                continue
            dur = starter_durations[s]
            if dur >= half_duration - 0.05:
                continue  # this starter played to the end, not pulled
            delta = abs(dur - sub_in_half)
            if delta > EDGE_TIMING_TOLERANCE_MIN:
                continue
            pos_match = (starter_positions[s] == sub_pos)
            # Sort key prefers smaller delta first, then position match.
            score = (delta, 0 if pos_match else 1)
            if best is None or score < (best[0], 0 if best[2] else 1):
                best = (delta, s, pos_match)
        if best is not None:
            matched[k] = best[1]
            used.add(best[1])
    return matched


def _build_team_subgraph(
    row: pd.Series,
    side: Side,
    kind: PlayerFeatureKind,
    use_stats: bool,
    use_coords: bool,
    half_duration: float,
    half_start_min: float,
) -> dict[str, Any]:
    """Build node-feature tensors and intra-team edges for one team-half.

    Returns a dict of CPU tensors ready to be merged into a paired graph
    or wrapped into a Data object directly.

    Node ordering (deterministic so paired-mode matchup edges below can
    reason about cross-team indices):

      * Slots 0..10 are the starters in template-slot order.
      * Slots 11..(10 + n_subs) are the substitutes in chronological order
        of entry.
    """
    formation = row[f"{side}_formation"]
    starter_positions = [row[f"{side}_position_{s+1}"] for s in range(N_STARTERS)]
    starter_perm = align_lineup_to_template(formation, starter_positions)
    template = formation_template(formation)  # list of (label, x, y) in slot order

    # Build the row mapping: template slot -> original starter index (0..10).
    # ``starter_perm[k]`` = index in starter_positions whose canonical role
    # best matches template slot ``k``. We rearrange starter rows to slot order.
    aligned_starter_idx = list(starter_perm)

    # Collect active subs (chronological, padded out).
    sub_slots: list[int] = []
    for k in range(N_SUB_SLOTS):
        if not _is_null(row[f"{side}_sub_player_{k+1}"]):
            sub_slots.append(k)
    # Sort by entry minute ascending (ties broken by original slot order).
    sub_slots.sort(key=lambda k: float(row[f"{side}_sub_start_min_{k+1}"]))
    n_subs = len(sub_slots)
    n_nodes = N_STARTERS + n_subs

    matched_starter = _resolve_subs_to_starters(row, side, half_duration, half_start_min)
    # Map: template slot index -> set of node indices whose neighbors should
    # cascade to that template's neighbors. Starts with the starter at that
    # slot; subs that replace that starter get appended.
    nodes_in_template_slot: list[list[int]] = [[k] for k in range(N_STARTERS)]
    # Track which starters were eventually subbed off (for was_substituted_off).
    starter_was_subbed: list[bool] = [False] * N_STARTERS
    # Sub node index per sub slot.
    sub_node_idx: dict[int, int] = {}
    for node_offset, sub_k in enumerate(sub_slots):
        sub_node = N_STARTERS + node_offset
        sub_node_idx[sub_k] = sub_node
        st = matched_starter[sub_k]
        if st >= 0:
            # Locate the *template slot* of that starter (its post-alignment slot).
            template_slot = aligned_starter_idx.index(st)
            nodes_in_template_slot[template_slot].append(sub_node)
            starter_was_subbed[st] = True

    # Node features
    x_cat = torch.empty(n_nodes, dtype=torch.long)
    x_num = torch.zeros((n_nodes, NODE_NUM_DIM), dtype=torch.float)
    x_pos = torch.zeros((n_nodes, 2), dtype=torch.float)
    x_stats: torch.Tensor | None = (
        torch.zeros((n_nodes, STAT_DIM), dtype=torch.float) if use_stats else None
    )

    # Starter nodes 0..10 in template-slot order.
    for k in range(N_STARTERS):
        orig = aligned_starter_idx[k]
        label = (
            row[f"{side}_archetype_{orig+1}"]
            if kind == "archetype"
            else row[f"{side}_position_{orig+1}"]
        )
        x_cat[k] = _category_index(label, kind)
        dur = float(row[f"{side}_starter_duration_{orig+1}"])
        x_num[k, 0] = 1.0
        x_num[k, 1] = dur / half_duration
        x_num[k, 2] = 0.0
        x_num[k, 3] = dur / half_duration
        x_num[k, 4] = 1.0 if starter_was_subbed[orig] else 0.0
        x_pos[k, 0], x_pos[k, 1] = _template_coords_at_slot(formation, k)
        if use_stats:
            for j, stat_name in enumerate(STAT_COLS):
                x_stats[k, j] = float(row[f"{side}_starter_p{orig+1}_{stat_name}"])

    # Sub nodes 11..(10 + n_subs).
    for node_offset, sub_k in enumerate(sub_slots):
        node = N_STARTERS + node_offset
        label = (
            row[f"{side}_sub_archetype_{sub_k+1}"]
            if kind == "archetype"
            else row[f"{side}_sub_position_{sub_k+1}"]
        )
        x_cat[node] = _category_index(label, kind)
        # ``sub_start_min`` is stored in match-clock minutes; convert to
        # "minutes since half kickoff" (which is what ``half_duration``
        # normalizes against) by subtracting ``half_start_min``.
        smin_in_half = float(row[f"{side}_sub_start_min_{sub_k+1}"]) - half_start_min
        dur = float(row[f"{side}_sub_duration_{sub_k+1}"])
        end_in_half = smin_in_half + dur
        x_num[node, 0] = 0.0
        x_num[node, 1] = dur / half_duration
        x_num[node, 2] = smin_in_half / half_duration
        x_num[node, 3] = end_in_half / half_duration
        x_num[node, 4] = 1.0 if end_in_half < half_duration - 0.05 else 0.0
        x_pos[node, 0], x_pos[node, 1] = _position_coords(
            row[f"{side}_sub_position_{sub_k+1}"]
        )
        if use_stats:
            for j, stat_name in enumerate(STAT_COLS):
                x_stats[node, j] = float(row[f"{side}_sub_p{sub_k+1}_{stat_name}"])

    # Per-player on-pitch intervals (for matchup-edge time overlap in paired mode).
    on_pitch_start = x_num[:, 2].clone() * half_duration
    on_pitch_end = x_num[:, 3].clone() * half_duration

    # Edges
    # Sub edges (the tactically-typed ones) get emitted first so we can
    # carve them out of the fully-connected intra-team pool below.
    src_list: list[int] = []
    dst_list: list[int] = []
    type_list: list[int] = []
    sub_min_list: list[float] = []
    same_pos_list: list[float] = []
    overlap_list: list[float] = []
    template_adj_list: list[float] = []

    sub_pair_set: set[tuple[int, int]] = set()

    # 1) Sub edges (bidirectional) between sub and the starter they replaced.
    for sub_k in sub_slots:
        st = matched_starter[sub_k]
        if st < 0:
            continue
        sub_node = sub_node_idx[sub_k]
        # template_slot of the replaced starter.
        template_slot = aligned_starter_idx.index(st)
        sub_pair_set.add((min(template_slot, sub_node), max(template_slot, sub_node)))
        # Sub edge's ``sub_minute`` is half-relative, normalized to [0, 1].
        smin = (
            float(row[f"{side}_sub_start_min_{sub_k+1}"]) - half_start_min
        ) / half_duration
        # Same-position iff sub's recorded position matches the template's
        # slot label at the replaced starter's template slot.
        template_label = template[template_slot][0]
        same_pos = 1.0 if row[f"{side}_sub_position_{sub_k+1}"] == template_label else 0.0
        src_list.extend([template_slot, sub_node])
        dst_list.extend([sub_node, template_slot])
        type_list.extend([EDGE_TYPE_SUB, EDGE_TYPE_SUB])
        sub_min_list.extend([smin, smin])
        same_pos_list.extend([same_pos, same_pos])
        overlap_list.extend([0.0, 0.0])
        # Sub-replacement is by definition same-template-slot, so the pair
        # is template-adjacent in a self-loop sense (but never appears in
        # the formation's pairwise template edges). We mark it 1 so the
        # model can read "this pair shares a tactical role" from a single
        # feature regardless of which edge type carries the pair.
        template_adj_list.extend([1.0, 1.0])

    # 2) Build the set of template-adjacent (node_i, node_j) pairs so the
    #    fully-connected intra-team edges below can carry a flag indicating
    #    when two players occupy roles the 17-rule template wires together
    #    (subs inherit their replaced starter's template slot for this).
    template_adjacent_set: set[tuple[int, int]] = set()
    for ti, tj in formation_edges(formation):
        for a in nodes_in_template_slot[ti]:
            for b in nodes_in_template_slot[tj]:
                template_adjacent_set.add((min(a, b), max(a, b)))

    # 3) Fully-connected intra-team edges. Every pair of nodes on the same
    #    team gets a bidirectional edge; the model uses attention (with the
    #    is_template_adjacent flag, the coord-derived dist/dx/dy, and the
    #    node-feature block) to figure out which pairings carry signal. Pairs
    #    already wired by a sub edge are skipped so the same pair isn't
    #    represented under two different edge types.
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if (i, j) in sub_pair_set:
                continue
            is_template = 1.0 if (i, j) in template_adjacent_set else 0.0
            src_list.extend([i, j])
            dst_list.extend([j, i])
            type_list.extend([EDGE_TYPE_INTRA, EDGE_TYPE_INTRA])
            sub_min_list.extend([0.0, 0.0])
            same_pos_list.extend([0.0, 0.0])
            overlap_list.extend([0.0, 0.0])
            template_adj_list.extend([is_template, is_template])

    n_edges = len(src_list)
    edge_index = (
        torch.tensor([src_list, dst_list], dtype=torch.long)
        if n_edges else torch.empty((2, 0), dtype=torch.long)
    )

    # Edge attr is filled in below by _finalize_edge_attr once we know the
    # global node-position tensor (for distance, dx, dy). The intra-team graph
    # carries same_team=1 throughout.
    return {
        "x_cat": x_cat,
        "x_num": x_num,
        "x_pos": x_pos,
        "x_stats": x_stats,
        "edge_index": edge_index,
        "edge_type": torch.tensor(type_list, dtype=torch.long) if type_list else torch.empty((0,), dtype=torch.long),
        "edge_sub_min": torch.tensor(sub_min_list, dtype=torch.float),
        "edge_same_pos": torch.tensor(same_pos_list, dtype=torch.float),
        "edge_overlap": torch.tensor(overlap_list, dtype=torch.float),
        "edge_template_adj": torch.tensor(template_adj_list, dtype=torch.float),
        "same_team": torch.ones(n_edges, dtype=torch.float),
        "on_pitch_start": on_pitch_start,
        "on_pitch_end": on_pitch_end,
        "n_nodes": n_nodes,
    }


def _build_matchup_edges(
    on_start_t1: torch.Tensor,
    on_end_t1: torch.Tensor,
    on_start_t2: torch.Tensor,
    on_end_t2: torch.Tensor,
    team2_offset: int,
    half_duration: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the matchup edge_index and per-edge feature components.

    Every (i, j) pair with on-pitch interval overlap > 0 emits a
    bidirectional edge. Cross-team edges carry same_team = 0,
    is_template_adjacent = 0 (template adjacency is intra-team), and the
    overlap fraction in the overlap slot.
    """
    n1 = on_start_t1.size(0)
    n2 = on_start_t2.size(0)
    if n1 == 0 or n2 == 0:
        empty_l = torch.empty((0,), dtype=torch.long)
        empty_f = torch.empty((0,), dtype=torch.float)
        return (
            torch.empty((2, 0), dtype=torch.long),
            empty_l, empty_f, empty_f, empty_f, empty_f, empty_f,
        )
    # All pairs (i, j), then mask by overlap > 0.
    starts1 = on_start_t1.unsqueeze(1).expand(n1, n2)
    ends1 = on_end_t1.unsqueeze(1).expand(n1, n2)
    starts2 = on_start_t2.unsqueeze(0).expand(n1, n2)
    ends2 = on_end_t2.unsqueeze(0).expand(n1, n2)
    overlap = torch.clamp(torch.minimum(ends1, ends2) - torch.maximum(starts1, starts2), min=0.0)
    mask = overlap > 0.0
    if not mask.any():
        empty_l = torch.empty((0,), dtype=torch.long)
        empty_f = torch.empty((0,), dtype=torch.float)
        return (
            torch.empty((2, 0), dtype=torch.long),
            empty_l, empty_f, empty_f, empty_f, empty_f, empty_f,
        )
    i_idx, j_idx = torch.nonzero(mask, as_tuple=True)
    overlap_vals = overlap[i_idx, j_idx]
    a = i_idx
    b = j_idx + team2_offset
    src = torch.cat([a, b])
    dst = torch.cat([b, a])
    edge_index = torch.stack([src, dst], dim=0).long()
    n_edges = edge_index.size(1)
    edge_type = torch.full((n_edges,), EDGE_TYPE_MATCHUP, dtype=torch.long)
    sub_min = torch.zeros(n_edges, dtype=torch.float)
    same_pos = torch.zeros(n_edges, dtype=torch.float)
    overlap_norm = torch.cat([overlap_vals, overlap_vals]).float() / float(half_duration)
    same_team = torch.zeros(n_edges, dtype=torch.float)
    template_adj = torch.zeros(n_edges, dtype=torch.float)
    return edge_index, edge_type, sub_min, same_pos, overlap_norm, same_team, template_adj


def _finalize_edge_attr(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    edge_type: torch.Tensor,
    sub_min: torch.Tensor,
    same_pos: torch.Tensor,
    overlap: torch.Tensor,
    same_team: torch.Tensor,
    template_adj: torch.Tensor,
    stats: torch.Tensor | None,
) -> torch.Tensor:
    """Assemble the (E, EDGE_BASE_DIM [+EDGE_STATS_DIM]) edge feature matrix."""
    if edge_index.numel() == 0:
        width = EDGE_BASE_DIM + (EDGE_STATS_DIM if stats is not None else 0)
        return torch.empty((0, width), dtype=torch.float)
    delta = pos[edge_index[1]] - pos[edge_index[0]]
    dist = delta.norm(dim=1, keepdim=True)
    type_onehot = torch.zeros((edge_index.size(1), N_EDGE_TYPES), dtype=torch.float)
    type_onehot.scatter_(1, edge_type.unsqueeze(1), 1.0)
    parts = [
        same_team.unsqueeze(1),
        dist, delta,
        type_onehot,
        sub_min.unsqueeze(1),
        same_pos.unsqueeze(1),
        overlap.unsqueeze(1),
        template_adj.unsqueeze(1),
    ]
    if stats is not None:
        # 4-axis stat-diff features (src vs dst). Cols:
        # src.dribble_success_rate - dst.def_duel_stop_rate,
        # dst.dribble_success_rate - src.def_duel_stop_rate,
        # src.pass_comp_rate - dst.pass_comp_rate,
        # src.aerial_duel_win_rate - dst.aerial_duel_win_rate.
        src_stats = stats[edge_index[0]]
        dst_stats = stats[edge_index[1]]
        dribble_ab = (src_stats[:, 7] - dst_stats[:, 8]).unsqueeze(1)
        dribble_ba = (dst_stats[:, 7] - src_stats[:, 8]).unsqueeze(1)
        pass_diff = (src_stats[:, 6] - dst_stats[:, 6]).unsqueeze(1)
        aerial_diff = (src_stats[:, 9] - dst_stats[:, 9]).unsqueeze(1)
        parts.extend([dribble_ab, dribble_ba, pass_diff, aerial_diff])
    return torch.cat(parts, dim=1)


def build_half_subs_graph(
    row: pd.Series,
    *,
    mode: GraphMode,
    kind: PlayerFeatureKind,
    use_stats: bool,
    use_coords: bool = True,
    target_col: str = "xg_team1_minus_team2_per_30",
) -> Data | tuple[Data, Data]:
    """Build the per-snapshot graph(s) for the half-with-subs model.

    Returns a single Data for paired mode, or a (team1, team2) tuple for
    single mode (each is its own Data; the trainer subtracts their
    predictions to enforce antisymmetry).
    """
    half_duration = float(row["period_duration_min"])
    # Match-clock minute that this half starts at; the snapshot builder
    # writes 0 for H1 and 45 for H2. Needed to bring ``sub_start_min``
    # (stored in match-clock minutes) into the same frame as the
    # half-relative ``starter_duration``.
    half_start_min = float(row["period_start_min"])

    team1_block = _build_team_subgraph(
        row, "team1", kind, use_stats, use_coords, half_duration, half_start_min,
    )
    if mode == "single":
        team2_block = _build_team_subgraph(
            row, "team2", kind, use_stats, use_coords, half_duration, half_start_min,
        )
        return (
            _block_to_data(team1_block, use_stats, half_duration, target=float(row[target_col])),
            _block_to_data(team2_block, use_stats, half_duration, target=None),
        )

    # paired: assemble the joint graph with matchup edges.
    team2_block = _build_team_subgraph(
        row, "team2", kind, use_stats, use_coords, half_duration, half_start_min,
    )
    n1 = team1_block["n_nodes"]
    n2 = team2_block["n_nodes"]

    x_cat = torch.cat([team1_block["x_cat"], team2_block["x_cat"]])
    x_num = torch.cat([team1_block["x_num"], team2_block["x_num"]], dim=0)
    # Rotate team-2 coords 180 degrees about the (normalized) pitch center so
    # opposing players in the same physical channel sit close together in the
    # shared coordinate frame. Without this flip, matchup edges' dist/dx/dy
    # features collapse to 0 (both teams at the same template coords) and the
    # model loses the geometric signal for opposing roles.
    pos_team2_flipped = 1.0 - team2_block["x_pos"]
    pos = torch.cat([team1_block["x_pos"], pos_team2_flipped], dim=0)
    if use_stats:
        x_stats_combined = torch.cat([team1_block["x_stats"], team2_block["x_stats"]], dim=0)
    else:
        x_stats_combined = None

    team_mask = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long),
    ])

    # Shift team-2's intra-team edges to the global index space.
    intra2_edge = team2_block["edge_index"].clone()
    if intra2_edge.numel():
        intra2_edge = intra2_edge + n1
    edge_index = torch.cat([team1_block["edge_index"], intra2_edge], dim=1)
    edge_type = torch.cat([team1_block["edge_type"], team2_block["edge_type"]])
    sub_min = torch.cat([team1_block["edge_sub_min"], team2_block["edge_sub_min"]])
    same_pos = torch.cat([team1_block["edge_same_pos"], team2_block["edge_same_pos"]])
    overlap = torch.cat([team1_block["edge_overlap"], team2_block["edge_overlap"]])
    same_team = torch.cat([team1_block["same_team"], team2_block["same_team"]])
    template_adj = torch.cat([team1_block["edge_template_adj"], team2_block["edge_template_adj"]])

    (matchup_ei, matchup_type, matchup_sub_min, matchup_same_pos,
     matchup_overlap, matchup_same_team, matchup_template_adj) = (
        _build_matchup_edges(
            team1_block["on_pitch_start"], team1_block["on_pitch_end"],
            team2_block["on_pitch_start"], team2_block["on_pitch_end"],
            team2_offset=n1,
            half_duration=half_duration,
        )
    )
    edge_index = torch.cat([edge_index, matchup_ei], dim=1)
    edge_type = torch.cat([edge_type, matchup_type])
    sub_min = torch.cat([sub_min, matchup_sub_min])
    same_pos = torch.cat([same_pos, matchup_same_pos])
    overlap = torch.cat([overlap, matchup_overlap])
    same_team = torch.cat([same_team, matchup_same_team])
    template_adj = torch.cat([template_adj, matchup_template_adj])

    edge_attr = _finalize_edge_attr(
        edge_index, pos, edge_type, sub_min, same_pos, overlap, same_team,
        template_adj, stats=x_stats_combined if use_stats else None,
    )

    data = Data(
        x=x_cat,
        x_num=x_num,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        team=team_mask,
    )
    if use_stats:
        data.stats = x_stats_combined
    data.y = torch.tensor([float(row[target_col])], dtype=torch.float)
    return data


def _block_to_data(
    block: dict[str, Any],
    use_stats: bool,
    half_duration: float,
    target: float | None,
) -> Data:
    """Wrap a single-team block into a PyG Data with finalized edge_attr."""
    edge_attr = _finalize_edge_attr(
        block["edge_index"],
        block["x_pos"],
        block["edge_type"],
        block["edge_sub_min"],
        block["edge_same_pos"],
        block["edge_overlap"],
        block["same_team"],
        block["edge_template_adj"],
        stats=block["x_stats"] if use_stats else None,
    )
    data = Data(
        x=block["x_cat"],
        x_num=block["x_num"],
        pos=block["x_pos"],
        edge_index=block["edge_index"],
        edge_attr=edge_attr,
    )
    if use_stats:
        data.stats = block["x_stats"]
    if target is not None:
        data.y = torch.tensor([target], dtype=torch.float)
    return data


def edge_attr_dim(use_stats: bool) -> int:
    """Width of edge_attr (matches what the model layers expect)."""
    return EDGE_BASE_DIM + (EDGE_STATS_DIM if use_stats else 0)
