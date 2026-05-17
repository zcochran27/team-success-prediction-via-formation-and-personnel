"""Tests for :mod:`features.build_graphs`."""

from __future__ import annotations

import pytest
import torch

from features.build_graphs import (
    ARCHETYPE_VOCAB,
    MISSING_ARCHETYPE,
    POSITION_VOCAB,
    STAT_DIM,
    STAT_EDGE_DIM,
    build_edge_index,
    build_node_features,
    build_snapshot_graph,
    vocab_size,
)
from graphs.matchups import build_intermatch_edges
from graphs.templates import formation_edges


def test_edge_index_is_undirected_long_tensor_for_4_3_3() -> None:
    """Undirected template edges are duplicated to (src, dst) and (dst, src)."""
    edge_index = build_edge_index("4-3-3")
    assert edge_index.dtype == torch.long
    assert edge_index.shape[0] == 2
    pairs = formation_edges("4-3-3")
    assert edge_index.shape[1] == 2 * len(pairs)

    cols = {(int(edge_index[0, k]), int(edge_index[1, k])) for k in range(edge_index.shape[1])}
    for i, j in pairs:
        assert (i, j) in cols and (j, i) in cols
    assert all(s != d for s, d in cols)


def test_build_node_features_position_and_archetype_shapes_and_dtype() -> None:
    """Node features are (11,) long indices into the matching vocabulary."""
    pos_labels = ["GK", "RB", "LB", "RCB", "LCB", "CM", "RW", "CM", "ST", "SS", "LW"]
    arch_labels = [
        "GK-1", "RWP-2", "LWD-1", "RWD-2", "CD-0",
        "CM-1", "RWP-0", "CM-2", "CF-1", "CF-2", "LWP-0",
    ]

    x_pos = build_node_features(pos_labels, "position")
    assert x_pos.shape == (11,)
    assert x_pos.dtype == torch.long
    assert int(x_pos.max()) < len(POSITION_VOCAB)

    x_arch = build_node_features(arch_labels, "archetype")
    assert x_arch.shape == (11,)
    assert x_arch.dtype == torch.long
    assert int(x_arch.max()) < len(ARCHETYPE_VOCAB)


def test_build_node_features_rejects_unknown_label() -> None:
    with pytest.raises(KeyError):
        build_node_features(["GK"] + ["NOT_A_POS"] * 10, "position")


def test_build_snapshot_graph_single_rejects_bad_lineup_length() -> None:
    with pytest.raises(ValueError):
        build_snapshot_graph(
            mode="single",
            kind="position",
            formation_team1="4-4-2",
            slot_labels_team1=["GK"],
        )


def test_build_snapshot_graph_single_has_expected_attributes() -> None:
    """Single-mode ``Data`` carries ``x``, ``edge_index``, ``pos``, ``edge_attr`` (3D), and ``y`` when given."""
    labels = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    data = build_snapshot_graph(
        mode="single",
        kind="position",
        formation_team1="4-4-2",
        slot_labels_team1=labels,
        target=1.7,
    )
    assert data.x.shape == (11,)
    assert data.edge_index.shape[0] == 2
    assert data.pos.shape == (11, 2)
    assert data.edge_attr.shape == (data.edge_index.shape[1], 3)
    assert hasattr(data, "y") and data.y.shape == (1,)
    assert float(data.y.item()) == pytest.approx(1.7)

    data_no_y = build_snapshot_graph(
        mode="single",
        kind="position",
        formation_team1="4-4-2",
        slot_labels_team1=labels,
    )
    assert getattr(data_no_y, "y", None) is None


def test_vocab_size_matches_vocab_constants() -> None:
    assert vocab_size("position") == len(POSITION_VOCAB)
    assert vocab_size("archetype") == len(ARCHETYPE_VOCAB)


def test_missing_archetype_is_in_vocab_and_embeddable() -> None:
    """NaN archetypes get mapped to MISSING_ARCHETYPE -- it must resolve to an embedding index."""
    assert MISSING_ARCHETYPE in ARCHETYPE_VOCAB
    labels = [MISSING_ARCHETYPE] * 11
    x = build_node_features(labels, "archetype")
    assert x.shape == (11,)
    assert int(x.max()) < len(ARCHETYPE_VOCAB)


def test_build_snapshot_graph_paired_combines_intra_and_inter_edges() -> None:
    """22-node graph: team1 intra + team2 intra (shifted) + inter-team matchups."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    data = build_snapshot_graph(
        mode="paired",
        kind="position",
        formation_team1="4-3-3",
        slot_labels_team1=t1,
        formation_team2="4-3-3",
        slot_labels_team2=t2,
        target=0.5,
    )

    assert data.x.shape == (22,) and data.x.dtype == torch.long
    assert data.team.tolist() == [0] * 11 + [1] * 11
    assert data.y.shape == (1,)

    intra1 = formation_edges("4-3-3")
    intra2 = formation_edges("4-3-3")
    inter = build_intermatch_edges(
        t1, t2, team2_offset=11,
        formation_team1="4-3-3", formation_team2="4-3-3",
    )
    expected_undirected = len(intra1) + len(intra2) + len(inter)
    assert data.edge_index.shape == (2, 2 * expected_undirected)
    # edge_attr columns = [same_team_flag, distance, dx, dy].
    assert data.edge_attr.shape == (2 * expected_undirected, 4)

    n_intra_directed = 2 * (len(intra1) + len(intra2))
    # Column 0 is the same-team flag: intra-team rows are 0.0, inter-team are 1.0.
    assert torch.all(data.edge_attr[:n_intra_directed, 0] == 0.0)
    assert torch.all(data.edge_attr[n_intra_directed:, 0] == 1.0)


def test_build_snapshot_graph_paired_team2_indices_are_shifted() -> None:
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "RM", "LW", "ST", "RW"]
    data = build_snapshot_graph(
        mode="paired",
        kind="position",
        formation_team1="4-3-3",
        slot_labels_team1=t1,
        formation_team2="4-4-2",
        slot_labels_team2=t2,
    )

    assert int(data.edge_index.max()) <= 21
    n_t1 = 2 * len(formation_edges("4-3-3"))
    n_t2 = 2 * len(formation_edges("4-4-2"))
    t2_intra_slice = data.edge_index[:, n_t1:n_t1 + n_t2]
    assert int(t2_intra_slice.min()) >= 11


def test_build_snapshot_graph_paired_requires_team2_inputs() -> None:
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    with pytest.raises(ValueError):
        build_snapshot_graph(
            mode="paired",
            kind="position",
            formation_team1="4-3-3",
            slot_labels_team1=t1,
        )


def test_build_snapshot_graph_paired_archetype_requires_position_labels() -> None:
    """Paired + archetype needs position labels (zone_of only knows position labels)."""
    arch_t1 = [
        "GK-1", "LWD-1", "CD-0", "CD-2", "RWD-2",
        "CM-2", "CM-3", "CM-1",
        "LWP-0", "CF-1", "RWP-0",
    ]
    arch_t2 = list(arch_t1)
    with pytest.raises(ValueError, match="position_labels"):
        build_snapshot_graph(
            mode="paired",
            kind="archetype",
            formation_team1="4-3-3",
            slot_labels_team1=arch_t1,
            formation_team2="4-3-3",
            slot_labels_team2=arch_t2,
        )


def test_build_snapshot_graph_paired_archetype_with_position_labels_builds() -> None:
    """With position labels supplied, paired + archetype produces the 22-node graph."""
    pos_t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    pos_t2 = list(pos_t1)
    arch_t1 = [
        "GK-1", "LWD-1", "CD-0", "CD-2", "RWD-2",
        "CM-2", "CM-3", "CM-1",
        "LWP-0", "CF-1", "RWP-0",
    ]
    arch_t2 = list(arch_t1)
    data = build_snapshot_graph(
        mode="paired",
        kind="archetype",
        formation_team1="4-3-3",
        slot_labels_team1=arch_t1,
        formation_team2="4-3-3",
        slot_labels_team2=arch_t2,
        position_labels_team1=pos_t1,
        position_labels_team2=pos_t2,
    )
    assert data.x.shape == (22,) and data.x.dtype == torch.long
    assert int(data.x.max()) < len(ARCHETYPE_VOCAB)
    assert data.team.tolist() == [0] * 11 + [1] * 11
    assert data.edge_attr.shape[1] == 4


def test_single_graph_has_normalized_coords_and_edge_features() -> None:
    """Single-mode graphs carry data.pos ([0,1]) and 3D edge_attr [dist, dx, dy]."""
    labels = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    data = build_snapshot_graph(
        mode="single",
        kind="position",
        formation_team1="4-4-2",
        slot_labels_team1=labels,
    )
    assert data.pos.shape == (11, 2)
    assert float(data.pos.min()) >= 0.0
    assert float(data.pos.max()) <= 1.0
    assert data.edge_attr is not None
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.edge_attr.shape[1] == 3
    # column 0 = distance, columns 1,2 = (dx, dy)
    delta = data.edge_attr[:, 1:3]
    expected_dist = delta.norm(dim=1)
    assert torch.allclose(data.edge_attr[:, 0], expected_dist, atol=1e-6)
    assert (data.edge_attr[:, 0] >= 0).all()


def test_paired_graph_team2_coords_are_flipped_and_edge_attr_is_4d() -> None:
    """Paired-mode graphs flip team-2 coords (1 - x, 1 - y) and carry 4D edge_attr."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = list(t1)
    data = build_snapshot_graph(
        mode="paired",
        kind="position",
        formation_team1="4-3-3",
        slot_labels_team1=t1,
        formation_team2="4-3-3",
        slot_labels_team2=t2,
    )
    assert data.pos.shape == (22, 2)
    team1_pos = data.pos[:11]
    team2_pos = data.pos[11:]
    # Each team-2 slot should sit at (1 - team1.x, 1 - team1.y) for identical formations.
    assert torch.allclose(team2_pos, 1.0 - team1_pos, atol=1e-6)
    # edge_attr columns: [same_team, dist, dx, dy].
    assert data.edge_attr.shape[1] == 4
    same_team = data.edge_attr[:, 0]
    # Some same-team (0.0) and some inter-team (1.0) edges should exist.
    assert bool((same_team == 0.0).any())
    assert bool((same_team == 1.0).any())
    # Distances must be non-negative.
    assert (data.edge_attr[:, 1] >= 0).all()


def _make_stat_grid(seed: int = 0) -> torch.Tensor:
    """Deterministic (11, STAT_DIM) tensor; each row's last 5 cols (rates) are in [0,1]."""
    g = torch.manual_seed(seed)
    rates = torch.rand(11, 5, generator=g)         # cols 5..9 are rates
    volumes = torch.rand(11, 5, generator=g) * 10  # cols 0..4 are per-90 volumes
    return torch.cat([volumes, rates], dim=1)


def test_single_graph_stat_edge_features_match_pairwise_formulas() -> None:
    """When stats are supplied, single graphs append 4 stat-diff cols built from
    STAT_COLS indices 6, 7, 8, 9 (pass_comp, dribble_success, def_duel_stop, aerial_win)."""
    labels = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    stats = _make_stat_grid(seed=42)
    data = build_snapshot_graph(
        mode="single",
        kind="position",
        formation_team1="4-4-2",
        slot_labels_team1=labels,
        stats_team1=stats,
    )
    assert data.edge_attr.shape == (data.edge_index.shape[1], 3 + STAT_EDGE_DIM)
    assert data.stats.shape == (11, STAT_DIM)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    # Stat block lives at cols [3:7]: [dribble_AB, dribble_BA, pass_diff, aerial_diff].
    expected_dribble_ab = stats[src, 7] - stats[dst, 8]
    expected_dribble_ba = stats[dst, 7] - stats[src, 8]
    expected_pass_diff = stats[src, 6] - stats[dst, 6]
    expected_aerial_diff = stats[src, 9] - stats[dst, 9]
    assert torch.allclose(data.edge_attr[:, 3], expected_dribble_ab, atol=1e-6)
    assert torch.allclose(data.edge_attr[:, 4], expected_dribble_ba, atol=1e-6)
    assert torch.allclose(data.edge_attr[:, 5], expected_pass_diff, atol=1e-6)
    assert torch.allclose(data.edge_attr[:, 6], expected_aerial_diff, atol=1e-6)


def test_paired_graph_stat_edge_features_present_on_intra_and_inter_edges() -> None:
    """Paired graphs prepend the same_team flag, so the stat block lives at cols [4:8].
    Stats must be applied uniformly across intra- AND inter-team edges."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = list(t1)
    stats1 = _make_stat_grid(seed=1)
    stats2 = _make_stat_grid(seed=2)
    data = build_snapshot_graph(
        mode="paired",
        kind="position",
        formation_team1="4-3-3",
        slot_labels_team1=t1,
        formation_team2="4-3-3",
        slot_labels_team2=t2,
        stats_team1=stats1,
        stats_team2=stats2,
    )
    assert data.edge_attr.shape[1] == 4 + STAT_EDGE_DIM
    assert data.stats.shape == (22, STAT_DIM)
    combined = torch.cat([stats1, stats2], dim=0)
    assert torch.allclose(data.stats, combined, atol=1e-6)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    expected_dribble_ab = combined[src, 7] - combined[dst, 8]
    expected_pass_diff = combined[src, 6] - combined[dst, 6]
    assert torch.allclose(data.edge_attr[:, 4], expected_dribble_ab, atol=1e-6)
    assert torch.allclose(data.edge_attr[:, 6], expected_pass_diff, atol=1e-6)
    # Inter-team edges (flag==1) must also be populated -- not just intra.
    inter_mask = data.edge_attr[:, 0] == 1.0
    assert bool(inter_mask.any())
    inter_pairs = (src[inter_mask].tolist(), dst[inter_mask].tolist())
    for s, d in zip(*inter_pairs):
        assert data.edge_attr[(src == s) & (dst == d), 4].numel() > 0


def test_no_stats_graph_keeps_legacy_edge_attr_width() -> None:
    """Without stats, single edge_attr stays 3D and paired stays 4D (regression check)."""
    labels = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    single = build_snapshot_graph(
        mode="single", kind="position",
        formation_team1="4-4-2", slot_labels_team1=labels,
    )
    assert single.edge_attr.shape[1] == 3
    assert getattr(single, "stats", None) is None

    paired = build_snapshot_graph(
        mode="paired", kind="position",
        formation_team1="4-3-3", slot_labels_team1=labels[:11],
        formation_team2="4-3-3", slot_labels_team2=labels[:11],
    )
    assert paired.edge_attr.shape[1] == 4
    assert getattr(paired, "stats", None) is None


def test_full_intra_topology_gives_55_undirected_edges() -> None:
    """Fully-connected intra graph has C(11, 2) = 55 undirected edges, doubled to 110 directed."""
    labels = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    data = build_snapshot_graph(
        mode="single",
        kind="position",
        formation_team1="4-4-2",
        slot_labels_team1=labels,
        intra_topology="full",
    )
    assert data.edge_index.shape == (2, 110)
    pairs = {(int(data.edge_index[0, k]), int(data.edge_index[1, k]))
             for k in range(data.edge_index.shape[1])}
    # Every i != j pair must appear (in both directions).
    for i in range(11):
        for j in range(11):
            if i == j:
                continue
            assert (i, j) in pairs


def test_radial_inter_topology_respects_threshold() -> None:
    """Paired + radial: every inter-team edge must satisfy ``||pos_i - pos_j|| < r``."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = list(t1)
    r = 0.35
    data = build_snapshot_graph(
        mode="paired",
        kind="position",
        formation_team1="4-3-3",
        slot_labels_team1=t1,
        formation_team2="4-3-3",
        slot_labels_team2=t2,
        intra_topology="full",
        inter_topology="radial",
        matchup_radius=r,
    )
    # Inter edges = same_team flag == 1; intra edges = 0.
    same_team = data.edge_attr[:, 0]
    inter_mask = same_team == 1.0
    assert bool(inter_mask.any()), "radial threshold produced no inter-team edges"
    inter_src = data.edge_index[0][inter_mask]
    inter_dst = data.edge_index[1][inter_mask]
    dists = (data.pos[inter_dst] - data.pos[inter_src]).norm(dim=1)
    assert float(dists.max()) < r + 1e-6
    # No matchup edges between same-team nodes.
    assert not torch.any((inter_src < 11) == (inter_dst < 11))


def test_full_bipartite_inter_edges_connect_every_matchup() -> None:
    """``inter_topology='full'`` produces every (team1_i, team2_j) pair -- 121 undirected, 242 directed."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2 = list(t1)
    data = build_snapshot_graph(
        mode="paired", kind="position",
        formation_team1="4-3-3", slot_labels_team1=t1,
        formation_team2="4-3-3", slot_labels_team2=t2,
        intra_topology="full", inter_topology="full",
    )
    same_team = data.edge_attr[:, 0]
    inter_count = int((same_team == 1.0).sum())
    assert inter_count == 11 * 11 * 2  # 11x11 undirected, doubled to directed
    # Every (i, 11 + j) and (11 + j, i) pair must appear exactly once.
    src = data.edge_index[0]
    dst = data.edge_index[1]
    inter_pairs = {
        (int(src[k]), int(dst[k]))
        for k in range(src.shape[0])
        if data.edge_attr[k, 0] == 1.0
    }
    for i in range(11):
        for j in range(11):
            assert (i, 11 + j) in inter_pairs
            assert (11 + j, i) in inter_pairs


def test_radial_widens_or_tightens_inter_edge_count_monotonically() -> None:
    """Increasing the matchup radius can only add inter-team edges, never remove."""
    t1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    kwargs = dict(
        mode="paired", kind="position",
        formation_team1="4-3-3", slot_labels_team1=t1,
        formation_team2="4-3-3", slot_labels_team2=list(t1),
        intra_topology="full", inter_topology="radial",
    )
    counts = []
    for r in (0.10, 0.25, 0.50, 1.00):
        data = build_snapshot_graph(**kwargs, matchup_radius=r)
        same_team = data.edge_attr[:, 0]
        counts.append(int((same_team == 1.0).sum()))
    assert counts == sorted(counts)
    # r=1.0 in normalized coords covers most of the pitch -> close to fully
    # bipartite (11 × 11 = 121 undirected -> 242 directed).
    assert counts[-1] >= 200
    # Tight threshold should still pick up at least the on-axis mirror pair
    # (team1 GK <-> team2 ST after flip) so the topology never collapses to 0.
    assert counts[0] >= 2


def test_build_snapshot_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_snapshot_graph(
            mode="trio",  # type: ignore[arg-type]
            kind="position",
            formation_team1="4-3-3",
            slot_labels_team1=["GK"] * 11,
        )
