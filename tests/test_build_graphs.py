"""Tests for :mod:`features.build_graphs`."""

from __future__ import annotations

import pytest
import torch

from features.build_graphs import (
    ARCHETYPE_VOCAB,
    MISSING_ARCHETYPE,
    POSITION_VOCAB,
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
    """Single-mode ``Data`` carries ``x``, ``edge_index``, ``y`` (when given), no ``edge_attr``."""
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
    assert getattr(data, "edge_attr", None) is None
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
    assert data.edge_attr.shape == (2 * expected_undirected, 1)

    n_intra_directed = 2 * (len(intra1) + len(intra2))
    assert torch.all(data.edge_attr[:n_intra_directed] == 0.0)
    assert torch.all(data.edge_attr[n_intra_directed:] == 1.0)


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
    assert data.edge_attr.shape[1] == 1


def test_build_snapshot_graph_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_snapshot_graph(
            mode="trio",  # type: ignore[arg-type]
            kind="position",
            formation_team1="4-3-3",
            slot_labels_team1=["GK"] * 11,
        )
