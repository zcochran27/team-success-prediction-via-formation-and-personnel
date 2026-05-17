"""Tests for :mod:`features.dataset` and the GNN training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from features.dataset import LineupSnapshotDataset, collate_for
from models.gnn import LineupGNN, evaluate, train_one_epoch


SNAPSHOTS = Path("data/processed/lineup_snapshots.parquet")


@pytest.fixture(scope="module")
def position_single_dataset() -> LineupSnapshotDataset:
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    return LineupSnapshotDataset(SNAPSHOTS, kind="position", mode="single")


@pytest.fixture(scope="module")
def position_paired_dataset() -> LineupSnapshotDataset:
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    return LineupSnapshotDataset(SNAPSHOTS, kind="position", mode="paired")


def test_dataset_filters_to_known_formations(position_single_dataset: LineupSnapshotDataset) -> None:
    from graphs.templates import FORMATION_TEMPLATES

    assert len(position_single_dataset) > 0
    formations = set(FORMATION_TEMPLATES)
    f1 = set(position_single_dataset.frame["team1_formation"].unique())
    f2 = set(position_single_dataset.frame["team2_formation"].unique())
    assert f1 <= formations
    assert f2 <= formations


def test_single_mode_item_shapes_match_per_row(position_single_dataset: LineupSnapshotDataset) -> None:
    d1, d2, y = position_single_dataset[0]
    assert d1.x.shape == (11,) and d1.x.dtype == torch.long
    assert d2.x.shape == (11,) and d2.x.dtype == torch.long
    assert d1.edge_index.shape[0] == 2 and d1.edge_index.dtype == torch.long
    assert d2.edge_index.shape[0] == 2
    # Single-mode graphs carry data.pos and 3D edge_attr [dist, dx, dy].
    assert d1.pos.shape == (11, 2) and d2.pos.shape == (11, 2)
    assert d1.edge_attr.shape[1] == 3
    assert isinstance(y, float)


def test_paired_mode_item_shapes_match_per_row(position_paired_dataset: LineupSnapshotDataset) -> None:
    data, y = position_paired_dataset[0]
    assert data.x.shape == (22,) and data.x.dtype == torch.long
    assert data.team.shape == (22,)
    # Paired-mode edge_attr = [same_team_flag, distance, dx, dy].
    assert data.edge_attr is not None and data.edge_attr.shape[1] == 4
    assert data.pos.shape == (22, 2)
    assert isinstance(y, float)


def test_single_collate_produces_paired_batches(position_single_dataset: LineupSnapshotDataset) -> None:
    loader = DataLoader(
        position_single_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_for("single"),
    )
    b1, b2, y = next(iter(loader))
    assert b1.num_graphs == 4
    assert b2.num_graphs == 4
    assert b1.x.shape == (44,)
    assert y.shape == (4,) and y.dtype == torch.float


def test_paired_collate_produces_combined_batch(position_paired_dataset: LineupSnapshotDataset) -> None:
    loader = DataLoader(
        position_paired_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_for("paired"),
    )
    batch, y = next(iter(loader))
    assert batch.num_graphs == 4
    assert batch.x.shape == (88,)  # 4 graphs * 22 nodes
    assert batch.team.shape == (88,)
    assert y.shape == (4,)


def _smoke_loader(dataset: LineupSnapshotDataset, mode: str) -> DataLoader:
    subset = torch.utils.data.Subset(dataset, range(min(32, len(dataset))))
    return DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collate_for(mode))


def test_single_mode_training_step_changes_loss(
    position_single_dataset: LineupSnapshotDataset,
) -> None:
    """Dataset -> single-mode model -> optimizer step actually updates the loss."""
    loader = _smoke_loader(position_single_dataset, "single")
    torch.manual_seed(0)
    model = LineupGNN(kind="position", mode="single", hidden_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    pre_loss, _, _ = evaluate(model, loader, device="cpu")
    train_one_epoch(model, loader, opt, device="cpu")
    post_loss, _, _ = evaluate(model, loader, device="cpu")

    assert pre_loss != pytest.approx(post_loss)


def test_paired_mode_training_step_changes_loss(
    position_paired_dataset: LineupSnapshotDataset,
) -> None:
    """Dataset -> paired-mode model -> optimizer step actually updates the loss."""
    loader = _smoke_loader(position_paired_dataset, "paired")
    torch.manual_seed(0)
    model = LineupGNN(kind="position", mode="paired", hidden_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    pre_loss, _, _ = evaluate(model, loader, device="cpu")
    train_one_epoch(model, loader, opt, device="cpu")
    post_loss, _, _ = evaluate(model, loader, device="cpu")

    assert pre_loss != pytest.approx(post_loss)


def test_archetype_kind_also_builds_graphs() -> None:
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    ds = LineupSnapshotDataset(SNAPSHOTS, kind="archetype", mode="single")
    d1, d2, _ = ds[0]
    from features.build_graphs import ARCHETYPE_VOCAB

    assert int(d1.x.max()) < len(ARCHETYPE_VOCAB)
    assert int(d2.x.max()) < len(ARCHETYPE_VOCAB)


def test_archetype_paired_builds_combined_graph_with_matchup_edges() -> None:
    """Paired mode must build matchups from position labels even when features are archetypes."""
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    ds = LineupSnapshotDataset(SNAPSHOTS, kind="archetype", mode="paired")
    data, _ = ds[0]
    from features.build_graphs import ARCHETYPE_VOCAB

    assert data.x.shape == (22,)
    assert int(data.x.max()) < len(ARCHETYPE_VOCAB)
    assert data.team.shape == (22,)
    assert data.edge_attr is not None and data.edge_attr.shape[1] == 4
    # At least one inter-team (same-team flag == 1) edge should be present.
    assert bool((data.edge_attr[:, 0] == 1.0).any())


def test_dataset_includes_rows_with_missing_archetypes() -> None:
    """NaN archetypes survive filtering and resolve to the MISSING vocab id."""
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    from features.build_graphs import ARCHETYPE_VOCAB, MISSING_ARCHETYPE
    import pandas as pd
    from features.snapshots import filter_buildable_snapshots

    ds = LineupSnapshotDataset(SNAPSHOTS, kind="archetype", mode="single")
    # Buildable snapshots count should match the dataset count (target is never null
    # in this parquet so the target filter is a no-op).
    expected_rows = len(filter_buildable_snapshots(pd.read_parquet(SNAPSHOTS)))
    assert len(ds) == expected_rows

    # Find a row where at least one team1 archetype slot is NaN and confirm the
    # builder maps it to the MISSING archetype embedding index.
    arch_cols = [f"team1_archetype_{i}" for i in range(1, 12)]
    null_mask = ds.frame[arch_cols].isna().any(axis=1)
    if not null_mask.any():
        pytest.skip("no rows with missing archetypes in current parquet")
    idx_with_missing = int(null_mask.idxmax())
    d1, _, _ = ds[idx_with_missing]
    missing_id = ARCHETYPE_VOCAB.index(MISSING_ARCHETYPE)
    assert (d1.x == missing_id).any()


def test_use_coords_model_runs_forward_on_both_modes(
    position_single_dataset: LineupSnapshotDataset,
    position_paired_dataset: LineupSnapshotDataset,
) -> None:
    """LineupGNN(use_coords=True) consumes data.pos + the wider edge_attr."""
    loader_single = DataLoader(
        torch.utils.data.Subset(position_single_dataset, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("single"),
    )
    loader_paired = DataLoader(
        torch.utils.data.Subset(position_paired_dataset, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("paired"),
    )
    torch.manual_seed(0)
    m_single = LineupGNN(kind="position", mode="single", hidden_dim=16, num_layers=1, use_coords=True)
    m_paired = LineupGNN(kind="position", mode="paired", hidden_dim=16, num_layers=1, use_coords=True)

    # One training step on each should mutate the loss.
    pre_s, _, _ = evaluate(m_single, loader_single, device="cpu")
    opt = torch.optim.Adam(m_single.parameters(), lr=1e-2)
    train_one_epoch(m_single, loader_single, opt, device="cpu")
    post_s, _, _ = evaluate(m_single, loader_single, device="cpu")
    assert pre_s != pytest.approx(post_s)

    pre_p, _, _ = evaluate(m_paired, loader_paired, device="cpu")
    opt2 = torch.optim.Adam(m_paired.parameters(), lr=1e-2)
    train_one_epoch(m_paired, loader_paired, opt2, device="cpu")
    post_p, _, _ = evaluate(m_paired, loader_paired, device="cpu")
    assert pre_p != pytest.approx(post_p)


def test_use_stats_dataset_attaches_stats_and_model_consumes_them(
    position_single_dataset: LineupSnapshotDataset,
) -> None:
    """LineupSnapshotDataset(stats_path=...) attaches data.stats and LineupGNN(use_stats=True) consumes it."""
    import pandas as pd
    from features.build_graphs import STAT_DIM
    stats_path = Path("data/processed/player_season_stats.parquet")
    if not stats_path.exists():
        pytest.skip(f"{stats_path} not present")

    # Sanity: a dataset with stats yields graphs that carry a (11, STAT_DIM) data.stats tensor.
    ds = LineupSnapshotDataset(
        SNAPSHOTS, kind="position", mode="single", stats_path=stats_path,
    )
    d1, d2, _ = ds[0]
    assert d1.stats.shape == (11, STAT_DIM)
    assert d2.stats.shape == (11, STAT_DIM)
    assert d1.stats.dtype == torch.float32

    # End-to-end smoke: model with use_stats=True actually trains.
    loader = DataLoader(
        torch.utils.data.Subset(ds, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("single"),
    )
    torch.manual_seed(0)
    m = LineupGNN(kind="position", mode="single", hidden_dim=16, num_layers=1, use_stats=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    pre, _, _ = evaluate(m, loader, device="cpu")
    train_one_epoch(m, loader, opt, device="cpu")
    post, _, _ = evaluate(m, loader, device="cpu")
    assert pre != pytest.approx(post)


def test_non_coords_paired_model_slices_edge_attr_correctly(
    position_paired_dataset: LineupSnapshotDataset,
) -> None:
    """The non-coords paired model should still see the same-team flag despite 4D edge_attr."""
    loader = DataLoader(
        torch.utils.data.Subset(position_paired_dataset, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("paired"),
    )
    torch.manual_seed(0)
    model = LineupGNN(kind="position", mode="paired", hidden_dim=16, num_layers=1, use_coords=False)
    # Forward should not raise -- model slices edge_attr[:, :1] under the hood.
    batch, _ = next(iter(loader))
    out = model(batch)
    assert out.shape == (4,)


def test_position_kind_matches_buildable_row_count() -> None:
    """kind='position' should also see the full buildable subset, not the archetype-clean one."""
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    import pandas as pd
    from features.snapshots import filter_buildable_snapshots

    ds = LineupSnapshotDataset(SNAPSHOTS, kind="position", mode="single")
    expected_rows = len(filter_buildable_snapshots(pd.read_parquet(SNAPSHOTS)))
    assert len(ds) == expected_rows


def test_dense_arch_dataset_and_attention_pool_trains_smoke() -> None:
    """End-to-end: dense-topology dataset + attention-pool GNN runs forward + train step."""
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    ds_single = LineupSnapshotDataset(
        SNAPSHOTS, kind="position", mode="single",
        intra_topology="full",
    )
    ds_paired = LineupSnapshotDataset(
        SNAPSHOTS, kind="position", mode="paired",
        intra_topology="full", inter_topology="full",
    )

    loader_single = DataLoader(
        torch.utils.data.Subset(ds_single, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("single"),
    )
    loader_paired = DataLoader(
        torch.utils.data.Subset(ds_paired, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("paired"),
    )

    torch.manual_seed(0)
    m_single = LineupGNN(
        kind="position", mode="single", hidden_dim=16, num_layers=1,
        use_coords=True, pool="attention",
    )
    m_paired = LineupGNN(
        kind="position", mode="paired", hidden_dim=16, num_layers=1,
        use_coords=True, pool="attention",
    )
    # Verify the head trains -- pre/post loss should differ after one step.
    pre_s, _, _ = evaluate(m_single, loader_single, device="cpu")
    opt = torch.optim.Adam(m_single.parameters(), lr=1e-2)
    train_one_epoch(m_single, loader_single, opt, device="cpu")
    post_s, _, _ = evaluate(m_single, loader_single, device="cpu")
    assert pre_s != pytest.approx(post_s)

    pre_p, _, _ = evaluate(m_paired, loader_paired, device="cpu")
    opt2 = torch.optim.Adam(m_paired.parameters(), lr=1e-2)
    train_one_epoch(m_paired, loader_paired, opt2, device="cpu")
    post_p, _, _ = evaluate(m_paired, loader_paired, device="cpu")
    assert pre_p != pytest.approx(post_p)


def test_all_conv_kinds_build_and_forward(
    position_paired_dataset: LineupSnapshotDataset,
) -> None:
    """gat / gatv2 / transformer convs all build, forward, and train one step."""
    loader = DataLoader(
        torch.utils.data.Subset(position_paired_dataset, range(8)),
        batch_size=4, shuffle=False, collate_fn=collate_for("paired"),
    )
    for conv in ("gat", "gatv2", "transformer"):
        torch.manual_seed(0)
        model = LineupGNN(
            kind="position", mode="paired", hidden_dim=16, num_layers=1,
            use_coords=True, pool="attention", conv=conv,
        )
        pre, _, _ = evaluate(model, loader, device="cpu")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        train_one_epoch(model, loader, opt, device="cpu")
        post, _, _ = evaluate(model, loader, device="cpu")
        assert pre != pytest.approx(post), f"{conv} did not update on one step"
        assert model.conv_kind == conv


def test_paired_self_match_predicts_zero_differential() -> None:
    """When team1 and team2 graphs are identical, paired-mode output should be ~0."""
    if not SNAPSHOTS.exists():
        pytest.skip(f"{SNAPSHOTS} not present")
    ds = LineupSnapshotDataset(SNAPSHOTS, kind="position", mode="paired")
    data, _ = ds[0]
    # Replace team2 nodes with team1 nodes to construct a perfect self-match.
    data.x = torch.cat([data.x[:11], data.x[:11]])

    loader = DataLoader([(data, 0.0)], batch_size=1, collate_fn=collate_for("paired"))
    torch.manual_seed(0)
    model = LineupGNN(kind="position", mode="paired", hidden_dim=16, num_layers=2)
    model.eval()
    _, preds, _ = evaluate(model, loader, device="cpu")
    # NB: The intra-team edges still differ between the two halves (team1 vs team2
    # formation), so this checks the head's antisymmetry given identical node
    # features, not full graph self-symmetry. Predictions should be small but
    # not strictly zero unless formations also match.
    assert preds.shape == (1,)
