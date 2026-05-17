"""Tests for the tabular ``+stats`` variants and the shared join helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archetypes.season_stats import STAT_COLS
from models import (
    _tabular_stats,
    tab_arch_ego,
    tab_arch_ego_stats,
    tab_arch_matchup_stats,
    tab_pos_ego,
    tab_pos_ego_stats,
    tab_pos_matchup_stats,
)


STATS_PATH = Path("data/processed/player_season_stats.parquet")


def _fake_snapshots() -> pd.DataFrame:
    """Two rows with a controllable mix of (player_id, season) keys."""
    return pd.DataFrame({
        # categoricals
        "team1_formation": ["4-3-3", "4-4-2"],
        "team2_formation": ["4-2-3-1", "4-4-2"],
        **{f"team1_position_{i}": ["GK"] * 2 for i in range(1, 12)},
        **{f"team2_position_{i}": ["GK"] * 2 for i in range(1, 12)},
        **{f"team1_archetype_{i}": ["GK-0"] * 2 for i in range(1, 12)},
        **{f"team2_archetype_{i}": ["GK-0"] * 2 for i in range(1, 12)},
        # ids: slot 1 -> known player, others -> -1 sentinel (no stats).
        "team1_player_1": pd.array([100, 200], dtype="Int64"),
        **{f"team1_player_{i}": pd.array([-1, -1], dtype="Int64") for i in range(2, 12)},
        **{f"team2_player_{i}": pd.array([-1, -1], dtype="Int64") for i in range(1, 12)},
        "season": pd.array([2023, 2023], dtype="Int64"),
        "team1_is_home": [1, 0],
        "match_id": [1, 2],
        "xg_team1_minus_team2_per_30": [0.1, -0.2],
    })


def _fake_stats() -> pd.DataFrame:
    """Stats for one (player_id=100, season=2023) row; everything else missing."""
    row = {c: float(i + 1) for i, c in enumerate(STAT_COLS)}
    row.update({"player_id": 100, "season": 2023, "minutes": 1000.0})
    return pd.DataFrame([row])


def test_stat_columns_for_listing_order_and_count() -> None:
    cols_one = _tabular_stats.stat_columns_for(("team1",))
    cols_two = _tabular_stats.stat_columns_for(("team1", "team2"))
    assert len(cols_one) == 11 * len(STAT_COLS)
    assert len(cols_two) == 22 * len(STAT_COLS)
    # Side-major, slot-major, stat-minor ordering.
    assert cols_one[0] == f"team1_p1_{STAT_COLS[0]}"
    assert cols_one[len(STAT_COLS)] == f"team1_p2_{STAT_COLS[0]}"
    assert cols_two[11 * len(STAT_COLS)] == f"team2_p1_{STAT_COLS[0]}"


def test_attach_stats_fills_missing_with_zero(tmp_path: Path) -> None:
    """Players absent from the stats parquet get a 0-filled stat vector."""
    stats_pq = tmp_path / "stats.parquet"
    _fake_stats().to_parquet(stats_pq, index=False)
    _tabular_stats._STATS_CACHE.pop(stats_pq, None)
    aug = _tabular_stats.attach_stats(
        _fake_snapshots(), sides=("team1",), stats_path=stats_pq,
    )
    # Slot 1 row 0 -> known player 100 -> stats are 1..10.
    for i, stat in enumerate(STAT_COLS):
        assert aug.loc[0, f"team1_p1_{stat}"] == pytest.approx(i + 1)
    # Slot 1 row 1 -> unknown player 200 -> all zeros.
    for stat in STAT_COLS:
        assert aug.loc[1, f"team1_p1_{stat}"] == 0.0
    # Slots 2..11 all sentinel -> zeros across both rows.
    for slot in range(2, 12):
        for stat in STAT_COLS:
            assert (aug[f"team1_p1_{stat}"][1], aug[f"team1_p{slot}_{stat}"][0]) == (0.0, 0.0)


def test_attach_stats_does_not_mutate_input(tmp_path: Path) -> None:
    stats_pq = tmp_path / "stats.parquet"
    _fake_stats().to_parquet(stats_pq, index=False)
    _tabular_stats._STATS_CACHE.pop(stats_pq, None)
    snapshots = _fake_snapshots()
    n_cols_before = snapshots.shape[1]
    _tabular_stats.attach_stats(snapshots, sides=("team1",), stats_path=stats_pq)
    assert snapshots.shape[1] == n_cols_before


def test_tab_pos_ego_stats_build_features_shape() -> None:
    """+stats variant adds exactly 11*STAT_DIM columns over the base variant."""
    if not STATS_PATH.exists():
        pytest.skip(f"{STATS_PATH} not present")
    snapshots = pd.read_parquet("data/processed/lineup_snapshots.parquet").head(64)

    X_base, y_base, g_base = tab_pos_ego.build_features(snapshots)
    X_stats, y_stats, g_stats = tab_pos_ego_stats.build_features(snapshots)

    assert X_stats.shape[1] == X_base.shape[1] + 11 * len(STAT_COLS)
    assert X_stats.shape[0] == X_base.shape[0]
    pd.testing.assert_series_equal(y_base, y_stats, check_names=False)
    pd.testing.assert_series_equal(g_base, g_stats, check_names=False)
    # Stat columns are numeric, not categorical.
    for c in tab_pos_ego_stats.STATS_COLS[:5]:
        assert pd.api.types.is_numeric_dtype(X_stats[c])
    # Categorical block survived.
    for c in tab_pos_ego_stats.CATEGORICAL_COLS:
        assert isinstance(X_stats[c].dtype, pd.CategoricalDtype)


def test_tab_arch_ego_stats_build_features_shape() -> None:
    if not STATS_PATH.exists():
        pytest.skip(f"{STATS_PATH} not present")
    snapshots = pd.read_parquet("data/processed/lineup_snapshots.parquet").head(64)
    X_base, _, _ = tab_arch_ego.build_features(snapshots)
    X_stats, _, _ = tab_arch_ego_stats.build_features(snapshots)
    assert X_stats.shape[1] == X_base.shape[1] + 11 * len(STAT_COLS)


def test_tab_matchup_stats_have_both_team_blocks() -> None:
    """Matchup variants append 22 stat blocks (one per slot for each team)."""
    if not STATS_PATH.exists():
        pytest.skip(f"{STATS_PATH} not present")
    snapshots = pd.read_parquet("data/processed/lineup_snapshots.parquet").head(32)
    X_pos, _, _ = tab_pos_matchup_stats.build_features(snapshots)
    X_arch, _, _ = tab_arch_matchup_stats.build_features(snapshots)

    expected_stat_cols = set(_tabular_stats.stat_columns_for(("team1", "team2")))
    assert expected_stat_cols.issubset(X_pos.columns)
    assert expected_stat_cols.issubset(X_arch.columns)
    # The new columns split evenly between team1 and team2.
    team1_stat = {c for c in expected_stat_cols if c.startswith("team1_p")}
    team2_stat = {c for c in expected_stat_cols if c.startswith("team2_p")}
    assert len(team1_stat) == len(team2_stat) == 11 * len(STAT_COLS)


def test_tab_stats_fit_predict_roundtrip_runs() -> None:
    """End-to-end smoke: fit each +stats model on a small slice and predict."""
    if not STATS_PATH.exists():
        pytest.skip(f"{STATS_PATH} not present")
    snapshots = pd.read_parquet("data/processed/lineup_snapshots.parquet").head(256)
    for mod in (tab_pos_ego_stats, tab_arch_ego_stats):
        X, y, _ = mod.build_features(snapshots)
        # XGBoost needs enough rows / variance; head(256) gives both.
        model = mod.fit(snapshots)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.isfinite(preds).all()
