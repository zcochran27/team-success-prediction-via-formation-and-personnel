"""Tests for :mod:`archetypes.season_stats`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from archetypes.season_stats import (
    STAT_COLS,
    _safe_rate,
    build_season_stats,
    tag_events,
)


SNAPSHOTS_DIR = Path("data/processed")
STATS_PATH = SNAPSHOTS_DIR / "player_season_stats.parquet"


@pytest.fixture()
def fixture_events() -> pd.DataFrame:
    """A tiny synthetic event log covering all five categories + their successes."""
    return pd.DataFrame({
        "matchId":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "player_id":  [10, 10, 10, 10, 10, 11, 11, 11, 11, 11],
        "type_primary":   ["shot", "shot", "pass", "pass", "duel",
                           "duel", "duel", "duel", "duel", "pass"],
        "type_secondary": [[], [], [], [], ["defensive_duel"],
                           ["aerial_duel"], ["defensive_duel"], [], ["aerial_duel"], []],
        "shot_isGoal":    [True, False, None, None, None, None, None, None, None, None],
        "pass_accurate":  [None, None, True, False, None, None, None, None, None, True],
        "groundDuel_takeOn":          [None, None, None, None, False, False, False, True, False, None],
        "groundDuel_progressedWithBall": [None, None, None, None, None, None, None, True, None, None],
        "groundDuel_stoppedProgress": [None, None, None, None, True, None, False, None, None, None],
        "aerialDuel_firstTouch":      [None, None, None, None, None, True, None, None, False, None],
    })


@pytest.fixture()
def fixture_matches() -> pd.DataFrame:
    return pd.DataFrame({"wyId": [1], "seasonId": [200]})


@pytest.fixture()
def fixture_seasons() -> pd.DataFrame:
    return pd.DataFrame({"seasonId": [200], "name": ["2024 Fall"]})


@pytest.fixture()
def fixture_formations() -> pd.DataFrame:
    """Two players: player 10 logs 90 min, player 11 logs 45 min."""
    return pd.DataFrame({
        "matchId":          [1, 1],
        "playerId":         [10, 11],
        "period_start_min": [0.0, 0.0],
        "period_end_min":   [90.0, 45.0],
    })


def test_tag_events_marks_each_category(fixture_events: pd.DataFrame) -> None:
    """Five category counters + their success counters should fire as expected."""
    tagged = tag_events(fixture_events)
    # Two shots (1 goal), three passes (2 complete -- rows 2, 9), one dribble (1 success).
    assert tagged["_shots"].sum() == 2
    assert tagged["_goals"].sum() == 1
    assert tagged["_passes"].sum() == 3
    assert tagged["_passes_complete"].sum() == 2
    assert tagged["_dribbles"].sum() == 1
    assert tagged["_dribbles_success"].sum() == 1
    # Two defensive duels (row 4 + row 6), of which row 4 stopped progress.
    assert tagged["_def_duels"].sum() == 2
    assert tagged["_def_duels_won"].sum() == 1
    # Two aerial duels (row 5, row 8), of which row 5 won the first touch.
    assert tagged["_aerial_duels"].sum() == 2
    assert tagged["_aerial_duels_won"].sum() == 1


def test_safe_rate_returns_zero_for_zero_denominator() -> None:
    """Per project convention, missing rates are 0.0 (not NaN, not inf)."""
    num = pd.Series([0, 5, 10], dtype=float)
    den = pd.Series([0, 10, 0],  dtype=float)
    rate = _safe_rate(num, den)
    assert rate.tolist() == [0.0, 0.5, 0.0]
    assert not np.isnan(rate).any()
    assert not np.isinf(rate).any()


def test_build_season_stats_end_to_end(
    fixture_events: pd.DataFrame,
    fixture_formations: pd.DataFrame,
    fixture_matches: pd.DataFrame,
    fixture_seasons: pd.DataFrame,
) -> None:
    """End-to-end: tiny synthetic export -> stat table with expected shape + rates."""
    stats = build_season_stats(
        events=fixture_events,
        formations=fixture_formations,
        matches=fixture_matches,
        seasons=fixture_seasons,
    )
    assert set(stats.columns) >= {"player_id", "season", "minutes", *STAT_COLS}
    # One row per (player, season). Two players, one season.
    assert len(stats) == 2

    p10 = stats[stats["player_id"] == 10].iloc[0]
    assert p10["minutes"] == pytest.approx(90.0)
    assert p10["shots_per_90"] == pytest.approx(2.0)
    # Player 10 had 1 goal in 2 shots.
    assert p10["shot_conv_rate"] == pytest.approx(0.5)
    # Player 10 had 2 passes (1 accurate, 1 not) in 90 min.
    assert p10["passes_per_90"] == pytest.approx(2.0)
    assert p10["pass_comp_rate"] == pytest.approx(0.5)

    p11 = stats[stats["player_id"] == 11].iloc[0]
    # Player 11 had 1 dribble (1 success).
    assert p11["dribble_success_rate"] == pytest.approx(1.0)
    # Player 11 played 45 min, so 1 dribble -> 2.0/90.
    assert p11["dribbles_per_90"] == pytest.approx(2.0)


def test_real_artifact_has_no_inf_or_nan() -> None:
    """The produced artifact should never expose inf / NaN to downstream callers."""
    if not STATS_PATH.exists():
        pytest.skip(f"{STATS_PATH} not present")
    stats = pd.read_parquet(STATS_PATH)
    feature_cols = list(STAT_COLS)
    assert not stats[feature_cols].isin([np.inf, -np.inf]).any().any()
    assert not stats[feature_cols].isna().any().any()
    # Rates must be in [0, 1].
    rate_cols = [c for c in STAT_COLS if c.endswith("_rate")]
    assert (stats[rate_cols] >= 0).all().all()
    assert (stats[rate_cols] <= 1).all().all()
