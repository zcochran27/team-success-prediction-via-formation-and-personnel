"""Per-(player, season) stat joins for the tabular ``+stats`` model variants.

Each base tabular model encodes a slot's personnel with a single
categorical token (raw position or archetype). The ``+stats`` variants
augment that with the player's 10-D per-season behavioral vector
(see :mod:`archetypes.season_stats`), flattened to one numeric column
per ``(slot, stat)`` cell:

  * ego variants  -- adds ``team1_p{1..11}_<stat>`` (110 columns)
  * matchup variants -- also adds ``team2_p{1..11}_<stat>`` (220 columns)

The join is keyed on ``(player_id, season)``; players missing from the
stats parquet resolve to all-zero vectors (matches the project-wide
0-fill convention used for the ``MISSING`` archetype).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from archetypes.season_stats import STAT_COLS


Side = Literal["team1", "team2"]

# Anchor the default to the repo layout, not the caller's cwd -- otherwise
# running this from ``models/notebooks/`` (jupyter's default kernel cwd)
# fails to find the parquet.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS_PATH = _PROJECT_ROOT / "data" / "processed" / "player_season_stats.parquet"

# Lightweight cache so repeated build_features calls (e.g. once for train,
# once for test in comparison.py) only hit the parquet once per process.
_STATS_CACHE: dict[Path, pd.DataFrame] = {}


def _load_stats(stats_path: Path) -> pd.DataFrame:
    path = Path(stats_path)
    if path not in _STATS_CACHE:
        _STATS_CACHE[path] = pd.read_parquet(path)
    return _STATS_CACHE[path]


def stat_columns_for(sides: tuple[Side, ...]) -> list[str]:
    """Return the flattened ``{side}_p{slot}_{stat}`` column names for ``sides``.

    Order is ``side -> slot -> stat`` so that successive slots stay contiguous --
    matches the layout produced by :func:`attach_stats` and the natural way
    a tree model would scan per-slot stat blocks.
    """
    cols: list[str] = []
    for side in sides:
        for slot in range(1, 12):
            for stat in STAT_COLS:
                cols.append(f"{side}_p{slot}_{stat}")
    return cols


def attach_stats(
    snapshots: pd.DataFrame,
    sides: tuple[Side, ...] = ("team1",),
    stats_path: Path | str = DEFAULT_STATS_PATH,
) -> pd.DataFrame:
    """Return a copy of ``snapshots`` with per-slot stat columns appended.

    For each ``side`` in ``sides`` and each slot ``1..11``, looks up the
    10-D stat vector by ``(team{side}_player_{slot}, season)`` and writes
    the values into ``{side}_p{slot}_{stat}`` columns. Missing keys
    (no events that season) resolve to ``0.0``.

    The input frame is not mutated.
    """
    stats_df = _load_stats(stats_path)
    stats_indexed = stats_df.set_index(["player_id", "season"])[list(STAT_COLS)]

    aug = snapshots.copy()
    # Snapshot columns are nullable Int64; fill any NaN with -1 (a sentinel that
    # won't match a real player_id) and cast to plain int64 so the MultiIndex
    # reindex below has matching types for both keys.
    season = aug["season"].fillna(-1).astype("int64").to_numpy()

    # Collect the new columns into a dict and concat at the end -- ~hundreds of
    # successive ``aug[col] = ...`` inserts trigger pandas' fragmentation warning.
    new_blocks: dict[str, np.ndarray] = {}
    for side in sides:
        for slot in range(1, 12):
            pid_col = f"{side}_player_{slot}"
            player_ids = aug[pid_col].fillna(-1).astype("int64").to_numpy()
            idx = pd.MultiIndex.from_arrays(
                [player_ids, season], names=["player_id", "season"],
            )
            slot_stats = stats_indexed.reindex(idx)
            for stat in STAT_COLS:
                new_blocks[f"{side}_p{slot}_{stat}"] = (
                    slot_stats[stat].fillna(0.0).to_numpy()
                )
    return pd.concat([aug, pd.DataFrame(new_blocks, index=aug.index)], axis=1)
