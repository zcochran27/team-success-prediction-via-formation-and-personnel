"""Shared snapshot filter: drop rows the graph builder cannot consume.

The graph models need every snapshot to meet three structural conditions
that the lineup parquet does not enforce:

  1. Both teams' formations are in graphs.templates.FORMATION_TEMPLATES, so
     the rule-based formation-edge builder has a topology to emit.
  2. Both templates have all 11 slots. A few registered formations (e.g.
     "4-4-1") only cover 10 slots and represent post-red-card states; the
     per-half modal collapser can emit such rows with 11 populated slots,
     which then crash graphs.alignment.align_lineup_to_template.
  3. All 22 position labels are non-null, so the embedding vocabulary can
     resolve every slot and the aligner has something to score against.

Archetype labels are intentionally not filtered. Many players have no
season-level archetype (too few sub minutes, missing season data, etc.),
and dropping those rows would shrink the dataset by about 60%. Instead, NaN
archetypes are replaced downstream with the "MISSING" sentinel so the
missingness becomes a learnable signal.

Both the GNN dataset and the tabular comparison apply this filter, so the
two pipelines train and validate on the same subset of the snapshot table.
"""

from __future__ import annotations

import pandas as pd

from graphs.templates import FORMATION_TEMPLATES, formation_template


_POSITION_COLS: tuple[str, ...] = tuple(
    f"team{t}_position_{i}" for t in (1, 2) for i in range(1, 12)
)
_FORMATION_COLS: tuple[str, str] = ("team1_formation", "team2_formation")

_FULL_FORMATIONS: frozenset[str] = frozenset(
    f for f in FORMATION_TEMPLATES if len(formation_template(f)) == 11
)


def filter_buildable_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that satisfy the graph builder's structural prerequisites.

    Keeps rows where both team1_formation and team2_formation map to an
    11-slot template in FORMATION_TEMPLATES and all 22 position columns are
    non-null. The index is reset so downstream code can use .iloc[i] safely.
    """
    keep = (
        df[_FORMATION_COLS[0]].isin(_FULL_FORMATIONS)
        & df[_FORMATION_COLS[1]].isin(_FULL_FORMATIONS)
        & df[list(_POSITION_COLS)].notna().all(axis=1)
    )
    return df.loc[keep].reset_index(drop=True)
