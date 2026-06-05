"""Per-(match, half, focal-team) dataset over the half-with-subs parquet.

Each side has 11 starter slots and 11 sub slots, with position, archetype,
season stats, duration (and sub_start_min for subs). The graph builder in
build_graphs_subs.py produces variable-size Data objects (11 to 22 nodes per
team) that PyG's Batch handles natively.

Only rows that survive filter_buildable_snapshots are exposed, so this stays
on the same buildable subset the rest of the project uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from features.build_graphs_subs import (
    GraphMode,
    PlayerFeatureKind,
    build_half_subs_graph,
)
from features.snapshots import filter_buildable_snapshots


class HalfSubsDataset(Dataset):
    """Variable-size per-half graphs with explicit substitution structure.

    Args:
        snapshots_path: path to lineup_snapshots_half_subs.parquet (or its
            train/test split files).
        kind: "position" or "archetype", which label drives the node
            embedding. Both label streams are on every node; this only
            controls which index is embedded.
        mode: "single" returns (team1_graph, team2_graph, y) per row (the
            model forwards each side and subtracts to enforce antisymmetry);
            "paired" returns (joint_graph, y) per row with cross-team matchup
            edges weighted by on-pitch time overlap.
        use_stats: append the 10-D season-stat vector to each node and the
            4-D stat-diff edge features.
        use_coords: kept for API symmetry; the builder always emits the
            coord-derived [dist, dx, dy] edge block, so this only documents
            intent. The model decides whether to project pos into embeddings.
        target_col: regression target, default the per-30 xG differential.
    """

    def __init__(
        self,
        snapshots_path: Path | str,
        kind: PlayerFeatureKind,
        mode: GraphMode,
        use_stats: bool = False,
        use_coords: bool = True,
        target_col: str = "xg_team1_minus_team2_per_30",
    ) -> None:
        if kind not in ("position", "archetype"):
            raise ValueError(f"unknown kind {kind!r}")
        if mode not in ("single", "paired"):
            raise ValueError(f"unknown mode {mode!r}")

        df = pd.read_parquet(snapshots_path)
        if target_col not in df.columns:
            raise KeyError(f"target column {target_col!r} not in snapshots parquet")
        df = filter_buildable_snapshots(df)
        df = df[df[target_col].notna()].reset_index(drop=True)
        self._df = df

        self.kind: PlayerFeatureKind = kind
        self.mode: GraphMode = mode
        self.use_stats = use_stats
        self.use_coords = use_coords
        self.target_col = target_col

    @property
    def frame(self) -> pd.DataFrame:
        """The filtered snapshot DataFrame (read-only by convention)."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(
        self, idx: int,
    ) -> tuple[Data, Data, float] | tuple[Data, float]:
        row = self._df.iloc[idx]
        result = build_half_subs_graph(
            row,
            mode=self.mode,
            kind=self.kind,
            use_stats=self.use_stats,
            use_coords=self.use_coords,
            target_col=self.target_col,
        )
        if self.mode == "single":
            d1, d2 = result
            return d1, d2, float(row[self.target_col])
        return result, float(row[self.target_col])


def paired_collate(samples: list[tuple[Data, Data, float]]) -> tuple[Batch, Batch, torch.Tensor]:
    """Collate (team1, team2, y) triples (mode="single")."""
    team1 = Batch.from_data_list([s[0] for s in samples])
    team2 = Batch.from_data_list([s[1] for s in samples])
    y = torch.tensor([s[2] for s in samples], dtype=torch.float)
    return team1, team2, y


def joint_collate(samples: list[tuple[Data, float]]) -> tuple[Batch, torch.Tensor]:
    """Collate (joint_graph, y) pairs (mode="paired")."""
    batch = Batch.from_data_list([s[0] for s in samples])
    y = torch.tensor([s[1] for s in samples], dtype=torch.float)
    return batch, y


def collate_for(mode: GraphMode):
    """Return the collate function matching mode."""
    if mode == "single":
        return paired_collate
    if mode == "paired":
        return joint_collate
    raise ValueError(f"unknown mode {mode!r}")
