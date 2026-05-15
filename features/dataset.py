"""Lazy PyG datasets that build graphs on-demand from lineup snapshots.

The full snapshot table fits in memory easily (~80k rows of light columns),
but materializing one PyG ``Data`` object per row up-front would be wasteful.
:class:`LineupSnapshotDataset` keeps only the source DataFrame in memory
and constructs ``Data`` objects inside ``__getitem__`` so the trainer never
holds more than ``batch_size`` graphs at once.

The dataset supports the full 2 x 2 model matrix via two arguments:

- ``kind`` -- ``"position"`` (raw lineup position) or ``"archetype"``.
- ``mode`` -- ``"single"`` (one 11-node graph per team, paired forward in
  the model) or ``"paired"`` (one 22-node graph per snapshot with
  inter-team matchup edges and a same-team edge flag).

Slot alignment
--------------
Raw Wyscout slot order does NOT match
:func:`graphs.templates.formation_template` slot order. Before building
graphs, the dataset calls :func:`graphs.alignment.align_lineup_to_template`
using the position labels (the canonical role tag) and applies the
resulting permutation to whichever label set (``position`` or
``archetype``) is driving node features. This guarantees node ``k`` of
the graph corresponds to template slot ``k``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from features.build_graphs import MISSING_ARCHETYPE, PlayerFeatureKind, build_snapshot_graph
from features.snapshots import filter_buildable_snapshots
from graphs.alignment import aligned, align_lineup_to_template


Side = Literal["team1", "team2"]
GraphMode = Literal["single", "paired"]


_POS_COLS = {
    "team1": tuple(f"team1_position_{i}" for i in range(1, 12)),
    "team2": tuple(f"team2_position_{i}" for i in range(1, 12)),
}
_ARCH_COLS = {
    "team1": tuple(f"team1_archetype_{i}" for i in range(1, 12)),
    "team2": tuple(f"team2_archetype_{i}" for i in range(1, 12)),
}
_FORMATION_COLS = {"team1": "team1_formation", "team2": "team2_formation"}


class LineupSnapshotDataset(Dataset):
    """Per-snapshot dataset configurable across the 2 x 2 model matrix.

    Filters out rows where either team's formation isn't in
    :data:`graphs.templates.FORMATION_TEMPLATES`, where any of the 22 slot
    labels is null, or where the target is null. Survivors are cached as a
    compact in-memory DataFrame; graphs are built lazily in ``__getitem__``.

    Parameters
    ----------
    snapshots_path
        Path to ``lineup_snapshots.parquet``.
    kind
        ``"position"`` (raw position labels) or ``"archetype"`` (archetype
        labels) -- selects the per-slot vocabulary for node features.
    mode
        ``"single"`` to yield ``(team1_graph, team2_graph, y)`` triples
        with separate 11-node graphs per team; ``"paired"`` to yield
        ``(combined_graph, y)`` pairs with a single 22-node graph that
        includes inter-team matchup edges and a same-team edge flag.
    target_col
        Column to use as the regression target. Defaults to the per-30 xG
        differential ``team1 - team2``.
    formations
        Optional iterable restricting allowed formation strings. Defaults
        to every formation in :data:`FORMATION_TEMPLATES`.
    """

    def __init__(
        self,
        snapshots_path: Path | str,
        kind: PlayerFeatureKind,
        mode: GraphMode,
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

        label_cols = (
            list(_POS_COLS["team1"]) + list(_POS_COLS["team2"])
            + list(_ARCH_COLS["team1"]) + list(_ARCH_COLS["team2"])
        )
        cols_to_keep = (
            [_FORMATION_COLS["team1"], _FORMATION_COLS["team2"], target_col]
            + label_cols
        )
        self._df = df[cols_to_keep].reset_index(drop=True)
        self.kind: PlayerFeatureKind = kind
        self.mode: GraphMode = mode
        self.target_col = target_col

    @property
    def frame(self) -> pd.DataFrame:
        """The filtered snapshot DataFrame (read-only by convention)."""
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def _aligned_labels(self, row: pd.Series, side: Side) -> tuple[list[str], list[str]]:
        """Permute lineup slots into template-slot order for ``side``.

        Returns ``(feature_labels, position_labels)``. ``feature_labels``
        is position-or-archetype depending on ``kind``; ``position_labels``
        is always the position stream (the paired graph builder needs them
        for matchup edges regardless of ``kind``). NaN archetype labels are
        replaced with :data:`features.build_graphs.MISSING_ARCHETYPE` so
        the embedding lookup has a valid index. Both label streams are
        permuted with the same alignment so node index ``k`` agrees across
        them.
        """
        formation = row[_FORMATION_COLS[side]]
        position_labels = [row[c] for c in _POS_COLS[side]]
        perm = align_lineup_to_template(formation, position_labels)
        aligned_pos = aligned(position_labels, perm)
        if self.kind == "position":
            return aligned_pos, aligned_pos
        arch_labels = [
            MISSING_ARCHETYPE if pd.isna(row[c]) else row[c]
            for c in _ARCH_COLS[side]
        ]
        return aligned(arch_labels, perm), aligned_pos

    def __getitem__(
        self, idx: int
    ) -> tuple[Data, Data, float] | tuple[Data, float]:
        row = self._df.iloc[idx]
        y = float(row[self.target_col])
        feat_t1, pos_t1 = self._aligned_labels(row, "team1")
        if self.mode == "single":
            d1 = build_snapshot_graph(
                mode="single",
                kind=self.kind,
                formation_team1=row[_FORMATION_COLS["team1"]],
                slot_labels_team1=feat_t1,
            )
            feat_t2, _ = self._aligned_labels(row, "team2")
            d2 = build_snapshot_graph(
                mode="single",
                kind=self.kind,
                formation_team1=row[_FORMATION_COLS["team2"]],
                slot_labels_team1=feat_t2,
            )
            return d1, d2, y

        feat_t2, pos_t2 = self._aligned_labels(row, "team2")
        data = build_snapshot_graph(
            mode="paired",
            kind=self.kind,
            formation_team1=row[_FORMATION_COLS["team1"]],
            slot_labels_team1=feat_t1,
            formation_team2=row[_FORMATION_COLS["team2"]],
            slot_labels_team2=feat_t2,
            position_labels_team1=pos_t1,
            position_labels_team2=pos_t2,
        )
        return data, y


def paired_collate(samples: list[tuple[Data, Data, float]]) -> tuple[Batch, Batch, torch.Tensor]:
    """Collate ``(team1, team2, y)`` triples (``mode="single"``)."""
    team1 = Batch.from_data_list([s[0] for s in samples])
    team2 = Batch.from_data_list([s[1] for s in samples])
    y = torch.tensor([s[2] for s in samples], dtype=torch.float)
    return team1, team2, y


def snapshot_collate(samples: list[tuple[Data, float]]) -> tuple[Batch, torch.Tensor]:
    """Collate ``(combined_graph, y)`` pairs (``mode="paired"``)."""
    batch = Batch.from_data_list([s[0] for s in samples])
    y = torch.tensor([s[1] for s in samples], dtype=torch.float)
    return batch, y


def collate_for(mode: GraphMode):
    """Return the collate function matching ``mode``."""
    if mode == "single":
        return paired_collate
    if mode == "paired":
        return snapshot_collate
    raise ValueError(f"unknown mode {mode!r}")
