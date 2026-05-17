"""Shared GNN backbone + train/predict helpers for the graph models.

The model grid is configured via three arguments on :class:`LineupGNN`:

- ``kind`` -- ``"position"`` or ``"archetype"`` (selects the embedding
  vocabulary).
- ``mode`` -- ``"single"`` (the 11-node single-team graph) or
  ``"paired"`` (the 22-node graph with inter-team matchup edges and a
  same-team edge flag).
- ``use_coords`` -- when ``True``, mix the per-node normalized template
  ``(x, y)`` coordinate into the node embedding via a learned linear
  projection, and consume coord-derived edge features (distance, ``dx``,
  ``dy``) in the message-passing layers.
- ``use_stats`` -- when ``True``, mix the per-(player, season) stat
  vector (10-D, see :mod:`archetypes.season_stats`) into the node
  embedding via a learned linear projection, and append 4 stat-diff
  edge features built by :func:`features.build_graphs._edge_features`
  (src/dst dribble-vs-defense in both directions, passing-comp diff,
  aerial-duel-win diff). Combinable with ``use_coords``; the node-side
  stat and coord projections are both added to the categorical
  embedding, and the edge-side stat features are concatenated to
  whatever coord-derived edge features the model is configured to use.

GATConv ``edge_dim`` is the sum of the active blocks:

==========  ====================  ==========
mode        edge blocks present   edge_dim
==========  ====================  ==========
single      none                  None
single      coords only           3
single      stats only            4
single      coords + stats        7
paired      same_team only        1
paired      same_team + coords    4
paired      same_team + stats     5
paired      all three             8
==========  ====================  ==========

For non-coords paired models the graph builder still produces ``edge_attr``
that starts with ``[same_team, dist, dx, dy]``; the forward keeps only the
columns this model is configured to use.

Paired-mode head
----------------
Per-snapshot node embeddings are pooled separately for team 1 and
team 2 (using ``data.team``); the prediction is
``head(team1_pool) - head(team2_pool)``. Sharing the head and subtracting
preserves antisymmetry: the same lineup facing itself predicts exactly
zero differential.

Pooling
-------
- ``pool="mean"`` (default) -- :func:`torch_geometric.nn.global_mean_pool`.
- ``pool="attention"`` -- :class:`~torch_geometric.nn.AttentionalAggregation`
  with a small gate MLP. A single pool instance is shared across the
  team-1 / team-2 pools in paired mode so the antisymmetry of the head
  still holds.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import (
    AttentionalAggregation,
    GATConv,
    GATv2Conv,
    TransformerConv,
    global_mean_pool,
)

from features.build_graphs import (
    GraphMode,
    PlayerFeatureKind,
    STAT_DIM,
    STAT_EDGE_DIM,
    vocab_size,
)


def _edge_dim_for(mode: GraphMode, use_coords: bool, use_stats: bool) -> int | None:
    """Return ``edge_dim`` for :class:`torch_geometric.nn.GATConv`.

    See the module docstring for the full table. In short:

    - single mode: sum of ``3`` (coords) and ``STAT_EDGE_DIM`` (stats), or
      ``None`` if neither block is active.
    - paired mode: always ``1`` (same-team flag) plus ``3`` (coords) and/or
      ``STAT_EDGE_DIM`` (stats).
    """
    if mode == "single":
        n = (3 if use_coords else 0) + (STAT_EDGE_DIM if use_stats else 0)
        return n if n > 0 else None
    if mode == "paired":
        return 1 + (3 if use_coords else 0) + (STAT_EDGE_DIM if use_stats else 0)
    raise ValueError(f"unknown mode {mode!r}")


PoolKind = Literal["mean", "attention"]
ConvKind = Literal["gat", "gatv2", "transformer"]


_CONV_CLASSES: dict[str, type[nn.Module]] = {
    "gat":         GATConv,
    "gatv2":       GATv2Conv,
    "transformer": TransformerConv,
}


def _make_conv(
    conv_kind: ConvKind,
    hidden_dim: int,
    heads: int,
    edge_dim: int | None,
) -> nn.Module:
    """Build one message-passing layer matching ``conv_kind``.

    All three layer types take the same ``(in, out, heads, concat, edge_dim)``
    signature and produce ``(N, hidden_dim)`` outputs when ``concat=False``,
    so they're drop-in interchangeable from the perspective of the rest of
    the model.
    """
    cls = _CONV_CLASSES.get(conv_kind)
    if cls is None:
        raise ValueError(f"unknown conv {conv_kind!r}")
    return cls(
        hidden_dim,
        hidden_dim,
        heads=heads,
        concat=False,
        edge_dim=edge_dim,
    )


class LineupGNN(nn.Module):
    """Embedding -> stacked graph conv -> pool -> shared MLP head.

    The conv family is configurable via ``conv``:

    - ``"gat"``         -- :class:`~torch_geometric.nn.GATConv` (default).
    - ``"gatv2"``       -- :class:`~torch_geometric.nn.GATv2Conv` (dynamic
      attention; fixes the static-attention pathology in vanilla GAT).
    - ``"transformer"`` -- :class:`~torch_geometric.nn.TransformerConv`
      (multi-head Q/K/V over neighbors with edge-feature conditioning).
    """

    def __init__(
        self,
        kind: PlayerFeatureKind,
        mode: GraphMode = "single",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 1,
        use_coords: bool = False,
        use_stats: bool = False,
        pool: PoolKind = "mean",
        conv: ConvKind = "gat",
    ) -> None:
        super().__init__()
        if mode not in ("single", "paired"):
            raise ValueError(f"unknown mode {mode!r}")
        if pool not in ("mean", "attention"):
            raise ValueError(f"unknown pool {pool!r}")
        if conv not in _CONV_CLASSES:
            raise ValueError(f"unknown conv {conv!r}")
        self.kind = kind
        self.mode: GraphMode = mode
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_coords = use_coords
        self.use_stats = use_stats
        self.pool_kind: PoolKind = pool
        self.conv_kind: ConvKind = conv

        self.embedding = nn.Embedding(vocab_size(kind), hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim) if use_coords else None
        self.stats_proj = nn.Linear(STAT_DIM, hidden_dim) if use_stats else None

        edge_dim = _edge_dim_for(mode, use_coords, use_stats)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(_make_conv(conv, hidden_dim, heads, edge_dim))

        if pool == "attention":
            gate_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.attn_pool = AttentionalAggregation(gate_nn=gate_nn)
        else:
            self.attn_pool = None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _pool(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        if self.pool_kind == "attention":
            return self.attn_pool(x, batch_index)
        return global_mean_pool(x, batch_index)

    def _resolve_edge_attr(self, batch: Batch) -> torch.Tensor | None:
        """Pick the slice of ``batch.edge_attr`` matching this model's config.

        The builder emits a fixed column layout per mode:

        - single: ``[dist, dx, dy]`` (+ 4 stat-diff cols when stats were
          supplied) -- coord block at ``[:3]``, stat block at ``[3:7]``.
        - paired: ``[same_team, dist, dx, dy]`` (+ 4 stat-diff cols) --
          flag at ``[:1]``, coord block at ``[1:4]``, stat block at
          ``[4:8]``.

        This helper concatenates the blocks the model is configured to
        consume; the resulting width matches :func:`_edge_dim_for`.
        """
        edge_attr = getattr(batch, "edge_attr", None)
        if edge_attr is None:
            return None
        parts: list[torch.Tensor] = []
        if self.mode == "single":
            if self.use_coords:
                parts.append(edge_attr[:, :3])
            if self.use_stats:
                parts.append(edge_attr[:, 3:3 + STAT_EDGE_DIM])
        else:  # paired
            parts.append(edge_attr[:, :1])  # same_team flag always
            if self.use_coords:
                parts.append(edge_attr[:, 1:4])
            if self.use_stats:
                parts.append(edge_attr[:, 4:4 + STAT_EDGE_DIM])
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def _encode(self, batch: Batch) -> torch.Tensor:
        x = self.embedding(batch.x)
        if self.use_coords:
            x = x + self.coord_proj(batch.pos)
        if self.use_stats:
            x = x + self.stats_proj(batch.stats)
        edge_attr = self._resolve_edge_attr(batch)
        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr=edge_attr) if edge_attr is not None \
                else conv(x, batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return one scalar per graph in ``batch``.

        - ``mode="single"``: pools all 11 nodes and applies the head; the
          training loop calls this once per team and subtracts.
        - ``mode="paired"``: pools team-1 and team-2 nodes separately
          using ``batch.team`` and returns ``head(t1) - head(t2)``.
        """
        node_emb = self._encode(batch)
        if self.mode == "single":
            graph_emb = self._pool(node_emb, batch.batch)
            return self.head(graph_emb).squeeze(-1)

        team_mask = batch.team
        t1_idx = (team_mask == 0).nonzero(as_tuple=False).squeeze(-1)
        t2_idx = (team_mask == 1).nonzero(as_tuple=False).squeeze(-1)
        pool_t1 = self._pool(node_emb[t1_idx], batch.batch[t1_idx])
        pool_t2 = self._pool(node_emb[t2_idx], batch.batch[t2_idx])
        return (self.head(pool_t1) - self.head(pool_t2)).squeeze(-1)


def _predict_diff(
    model: LineupGNN,
    loader_item,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass on a loader item; return ``(prediction, target)``."""
    if model.mode == "single":
        b1, b2, y = loader_item
        b1 = b1.to(device)
        b2 = b2.to(device)
        y = y.to(device)
        return model(b1) - model(b2), y
    batch, y = loader_item
    batch = batch.to(device)
    y = y.to(device)
    return model(batch), y


def train_one_epoch(
    model: LineupGNN,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> float:
    """Run one training epoch on ``loader``; return mean MSE."""
    model.train()
    total_loss = 0.0
    n_examples = 0
    for item in loader:
        optimizer.zero_grad(set_to_none=True)
        pred, y = _predict_diff(model, item, device)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n_examples += bs
    return total_loss / max(n_examples, 1)


@torch.no_grad()
def evaluate(
    model: LineupGNN,
    loader,
    device: torch.device | str,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return ``(mean_mse, predictions, targets)`` over ``loader``."""
    model.eval()
    total_loss = 0.0
    n_examples = 0
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for item in loader:
        pred, y = _predict_diff(model, item, device)
        loss = F.mse_loss(pred, y)
        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        n_examples += bs
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())
    return (
        total_loss / max(n_examples, 1),
        torch.cat(preds) if preds else torch.empty(0),
        torch.cat(targets) if targets else torch.empty(0),
    )


@torch.no_grad()
def predict(
    model: LineupGNN,
    loader,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the concatenated differentials over ``loader`` in eval mode."""
    model.eval()
    preds: list[torch.Tensor] = []
    for item in loader:
        pred, _ = _predict_diff(model, item, device)
        preds.append(pred.detach().cpu())
    return torch.cat(preds) if preds else torch.empty(0)
