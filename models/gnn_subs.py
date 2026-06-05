"""GNN for the half-with-subs dataset.

A typed-edge GAT/GATv2/TransformerConv stack with attention pooling over the
variable-size per-half graphs from build_half_subs_graph. Notable points:

  - Variable node count. Graphs span 11 to 22 nodes per team. PyG's Batch
    handles this; the only place the model touches n_nodes is the attention
    pool, which uses batch.batch indices.
  - Per-node block. On top of the categorical embedding and template coords,
    the model consumes a 5-D numeric block (is_starter, duration, start_min,
    end_min, was_substituted_off) via a learned linear projection added to
    the embedding. Set use_stats to also project the 10-D season-stat vector.
  - Typed edges. edge_attr carries a 3-D edge-type one-hot (formation, sub,
    matchup) plus per-type continuous features (sub minute, same-position
    flag, overlap fraction). All edges share the same edge_attr width;
    irrelevant columns are 0. GATConv variants consume edge_attr directly, so
    the layers can learn type-conditional attention.

The toggleable axes:
  - kind: position | archetype (selects the embedding vocab)
  - mode: single | paired (single forwards each team; paired builds a joint
    graph with matchup edges)
  - use_stats: include season stats and stat-diff edge features
  - use_coords: include template (x, y) in node embeddings

In paired mode, prediction is head(team1_pool) - head(team2_pool) so the
same lineup facing itself predicts exactly zero.
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

from features.build_graphs_subs import (
    EDGE_BASE_DIM,
    EDGE_STATS_DIM,
    NODE_NUM_DIM,
    edge_attr_dim,
)
from features.vocab import (
    GraphMode,
    PlayerFeatureKind,
    STAT_DIM,
    vocab_size,
)


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
    edge_dim: int,
) -> nn.Module:
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


class HalfSubsGNN(nn.Module):
    """Variable-size graph regressor for the half-with-subs dataset.

    Forward expects a PyG Batch whose nodes carry:
      x:         (N,) long, categorical embedding index
      x_num:     (N, 5) float (is_starter, duration, start, end, was_subbed_off)
      pos:       (N, 2) float (normalized template coords)
      stats:     (N, 10) float, present when use_stats
      edge_attr: (E, edge_dim) float from features.build_graphs_subs
      team:      (N,) long, paired mode only (0 = team1, 1 = team2)
    """

    def __init__(
        self,
        kind: PlayerFeatureKind,
        mode: GraphMode = "paired",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 1,
        use_coords: bool = True,
        use_stats: bool = False,
        pool: PoolKind = "attention",
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
        self.num_proj = nn.Linear(NODE_NUM_DIM, hidden_dim)
        self.coord_proj = nn.Linear(2, hidden_dim) if use_coords else None
        self.stats_proj = nn.Linear(STAT_DIM, hidden_dim) if use_stats else None

        # All edges carry the base block plus type one-hot; the stats-diff
        # block is appended when use_stats is set.
        edge_dim = edge_attr_dim(use_stats)
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

    def _encode_nodes(self, batch: Batch) -> torch.Tensor:
        x = self.embedding(batch.x)
        x = x + self.num_proj(batch.x_num)
        if self.use_coords:
            x = x + self.coord_proj(batch.pos)
        if self.use_stats:
            x = x + self.stats_proj(batch.stats)
        for conv in self.convs:
            x = conv(x, batch.edge_index, edge_attr=batch.edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return one scalar per graph in batch.

        - mode="single": pools every node in each graph; the trainer calls
          this once per team and subtracts the two predictions.
        - mode="paired": splits nodes by batch.team and returns
          head(team1_pool) - head(team2_pool).
        """
        node_emb = self._encode_nodes(batch)
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
    model: HalfSubsGNN,
    loader_item,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward one loader item, returning (prediction, target)."""
    if model.mode == "single":
        b1, b2, y = loader_item
        b1 = b1.to(device); b2 = b2.to(device); y = y.to(device)
        return model(b1) - model(b2), y
    batch, y = loader_item
    batch = batch.to(device); y = y.to(device)
    return model(batch), y


@torch.no_grad()
def predict(
    model: HalfSubsGNN,
    loader,
    device: torch.device | str,
) -> torch.Tensor:
    """Return concatenated predictions over loader in eval mode."""
    model.eval()
    preds: list[torch.Tensor] = []
    for item in loader:
        pred, _ = _predict_diff(model, item, device)
        preds.append(pred.detach().cpu())
    return torch.cat(preds) if preds else torch.empty(0)
