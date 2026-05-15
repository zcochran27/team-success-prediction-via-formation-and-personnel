"""Shared GNN backbone + train/predict helpers for the 4 graph models.

The 2 x 2 model matrix is configured via two arguments on
:class:`LineupGNN`:

- ``kind`` -- ``"position"`` or ``"archetype"`` (selects the embedding
  vocabulary).
- ``mode`` -- ``"single"`` (the 11-node single-team graph; team strength
  estimated per side and the loss supervises the difference) or
  ``"paired"`` (the 22-node graph with inter-team matchup edges and a
  same-team edge flag consumed via ``GATConv(edge_dim=1)``).

Paired-mode forward
-------------------
The 22-node graph is encoded with edge-aware ``GATConv`` layers, then the
per-snapshot node embeddings are mean-pooled separately for team 1 and
team 2 (using ``data.team``). The same MLP head is applied to each pool
and the prediction is ``head(team1_pool) - head(team2_pool)``. Sharing
the head and subtracting preserves the antisymmetry of single-mode: if
the same lineup faces itself the predicted differential is exactly zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

from features.build_graphs import GraphMode, PlayerFeatureKind, vocab_size


class LineupGNN(nn.Module):
    """Embedding -> stacked GATConv -> mean pool -> shared MLP head."""

    def __init__(
        self,
        kind: PlayerFeatureKind,
        mode: GraphMode = "single",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 1,
    ) -> None:
        super().__init__()
        if mode not in ("single", "paired"):
            raise ValueError(f"unknown mode {mode!r}")
        self.kind = kind
        self.mode: GraphMode = mode
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size(kind), hidden_dim)

        edge_dim = 1 if mode == "paired" else None
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, batch: Batch) -> torch.Tensor:
        x = self.embedding(batch.x)
        edge_attr = getattr(batch, "edge_attr", None) if self.mode == "paired" else None
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
            graph_emb = global_mean_pool(node_emb, batch.batch)
            return self.head(graph_emb).squeeze(-1)

        team_mask = batch.team
        t1_idx = (team_mask == 0).nonzero(as_tuple=False).squeeze(-1)
        t2_idx = (team_mask == 1).nonzero(as_tuple=False).squeeze(-1)
        pool_t1 = global_mean_pool(node_emb[t1_idx], batch.batch[t1_idx])
        pool_t2 = global_mean_pool(node_emb[t2_idx], batch.batch[t2_idx])
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
