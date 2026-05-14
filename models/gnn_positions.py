"""Model 3: GNN regression with raw positional node features.

Architecture (default)
----------------------
- 2-3 GNN layers (GCN or GAT)
- Global mean pooling over the 11 player nodes
- MLP head producing a single regression scalar (win rate or PPG)
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class GNNPositionsModel:
    """PyTorch Geometric GNN over team-season graphs with raw positional features."""

    def __init__(
        self,
        target: str,
        gnn_layers: int = 2,
        hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        **kwargs: Any,
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        target
            ``"win_rate"`` or ``"points_per_game"``.
        gnn_layers, hidden_dim, learning_rate, epochs
            Architecture / training hyperparameters.
        **kwargs
            Additional options (e.g. layer type, dropout) passed through.
        """
        self.target = target
        self.gnn_layers = gnn_layers
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.kwargs = kwargs
        # TODO: build the torch ``nn.Module`` + optimizer.
        self._module: Any = None
        self._optimizer: Any = None

    def fit(self, graphs: Iterable[Any], targets: Iterable[float] | None = None) -> "GNNPositionsModel":
        """Train the GNN on a collection of PyG ``Data`` objects.

        Parameters
        ----------
        graphs
            Iterable of PyG ``Data`` graphs.
        targets
            Optional explicit targets; if omitted, targets are read from ``graph.y``.
        """
        # TODO: build DataLoader, training loop with MSE loss.
        raise NotImplementedError

    def predict(self, graphs: Iterable[Any]) -> np.ndarray:
        """Run the GNN forward over each graph and return predicted scalars."""
        # TODO: eval mode, batched forward, concatenate outputs.
        raise NotImplementedError
