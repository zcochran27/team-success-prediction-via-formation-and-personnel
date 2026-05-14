"""Model 4: GNN regression with archetype-encoded node features.

Same architecture as :mod:`models.gnn_positions`; node features are archetype
encodings rather than raw positional encodings. Combines graph structure with
archetype enrichment — expected to be the strongest model.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class GNNArchetypesModel:
    """PyTorch Geometric GNN over team-season graphs with archetype node features."""

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

    def fit(self, graphs: Iterable[Any], targets: Iterable[float] | None = None) -> "GNNArchetypesModel":
        """Train the GNN on a collection of PyG ``Data`` objects."""
        # TODO: build DataLoader, training loop with MSE loss.
        raise NotImplementedError

    def predict(self, graphs: Iterable[Any]) -> np.ndarray:
        """Run the GNN forward over each graph and return predicted scalars."""
        # TODO: eval mode, batched forward, concatenate outputs.
        raise NotImplementedError
