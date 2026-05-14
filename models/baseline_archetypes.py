"""Model 2: tabular regression with archetype-encoded player features.

Identical structure to Model 1 but replaces each player's raw position with
their archetype label, isolating the marginal value of archetype enrichment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BaselineArchetypesModel:
    """Sklearn-compatible regression model over the 12-feature tabular dataset
    with archetype-encoded players.
    """

    def __init__(self, target: str, **model_kwargs: Any) -> None:
        """Initialize the model.

        Parameters
        ----------
        target
            ``"win_rate"`` or ``"points_per_game"``.
        **model_kwargs
            Forwarded to the underlying gradient boosting estimator.
        """
        self.target = target
        self.model_kwargs = model_kwargs
        # TODO: instantiate underlying estimator + preprocessing pipeline.
        self._pipeline: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineArchetypesModel":
        """Fit on archetype-encoded tabular features."""
        # TODO: preprocess (one-hot archetypes + formation), fit estimator.
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row in ``X``."""
        # TODO: run preprocessing + estimator forward pass.
        raise NotImplementedError
