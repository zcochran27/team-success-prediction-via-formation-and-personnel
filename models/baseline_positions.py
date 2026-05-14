"""Model 1: tabular regression with raw positional player features.

Lower-bound baseline. Isolates the signal available from the basic positional
composition of the modal lineup plus formation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class BaselinePositionsModel:
    """Sklearn-compatible regression model over the 12-feature tabular dataset.

    Primary candidate model type: gradient boosting (XGBoost or LightGBM).
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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselinePositionsModel":
        """Fit on raw-position tabular features."""
        # TODO: preprocess (one-hot positions + formation), fit estimator.
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target for each row in ``X``."""
        # TODO: run preprocessing + estimator forward pass.
        raise NotImplementedError
