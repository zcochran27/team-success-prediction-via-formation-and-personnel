"""Model-agnostic k-fold cross-validation harness.

Works for both:
  - sklearn-style models with a tabular ``pandas.DataFrame`` ``X`` and a ``y``
    series (Models 1 and 2)
  - PyG-style models with a list of ``Data`` graphs and per-graph targets
    (Models 3 and 4)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def kfold_indices(
    n: int,
    k: int,
    random_state: int = 0,
    shuffle: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``k`` ``(train_idx, val_idx)`` index pairs for a dataset of size ``n``."""
    # TODO: implement using ``sklearn.model_selection.KFold``.
    raise NotImplementedError


def cross_validate(
    model_factory: Any,
    data: pd.DataFrame | Sequence[Any],
    targets: pd.Series | Sequence[float],
    k: int,
    random_state: int = 0,
) -> dict[str, Any]:
    """Run k-fold CV for a single model on a single target.

    Parameters
    ----------
    model_factory
        Zero-argument callable returning a fresh, unfit model exposing
        ``fit`` / ``predict``.
    data
        Either a tabular ``DataFrame`` or a sequence of PyG ``Data`` graphs.
    targets
        Per-row / per-graph targets aligned with ``data``.
    k
        Number of folds.
    random_state
        Seed for fold assignment.

    Returns
    -------
    dict
        Keys: ``per_fold_metrics`` (list of metric dicts), ``aggregate``
        (mean and std per metric), ``predictions`` (concatenated out-of-fold
        predictions and ground truth).
    """
    # TODO: loop folds, fit/predict, compute metrics, aggregate.
    raise NotImplementedError
