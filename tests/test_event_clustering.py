"""Tests for :mod:`archetypes.event_clustering`."""

from __future__ import annotations


def test_build_event_feature_matrix_returns_numeric_frame() -> None:
    """Feature extraction produces a numeric DataFrame with no NaNs in required cols."""
    # TODO: fixture events, call builder, assert dtypes and shape.
    raise NotImplementedError


def test_fit_event_clusters_returns_predict_capable_model() -> None:
    """Fit model exposes a ``predict`` method returning labels in ``[0, n_clusters)``."""
    # TODO: small synthetic features, fit, call predict, assert label range.
    raise NotImplementedError


def test_fit_all_event_clusters_persists_models(tmp_path) -> None:
    """Fitting writes one model per (position group, event type) to disk."""
    # TODO: fixture events, run fit_all, assert files exist and load round-trips.
    raise NotImplementedError
