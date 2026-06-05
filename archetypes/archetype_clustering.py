"""Cluster per-player feature vectors into archetypes, per position group.

Clustering runs separately within each of the eight position groups, so an
archetype label only means something within its group (CD-0 and CM-0 are
unrelated).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .player_aggregation import feature_columns_for_group


def fit_archetype_clusters(
    player_features: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit StandardScaler + KMeans on one position group's feature vectors.

    player_features should already be restricted to one group. Returns
    {"pipeline": fitted Pipeline, "feature_cols": ...}.
    """
    X = player_features[feature_cols].values
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
        )),
    ])
    pipe.fit(X)
    return {"pipeline": pipe, "feature_cols": list(feature_cols)}


def fit_all_archetype_clusters(
    player_features: pd.DataFrame,
    event_types_per_group: dict[str, list[str]],
    n_clusters_per_event: dict[str, dict[str, int]],
    n_archetypes_per_group: dict[str, int],
    output_dir: Path,
    random_state: int = 0,
) -> dict[str, dict[str, Any]]:
    """Fit one archetype pipeline per position group and save each to
    output_dir as <group>.joblib.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, dict[str, Any]] = {}
    for group, n_clusters in n_archetypes_per_group.items():
        gdf = player_features[player_features["position_group"] == group]
        if len(gdf) < n_clusters:
            print(f"  [skip] {group}: only {len(gdf)} players (< n_clusters={n_clusters})")
            continue
        cols = feature_columns_for_group(event_types_per_group[group], n_clusters_per_event[group])
        model = fit_archetype_clusters(
            gdf,
            feature_cols=cols,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        joblib.dump(model, output_dir / f"{group}.joblib")
        models[group] = model
        print(f"  [fit ] {group}: {len(gdf):>5,} players -> k={n_clusters}")
    return models


def load_archetype_clusters(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Load previously fit per-position-group archetype models."""
    input_dir = Path(input_dir)
    models: dict[str, dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*.joblib")):
        models[path.stem] = joblib.load(path)
    return models
