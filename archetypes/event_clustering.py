"""Intra-event clustering.

For each (position group, event type) pair, fit a clustering model over
individual events to find distinct event subtypes (short vs long passes,
and so on). Each fitted pipeline (StandardScaler + MiniBatchKMeans) is
saved with its feature column order so player_aggregation.py can reapply
the same transform when assigning cluster labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .position_groups import filter_events_by_group


# Per-event-type feature engineering

def _safe_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _pass_features(e: pd.DataFrame) -> pd.DataFrame:
    sx = _safe_float(e["start_x"])
    sy = _safe_float(e["start_y"])
    ex = _safe_float(e["end_x"])
    ey = _safe_float(e["end_y"])
    length = _safe_float(e["pass_length"])
    angle = _safe_float(e["pass_angle"])
    dx = ex - sx
    dy = ey - sy
    out = pd.DataFrame({
        "start_x": sx,
        "start_y": sy,
        "end_x": ex,
        "end_y": ey,
        "length": length,
        "angle": angle,
        "delta_x": dx,
        "delta_y_abs": dy.abs(),
        "is_high": (e["pass_height"] == "high").astype("float"),
    }, index=e.index)
    return out


def _shot_features(e: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_x": _safe_float(e["start_x"]),
        "start_y": _safe_float(e["start_y"]),
        "right_foot": (e["shot_bodyPart"] == "right_foot").astype("float"),
        "left_foot": (e["shot_bodyPart"] == "left_foot").astype("float"),
        "head": (e["shot_bodyPart"] == "head_or_other").astype("float"),
    }, index=e.index)
    return out


def _offensive_duel_features(e: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_x": _safe_float(e["start_x"]),
        "start_y": _safe_float(e["start_y"]),
    }, index=e.index)
    return out


def _defensive_duel_features(e: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_x": _safe_float(e["start_x"]),
        "start_y": _safe_float(e["start_y"]),
    }, index=e.index)
    return out


def _aerial_duel_features(e: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_x": _safe_float(e["start_x"]),
        "start_y": _safe_float(e["start_y"]),
    }, index=e.index)
    return out


def _goalkeeper_exit_features(e: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "start_x": _safe_float(e["start_x"]),
        "start_y": _safe_float(e["start_y"]),
    }, index=e.index)
    return out


_FEATURE_BUILDERS = {
    "pass": _pass_features,
    "shot": _shot_features,
    "offensive_duel": _offensive_duel_features,
    "defensive_duel": _defensive_duel_features,
    "aerial_duel": _aerial_duel_features,
    "goalkeeper_exit": _goalkeeper_exit_features,
}


def build_event_feature_matrix(events: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Build the numeric feature matrix for clustering events of event_type.

    Rows with any NaN are dropped, so the model is fit only on complete rows.
    """
    if event_type not in _FEATURE_BUILDERS:
        raise KeyError(f"unknown event_type: {event_type!r}")
    feats = _FEATURE_BUILDERS[event_type](events)
    return feats.dropna()


# Model fit, save, load

def fit_event_clusters(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit StandardScaler + MiniBatchKMeans on event features.

    Returns {"pipeline": fitted Pipeline, "feature_cols": list[str]}. The
    column order is stored so player_aggregation can rebuild the same feature
    matrix at predict time.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            batch_size=4096,
        )),
    ])
    pipe.fit(features.values)
    return {"pipeline": pipe, "feature_cols": list(features.columns)}


def fit_all_event_clusters(
    events: pd.DataFrame,
    event_types_per_group: dict[str, list[str]],
    n_clusters_per_event: dict[str, dict[str, int]],
    output_dir: Path,
    random_state: int = 0,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Fit one pipeline per (position group, event type) and save each to
    output_dir as <group>__<event_type>.joblib.

    events must already have position_group and logical_event_type columns
    (see position_groups.py). n_clusters_per_event is nested as
    {group: {event_type: k}}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models: dict[tuple[str, str], dict[str, Any]] = {}
    for group, event_types in event_types_per_group.items():
        for et in event_types:
            k = n_clusters_per_event[group][et]
            subset = filter_events_by_group(events, group, [et])
            feats = build_event_feature_matrix(subset, et)
            if len(feats) < k:
                # not enough samples to cluster
                print(f"  [skip] {group}/{et}: only {len(feats)} usable events")
                continue
            model = fit_event_clusters(
                feats,
                n_clusters=k,
                random_state=random_state,
            )
            joblib.dump(model, output_dir / f"{group}__{et}.joblib")
            models[(group, et)] = model
            print(f"  [fit ] {group}/{et}: {len(feats):>9,} events -> k={k}")
    return models


def load_event_clusters(input_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load previously fit (position group, event type) cluster models."""
    input_dir = Path(input_dir)
    models: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(input_dir.glob("*__*.joblib")):
        stem = path.stem  # e.g. "CD__pass"
        group, _, event_type = stem.partition("__")
        models[(group, event_type)] = joblib.load(path)
    return models


def predict_event_clusters(events: pd.DataFrame, model: dict[str, Any], event_type: str) -> np.ndarray:
    """Predict cluster labels for events using model.

    Returns an int array the same length as events; rows without a full
    feature vector get -1.
    """
    feats = _FEATURE_BUILDERS[event_type](events)
    feats = feats[model["feature_cols"]]
    valid = feats.notna().all(axis=1)
    labels = np.full(len(events), -1, dtype=int)
    if valid.any():
        labels[valid.values] = model["pipeline"].predict(feats.loc[valid].values)
    return labels
