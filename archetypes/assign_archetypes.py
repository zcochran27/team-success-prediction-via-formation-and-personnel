"""Assign each player their archetype label using the fit cluster models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def assign_archetypes(
    player_features: pd.DataFrame,
    archetype_models: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Add an ``archetype`` column to the player feature table.

    Returns the input frame plus an ``archetype`` column of the form
    ``"<position_group>-<cluster_index>"``.
    """
    out = player_features.copy()
    out["archetype"] = pd.NA
    for group, model in archetype_models.items():
        mask = out["position_group"] == group
        if not mask.any():
            continue
        X = out.loc[mask, model["feature_cols"]].values
        labels = model["pipeline"].predict(X)
        out.loc[mask, "archetype"] = [f"{group}-{int(c)}" for c in labels]
    return out


def save_player_archetype_map(assignments: pd.DataFrame, output_path: Path) -> None:
    """Persist the player->archetype mapping to ``output_path`` as parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["player_id", "position_group", "archetype"]
    assignments[cols].to_parquet(output_path, index=False)


def load_player_archetype_map(input_path: Path) -> pd.DataFrame:
    """Load a previously saved player->archetype mapping."""
    return pd.read_parquet(input_path)
