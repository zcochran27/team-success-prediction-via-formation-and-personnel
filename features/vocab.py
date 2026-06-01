"""Shared vocabularies and dimension constants for the half-subs pipeline.

Contains the position / archetype label vocabularies the lineup-snapshot
data uses, the dimensions of the per-(player, season) stat vector and
its derived edge-feature block, plus the small ``vocab_size`` helper used
by both the graph builder and the model to size the embedding layer.

These constants are stable across snapshot variants -- they describe the
universe of slot labels and per-player features the project works in,
not the topology of any particular dataset.
"""

from __future__ import annotations

from typing import Literal


PlayerFeatureKind = Literal["position", "archetype"]
GraphMode = Literal["single", "paired"]

# Per-(player, season) stat vector width (see archetypes.season_stats.STAT_COLS).
STAT_DIM: int = 10

# Width of the stat-derived edge feature block appended by the graph builder
# when stats are supplied. See ``features.build_graphs_subs._finalize_edge_attr``
# for the column meanings.
STAT_EDGE_DIM: int = 4

# Union of (a) positions observed in the lineup-snapshot parquets and
# (b) slot labels used in graphs.templates.FORMATION_TEMPLATES.
POSITION_VOCAB: tuple[str, ...] = (
    "CAM", "CB", "CDM", "CM", "GK",
    "LAM", "LB", "LCB", "LM", "LW", "LWB", "LWF",
    "RAM", "RB", "RCB", "RM", "RW", "RWB", "RWF",
    "SS", "ST",
)

# Archetypes are ``"<position_group>-<cluster_idx>"`` plus a ``"MISSING"``
# sentinel. Many players don't have a season-level archetype assignment
# (sub minutes too low, missing season data, etc.) and the dataset
# deliberately keeps those rows -- the fact-of-missingness is itself a
# learnable signal, so NaN archetype labels are mapped to
# ``MISSING_ARCHETYPE`` before embedding.
MISSING_ARCHETYPE: str = "MISSING"
ARCHETYPE_VOCAB: tuple[str, ...] = (
    "CD-0", "CD-1", "CD-2",
    "CF-0", "CF-1", "CF-2",
    "CM-0", "CM-1", "CM-2", "CM-3",
    "GK-0", "GK-1",
    "LWD-0", "LWD-1", "LWD-2",
    "LWP-0", "LWP-1", "LWP-2",
    "RWD-0", "RWD-1", "RWD-2",
    "RWP-0", "RWP-1", "RWP-2",
    MISSING_ARCHETYPE,
)


def vocab_size(kind: PlayerFeatureKind) -> int:
    """Return ``num_embeddings`` for the matching ``nn.Embedding``."""
    if kind == "position":
        return len(POSITION_VOCAB)
    if kind == "archetype":
        return len(ARCHETYPE_VOCAB)
    raise ValueError(f"unknown kind {kind!r}")
