"""Assemble 12-feature team-season vectors for the tabular models.

Each team-season is represented as:
  - 11 player features (one per formation slot), each either a raw positional
    encoding (Model 1) or an archetype encoding (Model 2)
  - 1 formation feature (categorical, e.g. ``"4-3-3"``)

The same module produces both flavors via the ``player_feature_kind`` switch.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd


PlayerFeatureKind = Literal["position", "archetype"]


def get_modal_lineup(
    events_or_lineups: pd.DataFrame,
    team_id: str,
    season: str,
) -> pd.DataFrame:
    """Return the modal starting lineup for a single team-season.

    Parameters
    ----------
    events_or_lineups
        Source data containing starting lineups per match.
    team_id
        Team identifier.
    season
        Season identifier.

    Returns
    -------
    pandas.DataFrame
        One row per formation slot for the modal lineup, ordered by slot index.
    """
    # TODO: aggregate starting lineups across matches, pick the modal lineup.
    raise NotImplementedError


def get_modal_formation(
    events_or_lineups: pd.DataFrame,
    team_id: str,
    season: str,
) -> str:
    """Return the modal formation string (e.g. ``"4-3-3"``) for a team-season."""
    # TODO: count formations across matches and return the mode.
    raise NotImplementedError


def encode_player(
    player_row: pd.Series,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
) -> str | int:
    """Encode a single player as either a raw position or an archetype label.

    Parameters
    ----------
    player_row
        Row describing one player in the modal lineup.
    kind
        ``"position"`` for Model 1, ``"archetype"`` for Model 2.
    archetype_map
        Player→archetype mapping; required when ``kind == "archetype"``.

    Returns
    -------
    str | int
        Encoded player feature value.
    """
    # TODO: branch on ``kind`` and return the appropriate encoded value.
    raise NotImplementedError


def build_team_season_vector(
    events_or_lineups: pd.DataFrame,
    team_id: str,
    season: str,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
) -> pd.Series:
    """Assemble the 12-feature vector for one team-season."""
    # TODO: get modal lineup + formation, encode all 11 players, append formation.
    raise NotImplementedError


def build_tabular_dataset(
    events_or_lineups: pd.DataFrame,
    team_seasons: pd.DataFrame,
    kind: PlayerFeatureKind,
    archetype_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full tabular team-season dataset.

    Parameters
    ----------
    events_or_lineups
        Source data with per-match lineups and formations.
    team_seasons
        Index of all team-season rows to assemble.
    kind
        ``"position"`` for Model 1, ``"archetype"`` for Model 2.
    archetype_map
        Player→archetype mapping; required when ``kind == "archetype"``.

    Returns
    -------
    pandas.DataFrame
        One row per team-season with 12 feature columns plus targets joined in
        upstream.
    """
    # TODO: iterate team_seasons and concatenate ``build_team_season_vector`` rows.
    raise NotImplementedError
