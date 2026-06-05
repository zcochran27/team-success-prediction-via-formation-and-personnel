"""Per-(player, season) summary stats from the raw event log.

Where archetypes give each player-season a discrete role token, this builds
a continuous 10-dim feature vector summarizing how often and how well a
player performed the five most common on-ball actions.

Five per-90 volumes: shots, passes, dribbles, defensive duels, aerial
duels. Five success rates: shot conversion, pass completion, dribble
success, defensive-duel stop rate, aerial-duel win rate. Rates with a zero
denominator are set to 0.0 (matching the "MISSING" archetype convention).
The result is one row per (player_id, season), with no minimum-minutes
filter.

Output: data/processed/player_season_stats.parquet, with columns player_id,
season, minutes, then the ten stat columns in STAT_COLS order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Column order. The GNN dataset relies on this to build fixed-width
# per-player feature vectors, so don't reorder without updating callers.
STAT_COLS: tuple[str, ...] = (
    "shots_per_90",
    "passes_per_90",
    "dribbles_per_90",
    "def_duels_per_90",
    "aerial_duels_per_90",
    "shot_conv_rate",
    "pass_comp_rate",
    "dribble_success_rate",
    "def_duel_stop_rate",
    "aerial_duel_win_rate",
)


def _season_year_map(seasons: pd.DataFrame) -> pd.Series:
    """Map seasonId to a 4-digit calendar year (parsed from seasons.name)."""
    return seasons.set_index("seasonId")["name"].str[:4].astype(int)


def _attach_season(
    df: pd.DataFrame,
    match_id_col: str,
    matches: pd.DataFrame,
    seasons: pd.DataFrame,
) -> pd.Series:
    """Return a season Series aligned with df (NaN where unknown)."""
    season_id = df[match_id_col].map(matches.set_index("wyId")["seasonId"])
    return season_id.map(_season_year_map(seasons))


def _bool_eq_true(s: pd.Series) -> pd.Series:
    """Return s == True as a bool Series (NaN or non-True become False).

    Many Wyscout flag columns are object dtype with a mix of True/False/NaN.
    Using .eq(True) avoids pandas' deprecated object downcasting warning while
    keeping the same result as .fillna(False).astype(bool).
    """
    return s.eq(True)


def tag_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add boolean count columns flagging each event's category and success.

    Vectorized over the event log, so it runs fine on the full ~13M-row export.
    """
    out = events.copy()
    tp = out["type_primary"]

    # shots
    is_shot = tp == "shot"
    out["_shots"] = is_shot
    out["_goals"] = is_shot & _bool_eq_true(out["shot_isGoal"])

    # passes
    is_pass = tp == "pass"
    out["_passes"] = is_pass
    out["_passes_complete"] = is_pass & _bool_eq_true(out["pass_accurate"])

    # duels (defensive, aerial, dribble share the duel mask)
    is_duel = tp == "duel"
    sec_str = out["type_secondary"].astype(str)
    is_def_duel = is_duel & sec_str.str.contains("defensive_duel", regex=False, na=False)
    is_aerial_duel = is_duel & sec_str.str.contains("aerial_duel", regex=False, na=False)
    is_dribble = is_duel & _bool_eq_true(out["groundDuel_takeOn"])

    out["_def_duels"] = is_def_duel
    out["_def_duels_won"] = is_def_duel & _bool_eq_true(out["groundDuel_stoppedProgress"])
    out["_aerial_duels"] = is_aerial_duel
    out["_aerial_duels_won"] = is_aerial_duel & _bool_eq_true(out["aerialDuel_firstTouch"])
    out["_dribbles"] = is_dribble
    out["_dribbles_success"] = is_dribble & _bool_eq_true(out["groundDuel_progressedWithBall"])

    return out


def compute_minutes(
    formations: pd.DataFrame,
    matches: pd.DataFrame,
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Sum each player's on-pitch interval lengths per (player_id, season).

    Returns a DataFrame with player_id, season, minutes. Rows are dropped
    when the season can't be resolved.
    """
    intervals = formations.assign(
        minutes=formations["period_end_min"] - formations["period_start_min"],
    )
    intervals["season"] = _attach_season(intervals, "matchId", matches, seasons)
    return (
        intervals.dropna(subset=["season"])
                 .groupby(["playerId", "season"], as_index=False)["minutes"].sum()
                 .rename(columns={"playerId": "player_id"})
    )


def aggregate_events(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    seasons: pd.DataFrame,
) -> pd.DataFrame:
    """Sum the per-event count flags per (player_id, season).

    The input should already have been through tag_events (or have the same
    _-prefixed columns). The result has one row per (player_id, season) with
    one column per count.
    """
    flag_cols = [c for c in events.columns if c.startswith("_")]
    if not flag_cols:
        raise ValueError("events frame has no _-prefixed flag columns; run tag_events first")
    df = events.copy()
    df["season"] = _attach_season(df, "matchId", matches, seasons)
    return (
        df.dropna(subset=["season", "player_id"])
          .groupby(["player_id", "season"], as_index=False)[flag_cols].sum()
    )


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """numerator / denominator, with 0/0 returning 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(
            denominator > 0,
            numerator / denominator.replace(0, np.nan),
            0.0,
        )
    return np.where(np.isnan(rate), 0.0, rate)


def build_season_stats(
    events: pd.DataFrame | Path,
    formations: pd.DataFrame | Path,
    matches: pd.DataFrame | Path,
    seasons: pd.DataFrame | Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the per-(player, season) stat table.

    Any input may be a DataFrame or a parquet Path. This is the function the
    unit tests target, so the call sites stay flexible.
    """
    def _load(arg: pd.DataFrame | Path) -> pd.DataFrame:
        return arg if isinstance(arg, pd.DataFrame) else pd.read_parquet(arg)

    events_df = _load(events)
    formations_df = _load(formations)
    matches_df = _load(matches)
    seasons_df = _load(seasons)

    minutes = compute_minutes(formations_df, matches_df, seasons_df)
    tagged = tag_events(events_df)
    counts = aggregate_events(tagged, matches_df, seasons_df)

    # Players in formations but with no events get 0 counts; players with
    # events but no formation record are dropped (no minutes, no per-90).
    df = minutes.merge(counts, on=["player_id", "season"], how="left").fillna(0.0)
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    df["minutes"] = df["minutes"].astype(float)

    minutes_90 = df["minutes"] / 90.0
    df["shots_per_90"]        = _safe_rate(df["_shots"],        minutes_90)
    df["passes_per_90"]       = _safe_rate(df["_passes"],       minutes_90)
    df["dribbles_per_90"]     = _safe_rate(df["_dribbles"],     minutes_90)
    df["def_duels_per_90"]    = _safe_rate(df["_def_duels"],    minutes_90)
    df["aerial_duels_per_90"] = _safe_rate(df["_aerial_duels"], minutes_90)

    df["shot_conv_rate"]       = _safe_rate(df["_goals"],            df["_shots"])
    df["pass_comp_rate"]       = _safe_rate(df["_passes_complete"],  df["_passes"])
    df["dribble_success_rate"] = _safe_rate(df["_dribbles_success"], df["_dribbles"])
    df["def_duel_stop_rate"]   = _safe_rate(df["_def_duels_won"],    df["_def_duels"])
    df["aerial_duel_win_rate"] = _safe_rate(df["_aerial_duels_won"], df["_aerial_duels"])

    keep = ["player_id", "season", "minutes", *STAT_COLS]
    out = df[keep].reset_index(drop=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
    return out
