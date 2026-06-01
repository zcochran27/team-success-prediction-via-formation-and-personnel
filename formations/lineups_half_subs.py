"""Half-level lineup snapshot with explicit starter + substitute feature blocks.

Each (match, half) becomes two focal-perspective rows (one per side as
team1), with these per-team feature blocks:

  * **Starter block** -- 11 slots in formation-template order. Per slot:
    ``player_id``, ``position``, ``archetype``, the 10-D season stat
    vector, and ``duration`` (minutes on the pitch in this half).
    ``team1_position_1..11`` / ``team2_position_1..11`` mirror the
    column names the rest of the pipeline (and
    :func:`features.snapshots.filter_buildable_snapshots`) expect, so
    this dataset slots into the existing tabular comparison harness.
  * **Sub block** -- 11 slots in chronological-entry order (1st sub at
    slot 1, 11th sub at slot 11). Per slot: ``player_id``, ``position``
    (modal recorded position over their appearances), ``archetype``,
    season stats, ``start_min`` (when they came on, in match-clock
    minutes), and ``duration``. Unused slots are NaN-padded; in pro-league
    halves slot 11 is filled in <0.6% of rows.

Why 11 + 11 = 22 slots? The empirical 22-player cap covers ~99.4% of
pro-league half-sides; the remaining tail is dominated by NCAA matches
where the substitution rule is unlimited and re-entry is sometimes
allowed (which would also break the chronological-slot semantics). Any
subs past slot 11 in the source data are dropped with a warning logged
once per build.

Match-state features (xG, goals, per-30 rates) are recomputed at
half granularity by re-running :func:`formations.match_state.add_match_state_features`
with the half boundaries as the snapshot window.

Outputs (suffixed ``_half_subs`` so they coexist with the other variants):

  * ``data/processed/lineup_snapshots_half_subs.parquet``
  * ``data/processed/train_snapshots_half_subs.parquet``
  * ``data/processed/test_snapshots_half_subs.parquet``

Usage (from repo root)::

    python -m formations.lineups_half_subs
    python -m formations.lineups_half_subs --test-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from archetypes.season_stats import STAT_COLS
from features.snapshots import filter_buildable_snapshots
from graphs.alignment import align_lineup_to_template

from .match_state import add_match_state_features


# When a team's final period has ``period_end_min == NaN`` we replace it
# with the match's end-of-clock minute so half-clipping math stays finite.
# ``matches.duration`` only distinguishes regulation vs. ET, so we pick a
# generous stoppage allowance for each category. Penalties are excluded
# from the match clock (the shootout uses its own timeline), so a
# Penalties match's continuous clock still ends at the end of ET.
_DURATION_TO_END_MIN: dict[str, float] = {
    "Regular":   95.0,
    "ExtraTime": 125.0,
    "Penalties": 125.0,
}

N_STARTER_SLOTS = 11
N_SUB_SLOTS = 11
# Tolerance (in minutes) for treating a player's first appearance as
# "exactly at half-start". The formations table rounds period boundaries
# to ~0.1 min, so any non-zero EPS just absorbs that quantization.
_START_EPS = 0.05


def _match_to_season(matches_path: Path, seasons_path: Path) -> pd.Series:
    """Map ``match_id -> season_year`` (4-digit int from ``seasons.name``)."""
    matches = pd.read_parquet(matches_path, columns=["wyId", "seasonId"])
    seasons = pd.read_parquet(seasons_path, columns=["seasonId", "name"])
    seasons["season"] = seasons["name"].str.extract(r"(\d{4})")[0].astype(int)
    return (
        matches.merge(seasons[["seasonId", "season"]], on="seasonId", how="left")
               .set_index("wyId")["season"]
    )


def _clip_periods_to_half(
    team_periods: pd.DataFrame, half_start: float, half_end: float
) -> pd.DataFrame:
    """Return periods that overlap ``[half_start, half_end)`` with start/end clipped."""
    df = team_periods[
        (team_periods["period_start_min"] < half_end)
        & (team_periods["period_end_min"] > half_start)
    ].copy()
    if df.empty:
        return df
    df["_clip_start"] = df["period_start_min"].clip(lower=half_start)
    df["_clip_end"] = df["period_end_min"].clip(upper=half_end)
    df["_duration"] = df["_clip_end"] - df["_clip_start"]
    return df


def _per_player_summary(clipped: pd.DataFrame) -> pd.DataFrame:
    """One row per player_id with first_in / last_out / duration / modal position.

    Modal position breaks ties by picking the position from the
    *earliest* clipped period (so a starter who later shifted slots keeps
    their initial role unless they spent more time elsewhere).
    """
    # Per (playerId, position) total duration, then keep the (playerId, position)
    # with the largest duration per playerId. Ties broken by earliest _clip_start
    # via a secondary sort.
    pos_dur = (
        clipped.groupby(["playerId", "position"], as_index=False)
               .agg(dur=("_duration", "sum"), first=("_clip_start", "min"))
    )
    pos_dur = pos_dur.sort_values(["playerId", "dur", "first"], ascending=[True, False, True])
    modal_pos = pos_dur.drop_duplicates("playerId", keep="first").set_index("playerId")["position"]

    pp = (
        clipped.groupby("playerId", as_index=False)
               .agg(
                   first_in=("_clip_start", "min"),
                   last_out=("_clip_end", "max"),
                   duration=("_duration", "sum"),
               )
    )
    pp["position"] = pp["playerId"].map(modal_pos)
    return pp


def _starter_initial_positions(
    clipped: pd.DataFrame, starter_ids: list[int], starting_period_index: int
) -> dict[int, str]:
    """Position each starter held in the team's first period of the half.

    Falls back to the starter's overall modal position if a starter
    somehow isn't in the starting period (defensive guard -- in practice
    every starter is by definition in the starting period).
    """
    first = clipped[clipped["period_index"] == starting_period_index]
    pos_in_first = first.set_index("playerId")["position"].to_dict()
    return {pid: pos_in_first.get(pid) for pid in starter_ids}


def _zero_stats() -> np.ndarray:
    return np.zeros(len(STAT_COLS), dtype=np.float32)


def _build_team_half_block(
    side: str,
    clipped: pd.DataFrame,
    half_start: float,
    half_end: float,
    season: int,
    archetype_lookup: dict[tuple[int, int], str],
    stats_lookup: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any] | None:
    """Build the per-team column dict for one (match, team, half).

    Returns ``None`` when the half can't be assembled (e.g., team has no
    periods overlapping the half). Caller should drop those rows.
    """
    if clipped.empty:
        return None

    summary = _per_player_summary(clipped)

    # Starting period: the period with the smallest clipped start time.
    starting_idx = int(clipped.loc[clipped["_clip_start"].idxmin(), "period_index"])
    starting_formation = clipped.loc[clipped["period_index"] == starting_idx, "formation"].iloc[0]

    starter_mask = summary["first_in"] <= half_start + _START_EPS
    starters = summary[starter_mask].copy()
    subs = summary[~starter_mask].copy().sort_values("first_in")

    # Initial positions for starter slots come from the starting period --
    # this is what defines the starting formation, so it's the right thing
    # to align to the template (not the modal position).
    starter_ids = starters["playerId"].astype(int).tolist()
    initial_pos_by_pid = _starter_initial_positions(clipped, starter_ids, starting_idx)
    starters["initial_position"] = starters["playerId"].astype(int).map(initial_pos_by_pid)

    if len(starters) != N_STARTER_SLOTS or starters["initial_position"].isna().any():
        # Data quirk -- typically a team-half with a partial period. We
        # can't align without 11 starters, so emit a row with NaN starter
        # slots so the buildable filter catches and drops it downstream.
        starter_perm: list[int] | None = None
    else:
        try:
            starter_perm = align_lineup_to_template(
                starting_formation, starters["initial_position"].tolist(),
            )
        except (KeyError, ValueError):
            starter_perm = None

    row: dict[str, Any] = {f"{side}_formation": starting_formation}

    # ---- Starter slots in template order. ---------------------------------
    for k in range(N_STARTER_SLOTS):
        if starter_perm is not None:
            r = starters.iloc[starter_perm[k]]
            pid = int(r["playerId"])
            arch = archetype_lookup.get((pid, season))
            stats = stats_lookup.get((pid, season), _zero_stats())
            row[f"{side}_player_{k+1}"] = pid
            row[f"{side}_position_{k+1}"] = r["initial_position"]
            row[f"{side}_archetype_{k+1}"] = arch
            row[f"{side}_starter_duration_{k+1}"] = float(r["duration"])
            for j, stat_name in enumerate(STAT_COLS):
                row[f"{side}_starter_p{k+1}_{stat_name}"] = float(stats[j])
        else:
            row[f"{side}_player_{k+1}"] = pd.NA
            row[f"{side}_position_{k+1}"] = None
            row[f"{side}_archetype_{k+1}"] = None
            row[f"{side}_starter_duration_{k+1}"] = np.nan
            for stat_name in STAT_COLS:
                row[f"{side}_starter_p{k+1}_{stat_name}"] = 0.0

    # ---- Sub slots in chronological order, padded to N_SUB_SLOTS. ---------
    sub_records = subs.head(N_SUB_SLOTS)
    for k in range(N_SUB_SLOTS):
        if k < len(sub_records):
            r = sub_records.iloc[k]
            pid = int(r["playerId"])
            arch = archetype_lookup.get((pid, season))
            stats = stats_lookup.get((pid, season), _zero_stats())
            row[f"{side}_sub_player_{k+1}"] = pid
            row[f"{side}_sub_position_{k+1}"] = r["position"]
            row[f"{side}_sub_archetype_{k+1}"] = arch
            row[f"{side}_sub_start_min_{k+1}"] = float(r["first_in"])
            row[f"{side}_sub_duration_{k+1}"] = float(r["duration"])
            for j, stat_name in enumerate(STAT_COLS):
                row[f"{side}_sub_p{k+1}_{stat_name}"] = float(stats[j])
        else:
            row[f"{side}_sub_player_{k+1}"] = pd.NA
            row[f"{side}_sub_position_{k+1}"] = None
            row[f"{side}_sub_archetype_{k+1}"] = None
            row[f"{side}_sub_start_min_{k+1}"] = np.nan
            row[f"{side}_sub_duration_{k+1}"] = np.nan
            for stat_name in STAT_COLS:
                row[f"{side}_sub_p{k+1}_{stat_name}"] = 0.0

    return row


def _load_archetype_lookup(path: Path) -> dict[tuple[int, int], str]:
    amap = pd.read_parquet(path, columns=["player_id", "season", "archetype"])
    return {(int(p), int(s)): a for p, s, a in amap.itertuples(index=False)}


def _load_stats_lookup(path: Path) -> dict[tuple[int, int], np.ndarray]:
    df = pd.read_parquet(path)
    out: dict[tuple[int, int], np.ndarray] = {}
    cols = list(STAT_COLS)
    for row in df[["player_id", "season"] + cols].itertuples(index=False):
        out[(int(row[0]), int(row[1]))] = np.asarray(row[2:], dtype=np.float32)
    return out


def build_half_subs_snapshots(
    formations: pd.DataFrame,
    matches: pd.DataFrame,
    matches_path: Path,
    seasons_path: Path,
    archetype_map_path: Path,
    stats_path: Path,
    events_path: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build the half-level lineup-with-subs snapshot table (home/away framing)."""
    home_by_match = matches.set_index("wyId")[["home_team", "away_team"]]
    end_min_by_match = (
        matches.set_index("wyId")["duration"].map(_DURATION_TO_END_MIN).fillna(95.0)
    )
    season_by_match = _match_to_season(matches_path, seasons_path)

    print(f"[load] archetype map  {archetype_map_path}")
    archetype_lookup = _load_archetype_lookup(archetype_map_path)
    print(f"       {len(archetype_lookup):,} (player, season) -> archetype entries")
    print(f"[load] season stats   {stats_path}")
    stats_lookup = _load_stats_lookup(stats_path)
    print(f"       {len(stats_lookup):,} (player, season) -> stat-vec entries")

    print("[build] aggregating per (match, team, half)")
    rows: list[dict[str, Any]] = []
    n_subs_overflow = 0

    for match_id, mdf in formations.groupby("matchId", sort=False):
        if match_id not in home_by_match.index:
            continue
        home_id = int(home_by_match.loc[match_id, "home_team"])
        away_id = int(home_by_match.loc[match_id, "away_team"])
        team_ids_in_data = set(mdf["teamId"].unique())
        if not {home_id, away_id}.issubset(team_ids_in_data):
            continue

        match_end = float(end_min_by_match.loc[match_id])
        season = season_by_match.get(match_id)
        if pd.isna(season):
            continue
        season = int(season)

        # Cap each team's open-ended last period at the match end (mirrors
        # the canonical pipeline) so half-clipping math is finite.
        mdf2 = mdf.copy()
        mdf2["period_end_min"] = mdf2["period_end_min"].fillna(match_end)

        home_periods = mdf2[mdf2["teamId"] == home_id]
        away_periods = mdf2[mdf2["teamId"] == away_id]

        for half in (1, 2):
            half_start = 0.0 if half == 1 else 45.0
            half_end = min(45.0, match_end) if half == 1 else match_end
            if half_end <= half_start:
                continue

            home_clipped = _clip_periods_to_half(home_periods, half_start, half_end)
            away_clipped = _clip_periods_to_half(away_periods, half_start, half_end)
            home_block = _build_team_half_block(
                "home", home_clipped, half_start, half_end, season,
                archetype_lookup, stats_lookup,
            )
            away_block = _build_team_half_block(
                "away", away_clipped, half_start, half_end, season,
                archetype_lookup, stats_lookup,
            )
            if home_block is None or away_block is None:
                continue

            # Count overflow (subs we had to drop past slot N_SUB_SLOTS).
            home_n_subs = (
                home_clipped.groupby("playerId")["_clip_start"].min() > half_start + _START_EPS
            ).sum()
            away_n_subs = (
                away_clipped.groupby("playerId")["_clip_start"].min() > half_start + _START_EPS
            ).sum()
            n_subs_overflow += max(0, int(home_n_subs) - N_SUB_SLOTS)
            n_subs_overflow += max(0, int(away_n_subs) - N_SUB_SLOTS)

            row: dict[str, Any] = {
                "match_id": int(match_id),
                "season": season,
                "half": half,
                "period_start_min": float(half_start),
                "period_end_min": float(half_end),
                "period_duration_min": float(half_end - half_start),
                "home_team_id": home_id,
                "away_team_id": away_id,
            }
            row.update(home_block)
            row.update(away_block)
            rows.append(row)

    if n_subs_overflow:
        print(
            f"[note] dropped {n_subs_overflow:,} sub events past slot {N_SUB_SLOTS} "
            "(NCAA-style high-rotation halves)."
        )

    df = pd.DataFrame(rows)
    print(f"       {len(df):,} (match, half) rows assembled")

    # Recompute half-level xG / goal features by feeding the half windows
    # back through the canonical match-state aggregator. Only carry the
    # per-side totals + per-30 rates over -- the home/away differentials
    # would become stale after the focal flip, and ``period_duration_min``
    # is already on ``df`` from the row dict.
    print(f"[feat] half-level xG / goals from {events_path}")
    state_in = df[[
        "match_id", "period_start_min", "period_end_min",
        "home_team_id", "away_team_id",
    ]].copy()
    state_out = add_match_state_features(state_in, events_path)
    side_cols = [
        "home_xg", "away_xg", "home_goals", "away_goals",
        "home_xg_per_30", "away_xg_per_30",
        "home_goals_per_30", "away_goals_per_30",
    ]
    df = pd.concat([df, state_out[side_cols].reset_index(drop=True)], axis=1)

    # Focal-perspective doubling: one row per (match, half, focal_team).
    print("[focal] reframing to (team1=focal, team2=opponent)")
    focal = _to_focal_perspective(df)
    print(f"       {len(focal):,} focal rows")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        focal.to_parquet(output_path, index=False)
        print(f"[done] {output_path}")
    return focal


def _to_focal_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Return two rows per (match, half): home-as-focal + away-as-focal."""
    # Build column-name swap functions for the two perspectives. Anything
    # starting with home_ or away_ flips to team1_/team2_ (or vice versa).
    def _rename(col: str, *, swap: bool) -> str:
        if col.startswith("home_"):
            tail = col[len("home_"):]
            return f"{'team2' if swap else 'team1'}_{tail}"
        if col.startswith("away_"):
            tail = col[len("away_"):]
            return f"{'team1' if swap else 'team2'}_{tail}"
        return col

    # Perspective 1: home = team1.
    p1 = df.rename(columns={c: _rename(c, swap=False) for c in df.columns}).copy()
    p1["team1_is_home"] = True
    # Perspective 2: away = team1.
    p2 = df.rename(columns={c: _rename(c, swap=True) for c in df.columns}).copy()
    p2["team1_is_home"] = False

    focal = pd.concat([p1, p2], ignore_index=True)

    # Recompute team1-minus-team2 differentials in the focal frame.
    focal["xg_team1_minus_team2"] = focal["team1_xg"] - focal["team2_xg"]
    focal["xg_team2_minus_team1"] = focal["team2_xg"] - focal["team1_xg"]
    focal["goals_team1_minus_team2"] = focal["team1_goals"] - focal["team2_goals"]
    focal["goals_team2_minus_team1"] = focal["team2_goals"] - focal["team1_goals"]
    scale = 30.0 / focal["period_duration_min"]
    focal["xg_team1_minus_team2_per_30"] = focal["xg_team1_minus_team2"] * scale
    focal["xg_team2_minus_team1_per_30"] = focal["xg_team2_minus_team1"] * scale
    focal["goals_team1_minus_team2_per_30"] = focal["goals_team1_minus_team2"] * scale
    focal["goals_team2_minus_team1_per_30"] = focal["goals_team2_minus_team1"] * scale
    return focal


def _split_train_test(
    focal: pd.DataFrame, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match-blocked split (same logic as ``scripts.build_train_test_split``)."""
    match_ids = focal["match_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled = match_ids.copy()
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_frac))
    test_ids = set(shuffled[:n_test].tolist())
    is_test = focal["match_id"].isin(test_ids)
    return (
        focal.loc[~is_test].reset_index(drop=True),
        focal.loc[is_test].reset_index(drop=True),
    )


def run(
    config_path: Path = Path("configs/config.yaml"),
    test_frac: float = 0.20,
    seed: int = 0,
) -> None:
    cfg = yaml.safe_load(config_path.read_text())
    raw_dir = Path(cfg["data_paths"]["raw"])
    processed_dir = Path(cfg["data_paths"]["processed"])
    archetype_map = Path(cfg["data_paths"]["archetype_artifacts"]) / "player_archetype_map.parquet"
    stats_path = processed_dir / "player_season_stats.parquet"

    out_path = processed_dir / "lineup_snapshots_half_subs.parquet"
    train_path = processed_dir / "train_snapshots_half_subs.parquet"
    test_path = processed_dir / "test_snapshots_half_subs.parquet"

    print(f"[load] {raw_dir / 'formations.parquet'}")
    formations = pd.read_parquet(raw_dir / "formations.parquet")
    print(f"       {len(formations):,} rows across {formations['matchId'].nunique()} matches")
    print(f"[load] {raw_dir / 'matches.parquet'}")
    matches = pd.read_parquet(
        raw_dir / "matches.parquet",
        columns=["wyId", "home_team", "away_team", "duration"],
    )

    focal = build_half_subs_snapshots(
        formations=formations,
        matches=matches,
        matches_path=raw_dir / "matches.parquet",
        seasons_path=raw_dir / "seasons.parquet",
        archetype_map_path=archetype_map,
        stats_path=stats_path,
        events_path=raw_dir / "all_events.parquet",
        output_path=out_path,
    )

    print("[split] match-blocked train/test split")
    filtered = filter_buildable_snapshots(focal)
    print(f"       buildable rows: {len(filtered):,}  /  total: {len(focal):,}")
    train, test = _split_train_test(filtered, test_frac, seed)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    print(
        f"       train: {len(train):>6,} / {train['match_id'].nunique():>5,} matches "
        f"({len(train)/max(len(filtered),1):.1%}) -> {train_path}"
    )
    print(
        f"       test:  {len(test):>6,} / {test['match_id'].nunique():>5,} matches "
        f"({len(test)/max(len(filtered),1):.1%}) -> {test_path}"
    )
    overlap = set(train["match_id"].unique()) & set(test["match_id"].unique())
    if overlap:
        raise RuntimeError(f"match_id leakage detected ({len(overlap)} overlapping)")
    print(f"       no match_id leakage across split")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.config, args.test_frac, args.seed)
