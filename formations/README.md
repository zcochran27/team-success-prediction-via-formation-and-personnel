# formations/

Turns the raw formation log into the half-level lineup-snapshot table at
`data/processed/lineup_snapshots_half_subs.parquet`. Reads the archetype map
from [`archetypes/`](../archetypes/) and the season stats, and joins them onto
each player slot.

## Files

- [match_state.py](match_state.py): for each window, sum every shot's xG and
  count goals per team, then attach raw, differential, and per-30 xG and goal
  columns. End-of-match windows are clamped using `matches.duration`
  (Regular to 95 min, ExtraTime to 125 min).
- [lineups_half_subs.py](lineups_half_subs.py): build one snapshot per
  (match, half). Each side has 11 starter slots (in formation-template order)
  and 11 sub slots (in chronological entry order), each with player_id,
  position, archetype, season stats, and duration. Recomputes half-level
  match-state features through `match_state.py`, reframes to focal perspective
  (team1 vs team2), and writes the match-blocked train/test split.

## Running

```powershell
python -m formations.lineups_half_subs
```

Reads `data/raw/formations.parquet`, `all_events.parquet`, `matches.parquet`,
`seasons.parquet`, and the archetype map and season stats from
`data/processed/`. Writes `lineup_snapshots_half_subs.parquet` plus
`train_snapshots_half_subs.parquet` and `test_snapshots_half_subs.parquet`.
