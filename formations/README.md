# formations/

Stage 2 of the data pipeline. Pivots the raw formation log into the wide
lineup-snapshot training table at
`data/processed/lineup_snapshots.parquet`. Reads the archetype map
written by [`archetypes/`](../archetypes/) and joins it in.

## Pipeline stages (run in order)

| Module | Role |
|---|---|
| [lineups.py](lineups.py) | Pivot `data/raw/formations.parquet` into one row per `(match, joint-stable window)` — a contiguous interval during which *neither* team changes shape. Assigns each of the 22 players to a numbered slot 1..11 per team using the modern English shirt-number convention. |
| [match_state.py](match_state.py) | For each window, sum every shot's `shot_xg` and count goals per team. Attaches raw / differential / per-30 versions of xG and goals to the snapshot row. End-of-match windows are clamped using `matches.duration` (Regular → 95′, ExtraTime → 125′). |
| [cluster_collapse.py](cluster_collapse.py) | Merge consecutive sub-induced sub-windows (no formation change, only personnel) into single segments. Formation changes are hard barriers. Recorded personnel is the post-last-sub lineup of each segment. Drops the dataset from 174,694 focal rows to ~79,782 with the target std reduced by 34 %. |
| [player_archetypes.py](player_archetypes.py) | Tag each row with its calendar `season` and join the per-season archetype label onto each of the 22 player slots. NaN archetypes are preserved (informative missingness; see [`features/build_graphs.py`](../features/build_graphs.py)). |
| [run.py](run.py) | Orchestrates all of the above end-to-end. |

## Running

```powershell
python -m formations.run
```

Reads `data/raw/formations.parquet`, `all_events.parquet`,
`matches.parquet`, `seasons.parquet`, and
`data/processed/archetype_artifacts/player_archetype_map.parquet`.
Writes `data/processed/lineup_snapshots.parquet` (~79.8k focal-perspective
rows; ~73.6k of those are graph-buildable per
[`features.snapshots.filter_buildable_snapshots`](../features/snapshots.py)).
