# data/

All of `data/` is gitignored (only `.gitkeep` placeholders are tracked).
Populate it yourself before running the pipelines.

## Expected layout

```
data/
├── raw/                                # Wyscout exports (you drop these in)
│   ├── all_events.parquet              # per-event log
│   ├── formations.parquet              # per-player formation log
│   ├── matches.parquet                 # match metadata + duration flag
│   ├── seasons.parquet                 # season metadata
│   ├── leagues.parquet                 # league metadata
│   ├── team_info.parquet               # team metadata
│   └── player_info.parquet             # player metadata
│
├── processed/                          # written by the pipelines
│   ├── archetype_artifacts/            # archetypes/run_pipeline.py output
│   │   ├── event_clusters/             # fitted (StandardScaler, KMeans) per (group, event-type)
│   │   ├── archetype_clusters/         # fitted KMeans per position group
│   │   ├── player_features.parquet     # per-(player, season) feature vectors
│   │   └── player_archetype_map.parquet  # final (player_id, season) -> archetype
│   │
│   ├── lineup_snapshots.parquet        # formations/run.py output - the base training table
│   ├── train_snapshots.parquet         # scripts/build_train_test_split.py output (80% by match_id)
│   └── test_snapshots.parquet          # scripts/build_train_test_split.py output (20% held out)
│
└── team_seasons/                       # legacy directory from the per-season formulation; unused
```

## Build order

```powershell
python -m archetypes.run_pipeline       # raw events  -> archetype_artifacts/
python -m formations.run                # formations + archetypes -> lineup_snapshots.parquet
python -m scripts.build_train_test_split  # lineup_snapshots -> train_/test_snapshots.parquet
```

Stage 2 reads Stage 1's output; the split script reads Stage 2's output.
The full snapshot table is ~79k rows raw; the filter applied by
[`features.snapshots.filter_buildable_snapshots`](../features/snapshots.py)
brings it down to ~73.6k buildable rows, then the match-blocked 80/20
split lands at ~59.1k train / ~14.5k test.

## Notes

- `data/raw/` is read-only after the Wyscout drop — never overwrite.
- `data/processed/` is regenerable; safe to wipe and rebuild.
- `data/team_seasons/` is vestigial from the original per-season formulation; left in place for the `.gitkeep` but produces no artifacts in the current pipeline.
