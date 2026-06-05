# data/

All of `data/` is gitignored (only `.gitkeep` placeholders are tracked).
Populate it before running the pipelines.

## Expected layout

```
data/
├── raw/                                # Wyscout exports (you drop these in)
│   ├── all_events.parquet
│   ├── formations.parquet
│   ├── matches.parquet
│   ├── seasons.parquet
│   ├── leagues.parquet
│   ├── team_info.parquet
│   └── player_info.parquet
│
└── processed/                          # written by the pipelines
    ├── archetype_artifacts/            # archetypes/run_pipeline.py output
    │   ├── event_clusters/             # fitted (StandardScaler, KMeans) per (group, event type)
    │   ├── archetype_clusters/         # fitted KMeans per position group
    │   ├── player_features.parquet     # per-(player, season) feature vectors
    │   └── player_archetype_map.parquet  # final (player_id, season) -> archetype
    ├── player_season_stats.parquet     # per-(player, season) stat vectors (season_stats.py)
    ├── lineup_snapshots_half_subs.parquet   # formations output, all focal rows
    ├── train_snapshots_half_subs.parquet    # 80% by match_id
    └── test_snapshots_half_subs.parquet     # 20% held out
```

## Build order

```powershell
python -m archetypes.run_pipeline           # raw events -> archetype_artifacts/
# build player_season_stats.parquet via archetypes/season_stats.py
python -m formations.lineups_half_subs      # formations + archetypes + stats -> snapshots + split
```

The formation stage reads the archetype map and season stats, builds the
half-level snapshot table, and writes the match-blocked 80/20 train/test split
directly (no separate split step).

## Notes

- `data/raw/` is read-only after the Wyscout drop.
- `data/processed/` is regenerable; safe to wipe and rebuild.
