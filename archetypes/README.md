# archetypes/

Builds the data-driven player taxonomy used by the archetype model variants.
The taxonomy is computed per (player, season), so the same player can carry
different archetype labels in different seasons. The final output is
`data/processed/archetype_artifacts/player_archetype_map.parquet`, which the
formations stage joins onto every snapshot row.

## Pipeline (run in order)

- [position_groups.py](position_groups.py): map each Wyscout position label to
  one of 8 groups (CD, LWD, RWD, CM, CF, LWP, RWP, GK) and derive each row's
  logical event type (pass, offensive_duel, defensive_duel, aerial_duel, shot,
  goalkeeper_exit).
- [event_clustering.py](event_clustering.py): for each (position group, event
  type) pair, fit a StandardScaler + MiniBatchKMeans pipeline over the event
  features to find distinct subtypes (short vs long passes, and so on).
- [player_aggregation.py](player_aggregation.py): for each (player_id, season),
  turn the event cluster labels into a feature vector (event-type mix plus the
  subtype distribution per event type). Players below `min_events_per_player`
  drop out.
- [archetype_clustering.py](archetype_clustering.py): cluster the
  per-player-season vectors with KMeans within each position group. Each
  cluster is one archetype (CD-0, CM-1, and so on).
- [assign_archetypes.py](assign_archetypes.py): apply the fitted models to
  every player-season to produce `player_archetype_map.parquet`.
- [season_stats.py](season_stats.py): build the per-(player, season) 10-dim
  stat vector used as a separate feature block by the models.
- [run_pipeline.py](run_pipeline.py): runs the clustering stages end to end.

## Running

```powershell
python -m archetypes.run_pipeline
```

Reads `data/raw/all_events.parquet` and writes
`data/processed/archetype_artifacts/` (event-cluster models, the player
feature table, archetype cluster models, and the final
`player_archetype_map.parquet`).

## Notebooks

Each stage has a notebook under [notebooks/](notebooks/) for sanity checks
(elbow plots, silhouette scores, archetype profiles).
