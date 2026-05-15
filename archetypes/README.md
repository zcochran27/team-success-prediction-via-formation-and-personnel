# archetypes/

Builds the data-driven player taxonomy used by the archetype variants of
every downstream model. The taxonomy is computed **per (player, season)**,
so the same player can carry different archetype labels in different
seasons. The final output is the lookup table
`data/processed/archetype_artifacts/player_archetype_map.parquet`, which
`formations/` joins onto every snapshot row.

## Pipeline stages (run in order)

| Module | Role |
|---|---|
| [position_groups.py](position_groups.py) | Map each Wyscout position label to one of 8 groups (`CD`, `LWD`, `RWD`, `CM`, `CF`, `LWP`, `RWP`, `GK`) and derive each row's logical event type (`pass`, `offensive_duel`, `defensive_duel`, `aerial_duel`, `shot`, `goalkeeper_exit`). |
| [event_clustering.py](event_clustering.py) | For each `(position group, event type)` pair, fit a `StandardScaler -> MiniBatchKMeans` pipeline over the event-instance features to recover qualitatively distinct subtypes (short vs long passes, progressive vs lateral duels, …). |
| [player_aggregation.py](player_aggregation.py) | For each `(player_id, season)`, look up the cluster label of every event the player produced and reduce them to a feature vector: proportional event-type mix + intra-event subtype distribution per event type. Players below `min_events_per_player` drop out. |
| [archetype_clustering.py](archetype_clustering.py) | Cluster the per-player-season vectors with KMeans **within each position group**. Each resulting cluster is one archetype (`CD-0`, `CM-1`, …). |
| [assign_archetypes.py](assign_archetypes.py) | Apply the fitted models back to every player-season to produce the final `player_archetype_map.parquet`. |
| [run_pipeline.py](run_pipeline.py) | Orchestrates the four stages above end-to-end. |

## Running

```powershell
python -m archetypes.run_pipeline
```

Reads `data/raw/all_events.parquet`, writes
`data/processed/archetype_artifacts/` (event-cluster models, player feature
table, archetype cluster models, and the final `player_archetype_map.parquet`).

## Diagnostics

Each stage has a paired notebook under [notebooks/](notebooks/) for sanity
checks (elbow plots, silhouette scores, archetype profiles).
