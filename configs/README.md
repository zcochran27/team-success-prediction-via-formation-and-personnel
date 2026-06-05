# configs/

Project configuration in a single YAML file.

- [config.yaml](config.yaml): the data-path layout, the archetype-pipeline
  hyperparameters (per-group cluster counts, event-type vocabularies,
  `min_events_per_player`), and reference hyperparameter blocks.

## Who reads what

- The archetype pipeline
  ([`archetypes/run_pipeline.py`](../archetypes/run_pipeline.py)) reads the
  `event_clustering` and `archetype_clustering` blocks.
- The formation pipeline
  ([`formations/lineups_half_subs.py`](../formations/lineups_half_subs.py))
  reads the data paths.
- The GNN trainer takes its hyperparameters from CLI flags; the `model` and
  `baseline` blocks are reference defaults, not runtime inputs.

If you change a clustering hyperparameter (especially the per-group `k`
values), re-run `python -m archetypes.run_pipeline` to refit the affected
models and rebuild `player_archetype_map.parquet`.
