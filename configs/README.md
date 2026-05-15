# configs/

Project-wide configuration. Currently a single YAML file:

| File | Contents |
|---|---|
| [config.yaml](config.yaml) | Data-path layout, archetype-pipeline hyperparameters (per-group cluster counts, event-type vocabularies, `min_events_per_player`), formation-snapshot sub-cluster collapse window, evaluation defaults, and baseline GNN / XGBoost hyperparameter blocks. |

## Who reads what

- The archetype pipeline ([`archetypes/run_pipeline.py`](../archetypes/run_pipeline.py)) reads `event_clustering` and `archetype_clustering`.
- The formation pipeline ([`formations/run.py`](../formations/run.py)) reads `formations.sub_cluster_window_min`.
- The GNN trainer ([`scripts/train_gnn.py`](../scripts/train_gnn.py)) and tabular models accept hyperparameters via CLI flags; the `model` and `baseline` blocks in `config.yaml` are reference defaults rather than runtime inputs.

If you change any clustering hyperparameter (especially the per-group `k`
values), re-run `python -m archetypes.run_pipeline` to refit the affected
models and rebuild `player_archetype_map.parquet`.
