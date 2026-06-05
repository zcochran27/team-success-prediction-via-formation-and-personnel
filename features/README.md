# features/

Feature engineering and dataset assembly for the graph models. The tabular
models read their features directly off the snapshot table.

## Files

- [vocab.py](vocab.py): the position and archetype label vocabularies, the
  stat-vector dimensions, and `vocab_size()` for the embedding tables.
- [snapshots.py](snapshots.py): `filter_buildable_snapshots(df)`, the filter
  applied by both the GNN dataset and the tabular comparison. Drops rows whose
  formation isn't in [`graphs.templates.FORMATION_TEMPLATES`](../graphs/templates.py)
  or that have a null position label. Archetypes are not filtered (NaN maps to
  `"MISSING"` downstream).
- [build_graphs_subs.py](build_graphs_subs.py): `build_half_subs_graph(row, mode, kind, ...)`
  builds the per-half graph(s). Single mode returns a (team1, team2) pair;
  paired mode returns one joint graph with formation, sub, and time-overlap
  matchup edges.
- [dataset_subs.py](dataset_subs.py): `HalfSubsDataset`, a PyG dataset over a
  half-subs snapshot parquet, plus the collate functions `paired_collate`,
  `joint_collate`, and the dispatcher `collate_for(mode)`.

These are consumed by the GNN training script and by
[`models/comparison_half_subs.py`](../models/comparison_half_subs.py), which
calls `filter_buildable_snapshots` so the tabular side trains on exactly the
rows the GNN dataset consumes.
