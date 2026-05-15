# features/

Feature engineering and dataset assembly for the graph models. The
tabular models read their features directly off the snapshot table — see
[`../models/`](../models/) for those.

## Files

| Module | Role |
|---|---|
| [snapshots.py](snapshots.py) | `filter_buildable_snapshots(df)` — the canonical filter applied by *both* the GNN dataset and the tabular comparison. Drops rows whose formation isn't in [`graphs.templates.FORMATION_TEMPLATES`](../graphs/templates.py) or that have any null position label. Archetypes are deliberately *not* filtered (NaN gets mapped to `"MISSING"` downstream). |
| [build_graphs.py](build_graphs.py) | `build_snapshot_graph(mode, kind, …)` — single public entry point that produces either an 11-node single-team graph (Models 5 & 6) or a 22-node paired graph with same-team / matchup edges (Models 7 & 8). Also exports `POSITION_VOCAB`, `ARCHETYPE_VOCAB` (incl. `MISSING_ARCHETYPE`), and `vocab_size()` for the GNN embedding tables. |
| [dataset.py](dataset.py) | `LineupSnapshotDataset` — lazy PyG dataset wrapping a snapshot parquet (typically `train_snapshots.parquet` or `test_snapshots.parquet`). Arguments: `kind ∈ {position, archetype}` and `mode ∈ {single, paired}`. Aligns each lineup to template slot order via [`graphs.alignment.align_lineup_to_template`](../graphs/alignment.py) before handing the labels to the graph builder. Includes `paired_collate`, `snapshot_collate`, and the dispatcher `collate_for(mode)`. |

## Where it's used

- [`scripts/train_gnn.py`](../scripts/train_gnn.py) instantiates one
  `LineupSnapshotDataset` per parquet (train + test) and consumes
  `build_snapshot_graph` indirectly through it.
- [`models/comparison.py`](../models/comparison.py) calls
  `filter_buildable_snapshots` so the tabular side trains on exactly the
  rows the GNN dataset consumes.

## What's not here

- The original `features/build_tabular.py` (a stub) was removed in cleanup. Tabular features are produced by each model module under [`../models/`](../models/).
