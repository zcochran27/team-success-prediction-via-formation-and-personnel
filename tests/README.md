# tests/

Unit + smoke tests for the graph pipeline. The archetype and formation
pipelines are exercised end-to-end by `python -m archetypes.run_pipeline`
and `python -m formations.run`; tests here cover the parts most prone to
silent breakage (vocab drift, edge-set regressions, dataset shape, model
forward).

## Files

| File | Covers |
|---|---|
| [test_build_graphs.py](test_build_graphs.py) | The graph builder. Edge index shape and undirectedness, node-feature vocab + `MISSING_ARCHETYPE` round-trip, single-mode and paired-mode `build_snapshot_graph` invariants (incl. that paired + archetype demands an explicit position label list for the matchup edges). |
| [test_dataset.py](test_dataset.py) | `LineupSnapshotDataset` + the GNN training loop. Buildable-row count matches `filter_buildable_snapshots`, single and paired mode item shapes, collate batching, end-to-end training step actually changes loss, NaN archetypes resolve to the `MISSING` embedding index. |
| [test_matchups.py](test_matchups.py) | Inter-team matchup edge rules from [`graphs/matchups.py`](../graphs/matchups.py). Zone bucketing, lane-mirror correctness, GK exclusion, side-aware fullback ↔ opposing-striker refinement, back-three winger collapse. |

## Running

```powershell
pytest tests/                    # full suite, ~10s
pytest tests/test_matchups.py    # one file
pytest -k missing_archetype      # one test by substring
```

The dataset-level tests require `data/processed/lineup_snapshots.parquet`
to exist (built by `python -m formations.run`); they skip cleanly when
the parquet is missing.
