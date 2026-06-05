# models/

Model code and cross-model helpers for the half-with-subs pipeline. Every
model trains on
[`data/processed/{train,test}_snapshots_half_subs.parquet`](../data/processed/)
and lands under `artifacts/half_subs/`.

## Families

Three families share the same 2x2x2 design axes (kind, opponent/mode,
use_stats), so any pair of runs lines up directly:

- tab: XGBoost on the 11 starter slots per team; position or archetype; with or
  without season stats. Folder prefix `tab_<kind>_<opp>[_stats]`.
- tab_subs: the same axes plus all 11 sub slots (start_min, duration, stats).
  Folder prefix `tab_subs_<kind>_<opp>[_stats]`.
- gnn_subs: [`HalfSubsGNN`](gnn_subs.py) with typed edges (intra-team, sub,
  time-overlap matchup), fully connected intra-team attention, and sub-edge
  message passing. Folder prefix `gnn_subs_<kind>_<mode>_coords[_stats]`.

The GNN runs are trained by `scripts/train_gnn_subs.py` (one run) and
`scripts/train_gnn_subs_batch.py` (the full sweep). The tabular models are fit
inside the comparison notebook.

## Files

- [gnn_subs.py](gnn_subs.py): the `HalfSubsGNN` class, an embedding (position or
  archetype vocab) plus numeric/coord/stats projections, stacked GATConv over
  typed edges, attention pooling, and an antisymmetric head for paired graphs.
- [comparison_half_subs.py](comparison_half_subs.py): cross-model helpers
  (`load_runs`, `build_summary_table`, `plot_metric_bars`,
  `plot_pred_vs_target`, `plot_residual_hists`).
- [training_diagnostics.py](training_diagnostics.py): GNN per-epoch helpers
  (`load_all_logs`, `best_epoch_table`, `plot_loss_curves`,
  `plot_metric_overlays`, `plot_grad_norms`, `plot_epoch_timing`) over the
  `log.jsonl` files. Tabular runs don't emit per-epoch logs, so curves only
  exist for the 8 GNN variants.

## Notebooks

[notebooks/](notebooks/) holds thin demos that import the helpers above:

- [all_models_comparison.ipynb](notebooks/all_models_comparison.ipynb): the
  leaderboard, family and use-stats pivots, metric bar chart, predictions vs
  targets scatter grid, and residual histograms.
- [training_diagnostics.ipynb](notebooks/training_diagnostics.ipynb): per-epoch
  train/val curves, metric overlays, and grad-norm and timing traces for the 8
  GNN runs.
