# models/

Model code and cross-model helpers for the half-with-subs pipeline.
Every model in this folder trains on
[`data/processed/{train,test}_snapshots_half_subs.parquet`](../data/processed/)
and lands under [`artifacts/half_subs/`](../artifacts/half_subs/).

## Three families × 8 variants = 24 trained models

| Family | Trainer | Folder prefix | What it sees |
|---|---|---|---|
| **tab** | `scripts.train_tab_*` | `tab_<kind>_<opp>[_stats]` | 11 starter slots per team; position OR archetype; ±season stats. |
| **tab_subs** | `scripts.train_tab_subs_*` | `tab_subs_<kind>_<opp>[_stats]` | Same axes plus all 11 sub slots with `start_min`, `duration`, stats. |
| **gnn_subs** | `scripts.train_gnn_subs` (sweep: `train_gnn_subs_batch`) | `gnn_subs_<kind>_<mode>_coords[_stats]` | [`HalfSubsGNN`](gnn_subs.py): typed edges (intra-team / sub / time-overlap matchup), fully-connected intra-team with attention, sub-edge message passing. |

The 2×2×2 design axes (`kind`, `opp/mode`, `use_stats`) are shared
across families so any pair of runs lines up apples-to-apples.

## Files

| Module | Role |
|---|---|
| [`gnn_subs.py`](gnn_subs.py) | `HalfSubsGNN` class — `nn.Embedding` (position or k6-archetype vocab) + numeric/coord/stats projections, stacked `GATConv` over typed edges with `edge_attr`, attention pooling, antisymmetric head for paired graphs. |
| [`comparison_half_subs.py`](comparison_half_subs.py) | Cross-model helpers — `load_runs`, `build_summary_table`, `plot_metric_bars`, `plot_pred_vs_target`, `plot_residual_hists`. Drives the all-models comparison notebook. |
| [`training_diagnostics.py`](training_diagnostics.py) | GNN-only per-epoch helpers — `load_all_logs`, `best_epoch_table`, `plot_loss_curves`, `plot_metric_overlays`, `plot_grad_norms`, `plot_epoch_timing`. Consumes the `log.jsonl` written by `scripts.train_gnn_subs`. Tabular runs don't emit per-iteration logs, so curves only exist for the 8 GNN variants. |

## Notebooks

[`notebooks/`](notebooks/) holds the thin demos that import the helpers
above:

| Notebook | What it shows |
|---|---|
| [training_diagnostics.ipynb](notebooks/training_diagnostics.ipynb) | Per-epoch train/val curves, R² / Pearson / sign-acc overlays, grad-norm and wall-clock-per-epoch traces for the 8 GNN runs. |
| [all_models_comparison.ipynb](notebooks/all_models_comparison.ipynb) | 24-row leaderboard, family/use-stats pivots, metric bar chart, prediction-vs-target scatter grid, residual histograms. |

Earlier-iteration modules (joint-window snapshots, original `LineupGNN`,
separate `tab_*.py` modules) live under [`../legacy/`](../legacy/).
