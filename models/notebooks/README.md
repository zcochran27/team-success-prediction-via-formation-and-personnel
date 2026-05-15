# models/notebooks/

Result-rendering notebooks for the 8-way model comparison. All three are
deliberately thin — plotting and aggregation live in
[`../comparison.py`](../comparison.py),
[`../training_plots.py`](../training_plots.py), and
[`../_tabular_common.py`](../_tabular_common.py).

| Notebook | Covers |
|---|---|
| [tabular_models.ipynb](tabular_models.ipynb) | Models 1–4 in isolation. Builds each model's `(X, y, groups)`, runs 5-fold `GroupKFold` CV blocked by `match_id`, and renders a metric bar chart + pairwise ablation deltas. |
| [gnn_training.ipynb](gnn_training.ipynb) | Models 5–8 training diagnostics. Reads `artifacts/gnn_<kind>_<mode>/` for whichever runs exist and renders: summary table, train / test loss curves per model, multi-model metric overlays (MSE / MAE / Pearson / sign acc), optimization-health curves (grad norm + lr), best-epoch predictions-vs-targets scatter, residual histograms, and a reliability plot. |
| [all_models_comparison.ipynb](all_models_comparison.ipynb) | The head-to-head. Loads the shared `data/processed/{train,test}_snapshots.parquet`, fits tabular on train + predicts on test, reads the GNN best-epoch test predictions from `artifacts/`, and produces a unified summary table, metric bar chart, predictions-vs-targets grid, residual grid, and 3-axis ablation deltas (`tabular → graph`, `position → archetype`, `ego → matchup`). |

## Re-running

```powershell
python -m archetypes.run_pipeline                # if archetype map is missing
python -m formations.run                         # if lineup_snapshots.parquet is missing
python -m scripts.build_train_test_split         # if train/test parquets are missing
python -m scripts.train_gnn --kind position  --mode single --epochs 30
python -m scripts.train_gnn --kind archetype --mode single --epochs 30
python -m scripts.train_gnn --kind position  --mode paired --epochs 30
python -m scripts.train_gnn --kind archetype --mode paired --epochs 30
```

Then open any of the three notebooks and run all. Models with no artifacts
are silently skipped.
