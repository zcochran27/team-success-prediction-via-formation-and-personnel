# models/notebooks/

Thin demos for the half-with-subs model pipeline. Plotting and loading
helpers live in [`../training_diagnostics.py`](../training_diagnostics.py)
and [`../comparison_half_subs.py`](../comparison_half_subs.py) so these
notebooks stay short — they load, call, render.

| Notebook | Inspects | Reads from |
|---|---|---|
| [training_diagnostics.ipynb](training_diagnostics.ipynb) | Per-epoch curves (train + val MSE, R² / Pearson / sign-acc), gradient-norm trajectory, wall-clock per epoch, best-epoch table — for the 8 [`HalfSubsGNN`](../gnn_subs.py) variants. Tabular runs don't emit per-iteration logs, so curves only exist for GNNs. | [`../../artifacts/half_subs/gnn_subs_*/log.jsonl`](../../artifacts/half_subs/) |
| [all_models_comparison.ipynb](all_models_comparison.ipynb) | 24-row leaderboard (8 `tab` + 8 `tab_subs` + 8 `gnn_subs`), family × stats pivots, metric bars, prediction-vs-target scatter grid, residual histograms — all on the shared held-out test set. | every [`../../artifacts/half_subs/*/summary.json`](../../artifacts/half_subs/) + `test_preds.parquet` / `val_preds.parquet` |

Older notebooks (joint-window snapshots, separate per-model training
notebooks) were moved to [`../../legacy/models/notebooks/`](../../legacy/models/notebooks/)
when the repo refocused on the half-with-subs pipeline.
