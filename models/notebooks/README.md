# models/notebooks/

Thin demos for the half-with-subs model pipeline. The plotting and loading
helpers live in [`../training_diagnostics.py`](../training_diagnostics.py) and
[`../comparison_half_subs.py`](../comparison_half_subs.py), so these notebooks
just load, call, and render.

- [training_diagnostics.ipynb](training_diagnostics.ipynb): per-epoch curves
  (train and val MSE, R2, Pearson, sign accuracy), the gradient-norm
  trajectory, wall-clock per epoch, and the best-epoch table for the 8
  [`HalfSubsGNN`](../gnn_subs.py) variants. Reads
  `artifacts/half_subs/gnn_subs_*/log.jsonl`. Tabular runs don't emit per-epoch
  logs, so curves only exist for the GNNs.
- [all_models_comparison.ipynb](all_models_comparison.ipynb): the leaderboard
  (8 tab, 8 tab_subs, 8 gnn_subs), the family and stats pivots, metric bars,
  the predictions vs targets scatter grid, and residual histograms, all on the
  shared held-out test set. Reads every `artifacts/half_subs/*/summary.json`
  plus its `test_preds.parquet` or `val_preds.parquet`.
