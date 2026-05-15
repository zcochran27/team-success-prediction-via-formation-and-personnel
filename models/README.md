# models/

The 8 model variants of the 2 × 2 × 2 design — 4 tabular XGBoost models
plus 4 GNN variants served by a single `LineupGNN` class. The same
held-out test set drives all 8 (see [`comparison.py`](comparison.py)).

## Tabular models (Models 1–4)

| Module | Personnel | Opponent in view? |
|---|---|---|
| [tab_pos_ego.py](tab_pos_ego.py) | raw positions | no |
| [tab_pos_matchup.py](tab_pos_matchup.py) | raw positions | yes |
| [tab_arch_ego.py](tab_arch_ego.py) | archetypes | no |
| [tab_arch_matchup.py](tab_arch_matchup.py) | archetypes | yes |

Each exposes `build_features(snapshots) -> (X, y, groups)`,
`cross_validate(snapshots, n_splits=5)`, and `fit(snapshots)`. They
share an XGBoost configuration and the categorical-handling helper in
[`_tabular_common.py`](_tabular_common.py), so the difference between
any two is just which columns get selected and whether NaN archetypes
get coerced to the `"MISSING"` category.

## Graph models (Models 5–8)

All 4 graph variants live in [`gnn.py`](gnn.py) as a single
`LineupGNN(kind, mode)` class:

| Constructor call | Model |
|---|---|
| `LineupGNN(kind="position",  mode="single")` | Model 5 (raw positions, ego) |
| `LineupGNN(kind="archetype", mode="single")` | Model 6 (archetypes, ego) |
| `LineupGNN(kind="position",  mode="paired")` | Model 7 (raw positions, matchup) |
| `LineupGNN(kind="archetype", mode="paired")` | Model 8 (archetypes, matchup) |

Architecture: `nn.Embedding` (sized to the kind's vocab) → stacked
`GATConv` (`edge_dim=1` in paired mode so the same-team flag is
consumed) → mean-pool per team → shared MLP head. The paired forward
returns `head(team1_pool) − head(team2_pool)` so the prediction is
antisymmetric under team swap by construction.

Training and evaluation helpers (`train_one_epoch`, `evaluate`,
`predict`) live in the same module and branch on `model.mode`.

## Shared utilities

| Module | Role |
|---|---|
| [_tabular_common.py](_tabular_common.py) | `fit_xgb`, `cross_validate_xgb`, `coerce_categoricals`, `_sign_accuracy`. The default XGBoost params live here. |
| [comparison.py](comparison.py) | Cross-model helpers: `load_tabular_results(train_df, test_df)`, `load_gnn_results(artifacts_root)`, `build_summary_table`, `plot_metric_bars`, `plot_pred_vs_target_grid`, `plot_residual_hists`, `ablation_deltas`. |
| [training_plots.py](training_plots.py) | GNN-specific training diagnostics: `load_all_runs`, `plot_loss_curves`, `plot_metric_overlays`, `plot_pred_vs_target_grid`, `plot_residual_hists`, `plot_calibration`, `summary_table`. Consumes the `log.jsonl` / `val_preds.parquet` / `summary.json` written by [`../scripts/train_gnn.py`](../scripts/train_gnn.py). |

## Diagnostics + comparison

See [notebooks/](notebooks/):
- `tabular_models.ipynb` — Models 1–4 in isolation.
- `gnn_training.ipynb` — per-model training curves + best-epoch diagnostics for Models 5–8.
- `all_models_comparison.ipynb` — apples-to-apples head-to-head on the shared test set.
