"""Model implementations for the half-with-subs pipeline.

Three families are trained on
``data/processed/{train,test}_snapshots_half_subs.parquet`` and live
side-by-side under ``artifacts/half_subs/``:

* **tab** — XGBoost on the 8 starter-only feature configurations
  (position vs archetype × ego vs matchup × ±stats). The training entry
  points are in ``scripts.train_tab_*``; this package only carries the
  cross-model comparison helpers.
* **tab_subs** — XGBoost on the same 8 axes but with all 11 sub slots
  (position/archetype/start_min/duration/stats) joined into the feature
  matrix. Trained by ``scripts.train_tab_subs_*``.
* **gnn_subs** — :class:`~models.gnn_subs.HalfSubsGNN`, a typed-edge GAT
  with fully-connected intra-team edges, sub-edge message passing, and
  time-overlap matchup edges (paired mode only). Trained by
  ``scripts.train_gnn_subs``.

Cross-model helpers:

* :mod:`models.comparison_half_subs` — loader + leaderboard + plots over
  every ``artifacts/half_subs/*`` run, used by
  ``notebooks/all_models_comparison.ipynb``.
* :mod:`models.training_diagnostics` — per-epoch curves + best-epoch
  table from the GNN ``log.jsonl`` files, used by
  ``notebooks/training_diagnostics.ipynb``.

Earlier-iteration modules (joint-window snapshots, the original
``LineupGNN``, separate ``tab_*.py`` modules) were moved to ``legacy/``
when the repo refocused on this pipeline.
"""
