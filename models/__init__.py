"""Model implementations for the half-with-subs pipeline.

Three families are trained on
data/processed/{train,test}_snapshots_half_subs.parquet and live side by
side under artifacts/half_subs/:

- tab: XGBoost on the 8 starter-only feature configs (position vs archetype,
  ego vs matchup, with or without stats).
- tab_subs: XGBoost on the same axes but with all 11 sub slots
  (position/archetype/start_min/duration/stats) joined into the feature
  matrix.
- gnn_subs: HalfSubsGNN (gnn_subs.py), a typed-edge GAT with fully connected
  intra-team edges, sub-edge message passing, and time-overlap matchup edges
  (paired mode only).

Cross-model helpers:
- comparison_half_subs.py: loader, leaderboard, and plots over every
  artifacts/half_subs/* run.
- training_diagnostics.py: per-epoch curves and the best-epoch table from the
  GNN log.jsonl files.
"""
