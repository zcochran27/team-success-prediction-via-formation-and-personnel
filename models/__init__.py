"""Model implementations for the 2 x 2 x 2 experimental matrix.

Tabular variants (XGBoost regressors over the focal-perspective snapshot
table at ``data/processed/lineup_snapshots.parquet``):

  * Model 1 -- :mod:`models.tab_pos_ego`        (raw position, focal-only)
  * Model 2 -- :mod:`models.tab_pos_matchup`    (raw position, +opponent)
  * Model 3 -- :mod:`models.tab_arch_ego`       (archetypes, focal-only)
  * Model 4 -- :mod:`models.tab_arch_matchup`   (archetypes, +opponent)

Graph variants (Models 5-8, GNN over per-team graphs) are scaffolded
under :mod:`models.gnn_positions` / :mod:`models.gnn_archetypes` and
will be filled out as the model library expands.

Each tabular module exposes ``build_features``, ``cross_validate``, and
``fit`` with the same signatures so they're interchangeable from the
shared evaluation harness.
"""
