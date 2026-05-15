"""Model implementations for the 2 x 2 x 2 experimental matrix.

Tabular variants (XGBoost over the focal-perspective snapshot table at
``data/processed/lineup_snapshots.parquet``):

  * Model 1 -- :mod:`models.tab_pos_ego`        (raw position, focal-only)
  * Model 2 -- :mod:`models.tab_pos_matchup`    (raw position, +opponent)
  * Model 3 -- :mod:`models.tab_arch_ego`       (archetypes, focal-only)
  * Model 4 -- :mod:`models.tab_arch_matchup`   (archetypes, +opponent)

Graph variants (Models 5-8) are all served by the single
:mod:`models.gnn` module via :class:`~models.gnn.LineupGNN`. The 2 x 2
graph grid is configured at construction time:

  * Model 5: ``LineupGNN(kind="position",  mode="single")``
  * Model 6: ``LineupGNN(kind="archetype", mode="single")``
  * Model 7: ``LineupGNN(kind="position",  mode="paired")``
  * Model 8: ``LineupGNN(kind="archetype", mode="paired")``

Head-to-head plotting + summary tables live in :mod:`models.comparison`;
GNN-specific training plots live in :mod:`models.training_plots`.
"""
