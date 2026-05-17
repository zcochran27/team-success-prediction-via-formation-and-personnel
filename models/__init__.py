"""Model implementations for the 2 x 2 x 2 experimental matrix.

Tabular variants (XGBoost over the focal-perspective snapshot table at
``data/processed/lineup_snapshots.parquet``):

  * Model 1 -- :mod:`models.tab_pos_ego`        (raw position, focal-only)
  * Model 2 -- :mod:`models.tab_pos_matchup`    (raw position, +opponent)
  * Model 3 -- :mod:`models.tab_arch_ego`       (archetypes, focal-only)
  * Model 4 -- :mod:`models.tab_arch_matchup`   (archetypes, +opponent)

Each of the four tabular variants has a ``+stats`` companion that
appends the per-(player, season) 10-D behavioral vector (see
:mod:`archetypes.season_stats`) to every player slot:

  * :mod:`models.tab_pos_ego_stats`
  * :mod:`models.tab_pos_matchup_stats`
  * :mod:`models.tab_arch_ego_stats`
  * :mod:`models.tab_arch_matchup_stats`

The shared join lives in :mod:`models._tabular_stats`; the column block
is ``team{side}_p{slot}_<stat>`` -- 110 columns for ego variants, 220
for matchup variants.

Graph variants are all served by the single :mod:`models.gnn` module
via :class:`~models.gnn.LineupGNN`, configured at construction time:

  * ``LineupGNN(kind="position",  mode="single")``  -- pos · ego
  * ``LineupGNN(kind="archetype", mode="single")``  -- arch · ego
  * ``LineupGNN(kind="position",  mode="paired")``  -- pos · matchup
  * ``LineupGNN(kind="archetype", mode="paired")``  -- arch · matchup

Each of the four graph variants additionally supports ``use_coords``
(template ``(x, y)`` node + edge features) and ``use_stats`` (the same
10-D behavioral vector mixed into the node embedding and 4 stat-diff
edge features) flags -- yielding the 12 graph variants tracked by
:mod:`models.comparison`.

Head-to-head plotting + summary tables live in :mod:`models.comparison`;
GNN-specific training plots live in :mod:`models.training_plots`.
"""
