# graphs/notebooks/

Visualization notebooks for the formation-template graphs and the
paired matchup graphs. Both notebooks call helpers from
[`../visualize.py`](../visualize.py) so the rendering logic stays in a
reusable module.

| Notebook | Inspects |
|---|---|
| [formation_graphs.ipynb](formation_graphs.ipynb) | The single-team formation templates from [`../templates.py`](../templates.py). Shows the 17-rule edge set rendered on a pitch for each formation in `FORMATION_TEMPLATES`. |
| [paired_snapshot_graphs.ipynb](paired_snapshot_graphs.ipynb) | The 22-node paired graph used by the matchup-mode GNN ([`features.build_graphs.build_snapshot_graph`](../../features/build_graphs.py) with `mode="paired"`). Hand-picked snapshots from `data/processed/lineup_snapshots.parquet` — Team 1 (blue) at template coords, Team 2 (red) rotated 180° about the pitch center, intra-team edges per team color, inter-team matchup edges in orange. |
