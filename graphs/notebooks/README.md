# graphs/notebooks/

Visualization notebooks for the half-with-subs graph builder. Drawing
helpers live in [`../visualize_half_subs.py`](../visualize_half_subs.py)
so the notebooks stay thin -- they load a row, build, and render.

| Notebook | Inspects |
|---|---|
| [half_subs_graph_demo.ipynb](half_subs_graph_demo.ipynb) | One real row from [`data/processed/lineup_snapshots_half_subs.parquet`](../../data/processed/) → [`features.build_graphs_subs.build_half_subs_graph`](../../features/build_graphs_subs.py). Prints the node + edge tables, then renders both single-team graphs side by side and the joint paired graph with matchup edges weighted by on-pitch time overlap. |

Older notebooks describing the prior (joint-window snapshot) builder
were moved to [`../../legacy/graphs/notebooks/`](../../legacy/graphs/notebooks/)
when the repo refocused on the half-with-subs pipeline.
