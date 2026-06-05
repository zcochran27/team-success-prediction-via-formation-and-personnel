# graphs/notebooks/

Visualization notebook for the half-with-subs graph builder. The drawing
helpers live in [`../visualize_half_subs.py`](../visualize_half_subs.py) so the
notebook stays thin: it loads a row, builds the graph, and renders.

- [half_subs_graph_demo.ipynb](half_subs_graph_demo.ipynb): takes one real row
  from `data/processed/lineup_snapshots_half_subs.parquet`, builds the graph
  with `build_half_subs_graph`, prints the node and edge tables, then renders
  both single-team graphs side by side and the joint paired graph with matchup
  edges weighted by on-pitch time overlap.
