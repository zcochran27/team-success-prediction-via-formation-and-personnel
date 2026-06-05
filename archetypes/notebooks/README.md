# archetypes/notebooks/

Diagnostic notebooks for the archetype pipeline. Each one consumes artifacts
written by [`../run_pipeline.py`](../run_pipeline.py) and imports its plotting
helpers from the matching `archetypes/*.py` module, so the notebooks stay thin.

- [event_clustering.ipynb](event_clustering.ipynb): the fitted event-cluster
  models. Re-applies them to a sample of events and renders 2D projections and
  cluster centroids per (position group, event type) pair.
- [player_aggregation.ipynb](player_aggregation.ipynb): the per-player feature
  table. Checks coverage, NaN structure, and the event-type and subtype
  distributions across players.
- [archetype_clustering.ipynb](archetype_clustering.ipynb): the
  per-position-group archetype clustering. Elbow and silhouette diagnostics and
  a 2D PCA of the player vectors colored by archetype.
- [archetype_profiles.ipynb](archetype_profiles.ipynb): archetype
  interpretability. A heatmap of mean feature vectors per archetype and a radar
  of pct_total_<event_type> so each archetype gets a readable description.

## Running

Open in Jupyter from the repo root after running
`python -m archetypes.run_pipeline`.
