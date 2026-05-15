# archetypes/notebooks/

Diagnostic notebooks for the archetype pipeline. Each one consumes
artifacts written by [`../run_pipeline.py`](../run_pipeline.py) and
imports its plotting helpers from the corresponding `archetypes/*.py`
module — the notebooks themselves are intentionally thin.

| Notebook | Inspects |
|---|---|
| [event_clustering.ipynb](event_clustering.ipynb) | The fitted `StandardScaler → MiniBatchKMeans` event-cluster models. Re-applies them to a sample of events and renders 2D projections + cluster centroids per `(position group, event type)` pair. |
| [player_aggregation.ipynb](player_aggregation.ipynb) | The per-player feature table. Sanity-checks coverage, NaN structure, and event-type / subtype distributions across players. |
| [archetype_clustering.ipynb](archetype_clustering.ipynb) | The per-position-group archetype clustering. Elbow + silhouette diagnostics for `k ∈ [2, 10]` and a 2D PCA of the player vectors colored by archetype. |
| [archetype_profiles.ipynb](archetype_profiles.ipynb) | Archetype interpretability. Heatmap of mean feature vectors per archetype + radar of `pct_total_<event_type>` so each archetype gets a readable narrative. |

## Running

Open in Jupyter from the repo root after running
`python -m archetypes.run_pipeline`. Each notebook tolerates being run
standalone (it pulls helpers off the matching `.py` module so the
analysis is reusable outside Jupyter).
