# D1 Soccer Prediction

Predict NCAA Division I college soccer team performance — **win rate** and
**points per game** — from a team's formation and player personnel, using
Wyscout event-level tracking data.

The unit of observation is a **team-season**: one record per team per season.

---

## Research Design

The project answers two questions simultaneously:

1. Does representing team structure as a **graph** (vs. a flat feature vector)
   improve prediction?
2. Do richer **player archetypes** (vs. raw positional labels) improve
   prediction?

This produces a 2×2 model comparison:

|                   | Tabular  | Graph   |
|-------------------|----------|---------|
| **Raw Positions** | Model 1  | Model 3 |
| **Archetypes**    | Model 2  | Model 4 |

Comparisons:
- Models 1 vs. 2 and 3 vs. 4 → marginal value of archetypes
- Models 1 vs. 3 and 2 vs. 4 → marginal value of graph structure
- Models 1 vs. 4 → combined gain

Each model is trained independently for each of the two targets (win rate and
points per game), producing a final 8-way results table.

---

## Player Archetype Pipeline

Before any modeling, a data-driven archetype taxonomy is constructed. This is
a standalone preprocessing phase that runs once and produces artifacts
consumed by all downstream feature builders.

1. **Position Group Definition** — partition players into 5 groups:
   Center Defenders, Wide Defenders, Center Midfielders, Center Forwards,
   Wide Players (wide mids + wingers).
2. **Event Filtering** — for each position group, filter the event log to
   passes, dribbles, tackles, and shots produced by players in that group.
3. **Intra-Event Clustering** — within each event type, cluster individual
   event instances to identify qualitatively distinct subtypes (e.g. short vs.
   long passes, progressive vs. lateral dribbles).
4. **Player-Level Aggregation** — aggregate each player's event history into a
   feature vector: for each event type, record `(% of total actions,
   % in cluster 1, % in cluster 2, ...)`.
5. **Archetype Clustering** — cluster the per-player vectors within each
   position group to recover recurring behavioral profiles. Each cluster =
   one archetype. Archetypes are position-group-specific.
6. **Archetype Assignment** — assign each player their archetype label.

---

## Repo Structure

```
.
├── archetypes/          Player archetype pipeline (steps 1–6 above)
├── features/            Tabular and graph feature builders
├── models/              The 4 models (2 tabular + 2 GNN)
├── evaluation/          k-fold CV harness and metrics
├── notebooks/           Project-level EDA
├── configs/             config.yaml — paths, hyperparameters, k-fold k
├── tests/               Unit tests
└── data/                raw / processed / team_seasons (gitignored)
```

Each subpackage has a `notebooks/` directory for diagnostics local to that
stage (event-cluster inspection, graph rendering, results tables).

---

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch Geometric occasionally needs platform-specific wheels — see the
[PyG install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if `pip install torch-geometric` fails.

Drop the raw Wyscout event data into `data/raw/`. That directory is
gitignored; only a `.gitkeep` is tracked.

---

## Running the Archetype Pipeline

The archetype pipeline runs once and writes its artifacts into
`data/processed/archetype_artifacts/`. The downstream feature builders read
those artifacts.

Order:

1. `archetypes/position_groups.py` — assign each player a position group and
   filter events.
2. `archetypes/event_clustering.py` — fit intra-event cluster models, one per
   (position group, event type).
3. `archetypes/player_aggregation.py` — build the per-player feature table.
4. `archetypes/archetype_clustering.py` — fit one archetype clustering model
   per position group.
5. `archetypes/assign_archetypes.py` — write the final player→archetype map.

Inspect each stage with the notebooks in `archetypes/notebooks/`.

---

## Running the Models

Each of the four models exposes `fit` / `predict` and is compatible with the
shared CV harness in `evaluation/cross_validate.py`.

- Model 1 — `models.baseline_positions.BaselinePositionsModel` (tabular, raw positions)
- Model 2 — `models.baseline_archetypes.BaselineArchetypesModel` (tabular, archetypes)
- Model 3 — `models.gnn_positions.GNNPositionsModel` (GNN, raw positions)
- Model 4 — `models.gnn_archetypes.GNNArchetypesModel` (GNN, archetypes)

The tabular models consume the output of `features.build_tabular`; the GNN
models consume the output of `features.build_graphs`. Both feature builders
accept `kind="position"` or `kind="archetype"` so the four model variants
share machinery.

---

## Evaluation

`evaluation/cross_validate.py` runs k-fold CV over team-seasons for any
model, regardless of whether it consumes tabular rows or PyG graphs. Metrics
(MAE, RMSE, R²) live in `evaluation/metrics.py`; the final 8-way comparison
table is assembled in `evaluation/notebooks/results.ipynb`.

k-fold `k` and the random seed live in `configs/config.yaml`.

---

## Tests

```bash
pytest
```

Tests live in `tests/` and cover the position-group mapping, event clustering,
graph construction, and metric functions.
