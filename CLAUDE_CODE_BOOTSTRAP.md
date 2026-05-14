# D1 Soccer Prediction — Claude Code Bootstrap

Use this document to scaffold the full project repo. Create all files and directories exactly as specified, with docstrings and `TODO` stubs in every `.py` file. Do not implement any logic yet — the goal is a clean, navigable scaffold that makes the project's intent clear.

---

## Project Summary

**Goal:** Predict NCAA Division I college soccer team performance (win rate and points per game) from a team's formation and player personnel.

**Dataset:** Wyscout event-level tracking data, NCAA Division I.

**Unit of observation:** Team-season (one record per team per season).

**Target variables:**
- Win Rate — proportion of games won (continuous, [0, 1])
- Points per Game — total points / games played (continuous)

Both targets are modeled independently and compared.

---

## Research Design

The project answers two questions simultaneously:

1. Does representing team structure as a **graph** (vs. a flat feature vector) improve prediction?
2. Do richer **player archetypes** (vs. raw positional labels) improve prediction?

This produces a 2×2 model comparison:

|                   | Tabular  | Graph   |
|-------------------|----------|---------|
| **Raw Positions** | Model 1  | Model 3 |
| **Archetypes**    | Model 2  | Model 4 |

Comparisons:
- Models 1 vs. 2 and 3 vs. 4 → marginal value of archetypes
- Models 1 vs. 3 and 2 vs. 4 → marginal value of graph structure
- Models 1 vs. 4 → combined gain

---

## Player Archetype Pipeline

Before any modeling, a data-driven archetype taxonomy is constructed. This is a standalone preprocessing phase that runs once and produces artifacts consumed by all downstream models.

**Steps:**

1. **Position Group Definition** — partition players into 5 groups: Center Defenders, Wide Defenders, Center Midfielders, Center Forwards, Wide Players (wide mids + wingers)

2. **Event Filtering** — for each position group, filter the event log to only events by players in that group. Relevant event types: passes, dribbles, tackles, shots.

3. **Intra-Event Clustering** — within each event type, cluster individual event instances to identify qualitatively distinct subtypes (e.g. short vs. long passes, progressive vs. lateral dribbles). Produces cluster labels per event type.

4. **Player-Level Aggregation** — aggregate each player's event history into a feature vector: for each event type, record (% of total actions, % in cluster 1, % in cluster 2, ...). This encodes behavioral tendencies.

5. **Archetype Clustering** — cluster the per-player aggregation vectors within each position group to identify recurring behavioral profiles. Each cluster = one archetype. Archetypes are position-group-specific.

6. **Archetype Assignment** — assign each player their archetype label based on cluster membership.

---

## Model Descriptions

### Model 1: Tabular Baseline (Raw Positions)
- Input: 12-feature vector per team-season
  - 11 features: typical (modal) position of each player in each formation slot
  - 1 feature: formation encoding (e.g. 4-3-3, 4-2-3-1)
- Task: regression on win rate and points per game
- Purpose: lower bound; isolates signal from basic positional composition

### Model 2: Tabular Baseline (Archetypes)
- Same 12-feature structure as Model 1
- Player features replaced with archetype encodings instead of raw positions
- Purpose: isolates marginal value of archetype enrichment over raw positions

### Model 3: Graph Model (Raw Positions)
- Team represented as a graph; nodes = 11 players, node features = raw positional encodings
- Edge construction rules (formation governs topology):
  - Defenders connected to one another
  - Midfielders connected to one another
  - Attackers connected to one another
  - Cross-line edges where formation implies direct interaction (e.g. CM→ST, FB→W)
- GNN produces team-level embedding for regression
- Purpose: isolates marginal value of graph structure over tabular, holding features constant

### Model 4: Graph Model (Archetypes)
- Same graph structure as Model 3
- Node features replaced with archetype encodings
- Purpose: combines graph structure + archetype enrichment; strongest model

---

## Evaluation

- **Method:** K-fold cross-validation on the team-season dataset
- **Metrics:** MAE, RMSE, R²
- **Comparison:** all 4 models × 2 target variables = 8-way results table

---

## Repo Structure to Scaffold

Create the following structure. For every `.py` file: add a module-level docstring describing its role, define all key functions with docstrings and `TODO` bodies. For `.ipynb` files: create valid empty notebooks with a markdown cell at the top describing the notebook's purpose and a code cell stub. For `config.yaml`: populate with placeholder values and comments. For `requirements.txt`: include all likely dependencies. For `README.md`: write a full project README.

```
d1-soccer-prediction/
│
├── data/
│   ├── raw/                              # Raw Wyscout event logs (gitignored)
│   ├── processed/                        # Cleaned, filtered event data
│   └── team_seasons/                     # Final team-season observations
│
├── archetypes/
│   ├── position_groups.py                # Partition players into position groups; filter events
│   ├── event_clustering.py               # Intra-event clustering per event type
│   ├── player_aggregation.py             # Aggregate events into per-player feature vectors
│   ├── archetype_clustering.py           # Cluster player vectors into archetypes per position group
│   ├── assign_archetypes.py              # Assign archetype labels to each player
│   │
│   └── notebooks/
│       ├── event_clustering.ipynb        # Visualize cluster separation; inspect event subtypes
│       ├── player_aggregation.ipynb      # Inspect per-player vectors; sanity check aggregation
│       ├── archetype_clustering.ipynb    # Elbow/silhouette plots; PCA/UMAP of archetype clusters
│       └── archetype_profiles.ipynb      # Radar/bar charts per archetype; spot-check assignments
│
├── features/
│   ├── build_tabular.py                  # Assemble 12-feature team-season vectors (positions or archetypes)
│   ├── build_graphs.py                   # Construct PyG graphs from formation + player features
│   │
│   └── notebooks/
│       ├── graph_construction.ipynb      # Visualize sample team graphs; verify edge logic per formation
│       └── graph_comparison.ipynb        # Compare graph topology across formations
│
├── models/
│   ├── baseline_positions.py             # Model 1: tabular regression with raw positions
│   ├── baseline_archetypes.py            # Model 2: tabular regression with archetypes
│   ├── gnn_positions.py                  # Model 3: GNN with raw positional node features
│   └── gnn_archetypes.py                 # Model 4: GNN with archetype node features
│
├── evaluation/
│   ├── cross_validate.py                 # Model-agnostic k-fold CV harness
│   ├── metrics.py                        # MAE, RMSE, R² computation and comparison tables
│   │
│   └── notebooks/
│       └── results.ipynb                 # Cross-model results; 8-way comparison table and plots
│
├── notebooks/
│   └── eda.ipynb                         # Dataset overview: distributions, formation frequencies,
│                                         # event volume by position group
│
├── configs/
│   └── config.yaml                       # Paths, hyperparameters, k-fold k, cluster counts
│
├── tests/
│   ├── test_position_groups.py
│   ├── test_event_clustering.py
│   ├── test_build_graphs.py
│   └── test_metrics.py
│
├── README.md
└── requirements.txt
```

---

## Key Implementation Notes

**archetypes/position_groups.py**
- Define a mapping from Wyscout position labels to the 5 position groups
- Function to filter the event dataframe to a given position group and set of event types

**archetypes/event_clustering.py**
- For each position group × event type combination, fit a clustering model (e.g. KMeans)
- Number of clusters per event type should be configurable via config.yaml
- Save cluster models to disk for reuse in player_aggregation

**archetypes/player_aggregation.py**
- Load cluster models; for each player, assign cluster labels to their events
- Aggregate into a vector: (% event type A, % cluster 1 within A, % cluster 2 within A, ...) for each type
- Output: one row per player with position group label

**archetypes/archetype_clustering.py**
- For each position group, cluster the player aggregation vectors
- Number of archetypes per position group configurable via config.yaml
- Save archetype cluster models to disk

**archetypes/assign_archetypes.py**
- Load archetype cluster models; assign each player their archetype label
- Output: player → archetype mapping as a dataframe or dict

**features/build_tabular.py**
- Accept either raw position encodings or archetype encodings as player features (parameterized)
- For each team-season, look up the modal starting lineup and formation
- Assemble the 12-feature vector; encode formation as a categorical

**features/build_graphs.py**
- Use PyTorch Geometric (PyG) to construct Data objects
- Node features: either raw positional encodings or archetype encodings
- Edge index: built from formation-specific connectivity rules
- Store formation-to-edge-rule mapping; should be easy to extend with new formations
- Output: list of PyG Data objects, one per team-season

**models/baseline_positions.py and baseline_archetypes.py**
- Scikit-learn compatible regression pipeline
- Try gradient boosting (XGBoost or LightGBM) as the primary model type
- Expose fit() and predict() interfaces compatible with the CV harness

**models/gnn_positions.py and gnn_archetypes.py**
- PyTorch Geometric GNN
- Architecture: 2–3 GNN layers (GCN or GAT) → global mean pooling → MLP head
- Output: scalar regression (win rate or PPG)
- Expose fit() and predict() interfaces compatible with the CV harness

**evaluation/cross_validate.py**
- Accept any model and dataset; run k-fold CV
- Return per-fold and aggregate metrics
- Should work for both sklearn-style and PyG-style models

**evaluation/metrics.py**
- Compute MAE, RMSE, R² from predictions and ground truth
- Produce a formatted comparison table across all models and both targets

**configs/config.yaml**
- data_paths: raw, processed, team_seasons
- archetype_clustering: n_clusters per position group, n_event_clusters per event type
- evaluation: k_folds, random_seed
- model: gnn_layers, hidden_dim, learning_rate, epochs

**requirements.txt** should include at minimum:
- pandas, numpy, scikit-learn
- torch, torch-geometric
- xgboost or lightgbm
- networkx, matplotlib, seaborn
- umap-learn
- pyyaml
- jupyter

---

## Instructions for Claude Code

1. Initialize a git repo (`git init`)
2. Create the full directory and file structure above
3. For every `.py` file: write a module docstring, define all functions mentioned in "Key Implementation Notes" with full docstrings and `TODO` bodies — no logic implemented yet
4. For every `.ipynb` file: create a valid Jupyter notebook with a markdown cell describing the notebook's purpose and one stub code cell
5. Write `config.yaml` with placeholder values and inline comments explaining each field
6. Write `requirements.txt` with all dependencies listed above (pin to recent stable versions)
7. Write a full `README.md` covering: project overview, research design, repo structure, setup instructions, how to run the archetype pipeline, how to run models, how to evaluate
8. Stage and commit everything: `git add . && git commit -m "initial scaffold"`
