# D1 Soccer Prediction

Predict the **xG differential** between two teams during a stable
formation window from formation, personnel, and player archetypes,
using Wyscout event-level data.

The unit of observation is a **formation snapshot**: one row per
`(match, joint-stable formation window)`. A window starts/ends whenever
*either* team makes a formation change, so within each row both teams'
shapes and 11-man lineups are constant. The row is then reframed into
two **focal-perspective** rows (one per team) so each model predicts the
focal team's xG advantage during the window, rate-normalized to per 30
minutes (`xg_team1_minus_team2_per_30`).

See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for the detailed
research narrative; this file is the engineering quick-reference.

---

## Research design

Three orthogonal modeling choices define a **2 × 2 × 2 grid** of 8 model
variants:

| Axis | Levels |
|---|---|
| input representation | tabular (XGBoost) · graph (GNN) |
| personnel encoding | raw positions · per-season archetypes |
| opponent in view | ego (focal team only) · matchup (both teams) |

| # | Module / class | rep | personnel | opponent |
|---|---|---|---|---|
| 1 | [`models.tab_pos_ego`](models/tab_pos_ego.py) | tabular | position | ego |
| 2 | [`models.tab_pos_matchup`](models/tab_pos_matchup.py) | tabular | position | matchup |
| 3 | [`models.tab_arch_ego`](models/tab_arch_ego.py) | tabular | archetype | ego |
| 4 | [`models.tab_arch_matchup`](models/tab_arch_matchup.py) | tabular | archetype | matchup |
| 5 | `LineupGNN(kind="position",  mode="single")` | graph | position | ego |
| 6 | `LineupGNN(kind="archetype", mode="single")` | graph | archetype | ego |
| 7 | `LineupGNN(kind="position",  mode="paired")` | graph | position | matchup |
| 8 | `LineupGNN(kind="archetype", mode="paired")` | graph | archetype | matchup |

All four GNN variants share [`models.gnn.LineupGNN`](models/gnn.py);
`kind` and `mode` are constructor arguments.

Comparisons of interest:

- **tabular → graph** (1↔5, 2↔7, 3↔6, 4↔8) — does graph structure add signal?
- **position → archetype** (1↔3, 2↔4, 5↔6, 7↔8) — does archetype enrichment add signal?
- **ego → matchup** (1↔2, 3↔4, 5↔7, 6↔8) — does conditioning on the opponent add signal?

---

## Repo layout

Each subdirectory has its own `README.md` with module-level detail.

```
.
├── archetypes/    Stage 1 — per-(player, season) archetype pipeline
├── formations/    Stage 2 — joint-stable windows + match state + archetype enrichment
├── features/      Filter (snapshots.py), graph builder (build_graphs.py), and PyG Dataset (dataset.py)
├── graphs/        Topology primitives: formation templates, matchup edges, alignment, visualization
├── models/        4 tabular XGBoost variants + LineupGNN (4 variants) + comparison helpers
├── scripts/       CLIs: build_train_test_split, train_gnn, predict_gnn  (gitignored — see below)
├── configs/       config.yaml — paths + clustering hyperparameters
├── tests/         pytest suite (graph builder, dataset, matchup edges)
└── data/          raw/, processed/, team_seasons/  (gitignored)
```

> The whole `scripts/` directory is in `.gitignore`. Remove that line if
> you want the CLI scripts under version control.

---

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate on Unix
pip install -r requirements.txt
```

PyTorch Geometric occasionally needs platform-specific wheels — see the
[PyG install matrix](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
if `pip install torch-geometric` fails.

Drop the raw Wyscout exports into `data/raw/` (`all_events.parquet`,
`matches.parquet`, `seasons.parquet`, `formations.parquet`, `leagues.parquet`,
`team_info.parquet`, `player_info.parquet`). That directory is gitignored.

---

## End-to-end pipeline

```powershell
# 1. Player archetypes (~10 min, one-time per dataset)
python -m archetypes.run_pipeline
# -> data/processed/archetype_artifacts/player_archetype_map.parquet

# 2. Lineup snapshots + match state + archetype join
python -m formations.run
# -> data/processed/lineup_snapshots.parquet  (~79.8k focal rows)

# 3. Match-blocked 80/20 train/test split
python -m scripts.build_train_test_split
# -> data/processed/{train,test}_snapshots.parquet  (~59.1k / ~14.5k)

# 4. Train the 4 GNN variants (~20 min each for single, ~50 min for paired)
python -m scripts.train_gnn --kind position  --mode single --epochs 30
python -m scripts.train_gnn --kind archetype --mode single --epochs 30
python -m scripts.train_gnn --kind position  --mode paired --epochs 30
python -m scripts.train_gnn --kind archetype --mode paired --epochs 30
# -> artifacts/gnn_<kind>_<mode>/ : log.jsonl, checkpoint.pt, val_preds.parquet, summary.json
```

Tabular models are evaluated inside the comparison notebook; there's no
separate training script (XGBoost converges in a single fit, no
intermediate checkpoints to save).

---

## Evaluation

Both pipelines train on the same `train_snapshots.parquet` and report
metrics on the same `test_snapshots.parquet`. The shared filter
[`features.snapshots.filter_buildable_snapshots`](features/snapshots.py)
runs *before* the split so both model classes consume identical rows
(both formations in [`graphs.templates.FORMATION_TEMPLATES`](graphs/templates.py)
and all 22 position labels non-null). NaN archetypes are kept in both
pipelines: XGBoost encodes them as the `"MISSING"` category and the GNN's
archetype vocab includes a `"MISSING"` token.

Open [`models/notebooks/all_models_comparison.ipynb`](models/notebooks/all_models_comparison.ipynb)
to render:

- A unified summary table over all 8 models
- A metric bar chart (MAE / RMSE / R² / sign acc)
- 2 × 4 predictions-vs-targets scatter (tabular row vs GNN row)
- 2 × 4 residual histograms
- Pairwise ablation deltas on the three design axes

For GNN-specific training diagnostics (loss curves, gradient norms,
calibration plots) see
[`models/notebooks/gnn_training.ipynb`](models/notebooks/gnn_training.ipynb).

Primary metric: MAE on `xg_team1_minus_team2_per_30`. Secondary: RMSE,
R², sign accuracy (did the model identify the dominating team?).

Caveat: the GNN saves the best-test-MSE checkpoint, so it gets a mild
advantage versus the single-fit tabular model. Easy fix later by adding
a small inner validation slice carved out of `train_snapshots.parquet`.

---

## Tests

```powershell
pytest tests/
```

The dataset-level tests require `data/processed/lineup_snapshots.parquet`;
they skip cleanly when missing.
