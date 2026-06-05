# D1 Soccer Prediction

Predict the within-half xG differential between two teams from formation,
personnel, and data-driven player archetypes, using Wyscout event data for
NCAA Division I soccer.

The unit of observation is a half snapshot: one row per (match, half, focal
team), recording each team's formation, the 11 starters and any substitutes
(with minutes and entry times), each player's raw position and season
archetype, and per-season player stats. The target is the team1-minus-team2 xG
differential over the half, rate-normalized per 30 minutes
(`xg_team1_minus_team2_per_30`).

See [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md) for the full research
narrative; this file is the engineering quick-reference.

## Models

Three model families, each across the same 2x2x2 design axes (personnel:
position vs archetype; opponent: ego vs matchup; quality: with or without
season stats), for 24 trained runs:

- tab: XGBoost on the 11 starter slots per team.
- tab_subs: XGBoost on the starters plus all sub slots.
- gnn_subs: `HalfSubsGNN`, a typed-edge graph attention network over the
  per-half lineup graph.

The comparisons of interest are tab/tab_subs vs gnn_subs (does graph structure
help?), position vs archetype (does the archetype encoding help?), and ego vs
matchup (does conditioning on the opponent help?).

## Repo layout

```
archetypes/   Stage 1: per-(player, season) archetype pipeline + season stats
formations/   Stage 2: half snapshots + match state + train/test split
features/     snapshot filter, graph builder, PyG dataset, vocabularies
graphs/       formation templates, lineup-to-template alignment, visualization
models/       HalfSubsGNN + cross-model comparison and diagnostics helpers
configs/      config.yaml (paths + clustering hyperparameters)
report/       LaTeX report
scripts/      GNN train/predict CLIs (gitignored)
data/         raw/ and processed/ (gitignored)
```

## Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate on Unix
pip install -r requirements.txt
```

PyTorch Geometric sometimes needs platform-specific wheels; see the PyG
install docs if `pip install torch-geometric` fails.

Drop the raw Wyscout exports into `data/raw/` (`all_events.parquet`,
`matches.parquet`, `seasons.parquet`, `formations.parquet`, `leagues.parquet`,
`team_info.parquet`, `player_info.parquet`). That directory is gitignored.

## Pipeline

```powershell
# 1. Player archetypes -> data/processed/archetype_artifacts/
python -m archetypes.run_pipeline

# 2. Half snapshots + match state + match-blocked train/test split
python -m formations.lineups_half_subs

# 3. Train the GNN variants -> artifacts/half_subs/
python -m scripts.train_gnn_subs_batch
```

The tabular models are fit inside the comparison notebook.

## Evaluation

Open [models/notebooks/all_models_comparison.ipynb](models/notebooks/all_models_comparison.ipynb)
for the leaderboard, metric bars, predictions-vs-targets scatter, and residual
histograms over every run, and
[models/notebooks/training_diagnostics.ipynb](models/notebooks/training_diagnostics.ipynb)
for the GNN per-epoch curves.

Metrics: MAE, RMSE, R2, and sign accuracy (did the model pick the team with the
higher xG?) on the shared held-out test set.
