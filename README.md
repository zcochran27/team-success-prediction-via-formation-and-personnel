# Predicting xG Differential in D1 College Soccer

Formation, personnel, and player archetypes as predictors of within-match
chance creation.

Dataset: Wyscout. Level: NCAA Division I (over 7,000 matches, 2020 to 2024).

## 1. Overview

This project asks whether a team's formation, the players filling each slot,
and data-driven archetypes of those players can predict expected-goal (xG)
production within a match. We model the xG differential a team produces within
a half, using Wyscout event data and formation logs that record which players
were on the pitch and when. We compare several ways to encode a lineup and a
graph model against a tabular baseline.

Modeling at the half level (rather than per match or per season) is a
deliberate choice: NCAA teams make many substitutions, so the lineup changes
often within a match, and half-level rows keep the formation and personnel
changes at halftime as separate examples while staying dense enough for a
clean comparison of modeling choices.

## 2. Research questions

1. Personnel encoding: do data-driven player archetypes beat raw position
   labels?
2. Quality encoding: does a player's season statistical profile improve
   prediction?
3. Opponent conditioning: does adding the opponent's lineup and personnel
   help (matchup vs ego)?
4. Representation: does a graph neural network beat a tabular XGBoost model?

These define a 2x2x2 design grid (Section 5) so each choice can be assessed on
its own and in combination.

## 3. Unit of observation and target

### 3.1 Unit of observation

Each example is a half snapshot: one row per (match, half, focal team). A
snapshot records:

- the half bounds (`period_start_min`, `period_end_min`,
  `period_duration_min`), the match id, the season, and the team ids;
- each team's formation string (e.g. `4-2-3-1`, `4-4-2`);
- the 11 starters in formation-template order and any substitutes in
  chronological entry order, each with their raw position, season archetype,
  10-dim season stat vector, minutes played, and (for subs) the minute they
  entered;
- half-level match-state features: shots, xG, and goals per team plus their
  differentials.

Each (match, half) contributes two rows, one per team, kept together in
the train/test split by sharing `match_id`.

### 3.2 Target

The target is the team xG differential over the half, rate-normalized
per 30 minutes:

    y = xg_team1_minus_team2_per_30
      = (team1_xg - team2_xg) / period_duration_min * 30

The xG values are Wyscout's built-in xG, summed over each team's shots in the
half. xG is used instead of goals because goals are too sparse for half-level
regression while xG keeps directional information. The differential is
sign-symmetric (the opponent's row is the negation), so one regression serves
both perspectives, and the per-30 normalization removes the confound that
longer halves accumulate more xG.

## 4. Player archetype construction

A data-driven taxonomy of player roles is built before any modeling, per
(player, season), so a player who changes role between seasons gets different
archetypes in each.

1. Position groups. Players are partitioned into eight groups: center
   defenders (CD), left/right wide defenders (LWD/RWD), center midfielders
   (CM), center forwards (CF), left/right wide players (LWP/RWP), and
   goalkeepers (GK). The left/right split preserves the asymmetry of wide
   roles.
2. Intra-event clustering. For each (position group, event type) pair, a
   StandardScaler + MiniBatchKMeans pipeline clusters individual events
   (using start/end coordinates and similar features) into subtypes, for
   example short vs long passes.
3. Player-season aggregation. For each (player, season), the event cluster
   labels become a feature vector: the proportion of each event type and the
   distribution of those events across the clusters.
4. Archetype clustering. KMeans runs on the player-season vectors within each
   position group; each cluster is one archetype (CD-0, CM-1, and so on).
5. Season stats. Separately, `season_stats.py` builds a 10-dim per-(player,
   season) stat vector (five per-90 volumes and five success rates) used as a
   separate feature block.

The number of clusters per event type and the number of archetypes per group
are set in `configs/config.yaml`.

## 5. Snapshot construction and models

The formation logs are turned into half snapshots by
`formations/lineups_half_subs.py`, which assigns starters to template slots,
orders substitutes by entry time, attaches archetypes and season stats,
recomputes half-level match-state features, reframes to focal perspective, and
writes a match-blocked 80/20 train/test split.

Three model families share the same three design axes (personnel: position vs
archetype; opponent: ego vs matchup; quality: with or without season stats):

- tabular: XGBoost on the starters plus all substitute slots.
- graph nueral network: `HalfSubsGNN`, a typed-edge graph attention network over the
  per-half lineup graph. Players are nodes; edges are intra-team (template
  adjacency plus sub-inherited copies), sub (a substitute and the starter they
  replaced), and matchup (cross-team pairs weighted by shared time on the
  pitch, paired mode only). In paired mode the prediction is
  `head(team1) - head(team2)`.

## 6. Results

On the shared held-out test set, the best model is a GNN that combines
archetypes, the matchup graph, and season stats, reaching an R-squared of
0.283 and a sign accuracy of 0.659. Averaged over the design grid, season
stats help most (R-squared lift +0.166), then archetypes (+0.085), then the
GNN structure over the tabular baseline (+0.074); opponent conditioning helps
least (+0.045).

## 7. Limitations and future work

College soccer's high substitution rate adds noise to the lineup even within a
half, and the data spans every Division I conference, so team and player
quality vary widely. Match-level features (home/away, travel, pitch, weather)
are not captured. Promising directions: simulating how a new signing or a
position change would affect expected performance, learning archetypes
end-to-end inside the xG model, and applying the approach to professional
leagues with fewer substitutions.

## 8. Repo layout

```
archetypes/   Stage 1: per-(player, season) archetype pipeline + season stats
formations/   Stage 2: half snapshots + match state + train/test split
features/     snapshot filter, graph builder, PyG dataset, vocabularies
graphs/       formation templates, lineup-to-template alignment, visualization
models/       HalfSubsGNN + cross-model comparison and diagnostics helpers
configs/      config.yaml (paths + clustering hyperparameters)
report/       LaTeX report

```

## 9. Setup

Requires Python 3.11+.

```powershell
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate on Unix
pip install -r requirements.txt
```

## 10. Pipeline

```powershell
# 1. Player archetypes -> data/processed/archetype_artifacts/
python -m archetypes.run_pipeline

# 2. Half snapshots + match state + match-blocked train/test split
python -m formations.lineups_half_subs

# 3. Train the GNN variants -> artifacts/half_subs/
python -m scripts.train_gnn_subs_batch
```

The tabular models are fit inside the comparison notebook.

## 11. Evaluation

Open [models/notebooks/all_models_comparison.ipynb](models/notebooks/all_models_comparison.ipynb)
for the leaderboard, metric bars, predictions-vs-targets scatter, and residual
histograms over every run, and
[models/notebooks/training_diagnostics.ipynb](models/notebooks/training_diagnostics.ipynb)
for the GNN per-epoch curves.
