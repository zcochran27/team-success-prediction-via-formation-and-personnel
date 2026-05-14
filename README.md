# D1 Soccer Prediction

Predict the **xG differential** between two teams during a match window from
formation, personnel, and player archetypes, using Wyscout event-level data.

The unit of observation is a **formation snapshot**: one row per
`(match, joint-stable formation window)`. A window starts/ends whenever
*either* team makes a formation change, so within each row both teams' shapes
and 11-man lineups are constant. Each row predicts how lopsided chance
creation was during that window (`xg_home_minus_away`, ideally normalized to
`xg_home_minus_away_per_30` since window lengths vary).

---

## Research Design

The project answers three questions simultaneously:

1. Does representing team structure as a **graph** (vs. a flat feature vector)
   improve prediction?
2. Do richer **player archetypes** (vs. raw positional labels) improve
   prediction?
3. Does conditioning on the **opponent's formation and personnel** improve
   prediction beyond just knowing your own team's setup?

This produces a 2×2×2 model comparison (8 variants):

| Structure | Personnel encoding | Opponent included? | Model |
|-----------|--------------------|--------------------|-------|
| Tabular | Raw positions | No  | Model 1 |
| Tabular | Raw positions | Yes | Model 2 |
| Tabular | Archetypes    | No  | Model 3 |
| Tabular | Archetypes    | Yes | Model 4 |
| Graph   | Raw positions | No  | Model 5 |
| Graph   | Raw positions | Yes | Model 6 |
| Graph   | Archetypes    | No  | Model 7 |
| Graph   | Archetypes    | Yes | Model 8 |

Comparisons of interest:
- Odd→even pairs (1 vs 2, 3 vs 4, …) — marginal value of including the opponent
- 1 vs 3, 2 vs 4, 5 vs 7, 6 vs 8 — marginal value of archetypes
- 1 vs 5, 2 vs 6, 3 vs 7, 4 vs 8 — marginal value of graph structure
- 1 vs 8 — combined gain

The "no opponent" variants see only the home team's formation + 11 player
slots when predicting the home perspective (and symmetrically for away); the
"with opponent" variants see both sides simultaneously.

---

## Data Pipeline

Two preprocessing pipelines run before any modeling. They produce the
artifacts the feature builders consume.

### Stage 1 — Player Archetype Pipeline (`archetypes/`)

Builds a data-driven taxonomy of player roles, **per season** (so a player's
archetype can change year over year).

1. **Position group assignment** — partition each event/player into 8 groups:
   `CD` (center defenders), `LWD` / `RWD` (wide defenders), `CM` (center mids),
   `CF` (center forwards), `LWP` / `RWP` (wide players), `GK`.
2. **Logical event types** — `pass`, `offensive_duel`, `defensive_duel`,
   `aerial_duel`, `shot`, `goalkeeper_exit`. GK uses a smaller vocabulary
   (`pass`, `goalkeeper_exit`).
3. **Intra-event clustering** — within each `(group, event type)` pair, fit a
   KMeans on the event-instance features to identify qualitatively distinct
   subtypes (e.g. short vs. long passes, progressive vs. lateral duels).
4. **Per-(player, season) aggregation** — for each `(player_id, season)`
   pair, the player's primary position group that season is determined by
   relevant-event volume; their feature vector encodes
   `(% of total actions per event type, % in each sub-cluster)`. Player-seasons
   that don't clear `min_events_per_player` are dropped.
5. **Archetype clustering** — KMeans on the per-(player, season) vectors,
   independently within each position group.
6. **Assignment** — written to
   `data/processed/archetype_artifacts/player_archetype_map.parquet` with schema
   `player_id, season, position_group, player_position, archetype`.

Run with `python -m archetypes.run_pipeline`. Notebooks under
`archetypes/notebooks/` inspect each stage.

### Stage 2 — Lineup Snapshot Pipeline (`formations/`)

Pivots the per-player formation log (`data/raw/formations.parquet`) into the
wide training table at `data/processed/lineup_snapshots.parquet`. Each row
covers one match window during which **both** teams' formations and 11-man
lineups were constant.

1. **Joint-stable windows** — for each match, intersect the two teams'
   formation timelines so each window covers a `[start, end)` interval where
   neither team has changed shape.
2. **Slot pivot** — write each of the 22 players into a numbered slot using
   the modern English shirt-number convention: 1=GK, 2/3=RB/LB,
   4/5=RCB/LCB, 6=CDM, 7/11=RW/LW, 8=CM, 9=ST, 10=CAM. Slot assignment is
   driven by raw position label with `position_x` tiebreaks; ambiguous labels
   (un-prefixed `CB`, multiple `CM`s) are disambiguated by depth rank.
3. **Match-state features** — for each window, sum every shot's `shot_xg` and
   count goals attributed to each team via `matches.home_team` /
   `matches.away_team`. Adds raw counts (`home_xg`, `away_xg`, `home_goals`,
   `away_goals`), home–away / away–home diffs, and `_per_30` rate variants of
   all eight. `period_duration_min` is also stored.
4. **Season + archetype enrichment** — each match is tagged with its calendar
   season (extracted as the 4-digit year from `seasons.name`). For every one
   of the 22 player slots, the player's archetype for that season is looked
   up from the archetype map and stored as `home_archetype_<n>` /
   `away_archetype_<n>`. Players who didn't clear the archetype pipeline's
   per-season event threshold get `NaN` — this is informative missingness,
   not a bug.

Run with `python -m formations.run`. End-of-match windows are clamped using
`matches.duration` (`Regular → 95'`, `ExtraTime → 125'`) so the open-ended
final period of each team doesn't inflate the per-30 denominator.

#### Snapshot schema (selected columns)

```
match_id, season, period_start_min, period_end_min, period_duration_min,
home_team_id, away_team_id, home_formation, away_formation,
home_player_1..11, home_position_1..11, home_archetype_1..11,
away_player_1..11, away_position_1..11, away_archetype_1..11,
home_xg, away_xg, home_goals, away_goals,
xg_home_minus_away, xg_away_minus_home,
goals_home_minus_away, goals_away_minus_home,
{...all four}_per_30
```

---

## Target Variable

**Primary target**: `xg_home_minus_away` (or its rate-normalized form,
`xg_home_minus_away_per_30`). Predict how much one team out-created the other
in expected-goal terms during a stable-formation window.

Why this target:
- xG smooths the high variance of raw goals (most windows have zero of
  either) without losing directional signal.
- Differential framing makes the prediction symmetric — `away - home` is just
  the negative — so a single regression handles both perspectives.
- Per-30 normalization is required for windows with very different durations
  (median ~5 min, max ~42 min after end-of-match clamping).

Raw goals and their differentials are kept on the row as secondary targets
for sensitivity checks.

---

## Repo Structure

```
.
├── archetypes/          Stage 1 — per-(player, season) archetype pipeline
├── formations/          Stage 2 — lineup snapshot + match-state + archetype enrichment
├── features/            Tabular and graph feature builders (consume the snapshot table)
├── models/              The 8 model variants (2 structures × 2 personnel × 2 opponent-inclusion)
├── evaluation/          k-fold CV harness and metrics
├── notebooks/           Project-level EDA
├── configs/             config.yaml — paths, hyperparameters, k-fold k
├── tests/               Unit tests
└── data/                raw / processed (gitignored)
```

Each subpackage has a `notebooks/` directory for stage-local diagnostics
(event-cluster inspection, archetype profiles, snapshot QA, results tables).

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

Drop the raw Wyscout exports into `data/raw/` (`all_events.parquet`,
`matches.parquet`, `seasons.parquet`, `formations.parquet`, …). That directory
is gitignored; only a `.gitkeep` is tracked.

---

## Running the Pipelines

End-to-end build, in order:

```bash
python -m archetypes.run_pipeline    # Stage 1: writes player_archetype_map.parquet
python -m formations.run             # Stage 2: writes lineup_snapshots.parquet
```

Stage 2 reads the artifact written by Stage 1, so the order matters.

The lineup-snapshot table at `data/processed/lineup_snapshots.parquet` is the
**base training dataframe** for all 8 models.

---

## Running the Models

Each of the eight model variants exposes `fit` / `predict` and is compatible
with the shared CV harness in `evaluation/cross_validate.py`.

| Model | Module |
|-------|--------|
| 1 — Tabular · Raw positions · ego        | `models.tab_pos_ego` |
| 2 — Tabular · Raw positions · matchup    | `models.tab_pos_matchup` |
| 3 — Tabular · Archetypes · ego           | `models.tab_arch_ego` |
| 4 — Tabular · Archetypes · matchup       | `models.tab_arch_matchup` |
| 5 — Graph · Raw positions · ego          | `models.gnn_pos_ego` |
| 6 — Graph · Raw positions · matchup      | `models.gnn_pos_matchup` |
| 7 — Graph · Archetypes · ego             | `models.gnn_arch_ego` |
| 8 — Graph · Archetypes · matchup         | `models.gnn_arch_matchup` |

Tabular models consume `features.build_tabular`; GNN models consume
`features.build_graphs`. Each feature builder accepts:

- `personnel={"position", "archetype"}` — which columns to encode
- `include_opponent={False, True}` — whether to emit features for both teams
  or only the focal team

so the eight variants share machinery.

---

## Evaluation

`evaluation/cross_validate.py` runs k-fold CV over snapshots for any model,
regardless of whether it consumes tabular rows or PyG graphs. To prevent
leakage, folds are blocked by `match_id` so windows from the same match never
straddle train/test.

Primary metric: MAE on `xg_home_minus_away_per_30`. Secondary: RMSE, R²,
sign-accuracy (did the model get the dominance direction right?). All defined
in `evaluation/metrics.py`. The final 8-way comparison table is assembled in
`evaluation/notebooks/results.ipynb`.

k-fold `k` and the random seed live in `configs/config.yaml`.

---

## Tests

```bash
pytest
```

Tests live in `tests/` and cover the position-group mapping, event
clustering, snapshot joint-interval logic, slot-number assignment, and metric
functions.
