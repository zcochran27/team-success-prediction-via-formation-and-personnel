# Predicting xG Differential in D1 College Soccer

**Formation, Personnel, and Player Archetypes as Predictors of Within-Match Chance Creation**

**Dataset:** Wyscout &nbsp;•&nbsp; **Level:** NCAA Division I

---

## 1. Project Overview

This project investigates whether the tactical formation a team deploys, the
players occupying each positional slot, and the data-driven *archetypes* of
those players can predict expected-goal (xG) production during a match
window. Using event-level tracking data from Wyscout — together with
formation logs that record exactly which 11 players were on the pitch during
each interval of stable shape — the project develops a progression of models
of increasing structural sophistication, culminating in a graph-based
representation that encodes both tactical shape and inter-player
relationships, and that conditions on the opponent's setup as well as the
focal team's.

The project has evolved from an initial team-season formulation to a finer-
grained **formation-snapshot** formulation. This change was motivated by two
data realities: (a) NCAA teams change formation multiple times per match, so
a single per-season formation label discards substantial tactical signal,
and (b) season-level outcomes (win rate, points per game) are too noisy and
too sparse to support a clean comparison of modeling choices at the volume
of data available.

## 2. Research Objective

The project answers three interrelated questions:

1. **Tactical signal**: Do a team's formation and its players' positional
   composition predict short-window xG creation?
2. **Personnel encoding**: Do data-driven *player archetypes* (richer
   behavioral profiles) outperform raw positional labels as the player
   feature?
3. **Opponent conditioning**: Does explicitly modeling the opponent's
   formation and personnel improve prediction over a unilateral
   ("ego-only") view of the focal team?
4. **Representation**: Does a graph-based representation of team structure
   improve on a tabular baseline?

Together these define a 2×2×2 model grid (Section 5) that lets each
modeling choice be assessed independently and in combination.

## 3. Unit of Observation & Target Variable

### 3.1 Unit of observation

Each observation is a **formation snapshot**: one row per
`(match, joint-stable formation window)`. A "joint-stable" window is a
contiguous time interval during which *neither* team makes a formation
change — bounded above and below by formation transitions on either side.
Within a single window, both teams' shapes and 11-man lineups are constant,
so the snapshot is a coherent description of the matchup over that
interval.

For each window, the snapshot records:

- The interval bounds (`period_start_min`, `period_end_min`,
  `period_duration_min`), the match identifier, the season, and the home /
  away team IDs.
- Both teams' formation strings (e.g. `4-2-3-1`, `4-4-2`).
- For each team, the 11 players on the pitch, slotted into shirt-number
  conventions: 1 = GK, 2/3 = RB/LB, 4/5 = RCB/LCB, 6 = CDM, 7/11 = wingers,
  8 = CM, 9 = ST, 10 = CAM. Slot assignment uses the player's raw position
  label with depth tiebreaks.
- For each slot: the player's raw position label (e.g. `LCB`) and their
  archetype label for that season (e.g. `CD-2`).
- Match-state features for that window — shots, xG, and goals for each
  team, plus their home-minus-away and away-minus-home differentials.

### 3.2 Target variable

The primary target is **xG differential**:

$$
y = \text{xg\_home\_minus\_away\_per\_30} = \frac{\text{home\_xg} - \text{away\_xg}}{\text{period\_duration\_min}} \times 30
$$

That is, the home team's xG advantage during the window, rate-normalized to
a per-30-minute basis (since window lengths vary from seconds to tens of
minutes).

This choice is motivated by:

- **Signal density**. Across a sample of formation windows, the median
  window has zero goals scored by either team, but most have non-zero
  shot-xG. Goals are too sparse to support window-level regression; xG
  retains directional information.
- **Symmetry**. Predicting a differential makes the target sign-symmetric
  — the away perspective is just the negation — so a single regression
  serves both viewpoints.
- **Comparability across windows**. Per-30 normalization removes the
  obvious confound that longer windows accumulate more xG.

Raw goal counts (`home_goals`, `away_goals`) and the four other
differential variants are retained on each row as secondary targets for
sensitivity analysis.

## 4. Player Archetype Construction

A data-driven taxonomy of player roles is constructed before any modeling.
Crucially, the taxonomy is computed **per (player, season)** rather than
once per player: a player who shifts role between seasons (e.g., moves
from full-back to wing-back to wing) is assigned different archetypes in
each season.

### 4.1 Position groups

Players are partitioned into eight position groups, refined from the
original five to preserve left / right asymmetry:

- **CD** — Center Defenders
- **LWD**, **RWD** — Left and Right Wide Defenders (full-backs, wing-backs)
- **CM** — Center Midfielders (defensive, central, attacking)
- **CF** — Center Forwards
- **LWP**, **RWP** — Left and Right Wide Players (wingers, wide attacking
  mids, wide forwards)
- **GK** — Goalkeepers

The left / right split for the wide groups was added after diagnostic
inspection revealed materially different behavioral profiles between
left- and right-side wide players in this dataset.

### 4.2 Logical event types

For each position group, the event log is filtered to a small vocabulary
of *logical event types*:

- `pass`
- `offensive_duel`, `defensive_duel`, `aerial_duel` (ground duels split on
  Wyscout's offensive / defensive secondary tag; aerials kept separately)
- `shot`
- `goalkeeper_exit` (GK only)

Goalkeepers use a reduced vocabulary of `{pass, goalkeeper_exit}` since
duels and shots from GKs are too rare to support stable clustering.

### 4.3 Intra-event clustering

Within each `(position group, event type)` pair, individual event
instances are clustered with KMeans on a small set of spatial and
contextual features (`start_x`, `start_y`, `end_x`, `end_y`,
`pass_length`, `pass_angle`, `shot_xg`, duel outcomes, etc.). The
resulting cluster labels capture qualitatively distinct subtypes — short
versus long passes, progressive versus stopping defensive duels, and so
on. The number of sub-clusters is tuned per pair from elbow and
silhouette diagnostics.

### 4.4 Player-season aggregation

For each `(player_id, season)` pair, the player's primary position group
that season is determined by relevant-event volume. Their feature vector
records, for each event type relevant to that group:

- The proportion of total in-group actions that fall in that event type.
- Within that event type, the proportional distribution across its
  intra-event clusters.

Player-seasons with fewer than `min_events_per_player` relevant events
are dropped — so a player can appear in some seasons and not others.

### 4.5 Archetype clustering

Per-season player vectors are then clustered with KMeans within each
position group. Each resulting cluster constitutes a player archetype.
Because clustering is run independently per group, archetype labels are
group-scoped (e.g. `CD-2` and `CM-2` are unrelated). The number of
archetypes per group (between 2 and 4 in the current configuration) is
chosen from per-group silhouette diagnostics.

### 4.6 Assignment

The final mapping is one row per `(player_id, season)` with columns
`position_group, player_position, archetype` — written to
`data/processed/archetype_artifacts/player_archetype_map.parquet`. This
artifact is consumed by the downstream snapshot-enrichment stage.

## 5. Formation Snapshot Construction

A second preprocessing stage builds the actual training table. It
operates on the raw formation log (`formations.parquet`), which records
each player on the pitch during each interval of stable shape per team.

### 5.1 Joint-stable windows

For each match, the two teams' formation timelines are merged into a
single sequence of breakpoints, and each consecutive pair of breakpoints
defines a window during which neither team changes shape. End-of-match
windows are clamped using the `matches.duration` field — Regular →
95 minutes, ExtraTime → 125 minutes — so the open-ended final period of
each team does not inflate the per-30 denominator. Penalty-shootout
events are excluded from the timeline entirely.

### 5.2 Slot pivot

Each window's 22 players are slotted into 1..11 per team using the
modern English shirt-number convention described in Section 3.1.
Assignment proceeds in a "most-constrained-first" pass over the players,
with ambiguous labels (un-prefixed `CB`, multiple `CM`s, etc.)
disambiguated by `position_x` depth rank.

### 5.3 Match-state features

For each window, every shot occurring in `[period_start_min,
period_end_min)` is attributed to its shooting team. The window record
gains:

- `home_xg`, `away_xg` (sum of `shot_xg`)
- `home_goals`, `away_goals` (count of `shot_isGoal == True`)
- Four differential columns: `xg_home_minus_away`, `xg_away_minus_home`,
  `goals_home_minus_away`, `goals_away_minus_home`
- Per-30-minute rate versions of all eight, computed as
  `value × 30 / period_duration_min`

### 5.4 Season + archetype enrichment

Each match is tagged with its calendar season (4-digit year extracted
from `seasons.name`). Each of the 22 player slots is then joined against
the per-season archetype map to attach `home_archetype_<n>` /
`away_archetype_<n>`. Slots whose player did not clear the archetype
pipeline's per-season threshold receive `NaN`; this is informative
missingness, not an error.

### 5.5 Sub-cluster collapse

The joint-stable windows produced by Section 5.1 turned out to be
heavily over-fragmented for tactical modeling. Diagnostic inspection of
the raw snapshot table revealed that **92% of window boundaries were
substitutions, not formation changes**: each player swap forced a new
joint window even though both teams' shapes were unchanged. The
fragmentation was twofold:

- *Count distortion.* 39% of snapshot rows lasted under 3 minutes, yet
  those short rows represented only 8.5% of total match time. A naive
  regression would have treated them as 39% of the training signal.
- *Target distortion.* Rate-normalizing xG to per-30 minutes (Section
  3.2) blew up on tiny windows. A single 0.7-xG shot in a 30-second
  sub-induced window inflates to >40 xG/30. The raw target distribution
  had a standard deviation of 1.59 and 1st/99th percentiles of −4.0 /
  +5.1 — all driven by short-window outliers, not by real chance-
  creation variance.

The fix is a one-pass **sub-cluster collapse** applied to the snapshot
table before the focal-perspective reframing. Within each match,
consecutive snapshots whose only change from their predecessor was a
substitution are merged together as long as the cluster's first sub
occurred within a window of length T (configured at T = 10 minutes) of
the current snapshot. A new segment is opened whenever:

- the match boundary is crossed,
- *either* team's formation string changes (a hard barrier — never
  merged across), or
- the next sub event falls more than T minutes after the cluster's
  first sub (it's too late to belong to the current cluster, so it
  opens a new one).

Within each merged segment, the recorded personnel for every slot is
the lineup from the **last underlying snapshot** of the segment —
i.e., the post-last-sub lineup that the coach committed to after the
substitution flurry resolved. xG, goals, and duration are summed
across the underlying snapshots; differentials and per-30 rates are
recomputed from the merged totals.

**Why this is tactically sufficient.** When a coach makes a flurry of
substitutions in a 10-minute window, the meaningful tactical state is
the lineup that plays out the *remainder* of the segment after the
swaps are done. The intermediate snapshots inside the cluster (e.g.,
"two of three subs have happened; the third hasn't yet") are transient
artifacts of the order in which the changes were keyed into the match
log, not committed tactical states.

The information loss is bounded and small:

- The "extra credit" given to a player who came on in the middle or
  end of a cluster — i.e., the minutes they're credited with that they
  did not actually play — is **bounded above by T = 10 minutes per
  cluster, by construction.**
- Across all merged segments, the recorded post-cluster player was on
  the pitch for a mean of **94.3% of segment minutes** (median 100%).
  Only 11% of slot-segments saw a recorded player on for less than 80%
  of the segment.
- Every formation change is preserved as a hard barrier. No tactical-
  shape information is averaged across formation transitions.

**Before / after.**

| Metric | Raw joint-stable windows | After T=10 cluster collapse |
|---|---:|---:|
| Snapshot rows (home/away frame) | 87,347 | 39,891 |
| Training rows (focal frame) | 174,694 | 79,782 |
| Avg segments per match | 15.9 | 7.3 |
| Median segment duration | 3.9 min | 12.0 min |
| 25th-percentile duration | 1.8 min | 8.8 min |
| Share of segments under 3 min | 39.0% | 5.0% |
| Target std (`xg_team1_minus_team2_per_30`) | 1.59 | 1.05 |
| Target 99th percentile | 5.12 | 2.84 |
| Target 1st percentile | −4.01 | −2.84 |
| Mean post-cluster personnel coverage | — | 94.3% |
| Formation transitions preserved | yes | yes (hard barrier) |

The target standard deviation drops by 34% and the tails compress by
nearly half, while every formation transition is preserved and the
recorded personnel is on the pitch for ≥94% of segment minutes on
average.

The merged table — written to
`data/processed/lineup_snapshots.parquet` after the focal-perspective
reframing — is the base training data for all model variants in
Section 6.

## 6. Model Pipeline

Three orthogonal modeling choices define a **2 × 2 × 2 grid** of model
variants:

| Axis | Levels |
|---|---|
| Input representation | Tabular vs. Graph |
| Personnel encoding | Raw position vs. Archetype |
| Opponent inclusion | Ego (focal team only) vs. Matchup (both teams) |

This produces eight models:

| # | Representation | Personnel | Opponent | Module name |
|---|---|---|---|---|
| 1 | Tabular | Raw position | Ego | `models.tab_pos_ego` |
| 2 | Tabular | Raw position | Matchup | `models.tab_pos_matchup` |
| 3 | Tabular | Archetype | Ego | `models.tab_arch_ego` |
| 4 | Tabular | Archetype | Matchup | `models.tab_arch_matchup` |
| 5 | Graph | Raw position | Ego | `models.gnn_pos_ego` |
| 6 | Graph | Raw position | Matchup | `models.gnn_pos_matchup` |
| 7 | Graph | Archetype | Ego | `models.gnn_arch_ego` |
| 8 | Graph | Archetype | Matchup | `models.gnn_arch_matchup` |

### 6.1 Tabular models (1–4)

Each snapshot is reduced to a fixed-length feature vector. The vector
always contains the focal team's formation (one-hot or learned
embedding) plus 11 features encoding each starting slot.

- **Raw-position variants** encode each slot as its raw Wyscout position
  label.
- **Archetype variants** encode each slot as the player's per-season
  archetype label (with a dedicated category for `NaN` archetypes).
- **Ego variants** see only the focal team's 11 slots + formation.
- **Matchup variants** concatenate the opponent's 11 slots + formation
  to the focal-team feature vector, so prediction is conditioned on the
  full pitch state.

A regularized gradient-boosted regressor (XGBoost / LightGBM) is fit on
the per-30 xG differential.

### 6.2 Graph models (5–8)

Each snapshot is encoded as one or two graphs:

- **Nodes**: the 11 starting players per team. Each node carries the
  slot's personnel encoding (raw position or archetype), plus a small
  set of spatial features derived from the average `position_x` /
  `position_y` of that slot in the raw formation log.
- **Edges**: tactical proximity edges within the focal team:
  defenders-to-defenders, midfielders-to-midfielders, attackers-to-
  attackers, plus formation-implied cross-line edges (e.g. fullback ↔
  winger, CM ↔ ST).
- **Matchup variants** additionally include the opponent's graph and a
  cross-team edge type connecting matched positional groups (focal CM ↔
  opponent CM, etc.).

A graph neural network (GCN or GAT, configurable in `configs/config.yaml`)
produces a team-level embedding for the focal team — and, in matchup
variants, a joint embedding from both team graphs — which is then
passed through a regression head.

### 6.3 Variants of interest

The 2×2×2 design enables clean attribution of performance gains:

- **Odd → even pairs** (1 vs 2, 3 vs 4, 5 vs 6, 7 vs 8): marginal
  contribution of conditioning on the opponent.
- **1 vs 3, 2 vs 4, 5 vs 7, 6 vs 8**: marginal contribution of archetype
  enrichment over raw positions.
- **1 vs 5, 2 vs 6, 3 vs 7, 4 vs 8**: marginal contribution of graph
  structure over tabular.
- **1 vs 8**: combined gain of all three modeling choices.

## 7. Evaluation Strategy

All models are evaluated via k-fold cross-validation on the snapshot
dataset, with folds blocked by `match_id` to ensure that windows from
the same match never appear on both sides of a train / test split.
Without this blocking, leakage between within-match windows would
inflate apparent performance.

Performance metrics:

- **MAE** and **RMSE** on the per-30 xG differential — primary
  regression metrics.
- **R²** — proportion of explained variance.
- **Sign accuracy** — fraction of windows in which the predicted sign of
  the xG differential matches the observed sign. This is the practically
  meaningful binary outcome: did the model correctly identify which
  team out-created the other during this window?

Results are reported as an 8-way table (one row per model) per metric.
Secondary-target sensitivity analysis repeats the comparison with
`home_goals_per_30` and `goals_home_minus_away_per_30` as the dependent
variable to confirm conclusions are not artifacts of the xG estimator.

## 8. Pipeline Summary

End-to-end build, in order:

1. `python -m archetypes.run_pipeline` &nbsp; → &nbsp;
   `data/processed/archetype_artifacts/player_archetype_map.parquet`
2. `python -m formations.run` &nbsp; → &nbsp;
   `data/processed/lineup_snapshots.parquet` &nbsp; *(the base training table)*
3. `python -m evaluation.run` &nbsp; → &nbsp; 8-model results table
   *(forthcoming)*

The first two stages are implemented and tested; the modeling and
evaluation stages are scaffolded and will be filled in as the model
library is built out.
