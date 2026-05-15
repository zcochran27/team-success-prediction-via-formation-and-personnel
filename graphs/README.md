# graphs/

Topology + visualization primitives for the GNN models. This package
exposes only the *building blocks*; assembled PyG `Data` objects are
produced by [`features/build_graphs.py`](../features/build_graphs.py).

## Files

| Module | Role |
|---|---|
| [templates.py](templates.py) | One canonical 11-slot template per formation string (`4-4-2`, `4-3-3`, `3-5-2`, …). Each slot has a label (`GK`, `LB`, `CDM`, …) and template pitch coordinates. The intra-team edge set for a formation is derived from a fixed set of 17 tactical rules (CB-CB, GK-defenders, fullback ↔ nearest wide attacker, etc.). |
| [matchups.py](matchups.py) | Inter-team matchup edges. Buckets each player's position into a zone (`LD`, `CD`, `LM`, `CA`, …) and connects zones that share a physical lane on the pitch (`LM ↔ RD`, `CA ↔ CD`, etc.). Includes side-aware refinements for fullback ↔ opposing striker and back-three ↔ opposing winger. |
| [alignment.py](alignment.py) | `align_lineup_to_template(formation, labels)` — Hungarian-assigns each lineup slot to a template slot by minimizing squared Euclidean distance between canonical position coordinates. The GNN dataset uses this to permute raw Wyscout slot order into template order before building graphs. |
| [visualize.py](visualize.py) | Pitch-relative rendering of paired team graphs (`draw_paired_graph`). Team 2 is rotated 180° about the pitch center so the lane-mirror semantics of the inter-team edges line up physically. |

## Where it's used

- [`features.build_graphs.build_snapshot_graph`](../features/build_graphs.py) calls `templates.formation_edges`, `matchups.build_intermatch_edges`, and embedding-vocab helpers.
- [`features.dataset.LineupSnapshotDataset`](../features/dataset.py) calls `alignment.align_lineup_to_template` once per side per row.
- The notebooks under [notebooks/](notebooks/) call `visualize.draw_paired_graph` directly.

## Notes

- The 17 edge rules and the 13 matchup zone-pair rules are documented in the module docstrings; both are deterministic given a formation string (and labels for matchups).
- `FORMATION_TEMPLATES` is the whitelist used by `filter_buildable_snapshots` — adding a new formation string here automatically expands the dataset.
