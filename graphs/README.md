# graphs/

Topology and visualization primitives for the GNN models. The assembled PyG
Data objects are built in
[`features/build_graphs_subs.py`](../features/build_graphs_subs.py); this
package holds the pieces it uses.

## Files

- [templates.py](templates.py): one canonical 11-slot template per formation
  string (4-4-2, 4-3-3, 3-5-2, and so on). Each slot has a label and template
  pitch coordinates. The intra-team edge set for a formation comes from a fixed
  set of tactical rules (CB-CB, GK to defenders, fullback to nearest wide
  attacker, and so on).
- [alignment.py](alignment.py): `align_lineup_to_template(formation, labels)`
  Hungarian-assigns each lineup slot to a template slot by minimizing squared
  distance between canonical position coordinates. The graph builder uses this
  to permute raw Wyscout slot order into template order.
- [visualize_half_subs.py](visualize_half_subs.py): pitch-relative rendering of
  the half-subs graphs (`draw_single_team_graph`, `draw_paired_graph`). In
  paired mode team 2 is rotated 180 degrees about the pitch center so opposing
  roles line up physically.

## Notes

- The edge rules are documented in the `templates.py` module docstring and are
  deterministic given a formation string. Matchup (cross-team) edges are built
  in `build_graphs_subs.py`, not here.
- `FORMATION_TEMPLATES` is the whitelist used by `filter_buildable_snapshots`;
  adding a new formation string here expands the buildable dataset.
- The demo notebook under [notebooks/](notebooks/) calls the visualization
  helpers directly.
