"""Model implementations for the 2x2 experimental matrix.

- Model 1: :mod:`models.baseline_positions` (tabular, raw positions)
- Model 2: :mod:`models.baseline_archetypes` (tabular, archetypes)
- Model 3: :mod:`models.gnn_positions` (GNN, raw positional node features)
- Model 4: :mod:`models.gnn_archetypes` (GNN, archetype node features)

All four models expose ``fit`` / ``predict`` interfaces compatible with
:mod:`evaluation.cross_validate`.
"""
