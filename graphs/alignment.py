"""Permute lineup-snapshot slots into formation template-slot order.

The raw Wyscout snapshot data lists 11 players in an order that does not
match :func:`graphs.templates.formation_template` -- e.g. a 4-4-2 lineup
might arrive as ``[GK, RB, LB, RCB, LCB, CM, RW, CM, ST, SS, LW]`` while
the 4-4-2 template enumerates slots as ``[GK, LB, CB, CB, RB, LM, CM, CM,
RM, ST, ST]``. Without realignment the visualization and the GNN both
end up drawing or wiring nodes at the wrong template slot.

:func:`align_lineup_to_template` solves a one-to-one assignment between
the 11 lineup slots and the 11 template slots, minimizing the sum of
squared Euclidean distances between each template slot's ``(x, y)`` and
a canonical ``(x, y)`` for the lineup slot's position label. Hungarian
assignment (:func:`scipy.optimize.linear_sum_assignment`) gives the
globally optimal permutation in O(n^3); for n=11 this is essentially
free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from graphs.templates import formation_template


# Canonical (x, y) per position label, in the same coordinate frame as
# graphs.templates. Values are picked to match the typical pitch location
# of each role; ties between similar roles (e.g. LM vs LW) are resolved
# by depth (x coordinate).
_DEFAULT_POSITION_XY: dict[str, tuple[float, float]] = {
    "GK":  (6.0, 40.0),
    "LB":  (25.0, 68.0),
    "LWB": (30.0, 72.0),
    "LCB": (25.0, 56.0),
    "CB":  (25.0, 40.0),
    "RCB": (25.0, 24.0),
    "RB":  (25.0, 12.0),
    "RWB": (30.0, 8.0),
    "CDM": (45.0, 40.0),
    "CM":  (60.0, 40.0),
    "CAM": (75.0, 40.0),
    "LM":  (60.0, 68.0),
    "LAM": (75.0, 64.0),
    "RM":  (60.0, 12.0),
    "RAM": (75.0, 16.0),
    "LW":  (95.0, 68.0),
    "LWF": (95.0, 68.0),
    "RW":  (95.0, 12.0),
    "RWF": (95.0, 12.0),
    "ST":  (100.0, 40.0),
    "SS":  (95.0, 40.0),
}


def align_lineup_to_template(
    formation: str,
    lineup_labels: Sequence[str],
) -> list[int]:
    """Return ``perm`` such that ``lineup_labels[perm[k]]`` is the lineup slot
    best matching template slot ``k``.

    The assignment minimises the total squared distance between each
    template slot's ``(x, y)`` (from :func:`formation_template`) and the
    canonical ``(x, y)`` for the lineup slot's position label.
    """
    if len(lineup_labels) != 11:
        raise ValueError(f"expected 11 lineup labels, got {len(lineup_labels)}")

    template = formation_template(formation)
    cost = np.empty((11, 11), dtype=np.float64)
    for k in range(11):
        _, tx, ty = template[k]
        for i in range(11):
            lx, ly = _DEFAULT_POSITION_XY.get(lineup_labels[i], (50.0, 40.0))
            cost[k, i] = (tx - lx) ** 2 + (ty - ly) ** 2

    rows, cols = linear_sum_assignment(cost)
    perm = [0] * 11
    for k, i in zip(rows.tolist(), cols.tolist()):
        perm[k] = i
    return perm


def aligned(labels: Sequence[str], perm: Sequence[int]) -> list[str]:
    """Convenience: return ``[labels[perm[k]] for k in range(11)]``."""
    return [labels[perm[k]] for k in range(11)]
