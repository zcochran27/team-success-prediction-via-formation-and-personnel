"""Canonical graph topology per formation string.

Each formation has a fixed list of 11 player "slots" with template pitch
coordinates (x = depth, 0=own goal -> 100=opponent goal; y = lateral,
0..80 across the pitch). Edges between slots are derived from
**rule-based tactical connectivity** (not a distance-based k-NN cut),
so the graph encodes the connections a soccer coach would draw on a
chalkboard rather than a geometric neighborhood.

The rule set (applied in order; see :func:`_build_edges`):

  1. GK <-> every defender (CB / LCB / RCB / LB / RB / LWB / RWB).
  2. All CBs mutually connected; LB/LWB <-> leftmost CB; RB/RWB <->
     rightmost CB.
  3. Every CB <-> every CDM (or, in 3-back systems with no CDM, every CB
     <-> the deepest central midfielder so the back line is still wired
     into the midfield).
  4. Every CDM and every CAM <-> every other midfielder.
  5. Every central midfielder (CM / CDM / CAM) <-> every other.
  6. Wide midfielders (LM / LAM / RM / RAM) <-> every central mid.
  7. Same-side wide-mid mesh (LM <-> LAM; RM <-> RAM).
  8. LB/LWB <-> closest of {LM, LAM, LW, LWF}; fallback to closest
     same-side central mid. Symmetric for right.
  9. Wingers (LW/LWF, RW/RWF) <-> same-side wide mids.
 10. Wingers <-> strikers and CAMs.
 11. Strikers mutually connected (forward-line cohesion).
 12. Strikers <-> all central mids.

A formation's graph is identical across every match in which that
formation is observed; different formations produce different graphs.

This module is purely topological -- it doesn't attach any per-player
features. The downstream GNN feature builder is responsible for mapping
real-lineup slots (1..11) onto template indices and attaching the
position / archetype / etc. features per node.
"""

from __future__ import annotations

import math


# Canonical goalkeeper position, prepended to every formation template as
# template index 0. The (6, 40) coordinates come from the empirical mode
# of GK locations in the raw formation log.
GK_POSITION: tuple[str, float, float] = ("GK", 6.0, 40.0)

# Label groupings used by the rule-based edge builder.
_CB_LABELS = frozenset({"CB", "LCB", "RCB"})
_LEFT_FULLBACK_LABELS = frozenset({"LB", "LWB"})
_RIGHT_FULLBACK_LABELS = frozenset({"RB", "RWB"})
_CENTRAL_MID_LABELS = frozenset({"CM", "CDM", "CAM"})
_LEFT_WIDE_MID_LABELS = frozenset({"LM", "LAM"})
_RIGHT_WIDE_MID_LABELS = frozenset({"RM", "RAM"})
_LEFT_WINGER_LABELS = frozenset({"LW", "LWF"})
_RIGHT_WINGER_LABELS = frozenset({"RW", "RWF"})
_STRIKER_LABELS = frozenset({"ST", "SS"})
# A "left-side wide attacker" candidate for fullback edges (rule 8).
_LEFT_WIDE_ATTACKER_LABELS = _LEFT_WIDE_MID_LABELS | _LEFT_WINGER_LABELS
_RIGHT_WIDE_ATTACKER_LABELS = _RIGHT_WIDE_MID_LABELS | _RIGHT_WINGER_LABELS


FORMATION_TEMPLATES: dict[str, list[tuple[str, float, float]]] = {
    "4-4-2": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("LM", 60.0, 68.0), ("CM", 60.0, 52.0), ("CM", 60.0, 28.0), ("RM", 60.0, 12.0),
        ("ST", 100.0, 52.0), ("ST", 100.0, 28.0),
    ],
    "4-3-3": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CM", 55.0, 56.0), ("CDM", 50.0, 40.0), ("CM", 55.0, 24.0),
        ("LW", 95.0, 68.0), ("ST", 100.0, 40.0), ("RW", 95.0, 12.0),
    ],
    "4-2-3-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CDM", 45.0, 52.0), ("CDM", 45.0, 28.0),
        ("LAM", 75.0, 64.0), ("CAM", 75.0, 40.0), ("RAM", 75.0, 16.0),
        ("ST", 100.0, 40.0),
    ],
    "3-5-2": [
        ("CB", 25.0, 56.0), ("CB", 25.0, 40.0), ("CB", 25.0, 24.0),
        ("LWB", 45.0, 72.0), ("CM", 55.0, 52.0), ("CDM", 50.0, 40.0),
        ("CM", 55.0, 28.0), ("RWB", 45.0, 8.0),
        ("ST", 100.0, 52.0), ("ST", 100.0, 28.0),
    ],
    "3-4-3": [
        ("CB", 25.0, 56.0), ("CB", 25.0, 40.0), ("CB", 25.0, 24.0),
        ("LM", 55.0, 68.0), ("CM", 55.0, 52.0), ("CM", 55.0, 28.0), ("RM", 55.0, 12.0),
        ("LW", 95.0, 68.0), ("ST", 100.0, 40.0), ("RW", 95.0, 12.0),
    ],
    "4-1-4-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CDM", 45.0, 40.0),
        ("LM", 70.0, 68.0), ("CM", 70.0, 52.0), ("CM", 70.0, 28.0), ("RM", 70.0, 12.0),
        ("ST", 100.0, 40.0),
    ],
    "4-3-2-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CM", 50.0, 56.0), ("CDM", 45.0, 40.0), ("CM", 50.0, 24.0),
        ("LAM", 80.0, 52.0), ("RAM", 80.0, 28.0),
        ("ST", 100.0, 40.0),
    ],
    "5-3-2": [
        ("LWB", 30.0, 72.0), ("CB", 22.0, 56.0), ("CB", 22.0, 40.0),
        ("CB", 22.0, 24.0), ("RWB", 30.0, 8.0),
        ("CM", 55.0, 56.0), ("CM", 55.0, 40.0), ("CM", 55.0, 24.0),
        ("ST", 100.0, 52.0), ("ST", 100.0, 28.0),
    ],
    "5-4-1": [
        ("LWB", 30.0, 72.0), ("CB", 22.0, 56.0), ("CB", 22.0, 40.0),
        ("CB", 22.0, 24.0), ("RWB", 30.0, 8.0),
        ("LM", 60.0, 68.0), ("CM", 60.0, 52.0), ("CM", 60.0, 28.0), ("RM", 60.0, 12.0),
        ("ST", 100.0, 40.0),
    ],
    "4-1-3-2": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CDM", 45.0, 40.0),
        ("LAM", 70.0, 64.0), ("CAM", 70.0, 40.0), ("RAM", 70.0, 16.0),
        ("ST", 95.0, 52.0), ("SS", 95.0, 28.0),
    ],
    "4-1-2-3": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CDM", 45.0, 40.0),
        ("CM", 60.0, 52.0), ("CM", 60.0, 28.0),
        ("LW", 95.0, 68.0), ("ST", 100.0, 40.0), ("RW", 95.0, 12.0),
    ],
    "3-4-1-2": [
        ("CB", 25.0, 56.0), ("CB", 25.0, 40.0), ("CB", 25.0, 24.0),
        ("LM", 50.0, 68.0), ("CM", 50.0, 52.0), ("CM", 50.0, 28.0), ("RM", 50.0, 12.0),
        ("CAM", 75.0, 40.0),
        ("ST", 95.0, 52.0), ("ST", 95.0, 28.0),
    ],
    "4-4-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("LM", 60.0, 68.0), ("CM", 60.0, 52.0), ("CM", 60.0, 28.0), ("RM", 60.0, 12.0),
        ("ST", 100.0, 40.0),
    ],
    "4-5-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("LM", 55.0, 68.0), ("CM", 55.0, 52.0), ("CDM", 50.0, 40.0), ("CM", 55.0, 28.0), ("RM", 55.0, 12.0),
        ("ST", 100.0, 40.0),
    ],
    "3-4-2-1": [
        ("CB", 25.0, 56.0), ("CB", 25.0, 40.0), ("CB", 25.0, 24.0),
        ("LM", 50.0, 68.0), ("CM", 50.0, 52.0), ("CM", 50.0, 28.0), ("RM", 50.0, 12.0),
        ("LAM", 80.0, 52.0), ("RAM", 80.0, 28.0),
        ("ST", 100.0, 40.0),
    ],
    "4-4-1-1": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("LW", 55.0, 68.0), ("CM", 55.0, 52.0), ("CM", 55.0, 28.0), ("RW", 55.0, 12.0),
        ("CAM", 80.0, 40.0),
        ("ST", 100.0, 40.0),
    ],
    "4-3-1-2": [
        ("LB", 25.0, 68.0), ("CB", 25.0, 52.0), ("CB", 25.0, 28.0), ("RB", 25.0, 12.0),
        ("CM", 50.0, 56.0), ("CDM", 45.0, 40.0), ("CM", 50.0, 24.0),
        ("CAM", 75.0, 40.0),
        ("ST", 95.0, 52.0), ("ST", 95.0, 28.0),
    ],
}


def formation_template(formation: str) -> list[tuple[str, float, float]]:
    """Return the 11-node template for ``formation``, GK as index 0.

    Raises ``KeyError`` if the formation string isn't recognized.
    """
    return [GK_POSITION] + FORMATION_TEMPLATES[formation]


def _dist(template: list[tuple[str, float, float]], i: int, j: int) -> float:
    return math.hypot(template[i][1] - template[j][1], template[i][2] - template[j][2])


def _build_edges(template: list[tuple[str, float, float]]) -> list[tuple[int, int]]:
    """Apply the 12 tactical rules described in the module docstring.

    Distances are *only* used inside rule 8 to break the
    "closest same-side wide attacker" tie -- not as a primary edge criterion
    anywhere else. The output is the sorted set of undirected ``(i, j)``
    edges with ``i < j`` and no self-loops.
    """
    labels = [t[0] for t in template]
    edges: set[tuple[int, int]] = set()

    def add(i: int, j: int) -> None:
        if i != j:
            edges.add((i, j) if i < j else (j, i))

    def has(*wanted: str) -> list[int]:
        wanted_set = set(wanted)
        return [i for i, lab in enumerate(labels) if lab in wanted_set]

    def has_in(group: frozenset) -> list[int]:
        return [i for i, lab in enumerate(labels) if lab in group]

    gk = has("GK")
    cb = has_in(_CB_LABELS)
    lb = has_in(_LEFT_FULLBACK_LABELS)
    rb = has_in(_RIGHT_FULLBACK_LABELS)
    cdm = has("CDM")
    cam = has("CAM")
    cm = has("CM")
    central_mid = has_in(_CENTRAL_MID_LABELS)
    left_mid = has_in(_LEFT_WIDE_MID_LABELS)
    right_mid = has_in(_RIGHT_WIDE_MID_LABELS)
    all_mid = central_mid + left_mid + right_mid
    lw = has_in(_LEFT_WINGER_LABELS)
    rw = has_in(_RIGHT_WINGER_LABELS)
    st = has_in(_STRIKER_LABELS)
    defenders = cb + lb + rb

    # Rule 1 -- GK <-> every defender.
    for g in gk:
        for d in defenders:
            add(g, d)

    # Rule 2 -- All CBs mutually connected; fullbacks to their outer CB.
    for i, a in enumerate(cb):
        for b in cb[i + 1:]:
            add(a, b)
    if cb:
        leftmost_cb = max(cb, key=lambda i: template[i][2])
        rightmost_cb = min(cb, key=lambda i: template[i][2])
        for f in lb:
            add(f, leftmost_cb)
        for f in rb:
            add(f, rightmost_cb)

    # Rule 3 -- Every CB <-> every CDM. If no CDM exists (3-back formations
    # like 3-4-3 / 3-4-1-2), fall back to the deepest central midfielder(s)
    # so the back line is still wired into the midfield.
    if cdm:
        for c in cb:
            for d in cdm:
                add(c, d)
    elif cb and central_mid:
        deepest_x = min(template[i][1] for i in central_mid)
        deepest_mids = [i for i in central_mid if template[i][1] == deepest_x]
        for c in cb:
            for d in deepest_mids:
                add(c, d)

    # Rule 4 -- CDMs and CAMs <-> every other midfielder.
    for hub in cdm + cam:
        for m in all_mid:
            add(hub, m)

    # Rule 5 -- Central-midfield mesh.
    for i, a in enumerate(central_mid):
        for b in central_mid[i + 1:]:
            add(a, b)

    # Rule 6 -- Wide mids <-> all central mids.
    for w in left_mid + right_mid:
        for c in central_mid:
            add(w, c)

    # Rule 7 -- Same-side wide-mid mesh.
    for i, a in enumerate(left_mid):
        for b in left_mid[i + 1:]:
            add(a, b)
    for i, a in enumerate(right_mid):
        for b in right_mid[i + 1:]:
            add(a, b)

    # Rule 8 -- Each fullback to nearest same-side wide attacker.
    # If the formation has no LM/LAM/LW/LWF (or symmetric), fall back to the
    # nearest same-side central midfielder so wing-backs in formations like
    # 3-5-2 aren't stranded with only the back-line and the GK.
    def fb_target(fb_idx: int, primary: list[int], fallback: list[int]) -> int | None:
        candidates = primary if primary else fallback
        if not candidates:
            return None
        return min(candidates, key=lambda c: _dist(template, fb_idx, c))

    left_primary = left_mid + lw
    right_primary = right_mid + rw
    left_fallback = [i for i in central_mid if template[i][2] >= 40]
    right_fallback = [i for i in central_mid if template[i][2] <= 40]
    for f in lb:
        t = fb_target(f, left_primary, left_fallback)
        if t is not None:
            add(f, t)
    for f in rb:
        t = fb_target(f, right_primary, right_fallback)
        if t is not None:
            add(f, t)

    # Rule 9 -- Wingers <-> same-side wide mids.
    for w in lw:
        for m in left_mid:
            add(w, m)
    for w in rw:
        for m in right_mid:
            add(w, m)

    # Rule 10 -- Wingers <-> strikers and CAMs.
    for w in lw + rw:
        for s in st:
            add(w, s)
        for c in cam:
            add(w, c)

    # Rule 11 -- Strikers mutually connected.
    for i, a in enumerate(st):
        for b in st[i + 1:]:
            add(a, b)

    # Rule 12 -- Strikers <-> all central mids.
    for s in st:
        for c in central_mid:
            add(s, c)

    return sorted(edges)


def formation_edges(formation: str) -> list[tuple[int, int]]:
    """Return the canonical undirected edge list for ``formation``.

    Edges are built from a fixed set of tactical rules (see the module
    docstring); the graph for a given formation string is deterministic
    and identical across every match in which that formation is observed.
    Node indices are 0..10 matching the order in :func:`formation_template`
    (GK = 0, then the 10 outfield slots in template order).
    """
    return _build_edges(formation_template(formation))


def formation_graph(formation: str):
    """Return ``(nodes, edges)`` for ``formation``."""
    nodes = formation_template(formation)
    return nodes, _build_edges(nodes)


def all_formation_graphs() -> dict[str, tuple[list, list]]:
    """Return ``{formation_string: (nodes, edges)}`` for every template."""
    return {f: formation_graph(f) for f in FORMATION_TEMPLATES}
