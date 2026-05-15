"""Inter-team matchup edges: who-marks-whom across the two starting elevens.

Each position label is bucketed into a tactical *zone* defined by a (lane,
band) pair: lanes are LEFT / CENTER / RIGHT, bands are DEFENSE / MIDFIELD /
ATTACK plus GK. Inter-team edges then connect zones that physically share
the same lane on the pitch when the two teams face each other.

Lane-mirror reminder: when team1 attacks toward team2's goal, team1's *left*
lane is the same physical strip of grass as team2's *right* lane. So
team1's left attackers face team2's right defenders, and vice versa.

Zone-pair rules (each entry is applied symmetrically so one entry covers
both directions):

  1. CA <-> CD   strikers vs center backs
  2. LA <-> RD   left attackers vs right backs (same physical lane)
  3. RA <-> LD   right attackers vs left backs
  4. LM <-> RD   left wide-mids vs right backs
  5. RM <-> LD   right wide-mids vs left backs
  6. LM <-> RA   left wide-mids vs right wingers
  7. RM <-> LA   right wide-mids vs left wingers
  8. LM <-> RM   wide-mid mirror (lane-aligned wing battle)
  9. CM <-> CM   central midfield battle
 10. CA <-> CM   strikers also engage opposing center mids
 11. CD <-> LA   inverted left wingers driving at center backs
 12. CD <-> RA   inverted right wingers driving at center backs
 13. CD <-> CM   CMs / CAMs arriving between the lines or in the box

Side-aware rules (require the formation templates so left/right strikers
and back-three center-back slots can be identified by template y):

 14. LB <-> right ST, RB <-> left ST  (only when the opposing team has
     exactly 2 strikers; "right" = lower template y, "left" = higher).
     This refines the generic FB <-> CA matchup so each full-back picks
     up the lane-mirror striker rather than every striker.

 15. Back-three winger refinement: when a team plays 3 CBs, the CD <->
     opposing-winger edges from rules 11/12 collapse to a single
     lane-mirror outer-CB connection per winger -- LCB picks up the
     opposing right winger (RA), RCB picks up the opposing left winger
     (LA), and the middle CB stays out of the wide channel.

GK has no inter-team edges -- it is the last line of defense, not a
man-marker.

Edge indexing: when ``formation_*`` arguments are supplied, the caller
is expected to pass labels in *template-slot order* (i.e. already aligned
via :mod:`graphs.alignment`). The returned edges then index nodes in
template-slot order. If formations are omitted, edges only reflect the
zone-pair rules and inherit whatever order the input labels are in.
"""

from __future__ import annotations

from collections.abc import Sequence

from graphs.templates import formation_template


_ZONE_LABELS: dict[str, frozenset[str]] = {
    "GK": frozenset({"GK"}),
    "LD": frozenset({"LB", "LWB"}),
    "CD": frozenset({"CB", "LCB", "RCB"}),
    "RD": frozenset({"RB", "RWB"}),
    "LM": frozenset({"LM", "LAM"}),
    "CM": frozenset({"CM", "CDM", "CAM"}),
    "RM": frozenset({"RM", "RAM"}),
    "LA": frozenset({"LW", "LWF"}),
    "CA": frozenset({"ST", "SS"}),
    "RA": frozenset({"RW", "RWF"}),
}


_MATCHUP_PAIRS: tuple[tuple[str, str], ...] = (
    ("CA", "CD"),
    ("LA", "RD"),
    ("RA", "LD"),
    ("LM", "RD"),
    ("RM", "LD"),
    ("LM", "RA"),
    ("RM", "LA"),
    ("LM", "RM"),
    ("CM", "CM"),
    ("CA", "CM"),
    ("CD", "LA"),
    ("CD", "RA"),
    ("CD", "CM"),
)


def zone_of(label: str) -> str:
    """Return the matchup zone (e.g. ``'CA'``, ``'LD'``) for a position label."""
    for zone, members in _ZONE_LABELS.items():
        if label in members:
            return zone
    raise KeyError(f"position label {label!r} not in any matchup zone")


def build_intermatch_edges(
    team1_labels: Sequence[str],
    team2_labels: Sequence[str],
    team2_offset: int = 11,
    *,
    formation_team1: str | None = None,
    formation_team2: str | None = None,
) -> list[tuple[int, int]]:
    """Return the undirected inter-team edges as global ``(i, j)`` pairs.

    Team-1 nodes have indices ``0..len(team1_labels) - 1``; team-2 nodes are
    offset by ``team2_offset`` (default 11 so they sit at indices ``11..21``).
    Each pair in :data:`_MATCHUP_PAIRS` contributes the cross-product of its
    two zones in both directions (team1.A <-> team2.B and team1.B <-> team2.A).

    When both ``formation_team1`` and ``formation_team2`` are supplied, the
    labels are assumed to be in *template-slot order* (i.e. aligned via
    :func:`graphs.alignment.align_lineup_to_template`) and the side-aware
    striker rule is also applied: each team's full-backs pick up the
    opposing lane-mirror striker when the opposing team has exactly 2
    strikers.
    """
    if len(team1_labels) != 11 or len(team2_labels) != 11:
        raise ValueError(
            f"expected 11 labels per team, got {len(team1_labels)} / {len(team2_labels)}"
        )

    def bucket(labels: Sequence[str], offset: int) -> dict[str, list[int]]:
        buckets: dict[str, list[int]] = {z: [] for z in _ZONE_LABELS}
        for i, lab in enumerate(labels):
            buckets[zone_of(lab)].append(i + offset)
        return buckets

    t1 = bucket(team1_labels, 0)
    t2 = bucket(team2_labels, team2_offset)

    edges: set[tuple[int, int]] = set()
    for z_a, z_b in _MATCHUP_PAIRS:
        for i in t1[z_a]:
            for j in t2[z_b]:
                edges.add((i, j) if i < j else (j, i))
        for i in t1[z_b]:
            for j in t2[z_a]:
                edges.add((i, j) if i < j else (j, i))

    if formation_team1 is not None and formation_team2 is not None:
        _add_lateral_striker_edges(
            edges,
            formation_team1=formation_team1,
            formation_team2=formation_team2,
            team2_offset=team2_offset,
        )
        _refine_back_three_winger_edges(
            edges,
            formation_team1=formation_team1,
            team1_labels=team1_labels,
            formation_team2=formation_team2,
            team2_labels=team2_labels,
            team2_offset=team2_offset,
        )

    return sorted(edges)


def _team_striker_pair(formation: str) -> tuple[int, int] | None:
    """Return ``(left_st_slot, right_st_slot)`` if exactly 2 strikers, else ``None``.

    "Left ST" = higher template y; "right ST" = lower template y.
    """
    tmpl = formation_template(formation)
    strikers = [(k, y) for k, (lab, _, y) in enumerate(tmpl) if lab in _ZONE_LABELS["CA"]]
    if len(strikers) != 2:
        return None
    strikers.sort(key=lambda kv: kv[1], reverse=True)
    return strikers[0][0], strikers[1][0]


def _team_fullback_slots(formation: str) -> tuple[list[int], list[int]]:
    """Return ``(left_fullback_slots, right_fullback_slots)`` per template."""
    tmpl = formation_template(formation)
    lb = [k for k, (lab, _, _) in enumerate(tmpl) if lab in _ZONE_LABELS["LD"]]
    rb = [k for k, (lab, _, _) in enumerate(tmpl) if lab in _ZONE_LABELS["RD"]]
    return lb, rb


def _undirected(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _team_back_three_cb_slots(formation: str) -> tuple[int, int, int] | None:
    """Return ``(left, middle, right)`` CB template slots if exactly 3 CBs, else ``None``.

    "Left" CB = highest template y; "right" CB = lowest; middle is in between.
    """
    tmpl = formation_template(formation)
    cbs = [(k, y) for k, (lab, _, y) in enumerate(tmpl) if lab in _ZONE_LABELS["CD"]]
    if len(cbs) != 3:
        return None
    cbs.sort(key=lambda kv: kv[1], reverse=True)
    return cbs[0][0], cbs[1][0], cbs[2][0]


def _refine_back_three_winger_edges(
    edges: set[tuple[int, int]],
    *,
    formation_team1: str,
    team1_labels: Sequence[str],
    formation_team2: str,
    team2_labels: Sequence[str],
    team2_offset: int,
) -> None:
    """In back-three setups, collapse CD<->opposing-winger edges to the lane-mirror outer CB.

    Removes every CD<->winger edge for the back-three team, then re-adds the
    single lane-mirror edge per opposing winger (T1.LCB picks up T2.RA,
    T1.RCB picks up T2.LA, and vice versa). Middle CBs and wrong-side outer
    CBs are kept out of the wide channel.
    """
    t1_back3 = _team_back_three_cb_slots(formation_team1)
    t2_back3 = _team_back_three_cb_slots(formation_team2)

    def all_cb_slots(formation: str) -> list[int]:
        return [
            k for k, (lab, _, _) in enumerate(formation_template(formation))
            if lab in _ZONE_LABELS["CD"]
        ]

    if t1_back3 is not None:
        left_cb_t1, _middle_t1, right_cb_t1 = t1_back3
        for i, lab in enumerate(team2_labels):
            global_winger = i + team2_offset
            if lab in _ZONE_LABELS["LA"]:
                for cb in all_cb_slots(formation_team1):
                    edges.discard(_undirected(cb, global_winger))
                edges.add(_undirected(right_cb_t1, global_winger))
            elif lab in _ZONE_LABELS["RA"]:
                for cb in all_cb_slots(formation_team1):
                    edges.discard(_undirected(cb, global_winger))
                edges.add(_undirected(left_cb_t1, global_winger))

    if t2_back3 is not None:
        left_cb_t2_slot, _middle_t2, right_cb_t2_slot = t2_back3
        for i, lab in enumerate(team1_labels):
            if lab in _ZONE_LABELS["LA"]:
                for cb_local in all_cb_slots(formation_team2):
                    edges.discard(_undirected(i, cb_local + team2_offset))
                edges.add(_undirected(i, right_cb_t2_slot + team2_offset))
            elif lab in _ZONE_LABELS["RA"]:
                for cb_local in all_cb_slots(formation_team2):
                    edges.discard(_undirected(i, cb_local + team2_offset))
                edges.add(_undirected(i, left_cb_t2_slot + team2_offset))


def _add_lateral_striker_edges(
    edges: set[tuple[int, int]],
    *,
    formation_team1: str,
    formation_team2: str,
    team2_offset: int,
) -> None:
    """LB <-> opposing right ST, RB <-> opposing left ST (when 2 strikers).

    Each team's full-back template slots are looked up by label; the
    opposing team's striker pair is split into left/right by template y.
    Edges use template-slot indices and assume labels passed to
    :func:`build_intermatch_edges` were in template-slot order.
    """
    t1_fbs = _team_fullback_slots(formation_team1)
    t2_fbs = _team_fullback_slots(formation_team2)
    t1_strikers = _team_striker_pair(formation_team1)
    t2_strikers = _team_striker_pair(formation_team2)

    if t2_strikers is not None:
        left_st_t2, right_st_t2 = t2_strikers
        right_st_g = right_st_t2 + team2_offset
        left_st_g = left_st_t2 + team2_offset
        for lb in t1_fbs[0]:
            edges.add(_undirected(lb, right_st_g))
        for rb in t1_fbs[1]:
            edges.add(_undirected(rb, left_st_g))

    if t1_strikers is not None:
        left_st_t1, right_st_t1 = t1_strikers
        for lb_local in t2_fbs[0]:
            edges.add(_undirected(right_st_t1, lb_local + team2_offset))
        for rb_local in t2_fbs[1]:
            edges.add(_undirected(left_st_t1, rb_local + team2_offset))
