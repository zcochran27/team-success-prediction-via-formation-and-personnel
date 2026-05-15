"""Tests for :mod:`graphs.matchups`."""

from __future__ import annotations

import pytest

from graphs.matchups import build_intermatch_edges, zone_of


# Two 4-3-3 lineups in template-slot order (GK first):
#   0: GK  1: LB  2: CB  3: CB  4: RB
#   5: CM  6: CDM  7: CM
#   8: LW  9: ST  10: RW
_T1 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
_T2 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]


def _edges() -> set[tuple[int, int]]:
    return set(build_intermatch_edges(_T1, _T2, team2_offset=11))


def test_zone_of_maps_known_labels() -> None:
    assert zone_of("GK") == "GK"
    assert zone_of("LB") == "LD" and zone_of("LWB") == "LD"
    assert zone_of("LCB") == "CD" and zone_of("RCB") == "CD"
    assert zone_of("CDM") == "CM" and zone_of("CAM") == "CM"
    assert zone_of("SS") == "CA"
    assert zone_of("LWF") == "LA"


def test_zone_of_rejects_unknown_label() -> None:
    with pytest.raises(KeyError):
        zone_of("NOPE")


def test_no_self_loops_or_within_team_edges_in_intermatch_set() -> None:
    edges = _edges()
    for i, j in edges:
        assert i != j
        # team1 indices 0..10, team2 indices 11..21 -- every inter-team edge
        # must straddle the team boundary.
        assert (i < 11) != (j < 11)


def test_gk_has_no_inter_team_edges() -> None:
    edges = _edges()
    gk1, gk2 = 0, 11  # GK is template slot 0 in both teams
    for i, j in edges:
        assert gk1 not in (i, j)
        assert gk2 not in (i, j)


def test_strikers_face_opposing_center_backs() -> None:
    edges = _edges()
    # T1 ST = node 9; T1 CBs = nodes 2, 3
    # T2 ST = node 9 + 11 = 20; T2 CBs = nodes 2+11=13, 3+11=14
    assert (9, 13) in edges and (9, 14) in edges
    assert (2, 20) in edges and (3, 20) in edges


def test_left_winger_faces_opposing_right_back_not_left_back() -> None:
    edges = _edges()
    # T1 LW = 8, T1 LB = 1, T1 RB = 4; T2 LB = 12, T2 LW = 19, T2 RB = 15.
    # Lane mirror: T1 LW (left lane) <-> T2 RB (T2's right = T1's left lane).
    assert (8, 15) in edges
    assert (8, 12) not in edges  # left winger should not connect to opposing left back
    # Symmetric mirror: T1 RB <-> T2 LW.
    assert (4, 19) in edges
    # And the same-side-but-wrong-direction non-edges:
    assert (1, 19) not in edges  # both on left of their teams = opposite physical lanes
    assert (4, 15) not in edges  # both on right of their teams = opposite physical lanes


def test_center_mids_battle_centrally() -> None:
    edges = _edges()
    # T1 CMs = {5, 6, 7}, T2 CMs = {16, 17, 18}
    for i in (5, 6, 7):
        for j in (16, 17, 18):
            assert (min(i, j), max(i, j)) in edges


def test_intermatch_edges_are_deterministic_and_sorted() -> None:
    e1 = build_intermatch_edges(_T1, _T2, team2_offset=11)
    e2 = build_intermatch_edges(_T1, _T2, team2_offset=11)
    assert e1 == e2 == sorted(e1)


def test_inverted_wingers_face_center_backs_group_b() -> None:
    """B1/B2: T1's left and right attackers now connect to opposing CBs."""
    edges = _edges()
    # T1 LW = 8, T1 RW = 10; T2 CBs = nodes 13, 14
    assert (8, 13) in edges and (8, 14) in edges
    assert (10, 13) in edges and (10, 14) in edges
    # Symmetric: T1 CBs <-> T2 wingers.
    assert (2, 19) in edges and (3, 19) in edges  # T2 LW = 19
    assert (2, 21) in edges and (3, 21) in edges  # T2 RW = 21


def test_center_backs_connect_to_opposing_central_mids_group_c() -> None:
    """C1: CD <-> opposing CM connects late runners and #10s to the back line."""
    edges = _edges()
    # T1 CBs = {2, 3}, T2 CMs = {16, 17, 18}
    for cb in (2, 3):
        for cm in (16, 17, 18):
            assert (cb, cm) in edges
    # Symmetric: T2 CBs <-> T1 CMs.
    for cb in (13, 14):
        for cm in (5, 6, 7):
            assert (cm, cb) in edges


def test_side_aware_striker_rule_fires_only_with_two_strikers() -> None:
    """LB <-> opposing right ST and RB <-> opposing left ST when 2 STs."""
    # 4-4-2 has STs at template y=52 (left) and y=28 (right). LB at slot 1,
    # RB at slot 4. The labels below are in template-slot order.
    t1_442 = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    t2_442 = ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"]
    edges = set(build_intermatch_edges(
        t1_442, t2_442, team2_offset=11,
        formation_team1="4-4-2", formation_team2="4-4-2",
    ))
    # T1 LB (1) <-> T2 right ST (slot 10 -> global 21).
    assert (1, 21) in edges
    # T1 RB (4) <-> T2 left ST (slot 9 -> global 20).
    assert (4, 20) in edges
    # T2 LB (12) <-> T1 right ST (10).
    assert (10, 12) in edges
    # T2 RB (15) <-> T1 left ST (9).
    assert (9, 15) in edges


def test_back_three_collapses_winger_cb_edges_to_lane_mirror() -> None:
    """3-5-2 vs 4-3-3: T1's back-3 means only outer CBs face T2's wingers (lane-mirror)."""
    # 3-5-2 template after GK prepend: CBs at slots 1 (y=56), 2 (y=40), 3 (y=24).
    # Left CB = slot 1, middle = slot 2, right CB = slot 3.
    t1_352 = ["GK", "CB", "CB", "CB", "LWB", "CM", "CDM", "CM", "RWB", "ST", "ST"]
    # 4-3-3 has wingers at slots 8 (LW) and 10 (RW). T2 globals: 8+11=19 (LW), 10+11=21 (RW).
    t2_433 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]

    edges = set(build_intermatch_edges(
        t1_352, t2_433, team2_offset=11,
        formation_team1="3-5-2", formation_team2="4-3-3",
    ))

    # Lane-mirror edges PRESENT:
    #   T1 right CB (slot 3) <-> T2 LW (19)
    #   T1 left CB (slot 1)  <-> T2 RW (21)
    assert (3, 19) in edges
    assert (1, 21) in edges

    # All OTHER CB<->winger combinations REMOVED (would have been added by B1/B2).
    assert (1, 19) not in edges  # left CB <-> left winger: wrong physical lane
    assert (2, 19) not in edges  # middle CB <-> left winger
    assert (2, 21) not in edges  # middle CB <-> right winger
    assert (3, 21) not in edges  # right CB <-> right winger: wrong physical lane

    # Sanity: CD<->striker (rule 1) and CD<->CM (C1) are NOT pruned for the
    # back-three team -- middle CB still marks central runners and strikers.
    t2_st = 9 + 11  # T2 ST at slot 9 -> 20
    assert (2, t2_st) in edges  # middle CB <-> opposing striker
    for cm in (16, 17, 18):  # T2 CMs at slots 5/6/7 -> globals 16/17/18
        assert (2, cm) in edges  # middle CB <-> opposing CMs


def test_back_three_refinement_does_not_fire_for_back_four() -> None:
    """4-3-3 has 2 CBs; B1/B2 stay all-to-all between CBs and wingers."""
    t1_433 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2_433 = list(t1_433)
    edges = set(build_intermatch_edges(
        t1_433, t2_433, team2_offset=11,
        formation_team1="4-3-3", formation_team2="4-3-3",
    ))
    # Both T1 CBs (slots 2, 3) connect to both T2 wingers (globals 19, 21).
    for cb in (2, 3):
        for w in (19, 21):
            assert (cb, w) in edges


def test_side_aware_rule_does_not_fire_for_lone_striker() -> None:
    """4-3-3 has 1 ST; the LB / RB shouldn't get extra striker edges."""
    t1_433 = ["GK", "LB", "CB", "CB", "RB", "CM", "CDM", "CM", "LW", "ST", "RW"]
    t2_433 = list(t1_433)
    base = set(build_intermatch_edges(t1_433, t2_433, team2_offset=11))
    with_form = set(build_intermatch_edges(
        t1_433, t2_433, team2_offset=11,
        formation_team1="4-3-3", formation_team2="4-3-3",
    ))
    assert base == with_form


def test_offset_shifts_team2_indices() -> None:
    base = set(build_intermatch_edges(_T1, _T2, team2_offset=11))
    shifted = set(build_intermatch_edges(_T1, _T2, team2_offset=100))
    # Every team2 index should be 89 larger in the shifted variant.
    base_shifted = {
        (i, j + 89) if j >= 11 else (i + 89, j) if i >= 11 else (i, j)
        for i, j in base
    }
    # Normalize ordering after the shift.
    base_shifted = {(min(a, b), max(a, b)) for a, b in base_shifted}
    assert base_shifted == shifted
