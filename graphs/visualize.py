"""Plot paired team graphs on a pitch background.

Team-1 nodes are placed at their formation template coordinates; team-2
nodes are rotated 180 degrees about the pitch center so the two teams
face each other across the halfway line. Intra-team edges (from each
formation's tactical rules) and inter-team matchup edges (from
:mod:`graphs.matchups`) are drawn in different colors so the structure
of the combined graph is visible at a glance.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.patches as mpatches
from matplotlib.axes import Axes

from graphs.alignment import aligned, align_lineup_to_template
from graphs.matchups import build_intermatch_edges
from graphs.templates import formation_edges, formation_template


# Pitch extent in the same coordinate frame as graphs.templates.FORMATION_TEMPLATES.
PITCH_X = 100.0
PITCH_Y = 80.0

# Default horizontal gap between team1's back-line and team2's front-line on
# the rendered pitch -- team2 is shifted right until its deepest attackers
# sit this many units past team1's deepest defenders.
DEFAULT_TEAM_GAP = 5.0

_TEAM1_COLOR = "#1f78b4"   # blue
_TEAM2_COLOR = "#e31a1c"   # red
_INTRA_T1_EDGE = "#1f78b4"
_INTRA_T2_EDGE = "#e31a1c"
_INTER_EDGE = "#ff7f00"    # orange

_DEFENDER_LABELS = frozenset({"CB", "LCB", "RCB", "LB", "LWB", "RB", "RWB"})
_ATTACKER_LABELS = frozenset({"ST", "SS", "LW", "LWF", "RW", "RWF"})


def flip_to_team2(xy: tuple[float, float]) -> tuple[float, float]:
    """Rotate a template ``(x, y)`` 180 degrees about the pitch center.

    Used so that team-2's right lane sits on the same physical strip of
    grass as team-1's left lane (and vice versa), matching the lane-mirror
    semantics used by the inter-team matchup edges.
    """
    return PITCH_X - xy[0], PITCH_Y - xy[1]


def auto_team2_shift(
    formation_team1: str,
    formation_team2: str,
    gap: float = DEFAULT_TEAM_GAP,
) -> float:
    """Horizontal shift to apply to team-2 after the 180° flip.

    Picks the smallest shift that places team-2's deepest attacker
    ``gap`` units past team-1's deepest defender. Both ends are read
    from the formation templates so the result adapts to whichever
    formations the two teams are playing.
    """
    t1_def_x = max(
        x for label, x, _ in formation_template(formation_team1) if label in _DEFENDER_LABELS
    )
    t2_atk_flipped_min = min(
        PITCH_X - x
        for label, x, _ in formation_template(formation_team2)
        if label in _ATTACKER_LABELS
    )
    return (t1_def_x + gap) - t2_atk_flipped_min


def _draw_pitch(ax: Axes, x_max: float) -> None:
    width = x_max + 10
    ax.add_patch(
        mpatches.Rectangle(
            (-5, -5),
            width,
            PITCH_Y + 10,
            facecolor="#e6f2e6",
            edgecolor="gray",
            lw=0.5,
            zorder=0,
        )
    )
    ax.set_xlim(-6, x_max + 6)
    ax.set_ylim(-6, PITCH_Y + 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paired_graph(
    ax: Axes,
    formation_team1: str,
    position_labels_team1: Sequence[str],
    formation_team2: str,
    position_labels_team2: Sequence[str],
    title: str | None = None,
    display_labels_team1: Sequence[str] | None = None,
    display_labels_team2: Sequence[str] | None = None,
    team2_x_shift: float | None = None,
) -> None:
    """Render the 22-node paired graph for one snapshot on ``ax``.

    The raw lineup slot order does not match template-slot order, so each
    team's labels are permuted into template-slot order via
    :func:`graphs.alignment.align_lineup_to_template` before anything is
    plotted. After alignment, node positions come directly from each
    team's template; team-2 coords are flipped via :func:`flip_to_team2`.

    ``position_labels_*`` are used both for the alignment (so the
    plotted node positions are right) and for the matchup-zone bucketing.
    ``display_labels_*`` -- when provided -- override only the text drawn
    under each node (e.g. archetype strings); they are reordered with the
    same permutation as the position labels.
    """
    tmpl1 = formation_template(formation_team1)
    tmpl2 = formation_template(formation_team2)

    shift = (
        team2_x_shift
        if team2_x_shift is not None
        else auto_team2_shift(formation_team1, formation_team2)
    )

    pos: list[tuple[float, float]] = [(t[1], t[2]) for t in tmpl1]
    for t in tmpl2:
        fx, fy = flip_to_team2((t[1], t[2]))
        pos.append((fx + shift, fy))

    perm1 = align_lineup_to_template(formation_team1, position_labels_team1)
    perm2 = align_lineup_to_template(formation_team2, position_labels_team2)
    aligned_pos1 = aligned(position_labels_team1, perm1)
    aligned_pos2 = aligned(position_labels_team2, perm2)

    intra1 = formation_edges(formation_team1)
    intra2 = [(i + 11, j + 11) for i, j in formation_edges(formation_team2)]
    inter = build_intermatch_edges(
        aligned_pos1,
        aligned_pos2,
        team2_offset=11,
        formation_team1=formation_team1,
        formation_team2=formation_team2,
    )

    x_max = max(p[0] for p in pos)
    _draw_pitch(ax, x_max=x_max)

    for i, j in intra1:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.plot([x1, x2], [y1, y2], color=_INTRA_T1_EDGE, lw=1.0, alpha=0.55, zorder=1)
    for i, j in intra2:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.plot([x1, x2], [y1, y2], color=_INTRA_T2_EDGE, lw=1.0, alpha=0.55, zorder=1)
    for i, j in inter:
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        ax.plot([x1, x2], [y1, y2], color=_INTER_EDGE, lw=0.9, alpha=0.45, zorder=2)

    display1 = (
        aligned(list(display_labels_team1), perm1)
        if display_labels_team1 is not None
        else aligned_pos1
    )
    display2 = (
        aligned(list(display_labels_team2), perm2)
        if display_labels_team2 is not None
        else aligned_pos2
    )
    labels_all = display1 + display2
    for idx in range(22):
        x, y = pos[idx]
        color = _TEAM1_COLOR if idx < 11 else _TEAM2_COLOR
        ax.scatter(x, y, s=420, color=color, edgecolor="black", lw=1.0, zorder=3)
        ax.text(
            x, y, str(idx),
            ha="center", va="center", fontsize=8,
            color="white", fontweight="bold", zorder=4,
        )
        ax.text(
            x, y - 5.5, labels_all[idx],
            ha="center", va="top", fontsize=7, color="black", zorder=4,
        )

    if title:
        ax.set_title(title, fontsize=10)
