"""Plot helpers for the half-with-subs graphs.

Visualizes a single PyG Data object on a pitch background:

  - Nodes colored by side (team1 vs team2) and shaded by starter/substitute
    status. Starters sit at their normalized template coords; subs are offset
    slightly along y so they don't overlap the starter they replaced.
  - Edges drawn per type: formation edges (template adjacency plus
    sub-inherited copies) in gray; sub edges (starter to substitute) in
    orange, solid when the sub kept the slot and dashed when re-deployed;
    matchup edges (paired-mode cross-team) in green, with line width scaled
    by overlap_fraction (the share of half time both players were on
    together).

The helpers are small so the demo notebook
(graphs/notebooks/half_subs_graph_demo.ipynb) stays thin: it loads a row,
builds the graph, then calls draw_single_team_graph or draw_paired_graph.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch_geometric.data import Data

from features.build_graphs_subs import (
    EDGE_BASE_DIM,
    EDGE_TYPE_INTRA,
    EDGE_TYPE_MATCHUP,
    EDGE_TYPE_SUB,
    N_STARTERS,
)


_TEAM_COLORS: tuple[str, str] = ("#2c7fb8", "#d7301f")    # team1, team2
_SUB_COLORS:  tuple[str, str] = ("#a6cee3", "#fdbb84")    # team1-sub, team2-sub

# Per-edge-type rendering parameters. Intra-team edges are fully connected
# now, so the non-template subset gets very low alpha (the model uses
# attention to weight them; visually they're "background"). Template-
# adjacent intra edges get rendered more prominently as a hint to the
# reader. Matchup width is overridden per-edge by the overlap fraction.
_INTRA_STYLE_TEMPLATE = {"color": "#525252", "lw": 1.0, "alpha": 0.55}
_INTRA_STYLE_OTHER    = {"color": "#bdbdbd", "lw": 0.35, "alpha": 0.18}
_SUB_STYLE     = {"color": "#e6550d", "lw": 1.8, "alpha": 0.85}
_MATCHUP_STYLE = {"color": "#31a354", "lw": 0.5}  # width set per-edge

# Pitch dimensions in yards. Standard 120 x 80 (StatsBomb / mplsoccer convention).
PITCH_LENGTH = 120.0
PITCH_WIDTH  = 80.0

# Sub-node offset in yards (pitch is 80 yards wide): shift each sub along y
# so it doesn't overlap the starter it replaced. Direction alternates per
# sub so multiple subs at the same template slot fan out.
_SUB_DY_YARDS = 5.5


def _draw_pitch(ax: Axes) -> None:
    """Draw a 120 x 80 yard pitch with standard markings on ax.

    Coordinates are in yards: (0, 0) is the bottom-left corner, (120, 80) the
    top-right. Markings: pitch outline, halfway line, center circle (10 yd
    radius) and spot, 18-yard penalty areas, 6-yard goal areas, penalty spots
    (12 yd from goal line), and penalty arcs.
    """
    L, W = PITCH_LENGTH, PITCH_WIDTH
    # Field surface + outline.
    ax.add_patch(patches.Rectangle((0.0, 0.0), L, W,
                                   facecolor="#e8f5e9", edgecolor="black", lw=1.2))
    # Halfway line + center circle + center spot.
    ax.plot([L / 2, L / 2], [0.0, W], color="black", lw=0.8, alpha=0.6)
    ax.add_patch(patches.Circle((L / 2, W / 2), 10.0, fill=False,
                                color="black", lw=0.8, alpha=0.6))
    ax.add_patch(patches.Circle((L / 2, W / 2), 0.4, color="black", alpha=0.6))
    # 18-yard penalty boxes (18 yd deep x 44 yd wide, centered).
    pen_y0 = (W - 44.0) / 2  # = 18
    ax.add_patch(patches.Rectangle((0.0, pen_y0), 18.0, 44.0, fill=False,
                                   color="black", lw=0.7, alpha=0.6))
    ax.add_patch(patches.Rectangle((L - 18.0, pen_y0), 18.0, 44.0, fill=False,
                                   color="black", lw=0.7, alpha=0.6))
    # 6-yard goal areas (6 yd deep x 20 yd wide, centered).
    goal_y0 = (W - 20.0) / 2  # = 30
    ax.add_patch(patches.Rectangle((0.0, goal_y0), 6.0, 20.0, fill=False,
                                   color="black", lw=0.7, alpha=0.6))
    ax.add_patch(patches.Rectangle((L - 6.0, goal_y0), 6.0, 20.0, fill=False,
                                   color="black", lw=0.7, alpha=0.6))
    # Penalty spots (12 yd from each goal line).
    ax.add_patch(patches.Circle((12.0,     W / 2), 0.4, color="black", alpha=0.6))
    ax.add_patch(patches.Circle((L - 12.0, W / 2), 0.4, color="black", alpha=0.6))
    # Penalty arcs (10 yd radius, drawn outside the 18-yd box).
    ax.add_patch(patches.Arc((12.0,     W / 2), 20.0, 20.0,
                             theta1=-53, theta2=53, color="black", lw=0.7, alpha=0.6))
    ax.add_patch(patches.Arc((L - 12.0, W / 2), 20.0, 20.0,
                             theta1=127, theta2=233, color="black", lw=0.7, alpha=0.6))

    ax.set_xlim(-3.0, L + 3.0)
    ax.set_ylim(-3.0, W + 3.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _scale_to_pitch(
    pos: torch.Tensor, team: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map normalized [0, 1] template coords onto the 120 x 80 yard pitch.

    The template grid (see graphs/templates.py) was authored on a nominal
    100 x 80 tactical canvas where x=100 was the striker line.

    Single-team mode (team is None): inset x to (10, 110) yd so the GK lands
    about 16 yd from the own goal line and strikers land at x=110.

    Paired mode (team provided): the model receives team 2's coords rotated
    180 deg about the pitch center so matchup edges have nonzero geometric
    features. We render that as a matchup overlay: team 1 occupies x in
    [0, 100] (own goal at x=0, striker line at x=100), team 2 occupies x in
    [20, 120] (own goal at x=120, striker line at x=20). The ranges overlap
    in [20, 100], the band where matchups line up: team 1's ST near team 2's
    CBs, midfielders contesting midfielders, team 1's CBs near team 2's STs.
    Y axis is unchanged.
    """
    out = pos.clone()
    if team is None:
        inset = 10.0
        out[:, 0] = inset + out[:, 0] * (PITCH_LENGTH - 2 * inset)  # -> [10, 110]
    else:
        is_team2 = (team == 1)
        # Team 1 (unflipped): template x in [0, 1] -> pitch x in [0, 100].
        # GK (0.06) -> 6 yd (own 6-yd box); ST (1.0) -> 100 yd (in
        # team 2's defensive third, ~20 yd from the opposing goal line).
        out[~is_team2, 0] = out[~is_team2, 0] * 100.0                # -> [0, 100]
        # Team 2 (already flipped to 1 - x_orig): map onto [20, 120].
        # GK (flipped 0.94) -> 114 yd (own 6-yd box); ST (flipped 0.0)
        # -> 20 yd (in team 1's defensive third). Net effect: striker
        # vs opposing CB pairs land within ~5 yd of each other.
        out[is_team2,  0] = 20.0 + out[is_team2, 0] * 100.0          # -> [20, 120]
    out[:, 1] = out[:, 1] * PITCH_WIDTH                              # -> [0, 80]
    return out


def _node_positions(
    pos: torch.Tensor,
    is_starter_flag: torch.Tensor,
    team: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return display coords on the 120 x 80 pitch.

    Starters land at their scaled template coords; subs are offset in y by
    _SUB_DY_YARDS so they don't sit on top of the starter they replaced. Subs
    alternate +/- so multiple subs at the same slot fan out symmetrically.

    team switches between single-team scaling (full pitch inset) and
    paired-mode scaling (each team in their own half); see _scale_to_pitch.
    """
    out = _scale_to_pitch(pos, team=team)
    sub_count = 0
    for i, is_starter in enumerate(is_starter_flag.tolist()):
        if is_starter < 0.5:
            sign = 1.0 if sub_count % 2 == 0 else -1.0
            out[i, 1] = out[i, 1] + sign * _SUB_DY_YARDS
            sub_count += 1
    return out


def _node_colors(is_starter: torch.Tensor, team: torch.Tensor | None) -> list[str]:
    """Choose a marker color per node from the team + starter flags."""
    n = is_starter.size(0)
    teams = team.tolist() if team is not None else [0] * n
    starters = is_starter.tolist()
    out: list[str] = []
    for t, s in zip(teams, starters):
        if t == 0:
            out.append(_TEAM_COLORS[0] if s >= 0.5 else _SUB_COLORS[0])
        else:
            out.append(_TEAM_COLORS[1] if s >= 0.5 else _SUB_COLORS[1])
    return out


def _resolve_labels(
    data: Data, vocab: Sequence[str] | None,
) -> list[str]:
    """Return a human-readable label per node.

    If vocab is supplied (e.g. POSITION_VOCAB or ARCHETYPE_VOCAB), uses it to
    decode data.x. Otherwise returns the raw category index as a string.
    """
    cats = data.x.tolist()
    if vocab is None:
        return [str(c) for c in cats]
    return [vocab[c] for c in cats]


def draw_single_team_graph(
    data: Data,
    *,
    ax: Axes | None = None,
    vocab: Sequence[str] | None = None,
    title: str | None = None,
) -> Axes:
    """Render one team's graph (formation + sub edges) on a pitch background.

    data should be the per-team Data object returned by build_half_subs_graph
    in single mode. vocab is optional: pass POSITION_VOCAB or ARCHETYPE_VOCAB
    from features.vocab to label nodes with their text label rather than the
    raw embedding index.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5.6))
    _draw_pitch(ax)

    is_starter_col = data.x_num[:, 0]
    display_pos = _node_positions(data.pos, is_starter_col)
    colors = _node_colors(is_starter_col, team=None)
    labels = _resolve_labels(data, vocab)

    # Edges (drawn before nodes so markers sit on top). Intra-team edges
    # are fully connected, so non-template-adjacent ones get drawn very
    # faintly while template-adjacent and sub edges stay prominent.
    edge_attr = data.edge_attr
    edge_index = data.edge_index
    seen: set[tuple[int, int, int]] = set()  # de-dup undirected pairs by type
    for k in range(edge_index.size(1)):
        i = int(edge_index[0, k]); j = int(edge_index[1, k])
        etype = int(edge_attr[k, 4:7].argmax().item())
        key = (min(i, j), max(i, j), etype)
        if key in seen:
            continue
        seen.add(key)
        if etype == EDGE_TYPE_SUB:
            style = _SUB_STYLE
            same_pos_flag = float(edge_attr[k, 8].item())
            linestyle = "-" if same_pos_flag > 0.5 else "--"
        else:  # EDGE_TYPE_INTRA (matchup edges never appear in single mode)
            template_adj = float(edge_attr[k, 10].item())
            style = _INTRA_STYLE_TEMPLATE if template_adj > 0.5 else _INTRA_STYLE_OTHER
            linestyle = "-"
        ax.plot(
            [display_pos[i, 0], display_pos[j, 0]],
            [display_pos[i, 1], display_pos[j, 1]],
            color=style["color"], lw=style["lw"], linestyle=linestyle,
            alpha=style["alpha"], zorder=1,
        )

    # Nodes + labels.
    for i, (x, y) in enumerate(display_pos.tolist()):
        is_starter = is_starter_col[i].item() >= 0.5
        ax.scatter(x, y, s=240 if is_starter else 160,
                   color=colors[i], edgecolor="black", lw=0.8, zorder=2)
        ax.text(x, y, labels[i], ha="center", va="center",
                fontsize=7, color="white", zorder=3)

    if title:
        ax.set_title(title, fontsize=10)
    return ax


def draw_paired_graph(
    data: Data,
    *,
    ax: Axes | None = None,
    vocab: Sequence[str] | None = None,
    title: str | None = None,
    matchup_lw_scale: float = 2.5,
) -> Axes:
    """Render the joint team1+team2 graph with matchup edges weighted by overlap.

    data is the paired-mode Data from build_half_subs_graph (carries
    data.team distinguishing the two sides). Matchup-edge line width is scaled
    by overlap_fraction: full-half overlaps look thick, brief ones barely
    visible.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6.5))
    _draw_pitch(ax)

    is_starter_col = data.x_num[:, 0]
    display_pos = _node_positions(data.pos, is_starter_col, team=data.team)
    colors = _node_colors(is_starter_col, team=data.team)
    labels = _resolve_labels(data, vocab)

    edge_attr = data.edge_attr
    edge_index = data.edge_index
    seen: set[tuple[int, int, int]] = set()
    for k in range(edge_index.size(1)):
        i = int(edge_index[0, k]); j = int(edge_index[1, k])
        etype = int(edge_attr[k, 4:7].argmax().item())
        key = (min(i, j), max(i, j), etype)
        if key in seen:
            continue
        seen.add(key)
        if etype == EDGE_TYPE_MATCHUP:
            overlap = float(edge_attr[k, 9].item())
            color = _MATCHUP_STYLE["color"]
            lw = max(0.15, overlap * matchup_lw_scale)
            alpha = 0.18 + 0.45 * min(overlap, 1.0)
            linestyle = "-"
        elif etype == EDGE_TYPE_SUB:
            same_pos_flag = float(edge_attr[k, 8].item())
            color = _SUB_STYLE["color"]
            lw = _SUB_STYLE["lw"]
            alpha = _SUB_STYLE["alpha"]
            linestyle = "-" if same_pos_flag > 0.5 else "--"
        else:  # EDGE_TYPE_INTRA: fade non-template-adjacent pairs
            template_adj = float(edge_attr[k, 10].item())
            style = _INTRA_STYLE_TEMPLATE if template_adj > 0.5 else _INTRA_STYLE_OTHER
            color = style["color"]
            lw = style["lw"]
            alpha = style["alpha"]
            linestyle = "-"
        ax.plot(
            [display_pos[i, 0], display_pos[j, 0]],
            [display_pos[i, 1], display_pos[j, 1]],
            color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=1,
        )

    for i, (x, y) in enumerate(display_pos.tolist()):
        is_starter = is_starter_col[i].item() >= 0.5
        ax.scatter(x, y, s=220 if is_starter else 140,
                   color=colors[i], edgecolor="black", lw=0.8, zorder=2)
        ax.text(x, y, labels[i], ha="center", va="center",
                fontsize=6.5, color="white", zorder=3)

    if title:
        ax.set_title(title, fontsize=10)
    return ax


def legend_handles() -> tuple[list, list[str]]:
    """Return (handles, labels) for a shared legend describing edge/node types."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=_INTRA_STYLE_TEMPLATE["color"], lw=1.5,
               label="intra-team edge (template-adjacent)"),
        Line2D([0], [0], color=_INTRA_STYLE_OTHER["color"], lw=1.5,
               label="intra-team edge (non-template)"),
        Line2D([0], [0], color=_SUB_STYLE["color"], lw=1.5,
               label="sub edge (solid = same slot, dashed = re-deployed)"),
        Line2D([0], [0], color=_MATCHUP_STYLE["color"], lw=1.5,
               label="matchup edge (width ∝ time overlap)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TEAM_COLORS[0], markersize=10, markeredgecolor="black", label="team 1 starter"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_SUB_COLORS[0],  markersize=8,  markeredgecolor="black", label="team 1 sub"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TEAM_COLORS[1], markersize=10, markeredgecolor="black", label="team 2 starter"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_SUB_COLORS[1],  markersize=8,  markeredgecolor="black", label="team 2 sub"),
    ]
    labels = [h.get_label() for h in handles]
    return handles, labels
