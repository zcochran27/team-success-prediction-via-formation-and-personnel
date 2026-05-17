"""Cross-model comparison helpers spanning the tabular + GNN variants.

The 2 × 2 ``kind × opponent`` design is now crossed with a per-track
feature-block axis. Each ``(kind, opponent)`` corner has up to five
variants -- the base tabular model, its ``+stats`` counterpart, the
label-only GNN, the GNN with template ``(x, y)`` coords, and the GNN
with per-season behavioral stats. The 4-row × 4-col grid plots
collapse to a 5-row × 4-col matrix indexed by ``(_ROW_TIERS,
(kind, opponent))``.

All eight models train and evaluate on the same match-blocked train /
test split produced by :mod:`scripts.build_train_test_split` (loaded
from ``data/processed/train_snapshots.parquet`` and
``test_snapshots.parquet``):

- Tabular models fit XGBoost on the train parquet and predict on the
  test parquet; the test predictions are the basis for every reported
  metric.
- GNN models train on the train parquet and per-epoch evaluate on the
  test parquet via :mod:`scripts.train_gnn`; the best-epoch test
  predictions are written to ``artifacts/gnn_<kind>_<mode>/val_preds.parquet``.

Same rows, same held-out matches -- so the comparison is now an apples-to-apples
head-to-head. (Caveat: GNN early-stopping peeks at the test set across
epochs, while tabular runs a single fixed-budget XGBoost fit. This is a
mild advantage for the GNN that is easy to remove later by adding a
small inner validation slice.)
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models import (
    _tabular_common as _tc,
    tab_arch_ego,
    tab_arch_ego_stats,
    tab_arch_matchup,
    tab_arch_matchup_stats,
    tab_pos_ego,
    tab_pos_ego_stats,
    tab_pos_matchup,
    tab_pos_matchup_stats,
)


MODEL_KEYS: tuple[str, ...] = (
    "tab_pos_ego",
    "tab_pos_matchup",
    "tab_arch_ego",
    "tab_arch_matchup",
    "tab_pos_ego_stats",
    "tab_pos_matchup_stats",
    "tab_arch_ego_stats",
    "tab_arch_matchup_stats",
    "gnn_position_single",
    "gnn_archetype_single",
    "gnn_position_paired",
    "gnn_archetype_paired",
    "gnn_position_single_coords",
    "gnn_archetype_single_coords",
    "gnn_position_paired_coords",
    "gnn_archetype_paired_coords",
    "gnn_position_single_stats",
    "gnn_archetype_single_stats",
    "gnn_position_paired_stats",
    "gnn_archetype_paired_stats",
    # Dense architecture: fully-connected intra + radial-distance matchup
    # edges (paired) + attention pool. xy is always on so no _coords suffix.
    "gnn_dense_position_single",
    "gnn_dense_archetype_single",
    "gnn_dense_position_paired",
    "gnn_dense_archetype_paired",
    "gnn_dense_position_single_stats",
    "gnn_dense_archetype_single_stats",
    "gnn_dense_position_paired_stats",
    "gnn_dense_archetype_paired_stats",
)

GRAPH_ARCH: dict[str, str] = {k: ("dense" if k.startswith("gnn_dense_") else "template") for k in MODEL_KEYS}
USE_COORDS: dict[str, bool] = {
    k: ("_coords" in k) or k.startswith("gnn_dense_") for k in MODEL_KEYS
}
USE_STATS: dict[str, bool] = {k: k.endswith("_stats") for k in MODEL_KEYS}

def _gnn_kind(key: str) -> str:
    return "archetype" if "_archetype_" in key else "position"


def _gnn_opponent(key: str) -> str:
    return "matchup" if "_paired" in key else "ego"


def _is_dense(key: str) -> bool:
    return key.startswith("gnn_dense_")


MODEL_CLASS: dict[str, str] = {k: ("tabular" if k.startswith("tab_") else "graph") for k in MODEL_KEYS}


def _tab_kind(key: str) -> str:
    return "archetype" if "_arch_" in key else "position"


def _tab_opponent(key: str) -> str:
    return "matchup" if "_matchup" in key else "ego"


KIND: dict[str, str] = {
    **{k: _tab_kind(k) for k in MODEL_KEYS if k.startswith("tab_")},
    **{k: _gnn_kind(k) for k in MODEL_KEYS if k.startswith("gnn_")},
}

OPPONENT: dict[str, str] = {
    **{k: _tab_opponent(k) for k in MODEL_KEYS if k.startswith("tab_")},
    **{k: _gnn_opponent(k) for k in MODEL_KEYS if k.startswith("gnn_")},
}


def _pretty_gnn(key: str) -> str:
    kind = "pos" if KIND[key] == "position" else "arch"
    opp = OPPONENT[key]
    tail = []
    if _is_dense(key):
        # Dense already implies xy + attention pool; skip the redundant xy tag.
        head = f"gnn-dense · {kind} · {opp}"
    else:
        head = f"gnn · {kind} · {opp}"
        if USE_COORDS[key]:
            tail.append("xy")
    if USE_STATS[key]:
        tail.append("stats")
    return head + (" · " + " · ".join(tail) if tail else "")


def _pretty_tab(key: str) -> str:
    kind = "pos" if KIND[key] == "position" else "arch"
    opp = OPPONENT[key]
    base = f"tab · {kind} · {opp}"
    return base + (" · stats" if USE_STATS[key] else "")


PRETTY_LABEL: dict[str, str] = {
    **{k: _pretty_tab(k) for k in MODEL_KEYS if k.startswith("tab_")},
    **{k: _pretty_gnn(k) for k in MODEL_KEYS if k.startswith("gnn_")},
}

# Color palette: each (kind, opponent) corner has a hue; tabular gets the lightest
# shade, tabular+stats slightly darker, label-only GNN the medium shade, +xy
# a bit darker, +stats the darkest. Lets the five tiers stay distinguishable
# even when overlaid.
MODEL_COLORS: dict[str, str] = {
    # blue family — position / ego
    "tab_pos_ego":                "#c6dbef",
    "tab_pos_ego_stats":          "#9ecae1",
    "gnn_position_single":        "#6baed6",
    "gnn_position_single_coords": "#3182bd",
    "gnn_position_single_stats":  "#08519c",
    # green family — archetype / ego
    "tab_arch_ego":                "#c7e9c0",
    "tab_arch_ego_stats":          "#a1d99b",
    "gnn_archetype_single":        "#74c476",
    "gnn_archetype_single_coords": "#41ab5d",
    "gnn_archetype_single_stats":  "#006d2c",
    # orange family — position / matchup
    "tab_pos_matchup":            "#fdd0a2",
    "tab_pos_matchup_stats":      "#fdae6b",
    "gnn_position_paired":        "#fd8d3c",
    "gnn_position_paired_coords": "#e6550d",
    "gnn_position_paired_stats":  "#a63603",
    # red family — archetype / matchup
    "tab_arch_matchup":            "#fee0d2",
    "tab_arch_matchup_stats":      "#fcae91",
    "gnn_archetype_paired":        "#fb6a4a",
    "gnn_archetype_paired_coords": "#de2d26",
    "gnn_archetype_paired_stats":  "#a50f15",
    # Dense architecture — separate hue families per (kind, opponent) corner
    # so the new track is visually distinct from the template architecture.
    "gnn_dense_position_single":         "#bcbddc",
    "gnn_dense_position_single_stats":   "#54278f",
    "gnn_dense_archetype_single":        "#99d8c9",
    "gnn_dense_archetype_single_stats":  "#005824",
    "gnn_dense_position_paired":         "#fcc5c0",
    "gnn_dense_position_paired_stats":   "#7a0177",
    "gnn_dense_archetype_paired":        "#dfc27d",
    "gnn_dense_archetype_paired_stats":  "#543005",
}


_TAB_MODULES = {
    "tab_pos_ego": tab_pos_ego,
    "tab_pos_matchup": tab_pos_matchup,
    "tab_arch_ego": tab_arch_ego,
    "tab_arch_matchup": tab_arch_matchup,
    "tab_pos_ego_stats": tab_pos_ego_stats,
    "tab_pos_matchup_stats": tab_pos_matchup_stats,
    "tab_arch_ego_stats": tab_arch_ego_stats,
    "tab_arch_matchup_stats": tab_arch_matchup_stats,
}


# --------------------------------------------------------------------------- #
# Tabular: fit on train, predict on test                                      #
# --------------------------------------------------------------------------- #

def _fit_train_predict_test(
    mod: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fit one tabular model on ``train_df`` and predict on ``test_df``."""
    X_train, y_train, _ = mod.build_features(train_df)
    X_test, y_test, _ = mod.build_features(test_df)
    model = _tc.fit_xgb(X_train, y_train)
    preds = np.asarray(model.predict(X_test))
    targets = y_test.to_numpy()

    metrics = {
        "mae": float(mean_absolute_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "r2": float(r2_score(targets, preds)),
        "sign_acc": _tc._sign_accuracy(targets, preds),
    }
    return {
        "metrics": metrics,
        "test_preds": preds,
        "test_targets": targets,
    }


def load_tabular_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit each tabular model on ``train_df`` and predict on ``test_df``.

    Returns ``{key: {metrics, test_preds, test_targets}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    selected = keys if keys is not None else _TAB_MODULES.keys()
    for key in selected:
        if key not in _TAB_MODULES:
            continue
        out[key] = _fit_train_predict_test(_TAB_MODULES[key], train_df, test_df)
    return out


# --------------------------------------------------------------------------- #
# GNN: load training artifacts                                                #
# --------------------------------------------------------------------------- #

def _gnn_dir(artifacts_root: Path, key: str) -> Path:
    suffix = key.removeprefix("gnn_")
    return Path(artifacts_root) / f"gnn_{suffix}"


def load_gnn_results(
    artifacts_root: Path = Path("artifacts"),
    keys: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load best-epoch metrics + test predictions for each GNN run that exists."""
    selected = keys if keys is not None else [k for k in MODEL_KEYS if k.startswith("gnn_")]
    out: dict[str, dict[str, Any]] = {}
    for key in selected:
        if not key.startswith("gnn_"):
            continue
        run_dir = _gnn_dir(artifacts_root, key)
        log_path = run_dir / "log.jsonl"
        val_preds_path = run_dir / "val_preds.parquet"
        summary_path = run_dir / "summary.json"
        if not log_path.exists():
            continue
        log = pd.DataFrame(
            [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        )
        best_idx = int(log["val_mse"].idxmin())
        best = log.iloc[best_idx]
        bundle: dict[str, Any] = {
            "best_epoch": int(best["epoch"]),
            "mae": float(best["val_mae"]),
            "rmse": float(best["val_rmse"]),
            "r2": float(best["val_r2"]),
            "sign_acc": float(best["val_sign_acc"]),
            "log": log,
        }
        if val_preds_path.exists():
            vp = pd.read_parquet(val_preds_path)
            bundle["test_preds"] = vp["pred"].to_numpy()
            bundle["test_targets"] = vp["target"].to_numpy()
        if summary_path.exists():
            bundle["summary"] = json.loads(summary_path.read_text())
        out[key] = bundle
    return out


# --------------------------------------------------------------------------- #
# Unified summary table                                                       #
# --------------------------------------------------------------------------- #

def build_summary_table(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """One row per available model. All metrics on the shared held-out test set."""
    rows: list[dict[str, Any]] = []
    for key in MODEL_KEYS:
        if key.startswith("tab_") and key in tab_results:
            m = tab_results[key]["metrics"]
            rows.append({
                "model": PRETTY_LABEL[key],
                "key": key,
                "class": MODEL_CLASS[key],
                "kind": KIND[key],
                "opponent": OPPONENT[key],
                "MAE": m["mae"],
                "RMSE": m["rmse"],
                "R²": m["r2"],
                "sign acc": m["sign_acc"],
                "eval": "train fit / test predict",
            })
        elif key.startswith("gnn_") and key in gnn_results:
            r = gnn_results[key]
            rows.append({
                "model": PRETTY_LABEL[key],
                "key": key,
                "class": MODEL_CLASS[key],
                "kind": KIND[key],
                "opponent": OPPONENT[key],
                "MAE": r["mae"],
                "RMSE": r["rmse"],
                "R²": r["r2"],
                "sign acc": r["sign_acc"],
                "eval": f"best-of-{r.get('best_epoch', '?')} test eval",
            })
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_metric_bars(
    summary: pd.DataFrame,
    metrics: tuple[str, ...] = ("MAE", "RMSE", "R²", "sign acc"),
    figsize: tuple[float, float] = (14, 4),
) -> Figure:
    """One subplot per metric; bars colored by model class."""
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    labels = list(summary.index)
    colors = [MODEL_COLORS[k] for k in summary["key"]]
    for ax, metric in zip(axes, metrics):
        values = summary[metric].to_numpy()
        ax.bar(
            range(len(labels)),
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        higher_is_better = metric in ("R²", "sign acc")
        ax.set_title(
            f"{metric} ({'higher' if higher_is_better else 'lower'} is better)",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "All models on the shared held-out test set "
        "(matched-row apples-to-apples)",
        y=1.04,
    )
    fig.tight_layout()
    return fig


def _gather_preds(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return ``{key: (preds, targets)}`` for every model with test predictions."""
    paired: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, bundle in tab_results.items():
        preds = bundle.get("test_preds")
        targets = bundle.get("test_targets")
        if preds is None or targets is None:
            continue
        paired[key] = (np.asarray(preds), np.asarray(targets))
    for key, bundle in gnn_results.items():
        preds = bundle.get("test_preds")
        targets = bundle.get("test_targets")
        if preds is None or targets is None:
            continue
        paired[key] = (np.asarray(preds), np.asarray(targets))
    return paired


_ROW_TIERS: tuple[tuple[str, str], ...] = (
    ("tabular",            "tabular"),
    ("tabular_stats",      "tabular (+ stats)"),
    ("graph",              "GNN (label only)"),
    ("graph_xy",           "GNN (label + xy)"),
    ("graph_stats",        "GNN (label + stats)"),
    ("graph_dense",        "GNN-dense (xy, attn)"),
    ("graph_dense_stats",  "GNN-dense (xy, attn, + stats)"),
)


def _key_at(klass_tier: str, col_key: str) -> str | None:
    """Return the model key for a (row tier, (kind, opponent)) cell, if any."""
    for k in MODEL_KEYS:
        kind_opp = f"{KIND[k]}_{OPPONENT[k]}"
        if kind_opp != col_key:
            continue
        if MODEL_CLASS[k] == "tabular":
            if klass_tier == "tabular" and not USE_STATS[k]:
                return k
            if klass_tier == "tabular_stats" and USE_STATS[k]:
                return k
            continue
        # graph
        if _is_dense(k):
            if klass_tier == "graph_dense" and not USE_STATS[k]:
                return k
            if klass_tier == "graph_dense_stats" and USE_STATS[k]:
                return k
            continue
        # template-arch graph
        plain = not USE_COORDS[k] and not USE_STATS[k]
        if klass_tier == "graph" and plain:
            return k
        if klass_tier == "graph_xy" and USE_COORDS[k] and not USE_STATS[k]:
            return k
        if klass_tier == "graph_stats" and USE_STATS[k] and not USE_COORDS[k]:
            return k
    return None


def plot_pred_vs_target_grid(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (15, 22),
) -> Figure:
    """7 × 4 scatter grid covering every (row tier, (kind, opponent)) cell."""
    column_order = ("position_ego", "position_matchup", "archetype_ego", "archetype_matchup")
    fig, axes = plt.subplots(len(_ROW_TIERS), 4, figsize=figsize)
    preds = _gather_preds(tab_results, gnn_results)
    for row_idx, (tier, _label) in enumerate(_ROW_TIERS):
        for col_idx, col_key in enumerate(column_order):
            ax = axes[row_idx, col_idx]
            key = _key_at(tier, col_key)
            if key is None or key not in preds:
                ax.axis("off")
                continue
            p, t = preds[key]
            ax.scatter(t, p, color=MODEL_COLORS[key], alpha=0.25, s=8)
            lo = float(min(t.min(), p.min()))
            hi = float(max(t.max(), p.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
            r = float(np.corrcoef(t, p)[0, 1]) if len(t) >= 2 else float("nan")
            ax.set_title(f"{PRETTY_LABEL[key]} — r = {r:+.3f}", fontsize=9)
            ax.grid(alpha=0.3)
            if col_idx == 0:
                ax.set_ylabel("prediction")
            if row_idx == len(_ROW_TIERS) - 1:
                ax.set_xlabel("target")
    fig.suptitle(
        "Test-set predictions vs targets — tabular / +stats / GNN / +xy / +stats / dense / dense+stats",
        y=1.005,
    )
    fig.tight_layout()
    return fig


def plot_residual_hists(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (15, 22),
    bins: int = 60,
) -> Figure:
    """7 × 4 residual histograms aligned with the prediction grid."""
    column_order = ("position_ego", "position_matchup", "archetype_ego", "archetype_matchup")
    fig, axes = plt.subplots(len(_ROW_TIERS), 4, figsize=figsize, sharex=True)
    preds = _gather_preds(tab_results, gnn_results)
    for row_idx, (tier, _label) in enumerate(_ROW_TIERS):
        for col_idx, col_key in enumerate(column_order):
            ax = axes[row_idx, col_idx]
            key = _key_at(tier, col_key)
            if key is None or key not in preds:
                ax.axis("off")
                continue
            p, t = preds[key]
            residuals = p - t
            ax.hist(
                residuals, bins=bins,
                color=MODEL_COLORS[key], alpha=0.8,
                edgecolor="white", lw=0.4,
            )
            ax.axvline(0.0, color="k", lw=0.6, alpha=0.4)
            ax.set_title(
                f"{PRETTY_LABEL[key]} — μ = {residuals.mean():+.3f}, σ = {residuals.std():.3f}",
                fontsize=9,
            )
            ax.grid(alpha=0.3)
            if row_idx == len(_ROW_TIERS) - 1:
                ax.set_xlabel("pred − target")
    fig.suptitle(
        "Test-set residuals — tabular / +stats / GNN / +xy / +stats / dense / dense+stats",
        y=1.005,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Pairwise deltas                                                             #
# --------------------------------------------------------------------------- #

def ablation_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Pairwise deltas (B − A) for the three design axes.

    Reports ΔMAE / ΔRMSE / ΔR² / Δ sign acc on the shared test set.
    For MAE / RMSE a negative delta means B is better; for R² /
    sign acc, positive means B is better. Skips pairs where one side
    is missing.
    """
    by_key = summary.reset_index().set_index("key").to_dict("index")

    def pair(a: str, b: str, axis: str, label: str) -> dict[str, Any] | None:
        if a not in by_key or b not in by_key:
            return None
        ra, rb = by_key[a], by_key[b]
        return {
            "axis": axis,
            "comparison": label,
            "from": PRETTY_LABEL[a],
            "to": PRETTY_LABEL[b],
            "Δ MAE": rb["MAE"] - ra["MAE"],
            "Δ RMSE": rb["RMSE"] - ra["RMSE"],
            "Δ R²": rb["R²"] - ra["R²"],
            "Δ sign acc": rb["sign acc"] - ra["sign acc"],
        }

    pairs: list[dict[str, Any]] = []

    class_pairs = [
        ("tab_pos_ego",       "gnn_position_single",   "pos / ego"),
        ("tab_pos_matchup",   "gnn_position_paired",   "pos / matchup"),
        ("tab_arch_ego",      "gnn_archetype_single",  "arch / ego"),
        ("tab_arch_matchup",  "gnn_archetype_paired",  "arch / matchup"),
    ]
    for a, b, where in class_pairs:
        row = pair(a, b, "tabular → graph", where)
        if row is not None:
            pairs.append(row)

    kind_pairs = [
        ("tab_pos_ego",          "tab_arch_ego",          "tabular / ego"),
        ("tab_pos_matchup",      "tab_arch_matchup",      "tabular / matchup"),
        ("gnn_position_single",  "gnn_archetype_single",  "graph / ego"),
        ("gnn_position_paired",  "gnn_archetype_paired",  "graph / matchup"),
    ]
    for a, b, where in kind_pairs:
        row = pair(a, b, "position → archetype", where)
        if row is not None:
            pairs.append(row)

    opp_pairs = [
        ("tab_pos_ego",          "tab_pos_matchup",       "tabular / pos"),
        ("tab_arch_ego",         "tab_arch_matchup",      "tabular / arch"),
        ("gnn_position_single",  "gnn_position_paired",   "graph / pos"),
        ("gnn_archetype_single", "gnn_archetype_paired",  "graph / arch"),
    ]
    for a, b, where in opp_pairs:
        row = pair(a, b, "ego → matchup", where)
        if row is not None:
            pairs.append(row)

    # Coord-feature effect: each graph variant vs. its label+xy counterpart.
    coords_pairs = [
        ("gnn_position_single",   "gnn_position_single_coords",   "graph / pos · ego"),
        ("gnn_archetype_single",  "gnn_archetype_single_coords",  "graph / arch · ego"),
        ("gnn_position_paired",   "gnn_position_paired_coords",   "graph / pos · matchup"),
        ("gnn_archetype_paired",  "gnn_archetype_paired_coords",  "graph / arch · matchup"),
    ]
    for a, b, where in coords_pairs:
        row = pair(a, b, "label only → label + xy", where)
        if row is not None:
            pairs.append(row)

    # Stats-feature effect: each variant vs. its +stats counterpart, on both
    # the tabular and graph tracks. Reading these side-by-side answers
    # "does the stat block matter more for trees or for the GNN?".
    stats_pairs = [
        ("tab_pos_ego",           "tab_pos_ego_stats",           "tab / pos · ego"),
        ("tab_arch_ego",          "tab_arch_ego_stats",          "tab / arch · ego"),
        ("tab_pos_matchup",       "tab_pos_matchup_stats",       "tab / pos · matchup"),
        ("tab_arch_matchup",      "tab_arch_matchup_stats",      "tab / arch · matchup"),
        ("gnn_position_single",   "gnn_position_single_stats",   "graph / pos · ego"),
        ("gnn_archetype_single",  "gnn_archetype_single_stats",  "graph / arch · ego"),
        ("gnn_position_paired",   "gnn_position_paired_stats",   "graph / pos · matchup"),
        ("gnn_archetype_paired",  "gnn_archetype_paired_stats",  "graph / arch · matchup"),
    ]
    for a, b, where in stats_pairs:
        row = pair(a, b, "label only → label + stats", where)
        if row is not None:
            pairs.append(row)

    # Tabular → graph deltas with the +stats block held constant.
    stats_class_pairs = [
        ("tab_pos_ego_stats",      "gnn_position_single_stats",   "pos / ego"),
        ("tab_pos_matchup_stats",  "gnn_position_paired_stats",   "pos / matchup"),
        ("tab_arch_ego_stats",     "gnn_archetype_single_stats",  "arch / ego"),
        ("tab_arch_matchup_stats", "gnn_archetype_paired_stats",  "arch / matchup"),
    ]
    for a, b, where in stats_class_pairs:
        row = pair(a, b, "tabular+stats → graph+stats", where)
        if row is not None:
            pairs.append(row)

    # Dense architecture effect: every template-arch graph variant vs. its
    # gnn-dense counterpart (fully-connected intra, radial inter, attention
    # pool, xy always on). Read these to answer "does the new topology +
    # pool actually beat the template + mean-pool baseline?".
    dense_pairs = [
        ("gnn_position_single_coords",   "gnn_dense_position_single",
         "graph / pos · ego (+xy baseline)"),
        ("gnn_archetype_single_coords",  "gnn_dense_archetype_single",
         "graph / arch · ego (+xy baseline)"),
        ("gnn_position_paired_coords",   "gnn_dense_position_paired",
         "graph / pos · matchup (+xy baseline)"),
        ("gnn_archetype_paired_coords",  "gnn_dense_archetype_paired",
         "graph / arch · matchup (+xy baseline)"),
        ("gnn_position_single_stats",    "gnn_dense_position_single_stats",
         "graph / pos · ego (+stats baseline)"),
        ("gnn_archetype_single_stats",   "gnn_dense_archetype_single_stats",
         "graph / arch · ego (+stats baseline)"),
        ("gnn_position_paired_stats",    "gnn_dense_position_paired_stats",
         "graph / pos · matchup (+stats baseline)"),
        ("gnn_archetype_paired_stats",   "gnn_dense_archetype_paired_stats",
         "graph / arch · matchup (+stats baseline)"),
    ]
    for a, b, where in dense_pairs:
        row = pair(a, b, "template → dense", where)
        if row is not None:
            pairs.append(row)

    return pd.DataFrame(pairs).round(4)
