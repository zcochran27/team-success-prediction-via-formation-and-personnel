"""Cross-model comparison helpers spanning the 4 tabular + 4 GNN variants.

Eight models in a 2 × 2 × 2 design (``model_class × kind × opponent``):

==================  ===========  ==========  ===========
key                 model_class  kind        opponent
==================  ===========  ==========  ===========
tab_pos_ego         tabular      position    ego
tab_pos_matchup     tabular      position    matchup
tab_arch_ego        tabular      archetype   ego
tab_arch_matchup    tabular      archetype   matchup
gnn_position_single graph        position    ego
gnn_archetype_single graph       archetype   ego
gnn_position_paired graph        position    matchup
gnn_archetype_paired graph       archetype   matchup
==================  ===========  ==========  ===========

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
    tab_arch_matchup,
    tab_pos_ego,
    tab_pos_matchup,
)


MODEL_KEYS: tuple[str, ...] = (
    "tab_pos_ego",
    "tab_pos_matchup",
    "tab_arch_ego",
    "tab_arch_matchup",
    "gnn_position_single",
    "gnn_archetype_single",
    "gnn_position_paired",
    "gnn_archetype_paired",
)

MODEL_CLASS: dict[str, str] = {
    "tab_pos_ego": "tabular",
    "tab_pos_matchup": "tabular",
    "tab_arch_ego": "tabular",
    "tab_arch_matchup": "tabular",
    "gnn_position_single": "graph",
    "gnn_archetype_single": "graph",
    "gnn_position_paired": "graph",
    "gnn_archetype_paired": "graph",
}

KIND: dict[str, str] = {
    "tab_pos_ego": "position",
    "tab_pos_matchup": "position",
    "tab_arch_ego": "archetype",
    "tab_arch_matchup": "archetype",
    "gnn_position_single": "position",
    "gnn_archetype_single": "archetype",
    "gnn_position_paired": "position",
    "gnn_archetype_paired": "archetype",
}

OPPONENT: dict[str, str] = {
    "tab_pos_ego": "ego",
    "tab_pos_matchup": "matchup",
    "tab_arch_ego": "ego",
    "tab_arch_matchup": "matchup",
    "gnn_position_single": "ego",
    "gnn_archetype_single": "ego",
    "gnn_position_paired": "matchup",
    "gnn_archetype_paired": "matchup",
}

PRETTY_LABEL: dict[str, str] = {
    "tab_pos_ego":          "tab · pos · ego",
    "tab_pos_matchup":      "tab · pos · matchup",
    "tab_arch_ego":         "tab · arch · ego",
    "tab_arch_matchup":     "tab · arch · matchup",
    "gnn_position_single":  "gnn · pos · ego",
    "gnn_archetype_single": "gnn · arch · ego",
    "gnn_position_paired":  "gnn · pos · matchup",
    "gnn_archetype_paired": "gnn · arch · matchup",
}

MODEL_COLORS: dict[str, str] = {
    "tab_pos_ego":          "#9ecae1",
    "gnn_position_single":  "#08519c",
    "tab_arch_ego":         "#a1d99b",
    "gnn_archetype_single": "#006d2c",
    "tab_pos_matchup":      "#fdae6b",
    "gnn_position_paired":  "#a63603",
    "tab_arch_matchup":     "#fcae91",
    "gnn_archetype_paired": "#a50f15",
}


_TAB_MODULES = {
    "tab_pos_ego": tab_pos_ego,
    "tab_pos_matchup": tab_pos_matchup,
    "tab_arch_ego": tab_arch_ego,
    "tab_arch_matchup": tab_arch_matchup,
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
        "All 8 models on the shared held-out test set "
        f"(matched-row apples-to-apples)",
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


def plot_pred_vs_target_grid(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (15, 8),
) -> Figure:
    """2 × 4 scatter grid; row 1 = tabular, row 2 = graph, ordered by (kind, opponent)."""
    column_order = ("position_ego", "position_matchup", "archetype_ego", "archetype_matchup")
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    preds = _gather_preds(tab_results, gnn_results)
    for col_idx, col_key in enumerate(column_order):
        for row_idx, klass in enumerate(("tabular", "graph")):
            ax = axes[row_idx, col_idx]
            match = [
                k for k in MODEL_KEYS
                if MODEL_CLASS[k] == klass
                and f"{KIND[k]}_{OPPONENT[k]}" == col_key
            ]
            if not match or match[0] not in preds:
                ax.axis("off")
                continue
            key = match[0]
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
            if row_idx == 1:
                ax.set_xlabel("target")
    fig.suptitle("Test-set predictions vs targets — tabular (top) vs GNN (bottom)", y=1.02)
    fig.tight_layout()
    return fig


def plot_residual_hists(
    tab_results: Mapping[str, Mapping[str, Any]],
    gnn_results: Mapping[str, Mapping[str, Any]],
    figsize: tuple[float, float] = (15, 8),
    bins: int = 60,
) -> Figure:
    """2 × 4 residual histograms aligned with the prediction grid."""
    column_order = ("position_ego", "position_matchup", "archetype_ego", "archetype_matchup")
    fig, axes = plt.subplots(2, 4, figsize=figsize, sharex=True)
    preds = _gather_preds(tab_results, gnn_results)
    for col_idx, col_key in enumerate(column_order):
        for row_idx, klass in enumerate(("tabular", "graph")):
            ax = axes[row_idx, col_idx]
            match = [
                k for k in MODEL_KEYS
                if MODEL_CLASS[k] == klass
                and f"{KIND[k]}_{OPPONENT[k]}" == col_key
            ]
            if not match or match[0] not in preds:
                ax.axis("off")
                continue
            key = match[0]
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
            if row_idx == 1:
                ax.set_xlabel("pred − target")
    fig.suptitle("Test-set residuals — tabular (top) vs GNN (bottom)", y=1.02)
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

    return pd.DataFrame(pairs).round(4)
