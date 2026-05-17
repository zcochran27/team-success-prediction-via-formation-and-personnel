"""Plotting helpers for the GNN training notebook.

The training script :mod:`scripts.train_gnn` writes per-epoch metrics to
``log.jsonl``, the best-epoch validation predictions to
``val_preds.parquet``, and a frozen hyperparameter snapshot to
``config.json`` inside ``artifacts/gnn_{kind}_{mode}/`` for each of the
four GNN variants. The functions here consume those files so the
notebook stays thin.

Convention: the 4 model keys used everywhere are
``"position_single"`` / ``"archetype_single"`` / ``"position_paired"`` /
``"archetype_paired"``. Each maps to a directory by the same name
(``artifacts/gnn_position_single/`` etc.).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


MODEL_KEYS: tuple[str, ...] = (
    "position_single",
    "archetype_single",
    "position_paired",
    "archetype_paired",
    "position_single_coords",
    "archetype_single_coords",
    "position_paired_coords",
    "archetype_paired_coords",
    "position_single_stats",
    "archetype_single_stats",
    "position_paired_stats",
    "archetype_paired_stats",
    # Dense architecture (gnn_dense_*). xy is always on, attention pool.
    "dense_position_single",
    "dense_archetype_single",
    "dense_position_paired",
    "dense_archetype_paired",
    "dense_position_single_stats",
    "dense_archetype_single_stats",
    "dense_position_paired_stats",
    "dense_archetype_paired_stats",
)

# Stable palette: each (kind, mode) corner gets a hue family; base / +xy /
# +stats variants share the family at progressively darker shades. Dense
# variants get their own hue families per (kind, mode) corner so the new
# architecture stays visually distinct from the template baseline.
MODEL_COLORS: dict[str, str] = {
    # blues — position single
    "position_single":         "#9ecae1",
    "position_single_coords":  "#3182bd",
    "position_single_stats":   "#08519c",
    # greens — archetype single
    "archetype_single":        "#a1d99b",
    "archetype_single_coords": "#41ab5d",
    "archetype_single_stats":  "#006d2c",
    # oranges — position paired
    "position_paired":         "#fdae6b",
    "position_paired_coords":  "#e6550d",
    "position_paired_stats":   "#a63603",
    # reds — archetype paired
    "archetype_paired":        "#fcae91",
    "archetype_paired_coords": "#de2d26",
    "archetype_paired_stats":  "#a50f15",
    # purples — dense position single
    "dense_position_single":         "#bcbddc",
    "dense_position_single_stats":   "#54278f",
    # teals — dense archetype single
    "dense_archetype_single":        "#99d8c9",
    "dense_archetype_single_stats":  "#005824",
    # magentas — dense position paired
    "dense_position_paired":         "#fcc5c0",
    "dense_position_paired_stats":   "#7a0177",
    # browns — dense archetype paired
    "dense_archetype_paired":        "#dfc27d",
    "dense_archetype_paired_stats":  "#543005",
}

PRETTY_LABEL: dict[str, str] = {
    "position_single":         "position, single",
    "archetype_single":        "archetype, single",
    "position_paired":         "position, paired",
    "archetype_paired":        "archetype, paired",
    "position_single_coords":  "position, single + xy",
    "archetype_single_coords": "archetype, single + xy",
    "position_paired_coords":  "position, paired + xy",
    "archetype_paired_coords": "archetype, paired + xy",
    "position_single_stats":   "position, single + stats",
    "archetype_single_stats":  "archetype, single + stats",
    "position_paired_stats":   "position, paired + stats",
    "archetype_paired_stats":  "archetype, paired + stats",
    "dense_position_single":         "dense position, single",
    "dense_archetype_single":        "dense archetype, single",
    "dense_position_paired":         "dense position, paired",
    "dense_archetype_paired":        "dense archetype, paired",
    "dense_position_single_stats":   "dense position, single + stats",
    "dense_archetype_single_stats":  "dense archetype, single + stats",
    "dense_position_paired_stats":   "dense position, paired + stats",
    "dense_archetype_paired_stats":  "dense archetype, paired + stats",
}


def model_dir(artifacts_root: Path, key: str) -> Path:
    """Return ``artifacts_root / f"gnn_{key}"``."""
    return Path(artifacts_root) / f"gnn_{key}"


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def load_training_run(run_dir: Path) -> dict[str, object] | None:
    """Load ``{log, val_preds, config, summary}`` for one run, or ``None`` if missing."""
    run_dir = Path(run_dir)
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        return None
    bundle: dict[str, object] = {"log": _read_jsonl(log_path)}
    val_preds_path = run_dir / "val_preds.parquet"
    if val_preds_path.exists():
        bundle["val_preds"] = pd.read_parquet(val_preds_path)
    config_path = run_dir / "config.json"
    if config_path.exists():
        bundle["config"] = json.loads(config_path.read_text())
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        bundle["summary"] = json.loads(summary_path.read_text())
    return bundle


def load_all_runs(
    artifacts_root: Path = Path("artifacts"),
    keys: tuple[str, ...] = MODEL_KEYS,
) -> dict[str, dict[str, object]]:
    """Return ``{key: bundle}`` for every model that has a log file."""
    runs: dict[str, dict[str, object]] = {}
    for key in keys:
        bundle = load_training_run(model_dir(artifacts_root, key))
        if bundle is not None:
            runs[key] = bundle
    return runs


# --------------------------------------------------------------------------- #
# Per-model loss curves                                                       #
# --------------------------------------------------------------------------- #

def plot_loss_curves(
    runs: Mapping[str, Mapping[str, object]],
    figsize: tuple[float, float] = (16, 18),
) -> Figure:
    """5 x 4 grid of train vs test MSE curves, one panel per GNN variant."""
    fig, axes = plt.subplots(5, 4, figsize=figsize, sharex=True)
    for ax, key in zip(axes.flat, MODEL_KEYS):
        bundle = runs.get(key)
        if bundle is None:
            ax.set_title(f"{PRETTY_LABEL[key]} — (no log)", fontsize=10)
            ax.axis("off")
            continue
        log = bundle["log"]
        color = MODEL_COLORS[key]
        ax.plot(log["epoch"], log["train_mse"], color=color, lw=1.6, label="train")
        ax.plot(log["epoch"], log["val_mse"], color=color, lw=1.6, ls="--", label="val")
        best_idx = int(np.argmin(log["val_mse"]))
        ax.axvline(log["epoch"].iloc[best_idx], color="k", lw=0.6, alpha=0.3)
        ax.scatter(
            [log["epoch"].iloc[best_idx]],
            [log["val_mse"].iloc[best_idx]],
            color=color, edgecolor="k", lw=0.8, zorder=5, s=40,
        )
        ax.set_title(PRETTY_LABEL[key], fontsize=10)
        ax.set_ylabel("MSE")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.suptitle("Train / test MSE per GNN variant", y=1.005)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Multi-model comparison curves                                               #
# --------------------------------------------------------------------------- #

def plot_metric_overlays(
    runs: Mapping[str, Mapping[str, object]],
    metrics: tuple[str, ...] = ("val_mse", "val_mae", "val_pearson", "val_sign_acc"),
    metric_titles: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """One subplot per metric; all 4 models overlaid for direct comparison."""
    titles = dict({
        "val_mse": "Validation MSE (lower is better)",
        "val_mae": "Validation MAE (lower is better)",
        "val_rmse": "Validation RMSE (lower is better)",
        "val_r2": "Validation R² (higher is better)",
        "val_pearson": "Validation Pearson r (higher is better)",
        "val_spearman": "Validation Spearman r (higher is better)",
        "val_sign_acc": "Validation sign accuracy (higher is better)",
        "mean_grad_norm": "Mean gradient L2 norm",
    }, **(metric_titles or {}))

    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
    flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax, metric in zip(flat, metrics):
        for key in MODEL_KEYS:
            bundle = runs.get(key)
            if bundle is None:
                continue
            log = bundle["log"]
            if metric not in log.columns:
                continue
            ax.plot(
                log["epoch"], log[metric],
                color=MODEL_COLORS[key], lw=1.6, label=PRETTY_LABEL[key],
            )
        ax.set_title(titles.get(metric, metric), fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    for ax in list(flat)[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Final-epoch validation diagnostics                                          #
# --------------------------------------------------------------------------- #

def plot_pred_vs_target_grid(
    runs: Mapping[str, Mapping[str, object]],
    figsize: tuple[float, float] = (16, 20),
) -> Figure:
    """5 x 4 scatter grid of best-epoch test predictions vs targets per GNN variant."""
    fig, axes = plt.subplots(5, 4, figsize=figsize)
    for ax, key in zip(axes.flat, MODEL_KEYS):
        bundle = runs.get(key)
        if bundle is None or "val_preds" not in bundle:
            ax.set_title(f"{PRETTY_LABEL[key]} — (no val preds)", fontsize=10)
            ax.axis("off")
            continue
        vp = bundle["val_preds"]
        color = MODEL_COLORS[key]
        ax.scatter(vp["target"], vp["pred"], color=color, alpha=0.25, s=10)
        lo = float(min(vp["target"].min(), vp["pred"].min()))
        hi = float(max(vp["target"].max(), vp["pred"].max()))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--", alpha=0.5)
        r = float(np.corrcoef(vp["target"], vp["pred"])[0, 1])
        ax.set_title(f"{PRETTY_LABEL[key]} — r = {r:+.3f}", fontsize=10)
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        ax.grid(alpha=0.3)
    fig.suptitle("Best-epoch val predictions vs targets", y=1.005)
    fig.tight_layout()
    return fig


def plot_residual_hists(
    runs: Mapping[str, Mapping[str, object]],
    figsize: tuple[float, float] = (16, 18),
    bins: int = 60,
) -> Figure:
    """5 x 4 histograms of (pred - target) residuals at the best epoch per GNN variant."""
    fig, axes = plt.subplots(5, 4, figsize=figsize, sharex=True)
    for ax, key in zip(axes.flat, MODEL_KEYS):
        bundle = runs.get(key)
        if bundle is None or "val_preds" not in bundle:
            ax.set_title(f"{PRETTY_LABEL[key]} — (no val preds)", fontsize=10)
            ax.axis("off")
            continue
        vp = bundle["val_preds"]
        residuals = vp["residual"].to_numpy()
        color = MODEL_COLORS[key]
        ax.hist(residuals, bins=bins, color=color, alpha=0.75, edgecolor="white", lw=0.4)
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(
            f"{PRETTY_LABEL[key]} — μ = {residuals.mean():+.3f}, σ = {residuals.std():.3f}",
            fontsize=10,
        )
        ax.set_xlabel("pred − target")
        ax.grid(alpha=0.3)
    fig.suptitle("Residual distributions (best epoch)", y=1.005)
    fig.tight_layout()
    return fig


def plot_calibration(
    runs: Mapping[str, Mapping[str, object]],
    n_bins: int = 12,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 6),
) -> Figure:
    """Per-decile mean target vs mean prediction (all 4 models on one axes)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    for key in MODEL_KEYS:
        bundle = runs.get(key)
        if bundle is None or "val_preds" not in bundle:
            continue
        vp = bundle["val_preds"]
        order = np.argsort(vp["pred"].to_numpy())
        preds_sorted = vp["pred"].to_numpy()[order]
        targets_sorted = vp["target"].to_numpy()[order]
        bin_edges = np.linspace(0, len(preds_sorted), n_bins + 1, dtype=int)
        xs, ys = [], []
        for a, b in zip(bin_edges[:-1], bin_edges[1:]):
            if b <= a:
                continue
            xs.append(float(preds_sorted[a:b].mean()))
            ys.append(float(targets_sorted[a:b].mean()))
        ax.plot(xs, ys, "o-", color=MODEL_COLORS[key], lw=1.4, label=PRETTY_LABEL[key])
    span = ax.get_xlim()
    ax.plot(span, span, "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("mean prediction in bin")
    ax.set_ylabel("mean target in bin")
    ax.set_title("Reliability: mean target vs mean prediction by decile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


# --------------------------------------------------------------------------- #
# Summary table                                                               #
# --------------------------------------------------------------------------- #

def summary_table(runs: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Best-epoch metrics per model, indexed by pretty label."""
    rows: list[dict[str, object]] = []
    for key in MODEL_KEYS:
        bundle = runs.get(key)
        if bundle is None:
            continue
        log = bundle["log"]
        best_idx = int(log["val_mse"].idxmin())
        best = log.iloc[best_idx]
        rows.append({
            "model": PRETTY_LABEL[key],
            "best epoch": int(best["epoch"]),
            "val MSE": float(best["val_mse"]),
            "val MAE": float(best["val_mae"]),
            "val R²": float(best["val_r2"]),
            "val Pearson r": float(best["val_pearson"]),
            "val sign acc": float(best["val_sign_acc"]),
        })
    return pd.DataFrame(rows).set_index("model").round(4)
