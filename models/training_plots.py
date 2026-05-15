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
)

# Stable palette so the same model gets the same color in every panel.
MODEL_COLORS: dict[str, str] = {
    "position_single":  "#1f78b4",  # blue
    "archetype_single": "#33a02c",  # green
    "position_paired":  "#ff7f00",  # orange
    "archetype_paired": "#e31a1c",  # red
}

PRETTY_LABEL: dict[str, str] = {
    "position_single":  "position, single",
    "archetype_single": "archetype, single",
    "position_paired":  "position, paired",
    "archetype_paired": "archetype, paired",
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
    figsize: tuple[float, float] = (10, 7),
) -> Figure:
    """2 x 2 grid of train vs val MSE curves, one panel per model."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
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
    fig.suptitle("Train / val MSE per model", y=1.005)
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
    figsize: tuple[float, float] = (10, 9),
) -> Figure:
    """2 x 2 scatter grid of best-epoch val predictions vs targets."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
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
    figsize: tuple[float, float] = (10, 7),
    bins: int = 60,
) -> Figure:
    """2 x 2 histograms of (pred - target) residuals at the best epoch."""
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
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
