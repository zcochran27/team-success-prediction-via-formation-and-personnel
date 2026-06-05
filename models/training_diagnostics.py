"""Training diagnostics for the 8 HalfSubsGNN runs.

Each call to the GNN training script writes one log.jsonl next to the run's
summary.json. Each line is one epoch with keys: epoch, train_mse, val_mse,
val_mae, val_rmse, val_r2, val_pearson, val_spearman, val_sign_acc,
mean_grad_norm, lr, epoch_seconds, improved.

This module loads those logs and produces the plots the training-diagnostics
notebook renders. Tabular runs (tab_*, tab_subs_*) write only summary.json
and a test_preds.parquet, no per-epoch log, so per-epoch curves only exist
for the 8 GNN variants.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


_HALF_SUBS_ROOT = Path("artifacts/half_subs")


# Loading

def load_run_log(run_dir: Path) -> pd.DataFrame:
    """Return one DataFrame with one row per epoch."""
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines() if line]
    return pd.DataFrame(rows)


def load_all_logs(artifacts_root: Path = _HALF_SUBS_ROOT) -> dict[str, pd.DataFrame]:
    """Return {run_name: per-epoch DataFrame} for every gnn_subs_* run."""
    out: dict[str, pd.DataFrame] = {}
    for d in sorted(artifacts_root.glob("gnn_subs_*")):
        if (d / "log.jsonl").exists():
            out[d.name] = load_run_log(d)
    return out


def _short_label(run_name: str) -> str:
    """gnn_subs_archetype_paired_coords_stats -> arch · paired · stats."""
    rest = run_name[len("gnn_subs_"):]
    parts = rest.split("_")
    kind = "arch" if parts[0] == "archetype" else "pos"
    mode = parts[1]                          # 'single' | 'paired'
    has_stats = parts[-1] == "stats"
    base = f"{kind} · {mode}"
    return base + (" · stats" if has_stats else "")


def _style(run_name: str) -> dict:
    """Color by kind, linestyle by mode, alpha by stats - so the legend reads
    as a 2x2x2 grid rather than 8 arbitrary lines."""
    rest = run_name[len("gnn_subs_"):]
    parts = rest.split("_")
    kind = parts[0]
    mode = parts[1]
    has_stats = parts[-1] == "stats"
    color = "#1f77b4" if kind == "position" else "#d62728"   # blue / red
    ls = "-" if mode == "single" else "--"
    return {"color": color, "linestyle": ls, "alpha": 0.95 if has_stats else 0.45,
            "linewidth": 1.6 if has_stats else 1.1}


# Plots

def plot_loss_curves(
    logs: Mapping[str, pd.DataFrame],
    figsize: tuple[float, float] = (12, 4.4),
) -> Figure:
    """Two subplots: train MSE per epoch, val MSE per epoch. Best epoch marked."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for run_name, df in logs.items():
        style = _style(run_name)
        label = _short_label(run_name)
        axes[0].plot(df["epoch"], df["train_mse"], label=label, **style)
        axes[1].plot(df["epoch"], df["val_mse"],   label=label, **style)
        best_ep = int(df["val_mse"].idxmin()) + 1
        best_val = float(df["val_mse"].min())
        axes[1].scatter([best_ep], [best_val], color=style["color"],
                        edgecolor="black", s=42, zorder=5)
    axes[0].set_title("Train MSE per epoch"); axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train MSE"); axes[0].grid(alpha=0.3)
    axes[1].set_title("Val MSE per epoch - black-rim dot = best epoch")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val MSE"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def plot_metric_overlays(
    logs: Mapping[str, pd.DataFrame],
    metrics: tuple[str, ...] = ("val_r2", "val_pearson", "val_sign_acc"),
    figsize: tuple[float, float] = (14, 4.2),
) -> Figure:
    """One subplot per validation metric, all 8 runs overlaid."""
    titles = {
        "val_r2":       "Validation R²",
        "val_pearson":  "Validation Pearson r",
        "val_spearman": "Validation Spearman r",
        "val_sign_acc": "Validation sign accuracy",
        "val_mae":      "Validation MAE",
        "val_rmse":     "Validation RMSE",
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for run_name, df in logs.items():
        style = _style(run_name)
        for ax, m in zip(axes, metrics):
            if m in df.columns:
                ax.plot(df["epoch"], df[m], label=_short_label(run_name), **style)
    for ax, m in zip(axes, metrics):
        ax.set_title(titles.get(m, m), fontsize=10)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    return fig


def plot_grad_norms(
    logs: Mapping[str, pd.DataFrame],
    figsize: tuple[float, float] = (8, 4.2),
) -> Figure:
    """Mean-gradient-norm trajectory - flags exploding / vanishing dynamics."""
    fig, ax = plt.subplots(figsize=figsize)
    for run_name, df in logs.items():
        if "mean_grad_norm" not in df.columns:
            continue
        ax.plot(df["epoch"], df["mean_grad_norm"],
                label=_short_label(run_name), **_style(run_name))
    ax.set_title("Mean gradient norm per epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖∇‖"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_epoch_timing(
    logs: Mapping[str, pd.DataFrame],
    figsize: tuple[float, float] = (8, 4.2),
) -> Figure:
    """Wall-clock seconds per epoch - flags machine contention / GPU thrash."""
    fig, ax = plt.subplots(figsize=figsize)
    for run_name, df in logs.items():
        if "epoch_seconds" not in df.columns:
            continue
        ax.plot(df["epoch"], df["epoch_seconds"],
                label=_short_label(run_name), **_style(run_name))
    ax.set_title("Wall-clock seconds per epoch")
    ax.set_xlabel("epoch"); ax.set_ylabel("seconds"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


def best_epoch_table(logs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-run summary: best epoch (by val MSE), and the metrics at that epoch."""
    rows = []
    for run_name, df in logs.items():
        idx = int(df["val_mse"].idxmin())
        row = df.iloc[idx]
        rows.append({
            "run":          _short_label(run_name),
            "best epoch":   int(row["epoch"]),
            "train MSE":    float(row["train_mse"]),
            "val MSE":      float(row["val_mse"]),
            "val MAE":      float(row.get("val_mae",   np.nan)),
            "val R²":       float(row.get("val_r2",    np.nan)),
            "val Pearson":  float(row.get("val_pearson", np.nan)),
            "val sign acc": float(row.get("val_sign_acc", np.nan)),
            "epochs run":   int(df["epoch"].max()),
            "total minutes": float(df["epoch_seconds"].sum()) / 60.0,
        })
    return pd.DataFrame(rows).sort_values("val MSE").reset_index(drop=True)
