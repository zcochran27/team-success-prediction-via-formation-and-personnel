"""Cross-model comparison for the half-subs pipeline.

Loads every artifact under ``artifacts/half_subs/`` and produces:

  * ``build_summary_table(...)`` -- one DataFrame row per model with
    MAE / RMSE / R² / sign-acc on the shared held-out test parquet
    (``test_snapshots_half_subs.parquet``), tagged by model family and
    feature blocks.
  * ``plot_metric_bars(summary, ...)`` -- bar chart per metric.
  * ``plot_pred_vs_target(...)``  -- scatter grid by model.
  * ``plot_residual_hists(...)``  -- residual histograms by model.

Three model families live side-by-side under ``artifacts/half_subs/``:

  1. **tab** -- XGBoost on the 8 starter-only feature configurations
     (position vs archetype × ego vs matchup × ±stats). Folder names
     ``tab_<kind>_<opp>[_stats]``.
  2. **tab_subs** -- XGBoost on the 8 sub-aware feature configurations
     (same axes, but the feature set also exposes the 11 sub slots'
     position/archetype/start_min/duration/stats). Folder names
     ``tab_subs_<kind>_<opp>[_stats]``.
  3. **gnn_subs** -- HalfSubsGNN runs across the same 8 configurations
     (kind × mode × ±stats), with typed edges (formation / sub /
     time-overlap matchup) and node features
     (is_starter / duration / start_min / end_min / was_subbed_off).
     Folder names ``gnn_subs_<kind>_<mode>_coords[_stats]``.

Every model writes a ``summary.json`` and a per-test-row predictions
parquet alongside it -- the loader pulls metrics from ``summary.json``
and the ``test_preds.parquet`` / ``val_preds.parquet`` file for the
residual / scatter plots.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


_HALF_SUBS_ROOT = Path("artifacts/half_subs")


@dataclass(frozen=True)
class ModelRun:
    """Summary of one trained model + its held-out predictions."""

    key: str
    family: str            # "tab" | "tab_subs" | "gnn_subs"
    pretty: str
    kind: str              # "position" | "archetype"
    opponent: str          # "ego" | "matchup"  (paired/single for GNN -> ego/matchup)
    use_stats: bool
    metrics: Mapping[str, float]
    test_preds: np.ndarray | None
    test_targets: np.ndarray | None


# --------------------------------------------------------------------------- #
# Pretty labels                                                               #
# --------------------------------------------------------------------------- #

_FAMILY_PRETTY: dict[str, str] = {
    "tab":      "tab",
    "tab_subs": "tab-subs",
    "gnn_subs": "gnn-subs",
}


def _pretty(family: str, kind: str, opp: str, use_stats: bool) -> str:
    k = "pos" if kind == "position" else "arch"
    base = f"{_FAMILY_PRETTY[family]} · {k} · {opp}"
    return base + (" · stats" if use_stats else "")


# --------------------------------------------------------------------------- #
# Folder name parsing                                                         #
# --------------------------------------------------------------------------- #

def _parse_tab_name(name: str, *, family: str) -> tuple[str, str, bool] | None:
    """Parse ``tab_<kind>_<opp>[_stats]`` or ``tab_subs_<kind>_<opp>[_stats]``.

    Returns ``(kind, opp, use_stats)`` or ``None`` if it doesn't match.
    """
    prefix = f"{family}_"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    parts = rest.split("_")
    if len(parts) < 2:
        return None
    kind_short = parts[0]                                # 'pos' | 'arch'
    opp = parts[1]                                       # 'ego' | 'matchup'
    use_stats = (len(parts) >= 3 and parts[-1] == "stats")
    if kind_short not in ("pos", "arch") or opp not in ("ego", "matchup"):
        return None
    kind = "position" if kind_short == "pos" else "archetype"
    return kind, opp, use_stats


def _parse_gnn_subs_name(name: str) -> tuple[str, str, bool] | None:
    """Parse ``gnn_subs_<kind>_<mode>_coords[_stats]``.

    Returns ``(kind, opp, use_stats)`` where ``opp = "matchup"`` if the
    GNN was paired, ``"ego"`` if single -- matching the tabular axis.
    """
    if not name.startswith("gnn_subs_"):
        return None
    rest = name[len("gnn_subs_"):]
    parts = rest.split("_")
    if len(parts) < 3:
        return None
    kind = parts[0]                                      # 'position' | 'archetype'
    mode = parts[1]                                      # 'single' | 'paired'
    if kind not in ("position", "archetype") or mode not in ("single", "paired"):
        return None
    use_stats = parts[-1] == "stats"
    opp = "matchup" if mode == "paired" else "ego"
    return kind, opp, use_stats


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #

def _load_preds(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return ``(preds, targets)`` from either ``test_preds.parquet`` or
    ``val_preds.parquet``, whichever exists. Both schemas carry ``target``
    and ``pred`` columns.
    """
    for fname in ("test_preds.parquet", "val_preds.parquet"):
        p = run_dir / fname
        if p.exists():
            df = pd.read_parquet(p)
            if "target" in df.columns and "pred" in df.columns:
                return df["pred"].to_numpy(), df["target"].to_numpy()
    return None, None


def _gnn_metric_keys(metrics: dict[str, Any]) -> dict[str, float]:
    """GNN summary uses ``best_val_<metric>``; tabular uses bare ``<metric>``."""
    out: dict[str, float] = {}
    for short, src in (
        ("mae", "best_val_mae"),
        ("rmse", "best_val_rmse"),
        ("r2", "best_val_r2"),
        ("sign_acc", "best_val_sign_acc"),
    ):
        if src in metrics:
            out[short] = float(metrics[src])
    if "best_val_mse" in metrics and "rmse" not in out:
        out["rmse"] = float(np.sqrt(metrics["best_val_mse"]))
    return out


def _tab_metric_keys(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "mae":      float(metrics["mae"]),
        "rmse":     float(metrics["rmse"]),
        "r2":       float(metrics["r2"]),
        "sign_acc": float(metrics["sign_acc"]),
    }


def load_runs(artifacts_root: Path = _HALF_SUBS_ROOT) -> dict[str, ModelRun]:
    """Discover every ``<family>_…/summary.json`` in ``artifacts_root``.

    Returns ``{key: ModelRun}``. ``key`` is the folder name; the run's
    family is inferred from the folder prefix.
    """
    runs: dict[str, ModelRun] = {}
    for d in sorted(artifacts_root.iterdir()):
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())

        family: str | None = None
        parsed: tuple[str, str, bool] | None = None
        # Match longer prefix first so 'tab_subs_' doesn't get caught by 'tab_'.
        for fam in ("tab_subs", "gnn_subs", "tab"):
            if d.name.startswith(fam + "_"):
                family = fam
                if fam == "gnn_subs":
                    parsed = _parse_gnn_subs_name(d.name)
                else:
                    parsed = _parse_tab_name(d.name, family=fam)
                break
        if family is None or parsed is None:
            continue
        kind, opp, use_stats = parsed

        metrics = (
            _gnn_metric_keys(summary) if family == "gnn_subs"
            else _tab_metric_keys(summary)
        )
        preds, targets = _load_preds(d)
        runs[d.name] = ModelRun(
            key=d.name,
            family=family,
            pretty=_pretty(family, kind, opp, use_stats),
            kind=kind,
            opponent=opp,
            use_stats=use_stats,
            metrics=metrics,
            test_preds=preds,
            test_targets=targets,
        )
    return runs


# --------------------------------------------------------------------------- #
# Summary table + plots                                                       #
# --------------------------------------------------------------------------- #

def build_summary_table(runs: Mapping[str, ModelRun]) -> pd.DataFrame:
    """One row per run, sorted by MSE (lower-is-better). MSE is RMSE²."""
    rows: list[dict[str, Any]] = []
    for r in runs.values():
        rmse = r.metrics.get("rmse", float("nan"))
        rows.append({
            "model":    r.pretty,
            "key":      r.key,
            "family":   r.family,
            "kind":     r.kind,
            "opponent": r.opponent,
            "use_stats": r.use_stats,
            "MAE":      r.metrics.get("mae", float("nan")),
            "RMSE":     rmse,
            "MSE":      rmse * rmse if not np.isnan(rmse) else float("nan"),
            "R²":       r.metrics.get("r2", float("nan")),
            "sign acc": r.metrics.get("sign_acc", float("nan")),
        })
    df = pd.DataFrame(rows).sort_values("MSE").reset_index(drop=True)
    return df.set_index("model")


_FAMILY_COLOR: dict[str, str] = {
    "tab":      "#6baed6",  # blue
    "tab_subs": "#fd8d3c",  # orange
    "gnn_subs": "#74c476",  # green
}


def plot_metric_bars(
    summary: pd.DataFrame,
    metrics: tuple[str, ...] = ("MAE", "RMSE", "R²", "sign acc"),
    figsize: tuple[float, float] = (14, 4),
) -> Figure:
    """One subplot per metric; bars colored by model family."""
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    labels = list(summary.index)
    colors = [_FAMILY_COLOR[f] for f in summary["family"]]
    for ax, metric in zip(axes, metrics):
        values = summary[metric].to_numpy()
        ax.bar(range(len(labels)), values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        higher_is_better = metric in ("R²", "sign acc")
        ax.set_title(
            f"{metric} ({'higher' if higher_is_better else 'lower'} is better)",
            fontsize=10,
        )
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Half-subs model comparison — shared held-out test set", y=1.04)
    fig.tight_layout()
    return fig


def plot_pred_vs_target(
    runs: Mapping[str, ModelRun],
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Scatter ``pred`` vs ``target`` per model, in a grid sorted by MSE."""
    runs_sorted = sorted(
        [r for r in runs.values() if r.test_preds is not None],
        key=lambda r: r.metrics.get("rmse", float("inf")),
    )
    n = len(runs_sorted)
    if n == 0:
        raise ValueError("no runs with prediction parquets found")
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (ncols * 3.4, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    for idx, r in enumerate(runs_sorted):
        ax = axes[idx // ncols, idx % ncols]
        p, t = r.test_preds, r.test_targets
        ax.scatter(t, p, color=_FAMILY_COLOR[r.family], alpha=0.25, s=8)
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, alpha=0.5)
        pearson = float(np.corrcoef(t, p)[0, 1]) if len(t) >= 2 else float("nan")
        ax.set_title(f"{r.pretty} — r = {pearson:+.3f}", fontsize=8)
        ax.grid(alpha=0.3)
        if idx % ncols == 0:
            ax.set_ylabel("prediction")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("target")
    # Hide unused axes.
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Predictions vs targets — half-subs models", y=1.005)
    fig.tight_layout()
    return fig


def plot_residual_hists(
    runs: Mapping[str, ModelRun],
    figsize: tuple[float, float] | None = None,
    bins: int = 60,
) -> Figure:
    """One residual histogram per model, same grid layout as the scatter plot."""
    runs_sorted = sorted(
        [r for r in runs.values() if r.test_preds is not None],
        key=lambda r: r.metrics.get("rmse", float("inf")),
    )
    n = len(runs_sorted)
    if n == 0:
        raise ValueError("no runs with prediction parquets found")
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (ncols * 3.4, nrows * 3.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)
    for idx, r in enumerate(runs_sorted):
        ax = axes[idx // ncols, idx % ncols]
        resid = r.test_preds - r.test_targets
        ax.hist(resid, bins=bins, color=_FAMILY_COLOR[r.family], alpha=0.8,
                edgecolor="white", lw=0.4)
        ax.axvline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(
            f"{r.pretty} — μ = {resid.mean():+.3f}, σ = {resid.std():.3f}",
            fontsize=8,
        )
        ax.grid(alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("pred − target")
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Residuals — half-subs models", y=1.005)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _cli() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--artifacts-root", type=Path, default=_HALF_SUBS_ROOT)
    p.add_argument("--save-plots", type=Path, default=None,
                   help="Optional dir; saves bars / scatter / residuals PNGs there.")
    args = p.parse_args()

    runs = load_runs(args.artifacts_root)
    summary = build_summary_table(runs)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(summary.to_string())

    if args.save_plots is not None:
        args.save_plots.mkdir(parents=True, exist_ok=True)
        plot_metric_bars(summary).savefig(args.save_plots / "metric_bars.png", dpi=150, bbox_inches="tight")
        plot_pred_vs_target(runs).savefig(args.save_plots / "pred_vs_target.png", dpi=150, bbox_inches="tight")
        plot_residual_hists(runs).savefig(args.save_plots / "residuals.png", dpi=150, bbox_inches="tight")
        print(f"plots -> {args.save_plots}")


if __name__ == "__main__":
    _cli()
