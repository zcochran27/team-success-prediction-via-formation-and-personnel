# report/

LaTeX source for the DSC 148 final report.

| File | Role |
|---|---|
| [report.tex](report.tex) | The full two-column report (11pt). Figure includes point at the PNGs in `../artifacts/half_subs/_comparison_plots_v2/` and at `figures/`. |
| [figures/](figures/) | Standalone figures. Includes `graph_single_team1.png` and `graph_paired.png`, rendered from `features.build_graphs_subs` and `graphs.visualize_half_subs`. |
| [tables/](tables/) | Auto-generated LaTeX tables included via `\input{tables/...}`. `leaderboard.tex` is the 16-run head-to-head; `ablation_summary.tex` is the one-row-per-design-axis mean improvement. Regenerate from `models.comparison_half_subs`. |

## Build

```
cd report/
pdflatex report.tex
pdflatex report.tex
```

The second pass resolves the citation and reference numbers. The bibliography
is an inline `thebibliography` block, so there is no separate bibtex step.
