# report/

LaTeX outline for the DSC 148 final report.

| File | Role |
|---|---|
| [report.tex](report.tex) | Section-by-section outline with topic sentences, TODO bullets, and figure/table stubs. Compiles directly — figure includes point at the actual PNGs in `../artifacts/half_subs/_comparison_plots_v2/`. |
| [figures/](figures/) | Standalone figures included in the report. Pre-populated with `graph_single_team1.png` and `graph_paired.png` (rendered from `features.build_graphs_subs` + `graphs.visualize_half_subs`). The preamble adds this directory to `\graphicspath`. |
| [tables/](tables/) | Auto-generated LaTeX tables included via `\input{tables/...}`. `leaderboard.tex` is the full 16-run head-to-head; `ablation_summary.tex` is the one-row-per-design-axis mean improvement. Regenerate by running the snippet at the top of `report.tex` (see `scripts/` or rerun `models.comparison_half_subs`). |

## Build

```
cd report/
pdflatex report.tex
bibtex   report
pdflatex report.tex
pdflatex report.tex
```

The bibliography is a placeholder `thebibliography` block; replace with a real `.bib` file once citations are decided.

## Outline structure

| Section | Status | Notes |
|---|---|---|
| Abstract | TODO content | One paragraph; current placeholder includes the headline R²=0.283 result. |
| 1. Introduction | TODO content | Problem framing, 4 research questions, contributions. |
| 2. Related Work | TODO content | xG modeling, GNNs in soccer, player-role clustering. |
| 3. Data | TODO content | Wyscout NCAA D1, target variable definition. |
| 4. Player Archetype Construction | TODO content | The two-stage k-means pipeline. Mostly already written in `../PROJECT_DESCRIPTION.md` §4. |
| 5. Snapshot Construction | TODO content | Three-iteration journey: joint-stable → sub-cluster collapse → per-half sub-aware. |
| 6. Model Families | TODO content | `tab`, `tab_subs`, `gnn_subs`. Architecture details for `HalfSubsGNN`. |
| 7. Experiments | partial | Headline leaderboard table is **filled in** with real numbers from the comparison run; ablations are TODO. |
| 8. Discussion | TODO content | Three headline findings + limitations + future work. |
| 9. Conclusion | TODO content | One paragraph. |
| Appendix A | TODO content | Full 24-row leaderboard (auto-generate via `df.to_latex()`). |
| Appendix B | TODO content | Hyperparameters. |
| Appendix C | TODO content | Per-archetype profile plots. |

`\todo{...}` macro highlights placeholder content in red so unfilled sections are easy to spot when rendering.
