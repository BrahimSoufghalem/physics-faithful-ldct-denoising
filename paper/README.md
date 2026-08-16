# Paper — LaTeX source (arXiv two-column draft)

Mirrors the Notion draft v0.3 ("00. Full Paper — arXiv Draft").

## Build

```bash
cd paper
pdflatex main
pdflatex main   # second pass for cross-references
```

No BibTeX needed — references are embedded (`sections/references.tex`).

## Structure

- `main.tex` — preamble, title/author, inputs all sections
- `sections/0_abstract.tex` … `sections/6_discussion.tex` — paper body
- `sections/references.tex` — 12 verified references (`thebibliography`)
- `figures/fig1_architecture.tex` — Fig. 1 architecture schematic (TikZ, compiled inline)

## Figure files

The required PNG figures are stored in `paper/figures/`:

| File | Content |
| --- | --- |
| `figures/nps_C121.png` | Chest (C121) radial NPS, removed vs reference noise |
| `figures/nps_L006.png` | Abdomen (L006) radial NPS, removed vs reference noise |
| `figures/adaptive_gain_S4b.png` | S4b per-image adaptive gain curves by anatomy (RED-CNN) |
| `figures/adaptive_gain_S4.png` | S4 naive-adaptive saturation diagnostic |
| `figures/compare_C121_s196.png` | Chest qualitative panel (C121, slice 196) |
| `figures/compare_L006_s107.png` | Abdomen qualitative panel (L006, slice 107) |
| `figures/adaptive_gain_S4b_resnet.png` | S4b-on-ResNet gain curves (cross-trunk diagnosis) |

Fig. 1 needs no PNG file — it is drawn in TikZ.

## Remaining before submission

- [ ] Build the paper twice with `pdflatex` and inspect the resulting PDF
- [ ] Final visual pass on Fig. 1 TikZ spacing after first compile
- [ ] Decide arXiv category (eess.IV primary, cs.CV cross-list suggested)
- [ ] Make repository public at submission
