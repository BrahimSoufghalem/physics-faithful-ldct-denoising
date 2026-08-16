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

## ⚠️ Required figure files (add manually)

The PNG figures exist only as chat/Notion attachments and must be copied
from your local machine into `paper/figs/` with EXACTLY these names:

| File | Content |
| --- | --- |
| `figs/fig2_nps_chest.png` | Chest (C121) radial NPS, removed vs reference noise |
| `figs/fig3_nps_abdomen.png` | Abdomen (L006) radial NPS, removed vs reference noise |
| `figs/fig4_gain_curves_s4b.png` | S4b per-image adaptive gain curves by anatomy (RED-CNN) |
| `figs/fig5_gain_curves_s4_naive.png` | S4 naive-adaptive saturation diagnostic |
| `figs/fig6_qualitative_chest.png` | Chest qualitative panel (C121, slice 196) |
| `figs/fig7_qualitative_abdomen.png` | Abdomen qualitative panel (L006, slice 107) |
| `figs/fig8_gain_curves_resnet.png` | S4b-on-ResNet gain curves (cross-trunk diagnosis) |

Fig. 1 needs no file — it is drawn in TikZ.

## Remaining before submission

- [ ] Copy the 7 PNGs into `paper/figs/` (names above) and build
- [ ] Final visual pass on Fig. 1 TikZ spacing after first compile
- [ ] Decide arXiv category (eess.IV primary, cs.CV cross-list suggested)
- [ ] Make repository public at submission
