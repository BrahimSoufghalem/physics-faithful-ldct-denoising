"""Plot the accuracy/texture trade-off (Pareto) curve from run results.

Fill a small CSV by hand from your evaluate_image.py + evaluate_physics.py
tables (overall rows), one line per training run:

    label,nps_weight,psnr,ssim,vif,nps_logl1
    C0 (w=0),0,30.72,0.7448,0.3161,0.4797
    S4 (w=0.01),0.01,30.42,0.7327,0.3043,0.3390
    S2 (w=0.05),0.05,29.95,0.7177,0.2920,0.2096

Usage
-----
    python plot_pareto.py --csv pareto_points.csv --output pareto

Produces <output>.png with two panels: PSNR vs NPS_LogL1 and VIF vs
NPS_LogL1. The ideal corner is top-left (high accuracy, low spectral
distance).
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="pareto")
    args = p.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "label": row["label"],
                "w": float(row["nps_weight"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "vif": float(row["vif"]),
                "nps": float(row["nps_logl1"]),
            })
    if len(rows) < 2:
        raise RuntimeError("Need at least two runs in the CSV")
    rows.sort(key=lambda r: r["w"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    panels = [("psnr", "PSNR (dB)"), ("vif", "VIF")]
    for ax, (key, ylabel) in zip(axes, panels):
        xs = [r["nps"] for r in rows]
        ys = [r[key] for r in rows]
        ax.plot(xs, ys, "-o", color="#1f6feb", ms=6)
        for r in rows:
            ax.annotate(r["label"], (r["nps"], r[key]), fontsize=8,
                        textcoords="offset points", xytext=(6, 6))
        ax.set_xlabel("NPS_LogL1 (spectral distance, lower = better texture)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        # Ideal direction: top-left corner.
        ax.annotate("ideal", xy=(0.03, 0.93), xycoords="axes fraction",
                    fontsize=9, color="green")
        ax.annotate("", xy=(0.02, 0.98), xytext=(0.14, 0.86),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="green"))
    fig.suptitle("Pixel-accuracy vs noise-texture fidelity trade-off "
                 "(NPS weight sweep)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.output + ".png", dpi=200)
    print(f"Saved {args.output}.png")


if __name__ == "__main__":
    main()
