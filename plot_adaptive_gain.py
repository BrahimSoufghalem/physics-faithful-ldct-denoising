"""Analyze the v3 ADAPTIVE spectral head: per-image gain curves by anatomy.

The v3 head predicts a bounded per-image offset to the shared radial gain
curve from the LDCT input (see spectral_head.py). This script measures
whether it actually learned ANATOMY-DEPENDENT spectral behavior: it runs
the conditioning encoder on real test slices and plots the per-image gain
curves G_i(|f|) grouped by body region (chest vs abdomen).

Usage
-----
    HU_RANGE_PRESET=benchmark python plot_adaptive_gain.py \\
        --checkpoint runs_S4/redcnn/best_model.pt --arch redcnn \\
        --test-dir test --output adaptive_gain_S4

Produces ``<output>.png`` and ``<output>.csv`` and prints a quantitative
separation summary.

Reading the figure
------------------
* If the chest and abdomen bands clearly separate (between-anatomy gap
  larger than the within-anatomy spread), the head learned
  anatomy-dependent physics: this is the interpretability figure for the
  adaptive-head claim.
* If the bands overlap everywhere (max separation < ~0.01), the adaptive
  offsets are not anatomy-driven and the adaptive claim has no support.

Works on static-head checkpoints too (all curves collapse onto the shared
curve), which gives a null reference.
"""

import argparse
import csv
from glob import glob
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from benchmark_data import standardize_hu
from evaluate_image import get_test_set, load_checkpoint
from utils import (
    load_dicom_tensor, setup_reproducibility, get_device,
    sort_by_instance_number,
)


@torch.no_grad()
def collect_curves(model, test_dir, test_ids, slices_per_patient, device):
    head = model.head
    curves = {"Chest": [], "Abdomen": []}
    patients = sorted([
        d for d in Path(test_dir).iterdir()
        if d.is_dir() and d.name in test_ids and (d / "Low_Dose").exists()
    ])
    if not patients:
        raise RuntimeError(f"No test patients found in '{test_dir}'.")
    for d in patients:
        low = sort_by_instance_number(glob(str(d / "Low_Dose" / "*.dcm")))
        if not low:
            continue
        body = "Chest" if d.name.upper().startswith("C") else "Abdomen"
        step = max(1, len(low) // slices_per_patient)
        picked = low[::step][:slices_per_patient]
        for path in picked:
            low_hu = load_dicom_tensor(path).to(device)
            x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
            g = head.per_image_gain_curves(x)[0].detach().cpu().numpy()
            curves[body].append(g)
        print(f"  {d.name} ({body}): {len(picked)} slices")
    return curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="best_model.pt trained with --use-spectral-head")
    p.add_argument("--arch", default="redcnn")
    p.add_argument("--test-dir", default=cfg.TEST_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--slices-per-patient", type=int, default=20,
                   help="Evenly spaced slices sampled per patient.")
    p.add_argument("--output", default="adaptive_gain",
                   help="Output basename (writes <output>.png and <output>.csv)")
    args = p.parse_args()

    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError(
            "Run with HU_RANGE_PRESET=benchmark.\n"
            "Example: HU_RANGE_PRESET=benchmark python plot_adaptive_gain.py ..."
        )

    setup_reproducibility()
    device = get_device()
    model = load_checkpoint(args.checkpoint, args.arch, device)
    head = getattr(model, "head", None)
    if head is None:
        raise RuntimeError(
            "No spectral head in this checkpoint. "
            "Train with --use-spectral-head first."
        )
    if not getattr(head, "adaptive", False):
        print("NOTE: STATIC head -- every per-image curve equals the shared "
              "curve; expect zero separation (null reference).")

    test_ids = get_test_set(args.split)
    curves = collect_curves(model, args.test_dir, test_ids,
                            args.slices_per_patient, device)

    n_bins = int(head.n_bins)
    r = np.linspace(0.0, np.sqrt(2.0), n_bins)
    shared = head.gain_curve().detach().cpu().numpy()

    stats = {}
    plt.figure(figsize=(6.8, 4.4))
    colors = {"Chest": "#d62728", "Abdomen": "#1f6feb"}
    for body in ("Chest", "Abdomen"):
        if not curves[body]:
            continue
        arr = np.stack(curves[body])            # (N, n_bins)
        stats[body] = (arr.mean(axis=0), arr.std(axis=0), arr.shape[0])
        mean, std, n = stats[body]
        plt.plot(r, mean, color=colors[body], lw=1.8,
                 label=f"{body} mean (n={n})")
        plt.fill_between(r, mean - std, mean + std,
                         color=colors[body], alpha=0.18,
                         label=f"{body} +/- 1 std")
    plt.plot(r, shared, color="gray", ls="--", lw=1.2,
             label="shared base curve")
    plt.axhline(1.0, color="black", lw=0.8, ls=":", label="G = 1 (identity)")
    plt.xlabel("Radial spatial frequency |f|  (fraction of Nyquist)")
    plt.ylabel("Per-image gain G_i(|f|)")
    plt.title("v3 adaptive spectral head: per-image gain by anatomy")
    plt.legend(loc="best", fontsize=7)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output + ".png", dpi=200)

    with open(args.output + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radial_freq_nyquist", "shared_gain",
                    "chest_mean", "chest_std", "abdomen_mean", "abdomen_std"])
        for i in range(n_bins):
            row = [f"{r[i]:.6f}", f"{shared[i]:.8f}"]
            for body in ("Chest", "Abdomen"):
                if body in stats:
                    row += [f"{stats[body][0][i]:.8f}",
                            f"{stats[body][1][i]:.8f}"]
                else:
                    row += ["", ""]
            w.writerow(row)

    print(f"\nSaved {args.output}.png and {args.output}.csv")
    if "Chest" in stats and "Abdomen" in stats:
        sep = np.abs(stats["Chest"][0] - stats["Abdomen"][0])
        within = 0.5 * (stats["Chest"][1] + stats["Abdomen"][1])
        print(f"Anatomy separation |G_chest - G_abd|: "
              f"max = {sep.max():.4f} (at r = {r[sep.argmax()]:.3f} Nyquist), "
              f"mean = {sep.mean():.4f}")
        print(f"Mean within-anatomy std (spread)    : {within.mean():.4f}")
        if sep.max() < 0.01:
            print("VERDICT: no meaningful anatomy separation -- the adaptive "
                  "offsets are not anatomy-driven.")
        elif sep.max() > 2.0 * max(1e-8, float(within.mean())):
            print("VERDICT: clear anatomy separation (between-anatomy gap >> "
                  "within-anatomy spread) -- the head learned "
                  "anatomy-dependent spectral behavior.")
        else:
            print("VERDICT: some separation, but comparable to the "
                  "within-anatomy spread -- weak evidence.")


if __name__ == "__main__":
    main()
