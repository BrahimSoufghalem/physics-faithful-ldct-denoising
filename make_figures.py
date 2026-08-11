"""Generate thesis-ready comparison figures with ROIs, diff maps and NPS curves.

For each requested patient it produces:
  1. compare_<pid>_s<idx>.png : rows = [windowed images | difference maps
     (pred - NDCT, in HU) | one zoom row per ROI]; columns = LDCT, each
     model, NDCT. ROI rectangles are drawn on the full images and each zoom
     is annotated with mean +/- std HU inside the ROI.
  2. roi_stats_<pid>_s<idx>.csv : per model x ROI mean HU, std HU and
     residual std vs NDCT (noise magnitude).
  3. nps_<pid>.png : radial noise-power-spectrum curves of the REMOVED noise
     (LDCT - pred) for each model vs the reference residual (LDCT - NDCT),
     averaged over the patient's slices. This is the visual counterpart of
     evaluate_physics.py's NPS_LogL1/NPS_Corr numbers.

Usage
-----
    HU_RANGE_PRESET=benchmark python make_figures.py \\
        --test-dir test \\
        --runs-roots runs_paper_C0_redcnn runs_paper_S2_spectral \\
        --labels "RED-CNN" "RED-CNN+Spectral" \\
        --patients C121 L006 \\
        --slice-frac 0.5 \\
        --rois 160,200,96 300,260,96 \\
        --output figures

Notes
-----
* --runs-roots point to training output roots (each containing
  <arch>/best_model.pt). Spectral-head checkpoints are rebuilt automatically
  from checkpoint meta via evaluate_image.load_checkpoint.
* ROIs are x,y,size in pixel coordinates of the 512x512 slice. Look at the
  first rendered figure, then adjust ROI positions to land on a homogeneous
  soft-tissue region and a detail-rich region (lung vessels / organ edge).
* Clinical windows: Chest c=-600/w=1500, Abdomen c=50/w=400.
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
from matplotlib.patches import Rectangle

import config as cfg
from benchmark_data import standardize_hu, denormalize_to_pixel
from evaluate_image import load_checkpoint
from utils import (
    load_dicom_tensor, get_device, setup_reproducibility,
    sort_by_instance_number,
)

WINDOWS = {"Chest": (-600.0, 1500.0), "Abdomen": (50.0, 400.0)}
ROI_COLORS = ["red", "yellow", "cyan", "lime"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", default=cfg.TEST_DIR)
    p.add_argument("--runs-roots", nargs="+", required=True,
                   help="One or more training output roots "
                        "(each containing <arch>/best_model.pt).")
    p.add_argument("--labels", nargs="+", required=True,
                   help="Display name for each runs-root (same order).")
    p.add_argument("--arch", default="redcnn",
                   choices=["redcnn", "resnet", "dugan", "wganvgg", "transct"])
    p.add_argument("--patients", nargs="+", required=True,
                   help="Patient IDs, e.g. C121 L006.")
    p.add_argument("--slice-frac", type=float, default=0.5,
                   help="Slice position as a fraction of the series (0-1).")
    p.add_argument("--slice-index", type=int, default=None,
                   help="Explicit slice index (overrides --slice-frac).")
    p.add_argument("--rois", nargs="+", default=["160,200,96", "300,260,96"],
                   help="ROIs as x,y,size in pixels.")
    p.add_argument("--output", default="figures")
    p.add_argument("--nps-stride", type=int, default=4,
                   help="Use every Nth slice for the per-patient NPS curves "
                        "(1 = all slices, slower).")
    p.add_argument("--nps-crop", type=int, default=320)
    p.add_argument("--skip-nps", action="store_true",
                   help="Skip the per-patient radial NPS curve figure.")
    return p.parse_args()


def window_img(img_px: np.ndarray, center_hu: float, width_hu: float) -> np.ndarray:
    hu = img_px - 1024.0
    lo = center_hu - width_hu / 2.0
    return np.clip((hu - lo) / width_hu, 0.0, 1.0)


@torch.no_grad()
def predict_px(model, low_hu: torch.Tensor) -> np.ndarray:
    x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
    pred = model(x)
    pred_px = denormalize_to_pixel(pred.squeeze()).clamp(0.0, cfg.EVAL_DATA_RANGE)
    return pred_px.detach().cpu().numpy()


def radial_nps(residual: np.ndarray, crop: int, n_bins: int = 64):
    """Radially averaged power spectrum of a residual image (center crop,
    mean removed, Hann window). Returns (freq_centers, profile)."""
    h, w = residual.shape
    c = min(crop, h, w)
    y0, x0 = (h - c) // 2, (w - c) // 2
    r = residual[y0:y0 + c, x0:x0 + c].astype(np.float64)
    r = r - r.mean()
    win = np.outer(np.hanning(c), np.hanning(c))
    ps = np.abs(np.fft.fftshift(np.fft.fft2(r * win))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(c))
    rr = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    bins = np.linspace(0.0, 0.5 * np.sqrt(2.0), n_bins + 1)
    idx = np.clip(np.digitize(rr.ravel(), bins) - 1, 0, n_bins - 1)
    pw = np.bincount(idx, weights=ps.ravel(), minlength=n_bins)
    cnt = np.maximum(1, np.bincount(idx, minlength=n_bins))
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, pw / cnt


def list_slices(patient_dir: Path):
    low = sort_by_instance_number(glob(str(patient_dir / "Low_Dose" / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full) or not low:
        raise RuntimeError(f"slice mismatch or empty series in {patient_dir}")
    return low, full


def comparison_figure(pid, body, idx, low_px, full_px, preds, labels, rois,
                      out_dir: Path):
    c_hu, w_hu = WINDOWS[body]
    cols = [("LDCT", low_px)] + list(zip(labels, preds)) + [("NDCT", full_px)]
    n_cols = len(cols)
    n_rows = 2 + len(rois)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.1 * n_cols, 3.1 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    diff_rng = 100.0 if body == "Chest" else 50.0

    for j, (name, img) in enumerate(cols):
        ax = axes[0, j]
        ax.imshow(window_img(img, c_hu, w_hu), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        for k, (x, y, s) in enumerate(rois):
            ax.add_patch(Rectangle((x, y), s, s, fill=False, lw=1.4,
                                   edgecolor=ROI_COLORS[k % len(ROI_COLORS)]))
        ax.axis("off")

        ax = axes[1, j]
        if name == "NDCT":
            ax.axis("off")
        else:
            ax.imshow(img - full_px, cmap="coolwarm",
                      vmin=-diff_rng, vmax=diff_rng)
            ax.set_title(f"{name} \u2212 NDCT (\u00b1{diff_rng:.0f} HU)",
                         fontsize=9)
            ax.axis("off")

        for k, (x, y, s) in enumerate(rois):
            ax = axes[2 + k, j]
            crop = img[y:y + s, x:x + s]
            ax.imshow(window_img(crop, c_hu, w_hu), cmap="gray", vmin=0, vmax=1)
            mu = float(crop.mean()) - 1024.0
            sd = float(crop.std())
            ax.set_title(f"ROI{k + 1}: {mu:.1f}\u00b1{sd:.1f} HU", fontsize=9,
                         color=ROI_COLORS[k % len(ROI_COLORS)])
            ax.axis("off")

    win_txt = f"c={c_hu:.0f} / w={w_hu:.0f}"
    fig.suptitle(f"{pid} ({body}) \u2014 slice {idx} \u2014 window {win_txt}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = out_dir / f"compare_{pid}_s{idx}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  -> {path}")


def roi_stats_csv(pid, idx, low_px, full_px, preds, labels, rois, out_dir):
    path = out_dir / f"roi_stats_{pid}_s{idx}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "roi", "x", "y", "size",
                    "mean_hu", "std_hu", "resid_std_vs_ndct_hu"])
        for name, img in [("LDCT", low_px)] + list(zip(labels, preds)) + [("NDCT", full_px)]:
            for k, (x, y, s) in enumerate(rois):
                crop = img[y:y + s, x:x + s]
                resid = crop - full_px[y:y + s, x:x + s]
                w.writerow([name, k + 1, x, y, s,
                            f"{float(crop.mean()) - 1024.0:.2f}",
                            f"{float(crop.std()):.2f}",
                            f"{float(resid.std()):.2f}"])
    print(f"  -> {path}")


def nps_figure(pid, body, patient_dir, models, labels, device, stride, crop,
               out_dir):
    low_paths, full_paths = list_slices(patient_dir)
    ref_sum = None
    model_sums = [None] * len(models)
    n = 0
    for lp, fp in list(zip(low_paths, full_paths))[::max(1, stride)]:
        low_hu = load_dicom_tensor(lp).to(device)
        full_hu = load_dicom_tensor(fp).to(device)
        low_px = (low_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE).cpu().numpy()
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE).cpu().numpy()
        freqs, prof = radial_nps(low_px - full_px, crop)
        ref_sum = prof if ref_sum is None else ref_sum + prof
        for mi, model in enumerate(models):
            pred_px = predict_px(model, low_hu)
            _, p = radial_nps(low_px - pred_px, crop)
            model_sums[mi] = p if model_sums[mi] is None else model_sums[mi] + p
        n += 1
    if n == 0:
        return
    plt.figure(figsize=(6.6, 4.4))
    plt.semilogy(freqs, ref_sum / n, "k--", lw=2,
                 label="Reference noise (LDCT \u2212 NDCT)")
    for mi, lab in enumerate(labels):
        plt.semilogy(freqs, model_sums[mi] / n, lw=1.6,
                     label=f"Removed noise: {lab}")
    plt.xlabel("Radial spatial frequency (cycles/pixel)")
    plt.ylabel("Noise power (a.u.)")
    plt.title(f"Radial NPS of removed vs reference noise \u2014 {pid} ({body}, "
              f"{n} slices)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25, which="both")
    plt.tight_layout()
    path = out_dir / f"nps_{pid}.png"
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  -> {path}")


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Run with HU_RANGE_PRESET=benchmark.")
    if len(args.runs_roots) != len(args.labels):
        raise ValueError("--labels must match --runs-roots (same count)")
    rois = []
    for spec in args.rois:
        x, y, s = (int(v) for v in spec.split(","))
        rois.append((x, y, s))

    setup_reproducibility()
    device = get_device()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for root, lab in zip(args.runs_roots, args.labels):
        ckpt = Path(root) / args.arch / "best_model.pt"
        if not ckpt.exists():
            raise RuntimeError(f"{ckpt} not found")
        print(f"Loading {lab}: {ckpt}")
        models.append(load_checkpoint(str(ckpt), args.arch, device))

    for pid in args.patients:
        patient_dir = Path(args.test_dir) / pid
        if not patient_dir.is_dir():
            print(f"Skipping {pid}: {patient_dir} not found")
            continue
        body = "Chest" if pid.upper().startswith("C") else "Abdomen"
        low_paths, full_paths = list_slices(patient_dir)
        idx = (args.slice_index if args.slice_index is not None
               else int(round(args.slice_frac * (len(low_paths) - 1))))
        idx = max(0, min(idx, len(low_paths) - 1))
        print(f"\n{pid} ({body}): slice {idx + 1}/{len(low_paths)}")

        low_hu = load_dicom_tensor(low_paths[idx]).to(device)
        full_hu = load_dicom_tensor(full_paths[idx]).to(device)
        low_px = (low_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE).cpu().numpy()
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE).cpu().numpy()
        preds = [predict_px(m, low_hu) for m in models]

        comparison_figure(pid, body, idx, low_px, full_px, preds, args.labels,
                          rois, out_dir)
        roi_stats_csv(pid, idx, low_px, full_px, preds, args.labels, rois,
                      out_dir)
        if not args.skip_nps:
            nps_figure(pid, body, patient_dir, models, args.labels, device,
                       args.nps_stride, args.nps_crop, out_dir)

    print(f"\nAll figures -> {out_dir}/")


if __name__ == "__main__":
    main()
