"""Physics-fidelity evaluation: NPS match, per-tissue HU bias, noise magnitude.

WHY THIS SCRIPT EXISTS
----------------------
Standard metrics (PSNR/SSIM/VIF in evaluate_image.py) reward smoothing and are
blind to whether the denoised image is PHYSICALLY faithful. Physics-informed
losses (radial NPS matching, HU-bin bias) are NOT designed to raise PSNR --
they often trade against it by construction, because PSNR rewards exactly the
over-smoothing NPS penalizes. Judged only by PSNR/SSIM/VIF, physics additions
look useless BY CONSTRUCTION. This script measures what they actually target.

METRICS (per patient, averaged over slices)
-------------------------------------------
1. NPS_LogL1      : log-domain L1 distance between the radial noise-power
                    spectrum of the noise the model REMOVED (LDCT - pred) and
                    the true paired noise (LDCT - NDCT). Lower = the removed
                    noise has the correct spectral signature (no texture
                    shift, no anatomy leakage into the residual).
2. NPS_Corr       : Pearson correlation between the two radial NPS curves
                    (shape agreement, scale-invariant). Higher is better.
3. NoiseSTD_Ratio : std(removed) / std(true noise). Ideal = 1.0.
                    < 1 -> under-denoising; > 1 -> over-smoothing (the model
                    removed anatomy in addition to noise).
4. Bias_<tissue>  : mean HU error (pred - NDCT) inside fixed tissue intervals
                    of the NDCT image (AirLung / FatLow / Soft / Dense /
                    Bone). Quantitative HU-calibration check; ideal = 0 HU.

Usage
-----
HU_RANGE_PRESET=benchmark python evaluate_physics.py \\
    --test-dir test --runs-root runs \\
    --output eval_physics --split 100p

Run once per runs-root (e.g. pure vs +physics) and compare tables, or place
both checkpoints under one root as different arch directories.
"""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from benchmark_data import denormalize_to_pixel, standardize_hu
from evaluate_image import ARCH_MAP, get_test_set, load_checkpoint
from physics_losses import BatchRadialNPSLoss
from utils import (
    load_dicom_tensor, setup_reproducibility, get_device,
    sort_by_instance_number,
)


# Fixed physical tissue intervals (HU), same as physics_losses.HUBinBiasLoss.
_TISSUE_BINS = (
    ("AirLung", -1024.0, -500.0),
    ("FatLow",   -500.0, -200.0),
    ("Soft",     -200.0,  200.0),
    ("Dense",     200.0,  600.0),
    ("Bone",      600.0, 1900.0),
)
_MIN_PIXELS = 256
_EPS = 1e-8

# Reuse the training-time spectral machinery (Hann taper, radial binning) so
# the evaluation measures exactly what the NPS loss optimizes.
_NPS = BatchRadialNPSLoss()


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    """Center-crop the last two dims of a [1,1,H,W] tensor."""
    h, w = x.shape[-2:]
    size = min(size, h, w)
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


@torch.no_grad()
def _radial_nps(residual: torch.Tensor) -> torch.Tensor:
    """Radial noise-power spectrum of a [1,1,H,W] residual (Hann-tapered)."""
    return _NPS._radial_average(_NPS._power(residual))


@torch.no_grad()
def nps_metrics(removed: torch.Tensor, true_noise: torch.Tensor):
    """(log-L1 distance, Pearson correlation) between radial NPS curves."""
    p = _radial_nps(removed)
    t = _radial_nps(true_noise)
    log_l1 = float(F.l1_loss(torch.log(p + _EPS), torch.log(t + _EPS)))
    a = p - p.mean()
    b = t - t.mean()
    corr = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
    return log_l1, corr


@torch.no_grad()
def evaluate_patient(pid: str, patient_dir: Path, model, device,
                     nps_crop: int) -> dict:
    low  = sort_by_instance_number(glob(str(patient_dir / "Low_Dose"  / "*.dcm")))
    full = sort_by_instance_number(glob(str(patient_dir / "Full_Dose" / "*.dcm")))
    if len(low) != len(full):
        raise RuntimeError(f"[{pid}] slice mismatch: {len(low)} vs {len(full)}")

    body = "Chest" if pid.upper().startswith("C") else "Abdomen"

    log_l1s, corrs, std_ratios = [], [], []
    bias_sum = {name: 0.0 for name, _, _ in _TISSUE_BINS}
    bias_cnt = {name: 0   for name, _, _ in _TISSUE_BINS}

    for low_path, full_path in tqdm(
        zip(low, full), total=len(low), desc=f"  {pid}", leave=False
    ):
        low_hu  = load_dicom_tensor(low_path).to(device)
        full_hu = load_dicom_tensor(full_path).to(device)

        x      = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
        pred_z = model(x)

        pred_px = denormalize_to_pixel(pred_z.squeeze()).clamp(0.0, cfg.EVAL_DATA_RANGE)
        full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
        low_px  = (low_hu  + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)

        removed = (low_px - pred_px).unsqueeze(0).unsqueeze(0)
        true_n  = (low_px - full_px).unsqueeze(0).unsqueeze(0)
        if nps_crop > 0:
            removed_c = _center_crop(removed, nps_crop)
            true_c    = _center_crop(true_n,  nps_crop)
        else:
            removed_c, true_c = removed, true_n

        ll1, cc = nps_metrics(removed_c, true_c)
        log_l1s.append(ll1)
        corrs.append(cc)
        std_ratios.append(
            float(removed_c.std() / true_c.std().clamp_min(1e-6))
        )

        for name, lo_b, hi_b in _TISSUE_BINS:
            mask = (full_px >= lo_b + 1024.0) & (full_px < hi_b + 1024.0)
            n = int(mask.sum())
            if n < _MIN_PIXELS:
                continue
            bias_sum[name] += float((pred_px[mask] - full_px[mask]).sum())
            bias_cnt[name] += n

    row = {
        "PatientID":      pid,
        "BodyType":       body,
        "NumSlices":      len(low),
        "NPS_LogL1":      float(np.mean(log_l1s)),
        "NPS_Corr":       float(np.mean(corrs)),
        "NoiseSTD_Ratio": float(np.mean(std_ratios)),
    }
    for name, _, _ in _TISSUE_BINS:
        row[f"Bias_{name}"] = (
            bias_sum[name] / bias_cnt[name] if bias_cnt[name] else float("nan")
        )
    return row


def print_comparison(all_dfs: dict, split: str):
    n_label = "20" if split == "20p" else "100"
    metrics = ["NPS_LogL1", "NPS_Corr", "NoiseSTD_Ratio",
               "Bias_AirLung", "Bias_Soft", "Bias_Bone"]
    ideal   = {"NPS_LogL1": "0 (low)", "NPS_Corr": "1 (high)",
               "NoiseSTD_Ratio": "1.00", "Bias_AirLung": "0 HU",
               "Bias_Soft": "0 HU", "Bias_Bone": "0 HU"}
    print("\n" + "=" * 100)
    print(f"  {n_label}-PATIENT PHYSICS-FIDELITY COMPARISON")
    print("  (noise-power spectrum | noise magnitude | per-tissue HU bias)")
    print("=" * 100)
    header = f"  {'Model':<24}" + "".join(f"{m:>15}" for m in metrics)
    ideal_row = f"  {'(ideal)':<24}" + "".join(f"{ideal[m]:>15}" for m in metrics)
    for body in ["Chest", "Abdomen", "Overall"]:
        print(f"\n  [{body.upper()}]")
        print(header)
        print(ideal_row)
        print("  " + "-" * 96)
        for arch, df in all_dfs.items():
            sub = df if body == "Overall" else df[df["BodyType"] == body]
            if sub.empty:
                continue
            row   = sub[metrics].mean()
            label = ARCH_MAP.get(arch, arch)
            print(f"  {label:<24}" + "".join(f"{row[m]:>15.4f}" for m in metrics))
        print("  " + "-" * 96)
    print("\n  Read with evaluate_image.py side by side: the physics claim is")
    print("  'better physical fidelity at equal PSNR/SSIM/VIF'.")
    print("=" * 100)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--test-dir",  default=cfg.TEST_DIR)
    p.add_argument("--output",    default="eval_physics")
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--archs", default="redcnn,resnet",
                   help="Comma-separated arch directories under --runs-root.")
    p.add_argument("--nps-crop", type=int, default=320,
                   help="Center-crop size (px) for NPS estimation; 0 = full "
                        "slice. Cropping keeps the reconstruction-circle "
                        "background from dominating the spectrum.")
    args = p.parse_args()

    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError(
            "Run with HU_RANGE_PRESET=benchmark.\n"
            "Example: HU_RANGE_PRESET=benchmark python evaluate_physics.py ..."
        )

    setup_reproducibility()
    device   = get_device()
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)

    test_ids = get_test_set(args.split)
    test_patients = sorted([
        d for d in Path(args.test_dir).iterdir()
        if d.is_dir()
        and d.name in test_ids
        and (d / "Low_Dose").exists()
        and (d / "Full_Dose").exists()
    ])
    if not test_patients:
        raise RuntimeError(
            f"No test patients found in '{args.test_dir}' "
            f"matching the {args.split} split.\n"
            f"Expected IDs: {sorted(test_ids)}"
        )

    print(f"Split        : {args.split} ({len(test_patients)} patients found)")
    print(f"Test patients: {[d.name for d in test_patients]}")
    print(f"NPS crop     : {args.nps_crop or 'full slice'}")

    all_dfs: dict = {}
    for arch in [a.strip() for a in args.archs.split(",") if a.strip()]:
        ckpt = Path(args.runs_root) / arch / "best_model.pt"
        if not ckpt.exists():
            print(f"  Skipping {arch}: {ckpt} not found")
            continue
        print(f"\nEvaluating {ARCH_MAP.get(arch, arch)} ...")
        try:
            model = load_checkpoint(str(ckpt), arch, device)
        except Exception as e:
            print(f"  ERROR loading {arch}: {e}")
            continue
        rows = [
            evaluate_patient(d.name, d, model, device, args.nps_crop)
            for d in test_patients
        ]
        df = pd.DataFrame(rows)
        df["Model"] = ARCH_MAP.get(arch, arch)
        all_dfs[arch] = df
        df.to_csv(out_path / f"{arch}_physics.csv", index=False)

    if not all_dfs:
        print("No checkpoints found. Train first with train.py.")
        return

    print_comparison(all_dfs, args.split)

    summary_path = out_path / "physics_comparison.csv"
    pd.concat(all_dfs.values(), ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nFull report -> {summary_path}")


if __name__ == "__main__":
    main()
