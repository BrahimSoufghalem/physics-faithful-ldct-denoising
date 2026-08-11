"""Run many SMALL pilot trainings sequentially and summarize image + physics
metrics in one table.

Purpose: screen many configurations cheaply (few patients, few iterations)
before spending a full paper-protocol run on the winner. All pilots share the
same seed, the same deterministic patient subset and the same budget, so
their RANKING is meaningful. The absolute numbers are NOT reportable.

Default sweep (20 configs)
--------------------------
Base ablation:
    C0              pure trunk
    S1              spectral head only (mechanism without incentive)
    S2-n0.05        head + NPS loss 0.05
    S2h-n0.05-h0.1  head + NPS 0.05 + HU-bin 0.1
Loss attribution (losses WITHOUT head -- does the head matter or only the loss?):
    C1-n0.05        --nps-weight 0.05
    C1-n0.01        --nps-weight 0.01
    C1h-n0.005-h0.2 --nps-weight 0.005 --hu-bin-loss 0.2  (full-loss control)
    C2-h0.2         --hu-bin-loss 0.2                     (HU-bin only)
NPS weight sweep (with head):
    S2-n0.1 / S2-n0.03 / S2-n0.01 / S2-n0.005
HU-bin grid at nps=0.05:  h0.05 / h0.2   (h0.1 is in the base list)
HU-bin grid at nps=0.01:  h0.05 / h0.1 / h0.2
HU-bin grid at nps=0.005: h0.1 / h0.2

Usage
-----
    python run_pilot.py --data-dir dataset \\
        --train-patients 8 --val-patients 4 \\
        --iters 8000 --val-every 1000

Custom config list (one per line: "name | extra train.py flags"):
    python run_pilot.py --data-dir dataset --configs my_pilots.txt

Outputs
-------
- Each pilot trains into <output-root>/<name>/<arch>/best_model.pt; existing
  results are skipped (delete the folder or pass --force to retrain).
- Physics metrics (NPS_LogL1, NPS_Corr, NoiseSTD_Ratio, per-tissue HU bias)
  are computed on the SAME validation patient subset used for training
  validation -- NOT on the test set, so config selection does not leak test
  data. Cached per run in physics_val.csv.
- Final combined table (image + physics) printed and saved to
  <output-root>/pilot_summary.csv.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys

# Must be set before importing config/evaluate modules (physics summary).
os.environ.setdefault("HU_RANGE_PRESET", "benchmark")

PAPER_LR = "9.583417460320728e-05"

DEFAULT_CONFIGS = """\
# name | extra train.py flags
# -- Base ablation ------------------------------------------------------
C0              |
S1              | --use-spectral-head
S2-n0.05        | --use-spectral-head --nps-weight 0.05
S2h-n0.05-h0.1  | --use-spectral-head --nps-weight 0.05 --hu-bin-loss 0.1
# -- Loss attribution: losses WITHOUT the head --------------------------
C1-n0.05        | --nps-weight 0.05
C1-n0.01        | --nps-weight 0.01
C1h-n0.005-h0.2 | --nps-weight 0.005 --hu-bin-loss 0.2
C2-h0.2         | --hu-bin-loss 0.2
# -- NPS weight sweep (with head) ---------------------------------------
S2-n0.1         | --use-spectral-head --nps-weight 0.1
S2-n0.03        | --use-spectral-head --nps-weight 0.03
S2-n0.01        | --use-spectral-head --nps-weight 0.01
S2-n0.005       | --use-spectral-head --nps-weight 0.005
# -- HU-bin grid, nps fixed at 0.05 --------------------------------------
S2h-n0.05-h0.05 | --use-spectral-head --nps-weight 0.05 --hu-bin-loss 0.05
S2h-n0.05-h0.2  | --use-spectral-head --nps-weight 0.05 --hu-bin-loss 0.2
# -- HU-bin grid, nps fixed at 0.01 --------------------------------------
S2h-n0.01-h0.05 | --use-spectral-head --nps-weight 0.01 --hu-bin-loss 0.05
S2h-n0.01-h0.1  | --use-spectral-head --nps-weight 0.01 --hu-bin-loss 0.1
S2h-n0.01-h0.2  | --use-spectral-head --nps-weight 0.01 --hu-bin-loss 0.2
# -- HU-bin grid, nps fixed at 0.005 (winner region) ----------------------
S2h-n0.005-h0.1 | --use-spectral-head --nps-weight 0.005 --hu-bin-loss 0.1
S2h-n0.005-h0.2 | --use-spectral-head --nps-weight 0.005 --hu-bin-loss 0.2
"""

# Keep in sync with models/__init__.py ARCH_CHOICES (inline to avoid importing
# torch before the training subprocesses run).
_ARCH_CHOICES = ("redcnn", "resnet", "dugan", "wganvgg", "transct")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--arch", default="redcnn", choices=list(_ARCH_CHOICES),
                   help="Trunk architecture (see models/). NOTE: transct "
                        "requires --patch-size 512.")
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--train-patients", type=int, default=8)
    p.add_argument("--val-patients", type=int, default=4)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=73)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--output-root", default="runs_pilot")
    p.add_argument("--configs", default=None,
                   help="Path to a config list file. Each non-comment line: "
                        "'name | extra flags'. Defaults to the built-in "
                        "20-config sweep.")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if best_model.pt already exists.")
    p.add_argument("--skip-physics", action="store_true",
                   help="Skip physics metrics in the summary (image metrics "
                        "only).")
    p.add_argument("--phys-stride", type=int, default=4,
                   help="Use every Nth validation slice for physics metrics "
                        "(1 = all slices, slower).")
    p.add_argument("--phys-crop", type=int, default=320,
                   help="Center-crop (px) for NPS estimation; 0 = full slice.")
    return p.parse_args()


def load_configs(path):
    text = DEFAULT_CONFIGS if path is None else open(path).read()
    configs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, extra = line.split("|", 1)
        else:
            name, extra = line, ""
        name = name.strip()
        if not name:
            continue
        configs.append((name, extra.strip()))
    if not configs:
        raise RuntimeError("No configurations found")
    return configs


def train_one(args, name, extra, env):
    out_root = os.path.join(args.output_root, name)
    cmd = [
        sys.executable, "train.py",
        "--arch", args.arch,
        "--data-dir", args.data_dir,
        "--split", args.split,
        "--output-root", out_root,
        "--max-iterations", str(args.iters),
        "--iterations-before-val", str(args.val_every),
        "--batch-size", str(args.batch_size),
        "--patch-size", str(args.patch_size),
        "--val-patch-size", "128",
        "--lr", PAPER_LR,
        "--lr-schedule", "constant",
        "--select-by", "bench_ssim",
        "--num-workers", str(args.num_workers),
        "--cache-rate", str(args.cache_rate),
        "--train-patients", str(args.train_patients),
        "--val-patients", str(args.val_patients),
    ] + shlex.split(extra)
    print(f"\n[{name}] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[{name}] FAILED (returncode {result.returncode}) -- "
              f"continuing with the next config")


# ──────────────────────────────────────────────────────────────────
# Physics metrics on the VALIDATION patient subset (no test-set leakage).
# ──────────────────────────────────────────────────────────────────
_PHYS_FIELDS = ("nps_logl1", "nps_corr", "noise_std_ratio",
                "bias_airlung", "bias_fatlow", "bias_soft", "bias_dense",
                "bias_bone")


def _val_patient_list(data_dir, split, max_val):
    import config as cfg
    from benchmark_data import _limit_patients
    if split == "20p":
        from twenty_patient_split import TRAIN_20P, VAL_20P
        cfg.EXPECTED_TRAIN = TRAIN_20P
        cfg.EXPECTED_VAL = VAL_20P
    all_p = sorted(
        p for p in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, p))
    )
    val = [p for p in all_p if p in cfg.EXPECTED_VAL]
    return _limit_patients(val, max_val, "physics-val")


def _compute_physics(ckpt_path, arch, data_dir, split, max_val, stride, crop):
    """Per-patient physics rows for one checkpoint, on validation patients."""
    from glob import glob
    from pathlib import Path

    import numpy as np
    import torch

    import config as cfg
    from benchmark_data import denormalize_to_pixel, standardize_hu
    from evaluate_image import load_checkpoint
    from evaluate_physics import (
        nps_metrics, _center_crop, _TISSUE_BINS, _MIN_PIXELS,
    )
    from utils import get_device, load_dicom_tensor, sort_by_instance_number

    device = get_device()
    model = load_checkpoint(str(ckpt_path), arch, device)
    patients = _val_patient_list(data_dir, split, max_val)

    rows = []
    with torch.no_grad():
        for pid in patients:
            pdir = Path(data_dir) / pid
            low = sort_by_instance_number(
                glob(str(pdir / "Low_Dose" / "*.dcm")))
            full = sort_by_instance_number(
                glob(str(pdir / "Full_Dose" / "*.dcm")))
            if len(low) != len(full) or not low:
                print(f"  [physics] skipping {pid}: slice mismatch")
                continue
            pairs = list(zip(low, full))[::max(1, stride)]
            ll1s, corrs, ratios = [], [], []
            bias_sum = {n: 0.0 for n, _, _ in _TISSUE_BINS}
            bias_cnt = {n: 0 for n, _, _ in _TISSUE_BINS}
            for lp, fp in pairs:
                low_hu = load_dicom_tensor(lp).to(device)
                full_hu = load_dicom_tensor(fp).to(device)
                x = standardize_hu(low_hu).unsqueeze(0).unsqueeze(0)
                pred_px = denormalize_to_pixel(model(x).squeeze()).clamp(
                    0.0, cfg.EVAL_DATA_RANGE)
                full_px = (full_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
                low_px = (low_hu + 1024.0).clamp(0.0, cfg.EVAL_DATA_RANGE)
                removed = (low_px - pred_px).unsqueeze(0).unsqueeze(0)
                true_n = (low_px - full_px).unsqueeze(0).unsqueeze(0)
                if crop > 0:
                    removed = _center_crop(removed, crop)
                    true_n = _center_crop(true_n, crop)
                ll1, cc = nps_metrics(removed, true_n)
                ll1s.append(ll1)
                corrs.append(cc)
                ratios.append(float(
                    removed.std() / true_n.std().clamp_min(1e-6)))
                for nname, lo_b, hi_b in _TISSUE_BINS:
                    mask = ((full_px >= lo_b + 1024.0)
                            & (full_px < hi_b + 1024.0))
                    npx = int(mask.sum())
                    if npx < _MIN_PIXELS:
                        continue
                    bias_sum[nname] += float(
                        (pred_px[mask] - full_px[mask]).sum())
                    bias_cnt[nname] += npx
            row = {
                "patient": pid,
                "nps_logl1": float(np.mean(ll1s)),
                "nps_corr": float(np.mean(corrs)),
                "noise_std_ratio": float(np.mean(ratios)),
            }
            for nname, _, _ in _TISSUE_BINS:
                key = f"bias_{nname.lower()}"
                row[key] = (bias_sum[nname] / bias_cnt[nname]
                            if bias_cnt[nname] else float("nan"))
            rows.append(row)
    return rows


def physics_for_run(args, name):
    """Overall physics means for one pilot run, cached in physics_val.csv."""
    import numpy as np

    run_dir = os.path.join(args.output_root, name)
    ckpt = os.path.join(run_dir, args.arch, "best_model.pt")
    cache = os.path.join(run_dir, "physics_val.csv")
    if not os.path.exists(ckpt):
        return None
    if os.path.exists(cache):
        with open(cache, newline="") as f:
            rows = [{k: (v if k == "patient" else float(v))
                     for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        print(f"[{name}] computing physics metrics on validation subset ...")
        rows = _compute_physics(ckpt, args.arch, args.data_dir, args.split,
                                args.val_patients, args.phys_stride,
                                args.phys_crop)
        if rows:
            with open(cache, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    if not rows:
        return None
    return {k: float(np.nanmean([r[k] for r in rows])) for k in _PHYS_FIELDS}


# ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    configs = load_configs(args.configs)
    env = {**os.environ, "HU_RANGE_PRESET": "benchmark"}
    os.makedirs(args.output_root, exist_ok=True)

    print(f"\nPILOT SWEEP: {len(configs)} configs | "
          f"{args.train_patients} train / {args.val_patients} val patients | "
          f"{args.iters} iterations each")
    print("NOTE: pilot numbers rank configs; they are NOT reportable results.")
    print("Physics metrics are computed on the validation subset "
          "(no test-set leakage).\n")

    for name, extra in configs:
        best = os.path.join(args.output_root, name, args.arch, "best_model.pt")
        if os.path.exists(best) and not args.force:
            print(f"[{name}] already trained -> skipping (use --force to redo)")
            continue
        train_one(args, name, extra, env)

    # ── Combined summary ──────────────────────────────────────────────────────────────
    import torch

    rows = []
    for name, extra in configs:
        best = os.path.join(args.output_root, name, args.arch, "best_model.pt")
        if not os.path.exists(best):
            print(f"[{name}] no best_model.pt -> excluded from summary")
            continue
        payload = torch.load(best, map_location="cpu", weights_only=False)
        val = payload.get("val_detail", {})
        row = {
            "name": name,
            "flags": extra,
            "iter": payload.get("iteration", ""),
            "bench_ssim": val.get("bench_ssim", payload.get("score", 0.0)),
            "psnr": val.get("psnr", 0.0),
            "ssim": val.get("ssim", 0.0),
            "rmse": val.get("rmse", 0.0),
            "chest_ssim": val.get("chest_ssim", 0.0),
            "abdomen_ssim": val.get("abdomen_ssim", 0.0),
        }
        if not args.skip_physics:
            phys = physics_for_run(args, name)
            if phys:
                row.update(phys)
        rows.append(row)
    if not rows:
        print("\nNo pilot results to summarize.")
        return
    rows.sort(key=lambda r: r["bench_ssim"], reverse=True)

    has_phys = any("nps_logl1" in r for r in rows)
    print("\n" + "=" * 132)
    print("PILOT SUMMARY -- image + physics metrics on the validation subset "
          "(ranked by bench_ssim; ranking only, NOT reportable)")
    print("=" * 132)
    hdr = (f"{'config':<16} {'iter':>6} {'bSSIM':>8} {'PSNR':>7} {'SSIM':>7} "
           f"{'RMSE':>7}")
    if has_phys:
        hdr += (f" {'NPS_L1':>7} {'NPSCor':>7} {'STDrat':>7} {'B_Soft':>8} "
                f"{'B_Air':>8} {'B_Bone':>9}")
    print(hdr)
    print("-" * 132)
    ideal = f"{'(ideal)':<16} {'':>6} {'high':>8} {'high':>7} {'high':>7} {'low':>7}"
    if has_phys:
        ideal += (f" {'0':>7} {'1':>7} {'1.00':>7} {'0':>8} {'0':>8} {'0':>9}")
    print(ideal)
    for r in rows:
        line = (f"{r['name']:<16} {r['iter']:>6} {r['bench_ssim']:>8.5f} "
                f"{r['psnr']:>7.3f} {r['ssim']:>7.5f} {r['rmse']:>7.2f}")
        if has_phys and "nps_logl1" in r:
            line += (f" {r['nps_logl1']:>7.4f} {r['nps_corr']:>7.4f} "
                     f"{r['noise_std_ratio']:>7.4f} {r['bias_soft']:>8.2f} "
                     f"{r['bias_airlung']:>8.2f} {r['bias_bone']:>9.2f}")
        elif has_phys:
            line += " " + "(physics n/a)".rjust(50)
        print(line)
    print("-" * 132)

    csv_path = os.path.join(args.output_root, "pilot_summary.csv")
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary -> {csv_path}")
    print("Key comparisons: S1 vs C0 (head alone), C1/C1h/C2 vs S2/S2h at "
          "equal weights (head attribution), S2 sweep (Pareto knee), "
          "S2h grids (HU repair).")
    print("Next: retrain the winning config on the FULL split with the "
          "paper budget before reporting any numbers.")


if __name__ == "__main__":
    main()
