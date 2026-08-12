"""Matched-budget trainer for the physics-faithful LDCT denoising study.

Trains any of the five ldct-benchmark architectures (RED-CNN, ResNet,
DU-GAN generator, WGAN-VGG generator, TransCT -- see models/) with three
independently toggleable physics components:

  --use-spectral-head : DC-preserving learnable radial spectral head
                        (spectral_head.py, architectural contribution)
  --nps-weight W      : radial noise-power-spectrum matching loss
                        (physics_losses.BatchRadialNPSLoss)
  --hu-bin-loss W     : HU-bin bias loss on FIXED physical tissue intervals

Optional refinements (default OFF -> exact previous behavior):

  --freeze-dc-bins N  : pin the first N radial gain knots of the head to
                        G=1 (protects REGIONAL HU means near DC, anti-ring)
  --gain-tv-weight W  : quadratic smoothness penalty on the head's log-gain
                        knots (discourages sharp spectral transitions)
  --hu-bin-weights S  : per-bin weights for the HU-bin loss, comma-separated
                        in the order air,fat,soft,dense,bone (e.g. 1,1,1,3,1)
  --adaptive-head     : v3 -- make the head's radial gain IMAGE-ADAPTIVE: a
                        tiny conditioning encoder (~5k params) predicts
                        bounded per-image offsets to the gain knots from
                        the LDCT input (zero-initialized -> starts exactly
                        at the static head). Requires --use-spectral-head.

Architecture notes
------------------
'dugan' and 'wganvgg' are adversarially trained methods in the benchmark
paper; here only their GENERATORS are trained with the study loss (no
adversarial term), so treat them as trunk ablations (see
train_adversarial.py for the faithful adversarial reproduction). 'transct'
hard-codes 512x512 inputs: use --patch-size 512 --val-patch-size 512 and a
small batch.

Protocol notes
--------------
Hyperparameters follow the official configs/redcnn.yaml of
github.com/eeulig/ldct-benchmark (found by their hpopt), and the checkpoint
criterion (--select-by bench_ssim) replicates
ldctbench/methods/base.py save_checkpoint(to_optimize=\"SSIM\"): best OVERALL
validation SSIM computed WITHOUT clinical windowing.

This study uses a MATCHED training budget (30k iterations for every
configuration) rather than the paper's 92,994 iterations: all comparisons are
internally valid; absolute numbers are not directly comparable to the
published 100-patient benchmark table.

Usage
-----
# C0 baseline (pure MSE, official study protocol):
    HU_RANGE_PRESET=benchmark python train.py --arch redcnn \\
        --data-dir dataset --split 100p \\
        --max-iterations 30000 --iterations-before-val 1000 \\
        --batch-size 73 --patch-size 128 --val-patch-size 128 \\
        --lr 9.583417460320728e-05 --lr-schedule constant \\
        --select-by bench_ssim

# Full physics model (spectral head + NPS + HU-bin losses):
    ... --use-spectral-head --nps-weight 0.005 --hu-bin-loss 0.2

# Refined head (near-DC freeze + smooth gain) and weighted HU bins:
    ... --use-spectral-head --freeze-dc-bins 2 --gain-tv-weight 0.5 \\
        --nps-weight 0.005 --hu-bin-loss 0.2 --hu-bin-weights 1,1,1,3,1

# v3 ADAPTIVE head (anatomy-adaptive per-image gain curve):
    ... --use-spectral-head --adaptive-head --freeze-dc-bins 2 \\
        --nps-weight 0.005 --hu-bin-loss 0.2

# Losses-only control (no head):
    ... --nps-weight 0.005 --hu-bin-loss 0.2

# PILOT MODE: fast config screening on a small deterministic patient subset
# (chest/abdomen balanced). Ranking only -- NOT reportable numbers. See also
# run_pilot.py to sweep several configs sequentially with a summary table:
    ... --train-patients 8 --val-patients 4 --max-iterations 8000

# Multi-seed runs for reporting mean +/- std:
    ... --seed 1 --output-root runs_seed1

# Resume after interruption:
    ... --resume

Install (only when --ssim-weight > 0):
    pip install pytorch-msssim
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as _sk_ssim
from tqdm import tqdm

import config as cfg
from benchmark_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD,
    denormalize_to_pixel, prepare_benchmark_data,
)
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed, compute_rmse_hu,
    compute_vif_hu,
)
from models import ARCH_CHOICES, FIXED_INPUT_SIZE, build_benchmark_model
from physics_losses import BatchRadialNPSLoss
from spectral_head import SpectralResidualModel
from twenty_patient_split import TRAIN_20P, VAL_20P
from utils import setup_reproducibility, get_device, get_state_dict

try:
    from pytorch_msssim import ssim as _pytorch_ssim
    _HAS_MSSSIM = True
except ImportError:
    _HAS_MSSSIM = False


# FIX: constant SSIM data range in benchmark-standardized units.
# The physical evaluation range is EVAL_DATA_RANGE (= 2924 for the benchmark
# preset) in the HU+1024 pixel domain; dividing by the benchmark std expresses
# the same range in standardized units.
_SSIM_DATA_RANGE = float(cfg.EVAL_DATA_RANGE) / float(BENCHMARK_PIXEL_STD)

# Fixed physical tissue boundaries for the HU-bin bias loss (same intervals as
# physics_losses.HUBinBiasLoss).
_HU_BIN_BOUNDARIES_HU = (-1024.0, -500.0, -200.0, 200.0, 600.0, 1900.0)

_SELECT_CHOICES = ("ssim", "psnr", "vif", "chest_ssim", "chest_vif",
                   "bench_ssim")

# Shared radial-NPS loss instance (no parameters; window tensors are created
# on the input device, so a single lazily built module is safe).
_NPS_LOSS = None


def _get_nps_loss():
    global _NPS_LOSS
    if _NPS_LOSS is None:
        _NPS_LOSS = BatchRadialNPSLoss()
    return _NPS_LOSS


# ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Matched-budget trainer with optional physics components"
    )
    p.add_argument("--arch", required=True, choices=list(ARCH_CHOICES))
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations",        type=int,   default=100_000)
    p.add_argument("--iterations-before-val", type=int,   default=2_500)
    p.add_argument("--batch-size",            type=int,   default=64)
    p.add_argument("--patch-size",            type=int,   default=64)
    p.add_argument("--val-patch-size",        type=int,   default=128)
    p.add_argument("--lr",                    type=float, default=1e-4)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"],
                   default="constant",
                   help="LR schedule. 'cosine' decays from --lr to --min-lr "
                        "over --max-iterations (resume-safe: LR is a pure "
                        "function of the iteration counter). The ldct-benchmark "
                        "paper uses 'constant'.")
    p.add_argument("--min-lr", type=float, default=1e-6,
                   help="Final LR for --lr-schedule cosine.")
    p.add_argument("--num-workers",           type=int,   default=2)
    p.add_argument("--cache-rate",            type=float, default=1.0)
    p.add_argument("--output-root",           default="runs")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for weights, sampling and data order. "
                        "Use different --seed and --output-root values for "
                        "multi-seed reporting.")
    p.add_argument("--resume", action="store_true")

    # ── Pilot mode ────────────────────────────────────────────────────────────────────────────
    p.add_argument("--train-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: train on only N patients (deterministic "
                        "chest/abdomen-balanced subset). For fast config "
                        "screening: the RANKING of configs is meaningful, "
                        "the absolute numbers are NOT reportable. Retrain "
                        "the winning config on the full split before "
                        "reporting.")
    p.add_argument("--val-patients", type=int, default=None, metavar="N",
                   help="PILOT MODE: validate on only N patients "
                        "(deterministic chest/abdomen-balanced subset). "
                        "Speeds up the per-cycle validation pass.")

    # ── Checkpoint selection ────────────────────────────────────────────────────────────────
    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="ssim",
                   help="Validation metric used to select best_model.pt. "
                        "'bench_ssim' replicates the ldct-benchmark paper "
                        "saving criterion EXACTLY: overall validation SSIM "
                        "computed WITHOUT clinical windowing "
                        "(skimage structural_similarity, data_range=2924, on "
                        "denormalized images clipped to [0, 2924]), as in "
                        "ldctbench/methods/base.py "
                        "save_checkpoint(to_optimize='SSIM'). "
                        "'ssim'/'psnr' use clinically WINDOWED metrics and "
                        "are therefore NOT the paper criterion. "
                        "'chest_*' options fall back to the overall metric "
                        "when the validation set has no chest slices.")
    p.add_argument("--val-vif", action="store_true",
                   help="Compute VIF during validation even when --select-by "
                        "does not require it. VIF on validation crops is "
                        "indicative only; full-resolution evaluate_image.py "
                        "remains the ground truth.")

    # ── Physics components ──────────────────────────────────────────────────────────────────────
    p.add_argument("--use-spectral-head", action="store_true",
                   help="Physics-informed spectral residual head "
                        "(spectral_head.py): a learnable RADIAL gain G(|f|) "
                        "applied in the Fourier domain to the noise the base "
                        "model removes. The radial-only constraint mirrors "
                        "FBP noise physics (isotropic NPS); G is initialized "
                        "to 1 so training starts exactly at the base model. "
                        "ARCHITECTURAL addition: works with any --arch.")
    p.add_argument("--spectral-bins", type=int, default=32,
                   help="Number of radial gain knots for --use-spectral-head.")
    p.add_argument("--freeze-dc-bins", type=int, default=0, metavar="N",
                   help="Pin the first N radial gain knots of the spectral "
                        "head to G=1 (identity). Exact DC is always "
                        "preserved, but the lowest non-zero frequencies "
                        "carry REGIONAL HU means (whole-organ scale); "
                        "freezing them stops the head from trading regional "
                        "HU calibration for spectral fit and removes sharp "
                        "near-DC gain jumps (ring suspect). With 32 bins "
                        "each knot covers ~0.044 Nyquist units; try 2-3. "
                        "Requires --use-spectral-head. Default 0 = off.")
    p.add_argument("--gain-tv-weight", type=float, default=0.0, metavar="W",
                   help="Quadratic smoothness penalty on consecutive "
                        "log-gain knots of the spectral head (mean squared "
                        "difference of the effective curve). Discourages "
                        "sharp spectral transitions that can cause "
                        "concentric ring artifacts. Requires "
                        "--use-spectral-head. CAUTION: 0.5 was measured to "
                        "flatten the whole curve to ~identity (max |G-1| = "
                        "0.0175); useful range is roughly 0.02-0.1. "
                        "Default 0 = off.")
    p.add_argument("--adaptive-head", action="store_true",
                   help="v3: make the spectral head's radial gain "
                        "IMAGE-ADAPTIVE. A tiny conditioning encoder (~5k "
                        "params: 3 strided convs + GAP + one linear layer) "
                        "predicts bounded per-image offsets to the log-gain "
                        "knots from the LDCT input. The projection is "
                        "ZERO-INITIALIZED, so training starts EXACTLY at "
                        "the static head; frozen knots (--freeze-dc-bins) "
                        "stay frozen per image. Motivation: one shared "
                        "curve must compromise between anatomies (measured "
                        "chest NPS_LogL1 ~0.12 vs abdomen ~0.43). Requires "
                        "--use-spectral-head. NOTE: adds parameters to the "
                        "checkpoint; eval scripts rebuild the model from "
                        "checkpoint meta automatically.")
    p.add_argument("--adaptive-max-delta", type=float, default=0.25,
                   metavar="D",
                   help="Bound on the per-image |log-gain offset| for "
                        "--adaptive-head (tanh-scaled). 0.25 lets the "
                        "adaptive gain scale the shared curve by "
                        "exp(+/-0.25) ~ x0.78-x1.28 per band.")
    p.add_argument("--hu-bin-loss",    type=float, default=0.0, metavar="W",
                   help="HU-bin bias penalty weight on fixed physical tissue "
                        "intervals (-1024/-500/-200/200/600/1900 HU). "
                        "Architecture-independent (works with any --arch).")
    p.add_argument("--hu-bin-weights", type=str, default=None,
                   metavar="W1,W2,W3,W4,W5",
                   help="Comma-separated per-bin weights for --hu-bin-loss in "
                        "the order air/lung, fat/low, soft, dense, bone "
                        "(e.g. 1,1,1,3,1 to focus the stubborn dense-tissue "
                        "bin). A weight of 0 disables a bin. Default: "
                        "uniform weights (exact previous behavior).")
    p.add_argument("--nps-weight", type=float, default=0.0, metavar="W",
                   help="Radial noise-power-spectrum matching loss weight "
                        "(physics_losses.BatchRadialNPSLoss). Matches the "
                        "spectrum of the noise the model removes to the "
                        "paired LDCT-NDCT residual, preserving noise "
                        "texture. Additive on top of the base loss. "
                        "Pairs naturally with --use-spectral-head: the head "
                        "provides the mechanism, this loss provides the "
                        "training signal.")

    # ── Generic loss flags ──────────────────────────────────────────────────────────────────────
    p.add_argument("--ssim-weight", type=float, default=0.0, metavar="W",
                   help="SSIM loss weight (requires pytorch-msssim).")
    p.add_argument("--l1-weight",   type=float, default=0.0, metavar="W",
                   help="L1 loss weight. Remainder goes to MSE.")
    p.add_argument("--grad-weight", type=float, default=0.0, metavar="W",
                   help="Gradient edge loss weight (finite diff L1). Additive.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
def image_gradient_loss(pred, target):
    dx_pred   = pred  [:, :, :, 1:] - pred  [:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred   = pred  [:, :, 1:, :] - pred  [:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


def compute_loss(pred, target, ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                 hu_bin_loss_weight=0.0, hu_bin_weights=None,
                 input_img=None, nps_weight=0.0):
    mse_w = max(0.0, 1.0 - float(ssim_weight) - float(l1_weight))
    loss  = pred.new_zeros(())
    if mse_w > 0.0:
        loss = loss + mse_w * F.mse_loss(pred, target)
    if l1_weight > 0.0:
        loss = loss + float(l1_weight) * F.l1_loss(pred, target)
    if ssim_weight > 0.0:
        if not _HAS_MSSSIM:
            raise RuntimeError("pip install pytorch-msssim")
        # Constant benchmark-derived data range so the SSIM stability
        # constants are not batch-dependent.
        loss = loss + float(ssim_weight) * (
            1.0 - _pytorch_ssim(pred, target, data_range=_SSIM_DATA_RANGE,
                                size_average=True, nonnegative_ssim=True)
        )
    if grad_weight > 0.0:
        loss = loss + float(grad_weight) * image_gradient_loss(pred, target)
    if hu_bin_loss_weight > 0.0:
        loss = loss + hu_bin_loss_weight * hu_bin_bias_loss(
            pred, target, bin_weights=hu_bin_weights)
    if nps_weight > 0.0:
        if input_img is None:
            raise ValueError("--nps-weight requires the network input image")
        # Match the spectrum of what the model removed (input - pred) to the
        # paired LDCT-NDCT residual (input - target): texture preservation.
        loss = loss + float(nps_weight) * _get_nps_loss()(
            input_img - pred, input_img - target
        )
    return loss


def hu_bin_bias_loss(pred, target, bin_weights=None):
    """Mean-bias penalty inside FIXED physical tissue intervals.

    The boundaries are the fixed physical intervals also used by
    physics_losses.HUBinBiasLoss, converted to the benchmark-standardized
    domain. ``bin_weights`` (length 5, order air/lung, fat/low, soft, dense,
    bone) weights each bin's squared bias and the result is normalized by
    the total weight of the PRESENT bins; ``None`` means uniform weights,
    which reproduces the original unweighted behavior exactly.
    """
    edges = [
        (b + 1024.0 - BENCHMARK_PIXEL_MEAN) / BENCHMARK_PIXEL_STD
        for b in _HU_BIN_BOUNDARIES_HU
    ]
    if bin_weights is None:
        bin_weights = [1.0] * (len(edges) - 1)
    loss = pred.new_zeros(())
    wsum = 0.0
    for w, (lo, hi) in zip(bin_weights, zip(edges[:-1], edges[1:])):
        if w <= 0.0:
            continue
        mask = (target >= lo) & (target < hi)
        if int(mask.sum()) < 10:
            continue
        bias = (pred[mask] - target[mask]).mean()
        loss = loss + float(w) * bias * bias
        wsum += float(w)
    return loss / max(1.0, wsum)


def apply_split(split):
    if split == "20p":
        cfg.EXPECTED_TRAIN = TRAIN_20P
        cfg.EXPECTED_VAL   = VAL_20P
        return len(TRAIN_20P), len(VAL_20P)
    return len(cfg.EXPECTED_TRAIN), len(cfg.EXPECTED_VAL)


def build_model(arch, device, args):
    model = build_benchmark_model(arch, device)
    if args.use_spectral_head:
        freeze    = int(getattr(args, "freeze_dc_bins", 0))
        adaptive  = bool(getattr(args, "adaptive_head", False))
        max_delta = float(getattr(args, "adaptive_max_delta", 0.25))
        model = SpectralResidualModel(model, n_bins=args.spectral_bins,
                                      freeze_dc_bins=freeze,
                                      adaptive=adaptive,
                                      max_log_gain_delta=max_delta).to(device)
        print(f"  Spectral head : ON ({args.spectral_bins} radial gain knots, "
              f"init G=1 -> starts exactly at the base model"
              + (f"; first {freeze} knots FROZEN at G=1" if freeze > 0 else "")
              + (f"; ADAPTIVE per-image gain (max |dlog| {max_delta})"
                 if adaptive else "")
              + ")")
    return model


@torch.no_grad()
def validate(model, loader, device, with_vif=False):
    """Validate with overall AND per-region (Chest/Abdomen) metrics.

    ``bench_ssim`` is the ldct-benchmark paper metric: overall SSIM computed
    WITHOUT clinical windowing, exactly as in ldctbench/utils/metrics.py
    (skimage structural_similarity(target, pred, data_range=2924) on
    denormalized images clipped to [0, 2924]).
    """
    model.eval()
    sums    = dict(mse=0.0, psnr=0.0, ssim=0.0, rmse=0.0,
                   baseline_psnr=0.0, vif=0.0, bench_ssim=0.0)
    region  = {
        "Chest":   dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
        "Abdomen": dict(psnr=0.0, ssim=0.0, vif=0.0, n=0),
    }
    batches = samples = 0
    for batch in tqdm(loader, desc="  Val", leave=False, dynamic_ncols=True):
        x    = batch["image"].to(device, non_blocking=True)
        y    = batch["label"].to(device, non_blocking=True)
        pred = model(x)
        sums["mse"] += float(F.mse_loss(pred, y))
        batches += 1
        pred_px = denormalize_to_pixel(pred).clamp(0.0, cfg.EVAL_DATA_RANGE)
        y_px    = denormalize_to_pixel(y).clamp(0.0, cfg.EVAL_DATA_RANGE)
        x_px    = denormalize_to_pixel(x).clamp(0.0, cfg.EVAL_DATA_RANGE)
        body = batch.get("body_type", ["Abdomen"] * pred.shape[0])
        for i in range(pred.shape[0]):
            bt = "Chest" if str(body[i]).lower().startswith("c") else "Abdomen"
            ps = compute_psnr_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            ss = compute_ssim_windowed(pred_px[i].squeeze(), y_px[i].squeeze(), bt)
            vf = compute_vif_hu(pred_px[i].squeeze(), y_px[i].squeeze()) if with_vif else 0.0
            # ldct-benchmark paper SSIM: unwindowed, data_range = 2924, on
            # denormalized clipped images (see ldctbench/utils/metrics.py).
            t_np = y_px[i].squeeze().detach().cpu().numpy()
            p_np = pred_px[i].squeeze().detach().cpu().numpy()
            bss  = float(_sk_ssim(t_np, p_np,
                                  data_range=float(cfg.EVAL_DATA_RANGE)))
            sums["psnr"]          += ps
            sums["ssim"]          += ss
            sums["vif"]           += vf
            sums["bench_ssim"]    += bss
            sums["baseline_psnr"] += compute_psnr_windowed(x_px[i].squeeze(), y_px[i].squeeze(), bt)
            sums["rmse"]          += compute_rmse_hu(pred_px[i].squeeze(), y_px[i].squeeze())
            r = region[bt]
            r["psnr"] += ps
            r["ssim"] += ss
            r["vif"]  += vf
            r["n"]    += 1
            samples += 1
    n_b, n_s = max(1, batches), max(1, samples)
    out = {
        "mse":        sums["mse"]  / n_b,
        "psnr":       sums["psnr"] / n_s,
        "dpsnr":      (sums["psnr"] - sums["baseline_psnr"]) / n_s,
        "ssim":       sums["ssim"] / n_s,
        "rmse":       sums["rmse"] / n_s,
        "vif":        sums["vif"]  / n_s,
        "bench_ssim": sums["bench_ssim"] / n_s,
    }
    for name, r in region.items():
        key = name.lower()
        n   = max(1, r["n"])
        out[f"{key}_psnr"] = r["psnr"] / n
        out[f"{key}_ssim"] = r["ssim"] / n
        out[f"{key}_vif"]  = r["vif"]  / n
        out[f"{key}_n"]    = r["n"]
    return out


def selection_score(val, select_by):
    """Score used to pick best_model.pt. chest_* falls back to overall when
    the validation set contains no chest slices. 'bench_ssim' is the exact
    ldct-benchmark paper criterion (overall unwindowed SSIM)."""
    if select_by == "ssim":
        return val["ssim"]
    if select_by == "psnr":
        return val["psnr"]
    if select_by == "vif":
        return val["vif"]
    if select_by == "bench_ssim":
        return val["bench_ssim"]
    if select_by == "chest_ssim":
        return val["chest_ssim"] if val["chest_n"] > 0 else val["ssim"]
    if select_by == "chest_vif":
        return val["chest_vif"] if val["chest_n"] > 0 else val["vif"]
    raise ValueError(f"Unknown --select-by: {select_by}")


def lr_at(iteration, base_lr, min_lr, max_iter, schedule):
    """LR as a pure function of the iteration counter (resume-safe)."""
    if schedule != "cosine":
        return base_lr
    t = min(1.0, max(0.0, iteration / max(1, max_iter)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def train_cycle(model, loader, optimizer, device, iteration, max_iter,
                ssim_weight=0.0, l1_weight=0.0, grad_weight=0.0,
                hu_bin_loss_weight=0.0, hu_bin_weights=None,
                nps_weight=0.0, gain_tv_weight=0.0,
                base_lr=1e-4, min_lr=1e-6, lr_schedule="constant"):
    model.train()
    total = count = 0.0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if iteration >= max_iter:
            break
        lr_now = lr_at(iteration, base_lr, min_lr, max_iter, lr_schedule)
        for g in optimizer.param_groups:
            g["lr"] = lr_now
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = compute_loss(pred, y, ssim_weight=ssim_weight, l1_weight=l1_weight,
                            grad_weight=grad_weight,
                            hu_bin_loss_weight=hu_bin_loss_weight,
                            hu_bin_weights=hu_bin_weights,
                            input_img=x, nps_weight=nps_weight)
        if gain_tv_weight > 0.0:
            # Quadratic smoothness on the spectral head's log-gain knots.
            # main() only allows this weight with --use-spectral-head, so
            # ``model`` is a SpectralResidualModel here.
            loss = loss + float(gain_tv_weight) * model.head.smoothness_penalty()
        loss.backward()
        optimizer.step()
        iteration += 1
        total += float(loss.detach())
        count += 1
        bar.set_postfix(iter=iteration, loss=f"{loss.item():.6f}",
                        lr=f"{lr_now:.2e}")
    return iteration, total / max(1, count)


# ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    if args.ssim_weight + args.l1_weight > 1.0:
        raise ValueError("--ssim-weight + --l1-weight must not exceed 1.0")
    if args.ssim_weight > 0 and not _HAS_MSSSIM:
        raise RuntimeError("pip install pytorch-msssim")
    if args.nps_weight < 0.0:
        raise ValueError("--nps-weight must be >= 0")
    if args.spectral_bins < 2:
        raise ValueError("--spectral-bins must be >= 2")
    if args.freeze_dc_bins < 0:
        raise ValueError("--freeze-dc-bins must be >= 0")
    if args.freeze_dc_bins > 0 and not args.use_spectral_head:
        raise ValueError("--freeze-dc-bins requires --use-spectral-head")
    if args.use_spectral_head and args.freeze_dc_bins >= args.spectral_bins:
        raise ValueError("--freeze-dc-bins must be < --spectral-bins")
    if args.gain_tv_weight < 0.0:
        raise ValueError("--gain-tv-weight must be >= 0")
    if args.gain_tv_weight > 0.0 and not args.use_spectral_head:
        raise ValueError("--gain-tv-weight requires --use-spectral-head")
    if args.adaptive_head and not args.use_spectral_head:
        raise ValueError("--adaptive-head requires --use-spectral-head")
    if args.adaptive_max_delta <= 0.0:
        raise ValueError("--adaptive-max-delta must be > 0")
    hu_bin_weights = None
    if args.hu_bin_weights is not None:
        if args.hu_bin_loss <= 0.0:
            raise ValueError("--hu-bin-weights requires --hu-bin-loss > 0")
        try:
            hu_bin_weights = [float(v) for v in args.hu_bin_weights.split(",")]
        except ValueError:
            raise ValueError(
                "--hu-bin-weights must be comma-separated numbers")
        n_bins_expected = len(_HU_BIN_BOUNDARIES_HU) - 1
        if len(hu_bin_weights) != n_bins_expected:
            raise ValueError(
                f"--hu-bin-weights needs exactly {n_bins_expected} values "
                "(air/lung, fat/low, soft, dense, bone)")
        if any(w < 0.0 for w in hu_bin_weights):
            raise ValueError("--hu-bin-weights must be >= 0")
        if all(w == 0.0 for w in hu_bin_weights):
            raise ValueError("--hu-bin-weights must not be all zero")
    if args.train_patients is not None and args.train_patients < 1:
        raise ValueError("--train-patients must be >= 1")
    if args.val_patients is not None and args.val_patients < 1:
        raise ValueError("--val-patients must be >= 1")
    fixed = FIXED_INPUT_SIZE.get(args.arch)
    if fixed is not None and (args.patch_size != fixed
                              or args.val_patch_size != fixed):
        raise ValueError(
            f"--arch {args.arch} hard-codes {fixed}x{fixed} inputs "
            f"(see models/{args.arch}.py). Use --patch-size {fixed} "
            f"--val-patch-size {fixed} with a small --batch-size."
        )

    n_train, n_val = apply_split(args.split)
    if args.train_patients is not None:
        n_train = min(n_train, args.train_patients)
    if args.val_patients is not None:
        n_val = min(n_val, args.val_patients)
    pilot = args.train_patients is not None or args.val_patients is not None
    # Seed weights AND the data pipeline (benchmark_data reads cfg.SEED at
    # call time for the sampler generator and MONAI determinism).
    cfg.SEED = int(args.seed)
    setup_reproducibility(args.seed)
    device    = get_device()
    out_dir   = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")

    with_vif = bool(args.val_vif or args.select_by in ("vif", "chest_vif"))

    mse_w  = max(0.0, 1.0 - args.ssim_weight - args.l1_weight)
    if mse_w == 0.0 and (args.ssim_weight > 0.0 or args.l1_weight > 0.0):
        print(
            "\n" + "!" * 68 + "\n"
            "  WARNING: ssim_weight + l1_weight = 1.0 -> MSE weight is 0.\n"
            "  The benchmark protocol trains with pure MSE. Dropping MSE\n"
            "  entirely is an untested protocol deviation; keep MSE weight\n"
            "  >= 0.3 unless this is a deliberate, pre-registered ablation.\n"
            + "!" * 68 + "\n"
        )
    loss_parts = []
    if mse_w > 0.0:             loss_parts.append(f"{mse_w:.2f}*MSE")
    if args.l1_weight > 0.0:    loss_parts.append(f"{args.l1_weight:.2f}*L1")
    if args.ssim_weight > 0.0:  loss_parts.append(f"{args.ssim_weight:.2f}*SSIM")
    if args.grad_weight > 0.0:  loss_parts.append(f"{args.grad_weight:.2f}*Grad")
    loss_desc = " + ".join(loss_parts) if loss_parts else "1.00*MSE"
    if args.hu_bin_loss > 0.0:
        loss_desc += f" + {args.hu_bin_loss}*HU-bin"
        if hu_bin_weights is not None:
            loss_desc += f"(w={','.join(str(w) for w in hu_bin_weights)})"
    if args.nps_weight > 0.0:
        loss_desc += f" + {args.nps_weight}*NPS"
    if args.gain_tv_weight > 0.0:
        loss_desc += f" + {args.gain_tv_weight}*GainSmooth"

    active = []
    if args.use_spectral_head:
        head_desc = f"Spectral-Head(bins={args.spectral_bins}"
        if args.freeze_dc_bins > 0:
            head_desc += f", freezeDC={args.freeze_dc_bins}"
        if args.gain_tv_weight > 0.0:
            head_desc += f", tv={args.gain_tv_weight}"
        if args.adaptive_head:
            head_desc += f", adaptive(d={args.adaptive_max_delta})"
        active.append(head_desc + ")")
    if args.hu_bin_loss > 0.0:
        hb_desc = f"HU-bin(w={args.hu_bin_loss}"
        if hu_bin_weights is not None:
            hb_desc += f", bins={','.join(str(w) for w in hu_bin_weights)}"
        active.append(hb_desc + ")")
    if args.nps_weight > 0.0:  active.append(f"NPS(w={args.nps_weight})")
    active_str = ", ".join(active) if active else "none (baseline)"

    print(f"\n{'='*68}")
    print(f"  arch={args.arch.upper()} | split={args.split} | seed={args.seed}")
    print(f"  physics components: {active_str}")
    print(f"  Train patients : {n_train}  |  Val patients: {n_val}")
    print(f"  Data dir       : {args.data_dir}")
    print(f"  Output         : {out_dir}")
    print(f"  Loss           : {loss_desc}")
    print(f"  LR schedule    : {args.lr_schedule} (lr={args.lr:.2e}"
          + (f" -> {args.min_lr:.2e}" if args.lr_schedule == "cosine" else "")
          + ")")
    print(f"  Select best by : {args.select_by}"
          + (" [ldct-benchmark paper criterion]" if args.select_by == "bench_ssim" else "")
          + (" (+VIF in val)" if with_vif else ""))
    if pilot:
        print("  *** PILOT MODE : reduced patient subset. Use for config "
              "screening/ranking ONLY -- retrain the winner on the full "
              "split before reporting numbers. ***")
    print(f"{'='*68}\n")

    model     = build_model(args.arch, device, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    iteration  = 0
    best_score = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        iteration  = int(ckpt.get("iteration", 0))
        old_select = ckpt.get("select_by", "ssim")
        if old_select == args.select_by:
            best_score = float(ckpt.get("score", ckpt.get("ssim", -float("inf"))))
        else:
            print(f"  NOTE: checkpoint used --select-by {old_select}; "
                  f"resetting best score for {args.select_by}.")
        print(f"  Resumed at iter {iteration} | best {args.select_by} {best_score:.5f}")
    elif args.resume:
        print(f"  --resume: no checkpoint at {ckpt_path}, starting fresh.")

    if iteration >= args.max_iterations:
        print(f"  Training already complete ({iteration}/{args.max_iterations}).")
        return

    train_loader, val_loader = prepare_benchmark_data(
        in_dir=args.data_dir,
        train_patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        train_batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        iterations_before_val=args.iterations_before_val,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        max_train_patients=args.train_patients,
        max_val_patients=args.val_patients,
    )

    print(f"Loss      : {loss_desc}")
    print(f"Optimizer : Adam(lr={args.lr:.2e}, schedule={args.lr_schedule})")

    start = time.time()
    cycle = iteration // args.iterations_before_val

    while iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        iteration, train_loss = train_cycle(
            model, train_loader, optimizer, device,
            iteration, args.max_iterations,
            ssim_weight=args.ssim_weight,
            l1_weight=args.l1_weight,
            grad_weight=args.grad_weight,
            hu_bin_loss_weight=args.hu_bin_loss,
            hu_bin_weights=hu_bin_weights,
            nps_weight=args.nps_weight,
            gain_tv_weight=args.gain_tv_weight,
            base_lr=args.lr,
            min_lr=args.min_lr,
            lr_schedule=args.lr_schedule,
        )
        val   = validate(model, val_loader, device, with_vif=with_vif)
        score = selection_score(val, args.select_by)

        meta = {
            "architecture":    args.arch,
            "split":           args.split,
            "seed":            int(args.seed),
            "use_spectral_head": args.use_spectral_head,
            "spectral_bins":   args.spectral_bins,
            "freeze_dc_bins":  args.freeze_dc_bins,
            "gain_tv_weight":  args.gain_tv_weight,
            "adaptive_head":   args.adaptive_head,
            "adaptive_max_delta": args.adaptive_max_delta,
            "hu_bin_loss":     args.hu_bin_loss,
            "hu_bin_weights":  hu_bin_weights,
            "nps_weight":      args.nps_weight,
            "ssim_weight":     args.ssim_weight,
            "l1_weight":       args.l1_weight,
            "mse_weight":      mse_w,
            "grad_weight":     args.grad_weight,
            "ssim_data_range": _SSIM_DATA_RANGE,
            "lr_schedule":     args.lr_schedule,
            "min_lr":          args.min_lr,
            "select_by":       args.select_by,
            "normalization":   "benchmark_meanstd",
            "pixel_mean":      BENCHMARK_PIXEL_MEAN,
            "pixel_std":       BENCHMARK_PIXEL_STD,
            "pixel_domain":    "HU+1024",
            "hu_preset":       cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss":            loss_desc,
            "input_mode":      "2d",
            "n_train_patients": n_train,
            "n_val_patients":   n_val,
            "max_train_patients": args.train_patients,
            "max_val_patients":   args.val_patients,
            "pilot_mode":      pilot,
        }
        payload = {
            "model_state_dict": get_state_dict(model),
            "meta":      meta,
            "iteration": iteration,
            "ssim":      val["ssim"],
            "psnr":      val["psnr"],
            "val_mse":   val["mse"],
            "score":     score,
            "select_by": args.select_by,
            "val_detail": {k: v for k, v in val.items()},
        }
        torch.save(payload, os.path.join(out_dir, "last_model.pt"))
        if score > best_score:
            best_score = score
            torch.save(payload, os.path.join(out_dir, "best_model.pt"))
        torch.save({**payload, "optimizer_state": optimizer.state_dict()}, ckpt_path)

        elapsed = time.time() - t0
        region_str = (
            f"C-PSNR {val['chest_psnr']:.2f} C-SSIM {val['chest_ssim']:.4f} | "
            f"A-PSNR {val['abdomen_psnr']:.2f} A-SSIM {val['abdomen_ssim']:.4f}"
            if val["chest_n"] > 0 and val["abdomen_n"] > 0 else ""
        )
        vif_str = (
            f" | VIF {val['vif']:.4f}"
            + (f" (C {val['chest_vif']:.4f})" if val["chest_n"] > 0 else "")
            if with_vif else ""
        )
        print(
            f"Cycle {cycle:02d} | Iter {iteration:06d}/{args.max_iterations} | "
            f"Loss {train_loss:.6f} | Val MSE {val['mse']:.6f} | "
            f"PSNR {val['psnr']:.3f} | dPSNR {val['dpsnr']:+.3f} | "
            f"SSIM {val['ssim']:.5f} | bSSIM {val['bench_ssim']:.5f} | "
            f"RMSE {val['rmse']:.2f}"
            f"{vif_str} | {args.select_by} {score:.5f}"
            f"{(' | ' + region_str) if region_str else ''} | "
            f"{elapsed:.1f}s"
        )

    total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()}] in {total} | "
          f"best {args.select_by}={best_score:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")

    if pilot:
        print("\nPILOT MODE reminder: these numbers are for config "
              "ranking only. Retrain the winning config on the full "
              "split with the paper budget before reporting.")


if __name__ == "__main__":
    main()
