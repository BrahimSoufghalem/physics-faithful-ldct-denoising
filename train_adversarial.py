"""Faithful adversarial trainers for WGAN-VGG and DU-GAN.

train.py trains every architecture's GENERATOR with the study loss (trunk
ablation mode). This script instead REPRODUCES the adversarial training of
the two GAN methods of the ldct-benchmark reference implementation
(github.com/eeulig/ldct-benchmark, commit 09b1011bc2fb77ef4fc734cec1e961a20c754910),
wired into this repository's data pipeline, validation and checkpoint format
so evaluate_image.py / evaluate_physics.py / make_figures.py work unchanged.

WGAN-VGG (Yang et al. 2018) -- replicates ldctbench/methods/wganvgg/Trainer.py:
  - critic: 6-conv + 2-FC network on patches (models/wganvgg.py)
  - D loss: WGAN critic loss + gradient penalty (lam = 10)
  - G loss: lam_perc * VGG19 perceptual loss (L1 on layer-34 features,
    1->3 channel repeat) - D(G(x)). NO pixelwise MSE term (faithful).
  - Official hpopt hyperparameters (configs/wganvgg.yaml) as defaults:
    lr 7.0969e-05, Adam betas (0.32606, 0.999), batch 77, patch 77,
    n_d_train 1, lam_perc 0.68933, budget 67306 iterations.

DU-GAN (Huang et al. 2022) -- replicates ldctbench/methods/dugan/Trainer.py:
  - generator: RED-CNN (models/dugan.py, exactly as in the benchmark)
  - two spectral-norm U-Net discriminators (image + Sobel gradient domain),
    LSGAN objectives on encoder and decoder outputs, the LDCT input also
    penalized as fake, CutMix consistency regularization with linear warmup
  - G loss: lam_adv * (img + grad adversarial) + lam_px_im * MSE
    + lam_px_grad * L1(sobel)
  - Official hpopt hyperparameters (configs/dugan.yaml) as defaults:
    lr 1.24362e-05, batch 92, patch 128, n_d_train 2, lam_adv 0.080335,
    lam_px_grad 27.8145, lam_cutmix 2.64787, cutmix_prob 0.5,
    cutmix warmup 7615 iterations, budget 33875 iterations.
  - NOTE: the reference Trainer builds plain optim.Adam(params, lr) and
    IGNORES the adam_b1 value in its own yaml; we replicate the CODE, not
    the yaml.

All method hyperparameters default to the official values; override any of
them for matched-budget study runs (e.g. --max-iterations 30000). The
physics components (--use-spectral-head / --nps-weight / --hu-bin-loss) are
OPTIONAL additions to the generator objective and default to OFF, i.e. the
default run is the faithful reproduction.

Requires torchvision (pretrained VGG19) for --arch wganvgg.

Usage
-----
    HU_RANGE_PRESET=benchmark python train_adversarial.py --arch wganvgg --data-dir dataset --split 100p
    HU_RANGE_PRESET=benchmark python train_adversarial.py --arch dugan --data-dir dataset --split 100p

Matched-budget study variant:
    ... --max-iterations 30000 --iterations-before-val 1000 --select-by bench_ssim
"""

import argparse
import copy
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from adversarial_utils import (
    PerceptualLoss, SobelOperator, cutmix, ls_gan, mask_src_tgt,
    turn_on_spectral_norm,
)
from benchmark_data import (
    BENCHMARK_PIXEL_MEAN, BENCHMARK_PIXEL_STD, prepare_benchmark_data,
)
from models.dugan import DUGANGenerator, UNet
from models.wganvgg import Discriminator as WGANVGGCritic
from models.wganvgg import WGANVGGGenerator
from spectral_head import SpectralResidualModel
from train import (
    _SELECT_CHOICES, _get_nps_loss, apply_split, hu_bin_bias_loss,
    selection_score, validate,
)
from utils import get_device, get_state_dict, setup_reproducibility

ADV_ARCHS = ("wganvgg", "dugan")

# Official hpopt hyperparameters (ldct-benchmark configs at commit 09b1011).
OFFICIAL = {
    "wganvgg": dict(
        lr=7.096903581620458e-05,
        adam_b1=0.3260552006030955,
        adam_b2=0.999,
        batch_size=77,
        patch_size=77,
        n_d_train=1,
        max_iterations=67306,
        lam_perc=0.6893269329076448,
    ),
    "dugan": dict(
        lr=1.2436216786633454e-05,
        batch_size=92,
        patch_size=128,
        n_d_train=2,
        max_iterations=33875,
        lam_adv=0.080335201069619,
        lam_px_im=1.0,
        lam_px_grad=27.81452397771084,
        lam_cutmix=2.647865385857211,
        cutmix_prob=0.5,
        cutmix_warmup_iter=7615,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Faithful adversarial trainers (WGAN-VGG, DU-GAN). "
                    "Method hyperparameters default to the official "
                    "ldct-benchmark hpopt values."
    )
    p.add_argument("--arch", required=True, choices=list(ADV_ARCHS))
    p.add_argument("--data-dir", default=cfg.DATA_DIR)
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Default: official budget (wganvgg 67306, dugan "
                        "33875). Use 30000 for matched-budget study runs.")
    p.add_argument("--iterations-before-val", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patch-size", type=int, default=None,
                   help="Training patch size. Also fixes the WGAN-VGG critic "
                        "input size (it has a fully-connected head).")
    p.add_argument("--val-patch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--adam-b1", type=float, default=None)
    p.add_argument("--adam-b2", type=float, default=None)
    p.add_argument("--n-d-train", type=int, default=None)
    p.add_argument("--gp-lambda", type=float, default=10.0)
    p.add_argument("--lam-perc", type=float, default=None)
    p.add_argument("--lam-adv", type=float, default=None)
    p.add_argument("--lam-px-im", type=float, default=None)
    p.add_argument("--lam-px-grad", type=float, default=None)
    p.add_argument("--lam-cutmix", type=float, default=None)
    p.add_argument("--cutmix-prob", type=float, default=None)
    p.add_argument("--cutmix-warmup-iter", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--output-root", default="runs_adv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train-patients", type=int, default=None,
                   help="PILOT MODE: ranking only, not reportable.")
    p.add_argument("--val-patients", type=int, default=None)
    p.add_argument("--select-by", choices=list(_SELECT_CHOICES),
                   default="bench_ssim")
    p.add_argument("--val-vif", action="store_true")
    # Optional physics components (OFF by default = faithful reproduction).
    p.add_argument("--use-spectral-head", action="store_true")
    p.add_argument("--spectral-bins", type=int, default=32)
    p.add_argument("--hu-bin-loss", type=float, default=0.0)
    p.add_argument("--nps-weight", type=float, default=0.0)
    return p.parse_args()


def fill_official_defaults(args):
    for key, value in OFFICIAL[args.arch].items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def physics_terms(pred, target, inp, args):
    """Optional study additions to the generator objective (default OFF)."""
    extra = pred.new_zeros(())
    if args.hu_bin_loss > 0.0:
        extra = extra + args.hu_bin_loss * hu_bin_bias_loss(pred, target)
    if args.nps_weight > 0.0:
        extra = extra + args.nps_weight * _get_nps_loss()(
            inp - pred, inp - target
        )
    return extra


def build_generator(args, device):
    base = (WGANVGGGenerator() if args.arch == "wganvgg"
            else DUGANGenerator()).to(device)
    if args.use_spectral_head:
        return SpectralResidualModel(base, n_bins=args.spectral_bins).to(device)
    return base


class WGANVGGTrainer:
    """Replicates ldctbench/methods/wganvgg/Trainer.py."""

    loss_names = ("D", "GP", "G_adv", "G_perc")

    def __init__(self, args, device):
        self.args = args
        self.dev = device
        self.model = build_generator(args, device)
        self.critic = WGANVGGCritic(input_size=args.patch_size).to(device)
        self.perceptual = PerceptualLoss(
            network="vgg19", device=device, layers=[34], in_ch=1, norm="l1"
        )
        betas = (args.adam_b1, args.adam_b2)
        self.g_optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=args.lr, betas=betas)
        self.d_optimizer = torch.optim.Adam(self.critic.parameters(),
                                            lr=args.lr, betas=betas)
        self.iteration = 0

    def gradient_penalty(self, target, fake):
        a = torch.rand(target.size(0), 1, 1, 1, device=self.dev)
        interp = (a * target + (1 - a) * fake).requires_grad_(True)
        d_interp = self.critic(interp)
        gradients = torch.autograd.grad(
            outputs=d_interp,
            inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.args.gp_lambda

    def train_step(self, x, y):
        args = self.args
        # Train critic. The reference zero_grads once before the loop; with
        # the official n_d_train=1 this per-step zeroing is identical (and
        # correct for n_d_train > 1).
        for _ in range(args.n_d_train):
            self.d_optimizer.zero_grad(set_to_none=True)
            fakes = self.model(x)
            critic_loss = (torch.mean(self.critic(fakes))
                           - torch.mean(self.critic(y)))
            grad_p = self.gradient_penalty(y, fakes)
            loss_d = critic_loss + grad_p
            loss_d.backward()
            self.d_optimizer.step()

        # Train generator: perceptual + adversarial, NO pixelwise MSE
        # (faithful to the method).
        self.g_optimizer.zero_grad(set_to_none=True)
        fakes = self.model(x)
        loss_g_adv = -torch.mean(self.critic(fakes))
        loss_g_perc = self.perceptual(fakes, y)
        loss_g = (args.lam_perc * loss_g_perc + loss_g_adv
                  + physics_terms(fakes, y, x, args))
        loss_g.backward()
        self.g_optimizer.step()

        self.iteration += 1
        return {
            "D": float(loss_d.detach()),
            "GP": float(grad_p.detach()),
            "G_adv": float(loss_g_adv.detach()),
            "G_perc": float(loss_g_perc.detach()),
        }

    def set_train(self):
        self.model.train()
        self.critic.train()

    def state(self):
        return {
            "critic_state": self.critic.state_dict(),
            "g_optimizer_state": self.g_optimizer.state_dict(),
            "d_optimizer_state": self.d_optimizer.state_dict(),
        }

    def load_state(self, adv_state):
        if "critic_state" in adv_state:
            self.critic.load_state_dict(adv_state["critic_state"])
        if "g_optimizer_state" in adv_state:
            self.g_optimizer.load_state_dict(adv_state["g_optimizer_state"])
        if "d_optimizer_state" in adv_state:
            self.d_optimizer.load_state_dict(adv_state["d_optimizer_state"])


class DUGANTrainer:
    """Replicates ldctbench/methods/dugan/Trainer.py."""

    loss_names = ("D_img", "D_grad", "G_pix", "G_grad", "G")

    def __init__(self, args, device):
        self.args = args
        self.dev = device
        self.model = build_generator(args, device)
        disc = UNet(repeat_num=6, use_discriminator=True, conv_dim=64,
                    use_sigmoid=False).to(device)
        self.im_discriminator = turn_on_spectral_norm(disc)
        self.grad_discriminator = copy.deepcopy(self.im_discriminator)
        # The reference builds plain Adam(lr) (PyTorch default betas) for all
        # three optimizers, ignoring adam_b1 in its own yaml. Replicated.
        self.g_optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=args.lr)
        self.im_d_optimizer = torch.optim.Adam(
            self.im_discriminator.parameters(), lr=args.lr)
        self.grad_d_optimizer = torch.optim.Adam(
            self.grad_discriminator.parameters(), lr=args.lr)
        self.criterion = ls_gan
        self.sobel = SobelOperator().to(device)
        # Pre-drawn CutMix coin flips as in the reference. A seeded generator
        # is used here so --resume reproduces the same schedule.
        max_iter_upper = (args.max_iterations + args.iterations_before_val
                          - (args.max_iterations % args.iterations_before_val))
        gen = torch.Generator().manual_seed(int(args.seed))
        self.apply_cutmix_prob = torch.rand(max_iter_upper, generator=gen)
        self.iteration = 0

    def warmup(self):
        return min(
            self.iteration * self.args.cutmix_prob
            / self.args.cutmix_warmup_iter,
            self.args.cutmix_prob,
        )

    def train_discriminator(self, discriminator, optimizer, inputs, targets,
                            fakes):
        optimizer.zero_grad(set_to_none=True)
        real_enc, real_dec = discriminator(targets)
        fake_enc, fake_dec = discriminator(fakes.detach())
        source_enc, source_dec = discriminator(inputs)

        d_loss = (
            self.criterion(real_enc, 1.0)
            + self.criterion(real_dec, 1.0)
            + self.criterion(fake_enc, 0.0)
            + self.criterion(fake_dec, 0.0)
            + self.criterion(source_enc, 0.0)
            + self.criterion(source_dec, 0.0)
        )

        idx = min(self.iteration - 1, self.apply_cutmix_prob.numel() - 1)
        if self.apply_cutmix_prob[idx] < self.warmup():
            mask = cutmix(real_dec.size()).to(real_dec)
            cutmix_enc, cutmix_dec = discriminator(
                mask_src_tgt(targets, fakes.detach(), mask)
            )
            cutmix_disc_loss = (self.criterion(cutmix_enc, 0.0)
                                + self.criterion(cutmix_dec, mask))
            cr_loss = F.mse_loss(cutmix_dec,
                                 mask_src_tgt(real_dec, fake_dec, mask))
            d_loss = d_loss + cutmix_disc_loss + cr_loss * self.args.lam_cutmix

        d_loss.backward()
        optimizer.step()
        return float(d_loss.detach())

    def train_step(self, x, y):
        args = self.args
        self.iteration += 1

        gen_full_dose = self.model(x)
        grad_gen_full_dose = self.sobel(gen_full_dose)
        grad_low_dose = self.sobel(x)
        grad_full_dose = self.sobel(y)

        # Train image-domain discriminator (n_d_train times, as reference).
        for _ in range(args.n_d_train):
            im_d_loss = self.train_discriminator(
                self.im_discriminator, self.im_d_optimizer,
                x, y, gen_full_dose,
            )

        # Train generator (and the gradient-domain discriminator in between,
        # exactly as in the reference implementation).
        self.g_optimizer.zero_grad(set_to_none=True)
        img_gen_enc, img_gen_dec = self.im_discriminator(gen_full_dose)
        img_gen_loss = (self.criterion(img_gen_enc, 1.0)
                        + self.criterion(img_gen_dec, 1.0))

        grad_d_loss = self.train_discriminator(
            self.grad_discriminator, self.grad_d_optimizer,
            grad_low_dose, grad_full_dose, grad_gen_full_dose,
        )
        grad_gen_enc, grad_gen_dec = self.grad_discriminator(
            grad_gen_full_dose)
        grad_gen_loss = (self.criterion(grad_gen_enc, 1.0)
                         + self.criterion(grad_gen_dec, 1.0))

        # Pixelwise losses.
        pix_loss = F.mse_loss(gen_full_dose, y)
        grad_loss = F.l1_loss(grad_gen_full_dose, grad_full_dose)

        total_loss = (
            grad_gen_loss * args.lam_adv
            + img_gen_loss * args.lam_adv
            + pix_loss * args.lam_px_im
            + grad_loss * args.lam_px_grad
            + physics_terms(gen_full_dose, y, x, args)
        )
        total_loss.backward()
        self.g_optimizer.step()

        return {
            "D_img": im_d_loss,
            "D_grad": grad_d_loss,
            "G_pix": float(pix_loss.detach()),
            "G_grad": float(grad_loss.detach()),
            "G": float(total_loss.detach()),
        }

    def set_train(self):
        self.model.train()
        self.im_discriminator.train()
        self.grad_discriminator.train()

    def state(self):
        return {
            "im_discriminator_state": self.im_discriminator.state_dict(),
            "grad_discriminator_state": self.grad_discriminator.state_dict(),
            "g_optimizer_state": self.g_optimizer.state_dict(),
            "im_d_optimizer_state": self.im_d_optimizer.state_dict(),
            "grad_d_optimizer_state": self.grad_d_optimizer.state_dict(),
        }

    def load_state(self, adv_state):
        modules = (
            ("im_discriminator_state", self.im_discriminator),
            ("grad_discriminator_state", self.grad_discriminator),
        )
        for key, module in modules:
            if key in adv_state:
                module.load_state_dict(adv_state[key])
        optimizers = (
            ("g_optimizer_state", self.g_optimizer),
            ("im_d_optimizer_state", self.im_d_optimizer),
            ("grad_d_optimizer_state", self.grad_d_optimizer),
        )
        for key, opt in optimizers:
            if key in adv_state:
                opt.load_state_dict(adv_state[key])


def train_cycle(trainer, loader, device, max_iter):
    trainer.set_train()
    sums = {name: 0.0 for name in trainer.loss_names}
    count = 0
    bar = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for batch in bar:
        if trainer.iteration >= max_iter:
            break
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        losses = trainer.train_step(x, y)
        for name, value in losses.items():
            sums[name] += value
        count += 1
        bar.set_postfix(iter=trainer.iteration,
                        **{k: f"{v:.4f}" for k, v in losses.items()})
    return {name: value / max(1, count) for name, value in sums.items()}


def main():
    args = parse_args()
    if cfg.HU_RANGE_PRESET != "benchmark":
        raise RuntimeError("Set HU_RANGE_PRESET=benchmark.")
    fill_official_defaults(args)
    if args.nps_weight < 0.0:
        raise ValueError("--nps-weight must be >= 0")
    if args.spectral_bins < 2:
        raise ValueError("--spectral-bins must be >= 2")
    if args.train_patients is not None and args.train_patients < 1:
        raise ValueError("--train-patients must be >= 1")
    if args.val_patients is not None and args.val_patients < 1:
        raise ValueError("--val-patients must be >= 1")

    n_train, n_val = apply_split(args.split)
    if args.train_patients is not None:
        n_train = min(n_train, args.train_patients)
    if args.val_patients is not None:
        n_val = min(n_val, args.val_patients)
    pilot = args.train_patients is not None or args.val_patients is not None

    cfg.SEED = int(args.seed)
    setup_reproducibility(args.seed)
    device = get_device()
    out_dir = os.path.join(args.output_root, args.arch)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    with_vif = bool(args.val_vif or args.select_by in ("vif", "chest_vif"))

    if args.arch == "wganvgg":
        trainer = WGANVGGTrainer(args, device)
        loss_desc = (f"{args.lam_perc:g}*VGG19-perc(L1@34) + WGAN-adv "
                     f"(GP lam={args.gp_lambda:g}, n_d={args.n_d_train})")
    else:
        trainer = DUGANTrainer(args, device)
        loss_desc = (f"{args.lam_adv:g}*(img+grad adv) "
                     f"+ {args.lam_px_im:g}*MSE "
                     f"+ {args.lam_px_grad:g}*L1(sobel) "
                     f"+ CutMix(lam={args.lam_cutmix:g}, "
                     f"p={args.cutmix_prob:g}, "
                     f"warmup={args.cutmix_warmup_iter}), "
                     f"n_d={args.n_d_train}")
    if args.hu_bin_loss > 0.0:
        loss_desc += f" + {args.hu_bin_loss}*HU-bin"
    if args.nps_weight > 0.0:
        loss_desc += f" + {args.nps_weight}*NPS"

    print(f"\n{'='*68}")
    print(f"  FAITHFUL ADVERSARIAL TRAINING: {args.arch.upper()}")
    print(f"  split={args.split} | seed={args.seed} | "
          f"budget={args.max_iterations} iters")
    print(f"  batch={args.batch_size} | patch={args.patch_size} | "
          f"lr={args.lr:.6e}")
    print(f"  G loss : {loss_desc}")
    print(f"  Spectral head : {'ON' if args.use_spectral_head else 'off'}")
    print(f"  Select best by: {args.select_by}")
    if pilot:
        print("  *** PILOT MODE: ranking only, not reportable. ***")
    print(f"{'='*68}\n")

    best_score = -float("inf")
    if args.resume and os.path.exists(ckpt_path):
        print(f"  Resuming from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.load_state(ckpt.get("adv_state", {}))
        trainer.iteration = int(ckpt.get("iteration", 0))
        if ckpt.get("select_by") == args.select_by:
            best_score = float(ckpt.get("score", -float("inf")))
        else:
            print(f"  NOTE: checkpoint used --select-by "
                  f"{ckpt.get('select_by')}; resetting best score.")
        print(f"  Resumed at iter {trainer.iteration}")
    elif args.resume:
        print(f"  --resume: no checkpoint at {ckpt_path}, starting fresh.")

    if trainer.iteration >= args.max_iterations:
        print(f"  Training already complete "
              f"({trainer.iteration}/{args.max_iterations}).")
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

    start = time.time()
    cycle = trainer.iteration // args.iterations_before_val

    while trainer.iteration < args.max_iterations:
        cycle += 1
        t0 = time.time()
        train_losses = train_cycle(trainer, train_loader, device,
                                   args.max_iterations)
        val = validate(trainer.model, val_loader, device, with_vif=with_vif)
        score = selection_score(val, args.select_by)

        meta = {
            "architecture": args.arch,
            "trainer": "adversarial",
            "split": args.split,
            "seed": int(args.seed),
            "use_spectral_head": args.use_spectral_head,
            "spectral_bins": args.spectral_bins,
            "hu_bin_loss": args.hu_bin_loss,
            "nps_weight": args.nps_weight,
            "method_hparams": {key: getattr(args, key)
                               for key in OFFICIAL[args.arch]},
            "gp_lambda": (args.gp_lambda if args.arch == "wganvgg"
                          else None),
            "select_by": args.select_by,
            "normalization": "benchmark_meanstd",
            "pixel_mean": BENCHMARK_PIXEL_MEAN,
            "pixel_std": BENCHMARK_PIXEL_STD,
            "pixel_domain": "HU+1024",
            "hu_preset": cfg.HU_RANGE_PRESET,
            "eval_data_range": cfg.EVAL_DATA_RANGE,
            "loss": loss_desc,
            "input_mode": "2d",
            "n_train_patients": n_train,
            "n_val_patients": n_val,
            "max_train_patients": args.train_patients,
            "max_val_patients": args.val_patients,
            "pilot_mode": pilot,
        }
        payload = {
            "model_state_dict": get_state_dict(trainer.model),
            "meta": meta,
            "iteration": trainer.iteration,
            "ssim": val["ssim"],
            "psnr": val["psnr"],
            "val_mse": val["mse"],
            "score": score,
            "select_by": args.select_by,
            "val_detail": {k: v for k, v in val.items()},
        }
        torch.save(payload, os.path.join(out_dir, "last_model.pt"))
        if score > best_score:
            best_score = score
            torch.save(payload, os.path.join(out_dir, "best_model.pt"))
        torch.save({**payload, "adv_state": trainer.state()}, ckpt_path)

        loss_str = " ".join(f"{k} {v:.4f}" for k, v in train_losses.items())
        print(
            f"Cycle {cycle:02d} | "
            f"Iter {trainer.iteration:06d}/{args.max_iterations} | "
            f"{loss_str} | PSNR {val['psnr']:.3f} | "
            f"SSIM {val['ssim']:.5f} | bSSIM {val['bench_ssim']:.5f} | "
            f"RMSE {val['rmse']:.2f} | "
            f"{args.select_by} {score:.5f} | {time.time() - t0:.1f}s"
        )

    total = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print(f"\nDone [{args.arch.upper()} adversarial] in {total} | "
          f"best {args.select_by}={best_score:.5f}")
    print(f"Checkpoint -> {os.path.join(out_dir, 'best_model.pt')}")

    if pilot:
        print("\nPILOT MODE reminder: ranking only. Retrain the winner on "
              "the full split before reporting.")


if __name__ == "__main__":
    main()
