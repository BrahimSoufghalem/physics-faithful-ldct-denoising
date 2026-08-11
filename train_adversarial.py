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
            self.grad_discriminator.parameters(