"""Adversarial-training utilities copied from the ldct-benchmark reference
(github.com/eeulig/ldct-benchmark, commit 09b1011bc2fb77ef4fc734cec1e961a20c754910).

- PerceptualLoss, repeat_ch : ldctbench/utils/training_utils.py
- SobelOperator, cutmix, mask_src_tgt, ls_gan, turn_on_spectral_norm :
  ldctbench/methods/dugan/utils.py

The logic is verbatim. Only wandb/distributed helpers were dropped, the
torchvision import was made lazy (only needed for the WGAN-VGG perceptual
loss), and the VGG weights are frozen (the reference never steps them, so
this is mathematically identical but saves memory).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ---- WGAN-VGG perceptual loss (ldctbench/utils/training_utils.py) ----


class repeat_ch:
    """Repeat input 3x in channel dimension if ``in_ch == 1``."""

    def __init__(self, in_ch):
        self.in_ch = in_ch

    def __call__(self, x):
        if self.in_ch == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


class PerceptualLoss(nn.Module):
    """Perceptual loss used for the WGAN-VGG training.

    In Yang et al. 2018 the content loss is evaluated in VGG19 after the
    16th (last) conv layer, i.e. torchvision feature index 34. The
    benchmark trainer instantiates this with network="vgg19", layers=[34],
    in_ch=1 (1->3 channel repeat) and norm="l1".
    """

    def __init__(self, network, device, in_ch=3, layers=(3, 8, 15, 22),
                 norm="l1", return_features=False):
        super().__init__()
        try:
            import torchvision.models as models
        except ImportError as exc:
            raise RuntimeError(
                "PerceptualLoss requires torchvision: pip install torchvision"
            ) from exc
        if network == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        elif network == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
        else:
            raise ValueError(f"Unknown network {network}")
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad_(False)

        self.vgg_features = vgg.features
        self.layers = [str(layer) for layer in layers]
        if norm == "l1":
            self.norm = nn.L1Loss()
        elif norm == "mse":
            self.norm = nn.MSELoss()
        else:
            raise ValueError(f"Norm {norm} not known for PerceptualLoss")
        self.transform = repeat_ch(in_ch)
        self.return_features = return_features

    def forward(self, input, target):
        input = self.transform(input)
        target = self.transform(target)

        loss = 0.0
        if self.return_features:
            features = {"input": [], "target": []}

        for i, m in self.vgg_features._modules.items():
            input = m(input)
            target = m(target)

            if i in self.layers:
                loss += self.norm(input, target)
                if self.return_features:
                    features["input"].append(input.clone())
                    features["target"].append(target.clone())
                if i == self.layers[-1]:
                    break

        return (loss, features) if self.return_features else loss


# ---- DU-GAN utilities (ldctbench/methods/dugan/utils.py) ----


def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target


def cutmix(mask_size):
    mask = torch.ones(mask_size)
    lam = np.random.beta(1.0, 1.0)
    _, _, height, width = mask_size
    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))
    mask[:, :, y0:y1, x0:x1] = 0
    return mask


class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

        self.register_buffer(
            "conv_x",
            torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 4,
        )
        self.register_buffer(
            "conv_y",
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 4,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x**2 + grad_y**2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


def turn_on_spectral_norm(module):
    module_output = module
    if isinstance(module, torch.nn.Conv2d):
        if module.out_channels != 1 and module.in_channels > 4:
            module_output = nn.utils.spectral_norm(module)
    for name, child in module.named_children():
        module_output.add_module(name, turn_on_spectral_norm(child))
    del module
    return module_output


def ls_gan(inputs, targets):
    return torch.mean((inputs - targets) ** 2)
