"""Benchmark architectures package.

The five ldct-benchmark architectures (github.com/eeulig/ldct-benchmark,
commit 09b1011bc2fb77ef4fc734cec1e961a20c754910), one file per model:

  models/redcnn.py   RED-CNN (Chen et al. 2017)
  models/resnet.py   ResNet  (Park et al. 2017)
  models/dugan.py    DU-GAN  (Huang et al. 2021)  generator (= RED-CNN) + U-Net discriminator
  models/wganvgg.py  WGAN-VGG (Yang et al. 2018)  generator + critic
  models/transct.py  TransCT (Zhang et al. 2021)  -- requires 512x512 inputs

Every trunk maps a 1-channel image to a 1-channel image, so the DC-preserving
spectral residual head (spectral_head.SpectralResidualModel) can wrap ANY of
them: pass --use-spectral-head to train.py with any --arch.

NOTE: DU-GAN and WGAN-VGG are adversarially trained methods in the benchmark
paper. train.py trains their GENERATORS with the study loss (MSE + optional
physics losses) only; results are therefore ablations of the trunk
architecture, not reproductions of the adversarial methods.
"""

from models.dugan import DUGANGenerator
from models.redcnn import RedCNN
from models.resnet import ResNet
from models.transct import TransCT
from models.wganvgg import WGANVGGGenerator

ARCHITECTURES = {
    "redcnn":  RedCNN,
    "resnet":  ResNet,
    "dugan":   DUGANGenerator,
    "wganvgg": WGANVGGGenerator,
    "transct": TransCT,
}

ARCH_CHOICES = tuple(ARCHITECTURES.keys())

# Architectures that only accept a fixed input size (see models/transct.py).
FIXED_INPUT_SIZE = {"transct": 512}


def build_bare_model(name: str):
    """Instantiate an architecture by name (no device placement, no prints)."""
    key = name.lower().strip()
    if key not in ARCHITECTURES:
        raise ValueError(
            f"Unknown benchmark architecture: '{name}'. "
            f"Use one of: {', '.join(ARCH_CHOICES)}."
        )
    return ARCHITECTURES[key]()


def build_benchmark_model(name: str, device):
    """Instantiate an architecture, move it to `device` and print a summary."""
    model = build_bare_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : {name.upper()}")
    print(f"  Parameters   : {n_params:,}")
    return model
