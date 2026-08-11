"""
Benchmark Architectures - RED-CNN and ResNet
=============================================
Direct copies of the architectures from eeulig/ldct-benchmark
(commit 09b1011bc2fb77ef4fc734cec1e961a20c754910).

Sources:
  ldctbench/methods/redcnn/network.py
  ldctbench/methods/resnet/network.py

No architectural changes. Plain nn.Module subclasses without Namespace arg.
"""

import torch.nn as nn


# ===========================================================================
# RED-CNN
# ===========================================================================
class RedCNN(nn.Module):
    """RED-CNN (Chen et al. 2017).

    Copied exactly from ldctbench/methods/redcnn/network.py.
    5 Conv + 5 ConvTranspose with skip connections.
    Direct output (not residual subtraction).
    Final ReLU omitted to allow zero-mean normalised data.
    """

    def __init__(self, out_ch: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out += residual_1
        return out

    @staticmethod
    def model_config():
        return {"out_ch": 96}


# ===========================================================================
# ResNet
# ===========================================================================
class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, groups=8),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1),
        )

    def forward(self, x):
        return self.layers(x) + x


class ResNet(nn.Module):
    """Residual Network (Park et al. 2017).

    Copied exactly from ldctbench/methods/resnet/network.py.
    10 residual blocks, 128 channels.
    NOISE SUBTRACTION: output = input - predicted_noise.
    """

    def __init__(self, n_channels: int = 128, n_blocks: int = 10):
        super().__init__()
        self.in_conv = nn.Conv2d(1, n_channels, 9, padding=4)
        self.blocks = nn.ModuleList([_ResBlock(n_channels) for _ in range(n_blocks)])
        self.out_conv = nn.Conv2d(n_channels, 1, 3, padding=1)

    def forward(self, x):
        res = x
        out = self.in_conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.out_conv(out)
        return res - out

    @staticmethod
    def model_config():
        return {"n_channels": 128, "n_blocks": 10}


# ===========================================================================
# Factory
# ===========================================================================
def build_benchmark_model(name: str, device):
    name = name.lower().strip()
    if name == "redcnn":
        model = RedCNN()
    elif name == "resnet":
        model = ResNet()
    else:
        raise ValueError(f"Unknown benchmark architecture: '{name}'. Use 'redcnn' or 'resnet'.")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Architecture : {name.upper()}")
    print(f"  Parameters   : {n_params:,}")
    return model
