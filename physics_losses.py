"""Ablation-safe medical-physics losses for reconstructed CT images.

MSE is always present. Radial NPS matching and HU-bin bias are independently
opt-in so architectural and physical contributions can be measured separately.
No patient metadata is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchRadialNPSLoss(nn.Module):
    """Match batch-mean radial NPS of removed and paired-reference residuals."""

    def __init__(self, eps=1e-8, remove_mean=True, hann_window=True):
        super().__init__()
        self.eps = float(eps)
        self.remove_mean = bool(remove_mean)
        self.hann_window = bool(hann_window)

    def _power(self, x):
        x = x.float()
        if self.remove_mean:
            x = x - x.mean(dim=(-2, -1), keepdim=True)
        if self.hann_window:
            h, w = x.shape[-2:]
            wy = torch.hann_window(h, periodic=False, device=x.device, dtype=x.dtype)
            wx = torch.hann_window(w, periodic=False, device=x.device, dtype=x.dtype)
            window = (wy[:, None] * wx[None, :]).view(1, 1, h, w)
            # Preserve comparable power after tapering.
            x = x * window / window.square().mean().sqrt().clamp_min(self.eps)
        fft = torch.fft.fft2(x, norm="ortho")
        return fft.real.square() + fft.imag.square()

    def _radial_average(self, power):
        # Average channels and batch first: the physical target is a population
        # spectrum, not a noisy per-patient fingerprint.
        p = power.mean(dim=(0, 1))
        h, w = p.shape
        fy = torch.fft.fftfreq(h, device=p.device)
        fx = torch.fft.fftfreq(w, device=p.device)
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
        n_bins = max(8, min(h, w) // 2)
        indices = torch.clamp((radius / 0.5 * (n_bins - 1)).long(), 0, n_bins - 1)
        sums = torch.zeros(n_bins, device=p.device, dtype=p.dtype)
        counts = torch.zeros(n_bins, device=p.device, dtype=p.dtype)
        sums.scatter_add_(0, indices.reshape(-1), p.reshape(-1))
        counts.scatter_add_(0, indices.reshape(-1), torch.ones_like(p).reshape(-1))
        return sums / counts.clamp_min(1.0)

    def forward(self, predicted_noise, reference_noise):
        pred_nps = self._radial_average(self._power(predicted_noise))
        true_nps = self._radial_average(self._power(reference_noise))
        # Log domain balances low- and high-energy frequency bins.
        return F.l1_loss(
            torch.log(pred_nps + self.eps),
            torch.log(true_nps + self.eps),
        )


class HUBinBiasLoss(nn.Module):
    """Mean-HU preservation within fixed tissue-intensity intervals."""

    def __init__(self, a_min, a_max,
                 boundaries=(-1024.0, -500.0, -200.0, 200.0, 600.0, 1900.0),
                 min_pixels=64):
        super().__init__()
        self.a_min = float(a_min)
        self.a_max = float(a_max)
        self.boundaries = tuple(float(v) for v in boundaries)
        self.min_pixels = int(min_pixels)

    def forward(self, pred, target):
        target_hu = target.float() * (self.a_max - self.a_min) + self.a_min
        pred_hu = pred.float() * (self.a_max - self.a_min) + self.a_min
        terms = []
        for lo, hi in zip(self.boundaries[:-1], self.boundaries[1:]):
            mask = (target_hu >= lo) & (target_hu < hi)
            count = mask.sum()
            if int(count.detach()) < self.min_pixels:
                continue
            terms.append((pred_hu[mask].mean() - target_hu[mask].mean()).abs())
        if not terms:
            return pred_hu.sum() * 0.0
        # Return in normalized-image units so lambda_hu has an intuitive scale.
        return torch.stack(terms).mean() / (self.a_max - self.a_min)


class PhysicsInformedCTLoss(nn.Module):
    def __init__(self, a_min, a_max, lambda_nps=0.0, lambda_hu=0.0):
        super().__init__()
        self.lambda_nps = float(lambda_nps)
        self.lambda_hu = float(lambda_hu)
        self.nps = BatchRadialNPSLoss()
        self.hu = HUBinBiasLoss(a_min, a_max)

    def describe(self):
        return (
            f"1*MSE + {self.lambda_nps:g}*batch-radial-NPS "
            f"+ {self.lambda_hu:g}*HU-bin-bias"
        )

    def forward(self, pred_img, target_img, input_img):
        pred = pred_img.float()
        target = target_img.float()
        inp = input_img.float()
        mse = F.mse_loss(pred, target)
        nps = pred.new_zeros(())
        hu = pred.new_zeros(())
        if self.lambda_nps > 0:
            # What the model removed versus the paired LDCT-NDCT residual.
            nps = self.nps(inp - pred, inp - target)
        if self.lambda_hu > 0:
            hu = self.hu(pred, target)
        total = mse + self.lambda_nps * nps + self.lambda_hu * hu
        return total, {
            "MSE": float(mse.detach()),
            "NPS": float(nps.detach()),
            "HU": float(hu.detach()),
            "Total": float(total.detach()),
        }
