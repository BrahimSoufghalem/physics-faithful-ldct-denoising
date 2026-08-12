"""Physics-informed spectral residual head (architectural physics).

Physics background
------------------
In CT, noise texture is set by the FBP reconstruction kernel: a radially
symmetric filter in the 2-D frequency plane (ramp + apodization). Because
backprojection averages over all view angles, the image-domain noise power
spectrum (NPS) is approximately isotropic: a function of the radial spatial
frequency |f| only.

Mechanism
---------
``SpectralResidualHead`` applies a LEARNABLE RADIAL GAIN ``G(|f|)`` to the
noise a base denoiser wants to remove::

    y_hat  = base(x)                       # any denoiser (e.g. RED-CNN)
    n_hat  = x - y_hat                     # noise the base wants to remove
    m      = mean(n_hat)                   # per-image DC (HU calibration)
    n'     = IFFT2( G(|f|) * FFT2(n_hat - m) ) + m
    output = x - n'

Structural constraints (what a plain CNN cannot bypass):

* ``G`` depends only on the radial frequency ``|f|`` -- the same rotational
  symmetry that FBP physics imposes on the NPS. The head cannot build a
  directional or spatially varying filter.
* DC-PRESERVING (v2): the per-image mean of the removed noise passes
  through UNCHANGED. HU calibration is a zero-frequency property; the head
  shapes noise TEXTURE only. (In v1 the gain multiplied the DC component
  too; under pure MSE it drifted and introduced a soft-tissue HU bias of a
  few HU. Found experimentally with evaluate_physics.py.)
* ``G`` is parameterized by ``n_bins`` (~32) interpolated log-gain knots and
  initialized to ``G == 1``, so the wrapped model starts EXACTLY equal to
  the base model (safe start; cannot begin worse than the trunk).
* Frequencies are normalized by the Nyquist frequency, so the same curve
  serves 128x128 training patches and 512x512 test slices (no train/test
  extent shift).

Refinements (v2.1, both optional and OFF by default):

* ``freeze_dc_bins=N`` pins the first N knots to G=1. Exact DC is always
  preserved, but the lowest non-zero frequencies carry REGIONAL HU means
  (whole-organ scale); freezing them prevents the head from trading
  regional HU calibration for spectral fit (observed experimentally as a
  chest soft-tissue bias of several HU) and removes the sharp near-DC gain
  jump suspected of causing faint concentric rings in difference maps.
* ``smoothness_penalty()`` returns a quadratic smoothness term on the
  (effective) log-gain knots, for use as a training regularizer (see
  train.py --gain-tv-weight). Because it is computed on the EFFECTIVE
  curve, it also pulls the first free knot toward the frozen G=1 region.
  With the v3 adaptive head it regularizes the SHARED base curve only.

Adaptive conditioning (v3, optional and OFF by default):

* ``adaptive=True`` makes the radial gain IMAGE-ADAPTIVE: a tiny
  conditioning encoder (~5k parameters: three strided 3x3 convs + global
  average pooling + one zero-initialized linear layer) predicts a bounded
  per-image offset to the shared log-gain knots from the network INPUT
  (the LDCT image)::

      G_i(|f|) = exp( log_gain + max_log_gain_delta * tanh(proj(enc(x_i))) )

* Motivation (measured in this study): with one shared curve the head must
  compromise between anatomies whose noise differs strongly (chest
  NPS_LogL1 ~0.12 vs abdomen ~0.43 for the same model). A static radial
  curve can in principle be absorbed by the trunk; a per-image curve is
  capacity the trunk does not expose in an interpretable,
  physics-constrained form.
* All guarantees hold PER IMAGE: exact DC preservation, radial symmetry,
  ``tanh``-bounded offset magnitude, and ``freeze_dc_bins`` also pins the
  ADAPTIVE offsets of the frozen knots to 0.
* The projection layer is ZERO-INITIALIZED, so at initialization the
  adaptive head is EXACTLY the static head (which itself starts at the
  identity): training cannot start worse than v2.1.
* Interpretability: ``per_image_gain_curves()`` returns the per-image
  curves; plotting them for chest vs abdomen slices shows whether the head
  learned anatomy-dependent physics.

Checkpoint compatibility: freezing is implemented by recomposing the
log-gain vector at run time (frozen entries contribute log-gain 0 and
receive no gradient), so the state dict layout is UNCHANGED by
``freeze_dc_bins``. Checkpoints trained with or without freezing load into
either configuration; a frozen-trained checkpoint evaluates identically in
an unfrozen model because its frozen entries remain at 0 (G = exp(0) = 1).
The v3 adaptive head ADDS parameters (``cond_encoder``/``cond_proj``), so
adaptive checkpoints require ``adaptive=True`` at load time; train.py
stores ``adaptive_head`` and ``adaptive_max_delta`` in the checkpoint meta
and the eval scripts rebuild the model from it. Static checkpoints do NOT
load into adaptive models and vice versa.

Interpretability: after training, plot the learned curve with
``plot_spectral_gain.py``. ``G < 1`` at a band means the head RETURNS part
of the energy the trunk wanted to remove there (protects detail /
noise-texture); ``G > 1`` means it removes more.

Compatibility: v1 checkpoints load into v2 (identical parameters), but the
forward pass differs (DC preservation); re-train or at least re-evaluate
before comparing numbers across versions.

Limitations: the radial-symmetry constraint is exact for FBP
reconstructions (our benchmark data); for iterative/deep reconstructions
the NPS is less isotropic and a low-parameter 2-D gain would be the natural
extension.
"""

import torch
import torch.nn as nn


class SpectralResidualHead(nn.Module):
    """Learnable radial gain applied in the Fourier domain (DC-preserving).

    Parameters
    ----------
    n_bins : int
        Number of radial gain knots. The knots span radial frequencies
        ``[0, sqrt(2)]`` in units of the (axis) Nyquist frequency; values
        between knots are linearly interpolated.
    freeze_dc_bins : int
        Pin the first ``freeze_dc_bins`` knots to G=1 (identity). With 32
        bins each knot covers ~0.044 Nyquist units, so e.g. 2 protects
        radial frequencies below ~0.09 Nyquist. Applies to the adaptive
        offsets too. Default 0 (no freezing, previous behavior).
    adaptive : bool
        v3: predict bounded PER-IMAGE offsets to the log-gain knots from a
        conditioning image (see module docstring). Default False (static
        curve, previous behavior).
    cond_channels : int
        Width of the conditioning encoder (adaptive=True only).
    max_log_gain_delta : float
        Bound on the per-image |log-gain offset| (adaptive=True only).
        0.25 allows scaling the shared curve by exp(+/-0.25) ~ x0.78-x1.28
        per band.
    """

    _RMAX = 2.0 ** 0.5  # corner of the 2-D frequency plane, Nyquist units

    def __init__(self, n_bins: int = 32, freeze_dc_bins: int = 0,
                 adaptive: bool = False, cond_channels: int = 16,
                 max_log_gain_delta: float = 0.25):
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        if not 0 <= int(freeze_dc_bins) < int(n_bins):
            raise ValueError(
                f"freeze_dc_bins must be in [0, n_bins), got {freeze_dc_bins}"
            )
        if max_log_gain_delta <= 0.0:
            raise ValueError(
                f"max_log_gain_delta must be > 0, got {max_log_gain_delta}"
            )
        self.n_bins = int(n_bins)
        self.freeze_dc_bins = int(freeze_dc_bins)
        self.adaptive = bool(adaptive)
        self.max_log_gain_delta = float(max_log_gain_delta)
        # Log-parameterization: G = exp(log_gain) > 0; init log_gain = 0
        # -> G == 1 -> the head is the identity at initialization.
        self.log_gain = nn.Parameter(torch.zeros(self.n_bins))
        if self.adaptive:
            c = int(cond_channels)
            # Tiny conditioning encoder: enough to tell anatomies / noise
            # levels apart, far too small to denoise anything itself.
            self.cond_encoder = nn.Sequential(
                nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.cond_proj = nn.Linear(c, self.n_bins)
            # ZERO-INIT: the adaptive offset starts at exactly 0, so the
            # adaptive head begins as the static head (safe start).
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        # Cache of interpolation grids keyed by (H, W, device); tensors are
        # not parameters and are rebuilt lazily per resolution/device.
        self._grid_cache = {}

    def effective_log_gain(self) -> torch.Tensor:
        """log-gain with the first ``freeze_dc_bins`` knots pinned to 0.

        Recomposition (not an in-place mask): frozen entries contribute a
        constant 0 and receive no gradient, so they stay at their initial
        value in the state dict and checkpoints remain fully compatible in
        both directions.
        """
        if self.freeze_dc_bins == 0:
            return self.log_gain
        return torch.cat([
            self.log_gain.new_zeros(self.freeze_dc_bins),
            self.log_gain[self.freeze_dc_bins:],
        ])

    def log_gain_delta(self, cond: torch.Tensor) -> torch.Tensor:
        """Bounded per-image log-gain offsets, shape (B, n_bins).

        ``cond`` is the conditioning image, (B, 1, H, W). Frozen knots
        (``freeze_dc_bins``) get offset 0, mirroring the static freezing.
        """
        if not self.adaptive:
            raise RuntimeError("log_gain_delta requires adaptive=True")
        feat = self.cond_encoder(cond).flatten(1)          # (B, C)
        delta = self.max_log_gain_delta * torch.tanh(self.cond_proj(feat))
        if self.freeze_dc_bins > 0:
            delta = torch.cat([
                delta.new_zeros(delta.shape[0], self.freeze_dc_bins),
                delta[:, self.freeze_dc_bins:],
            ], dim=1)
        return delta

    def gain_curve(self) -> torch.Tensor:
        """Shared (static) radial gain G at the knots, shape (n_bins,).

        For the adaptive head this is the SHARED base curve; per-image
        curves are available via ``per_image_gain_curves()``.
        """
        return torch.exp(self.effective_log_gain())

    @torch.no_grad()
    def per_image_gain_curves(self, cond: torch.Tensor) -> torch.Tensor:
        """Per-image G at the knots, shape (B, n_bins). Analysis/plots.

        For a static head this is the shared curve expanded over the batch.
        Plotting these for chest vs abdomen slices shows whether the head
        learned anatomy-dependent spectral behavior.
        """
        if not self.adaptive:
            g = self.gain_curve()
            return g[None, :].expand(cond.shape[0], -1).clone()
        return torch.exp(
            self.effective_log_gain()[None, :] + self.log_gain_delta(cond)
        )

    def smoothness_penalty(self) -> torch.Tensor:
        """Quadratic smoothness of the effective log-gain curve.

        Mean squared difference of consecutive knots. Discourages sharp
        spectral transitions (ring artifacts, abrupt near-DC jumps). Using
        the EFFECTIVE curve also pulls the first free knot toward the
        frozen G=1 region when ``freeze_dc_bins > 0``. With adaptive=True
        this regularizes the SHARED base curve only.
        """
        lg = self.effective_log_gain()
        d = lg[1:] - lg[:-1]
        return (d * d).mean()

    def _interp_grid(self, h: int, w: int, device):
        key = (h, w, str(device))
        cached = self._grid_cache.get(key)
        if cached is not None:
            return cached
        fy = torch.fft.fftfreq(h, device=device)      # cycles/pixel, (H,)
        fx = torch.fft.rfftfreq(w, device=device)     # cycles/pixel, (W//2+1,)
        r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
        # r = 0.5 cycles/pixel is the Nyquist along an axis. Normalizing by
        # it makes the curve resolution-independent; the 2-D plane extends
        # to sqrt(2) Nyquist units at the corners.
        pos = (r / 0.5) / self._RMAX * (self.n_bins - 1)
        idx0 = pos.floor().long().clamp(0, self.n_bins - 1)
        idx1 = (idx0 + 1).clamp(0, self.n_bins - 1)
        frac = (pos - idx0.to(pos.dtype)).clamp(0.0, 1.0)
        cached = (idx0, idx1, frac)
        self._grid_cache[key] = cached
        return cached

    def forward(self, noise: torch.Tensor,
                cond: torch.Tensor = None) -> torch.Tensor:
        """Shape the spectrum of ``noise`` (B, C, H, W) with G(|f|).

        DC-preserving: the per-image spatial mean of ``noise`` is removed
        before shaping and added back unchanged, so the head cannot alter
        HU calibration (zero-frequency content).

        ``cond`` (adaptive=True only): conditioning image for the per-image
        gain offsets; the wrapper passes the network INPUT x. Falls back to
        ``noise`` itself when omitted. Ignored by the static head.
        """
        h, w = noise.shape[-2], noise.shape[-1]
        dc = noise.mean(dim=(-2, -1), keepdim=True)
        ac = noise - dc
        idx0, idx1, frac = self._interp_grid(h, w, noise.device)
        if self.adaptive:
            cond_in = cond if cond is not None else noise
            lg = self.effective_log_gain()[None, :] + self.log_gain_delta(cond_in)
            g = torch.exp(lg)                                  # (B, n_bins)
            gain2d = (1.0 - frac) * g[:, idx0] + frac * g[:, idx1]  # (B,H,W')
            gain2d = gain2d[:, None]                           # (B,1,H,W')
        else:
            g = self.gain_curve()
            gain2d = (1.0 - frac) * g[idx0] + frac * g[idx1]   # (H, W//2+1)
        spec = torch.fft.rfft2(ac, norm="ortho")
        shaped = torch.fft.irfft2(spec * gain2d, s=(h, w), norm="ortho")
        return shaped + dc


class SpectralResidualModel(nn.Module):
    """Wrap any image-domain denoiser with the spectral residual head.

    ``forward``: ``x - head(x - base(x))``. With the head at initialization
    (G == 1) this is exactly ``base(x)``. The adaptive head is conditioned
    on the network input x (anatomy + noise level information).
    """

    def __init__(self, base: nn.Module, n_bins: int = 32,
                 freeze_dc_bins: int = 0, adaptive: bool = False,
                 cond_channels: int = 16, max_log_gain_delta: float = 0.25):
        super().__init__()
        self.base = base
        self.head = SpectralResidualHead(
            n_bins=n_bins,
            freeze_dc_bins=freeze_dc_bins,
            adaptive=adaptive,
            cond_channels=cond_channels,
            max_log_gain_delta=max_log_gain_delta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_hat = self.base(x)
        noise = x - y_hat
        return x - self.head(noise, cond=x)


def wrap_with_spectral_head(base: nn.Module, n_bins: int = 32,
                            freeze_dc_bins: int = 0, adaptive: bool = False,
                            cond_channels: int = 16,
                            max_log_gain_delta: float = 0.25
                            ) -> SpectralResidualModel:
    """Convenience constructor used by train/eval scripts."""
    return SpectralResidualModel(base, n_bins=n_bins,
                                 freeze_dc_bins=freeze_dc_bins,
                                 adaptive=adaptive,
                                 cond_channels=cond_channels,
                                 max_log_gain_delta=max_log_gain_delta)
