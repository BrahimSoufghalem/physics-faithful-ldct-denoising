# Physics-Faithful Low-Dose CT Denoising

A study of **physical fidelity** in deep-learning LDCT denoising, built on the
[ldct-benchmark](https://github.com/eeulig/ldct-benchmark) protocol (Eulig et
al.). Standard metrics (PSNR/SSIM) reward smoothing and are blind to two
clinically relevant failure modes:

1. **Noise-texture distortion** — the denoised image's noise-power spectrum
   (NPS) no longer matches the FBP reconstruction physics.
2. **HU-calibration drift** — mean Hounsfield-Unit bias inside tissue classes
   (air/lung, fat, soft tissue, dense tissue, bone).

This repository adds one **architectural** contribution and two **loss**
contributions on top of exact benchmark trunks (RED-CNN, ResNet), plus a
physics-fidelity evaluation suite that measures what they target.

## Contributions

### 1. DC-preserving spectral residual head (architectural) — `spectral_head.py`
A learnable **radial** gain `G(|f|)` applied in the Fourier domain to the
noise the trunk removes: `output = x - IFFT(G(|f|) * FFT(x - trunk(x)))`.

- **Radial-only** by construction — the same rotational symmetry FBP physics
  imposes on the NPS. The head cannot build directional or spatially varying
  filters.
- **DC-preserving** — the per-image mean of the removed noise passes through
  unchanged, so the head cannot alter HU calibration (zero-frequency content).
- **Safe start** — `G ≡ 1` at initialization, so the wrapped model starts
  exactly at the trunk.
- **Interpretable** — plot the learned curve with `plot_spectral_gain.py`.

### 2. Radial NPS matching loss — `physics_losses.py`
Log-domain L1 distance between the radial NPS of the removed noise
(LDCT − pred) and the paired reference residual (LDCT − NDCT).

### 3. HU-bin bias loss
Mean-HU preservation inside fixed physical tissue intervals
(−1024 / −500 / −200 / 200 / 600 / 1900 HU).

### 4. Physics-fidelity evaluation — `evaluate_physics.py`
NPS_LogL1, NPS_Corr, NoiseSTD_Ratio and per-tissue HU bias, complementing the
standard windowed PSNR/SSIM/RMSE/VIF of `evaluate_image.py`.

> **Claim scope**: better *physical faithfulness* (noise texture + HU
> calibration) at essentially equal PSNR/SSIM — not a PSNR improvement.

## Repository structure

| File | Purpose |
| --- | --- |
| `config.py` | Central constants: benchmark HU preset, splits, paths |
| `download.py` | Download the Mayo/TCIA `LDCT-and-projection-data` patients (NBIA) |
| `benchmark_data.py` | Benchmark-aligned data pipeline (MONAI, mean/std standardization) |
| `benchmark_architectures.py` | Exact RED-CNN / ResNet trunks from ldct-benchmark |
| `spectral_head.py` | DC-preserving learnable radial spectral head |
| `physics_losses.py` | Radial NPS matching + HU-bin bias losses |
| `train.py` | Matched-budget trainer (paper hyperparameters, `bench_ssim` selection, `--seed`) |
| `run_pilot.py` | Sequential pilot sweep (20 default configs) with combined image+physics summary |
| `evaluate_image.py` | Test-set image-quality metrics (windowed PSNR/SSIM, RMSE, VIF) |
| `evaluate_physics.py` | Test-set physics-fidelity metrics (NPS, HU bias) |
| `make_figures.py` | Comparison figures: windowed images, diff maps, ROI zooms, NPS curves |
| `plot_spectral_gain.py` | Plot the learned `G(|f|)` curve of a spectral-head checkpoint |
| `plot_pareto.py` | Accuracy-vs-texture trade-off plot across NPS weights |
| `metrics.py`, `utils.py`, `twenty_patient_split.py` | Shared metrics, helpers, small-split IDs |

## Setup

```bash
pip install -r requirements.txt
# Only if you use --ssim-weight > 0:
pip install pytorch-msssim
```

Download the data (Mayo Clinic LDCT Grand Challenge patients, TCIA collection
`LDCT-and-projection-data`; 90 train/val patients into `dataset/`, 10 test
patients into `test/`):

```bash
python download.py
```

## Protocol

Hyperparameters follow the official `configs/redcnn.yaml` of ldct-benchmark
(batch 73, patch 128, lr 9.583e-5, constant schedule, pure MSE base loss,
best-checkpoint selection by overall unwindowed SSIM). This study uses a
**matched training budget of 30,000 iterations** for every configuration:
all internal comparisons are valid; absolute numbers are not directly
comparable to the published full-budget benchmark table.

### Main runs (matched budget, 30k iterations)

```bash
# C0 — baseline (pure MSE)
HU_RANGE_PRESET=benchmark python train.py --arch redcnn \
    --data-dir dataset --split 100p \
    --max-iterations 30000 --iterations-before-val 1000 \
    --batch-size 73 --patch-size 128 --val-patch-size 128 \
    --lr 9.583417460320728e-05 --lr-schedule constant \
    --select-by bench_ssim --output-root runs_C0

# Full physics model — spectral head + NPS + HU-bin losses
... --use-spectral-head --nps-weight 0.005 --hu-bin-loss 0.2 --output-root runs_S2h

# C1h — losses only, no head (head attribution control)
... --nps-weight 0.005 --hu-bin-loss 0.2 --output-root runs_C1h

# C2 — HU-bin loss only
... --hu-bin-loss 0.2 --output-root runs_C2
```

For multi-seed reporting add `--seed 1 --output-root runs_C0_seed1`, etc.

### Pilot screening (cheap config ranking, not reportable)

```bash
python run_pilot.py --data-dir dataset \
    --train-patients 8 --val-patients 4 \
    --iters 8000 --val-every 1000
```

### Evaluation

```bash
# Image quality (windowed PSNR/SSIM, RMSE in HU, VIF)
HU_RANGE_PRESET=benchmark python evaluate_image.py \
    --test-dir test --runs-root runs_S2h --output eval_S2h

# Physics fidelity (NPS match, per-tissue HU bias)
HU_RANGE_PRESET=benchmark python evaluate_physics.py \
    --test-dir test --runs-root runs_S2h --output eval_physics_S2h
```

### Figures

```bash
HU_RANGE_PRESET=benchmark python make_figures.py \
    --test-dir test --runs-roots runs_C0 runs_S2h \
    --labels "RED-CNN" "RED-CNN+Physics" \
    --patients C121 L006 --output figures

python plot_spectral_gain.py --checkpoint runs_S2h/redcnn/best_model.pt
python plot_pareto.py --csv pareto_points.csv
```

## Results (test set, 10 patients, matched 30k budget, RED-CNN trunk)

| Metric | C0 (pure MSE) | + Head + NPS 0.005 + HU-bin 0.2 | Ideal |
| --- | --- | --- | --- |
| PSNR (windowed) | 30.72 | 30.58 | high |
| SSIM (windowed) | 0.7445 | 0.7373 | high |
| NPS_LogL1 | 0.4550 | **0.2803** | 0 |
| NPS_Corr | 0.7476 | **0.7986** | 1 |
| NoiseSTD_Ratio | 0.8669 | **0.9087** | 1.00 |
| Bias_Bone (HU) | −39.16 | **−5.41** | 0 |
| Bias_Soft (HU) | −2.06 | −1.74 | 0 |

The physics model matches the baseline on standard metrics (ΔPSNR ≈ −0.14 dB)
while substantially improving noise-texture fidelity and reducing bone HU bias
by ~86%.

Known limitations: dense-tissue bias (~−18 HU) is not fully repaired;
abdominal bone bias changes sign vs baseline; a single seed per config so far.

## Acknowledgements

- Benchmark protocol and trunk architectures:
  [eeulig/ldct-benchmark](https://github.com/eeulig/ldct-benchmark)
- Data: *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*,
  Mayo Clinic / TCIA. Use of the data is subject to the TCIA data usage policy.

## License

MIT — see [LICENSE](LICENSE).
