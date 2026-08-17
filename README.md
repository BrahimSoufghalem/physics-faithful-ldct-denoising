# Physics-Faithful Low-Dose CT Denoising

**A DC-preserving, anatomy-adaptive spectral head + physics losses that expose and reduce the physical under-determination of appearance-trained LDCT denoisers.**

Built on the [ldct-benchmark](https://github.com/eeulig/ldct-benchmark) protocol (Eulig et al., *Medical Physics* 2024). The full paper source lives in [`paper/`](paper/).

---

## The problem

Standard image metrics (PSNR/SSIM) reward smoothing and are blind to two clinically relevant failure modes of deep LDCT denoisers:

1. **Noise-texture distortion** — the noise-power spectrum (NPS) of the denoised image no longer matches CT reconstruction physics. Our benchmark-faithful RED-CNN baseline leaves **16% of the noise magnitude unremoved** (26% in the abdomen) with a **~7× low-frequency under-removal hole** and an abdominal spectral-shape correlation of only 0.51 — while being the classically strongest model in the study.
2. **HU-calibration drift** — mean Hounsfield-Unit bias inside tissue classes (air/lung, fat, soft tissue, dense tissue, bone). The same baseline shows **−53 HU chest bone bias**, invisible to PSNR/SSIM.

Worse: with appearance-only training, seeds that are classically indistinguishable (PSNR std 0.04 dB, SSIM std 0.0004) differ by **±13% in spectral fidelity** — the physics of the solution is under-determined by the objective.

## The contributions

### 1. DC-preserving adaptive spectral residual head (architectural) — [`spectral_head.py`](spectral_head.py)

A learnable **radial** gain `G(|f|)` applied in the Fourier domain to the noise the trunk removes:

```
n̂ = x − trunk(x)          # removed noise
out = x − IFFT(G(|f|) · FFT(n̂))
```

Four structural guarantees hold **per image, by construction** (not by training):

- **Exact DC preservation of the residual correction** — the gain applies to non-zero frequencies only, so the head cannot alter the spatial mean of the removed residual. Scope: this guarantees the *head* introduces no global HU shift; absolute HU calibration of the final image is targeted by the HU-bin loss, not by this constraint.
- **Near-DC freezing** (`--freeze-dc-bins 2`) — the first two knots are pinned at G = 1, restricting the head's capacity to alter large-area regional HU means (introduced after diagnosing a −7 HU regional chest bias; −7.0 → −3.5 HU after freezing).
- **Radial-only parameterization** — 32 interpretable scalars; the head introduces no new directional frequency content and restricts the correction to an isotropic modulation of the predicted residual.
- **Bounded adaptivity** (`--adaptive-head`) — a tiny conditioning encoder (~5.3k params, ≤10⁻³ of trunk size) predicts per-image, tanh-bounded log-gain offsets (|Δlog G| ≤ 0.25), zero-initialized to start exactly at the static head.

**Batch-centering penalty** (`--adaptive-center-weight 0.05`): trained naively, the adaptive offsets saturate into a static global shift that the trunk absorbs (image metrics improve, physics regresses — measured, config S4). Centering pulls the batch-mean offset to zero per bin, reserving the adaptive path for genuine between-image differences. With it, the head learns **anatomy-discriminative** gain curves (chest/abdomen separation ≈ 4× the within-anatomy spread) — measured, not asserted (`plot_adaptive_gain.py`).

### 2. Radial NPS matching loss — [`physics_losses.py`](physics_losses.py)

Log-domain L1 distance between the radial NPS of the removed noise (LDCT − pred) and the paired reference noise (LDCT − NDCT). *Reference noise*, deliberately: the paired residual is the standard supervised noise surrogate, but strictly contains all differences between the two reconstructions, not detector noise alone.

### 3. HU-bin bias loss

Mean-HU preservation inside fixed physical tissue intervals (−1024 / −500 / −200 / 200 / 600 / 1900 HU), with optional per-bin weights (`--hu-bin-weights`).

### 4. Physics-fidelity evaluation suite — [`evaluate_physics.py`](evaluate_physics.py)

NPS_LogL1 (**pre-specified primary endpoint**), NPS_Corr, NoiseSTD_Ratio and per-tissue HU bias, complementing the windowed PSNR/SSIM/RMSE/VIF of [`evaluate_image.py`](evaluate_image.py). Reporting rule: always per anatomy — overall aggregates can cancel opposite-signed biases.

> **Claim scope**: better *physical faithfulness* (noise texture + HU calibration) at essentially equal PSNR/SSIM — not a PSNR improvement. The head's **interface** is trunk-agnostic and its structural guarantees hold on any trunk by construction; its **quantitative benefit is measured per trunk** (see cross-trunk results below).

---

## Headline results

**Matched-weight attribution** (the decisive experiment): identical losses (w_NPS = 0.01, w_HU = 0.2), identical 30k budget, identical protocol — the only difference is the head. Mean ± std over seeds 0/1/2, 10 test patients, RED-CNN trunk:

| Metric | Control (losses only) | **+ Spectral head (S4b)** | Δ |
| --- | --- | --- | --- |
| NPS_LogL1 ↓ (primary) | 0.296 ± 0.017 | **0.260 ± 0.027** | **−12%** |
| — chest | 0.125 ± 0.003 | **0.101 ± 0.005** | −19% |
| — abdomen | 0.467 ± 0.036 | **0.419 ± 0.050** | −10% |
| NoiseSTD_Ratio → 1 | 0.913 ± 0.005 | **0.931 ± 0.010** | + |
| Bias_Soft chest (HU) | −5.2 ± 3.8 | **−2.9 ± 0.6** | ÷1.8 |
| Bias_Bone chest (HU) | −36.3 ± 8.0 | **−25.7 ± 2.7** | ÷1.4 |
| PSNR | 30.465 ± 0.056 | **30.509 ± 0.019** | +0.04 dB |

The head wins the primary endpoint on **all 30 patient–seed comparisons** (10 patients × 3 seeds; exact two-sided Wilcoxon signed-rank p = 0.002 within each seed).

**The loss ceiling**: without the head, doubling w_NPS 0.005 → 0.01 makes *everything worse* (NPS 0.267 → 0.287, PSNR −0.17 dB); with the head, the same doubled weight yields the study's best physics. The tested trunk cannot exploit the stronger spectral incentive without degradation — the head provides the missing low-dimensional degrees of freedom.

**Seed determinism**: the physics constraints reduce the seed variance of spectral fidelity 2.5–4× (C0: NPS_LogL1 0.445–0.571 across seeds; S4b: std 0.027), making physical fidelity a property of the method rather than an accident of initialization.

**Cross-trunk replication (ResNet, same protocol, no re-tuning)**: the baseline pathology is trunk-generic; the physics **losses transfer fully** (NPS_LogL1 0.459 → 0.278, −39%); the **head's benefit is trunk-dependent** — it improves spectral shape, chest bone bias and PSNR but concedes the radial-NPS fit (its gain curve settles on broadband attenuation instead of targeted reshaping; head hyperparameters were tuned on RED-CNN and applied unchanged). Reported as a finding, diagnosable only because the gain curve is interpretable.

**Known limitations** (see paper §6): abdomen bone-bias sign flip (+11…+15 HU) systematic to the HU-bin loss; dense-tissue bias (~−18 HU) unresolved; 10 test patients, single dataset, single dose level; 30k budget (vs 92,994 full protocol); radial-only spectral modeling (no anisotropy); design hyperparameters fixed by pilot diagnosis, not swept; no reader study or task-based assessment.

---

## Repository structure

| Path | Purpose |
| --- | --- |
| `paper/` | **Full LaTeX paper** (`main.tex`, `sections/`, `figures/` incl. the TikZ architecture diagram) |
| `config.py` | Central constants: benchmark HU preset, splits, paths |
| `download.py` | Download the Mayo/TCIA `LDCT-and-projection-data` patients (NBIA) |
| `benchmark_data.py` | Benchmark-aligned data pipeline (MONAI, mean/std standardization) |
| `models/` | The five ldct-benchmark trunks: RED-CNN, ResNet, DU-GAN, WGAN-VGG, TransCT |
| `spectral_head.py` | DC-preserving radial spectral head (static + adaptive + centering support) |
| `physics_losses.py` | Radial NPS matching + HU-bin bias losses |
| `train.py` | Matched-budget trainer (paper hyperparameters, `bench_ssim` selection, `--seed`, `--resume`) |
| `train_adversarial.py` | Faithful adversarial trainers for WGAN-VGG and DU-GAN (official benchmark protocol) |
| `adversarial_utils.py` | VGG19 perceptual loss + DU-GAN utilities (verbatim from the benchmark) |
| `run_pilot.py` | Sequential pilot sweep (config screening on 8/4 patients; ranking only, not reportable) |
| `evaluate_image.py` | Test-set image-quality metrics (windowed PSNR/SSIM, RMSE, VIF) |
| `evaluate_physics.py` | Test-set physics-fidelity metrics (NPS, HU bias) |
| `make_figures.py` | Comparison figures: windowed images, diff maps, ROI zooms, NPS curves |
| `plot_spectral_gain.py` | Plot the learned shared `G(\|f\|)` curve of a checkpoint |
| `plot_adaptive_gain.py` | Plot per-image adaptive gain curves grouped by anatomy (the adaptivity measurement) |
| `plot_pareto.py` | Accuracy-vs-texture trade-off plot across NPS weights |
| `results/` | Evaluation outputs |
| `metrics.py`, `utils.py`, `twenty_patient_split.py` | Shared metrics, helpers, small-split IDs |

---

## Setup

```bash
pip install -r requirements.txt
# Only if you use --ssim-weight > 0:
pip install pytorch-msssim
# Only for train_adversarial.py --arch wganvgg (VGG19 perceptual loss):
pip install torchvision
```

Download the data (Mayo Clinic LDCT Grand Challenge patients, TCIA collection `LDCT-and-projection-data`; 90 train/val patients into `dataset/`, 10 test patients into `test/`):

```bash
python download.py
```

Test patients (fixed, anatomy-balanced): chest C121, C135, C170, C249, C280; abdomen L006, L107, L220, L221, L241.

---

## Protocol

Hyperparameters follow the official `configs/redcnn.yaml` of ldct-benchmark (batch 73, patch 128, lr 9.583e-5, constant schedule, pure MSE base loss, best-checkpoint selection by overall unwindowed SSIM = `--select-by bench_ssim`). This study uses a **matched training budget of 30,000 iterations** for every configuration: all internal comparisons are valid; absolute numbers are not directly comparable to the published full-budget (92,994-iteration) benchmark table, and cross-budget comparisons are disallowed by protocol (an under-trained model removes less noise and spuriously "wins" NPS metrics).

Statistics: paired per-patient Wilcoxon signed-rank tests on the primary endpoint (n = 10, two-sided). Seeds are repeats of the same 10 patients, so tests are per seed, never pooled.

## Reproducing the paper configurations

Shared protocol flags for every run:

```bash
PROTO="--data-dir dataset --split 100p \
  --max-iterations 30000 --iterations-before-val 1000 \
  --batch-size 73 --patch-size 128 --val-patch-size 128 \
  --lr 9.583417460320728e-05 --lr-schedule constant \
  --select-by bench_ssim"
```

| Paper ID | Role | Command |
| --- | --- | --- |
| **C0** | Baseline (pure MSE) | `HU_RANGE_PRESET=benchmark python train.py --arch redcnn $PROTO --output-root runs_C0` |
| **C2** | HU-bin loss only | `… --hu-bin-loss 0.2 --output-root runs_C2` |
| **C1h** | Best loss-only recipe | `… --nps-weight 0.005 --hu-bin-loss 0.2 --output-root runs_C1h` |
| **C1h-w0.01** | **Matched-weight control** | `… --nps-weight 0.01 --hu-bin-loss 0.2 --output-root runs_C1h_w001` |
| **S3b** | Static head | `… --use-spectral-head --freeze-dc-bins 2 --nps-weight 0.005 --hu-bin-loss 0.2 --output-root runs_S3b` |
| **S4** | Saturation diagnostic (naive adaptive) | `… --use-spectral-head --adaptive-head --freeze-dc-bins 2 --nps-weight 0.005 --hu-bin-loss 0.2 --output-root runs_S4` |
| **S4b** | **Proposed** | `… --use-spectral-head --adaptive-head --freeze-dc-bins 2 --adaptive-center-weight 0.05 --nps-weight 0.01 --hu-bin-loss 0.2 --output-root runs_S4b` |

- **Multi-seed** (C0, C1h-w0.01, S4b in the paper): add `--seed 1 --output-root runs_S4b_seed1`, etc.
- **Cross-trunk replication**: replace `--arch redcnn` with `--arch resnet` (same three arms, same flags, no re-tuning).
- **Pilot screening** (cheap ranking, never reportable): `python run_pilot.py --data-dir dataset --train-patients 8 --val-patients 4 --iters 8000 --val-every 1000`.
- Other head options exist for constraint studies: `--spectral-bins` (default 32), `--adaptive-max-delta` (default 0.25), `--gain-tv-weight`, `--hu-bin-weights` (see `python train.py --help`).

### Faithful adversarial reproductions (WGAN-VGG, DU-GAN)

```bash
# Official benchmark protocol (hpopt hyperparameters and budgets by default)
HU_RANGE_PRESET=benchmark python train_adversarial.py --arch wganvgg --data-dir dataset --split 100p
HU_RANGE_PRESET=benchmark python train_adversarial.py --arch dugan   --data-dir dataset --split 100p

# Matched-budget study variant (comparable to the 30k runs above)
HU_RANGE_PRESET=benchmark python train_adversarial.py --arch dugan --data-dir dataset --split 100p \
    --max-iterations 30000 --iterations-before-val 1000
```

`train.py` trains only their generators with the study loss (trunk ablations, no adversarial term); `train_adversarial.py` reproduces the benchmark's adversarial training faithfully. The physics components are optional additions and default to OFF. `transct` hard-codes 512×512 inputs: use `--patch-size 512 --val-patch-size 512` with a small batch.

### Evaluation

```bash
# Image quality (windowed PSNR/SSIM, RMSE in HU, VIF)
HU_RANGE_PRESET=benchmark python evaluate_image.py \
    --test-dir test --runs-root runs_S4b --output eval_S4b

# Physics fidelity (NPS match, per-tissue HU bias)
HU_RANGE_PRESET=benchmark python evaluate_physics.py \
    --test-dir test --runs-root runs_S4b --output eval_physics_S4b
```

### Figures

```bash
# Qualitative panels + NPS curves (paper Figs. 2-3)
HU_RANGE_PRESET=benchmark python make_figures.py \
    --test-dir test --runs-roots runs_C0 runs_S4b \
    --labels "RED-CNN" "RED-CNN+Spectral" \
    --patients C121 L006 --output figures

# Learned gain curves (paper Fig. 4: adaptivity measurement)
python plot_adaptive_gain.py --checkpoint runs_S4b/redcnn/best_model.pt
python plot_spectral_gain.py --checkpoint runs_S3b/redcnn/best_model.pt
python plot_pareto.py --csv pareto_points.csv
```

---

## The paper

LaTeX source in [`paper/`](paper/): `main.tex` + `sections/0_abstract … 6_discussion, references` + `figures/` (TikZ architecture diagram `fig1_architecture.tex` and result PNGs). Build:

```bash
cd paper
pdflatex main.tex && pdflatex main.tex
```

Section map: §1 clinical motivation & contributions · §2 related work & gap · §3 method (interface, head, guarantees, centering, objective) · §4 setup (data, matched-budget protocol, configurations, metrics & statistics) · §5 results (baseline audit, ablation, matched-weight attribution, loss ceiling, per-anatomy physics, adaptivity, seeds, cross-trunk) · §6 discussion, lessons & limitations.

---

## Acknowledgements

- Benchmark protocol and trunk architectures: [eeulig/ldct-benchmark](https://github.com/eeulig/ldct-benchmark)
- Data: *Low Dose CT Image and Projection Data (LDCT-and-Projection-data)*, Mayo Clinic / TCIA. Use of the data is subject to the TCIA data usage policy.

## License

MIT — see [LICENSE](LICENSE).
