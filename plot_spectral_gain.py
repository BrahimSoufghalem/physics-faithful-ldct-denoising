"""Plot the learned radial gain curve G(|f|) of a spectral-head checkpoint.

Usage
-----
    python plot_spectral_gain.py \\
        --checkpoint runs_paper_S1_spectral/redcnn/best_model.pt \\
        --output spectral_gain_S1

Produces ``<output>.png`` and ``<output>.csv``.

Reading the curve
-----------------
* ``G = 1``  : the head leaves that band untouched (identical to the trunk).
* ``G < 1``  : the head RETURNS part of the energy the trunk wanted to
               remove at that band (protects detail / noise texture --
               expected at mid-to-high frequencies if the trunk
               over-smooths).
* ``G > 1``  : the head removes more than the trunk at that band.
"""

import argparse
import csv

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a best_model.pt trained with --use-spectral-head")
    p.add_argument("--output", default="spectral_gain",
                   help="Output basename (writes <output>.png and <output>.csv)")
    args = p.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("model_state_dict", state)
    meta = state.get("meta", {}) if isinstance(state, dict) else {}

    key = next((k for k in weights if k.endswith("head.log_gain")), None)
    if key is None:
        raise RuntimeError(
            "No spectral head in this checkpoint. "
            "Train with --use-spectral-head first."
        )
    gain = torch.exp(weights[key].detach()).numpy()
    n_bins = int(gain.shape[0])
    # Knots span [0, sqrt(2)] in units of the axis Nyquist frequency.
    r = np.linspace(0.0, np.sqrt(2.0), n_bins)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(r, gain, marker="o", ms=3.5, lw=1.5, color="#1f6feb")
    plt.axhline(1.0, color="gray", lw=0.9, ls="--",
                label="G = 1 (identity = trunk alone)")
    plt.axvline(1.0, color="gray", lw=0.9, ls=":", label="axis Nyquist")
    plt.xlabel("Radial spatial frequency |f|  (fraction of Nyquist)")
    plt.ylabel("Learned gain G(|f|)")
    arch = str(meta.get("architecture", "model")).upper()
    it = state.get("iteration", "?")
    plt.title(f"Spectral residual head gain \u2014 {arch} (iter {it})")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output + ".png", dpi=200)

    with open(args.output + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["radial_freq_nyquist", "gain"])
        for ri, gi in zip(r, gain):
            w.writerow([f"{ri:.6f}", f"{gi:.8f}"])

    dev = float(np.abs(gain - 1.0).max())
    print(f"Saved {args.output}.png and {args.output}.csv")
    print(f"n_bins={n_bins} | max |G-1| = {dev:.4f}"
          + ("  (WARNING: G stayed ~1 everywhere -> the head learned "
             "nothing; consider pairing with --nps-weight)" if dev < 0.01 else ""))


if __name__ == "__main__":
    main()
