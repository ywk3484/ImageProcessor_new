"""Sweep CD accuracy over hole density and noise on synthetic Gaussian lattices.

Writes a Markdown table of valid fraction and CD bias +- spread per method for
the previous per-cell border background and the deblended model background.

    python scripts/validate_cd_accuracy.py
    python scripts/validate_cd_accuracy.py --ratios 2.67 2.0 1.6 1.4 --snr 0 20 10 --output artifacts/cd_accuracy.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import measure_cds


def lattice(ratio: float, *, fwhm: float, snr: float, shape=(200, 600), seed: int = 0):
    """Staggered Gaussian holes (amplitude 50 on background 20) with a 15 px hole-free margin."""
    rng = np.random.default_rng(seed)
    pitch = ratio * fwhm
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    xs = np.arange(15, shape[1] - 15, pitch)
    ys = np.arange(15, shape[0] - 15, pitch)
    xx, yy = np.meshgrid(xs, ys)
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    y, x = np.indices(shape, dtype=float)
    image = np.full(shape, 20.0)
    for cx, cy in centers:
        image += 50 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    if snr > 0:
        image += rng.normal(0, 50 / snr, shape)
    return image, centers


def main(argv=None) -> None:
    from scipy.spatial import cKDTree

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", type=float, nargs="+", default=[2.67, 2.0, 1.6, 1.4], help="pitch / FWHM")
    parser.add_argument("--snr", type=float, nargs="+", default=[0, 20, 10], help="amplitude / noise std; 0 = noise-free")
    parser.add_argument("--fwhm", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cd_accuracy.md")
    args = parser.parse_args(argv)

    configs = [("border", "threshold"), ("model", "threshold"), ("model", "peaks")]
    lines = ["| pitch/FWHM | SNR | background | seeds | holes | method | valid | CD bias (px) | CD std (px) | time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ratio in args.ratios:
        for snr in args.snr:
            image, centers = lattice(ratio, fwhm=args.fwhm, snr=snr)
            for background, seeds in configs:
                start = time.perf_counter()
                result = measure_cds(image, background=background, seeds=seeds, threshold="triangle", area_max=400)
                seconds = time.perf_counter() - start
                distance = cKDTree(centers).query(result.seeds_xy)[0] if len(result.seeds_xy) else np.zeros(0)
                for method, measured in result.measurements.items():
                    # Gradient edges sit at sigma, not at half maximum.
                    expected = args.fwhm / np.sqrt(2 * np.log(2)) if method == "gradient" else args.fwhm
                    use = measured.valid & (distance < 1)
                    error = measured.metrics["cd_equivalent_px"][use] - expected
                    bias = f"{error.mean():+.3f}" if use.any() else "n/a"
                    spread = f"{error.std():.3f}" if use.any() else "n/a"
                    lines.append(f"| {ratio:g} | {snr:g} | {background} | {seeds} | {len(centers)} | {method} | "
                                 f"{use.sum()}/{len(centers)} | {bias} | {spread} | {seconds:.1f} |")
                print(f"pitch/FWHM {ratio:g}, SNR {snr:g}, {background}/{seeds}: {seconds:.1f} s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
