"""How well each CD method follows the true hole size while the light pulse fluctuates per hole.

Synthetic ACTIS-like images, rendered exactly: round holes (~100 nm, 5 % random size spread) on a square
lattice, imaged through a circular-pupil blur and averaged over 21 nm pixels. Every hole gets its own random
pulse strength (5 % spread) that lights the hole and the absorber around it; detector offset 6 counts,
absorber light 10, hole peak about 59 counts above it, shot + read noise, 8-bit rounding.

For each blur size the table says, per method:
  follows true CD   how many nm the reading moves when the hole grows 1 nm (1.00 = faithful)
  scatter (nm)      typical per-hole error after the reading is scaled to nm by that factor
  pulse +10 % (nm)  how many nm a 10 % stronger pulse moves that scaled reading (0 = immune)

    python scripts/validate_invariant_cd.py
    python scripts/validate_invariant_cd.py --blur-nm 50 75 100 --output artifacts/invariant_cd.md
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


def exact_disks(shape, centers, diameters, weights, psf_fwhm):
    """Disks (global [x, y] centers, px) through an incoherent Airy blur, pixel-integrated.

    Exact up to periodic wrap-around: the blurred spectrum is band-limited, so sampling it on the pixel grid
    loses nothing.
    """
    from scipy.special import j1

    height, width = shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    f = np.hypot(fx, fy)
    fc = 1.029 / psf_fwhm
    nu = np.minimum(f / fc, 1)
    otf = np.where(f < fc, (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu * nu)), 0) * np.sinc(fx) * np.sinc(fy)
    fsafe = np.where(f > 0, f, 1)
    spectrum = np.zeros(f.shape, complex)
    for (x, y), d, a in zip(centers, diameters, weights):
        r = d / 2
        disk = np.where(f > 0, r * j1(2 * np.pi * fsafe * r) / fsafe, np.pi * r * r)
        spectrum += a * disk * np.exp(-2j * np.pi * (fx * x + fy * y))
    return np.fft.irfft2(spectrum * otf, s=(height, width))


def pulsed_lattice(*, psf_fwhm, cd=4.76, pitch=10.0, shape=(110, 150), pulse=0.05, cd_spread=0.05,
                   absorber=10.0, dark=6.0, peak=59.0, noise=True, seed=0, pad=16):
    """Image, true centers, true diameters and pulse strengths (all holes, incl. partly outside the image)."""
    from scipy.spatial import cKDTree
    from subpx._cpu.invariant_cd import DiskTable

    rng = np.random.default_rng(seed)
    height, width = shape[0] + 2 * pad, shape[1] + 2 * pad
    ys, xs = np.mgrid[pitch / 2:height:pitch, pitch / 2:width:pitch]
    centers = np.column_stack((xs.ravel(), ys.ravel())) + rng.uniform(-0.3, 0.3, (xs.size, 2))
    d = cd * (1 + cd_spread * rng.standard_normal(len(centers)))
    a = 1 + pulse * rng.standard_normal(len(centers))
    amplitude = peak / DiskTable(psf_fwhm, 3 * cd, 2).lookup(np.array([cd]), np.zeros(1), False)[0]
    yy, xx = np.mgrid[:height, :width]
    _, cell = cKDTree(centers).query(np.column_stack((xx.ravel(), yy.ravel())))
    image = amplitude * exact_disks((height, width), centers, d, a, psf_fwhm) + dark + absorber * a[cell].reshape(height, width)
    if noise:
        image += rng.standard_normal(image.shape) * np.sqrt(1 + 0.05 * np.maximum(image - dark, 0))
    image = np.clip(np.rint(image), 0, 255)[pad:-pad, pad:-pad]
    return image, centers - pad, d, a


def matched(result, centers, d, a, margin=6):
    """Interior measured cells matched to their true hole: mask, true diameter, pulse strength."""
    from scipy.spatial import cKDTree

    distance, index = cKDTree(centers).query(result.seeds_xy)
    height, width = result.image.shape
    x, y = result.seeds_xy.T
    use = (distance < 1.5) & (x > margin) & (y > margin) & (x < width - margin) & (y < height - margin)
    return use, d[index], a[index]


def against_truth(values, truth, pulse, use):
    """Slope vs true CD, scatter in true-CD units, and fractional CD change per unit pulse change."""
    ok = use & np.isfinite(values)
    slope, intercept = np.polyfit(truth[ok], values[ok], 1)
    residual = (values[ok] - slope * truth[ok] - intercept) / slope
    sensitivity = np.polyfit(pulse[ok] - 1, residual / np.median(truth[ok]), 1)[0]
    return slope, residual.std(), sensitivity


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--blur-nm", type=float, nargs="+", default=[50, 75, 100], help="blur FWHM in nm")
    parser.add_argument("--cd-nm", type=float, default=100.0)
    parser.add_argument("--pixel-nm", type=float, default=21.0)
    parser.add_argument("--seeds", type=int, default=3, help="images per blur")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/invariant_cd.md")
    args = parser.parse_args(argv)

    methods = ("gaussian", "halfmax", "disk", "flux_ratio", "projection")
    lines = [f"Hole {args.cd_nm:g} nm, pixel {args.pixel_nm:g} nm, 5 % pulse spread; disk/flux_ratio calibrated with "
             f"--nominal-cd {args.cd_nm:g} (blur found from the data). Mean over {args.seeds} images.", "",
             "| blur (nm) | method | valid | follows true CD | scatter (nm) | pulse +10 % (nm) |",
             "|---|---|---|---|---|---|"]
    for blur in args.blur_nm:
        stats = {m: [] for m in methods}
        found_blur = []
        start = time.perf_counter()
        for seed in range(args.seeds):
            image, centers, d, a = pulsed_lattice(psf_fwhm=blur / args.pixel_nm, cd=args.cd_nm / args.pixel_nm, seed=seed)
            result = measure_cds(image, methods=methods, area_max=200, nominal_cd=args.cd_nm, pixel_size=args.pixel_nm,
                                 unit="nm", dark_level=6.0)
            found_blur.append(result.meta["psf_fwhm_px"] * args.pixel_nm)
            use, truth, pulse = matched(result, centers, d, a)
            for m in methods:
                measured = result.measurements[m]
                slope, scatter, sensitivity = against_truth(measured.metrics["cd_equivalent_px"], truth, pulse,
                                                            use & measured.valid)
                stats[m].append((measured.valid[use].mean(), slope, scatter * args.pixel_nm,
                                 0.1 * sensitivity * args.cd_nm))
        for m in methods:
            valid, slope, scatter, shift = np.mean(stats[m], axis=0)
            lines.append(f"| {blur:g} | {m} | {valid:.0%} | {slope:.2f} | {scatter:.1f} | {shift:+.1f} |")
        print(f"blur {blur:g} nm: found {np.mean(found_blur):.1f} nm, {time.perf_counter() - start:.1f} s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
