"""Does preprocessing keep real CD differences? A synthetic page with known truth, measured raw and preprocessed.

The page: pitch 8.2 px, holes 4.8 px wide blurred by a Gaussian PSF (sigma 1.4 px), 4 % hole-to-hole size spread,
holes in the right half 5 % larger, 10 % brightness flicker, noise 4 % of the peak, rounded to 8 bits. Each
version is measured with ``measure_cds`` and compared with the CDs measured on the noise-free page:

- step kept: the measured right/left CD step divided by the noise-free page's step (1 = the +5 % step survives);
- slope: per-hole CD against the noise-free CD (1 = hole-to-hole differences survive at full size);
- scatter: per-hole error around that line, px.

A smaller scatter is better precision only when step kept and slope stay near 1.

    python scripts/preprocess_check.py [--layout square|staggered|mixed] [--methods half_area projection]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subpx.cd import measure_cds  # noqa: E402
from subpx.preprocess import denoise_mean_hole, denoise_wiener, gaussian_kernel, smooth_gaussian  # noqa: E402


def render(layout="square", h=360, w=520, pitch=8.2, diameter=4.8, psf=1.4, seed=1):
    """(noise-free page, noisy 8-bit page, hole centres [x, y]) on a 4x supersampled canvas."""
    rng = np.random.default_rng(seed)
    up = 4
    canvas = np.zeros((h * up, w * up))
    centers = []
    for row, cy in enumerate(np.arange(pitch * 0.7, h - 2, pitch)):
        shift = pitch / 2 * (row % 2) if layout != "square" else 0.0
        for cx in np.arange(pitch * 0.6 + shift, w - 2, pitch):
            size = diameter * (1 + 0.04 * rng.normal()) * (1.05 if cx > w / 2 else 1.0)
            gain = 1 + 0.10 * rng.normal()
            x, y = cx + rng.normal(0, 0.05), cy + rng.normal(0, 0.05)
            kind = ("disc", "square", "ellipse")[len(centers) % 3] if layout == "mixed" else "disc"
            x0, y0 = max(int((x - size) * up), 0), max(int((y - size) * up), 0)
            sx = (np.arange(x0, min(int((x + size) * up) + 2, w * up)) + 0.5) / up - 0.5
            sy = (np.arange(y0, min(int((y + size) * up) + 2, h * up)) + 0.5) / up - 0.5
            dx, dy = sx[None, :] - x, sy[:, None] - y
            if kind == "disc":
                inside = dx ** 2 + dy ** 2 <= (size / 2) ** 2
            elif kind == "square":
                a = np.deg2rad(20)
                u, v = dx * np.cos(a) + dy * np.sin(a), -dx * np.sin(a) + dy * np.cos(a)
                inside = (np.abs(u) <= size * 0.443) & (np.abs(v) <= size * 0.443)
            else:
                inside = (dx / (size * 0.6)) ** 2 + (dy / (size * 0.42)) ** 2 <= 1
            canvas[y0:y0 + inside.shape[0], x0:x0 + inside.shape[1]] += gain * inside
            centers.append((x, y))
    kernel = gaussian_kernel(psf * up)
    for axis in (0, 1):
        canvas = np.apply_along_axis(lambda line: np.convolve(np.pad(line, len(kernel) // 2, mode="symmetric"),
                                                              kernel, mode="valid"), axis, canvas)
    truth = canvas.reshape(h, up, w, up).mean(axis=(1, 3))
    truth = 12.0 + truth / truth.max() * 180.0
    noisy = np.clip(np.rint(truth + rng.normal(0, 0.04 * 180.0, truth.shape)), 0, 255).astype(np.uint8)
    return truth, noisy, np.array(centers)


def cds(image, centers, methods):
    """Per-hole equivalent CD per method, matched to ``centers`` (NaN where not found or failed)."""
    result = measure_cds(image, methods=methods, backend="cpu", area_min=3, area_max=400)
    distance = np.hypot(*(result.seeds_xy[:, None, :] - centers[None, :, :]).transpose(2, 0, 1))
    nearest, close = distance.argmin(axis=1), distance.min(axis=1) < 2
    out = {}
    for name, measured in result.measurements.items():
        values = np.full(len(centers), np.nan)
        ok = close & measured.valid
        values[nearest[ok]] = measured.metrics["cd_equivalent_px"][ok]
        out[name] = values
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", choices=("square", "staggered", "mixed"), default="square")
    parser.add_argument("--methods", nargs="+", default=["half_area", "projection"])
    args = parser.parse_args(argv)
    truth, noisy, centers = render(args.layout)
    right = centers[:, 0] > truth.shape[1] / 2
    inner = ((centers > 10) & (centers < np.array(truth.shape[::-1]) - 10)).all(axis=1)
    reference = cds(truth, centers, args.methods)
    versions = {
        "raw": noisy,
        "Gaussian 0.6 px": smooth_gaussian(noisy, 0.6).image,
        "Wiener, auto": denoise_wiener(noisy).image,
    }
    auto = denoise_mean_hole(noisy, centers)
    versions[f"mean hole, auto K = {auto.meta['components']}"] = auto.image
    for k in (16, 32):
        versions[f"mean hole, K = {k}"] = denoise_mean_hole(noisy, centers, components=k).image
    print(f"layout {args.layout}, {len(centers)} holes, planted step +5 %")
    print(f"{'version':24s} {'method':12s} {'step kept':>9s} {'slope':>6s} {'scatter px':>10s}")
    for label, image in versions.items():
        measured = cds(image, centers, args.methods)
        for name in args.methods:
            t, g = reference[name], measured[name]
            ok = np.isfinite(t) & np.isfinite(g) & inner
            step = (g[ok & right].mean() - g[ok & ~right].mean()) / (t[ok & right].mean() - t[ok & ~right].mean())
            slope, offset = np.polyfit(t[ok], g[ok], 1)
            scatter = np.std(g[ok] - (slope * t[ok] + offset))
            print(f"{label:24s} {name:12s} {step:9.2f} {slope:6.2f} {scatter:10.3f}")


if __name__ == "__main__":
    main()
