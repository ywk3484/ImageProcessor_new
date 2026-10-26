"""A synthetic contact-hole page on which GPU deblending never settles, as on the user's real pages.

    python scripts/drift_page.py              # 1378 x 2048, 39,042 holes: about 26 s here
    python scripts/drift_page.py --small      # 512 x 768, 5,400 holes, the same behaviour: about 5 s

On the user's median page the deblending loop runs all 30 rounds (10 + 20, 2 passes) without settling, with no
background tile switch, no fit status flip and no cycle, its p99 step still 9e-4 at round 30 (cd_diagnose's y token
0/0/0/9), and 1-3 more rounds move a page mean by 0.17 nm and a hole's p99 by 0.64 nm (k token 0.17/0.64). The
default page here does the same. Its holes fill the frame to the edges, so no background is visible: no tile gets
enough free pixels and the background map is "inferred" from all residual pixels. Wider fits then lower the map and
a lower map widens the fits: over the 30 rounds the map's mean falls from 37.4 to 32.8 grey levels (true level 40)
while the median fitted sigma rises from 1.92 to 1.98 px and the amplitude from 158 to 162, by a step that shrinks
only 4 % per round. With ``background_level=40`` the same page settles in 5 rounds.

What each ingredient did (32 configurations, the rows below carry the findings):

- No visible background is decisive. A 15 px empty margin (c01b, c03b) or 7.75 px strips at two edges (f09: 41 of
  2646 tiles measured, 1.5 %) settle in 7 rounds at a step decay of 0.40-0.45; every page that measured no tile
  decayed at 0.86-0.97. Tiles measured inside the lattice settle too (pitch 10-11: 93-100 % measured).
- Density: square pitch 9 settles at round 21 (decay 0.94), 8.5 never (0.96); pitch 10 measures most tiles. A
  rectangular 8.5 x 11 lattice puts the middle of its row gap at the free-pixel distance (1.25 FWHM = 5.5 px): with
  cut edges it stays inferred (f07, 24 rounds); inside the frame its tiles switch 2575 times and end 97.5 %
  measured (f08b, settled in 9 rounds).
- Hole shape: disk holes drift longer than Gaussian holes of the same fitted FWHM (pitch 9: 20 vs 15 rounds, decay
  0.91 vs 0.86; pitch 8.5: never vs 25 rounds, both 0.95). At 5.5 px the fitted Gaussian's tail is twice the
  disk's (1.3 % against 0.66 % of the peak).
- Noise is not monotone: 4 / 7.5 / 10 / 15 grey levels settle at 29 / 21 / 14 / never (pitch 9), as if it acts
  through the start (each cell border's 10th percentile: raised by the neighbours' tails, lowered by noise). At 15,
  foot_area becomes unstable (18,413 of 34,502 holes valid, hole p99 move 20 nm) and the cell stack 20x21.
- 5 % hole-to-hole amplitude: no effect (19 vs 20 rounds); a +-5 grey-level background ripple: a little longer (24).
- Cut edges (margin None): the same drift plus 0-15 status flips in the last 12 rounds, all edge holes whose fitted
  centre sits on the frame edge (checked on f06). The default margin 2 keeps every centre inside: 0 flips.

    id   page                                            size  rounds   set   sw fl cy step decay measured  move
    user the user's median page (real, ~29,600 holes)    full  10+20    n      0  0  0    9  ~0.9 ?         0.17
    c01b Gaussian 4.2, staggered 9.5, margin 15          small 7        y      0  0  0    3  0.41 66/294    0
    c03b disk, square 9, margin 15                       small 7        y      0  0  0    3  0.40 70/330    0
    c08  Gaussian, square 9, cut edges                   small 10+5     y      0  0  0   10  0.86 0/330     -
    c05  disk, square 9, cut edges                       small 10+10    y      0  2  0   22  0.91 0/330     -
    c10  disk, square 10, cut edges                      small 10+2     y      0  0  0    6     - 260/280   -
    c11  disk, square 11, cut edges                      small 8        y    114  9  0    3  0.43 216/216   -
    f01  disk, square 9, cut edges                       full  10+11    y      0  0  0    6  0.94 0/2360    0
    f05  f01 with noise 4                                full  10+19    y      0  0  0    6  0.94 0/2223    0
    f04  f01 with noise 10                               full  10+4     y      0  0  0    6  0.90 0/2360    0
    f02  f01 with noise 15                               full  10+20    n      0  4  0   10  0.95 0/2360    0.17
    f07  disk, rectangular 8.5 x 11, cut edges           full  10+14    y      0  2  0    6  0.91 0/2646    0
    f08b disk, rectangular 8.5 x 11, margin 2            full  9        y   2575  0  0    4  0.50 2579/2646 0
    f09  disk, square 8.5, margin 4.25 (7.75 px strips)  full  7        y     12  0  0    6  0.45 41/2646   0
    f06  disk, square 8.5, cut edges                     full  10+20    n      0  7  0    8  0.96 0/2646    0.16
    f10  f06 with seed 1                                 full  10+10+20 n      0 15  0    8  0.96 0/2646    0.16
    f11  f06 + 5 % amplitude + 5 grey-level ripple       full  10+10+20 n      0 11  0   10  0.97 0/2646    0.16
    c19  Gaussian, square 8.5, cut edges                 small 10+15    y      0  0  0    6  0.95 0/384     0
    f12  disk, square 8.5, margin 2: the default         full  10+20    n      0  0  0    7  0.96 0/2646    0.16
    s12  the default, small                              small 10+20    n      0  0  0    7  0.95 0/384     0.14

disk: a 4.8 px disk blurred by a 3 px FWHM PSF (fitted Gaussian FWHM 4.4); Gaussian: FWHM 4.4 unless given. All
pages are 8-bit, peak 150 on 40, noise 7.5, jitter 0.3, unless given. cut edges: ``margin=None``; margin m: the
lattice is centred, its outermost centres at least m px inside; c01b is about scripts/benchmark_cd.lattice's page.
small: 512 x 768, full: 1378 x 2048. step: the last p99 step over all holes, as relaxed by the loop (cd_diagnose's
y token, x1e4); the stop rule needs the raw step over the ok holes (about this / 0.6) below 1e-3. decay: that
step's ratio per round over the last 8 rounds of the last pass (fewer if shorter). move: the largest page-mean
change, nm, for 1-3 more rounds (0: settled before round 30; -: not measured). The default page differs from the
user's tokens in the hole p99 move (1.0 nm against 0.64), its 39,042 holes against ~29,600 and a step decay of 0.96
where ~0.9 was estimated. Whether the user's pages are "inferred" too (0 measured tiles) is not known yet: every
page here with a few tiles of visible background settled.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _profile(r, diameter: float, sigma: float):
    """Peak-normalized radial profile of a flat disk of ``diameter`` blurred by a Gaussian of ``sigma`` (diameter 0:
    the Gaussian): the chance that a point blurred from distance ``r`` lands inside the disk."""
    from scipy.stats import ncx2

    if diameter <= 0:
        return np.exp(-0.5 * (r / sigma) ** 2)
    k = (diameter / 2 / sigma) ** 2
    return ncx2.cdf(k, 2, np.maximum(r / sigma, 1e-6) ** 2) / ncx2.cdf(k, 2, 1e-12)


def drifting_page(
    shape=(1378, 2048), seed=0, *, pitch=8.5, pitch_y=None, lattice="square", diameter=4.8, blur=3.0,
    amplitude=150.0, amplitude_cv=0.0, background=40.0, ripple=0.0, noise=7.5, jitter=0.3, margin=2.0,
) -> np.ndarray:
    """Bright contact holes on a dark 8-bit (H, W) page, deterministic from ``seed``.

    Each hole is a flat disk of ``diameter`` px blurred by a Gaussian PSF of FWHM ``blur`` px (pixel area included as
    its variance, 1/12 px^2), ``amplitude`` grey levels above ``background`` at its peak, times 1 + ``amplitude_cv``
    * N(0, 1); ``diameter`` 0 gives Gaussian holes of FWHM ``blur``. Centres form a square lattice of ``pitch`` px
    (rows ``pitch_y`` apart; ``lattice="staggered"`` shifts every other row by half a pitch), each moved by up to
    ``jitter`` px. ``margin``: the lattice is centred with its outermost centres at least this far inside the frame
    (2: holes up to the edges, no strip of background; 15: like scripts/benchmark_cd.lattice); None: it runs on
    past the edges, which cut the holes there. ``ripple`` adds four random plane waves (wavelengths 150-450 px) to
    the background, at most that many grey levels; ``noise`` is Gaussian, in grey levels, before rounding.
    """
    rng = np.random.default_rng(seed)
    height, width = shape
    py = pitch if pitch_y is None else pitch_y
    if margin is None:
        xs, ys = np.arange(-pitch, width + pitch, pitch), np.arange(-py, height + py, py)
    else:
        nx, ny = int((width - 1 - 2 * margin) // pitch) + 1, int((height - 1 - 2 * margin) // py) + 1
        xs = (width - 1 - (nx - 1) * pitch) / 2 + pitch * np.arange(nx)
        ys = (height - 1 - (ny - 1) * py) / 2 + py * np.arange(ny)
    xx, yy = np.meshgrid(xs, ys)
    if lattice == "staggered":
        xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    if margin is not None:
        centers = centers[centers[:, 0] <= width - 1 - margin]
    centers = centers + rng.uniform(-jitter, jitter, centers.shape)
    amps = amplitude * np.clip(1 + amplitude_cv * rng.standard_normal(len(centers)), 0.2, None)
    sigma = np.sqrt((blur * FWHM_TO_SIGMA) ** 2 + 1 / 12)
    radius = int(np.ceil(diameter / 2 + 5 * sigma)) + 1
    table_r = np.linspace(0, radius * 1.5 + 2, 4001)  # beyond the farthest pixel of a hole's square
    table = _profile(table_r, diameter, sigma)
    ox, oy = (a.ravel() for a in np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)))
    image = np.zeros(height * width)
    for chunk in np.array_split(np.arange(len(centers)), max(1, len(centers) // 5000)):
        c = centers[chunk]
        ix = np.floor(c[:, 0]).astype(int)[:, None] + ox[None]
        iy = np.floor(c[:, 1]).astype(int)[:, None] + oy[None]
        values = amps[chunk][:, None] * np.interp(np.hypot(ix - c[:, :1], iy - c[:, 1:]), table_r, table)
        ok = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        image += np.bincount((iy * width + ix)[ok], values[ok], minlength=image.size)
    image = image.reshape(shape) + background
    if ripple:
        gy, gx = np.mgrid[0:height, 0:width].astype(float)
        field = np.zeros(shape)
        for _ in range(4):
            angle, phase = rng.uniform(0, 2 * np.pi, 2)
            wave = 150 * rng.uniform(1, 3)
            field += np.cos(2 * np.pi * (gx * np.cos(angle) + gy * np.sin(angle)) / wave + phase)
        image += ripple * field / np.abs(field).max()
    image += rng.normal(0, noise, shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--small", action="store_true", help="512 x 768 instead of the full 1378 x 2048 page")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from scripts.cd_diagnose import SYNTHETIC_SETTINGS, Probe, compare, num
    from subpx import measure_cds
    from subpx._gpu.deblend import deblend

    relaxation = inspect.signature(deblend).parameters["relaxation"].default  # the probed steps are relaxed
    shape = (512, 768) if args.small else (1378, 2048)
    start = time.perf_counter()
    image = drifting_page(shape, args.seed)
    rendered = time.perf_counter() - start
    measure_cds(image[:200, :300], backend="gpu", **SYNTHETIC_SETTINGS)  # loads the GPU kernels outside the timing
    start = time.perf_counter()
    with Probe() as probe:
        result = measure_cds(image, backend="gpu", **SYNTHETIC_SETTINGS)
    probed = time.perf_counter() - start
    s, meta = probe.summary(), result.meta
    steps = s["steps"]
    k = min(8, len(steps) - 1)
    decay = (steps[-1] / steps[-1 - k]) ** (1 / k) if k > 0 and steps[-1 - k] > 0 else float("nan")
    moves, seconds = [], []
    for extra in (1, 2, 3):
        start = time.perf_counter()
        more = measure_cds(image, backend="gpu", **{**SYNTHETIC_SETTINGS, "deblend_iterations": 30 + extra})
        seconds.append(time.perf_counter() - start)
        moves.append(compare(result, more))
    print(f"page            {shape[0]} x {shape[1]}, seed {args.seed}: {len(result.seeds_xy)} holes, rendered in "
          f"{rendered:.1f} s, measured in {probed:.1f} s (probed)")
    print(f"rounds          {s['rounds']} (passes {' + '.join(str(p['used']) for p in s['passes'])})")
    print(f"settled         {'yes' if meta['deblend_converged'] else 'no'}")
    print(f"switches        {s['switches']} (background tiles measured <-> interpolated, last 12 rounds)")
    print(f"flips           {s['status_flips']} (fit statuses changed, last 12 rounds)")
    print(f"cycle           {s['cycle']} (0: none)")
    print(f"last step       {steps[-1]:.1e} (p99 over holes, relaxed: y token {1e4 * steps[-1]:.0f}); raw "
          f"{steps[-1] / relaxation:.1e} (the loop stops below 1e-3)")
    print(f"decay           {decay:.3f} per round (last {k} rounds)")
    print(f"measured tiles  {meta['background_measured_tiles']} / {meta['background_tiles']}, background "
          f"{meta['background_source']}")
    print("+1..+3 rounds   page mean " + " / ".join(num(m["page_mean"], 3) for m in moves) + " nm, hole p99 "
          + " / ".join(num(m["hole_p99"], 2) for m in moves) + f" nm ({np.median(seconds):.1f} s per measure)")


if __name__ == "__main__":
    main()
