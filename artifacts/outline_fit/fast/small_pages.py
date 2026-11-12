"""Scratch: the one-pass fit where the report pipeline will use it: seeds handed in by measure_cds (not its own
finder), small images (the library's test page of 120 x 170 px and a few larger ones), few holes.

    python small_pages.py [existing=1]

Prints, per page: holes, the page CD against the mean true CD, the page fit's leftover, seconds; with existing=1
also the library's outline_fit on the same page through measure_cds (CPU for the smallest page, GPU otherwise).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from gpufit import ROOT, sync
from pipeline import fit_page

sys.path.insert(0, str(ROOT))
from subpx import measure_cds  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402

PIXEL_NM = 21.0


def truth_for(page, centres, mask):
    from scipy.spatial import cKDTree

    d, j = cKDTree(page.centers_xy).query(centres)
    ok = mask & (d < 1.5)
    return float((np.sqrt(page.size_px[j[ok], 0] * page.size_px[j[ok], 1]) * PIXEL_NM).mean()), int(ok.sum())


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    cases = [("library test page", (120, 170), dict(size_nm=(100, 100), pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=5)),
             ("300 x 400", (300, 400), dict(size_nm=(100, 100), pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=6)),
             ("600 x 800", (600, 800), dict(size_nm=(100, 100), pitch_nm=(200, 200), cd_sigma_nm=2.5, seed=7)),
             ("60 x 90 (few holes)", (60, 90), dict(size_nm=(100, 100), pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=8))]
    fit_page(render_hole_array((300, 400), hole="circle", size_nm=(100, 100), pitch_nm=(210, 210), layout="square",
                               seed=1, backend="cpu").image)  # warm-up
    for name, shape, kw in cases:
        page = render_hole_array(shape, hole="circle", layout="square", backend="cpu", **kw)
        base = measure_cds(page.image, methods=("half_area",), pixel_size=PIXEL_NM, unit="nm", backend="cpu", area_max=200)
        seeds = base.seeds_xy
        print(f"{name}: {shape[1]} x {shape[0]} px, {len(page.centers_xy)} true holes, {len(seeds)} seeds from measure_cds", flush=True)
        for label, s in (("report's seeds", seeds), ("own seeds", None)):
            for family in ("gauss2", "lens"):
                try:
                    sync()
                    t0 = time.perf_counter()
                    pg, out, res = fit_page(page.image, family=family, level=1, seeds=s)
                    sync()
                    dt = time.perf_counter() - t0
                    centres = cp.asnumpy(pg.c)
                    tm, n = truth_for(page, centres, cp.asnumpy(pg.use))
                    print(f"   one pass, {label:14s} {family:6s}: page CD {out['cd']:.3f} nm, error {out['cd'] - tm:+.3f} ({n} holes stacked of "
                          f"{pg.n}), blur {out['blur'][0]:.1f}/{out['blur'][1]:.1f}, chi2 per bin {out['chi2'] / max(out['bins'], 1):.2f}, "
                          f"first fit {out['first']['cd'] - tm:+.3f}, window +-{pg.R}, box {pg.L}, {dt * 1e3:.0f} ms", flush=True)
                except Exception as error:  # what a small page does to the prototype is the finding
                    print(f"   one pass, {label:14s} {family:6s}: FAILED {type(error).__name__}: {str(error)[:160]}", flush=True)
        if args.get("existing", "0") == "1" and len(seeds) >= 10:
            backend = "cpu" if shape[0] * shape[1] < 50_000 else "gpu"
            t0 = time.perf_counter()
            both = measure_cds(page.image, methods=("half_area", "outline_fit"), pixel_size=PIXEL_NM, unit="nm",
                               backend=backend, area_max=200)
            dt = time.perf_counter() - t0
            m = both.measurements["outline_fit"]
            meta = both.meta["outline_fit"]
            tm, n = truth_for(page, m.centers_xy, m.valid)
            print(f"   library outline_fit ({backend}): outline CD {meta['outline_cd_px'] * PIXEL_NM:.3f} nm, mean of the valid holes "
                  f"{np.nanmean(m.metrics['cd_equivalent_px'][m.valid]) * PIXEL_NM:.3f}, error {meta['outline_cd_px'] * PIXEL_NM - tm:+.3f} "
                  f"({n} holes), rounds {meta['rounds_run']}, converged {meta['converged']}, {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
