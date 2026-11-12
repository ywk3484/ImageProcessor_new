"""Scratch: the one-pass fit where the report pipeline will use it: seeds handed in by measure_cds (not its own
finder), small images (the library's test page of 120 x 170 px and a few larger ones), few holes.

    python small_pages.py [existing=1]

Prints, per page: holes, the page CD against the mean true CD with the fit's own standard error of it (from the
final fit's curvature), the page fit's leftover, the centre error of the stacked holes after the centring step (rms
over both axes together, in nm) beside that of the seeds, seconds; with
existing=1 also the library's outline_fit on the same page through measure_cds (CPU for the smallest page, GPU
otherwise).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from gpufit import ROOT, find_seeds, sync
from pipeline import describe, fit_page, fitter_for

sys.path.insert(0, str(ROOT))
from subpx import measure_cds  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402

PIXEL_NM = 21.0


def truth_for(page, centres, mask):
    from scipy.spatial import cKDTree

    d, j = cKDTree(page.centers_xy).query(centres)
    ok = mask & (d < 1.5)
    return float((np.sqrt(page.size_px[j[ok], 0] * page.size_px[j[ok], 1]) * PIXEL_NM).mean()), int(ok.sum())


def centre_rms_nm(page, centres, mask):
    """Distance of the holes' centres from the true ones, rms over the masked holes (both axes together), in nm."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(page.centers_xy).query(centres)
    ok = mask & (d < 1.5)
    return float(np.sqrt((d[ok] ** 2).mean()) * PIXEL_NM) if ok.any() else float("nan")


def page_cd_se(pg, family, theta):
    """Standard error of the page CD (nm) from the final fit's Gauss-Newton matrix with the priors (cost in
    chi-square units; the stack's noise estimate is about 30 % high, so this is a little large)."""
    sf = fitter_for(pg, family)
    free, _, tm = sf._stage("full")
    data = sf.prepare(pg.pair_map(sf)[None], *[v[None] for v in pg.stack_data(sf)])
    A = sf.evaluate(theta[None], data, tm)[4][0]
    rows = sf.default_prior()
    if rows:
        pj = np.stack([r[0] for r in rows]) @ tm
        A = A + (pj.T * np.array([1.0 / r[2] ** 2 for r in rows])[None, :]) @ pj
    grad = np.array([(describe(pg, theta + 1e-4 * tm[:, i], PIXEL_NM)["cd"] - describe(pg, theta - 1e-4 * tm[:, i], PIXEL_NM)["cd"])
                     / 2e-4 for i in range(len(free))])
    return float(np.sqrt(max(grad @ np.linalg.pinv(A) @ grad, 0.0)))


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
                    used = cp.asnumpy(pg.use)
                    tm, n = truth_for(page, centres, used)
                    start = (cp.asnumpy(find_seeds(cp.asarray(page.image).astype(cp.float32))) if s is None
                             else np.asarray(s, float))  # the method's own seeds are found the same way again
                    moved = (f"centres {centre_rms_nm(page, centres, used):.2f} nm rms from the truth (seeds "
                             f"{centre_rms_nm(page, start, used):.2f})" if len(start) == len(centres) else "centres: seeds not kept")
                    se = page_cd_se(pg, family, np.asarray(out["theta"]))
                    print(f"   one pass, {label:14s} {family:6s}: page CD {out['cd']:.3f} nm, error {out['cd'] - tm:+.3f} +- {se:.3f} ({n} holes stacked of "
                          f"{pg.n}), blur {out['blur'][0]:.1f}/{out['blur'][1]:.1f}, chi2 per bin {out['chi2'] / max(out['bins'], 1):.2f}, "
                          f"first fit {out['first']['cd'] - tm:+.3f}, {moved}, window +-{pg.R}, box {pg.L}, {dt * 1e3:.0f} ms",
                          flush=True)
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
            th = np.asarray(meta["theta"], float)  # gauss2: ..., log s1 x, y, log s2 x, y, logit w2
            print(f"   library outline_fit ({backend}): outline CD {meta['outline_cd_px'] * PIXEL_NM:.3f} nm, mean of the valid holes "
                  f"{np.nanmean(m.metrics['cd_equivalent_px'][m.valid]) * PIXEL_NM:.3f}, error {meta['outline_cd_px'] * PIXEL_NM - tm:+.3f} "
                  f"({n} holes), rounds {meta['rounds_run']}, converged {meta['converged']}, its blur: s1 {np.exp(th[-5]):.2f}/"
                  f"{np.exp(th[-4]):.2f}, s2 {np.exp(th[-3]):.2f}/{np.exp(th[-2]):.2f} px, w2 {1 / (1 + np.exp(-th[-1])):.3f}, "
                  f"{dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
