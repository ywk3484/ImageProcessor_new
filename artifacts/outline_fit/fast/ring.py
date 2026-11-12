"""Scratch: do the holes near the image border bias a small image's page CD (light of holes beyond the border is
not in the model)? Same page, same centres, same model; only the set of stacked holes changes:

    all     window inside the image (the default)
    inner   every neighbour within the pair reach is a whole hole inside the image
    ring    the rest (the holes near the border)

    python ring.py [family=lens,gauss2]

If the border biased the page, "ring" would read away from "inner" by more than the scatter between images.
"""
from __future__ import annotations

import sys

import cupy as cp
import numpy as np
from scipy.spatial import cKDTree

from gpufit import ROOT
from pipeline import describe, exchanged, fit_page, page_fit, second_is_narrower

sys.path.insert(0, str(ROOT))
from subpx.synthetic_arrays import render_hole_array  # noqa: E402

PIXEL_NM = 21.0
CASES = [("120 x 170 (library test page)", (120, 170), dict(pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=5)),
         ("300 x 400", (300, 400), dict(pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=6)),
         ("600 x 800", (600, 800), dict(pitch_nm=(200, 200), cd_sigma_nm=2.5, seed=7)),
         ("900 x 1200", (900, 1200), dict(pitch_nm=(210, 210), cd_sigma_nm=3.0, seed=9))]


def refit(pg, family, start):
    kw = dict(stages=("full",), grid=False, max_iter=(300,), ftol=1e-12)
    th, _, _, info = page_fit(pg, family, theta0=np.asarray(start), **kw)
    if family == "gauss2" and second_is_narrower(pg.t, th):
        th2, _, _, info2 = page_fit(pg, family, theta0=exchanged(pg.t, th), **kw)
        if info2["chi2"] < info["chi2"]:
            th, info = th2, info2
    return th, info


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    for name, shape, kw in CASES:
        page = render_hole_array(shape, hole="circle", size_nm=(100, 100), layout="square", backend="cpu", **kw)
        true_cd = np.sqrt(page.size_px[:, 0] * page.size_px[:, 1]) * PIXEL_NM
        for family in args.get("family", "lens,gauss2").split(","):
            pg, out, _ = fit_page(page.image, family=family, level=1)
            c = cp.asnumpy(pg.c)
            d, j = cKDTree(page.centers_xy).query(c)
            good = cp.asnumpy(pg.use)
            m = pg.reach + pg.spacing
            inner = good & (c[:, 0] >= m) & (c[:, 0] <= pg.w - 1 - m) & (c[:, 1] >= m) & (c[:, 1] <= pg.h - 1 - m)
            line = f"{name:30s} {family:6s}:"
            for label, mask in (("all", good), ("inner", inner), ("ring", good & ~inner)):
                ok = mask & (d < 1.5)
                if ok.sum() < 10:
                    line += f"  {label}: {int(ok.sum())} holes, not fitted;"
                    continue
                use = cp.asarray(mask)
                pg.use = use
                pg.stack, pg.wsum, pg.sw, pg.nrm = pg._stack(use)
                pg.noise = float(cp.median(pg.hole_rms[use]))
                th, info = refit(pg, family, out["theta"])
                line += (f"  {label}: {describe(pg, th, PIXEL_NM)['cd'] - float(true_cd[j[ok]].mean()):+.3f} nm "
                         f"({int(ok.sum())} holes, chi2 per bin {info['chi2'] / info['bins']:.2f});")
            print(line, flush=True)


if __name__ == "__main__":
    main()
