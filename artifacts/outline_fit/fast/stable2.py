"""Scratch: how far one centring round is from the settled page CD, against the number of passes in that round."""
import sys
import cupy as cp, numpy as np, tifffile
from gpufit import ROOT, GpuPage
from pipeline import describe, lib_template, page_fits
with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as t:
    images = {p: t.pages[p].asarray() for p in (0, 24)}
for p, image in images.items():
    for sweeps in (1, 2, 3):
        pg = GpuPage(image, tt="float", lib_template=lib_template("gauss2"))
        pg.build_stack(recentre=False)
        (theta, scale, const, fa), = page_fits([pg], "gauss2", stages=("round", "super"), max_iter=(10, 12), ftol=1e-6)
        line = f"page {p}, {sweeps} pass(es) in the first round: first fit {describe(pg, theta, 21.0)['cd']:.4f}"
        for cycle in range(4):
            pg.set_template(theta, scale, const, keep_holes=cycle > 0)
            pg.fit_holes(sweeps=sweeps if cycle == 0 else 1, size=False)
            pg.restack(pg.good_mask(), scale)
            (theta, scale, const, fc), = page_fits([pg], "gauss2", theta0=[theta], stages=("full",), grid=False, max_iter=(30,))
            line += f" -> {describe(pg, theta, 21.0)['cd']:.4f}"
        print(line, flush=True)
