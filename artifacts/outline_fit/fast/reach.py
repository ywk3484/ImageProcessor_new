"""Scratch: how far the stack's model follows the neighbours' light (the pair reach; default window half width + 13
px), for the lens form, whose blur has faint far rings.

    python reach.py ends [pages=c100,stag,c80,c100b72] [reach=0,30,40]      0 = the default
    python reach.py time [reach=0,30,40]

ends: on a generated full page whose blur is in the lens family (a Gaussian smear times a circular pupil with a
      cut-off of 0.48 cycles/px), the final fit from three starts (pupil cut-off 0.5; 0.9; 3 = no pupil inside the
      band): the page CD's error, the cost, the fitted smear and cut-off, per reach. The generator's own values are
      printed in the heading: the end that has them is the right one. Last line per reach: the blur held at the
      generator's values and only the outline fitted (the average hole is smeared a little more than one hole, so
      this fits worse than the free fit: an oracle test must leave the smear free).
time: ms per page for the page values, 20 pages of the test file per call, both blur forms, per reach; and how far
      the page CD moves.
"""
from __future__ import annotations

import json
import sys
import time

import cupy as cp
import numpy as np

import pipeline
from basins import PAGES
from gpufit import ROOT, sync
from pipeline import describe, fit_page, fit_pages, fitter_for, page_fit
from truth_gpu import DEFAULTS, PIXEL_NM, match, render


def outline_only(pg, start):
    """The final stage with the blur held where ``start`` has it: the stage map is replaced for one call."""
    sf = fitter_for(pg, "lens")
    j = pg.t.n_shape
    keep = sf._stage

    def stage(name):
        tm = np.zeros((sf.P, j))
        tm[np.arange(j), np.arange(j)] = 1.0
        return list(range(j)), {}, tm

    sf._stage = stage
    try:
        return page_fit(pg, "lens", theta0=np.asarray(start), stages=("full",), grid=False, max_iter=(200,), ftol=1e-12)
    finally:
        sf._stage = keep


def ends(args):
    from subpx.synthetic_arrays import LENS_FWHM_FACTOR, blur_smear_sigma

    reaches = [float(v) for v in args.get("reach", "0,30,40").split(",")]
    for key in args.get("pages", "c100,stag,c80,c100b72").split(","):
        name, over = PAGES[key]
        a = {**DEFAULTS, **over}
        page, image, _ = render(name, (1378, 2048), a)
        true_cd = np.sqrt(page.size_px[:, 0] * page.size_px[:, 1]) * PIXEL_NM
        smear = blur_smear_sigma(a["blur"] / PIXEL_NM, a["lens"] / PIXEL_NM)
        cut = LENS_FWHM_FACTOR / (a["lens"] / PIXEL_NM)
        print(f"{name}, blur {a['blur']:.0f} nm: generated with smear {smear:.4f} px and pupil cut-off {cut:.4f} cycles/px; every "
              f"hole's light is drawn on a patch of {page.meta.get('patch_px')} px", flush=True)
        for reach in reaches:
            pipeline.FITTERS.clear()
            pg, out, _ = fit_page(image, family="lens", level=1, pair_reach_px=reach or None)
            j = pg.t.n_shape
            idx = match(cp.asnumpy(pg.c), page.centers_xy)
            used = cp.asnumpy(pg.use) & (idx >= 0)
            tm = float(true_cd[idx[used]].mean())
            end = np.asarray(out["theta"])
            print(f"   pair reach {pg.reach:.0f} px (fit box {pg.L} px, window +-{pg.R} px); the pipeline: {out['cd'] - tm:+.3f} nm")
            for label, cutoff, width in (("pupil cut-off 0.5", 0.5, 1.0), ("cut-off 0.9, smear x 1.15", 0.9, 1.15),
                                         ("cut-off 3 (no pupil), smear x 1.3", 3.0, 1.3)):
                th0 = end.copy()
                th0[j + 2] = np.log(cutoff)
                th0[j:j + 2] += np.log(width)
                th, _, _, info = page_fit(pg, "lens", theta0=th0, stages=("full",), grid=False, max_iter=(400,), ftol=1e-12)
                print(f"      start {label:34s}: {describe(pg, th, PIXEL_NM)['cd'] - tm:+.3f} nm, cost {info['chi2']:.1f}, smear "
                      f"{np.exp(th[j]):.4f}/{np.exp(th[j + 1]):.4f} px, cut-off {np.exp(th[j + 2]):.4f}", flush=True)
            th0 = end.copy()
            th0[j:j + 2] = np.log(smear)
            th0[j + 2] = np.log(cut)
            th, _, _, info = outline_only(pg, th0)
            print(f"      {'blur held at the generated values':40s}: {describe(pg, th, PIXEL_NM)['cd'] - tm:+.3f} nm, cost "
                  f"{info['chi2']:.1f}", flush=True)
    pipeline.FITTERS.clear()


def timing(args):
    import tifffile

    reaches = [float(v) for v in args.get("reach", "0,30,40").split(",")]
    ref = json.loads((ROOT / "artifacts" / "outline_fit" / "fast" / "runs" / "ref" / "ref.json").read_text())
    pages = sorted(r["page"] for r in ref if not r.get("skipped") and r["holes"] > 30000)[:20]
    with tifffile.TiffFile(ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif") as t:
        images = [t.pages[p].asarray() for p in pages]
    for family in ("gauss2", "lens"):
        base = None
        for reach in reaches:
            pipeline.FITTERS.clear()
            kw = dict(family=family, level=1, pair_reach_px=reach or None)
            fit_pages(images[:4], **kw)  # compile, warm up
            best = None
            for _ in range(3):
                sync()
                t0 = time.perf_counter()
                res = fit_pages(images, **kw)
                sync()
                dt = (time.perf_counter() - t0) / len(images) * 1e3
                best = dt if best is None else min(best, dt)
            cds = np.array([o["cd"] for _, o, _ in res])
            base = cds if base is None else base
            pg = res[0][0]
            print(f"{family:6s} pair reach {pg.reach:.0f} px (fit box {pg.L} px): {best:5.1f} ms per page; page CD against the first "
                  f"reach: mean {np.mean(cds - base):+.3f}, largest {np.abs(cds - base).max():.3f} nm", flush=True)
    pipeline.FITTERS.clear()


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "ends"
    {"ends": ends, "time": timing}[what](dict(a.split("=", 1) for a in sys.argv[2:]))
