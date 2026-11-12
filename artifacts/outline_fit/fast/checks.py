"""Scratch: further checks of the one-pass fit on the test file.

    python checks.py other      pages of the test file without the hole pattern: what the seed finder and the fit do
    python checks.py stable     a second centring pass + page fit after the first: how much the page's CD still moves
    python checks.py load       CPU seconds and GPU memory per page, time against batch size
    python checks.py repeat     the same pages twice: do the numbers repeat
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import tifffile

from gpufit import ROOT, GpuPage, find_seeds, sync
from pipeline import describe, fit_pages, lib_template, page_fits

TIF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"
REF = Path(__file__).parent / "runs" / "ref"


def load(pages):
    with tifffile.TiffFile(TIF) as t:
        return [t.pages[p].asarray() for p in pages]


def other():
    for p, image in zip((3, 43, 46, 47), load((3, 43, 46, 47))):
        img = cp.asarray(image).astype(cp.float32)
        s = find_seeds(img)
        lo, hi = (float(v) for v in cp.percentile(img[::4, ::4], cp.asarray([5.0, 99.8])))
        line = f"page {p}: mean {image.mean():.1f}, 5 % / 99.8 % levels {lo:.0f} / {hi:.0f}, seeds {int(s.shape[0]):,}"
        if int(s.shape[0]) >= 10:
            try:
                pg, out, res = fit_pages([image], level=1)[0]
                line += (f"; fit: spacing {out['spacing']:.1f} px, stacked {out['stacked']:,}, outline CD {out['cd']:.2f} nm, blur "
                         f"{out['blur'][0]:.1f}/{out['blur'][1]:.1f}, chi2 {out['chi2']:.0f}/{out['bins']}, first to second fit "
                         f"{out['cd'] - out['first']['cd']:+.3f} nm")
            except Exception as error:  # what a page without the pattern does to the fit is the finding
                line += f"; fit failed: {type(error).__name__}: {str(error)[:120]}"
        print(line, flush=True)


def stable():
    pages = (0, 12, 24, 36)
    family = sys.argv[2] if len(sys.argv) > 2 else "gauss2"
    for p, image in zip(pages, load(pages)):
        pg = GpuPage(image, blur=family, tt="float", lib_template=lib_template(family))
        pg.build_stack(recentre=False)
        (theta, scale, const, fa), = page_fits([pg], family, stages=("round", "super"), max_iter=(10, 12), ftol=1e-6)
        line = f"page {p} [{family}]: first fit {describe(pg, theta, 21.0)['cd']:.4f}"
        for cycle in range(4):
            pg.set_template(theta, scale, const, keep_holes=cycle > 0)
            pg.fit_holes(sweeps=1, size=False)
            pg.restack(pg.good_mask(), scale)
            (theta, scale, const, fc), = page_fits([pg], family, theta0=[theta], stages=("full",), grid=False, max_iter=(30,))
            d = describe(pg, theta, 21.0)
            line += f" -> {d['cd']:.4f} (blur {d['blur'][0]:.2f}/{d['blur'][1]:.2f})"
        print(line, flush=True)


def load_test():
    import os

    rows = [r["page"] for r in json.loads((REF / "ref.json").read_text()) if not r.get("skipped") and r["holes"] > 30000]
    images = load(rows)
    fit_pages(images[:4])  # warm-up
    for level in (1, 2):
        for batch in (4, 10, 20, 40):
            cp.get_default_memory_pool().free_all_blocks()
            sync()
            t0, c0 = time.perf_counter(), time.process_time()
            for s in range(0, len(images), batch):
                fit_pages(images[s:s + batch], level=level, sweeps_d=3, relax=1.3)
            sync()
            dt, dc = time.perf_counter() - t0, time.process_time() - c0
            pool = cp.get_default_memory_pool()
            print(f"level {level}, batches of {batch:2d}: {dt / len(images) * 1e3:6.1f} ms per page, CPU {dc / len(images) * 1e3:6.1f} ms per page "
                  f"({dc / dt:.2f} cores busy), GPU memory pool {pool.total_bytes() / 2 ** 30:.2f} GB", flush=True)


def repeat():
    pages = (0, 5, 24, 30)
    images = load(pages)
    for family in ("gauss2", "lens"):
        a = fit_pages(images, family=family, level=2, sweeps_d=3, relax=1.3)
        b = fit_pages(images, family=family, level=2, sweeps_d=3, relax=1.3)
        for p, (pa, oa, ra), (pb, ob, rb) in zip(pages, a, b):
            d = (ra["cd_px"] - rb["cd_px"]) * 21.0
            print(f"page {p} [{family}]: outline CD {oa['cd']:.5f} / {ob['cd']:.5f} (apart {abs(oa['cd'] - ob['cd']):.5f} nm), per-hole CD apart "
                  f"rms {np.sqrt(np.nanmean(d * d)):.5f}, largest {np.nanmax(np.abs(d)):.5f} nm; blur {oa['blur'][0]:.2f}/{oa['blur'][1]:.2f}", flush=True)


if __name__ == "__main__":
    {"other": other, "stable": stable, "load": load_test, "repeat": repeat}[sys.argv[1]]()
