"""Scratch: what one hole sweep costs, kernel by kernel (one page of the test file, single-precision templates)."""
import sys, time
import numpy as np, cupy as cp, tifffile
from gpufit import GpuPage, sync, ROOT
from pipeline import fit_page, lib_template, page_fit
with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as t:
    image = t.pages[0].asarray()
fit_page(image)  # warm-up
pg = GpuPage(image, tt="float", lib_template=lib_template("gauss2"))
pg.build_stack(recentre=False)
theta, scale, const, fa = page_fit(pg, "gauss2", stages=("round", "super"), max_iter=(10, 12))
pg.set_template(theta, scale, const)
chunks = pg.colours()
print(f"{pg.n:,} holes, {len(chunks)} colours: sizes {[int(c.shape[0]) for c in chunks]}")
print(f"window R {pg.R} ({(2 * pg.R + 1) ** 2} px), render Rr {pg.Rr} ({(2 * pg.Rr + 1) ** 2} px, light within {pg.r_light:.1f} px), template grid {pg.tN}")
# time the kernels of one sweep by wrapping get_function
times = {}
orig = pg.mod.get_function
class Timed:
    def __init__(self, name, fn):
        self.name, self.fn = name, fn
    def __call__(self, *a, **k):
        sync(); t0 = time.perf_counter()
        r = self.fn(*a, **k)
        sync(); times[self.name] = times.get(self.name, 0.0) + time.perf_counter() - t0
        return r
class Mod:
    def get_function(self, name):
        return Timed(name, orig(name))
pg.mod = Mod()
for size in (False, True):
    times.clear()
    sync(); t0 = time.perf_counter()
    log = pg.fit_holes(sweeps=4, size=size)
    sync(); dt = time.perf_counter() - t0
    print(f"size {'on' if size else 'off'}: 4 sweeps + set-up {dt * 1e3:.1f} ms (set-up {pg.times['holes_setup'] * 1e3:.1f}, sweeps {pg.times['holes_sweeps'] * 1e3:.1f}); kernels: "
          + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in times.items()))
    pg.times.pop("holes_setup"); pg.times.pop("holes_sweeps")
