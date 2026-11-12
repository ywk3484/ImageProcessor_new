"""Scratch: where the one-pass prototype's time goes (one page of the test file): shape fit against batch size,
hole sweeps in double / single precision."""
import sys, time
import numpy as np, cupy as cp, tifffile
from pathlib import Path
from gpufit import GpuPage, make_fitter, sync, ROOT
from pipeline import fitter_for, fit_page
with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as t:
    image = t.pages[0].asarray()
# warm-up (kernel compilation)
for tt in ("double", "float"):
    fit_page(image, tt=tt, sweeps_b=1, sweeps_d=2)
pg = GpuPage(image)
pg.build_stack(recentre=False)
sf = fitter_for(pg, "gauss2")
gamma = pg.pair_map(sf)
y, w = pg.stack_data(sf)
th0 = pg.start_theta()
for B in (1, 4, 16, 32, 4, 16, 32):
    g = cp.broadcast_to(gamma[None], (B,) + gamma.shape).copy(); yy = cp.broadcast_to(y[None], (B,) + y.shape).copy(); ww = cp.broadcast_to(w[None], (B,) + w.shape).copy()
    sync(); t0 = time.perf_counter(); sf.evals = 0
    theta, scale, const, cost, info = sf.fit(np.repeat(th0[None], B, 0), g, yy, ww)
    sync(); dt = time.perf_counter() - t0
    its = sum(i for _, i in info)
    print(f"shape fit, {B:2d} pages at once: {dt * 1e3:7.1f} ms = {dt * 1e3 / B:6.1f} ms per page; {its} iterations {info}, {dt * 1e3 / its:.2f} ms per iteration, CD {pg.t.cd_px(theta[0]) * 21:.4f}")
# one evaluation alone
tm = np.eye(sf.P)
for B in (1, 16, 32):
    th = np.repeat(theta[:1], B, 0); g = cp.broadcast_to(gamma[None], (B,) + gamma.shape).copy(); yy = cp.broadcast_to(y[None], (B,) + y.shape).copy(); ww = cp.broadcast_to(w[None], (B,) + w.shape).copy()
    data = sf.prepare(g, yy, ww)
    for name, fn in (("outline kernel", lambda: sf.outline_hat(th, True)), ("template_hat", lambda: sf.template_hat(th, tm)),
                     ("evaluate (cost, g, A)", lambda: sf.evaluate(th, data, tm)), ("grid of starts", lambda: sf.grid_start(th.copy(), data))):
        fn(); sync(); t0 = time.perf_counter()
        for _ in range(10):
            fn()
        sync()
        print(f"   {name:<26s} B={B:2d}: {(time.perf_counter() - t0) * 100:.2f} ms")
# hole sweeps
for tt in ("double", "float"):
    for rep in range(2):
        pg2, out, res = fit_page(image, tt=tt, sweeps_b=1, sweeps_d=4)
    tms = out["times"]
    print(f"tt={tt}: total {out['seconds'] * 1e3:.0f} ms; " + ", ".join(f"{k} {v * 1e3:.0f}" for k, v in tms.items()) + f"; CD {out['cd']:.4f}, per-hole sd {out['cd_sd']:.4f}, Rr {out['Rr']}")
