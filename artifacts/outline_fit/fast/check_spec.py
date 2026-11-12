"""Scratch check: the analytic template (spec.py) against the library's Template.render on its own 24 px box."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]  # the repo root (this file sits in artifacts/outline_fit/fast)
sys.path.insert(0, str(ROOT))
from subpx._outline.model import Template
from spec import Spec, Band

rng = np.random.default_rng(1)
for blur, box in (("gauss2", 24), ("lens", 48)):
    t = Template(upsample=8, size_px=box, blur=blur)
    sp = Spec(t, n_poly=512)
    band = Band(box_px=box, f_max=1.0, upsample=8)
    worst = 0.0
    for trial in range(6):
        a, b = rng.uniform(1.6, 3.2, 2)
        p = rng.uniform(1.7, 3.5)
        phi = rng.uniform(-0.3, 0.3)
        coeffs = rng.uniform(-0.03, 0.03, (2, 2))
        if blur == "gauss2":
            bp = t.blur_pack(s1x=rng.uniform(1.0, 1.5), s1y=rng.uniform(1.0, 1.5), s2x=rng.uniform(2, 3), s2y=rng.uniform(2, 3), w2=rng.uniform(0.02, 0.2))
        else:
            bp = t.blur_pack(sx=rng.uniform(0.6, 1.2), sy=rng.uniform(0.6, 1.2), cutoff=rng.uniform(0.35, 0.6))
        theta = t.pack(a, b, p, phi, coeffs, bp)
        for delta, bx in ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.125)):
            t0 = time.perf_counter()
            lib = t.render(theta, delta, box_px=bx)
            t1 = time.perf_counter()
            mine = band.to_grid(sp.template_hat(theta, band.fx, band.fy, delta, box_px=bx))
            t2 = time.perf_counter()
            d = np.abs(mine - lib).max() / lib.max()
            worst = max(worst, d)
            if trial == 0:
                print(f"{blur} delta {delta:+.1f} box {bx}: peak {lib.max():.4f}, max |diff| / peak {d:.2e}; lib {1e3*(t1-t0):.1f} ms, analytic {1e3*(t2-t1):.1f} ms ({band.fx.size} freqs)")
    print(f"{blur}: worst over 18 renders {worst:.2e}")
    # derivative templates as the library builds them
    th = theta
    t0l, tdl, t2l = t.render_quad(th)
    h = 0.1
    m0 = band.to_grid(sp.template_hat(th, band.fx, band.fy, 0.0))
    mp = band.to_grid(sp.template_hat(th, band.fx, band.fy, h))
    mm = band.to_grid(sp.template_hat(th, band.fx, band.fy, -h))
    print(f"   T1 diff / max {np.abs((mp-mm)/(2*h) - tdl).max()/np.abs(tdl).max():.2e}, T2 diff / max {np.abs((mp+mm-2*m0)/(2*h*h) - t2l).max()/np.abs(t2l).max():.2e}")
