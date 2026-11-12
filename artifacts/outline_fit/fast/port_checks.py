"""Scratch: the numbers quoted for the port's tests in docs/outline_fit_fast_handoff.md (section 5.4, 5.5).

    python port_checks.py

1. ShapeFitter under NumPy: the model's derivatives against central differences at a clearly non-round theta.
2. ShapeFitter.fit under NumPy on a stack synthesised from the model (a hole without neighbours), with noise: the
   outline CD recovered from the grid of starts and from a start at the truth.
3. One full page per call on the GPU (steps A to C): seconds.
4. gauss2 with its two Gaussians exchanged (pipeline.exchanged) is the same template; no GPU needed.
"""
from __future__ import annotations

import time

import numpy as np

from gpufit import ROOT
from shapefit import ShapeFitter
from subpx._outline.model import Template


def jacobian(family):
    t = Template(upsample=8, size_px=24, blur=family)
    sf = ShapeFitter(t, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np)
    bp = t.blur_pack(s1x=1.2, s1y=1.35, s2x=2.4, s2y=2.1, w2=0.08, sx=1.1, sy=1.25, cutoff=0.45)
    th = t.pack(2.9, 2.1, 2.6, 0.3, [[0.03, -0.02], [0.02, 0.015]], bp)
    gamma = np.ones((1, sf.U, sf.U, sf.nky, sf.nkx), dtype=complex)
    tm = np.eye(sf.P)
    M = sf.synth(sf.template_hat(th[None], tm), gamma)[0]  # (1 + P, U, U, np, np)
    worst = []
    for k in range(sf.P):
        h = 1e-5
        tp, tn = th.copy(), th.copy()
        tp[k] += h
        tn[k] -= h
        num = (sf.synth(sf.template_hat(tp[None]), gamma)[0, 0] - sf.synth(sf.template_hat(tn[None]), gamma)[0, 0]) / (2 * h)
        worst.append(float(np.abs(M[1 + k] - num).max() / np.abs(num).max()))
    print(f"1. Jacobian [{family}]: largest relative error per parameter " + ", ".join(f"{n} {e:.1e}" for n, e in zip(t.names, worst)))


def recover(family, seed=0):
    rng = np.random.default_rng(seed)
    t = Template(upsample=8, size_px=24, blur=family)
    sf = ShapeFitter(t, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np)
    bp = t.blur_pack(s1x=1.27, s1y=1.27, s2x=2.5, s2y=2.5, w2=0.05, sx=1.2, sy=1.2, cutoff=0.45)
    th = t.pack(2.4, 2.3, 2.2, 0.2, np.zeros((2, 2)), bp)
    gamma = np.ones((1, sf.U, sf.U, sf.nky, sf.nkx), dtype=complex)
    clean = 180.0 * sf.synth(sf.template_hat(th[None]), gamma)[0, 0] + 40.0  # (U, U, np, np)
    mask = sf.pad[:, None, :, None] & sf.pad[None, :, None, :]
    sd = 0.16  # grey levels per bin: 7.7 per pixel over about 2,300 holes
    y = np.where(mask, clean + rng.normal(0, sd, clean.shape), 0.0)
    w = np.where(mask, 1.0 / sd ** 2, 0.0)
    true_cd = t.cd_px(th) * 21.0
    for label, start, grid in (("grid of starts", t.start(1.7, 1.0), True), ("start at the truth", th, False)):
        t0 = time.perf_counter()
        theta, scale, const, cost, info = sf.fit(start[None], gamma, y[None], w[None], grid=grid)
        d = t.cd_px(theta[0]) * 21.0
        print(f"2. recover [{family}, {label}]: CD {d:.4f} nm for {true_cd:.4f} ({d - true_cd:+.4f}), scale {scale[0]:.2f}, const {const[0]:.2f}, "
              f"chi2 per bin {cost[0] / mask.sum():.3f}, {info}, {time.perf_counter() - t0:.1f} s")


def one_page():
    import tifffile

    from gpufit import sync
    from pipeline import fit_pages

    with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as tif:
        image = tif.pages[0].asarray()
    fit_pages([image], level=1)
    for family in ("gauss2", "lens"):
        times = []
        for _ in range(4):
            sync()
            t0 = time.perf_counter()
            (pg, out, res), = fit_pages([image], family=family, level=1)
            sync()
            times.append(time.perf_counter() - t0)
        print(f"3. one full page per call [{family}]: {min(times[1:]) * 1e3:.0f} to {max(times[1:]) * 1e3:.0f} ms (CD {out['cd']:.3f} nm)")


def exchange():
    from pipeline import exchanged, second_is_narrower

    t = Template(upsample=8, size_px=24, blur="gauss2")
    sf = ShapeFitter(t, box_px=56, f_max=0.6, U=4, R=8, n_poly=256, xp=np)
    th = t.pack(2.4, 2.3, 2.2, 0.1, [[0.01, -0.02], [0.015, 0.005]], t.blur_pack(s1x=1.6, s1y=1.5, s2x=1.2, s2y=1.1, w2=0.45))
    ex = exchanged(t, th)
    a, b = sf.template_hat(th[None])[0, :, 0], sf.template_hat(ex[None])[0, :, 0]
    edge = exchanged(t, t.pack(2.4, 2.3, 2.2, 0.1, np.zeros((2, 2)), t.blur_pack(w2=0.05)))
    print(f"4. gauss2 exchange: second Gaussian the narrower before {second_is_narrower(t, th)}, after {second_is_narrower(t, ex)}; "
          f"the two template spectra differ by {np.abs(a - b).max() / np.abs(a).max():.1e} of the peak; exchanged twice gives "
          f"theta back to {np.abs(exchanged(t, ex) - th).max():.1e}; a weight of 0.05 becomes {1 / (1 + np.exp(-edge[-1])):.2f} "
          f"(the 0.6 bound: then it is not the same blur)")


if __name__ == "__main__":
    exchange()
    for fam in ("gauss2", "lens"):
        jacobian(fam)
    for fam in ("gauss2", "lens"):
        recover(fam)
    one_page()
