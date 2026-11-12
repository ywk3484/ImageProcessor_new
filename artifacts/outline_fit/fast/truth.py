"""Scratch: the one-pass page fit against the truth on generated pages (subpx.synthetic_arrays.render_hole_array).

    python truth.py NAME [family=gauss2|lens] [cd=2.5] [blur=65] [lens=45] [place=1.0] [bright=0.05] [rot=0] [seed=1]
                    [centres=seeds|truth] [shape=1378,2048] [existing=1]

Prints the page's outline CD against the mean true CD of the stacked holes. existing=1 also runs the existing
outline fit (same-numbers prototype, GPU) on the same page and seeds.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from fastfit import ROOT, FastFit
from shapefit import ShapeFitter

sys.path.insert(0, str(ROOT))
from scripts.pitch_signature_test import PATTERNS, PIXEL_NM, detection_limits, rows_apart  # noqa: E402
from subpx import detect_centers  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402


def render(name, shape, a):
    spec = PATTERNS[name]
    return render_hole_array(
        tuple(shape), hole=spec["hole"], size_nm=spec["size_nm"], pitch_nm=spec["pitch_nm"], layout=spec["layout"],
        pixel_nm=PIXEL_NM, blur_nm=a["blur"], lens_nm=a["lens"], photons_per_level=a["photons"],
        read_noise=a["read_noise"], cd_sigma_nm=a["cd"], placement_sigma_nm=a["place"],
        brightness_sigma=a["bright"], row_shift_nm=0.0, row_shift_period_nm=1000.0,
        row_shift_every=rows_apart(spec["pitch_nm"], 1000.0, 5), rotation_deg=a["rot"],
        seed=a["seed"] + list(PATTERNS).index(name), backend="auto")


def fast_page(image, seeds, family, *, U=4, f_max=0.6, centres=None, xp=np, log=None):
    """The one-pass page fit: (theta, describe dict, FastFit, seconds)."""
    t0 = time.perf_counter()
    f = FastFit(image, seeds, blur=family, U=U, f_max=f_max)
    if centres is not None:
        f.c = np.array(centres, dtype=float)
    f.build_stack(passes=2)
    sf = ShapeFitter(f.t, box_px=48, f_max=f_max, U=f.U, R=f.R, n_poly=256, xp=xp)
    y, mask = sf.to_classes(xp.asarray(np.where(np.isfinite(f.stack), f.stack, 0.0)))
    w, _ = sf.to_classes(xp.asarray(f.weight))
    noise2 = float(np.median(f.rms[f.use])) ** 2
    w = xp.where(mask, w, 0.0) / noise2
    gam = xp.asarray(np.fft.fftshift(f.gamma, axes=-2).reshape(1, f.U, f.U, sf.nky, sf.nkx))
    theta, scale, const, cost, info = sf.fit(f.start_theta()[None], gam, y[None], w[None], log=log)
    d = f.t.describe(theta[0], PIXEL_NM)
    out = dict(cd_nm=d["cd_nm"], blur=[float(v) for v in d["blur_fwhm_nm"]], p=d["superellipse"]["p"],
               extent=(d["extent_x_nm"], d["extent_y_nm"]), scale=float(scale[0]), const=float(const[0]),
               chi2=float(cost[0]), bins=int((w > 0).sum()), evals=sf.evals, stages=info, good=int(f.use.sum()))
    return theta[0], out, f, time.perf_counter() - t0


def match_truth(points, page, tol):
    from scipy.spatial import cKDTree

    d, j = cKDTree(page.centers_xy).query(points, distance_upper_bound=tol)
    return np.where(np.isfinite(d), j, -1)


def main():
    name = sys.argv[1]
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    a = dict(cd=float(args.get("cd", 2.5)), blur=float(args.get("blur", 65.0)), lens=float(args.get("lens", 45.0)),
             place=float(args.get("place", 1.0)), bright=float(args.get("bright", 0.05)), rot=float(args.get("rot", 0.0)),
             seed=int(args.get("seed", 1)), photons=float(args.get("photons", 30.0)), read_noise=float(args.get("read_noise", 0.65)))
    family = args.get("family", "gauss2")
    shape = tuple(int(v) for v in args.get("shape", "1378,2048").split(","))
    spec = PATTERNS[name]
    t0 = time.perf_counter()
    page = render(name, shape, a)
    t_render = time.perf_counter() - t0
    area_min, area_max = detection_limits(spec)
    t0 = time.perf_counter()
    det = detect_centers(page.image, backend="cpu", threshold="otsu", invert=False, area_min=area_min, area_max=area_max,
                         pad=3, refine="logquad")
    seeds = det.centers_xy
    t_seed = time.perf_counter() - t0
    true_cd = np.sqrt(page.size_px[:, 0] * page.size_px[:, 1]) * PIXEL_NM
    print(f"{name}: {len(page.centers_xy):,} true holes, {len(seeds):,} seeds, true CD mean {true_cd.mean():.3f} sd {true_cd.std():.3f} nm, "
          f"blur {a['blur']} lens {a['lens']}, rendered in {t_render:.1f} s, seeds {t_seed:.1f} s", flush=True)
    for centres in args.get("centres", "seeds").split(","):
        c = None
        if centres == "truth":
            idx = match_truth(seeds, page, 2.0)
            c = np.where(idx[:, None] >= 0, page.centers_xy[np.clip(idx, 0, None)], seeds)
        theta, d, f, dt = fast_page(page.image, seeds, family, centres=c)
        idx = match_truth(f.c, page, 2.0)
        used = f.use & (idx >= 0)
        t_mean = float(true_cd[idx[used]].mean())
        # the stack weights every hole by its brightness a: the a-weighted mean of the true CDs is what it averages
        t_wmean = float(np.average(true_cd[idx[used]], weights=f.a[used]))
        print(f"  one pass [{family}, centres {centres}]: outline CD {d['cd_nm']:.3f} nm; true mean {t_mean:.3f} (brightness-weighted "
              f"{t_wmean:.3f}) -> error {d['cd_nm'] - t_mean:+.3f} nm; blur {d['blur'][0]:.1f}/{d['blur'][1]:.1f}, p {d['p']:.2f}, "
              f"chi2 {d['chi2']:.0f} / {d['bins']} bins, {d['evals']} evaluations {d['stages']}, {dt:.1f} s", flush=True)
    if args.get("existing", "0") == "1":
        sys.path.insert(0, str(Path(__file__).parent.parent / "speed"))
        import tier1

        template_class = tier1.install(render="exact", gpu=True)
        import cupy as cp
        import subpx._outline.page as P
        from subpx.outline import BLUR_FAMILIES, UPSAMPLE

        t0 = time.perf_counter()
        fitter = P.PageFit(page.image.astype(np.float64), seeds, pixel_nm=PIXEL_NM, unit="nm",
                           template=template_class(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[family], blur=family), xp=cp)
        res = fitter.run(rounds=14, iterations=5)
        cp.cuda.get_current_stream().synchronize()
        dt = time.perf_counter() - t0
        idx = match_truth(res["centers_xy"], page, 2.0)
        good = res["good"] & (idx >= 0)
        cd = res["cd_px"] * PIXEL_NM
        t = true_cd[idx[good]]
        slope = np.polyfit(t, cd[good], 1)[0] if t.std() > 1e-6 else float("nan")
        dd = res["describe"]
        print(f"  existing fit [{family}]: outline CD {dd['cd_nm']:.3f} nm; true mean {t.mean():.3f} -> error {dd['cd_nm'] - t.mean():+.3f} nm "
              f"(per-hole mean error {cd[good].mean() - t.mean():+.3f}, slope {slope:.3f}); blur {dd['blur_fwhm_nm'][0]:.1f}/{dd['blur_fwhm_nm'][1]:.1f}, "
              f"p {dd['superellipse']['p']:.2f}, {len(fitter.history)} rounds, converged {res['converged']}, {dt:.1f} s", flush=True)


if __name__ == "__main__":
    main()
