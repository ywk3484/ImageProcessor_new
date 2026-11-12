"""Scratch: the GPU one-pass fit against the truth on generated pages (subpx.synthetic_arrays.render_hole_array).

    python truth_gpu.py NAME [family=gauss2|lens] [cd=2.5] [blur=65] [lens=45] [place=1.0] [bright=0.05] [rot=0]
                        [seed=1] [pitch=X[,Y]] [size=X[,Y]] [variants=fast,truth,existing] [sweeps_b=2] [sweeps_d=3] [relax=1.3]
                        [part=0.5] [flicker=0.1]

    fast       the one-pass fit (pipeline.fit_page): the page's outline CD against the mean true CD of the stacked
               holes; the per-hole CDs' slope against the true CD, scatter, move with brightness and centre error
    truth      the page fit alone with the generator's true centres (what perfect centring would give)
    existing   the existing outline fit (same-numbers prototype) on the same page
part=F keeps holes only in the left fraction F of the page (the rest is the bare background with its noise);
flicker=A multiplies the light above the background by 1 + A sin (a slow wave across the page: source flicker).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

from gpufit import ROOT, GpuPage, sync
from pipeline import describe, fit_page, lib_template, page_fit

sys.path.insert(0, str(ROOT))
from scripts.pitch_signature_test import PATTERNS, PIXEL_NM, detection_limits, rows_apart  # noqa: E402
from subpx.synthetic_arrays import render_hole_array  # noqa: E402

DEFAULTS = dict(cd=2.5, blur=65.0, lens=45.0, place=1.0, bright=0.05, rot=0.0, seed=1, photons=30.0, read_noise=0.65,
                pitch=None, size=None, part=None, flicker=0.0)


def render(name, shape, a):
    spec = dict(PATTERNS[name])
    if a.get("pitch"):
        spec["pitch_nm"] = a["pitch"]
    if a.get("size"):
        spec["size_nm"] = a["size"]
    page = render_hole_array(
        tuple(shape), hole=spec["hole"], size_nm=spec["size_nm"], pitch_nm=spec["pitch_nm"], layout=spec["layout"],
        pixel_nm=PIXEL_NM, blur_nm=a["blur"], lens_nm=a["lens"], photons_per_level=a["photons"],
        read_noise=a["read_noise"], cd_sigma_nm=a["cd"], placement_sigma_nm=a["place"],
        brightness_sigma=a["bright"], row_shift_nm=0.0, row_shift_period_nm=1000.0,
        row_shift_every=rows_apart(spec["pitch_nm"], 1000.0, 5), rotation_deg=a["rot"],
        seed=a["seed"] + list(PATTERNS).index(name), backend="auto")
    image = page.image
    if a.get("flicker"):
        yy, xx = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        wave = 1.0 + a["flicker"] * np.sin(2 * np.pi * (xx / 700.0 + yy / 1100.0))
        bg = float(np.percentile(image, 1))
        image = np.clip(np.rint(bg + (image.astype(float) - bg) * wave), 0, 255).astype(image.dtype)
    if a.get("part"):
        cut = int(image.shape[1] * a["part"])
        # the pattern ends between two columns of holes (whole holes only, as at the edge of a real array)
        xs = np.sort(page.centers_xy[:, 0])
        gaps = np.flatnonzero(np.diff(xs) > 0.5 * min(spec["pitch_nm"]) / PIXEL_NM)
        if len(gaps) and a.get("part_whole", 1):
            mids = 0.5 * (xs[gaps] + xs[gaps + 1])
            cut = int(round(mids[np.argmin(np.abs(mids - cut))]))
        bg = float(np.percentile(image, 1))
        rng = np.random.default_rng(5)
        blank = np.clip(np.rint(bg + rng.normal(0, 2.0, (image.shape[0], image.shape[1] - cut))), 0, 255)
        image = image.copy()
        image[:, cut:] = blank.astype(image.dtype)
    return page, image, spec


def match(points, truth_xy, tol=2.0):
    from scipy.spatial import cKDTree

    d, j = cKDTree(truth_xy).query(points, distance_upper_bound=tol)
    return np.where(np.isfinite(d), j, -1)


def per_hole(res, page, true_cd):
    truth_xy = page.centers_xy
    idx = match(res["centers_xy"], truth_xy)
    ok = res["good"] & ~res["at_clip"] & (idx >= 0)
    t = true_cd[idx[ok]]
    b = page.brightness[idx[ok]]
    cd = res["cd_px"][ok] * PIXEL_NM
    slope, icpt = np.polyfit(t, cd, 1) if t.std() > 1e-6 else (float("nan"), float("nan"))
    if np.isfinite(slope):
        r = (cd - slope * t - icpt) / slope
        scat = float(np.std(r))
        light = float(np.polyfit(b - 1, r / np.median(t), 1)[0] * 0.1 * np.median(t)) if b.std() > 1e-9 else float("nan")
    else:
        scat = light = float("nan")
    cen = (res["centers_xy"][ok] - truth_xy[idx[ok]]) * PIXEL_NM
    return dict(n=int(ok.sum()), mean_error=float(cd.mean() - t.mean()), slope=float(slope), scatter=scat, light10=light,
                centre_sd=float(np.hypot(cen[:, 0].std(), cen[:, 1].std()) / np.sqrt(2)), true_mean=float(t.mean()))


def run_case(name, family="gauss2", variants=("fast",), shape=(1378, 2048), sweeps_b=2, sweeps_d=3, weighting="raw",
             say=print, relax=1.3, **over):
    a = {**DEFAULTS, **over}
    page, image, spec = render(name, shape, a)
    true_cd = np.sqrt(page.size_px[:, 0] * page.size_px[:, 1]) * PIXEL_NM
    if spec["hole"] == "square":  # the fit's CD is the area-equivalent diameter: a square of side s reads 2 s / sqrt(pi)
        true_cd = true_cd * 2.0 / np.sqrt(np.pi)
    truth_xy = page.centers_xy
    say(f"{name} {spec['size_nm']} nm, pitch {spec['pitch_nm']} nm, {spec['layout']}: {len(truth_xy):,} holes, true CD mean "
        f"{true_cd.mean():.3f} sd {true_cd.std():.3f} nm, placement sd {a['place']} nm, blur {a['blur']} / lens {a['lens']}, "
        f"rotation {a['rot']} deg, family {family}" + (f", holes in the left {a['part']:.0%}" if a["part"] else "")
        + (f", flicker {a['flicker']:.0%}" if a["flicker"] else ""))
    out = dict(name=name, family=family, settings={k: v for k, v in a.items()}, true_mean=float(true_cd.mean()),
               true_sd=float(true_cd.std()), holes=int(len(truth_xy)))

    def truth_mean(pg):
        idx = match(cp.asnumpy(pg.c), truth_xy)
        used = cp.asnumpy(pg.use) & (idx >= 0)
        return float(true_cd[idx[used]].mean())

    for v in variants:
        if v == "fast":
            for rep in range(2):  # the second run is timed without kernel compilation
                pg, o, res = fit_page(image, family=family, level=2, sweeps_b=sweeps_b, sweeps_d=sweeps_d,
                                      weighting=weighting, relax=relax)
            tm = truth_mean(pg)
            ph = per_hole(res, page, true_cd)
            f = o["first"]
            out["fast"] = dict(cd=o["cd"], error=o["cd"] - tm, first_error=f["cd"] - tm, blur=o["blur"], p=o["p"],
                               extent=o["extent"], chi2=o["chi2"], bins=o["bins"], seconds=o["seconds"], seeds=pg.n,
                               stacked=o["stacked"], per_hole=ph, at_clip=o["at_clip"], window=pg.R)
            sd_rel = float(np.nanmedian(res["delta_sd"])) / (0.5 * res["cd_outline_px"])
            out["fast"]["offset_se_over_radius"] = sd_rel
            say(f"  (median standard error of a hole's edge offset: {sd_rel:.3f} of the outline radius; the library switches per-hole sizes off above 0.3)")
            say(f"  one pass: outline CD {o['cd']:.3f} nm, error {o['cd'] - tm:+.3f} (first page fit {f['cd'] - tm:+.3f}); blur "
                f"{o['blur'][0]:.1f}/{o['blur'][1]:.1f}, p {o['p']:.2f}, x/y {o['extent'][0]:.1f}/{o['extent'][1]:.1f}, chi2 {o['chi2']:.0f}/{o['bins']}, "
                f"{f['stages']} + {o['stages']}\n"
                f"            per hole ({ph['n']:,} of {pg.n:,} found): mean error {ph['mean_error']:+.3f} nm, slope {ph['slope']:.3f}, scatter "
                f"{ph['scatter']:.2f} nm, +10 % light {ph['light10']:+.3f} nm, centres {ph['centre_sd']:.2f} nm, at the offset bound {o['at_clip']}\n"
                f"            {o['seconds'] * 1e3:.0f} ms: " + ", ".join(f"{k} {t * 1e3:.0f}" for k, t in o["times"].items()))
        elif v == "truth":
            pg = GpuPage(image, blur=family, seeds=truth_xy, weighting=weighting, lib_template=lib_template(family))
            pg.build_stack(recentre=False)
            theta, scale, const, fa = page_fit(pg, family)
            d = describe(pg, theta, PIXEL_NM)
            tm = truth_mean(pg)
            out["truth_centres"] = dict(cd=d["cd"], error=d["cd"] - tm)
            say(f"  page fit with the true centres: outline CD {d['cd']:.3f} nm, error {d['cd'] - tm:+.3f}; blur {d['blur'][0]:.1f}/"
                f"{d['blur'][1]:.1f}, p {d['p']:.2f}, chi2 {fa['chi2']:.0f}/{fa['bins']}")
        elif v == "existing":
            sys.path.insert(0, str(Path(__file__).parent.parent / "speed"))
            import tier1
            from subpx import detect_centers

            template_class = tier1.install(render="exact", gpu=True)
            import subpx._outline.page as P
            from subpx.outline import BLUR_FAMILIES, UPSAMPLE

            area_min, area_max = detection_limits(spec)
            seeds = detect_centers(image, backend="cpu", threshold="otsu", invert=False, area_min=area_min,
                                   area_max=area_max, pad=3, refine="logquad").centers_xy
            t0 = time.perf_counter()
            fitter = P.PageFit(image.astype(np.float64), seeds, pixel_nm=PIXEL_NM, unit="nm",
                               template=template_class(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[family], blur=family), xp=cp)
            res = fitter.run(rounds=14, iterations=5)
            sync()
            dt = time.perf_counter() - t0
            ph = per_hole(dict(centers_xy=res["centers_xy"], good=res["good"], at_clip=res["at_clip"], cd_px=res["cd_px"]),
                          page, true_cd)
            dd = res["describe"]
            out["existing"] = dict(cd=dd["cd_nm"], error=dd["cd_nm"] - ph["true_mean"], blur=[float(v) for v in dd["blur_fwhm_nm"]],
                                   p=dd["superellipse"]["p"], rounds=len(fitter.history), converged=bool(res["converged"]),
                                   seconds=dt, per_hole=ph, per_hole_size=bool(res["per_hole_size"]))
            say(f"  existing: outline CD {dd['cd_nm']:.3f} nm, error {dd['cd_nm'] - ph['true_mean']:+.3f}; blur {dd['blur_fwhm_nm'][0]:.1f}/"
                f"{dd['blur_fwhm_nm'][1]:.1f}, p {dd['superellipse']['p']:.2f}; {len(fitter.history)} rounds, converged {res['converged']}, "
                f"per-hole size {'on' if res['per_hole_size'] else 'off'}\n"
                f"            per hole ({ph['n']:,}): mean error {ph['mean_error']:+.3f} nm, slope {ph['slope']:.3f}, scatter {ph['scatter']:.2f} nm, "
                f"+10 % light {ph['light10']:+.3f} nm, centres {ph['centre_sd']:.2f} nm\n            {dt * 1e3:.0f} ms")
            del fitter
            from subpx._gpu._cuda import release_memory

            release_memory()
    return out


def main():
    name = sys.argv[1]
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    pair = lambda key: tuple(float(v) for v in (args[key].split(",") * 2)[:2]) if key in args else None  # noqa: E731
    over = {k: float(args[k]) for k in ("cd", "blur", "lens", "place", "bright", "rot", "photons", "read_noise", "part", "flicker")
            if k in args}
    if "seed" in args:
        over["seed"] = int(args["seed"])
    over.update(pitch=pair("pitch"), size=pair("size"))
    run_case(name, family=args.get("family", "gauss2"), variants=tuple(args.get("variants", "fast,truth").split(",")),
             shape=tuple(int(v) for v in args.get("shape", "1378,2048").split(",")), sweeps_b=int(args.get("sweeps_b", 2)),
             sweeps_d=int(args.get("sweeps_d", 3)), weighting=args.get("weighting", "raw"), relax=float(args.get("relax", 1.3)),
             say=lambda s: print(s, flush=True), **over)


if __name__ == "__main__":
    main()
