"""Scratch: where the final page fit can end. On the pipeline's final stack of a page the final stage is started
from 81 points round the pipeline's end (sizes x 0.92 / 1 / 1.08, exponent 1.7 / 2 / 2.5, blur widths x 0.8 / 1 /
1.25, three settings of the blur's second part) and run to its end (400 iterations at most, ftol 1e-12); gauss2 ends
with the narrower Gaussian second are exchanged and fitted again, as the pipeline does. The ends are grouped by page
CD (0.02 nm) and listed by cost (chi-square plus priors).

    python basins.py [pages=c100,c80,c100b72,c80b72,stag,sq150,ell,c250,file] [family=lens,gauss2]

One group = the fit has one answer on that page. Several groups = the answer depends on the start; the pipeline's
own end is printed first. Pages: generated full pages with known sizes (truth_gpu.render) and pages 0 and 22 of the
test file.
"""
from __future__ import annotations

import sys

import cupy as cp
import numpy as np

from gpufit import ROOT
from pipeline import describe, exchanged, fit_page, fitter_for, second_is_narrower
from truth_gpu import DEFAULTS, PIXEL_NM, match, render

PAGES = {"c100": ("circle100_square", {}), "c80": ("circle80_square", {}), "c80b72": ("circle80_square", dict(blur=72.0)),
         "c100b72": ("circle100_square", dict(blur=72.0)), "stag": ("circle100_stagger", {}),
         "sq150": ("square150_stagger", {}), "ell": ("ellipse160x80_square", {}), "c250": ("circle250_square", {})}


def starts_round(pg, family, end):
    j = pg.t.n_shape
    seconds = [0.3, 0.5, 0.8] if family == "lens" else [(2.0, 0.05), (3.0, 0.2), (1.3, 0.4)]
    out = []
    for size in (0.92, 1.0, 1.08):
        for p in (1.7, 2.0, 2.5):
            for width in (0.8, 1.0, 1.25):
                for second in seconds:
                    th = end.copy()
                    th[0:2] += np.log(size)
                    th[2] = np.log(p - 1.0)
                    if family == "lens":  # (log smear x, log smear y, log pupil cut-off)
                        th[j:j + 2] += np.log(width)
                        th[j + 2] = np.log(second)
                    else:  # (log s1 x, y, log s2 x, y, logit w2): the second Gaussian as a ratio and a weight
                        s1 = np.exp(end[j:j + 2]) * width
                        th[j:j + 2] = np.log(s1)
                        th[j + 2:j + 4] = np.log(s1 * second[0])
                        th[j + 4] = np.log(second[1] / (1 - second[1]))
                    out.append(th)
    return np.stack(out)


def blur_text(pg, family, th):
    j = pg.t.n_shape
    if family == "lens":
        return f"smear {np.exp(th[j]):.3f}/{np.exp(th[j + 1]):.3f} px, pupil cut-off {np.exp(th[j + 2]):.3f} cycles/px"
    return (f"s1 {np.exp(th[j]):.2f}/{np.exp(th[j + 1]):.2f}, s2 {np.exp(th[j + 2]):.2f}/{np.exp(th[j + 3]):.2f} px, "
            f"w2 {1 / (1 + np.exp(-th[j + 4])):.3f}")


def ends_of(image, family, label, truth=None, cap=400, say=print):
    pg, out, _ = fit_page(image, family=family, level=1)
    sf = fitter_for(pg, family)
    ref = truth(pg) if truth else 0.0
    th0 = starts_round(pg, family, np.asarray(out["theta"]))
    n = len(th0)
    gam = pg.pair_map(sf)
    y, w = pg.stack_data(sf)
    G, Y, W = cp.stack([gam] * n), cp.stack([y] * n), cp.stack([w] * n)
    th, _, _, cost, info = sf.fit(th0, G, Y, W, stages=("full",), grid=False, max_iter=(cap,), ftol=1e-12)
    swapped = np.array([family == "gauss2" and second_is_narrower(pg.t, t) for t in th])
    if swapped.any():
        th2, _, _, cost2, _ = sf.fit(np.stack([exchanged(pg.t, t) for t in th]), G, Y, W, stages=("full",), grid=False,
                                     max_iter=(cap,), ftol=1e-12)
        take = swapped & (cost2 < cost)
        th[take], cost[take] = th2[take], cost2[take]
    cds = np.array([describe(pg, t, PIXEL_NM)["cd"] for t in th]) - ref
    groups = []
    for i in np.argsort(cost):
        for g in groups:
            if abs(cds[i] - g["cd"]) < 0.02:
                g["n"] += 1
                g["cost_hi"] = max(g["cost_hi"], float(cost[i]))
                break
        else:
            groups.append(dict(cd=float(cds[i]), n=1, cost=float(cost[i]), cost_hi=float(cost[i]), theta=th[i]))
    # an end is a stray when its cost is far above the best one (a start that fell off: outline on a bound)
    best = groups[0]["cost"]
    real = [g for g in groups if g["cost"] < 1.5 * best]
    say(f"{label}, {family}: the pipeline ends at {out['cd'] - ref:+.3f} nm (cost {out['chi2']:.1f}, {out['bins']} bins, "
        f"{out['stacked']} holes); {n} starts, {int(swapped.sum())} ends exchanged")
    for g in real:
        span = f" to {g['cost_hi']:.1f}" if g["cost_hi"] - g["cost"] > 0.05 else ""
        say(f"     {g['n']:3d} starts end at {g['cd']:+.3f} nm, cost {g['cost']:.1f}{span}; p {1 + np.exp(g['theta'][2]):.2f}, "
            f"{blur_text(pg, family, g['theta'])}")
    strays = sum(g["n"] for g in groups) - sum(g["n"] for g in real)
    if strays:
        say(f"     {strays} starts fell off (cost above 1.5 x the best)")
    return dict(label=label, family=family, pipeline=out["cd"] - ref, pipeline_cost=out["chi2"],
                ends=[dict(n=g["n"], cd=g["cd"], cost=g["cost"]) for g in real], strays=strays)


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    families = args.get("family", "lens,gauss2").split(",")
    rows = []
    for key in args.get("pages", "c100,c80,c100b72,c80b72,stag,sq150,ell,c250,file").split(","):
        if key == "file":
            import tifffile

            with tifffile.TiffFile(ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif") as t:
                for p in (0, 22):
                    image = t.pages[p].asarray()
                    for family in families:
                        rows.append(ends_of(image, family, f"test file page {p} (page CD in nm, no truth)"))
            continue
        name, over = PAGES[key]
        page, image, spec = render(name, (1378, 2048), {**DEFAULTS, **over})
        true_cd = np.sqrt(page.size_px[:, 0] * page.size_px[:, 1]) * PIXEL_NM
        if spec["hole"] == "square":
            true_cd = true_cd * 2.0 / np.sqrt(np.pi)

        def truth(pg, true_cd=true_cd, xy=page.centers_xy):
            idx = match(cp.asnumpy(pg.c), xy)
            used = cp.asnumpy(pg.use) & (idx >= 0)
            return float(true_cd[idx[used]].mean())

        label = name + (" " + ", ".join(f"{k} {v}" for k, v in over.items()) if over else "") + " (error of the page CD)"
        for family in families:
            rows.append(ends_of(image, family, label, truth))
    if args.get("out"):
        import json
        from pathlib import Path

        (Path(__file__).parent / "runs" / f"{args['out']}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
