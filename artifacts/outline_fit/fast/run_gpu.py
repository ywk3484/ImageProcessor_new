"""Scratch: the GPU one-pass fit on pages of the test file, against the existing fit's saved results (runs/ref).

    python run_gpu.py pages=0,24 [blur=gauss2] [sweeps=2] [seeds=own|ref] [weighting=uniform|a2] [tt=double|float]
                      [holes=1] [repeat=1] [verbose=1]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import tifffile

from gpufit import GpuPage, make_fitter, sync, ROOT

TIF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"


def one_page(image, args, sf_cache, seeds=None):
    """(page object, result dict, seconds by stage)"""
    blur = args.get("blur", "gauss2")
    t_all = time.perf_counter()
    pg = GpuPage(image, blur=blur, U=int(args.get("U", 4)), f_max=float(args.get("fmax", 0.6)), seeds=seeds,
                 weighting=args.get("weighting", "uniform"), tt=args.get("tt", "double"), lib_template=sf_cache.get("t"))
    sf_cache.setdefault("t", pg.t)
    pg.build_stack(recentre=args.get("recentre", "1") == "1")
    key = (blur, pg.R)
    if key not in sf_cache:
        sf_cache[key] = make_fitter(blur, pg.R, U=pg.U, f_max=pg.f_max, lib_template=pg.t)
    sf = sf_cache[key]
    gamma = pg.pair_map(sf)
    t0 = time.perf_counter()
    y, w = pg.stack_data(sf)
    theta, scale, const, cost, info = sf.fit(pg.start_theta()[None], gamma[None], y[None], w[None])
    sync()
    pg.times["shape"] = time.perf_counter() - t0
    d = pg.t.describe(theta[0], 21.0)
    out = dict(cd_nm=d["cd_nm"], blur=[float(v) for v in d["blur_fwhm_nm"]], p=d["superellipse"]["p"], scale=float(scale[0]),
               const=float(const[0]), chi2=float(cost[0]), bins=int((w > 0).sum()), stages=info, holes=pg.n,
               used=int(pg.use.sum()), spacing=pg.spacing, R=pg.R)
    res = None
    if args.get("holes", "1") == "1":
        pg.set_template(theta[0], float(scale[0]), float(const[0]))
        out["sweeps"] = pg.fit_holes(sweeps=int(args.get("sweeps", 2)))
        res = pg.results()
        g = res["good"] & ~res["at_clip"]
        out.update(cd_mean_nm=float(res["cd_px"][g].mean() * 21.0), cd_sd_nm=float(res["cd_px"][g].std() * 21.0),
                   delta_mean_nm=res["delta_mean"] * 21.0, good=int(g.sum()), Rr=pg.Rr)
    sync()
    out["total_s"] = time.perf_counter() - t_all
    return pg, out, res, theta[0]


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    pages = [int(v) for v in args.get("pages", "0").split(",")]
    sf_cache = {}
    rows = []
    with tifffile.TiffFile(TIF) as t:
        images = {p: t.pages[p].asarray() for p in pages}
    for rep in range(int(args.get("repeat", 1))):
        for p in pages:
            ref = np.load(Path("runs/ref") / f"ref_p{p}.npz")
            seeds = ref["seeds"] if args.get("seeds", "own") == "ref" else None
            pg, out, res, theta = one_page(images[p], args, sf_cache, seeds)
            ref_cd = float(ref["outline_cd_px"]) * 21.0
            line = (f"page {p}: {out['holes']:,} holes ({out['used']:,} stacked), outline CD {out['cd_nm']:.3f} nm (existing {ref_cd:.3f}, "
                    f"diff {out['cd_nm'] - ref_cd:+.3f}), blur {out['blur'][0]:.2f}/{out['blur'][1]:.2f}, p {out['p']:.3f}, "
                    f"chi2 {out['chi2']:.0f}/{out['bins']}, {out['stages']}")
            if res is not None:
                # per-hole comparison with the existing fit: match by position
                from scipy.spatial import cKDTree

                d, j = cKDTree(ref["centers"]).query(res["centers_xy"], distance_upper_bound=1.5)
                ok = np.isfinite(d) & res["good"] & ~res["at_clip"]
                ok &= ref["good"][np.clip(j, 0, len(ref["good"]) - 1)]
                jj = j[ok]
                dcd = (res["cd_px"][ok] - ref["cd_px"][jj]) * 21.0
                dc = (res["centers_xy"][ok] - ref["centers"][jj]) * 21.0
                slope = np.polyfit(ref["cd_px"][jj], res["cd_px"][ok], 1)[0]
                line += (f"\n      per hole vs existing ({ok.sum():,} holes): CD mean {dcd.mean():+.3f}, sd {dcd.std():.3f} nm, slope {slope:.4f}, "
                         f"centres sd {dc[:, 0].std():.3f}/{dc[:, 1].std():.3f} nm; per-hole CD sd {out['cd_sd_nm']:.3f} nm; "
                         f"mean edge offset {out['delta_mean_nm']:+.3f} nm; sweeps {[(s['sweep'], round(s['move99'], 4), round(s['delta99'], 4)) for s in out['sweeps']]}")
            tm = pg.times
            line += f"\n      {out['total_s'] * 1e3:.0f} ms: " + ", ".join(f"{k} {v * 1e3:.1f}" for k, v in tm.items())
            if rep == int(args.get("repeat", 1)) - 1 or args.get("verbose", "0") == "1":
                print(line, flush=True)
            rows.append(dict(page=p, rep=rep, **{k: v for k, v in out.items() if k != "sweeps"}, times=tm))
            del pg
    if "out" in args:
        Path("runs").mkdir(exist_ok=True)
        (Path("runs") / f"{args['out']}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
