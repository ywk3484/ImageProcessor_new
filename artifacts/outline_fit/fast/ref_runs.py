"""Scratch: reference fits of real pages by the existing outline fit (the same-numbers prototype of the earlier
session, ../speed/tier1.py: same procedure and stop rule as subpx._outline, 10-15 s per page instead of 104-150).

    python ref_runs.py pages=0:48 [blur=gauss2] [tag=ref]

Saves runs/ref/<tag>_pN.npz (seeds, centres, per-hole cd / delta / a / level / rms / good / at_clip, theta, summary)
and runs/ref/<tag>.json (one row per page). Pages without holes (fewer than 1000 seeds) are skipped.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # the repo root (this file sits in artifacts/outline_fit/fast)
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "speed"))
TIF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"
OUT = HERE / "runs" / "ref"


def load(page):
    import tifffile

    with tifffile.TiffFile(TIF) as t:
        return t.pages[page].asarray()


def parse_pages(text):
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(v) for v in text.split(",")]


def seeds_for(image):
    from subpx import detect_centers

    det = detect_centers(image, backend="cpu", threshold="otsu", invert=False, area_min=3, area_max=200, pad=3,
                         refine="logquad")
    return det.centers_xy


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    pages = parse_pages(args.get("pages", "0:48"))
    blur = args.get("blur", "gauss2")
    tag = args.get("tag", "ref")
    import tier1

    template_class = tier1.install(render="exact", gpu=True)
    import cupy as cp
    import subpx._outline.page as P
    from subpx.outline import BLUR_FAMILIES, UPSAMPLE

    cp.fft.rfft2(cp.zeros((64, 64)))
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pages:
        f = OUT / f"{tag}_p{p}.npz"
        if f.exists():
            continue
        raw = load(p)
        t0 = time.perf_counter()
        seeds = seeds_for(raw)
        t_seed = time.perf_counter() - t0
        if len(seeds) < 1000:
            print(f"page {p}: {len(seeds)} seeds - skipped", flush=True)
            rows.append(dict(page=p, holes=int(len(seeds)), skipped=True))
            continue
        image = raw.astype(np.float64)
        t0 = time.perf_counter()
        fitter = P.PageFit(image, seeds, pixel_nm=21.0, unit="nm",
                           template=template_class(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[blur], blur=blur), xp=cp)
        res = fitter.run(rounds=14, iterations=5)
        cp.cuda.get_current_stream().synchronize()
        dt = time.perf_counter() - t0
        d = res["describe"]
        good = res["good"]
        cd = res["cd_px"] * 21.0
        row = dict(page=p, seconds=dt, seed_seconds=t_seed, rounds=len(fitter.history), converged=bool(res["converged"]),
                   outline_cd_nm=d["cd_nm"], blur_x=d["blur_fwhm_nm"][0], blur_y=d["blur_fwhm_nm"][1],
                   p=d["superellipse"]["p"], cd_mean_nm=float(cd[good].mean()), cd_sd_nm=float(cd[good].std()),
                   holes=int(len(seeds)), good=int(good.sum()), at_clip=int(res["at_clip"].sum()),
                   spacing=fitter.spacing, window=fitter.R, render=fitter.Rr,
                   history=[dict(round=h["round"], cd=h["outline_cd"], chi2=h["chi2"], change=h["proposal_change"])
                            for h in fitter.history])
        rows.append(row)
        np.savez(f, seeds=seeds, centers=res["centers_xy"], cd_px=res["cd_px"], delta=res["edge_offset_px"],
                 a=res["brightness"], level=res["level"], rms=res["rms"], good=good, at_clip=res["at_clip"],
                 theta=res["theta"], outline_cd_px=res["cd_outline_px"], delta_mean=res["edge_offset_mean_px"],
                 var_line=np.asarray(res["var_line"]), stack=res["stack"], stack_weight=res["stack_weight"],
                 template=res["template"])
        print(f"page {p}: seeds {t_seed:.1f} s, fit {dt:5.1f} s, {row['rounds']} rounds, converged {row['converged']}, "
              f"outline CD {d['cd_nm']:.3f} nm, blur {d['blur_fwhm_nm'][0]:.2f}/{d['blur_fwhm_nm'][1]:.2f}, "
              f"p {row['p']:.3f}, per-hole sd {row['cd_sd_nm']:.3f}", flush=True)
        del fitter
        (OUT / f"{tag}.json").write_text(json.dumps(rows, indent=1))
    old = OUT / f"{tag}.json"
    if rows:
        old.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
