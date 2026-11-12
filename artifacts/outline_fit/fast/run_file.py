"""Scratch: the one-pass fit over the hole pages of the test file, in batches, against the existing fit (runs/ref).

    python run_file.py [pages=all|0,1,2] [batch=16] [level=2] [family=gauss2] [sweeps_b=2] [sweeps_d=3] [relax=1.3] [tt=float]
                       [repeat=2] [out=TAG]

Prints, per page, the outline CD against the existing fit's, and (level 2) the per-hole CDs against the existing
fit's; then the time per page of the last repeat (pages already in memory: reading the file is not part of it).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np
import tifffile

from gpufit import ROOT, sync
from pipeline import fit_pages

TIF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"
REF = Path(__file__).parent / "runs" / "ref"


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    ref_rows = {r["page"]: r for r in json.loads((REF / "ref.json").read_text()) if not r.get("skipped") and r["holes"] > 30000}
    pages = sorted(ref_rows) if args.get("pages", "all") == "all" else [int(v) for v in args["pages"].split(",")]
    batch = int(args.get("batch", 16))
    level = int(args.get("level", 2))
    family = args.get("family", "gauss2")
    with tifffile.TiffFile(TIF) as t:
        images = {p: t.pages[p].asarray() for p in pages}
    kw = dict(family=family, level=level, sweeps_b=int(args.get("sweeps_b", 2)), sweeps_d=int(args.get("sweeps_d", 3)),
              tt=args.get("tt", "float"), relax=float(args.get("relax", 1.3)))
    rows = []
    for rep in range(int(args.get("repeat", 2))):
        rows = []
        sync()
        t0 = time.perf_counter()
        for s in range(0, len(pages), batch):
            chunk = pages[s:s + batch]
            for p, (pg, out, res) in zip(chunk, fit_pages([images[p] for p in chunk], **kw)):
                row = dict(page=p, phase=out["phase"], cd=out["cd"], first_cd=out["first"]["cd"], blur=out["blur"], p=out["p"], chi2=out["chi2"],
                           bins=out["bins"], holes=out["holes"], stacked=out["stacked"], times=out["times"],
                           stages=str(out["first"]["stages"]) + " + " + str(out["stages"]))
                if res is not None:
                    row.update(cd_mean=out["cd_mean"], cd_sd=out["cd_sd"], good=out["good"], at_clip=out["at_clip"])
                    if rep == int(args.get("repeat", 2)) - 1 and (REF / f"ref_p{p}.npz").exists():
                        from scipy.spatial import cKDTree

                        ref = np.load(REF / f"ref_p{p}.npz")
                        d, j = cKDTree(ref["centers"]).query(res["centers_xy"], distance_upper_bound=1.5)
                        ok = np.isfinite(d) & res["good"] & ~res["at_clip"]
                        ok &= ref["good"][np.clip(j, 0, len(ref["good"]) - 1)]
                        jj = j[ok]
                        dcd = (res["cd_px"][ok] - ref["cd_px"][jj]) * 21.0
                        dc = (res["centers_xy"][ok] - ref["centers"][jj]) * 21.0
                        row.update(hole_n=int(ok.sum()), hole_dcd_mean=float(dcd.mean()), hole_dcd_sd=float(dcd.std()),
                                   hole_slope=float(np.polyfit(ref["cd_px"][jj], res["cd_px"][ok], 1)[0]),
                                   centre_sd=float(np.hypot(dc[:, 0].std(), dc[:, 1].std()) / np.sqrt(2)))
                rows.append(row)
                del pg
        sync()
        total = time.perf_counter() - t0
        print(f"repeat {rep + 1}: {len(pages)} pages in {total:.2f} s = {total / len(pages) * 1e3:.0f} ms per page "
              f"(batches of {batch}, level {level}, {family})", flush=True)
    for r in rows:
        ref_cd = ref_rows[r["page"]]["outline_cd_nm"]
        if args.get("lines", "1") != "1":
            break
        line = (f"page {r['page']:2d}: outline CD {r['cd']:.3f} (existing {ref_cd:.3f}, diff {r['cd'] - ref_cd:+.3f}; first fit "
                f"{r['first_cd'] - ref_cd:+.3f}), blur {r['blur'][0]:.2f}/{r['blur'][1]:.2f}, p {r['p']:.3f}, chi2 {r['chi2']:.0f}/{r['bins']}")
        if "hole_n" in r:
            line += (f"; per hole vs existing: mean {r['hole_dcd_mean']:+.3f}, sd {r['hole_dcd_sd']:.3f} nm, slope {r['hole_slope']:.4f}, "
                     f"centres {r['centre_sd']:.3f} nm, sd {r['cd_sd']:.3f}")
        print(line, flush=True)
    if rows and "hole_n" in rows[0]:
        print(f"per hole vs existing over the pages: sd of the difference {np.mean([r['hole_dcd_sd'] for r in rows]):.3f} nm, slope "
              f"{np.mean([r['hole_slope'] for r in rows]):.4f}, centres {np.mean([r['centre_sd'] for r in rows]):.3f} nm")
    cd = np.array([r["cd"] for r in rows])
    ref = np.array([ref_rows[r["page"]]["outline_cd_nm"] for r in rows])
    print(f"\npage CD over {len(rows)} pages: one pass mean {cd.mean():.3f} sd {cd.std():.3f} nm; existing mean {ref.mean():.3f} sd "
          f"{ref.std():.3f} nm; difference mean {np.mean(cd - ref):+.3f} sd {np.std(cd - ref):.3f}, largest {np.abs(cd - ref).max():.3f} nm; "
          f"correlation {np.corrcoef(cd, ref)[0, 1]:.2f}")
    keys = list(rows[0]["times"].keys())
    print("ms per page by stage: " + ", ".join(f"{k} {np.mean([r['times'].get(k, 0) for r in rows]) * 1e3:.1f}" for k in keys)
          + f"  (sum {sum(np.mean([r['times'].get(k, 0) for r in rows]) for k in keys) * 1e3:.0f})")
    print("ms per page by phase: " + ", ".join(f"{k} {np.mean([r['phase'][k] for r in rows]) * 1e3:.1f}" for k in rows[0]["phase"]))
    if args.get("quiet", "0") == "1":
        return
    if "out" in args:
        (Path(__file__).parent / "runs" / f"{args['out']}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
