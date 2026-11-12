"""Scratch: collects the proposal page's numbers (runs/*.json) into runs/page_data.json and draws its picture
(runs/fig_average_hole.png): a piece of a test-file page, its average hole with the neighbours' light in it, and what
the one-pass page fit leaves of it.

    python make_page_data.py [picture=1]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SQ = 2.0 / math.sqrt(math.pi)  # a square of side s has the area-equivalent diameter s x 2 / sqrt(pi)

LABELS = {
    "circle100_square": "100 nm circles, square array",
    "circle80_square": "80 nm circles, square array",
    "circle100_stagger": "100 nm circles, staggered",
    "square150_stagger": "150 nm squares, staggered",
    "circle250_square": "250 nm circles, square array",
    "ellipse160x80_square": "160 x 80 nm ellipses",
    "ellipse100x200_stagger": "100 x 200 nm ellipses",
    "circle60_dense": "60 nm circles, dense",
    "circle50_dense": "50 nm circles, dense",
    "ellipse90x60_stagger": "90 x 60 nm ellipses, dense",
}


def load(tag):
    f = RUNS / f"{tag}.json"
    return json.loads(f.read_text()) if f.exists() else []


def key(r):
    s = r.get("settings", {})
    extra = tuple(sorted((k, str(v)) for k, v in s.items()
                         if k in ("rot", "part", "flicker", "pitch", "place", "cd", "bright", "blur")
                         and v not in (None, 0.0, 2.5, 1.0, 0.05, 65.0)))
    return (r["name"], r["family"], extra)


def existing_fixed(r):
    """The existing fit's scores with squares scored against the area-equivalent truth (as the one-pass runs are)."""
    d = dict(r["existing"])
    ph = dict(d["per_hole"])
    if r["name"].startswith("square"):
        true = ph["true_mean"] * SQ
        d["error"] = d["cd"] - true
        ph["slope"] = ph["slope"] / SQ
        ph["scatter"] = ph["scatter"] * SQ
    d["per_hole"] = ph
    return d


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    fast = {key(r): r["fast"] for tag in ("truth_main_v4", "truth_blur_v4", "truth_more_v4") for r in load(tag) if "fast" in r}
    exist = {key(r): existing_fixed(r) for tag in ("truth_main", "truth_more", "truth_extra_existing", "truth_part")
             for r in load(tag) if "existing" in r}
    truth = []
    for k in fast:
        name, family, extra = k
        row = dict(name=name, label=LABELS.get(name, name), family=family, extra=dict(extra),
                   one_pass=dict(error=fast[k]["error"], slope=fast[k]["per_hole"]["slope"],
                                 scatter=fast[k]["per_hole"]["scatter"], light10=fast[k]["per_hole"]["light10"],
                                 centres=fast[k]["per_hole"]["centre_sd"], blur=fast[k]["blur"], seconds=fast[k]["seconds"]))
        if k in exist:
            e = exist[k]
            row["existing"] = dict(error=e["error"], slope=e["per_hole"]["slope"], scatter=e["per_hole"]["scatter"],
                                   light10=e["per_hole"]["light10"], centres=e["per_hole"]["centre_sd"], blur=e["blur"],
                                   seconds=e["seconds"], rounds=e["rounds"], converged=e["converged"])
        truth.append(row)
    ref = {r["page"]: r for r in load("ref/ref") if not r.get("skipped") and r.get("holes", 0) > 30000}
    one = {r["page"]: r for r in load("file_l2_v4")}
    lens = {r["page"]: r for r in load("file_l2_lens_v4")}
    pages = sorted(one)
    file_rows = [dict(page=p, existing=ref[p]["outline_cd_nm"], one_pass=one[p]["cd"], one_pass_lens=lens.get(p, {}).get("cd"),
                      existing_rounds=ref[p]["rounds"], existing_converged=ref[p]["converged"],
                      existing_seconds=ref[p]["seconds"], hole_sd=one[p].get("hole_dcd_sd"), hole_slope=one[p].get("hole_slope"),
                      centre_sd=one[p].get("centre_sd")) for p in pages]
    ex = np.array([r["existing"] for r in file_rows])
    op = np.array([r["one_pass"] for r in file_rows])
    summary = dict(pages=len(pages), existing_mean=float(ex.mean()), existing_sd=float(ex.std()),
                   one_pass_mean=float(op.mean()), one_pass_sd=float(op.std()), diff_mean=float((op - ex).mean()),
                   diff_sd=float((op - ex).std()), diff_max=float(np.abs(op - ex).max()),
                   hole_sd=float(np.mean([r["hole_sd"] for r in file_rows])),
                   hole_slope=float(np.mean([r["hole_slope"] for r in file_rows])),
                   centre_sd=float(np.mean([r["centre_sd"] for r in file_rows])),
                   existing_seconds=float(np.mean([r["existing_seconds"] for r in file_rows])),
                   phase_ms={k: float(np.mean([one[p]["phase"][k] for p in pages]) * 1e3) for k in one[pages[0]]["phase"]})
    data = dict(truth=truth, file=file_rows, file_summary=summary)
    (RUNS / "page_data.json").write_text(json.dumps(data, indent=1, default=float))
    print(f"{len(truth)} truth rows ({sum('existing' in r for r in truth)} with the existing fit), {len(file_rows)} file pages")
    print(json.dumps(summary, indent=1))
    if args.get("picture", "1") == "1":
        picture()


def picture():
    import base64

    import cupy as cp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile

    from gpufit import ROOT, GpuPage
    from pipeline import fitter_for, lib_template, page_fits

    with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as t:
        image = t.pages[0].asarray()
    pg = GpuPage(image, tt="float", lib_template=lib_template("gauss2"))
    pg.build_stack(recentre=False)
    (theta, scale, const, fa), = page_fits([pg], "gauss2", stages=("round", "super"), max_iter=(10, 12), ftol=1e-6)
    pg.set_template(theta, scale, const)
    pg.fit_holes(sweeps=1, size=False)
    pg.restack(pg.good_mask(), scale)
    (theta, scale, const, fc), = page_fits([pg], "gauss2", theta0=[theta], stages=("full",), grid=False, max_iter=(30,))
    sf = fitter_for(pg, "gauss2")
    gamma = pg.pair_map(sf)
    y, w = pg.stack_data(sf)
    data = sf.prepare(gamma[None], y[None], w[None])
    M = cp.asnumpy(sf.synth(sf.template_hat(theta[None]), data["gamma"]))[0, 0]  # (U, U, np, np)
    alone = cp.asnumpy(sf.synth(sf.template_hat(theta[None]), cp.ones_like(data["gamma"])))[0, 0]  # the hole without neighbours
    # back from the class layout to the bin grid
    nb = 2 * sf.nmax + 1
    grid_m = np.full((nb, nb), np.nan)
    grid_a = np.full((nb, nb), np.nan)
    for qy in range(sf.U):
        for qx in range(sf.U):
            iy, ix = sf.idx[qy][sf.pad[qy]], sf.idx[qx][sf.pad[qx]]
            grid_m[np.ix_(iy, ix)] = M[qy, qx][np.ix_(np.flatnonzero(sf.pad[qy]), np.flatnonzero(sf.pad[qx]))]
            grid_a[np.ix_(iy, ix)] = alone[qy, qx][np.ix_(np.flatnonzero(sf.pad[qy]), np.flatnonzero(sf.pad[qx]))]
    ms = pg.Ms
    lo = ms // 2 - sf.nmax
    stack = cp.asnumpy(pg.stack)[lo:lo + nb, lo:lo + nb]
    count = cp.asnumpy(pg.wsum)[lo:lo + nb, lo:lo + nb]
    model = scale * grid_m + const
    left = stack - model
    noise_sd = 7.7  # grey levels per pixel on this file (the existing fit's own reading: variance 59.4)
    expected = noise_sd / np.sqrt(np.maximum(count, 1))
    print(f"picture: stack {stack.min():.1f}..{stack.max():.1f}, leftover rms {np.sqrt(np.nanmean(left ** 2)):.3f} grey levels, "
          f"noise alone {np.sqrt(np.mean(expected ** 2)):.3f}; holes per bin {np.median(count):.0f}")
    ext = sf.nmax / sf.U
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), dpi=100)
    ax = axes.ravel()
    y0, x0 = 300, 500
    crop = image[y0:y0 + 44, x0:x0 + 58]
    ax[0].imshow(crop, cmap="gray", vmin=30, vmax=230, interpolation="nearest")
    c = cp.asnumpy(pg.c)
    near = c[(c[:, 0] > x0 + 20) & (c[:, 0] < x0 + 38) & (c[:, 1] > y0 + 14) & (c[:, 1] < y0 + 30)]
    cx, cy = np.rint(near[0])
    R = pg.R
    ax[0].add_patch(plt.Rectangle((cx - x0 - R - 0.5, cy - y0 - R - 0.5), 2 * R + 1, 2 * R + 1, fill=False, ec="#eb6834", lw=2.2))
    ax[0].set_title("(a) piece of a page, one hole's window", fontsize=13)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    im = ax[1].imshow(stack, cmap="gray", extent=(-ext, ext, ext, -ext), interpolation="nearest")
    ax[1].set_title(f"(b) average of {int(pg.use.sum()):,} windows", fontsize=13)
    im2 = ax[2].imshow(scale * grid_a + const, cmap="gray", extent=(-ext, ext, ext, -ext), interpolation="nearest",
                       vmin=stack.min(), vmax=stack.max())
    ax[2].set_title("(c) fitted hole alone: outline x blur", fontsize=13)
    t = pg.t
    ang = np.linspace(-np.pi, np.pi, 361)
    r, _ = t._radius_at(theta, ang)
    ax[2].plot(r * np.cos(ang), r * np.sin(ang), color="#eb6834", lw=2.0)
    im3 = ax[3].imshow(left / expected, cmap="RdBu_r", vmin=-4, vmax=4, extent=(-ext, ext, ext, -ext), interpolation="nearest")
    ax[3].set_title("(d) average - model, in units of the noise", fontsize=13)
    for a_ in ax[1:]:
        a_.set_xlabel("px from the hole's centre")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03, label="grey levels")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.03, label="grey levels")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.03, label="noise sd")
    fig.tight_layout()
    out = RUNS / "fig_average_hole.png"
    fig.savefig(out, facecolor="white")
    print("wrote", out, out.stat().st_size, "bytes; base64", len(base64.b64encode(out.read_bytes())))
    d = pg.t.describe(theta, 21.0)
    (RUNS / "fig_average_hole.json").write_text(json.dumps(dict(
        holes=int(pg.use.sum()), per_bin=float(np.median(count)), leftover_rms=float(np.sqrt(np.nanmean(left ** 2))),
        noise_alone=float(np.sqrt(np.mean(expected ** 2))), cd=d["cd_nm"], blur=[float(v) for v in d["blur_fwhm_nm"]],
        window=int(2 * R + 1), bins_per_px=int(sf.U), spacing=pg.spacing), indent=1))


if __name__ == "__main__":
    main()
