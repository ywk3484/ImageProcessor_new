"""Check outline_fit_fast's page CD on a folder of PNG images: does it depend on where its final fit starts or stops?

    python scripts/outline_fast_check.py FOLDER --pixel-size 21 --unit nm

Use the options the report was run with (--outline-blur, --seeds, --invert). GPU only. Nothing in the library is
changed by this script; it measures.

Why: outline_fit_fast fits one outline and one blur per image in two stages. The second (final) stage starts from
where the first stage ended and runs for at most 30 iterations. On generated images of the same holes the first
stage leaves the blur's second part (gauss2: the second Gaussian; lens: the pupil's cut-off) at a different place
from image to image, and from some of those starts the final fit ends at another, worse-fitting outline: the page CD
of repeated images then falls on two values (0.9 nm apart in one generated case) although nothing about the holes
changed. This script shows whether that is what happens on the images given.

Per image the one-pass fit is run exactly as the reports run it (measure_cds: the same seeds, the same steps); then
its final stage is repeated (a) from the same start without the cap, and (b) from several other starts of the blur's
second part (gauss2: 9, lens: 4), each run to its end with the fit's own rule for an exchanged gauss2 pair:

    rep    the page CD the report shows (one start, at most 30 iterations)
    end    the same start run to its end (--cap iterations at most)
    best   the end with the lowest leftover of all starts
    ends   how many different ends the starts reach (page CDs more than --same apart)

One line per image, the details of every start in a JSON file (--output), and as the last line the folder:

    check N=8 | rep 97.13-98.89 3s 1.82 nc 8 | end 3s 1.80 it<=212 | best 98.02-98.11 3s 0.09 moved 5 | ends 1-3 gap 344 | left 1.3-2.0 | rblur -0.93

    rep A-B 3s X nc K     lowest and highest page CD of the images, 3 x their standard deviation (N - 1), and how
                          many images stopped at the cap of 30 iterations ("NOT converged" in the report)
    end 3s X it<=M        the same 3 s with the same start run to its end; M = the most iterations an image needed
    best A-B 3s X moved K the best ends; K images whose best end is more than --same from the reported CD
    ends a-b gap G        fewest and most ends per image; G = how much worse the next end fits than the best one, in
                          chi-square units, smallest over the images with two ends (a few units = the two fit alike)
    left a-b              leftover of the best end (chi-square per bin: 1 = the average hole is fitted to its noise)
    rblur R               correlation over the images between the best page CD and its fitted blur width

How to read it (each case was planted on generated images; docs/outline_fit.md):

    best 3s far below rep 3s, moved > 0   the reported CD depends on the final fit's start: the fit should start
                                          from several points and keep the best end
    end 3s far below rep 3s               the cap of 30 iterations cuts the fit short: raise it
    rep = end = best, one end             the fit is at its end; a spread between images is in the images or in what
                                          the model cannot describe (left far above 1: the model does not fit the
                                          average hole)
    two ends with a gap of a few units    two outlines fit these images equally well: the model, not the procedure
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_CAP = 30  # the library's cap on the final fit (subpx._gpu.outline_onepass.FINAL_ITERATIONS)
GAUSS2_STARTS = tuple(itertools.product((1.3, 2.0, 3.0), (0.05, 0.2, 0.45)))  # second width / first, second's weight
LENS_STARTS = (0.35, 0.5, 0.9, 3.0)  # pupil cut-off, cycles per pixel


@contextmanager
def tapped_seeds(box: dict):
    """While active, ``fit_outline_fast`` records the seeds ``measure_cds`` hands it (its full seed list, with the
    holes cut by the frame, is not part of the result) and runs unchanged."""
    import subpx.outline as outline

    original = outline.fit_outline_fast

    def tapped(image, seeds_xy=None, **kw):
        box["seeds"] = None if seeds_xy is None else np.array(seeds_xy, dtype=np.float64)
        return original(image, seeds_xy, **kw)

    outline.fit_outline_fast = tapped
    try:
        yield
    finally:
        outline.fit_outline_fast = original


def other_starts(template, theta0, blur):
    """Starts for the final fit: the first fit's result with the blur's second part put to standard places.
    Returns (starts (K, P), labels)."""
    j = template.n_shape
    lo, hi = template.bounds()
    starts, labels = [], []
    if blur == "gauss2":
        for ratio, weight in GAUSS2_STARTS:
            theta = np.asarray(theta0, dtype=float).copy()
            theta[j + 2], theta[j + 3] = theta[j] + math.log(ratio), theta[j + 1] + math.log(ratio)
            theta[j + 4] = math.log(weight / (1 - weight))
            starts.append(theta)
            labels.append(f"second Gaussian {ratio:g} x as wide, {100 * weight:.0f} %")
    else:
        for cutoff in LENS_STARTS:
            theta = np.asarray(theta0, dtype=float).copy()
            theta[j + 2] = math.log(cutoff)
            starts.append(theta)
            labels.append(f"pupil cut-off {cutoff:g}")
    return np.clip(np.array(starts), lo + 1e-6, hi - 1e-6), labels


def fit_from(page, blur, starts, cap):
    """The final stage of one image from every start (in lockstep), then the library's rule for gauss2 fits that
    end with the two Gaussians exchanged (fitted again from the exchanged labels; the lower cost is kept).
    Returns theta (K, P), cost (K,), stopped by its tolerance (K,), iterations of the slowest start, bins."""
    import cupy as cp

    from subpx._gpu.outline_onepass import fitter_for
    from subpx._outline.onepass import exchanged, second_is_narrower

    fitter = fitter_for(page, blur)
    gamma = page.pair_map(fitter)
    y, w = page.stack_data(fitter)

    def run(points):
        k = len(points)
        copies = [cp.ascontiguousarray(cp.broadcast_to(a, (k,) + a.shape)) for a in (gamma, y, w)]
        theta, _, _, cost, info = fitter.fit(np.asarray(points, dtype=float), *copies, stages=("full",), grid=False,
                                             max_iter=(int(cap),))
        return theta.copy(), np.array(cost, dtype=float), np.array(info[-1][2], dtype=bool), int(info[-1][1])

    theta, cost, stopped, iterations = run(starts)
    if blur == "gauss2":
        narrow = [i for i in range(len(theta)) if second_is_narrower(page.t, theta[i])]
        if narrow:
            again, cost2, stopped2, iterations2 = run([exchanged(page.t, theta[i]) for i in narrow])
            iterations = max(iterations, iterations2)
            for n, i in enumerate(narrow):
                if cost2[n] < cost[i]:
                    theta[i], cost[i], stopped[i] = again[n], cost2[n], stopped2[n]
    bins = int(cp.asnumpy((w > 0).sum()))
    return theta, cost, stopped, iterations, bins


def first_stage(signal, seeds, blur):
    """Steps A and B of the one-pass fit for one image, as ``subpx._gpu.outline_onepass.fit_images`` runs them:
    the page with its stack rebuilt from the centred holes, and the first fit's theta (the final fit's start)."""
    from subpx._gpu import outline_onepass as onepass

    page = onepass.GpuPage(signal, blur=blur, seeds=seeds)
    page.build_stack()
    theta, scale, const, _ = onepass.page_fits([page], blur, stages=onepass.FIRST_STAGES,
                                               max_iter=onepass.FIRST_ITERATIONS, ftol=onepass.FIRST_FTOL)[0]
    page.set_template(theta, scale, const)
    page.fit_holes(sweeps=onepass.CENTRING_SWEEPS, size=False)
    page.restack(page.good_mask(), scale)
    return page, theta


def group_ends(cds, costs, same):
    """Ends of the starts grouped by page CD (within ``same`` of the neighbour in sorted order): per group the mean
    CD, its lowest cost and how many starts reached it; lowest cost first."""
    order = np.argsort(cds)
    groups = []
    for i in order:
        if groups and cds[i] - groups[-1]["last"] <= same:
            g = groups[-1]
            g["members"].append(int(i))
            g["last"] = float(cds[i])
        else:
            groups.append({"members": [int(i)], "last": float(cds[i])})
    out = [{"cd": float(np.mean(cds[g["members"]])), "cost": float(np.min(costs[g["members"]])),
            "starts": len(g["members"]), "best_start": int(g["members"][int(np.argmin(costs[g["members"]]))])}
           for g in groups]
    return sorted(out, key=lambda g: g["cost"])


def check_image(image, options, blur, cap, same, device=0):
    """One image: the report's reading and the final fit from the other starts. Lengths in ``options['unit']``."""
    import cupy as cp

    from subpx import measure_cds
    from subpx._gpu._cuda import release_memory
    from subpx.backends import gpu_device
    from subpx.outline import blur_second

    scale = float(options["pixel_size"])
    box = {}
    with tapped_seeds(box):
        result = measure_cds(image, methods=("outline_fit_fast",), outline_blur=blur, backend="gpu", device=device,
                             **options)
    meta = result.meta["outline_fit_fast"]
    if not meta.get("fitted"):
        return {"fitted": False, "reason": meta.get("reason", "")}
    signal = np.asarray(image).astype(np.float64)
    if options.get("invert"):
        signal = float(signal.max()) - signal
    with gpu_device(device):
        try:
            page, theta0 = first_stage(signal, box["seeds"], blur)
            template = page.t
            own_theta, own_cost, own_stopped, own_iterations, bins = fit_from(page, blur, theta0[None], cap)
            starts, labels = other_starts(template, theta0, blur)
            theta, cost, stopped, _, _ = fit_from(page, blur, starts, cap)
        finally:
            cp.get_default_memory_pool().free_all_blocks()
            release_memory()
    theta = np.concatenate([own_theta, theta])
    cost = np.concatenate([own_cost, cost])
    stopped = np.concatenate([own_stopped, stopped])
    labels = ["as the report starts"] + labels
    cds = np.array([template.cd_px(t) for t in theta]) * scale
    ends = group_ends(cds, cost, same)
    best = ends[0]
    best_theta = theta[best["best_start"]]
    fwhm = float(np.mean(template.blur_fwhm_px(best_theta))) * scale
    return {
        "fitted": True, "holes": int(meta["holes_fitted"]), "stacked": int(meta["holes_stacked"]), "bins": bins,
        "rep": float(meta["outline_cd_px"]) * scale, "rep_converged": bool(meta["converged"]),
        "rep_iterations": int(meta["fit_iterations"]), "rep_leftover": float(meta["leftover_chi2"]),
        "rep_second": blur_second(meta["theta"], blur), "first": float(meta["first_cd_px"]) * scale,
        "end": float(cds[0]), "end_iterations": own_iterations, "end_stopped": bool(own_stopped[0]),
        "end_leftover": float(cost[0]) / bins,
        "best": best["cd"], "best_leftover": best["cost"] / bins, "best_blur_fwhm": fwhm,
        "best_second": blur_second(best_theta, blur),
        "ends": [{"cd": g["cd"], "leftover": g["cost"] / bins, "cost": g["cost"], "starts": g["starts"],
                  "second": blur_second(theta[g["best_start"]], blur)} for g in ends],
        "gap": float(ends[1]["cost"] - ends[0]["cost"]) if len(ends) > 1 else None,
        "starts": [{"start": label, "cd": float(c), "cost": float(k), "stopped": bool(s),
                    "theta": [float(v) for v in t]} for label, c, k, s, t in zip(labels, cds, cost, stopped, theta)],
        "unit": options["unit"],
    }


def image_line(name, r, digits):
    """One ASCII line on an image's check."""
    if not r["fitted"]:
        return f"{name}: not fitted ({r['reason']})"
    u = r["unit"]
    state = "converged" if r["rep_converged"] else "NOT converged"
    ends = ", ".join(f"{g['cd']:.{digits}f} from {g['starts']} (leftover {g['leftover']:.2f})" for g in r["ends"])
    return (f"{name}: rep {r['rep']:.{digits}f} {u} ({state} in {r['rep_iterations']} it, leftover {r['rep_leftover']:.2f}) | "
            f"end {r['end']:.{digits}f} in {r['end_iterations']} it | best {r['best']:.{digits}f} "
            f"(leftover {r['best_leftover']:.2f}; blur {r['best_blur_fwhm']:.1f} {u}; {r['best_second']}) | "
            f"{len(r['ends'])} end{'s' if len(r['ends']) > 1 else ''} of {len(r['starts'])} starts: {ends}")


def folder_line(results, same, digits):
    """The folder's line: short, ASCII, to be typed back (see the module docstring for its fields)."""
    done = [r for r in results if r.get("fitted")]
    if not done:
        return "check N=0: no image was fitted"

    def spread(key):
        v = np.array([r[key] for r in done])
        return v, (3 * float(v.std(ddof=1)) if len(v) > 1 else float("nan"))

    rep, rep3 = spread("rep")
    _, end3 = spread("end")
    best, best3 = spread("best")
    gaps = [r["gap"] for r in done if r["gap"] is not None]
    counts = [len(r["ends"]) for r in done]
    left = [r["best_leftover"] for r in done]
    blur = np.array([r["best_blur_fwhm"] for r in done])
    rblur = float(np.corrcoef(best, blur)[0, 1]) if len(done) > 2 and np.ptp(best) > 0 and np.ptp(blur) > 0 else float("nan")

    def f(value, d=digits):
        return "-" if not np.isfinite(value) else f"{value:.{d}f}"

    return (f"check N={len(done)} | rep {f(rep.min())}-{f(rep.max())} 3s {f(rep3)} nc {sum(not r['rep_converged'] for r in done)} | "
            f"end 3s {f(end3)} it<={max(r['end_iterations'] for r in done)} | best {f(best.min())}-{f(best.max())} 3s {f(best3)} "
            f"moved {sum(abs(r['best'] - r['rep']) > same for r in done)} | ends {min(counts)}-{max(counts)} "
            f"gap {f(min(gaps), 0) if gaps else '-'} | left {f(min(left), 1)}-{f(max(left), 1)} | rblur {f(rblur, 2)}")


def main(argv=None) -> int:
    from scripts.analyze_contact_holes import load_image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the .png images (the one given to the report)")
    parser.add_argument("--output", type=Path, help="JSON file for the details of every start (default: under artifacts/)")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--outline-blur", choices=("gauss2", "lens"), default="gauss2")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cap", type=int, default=300, help="Most iterations of a final fit run 'to its end'")
    parser.add_argument("--same", type=float, default=None,
                        help="Page CDs closer than this are the same end (default: 0.001 px in --unit)")
    args = parser.parse_args(argv)
    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".png") if args.folder.is_dir() else []
    if not images:
        parser.error(f"No .png images in {args.folder}")
    from subpx.outline import _gpu_usable

    if not _gpu_usable():
        parser.error("this check needs the GPU (CuPy): outline_fit_fast has no CPU path")
    same = float(args.same) if args.same is not None else 0.001 * args.pixel_size
    digits = 2 if args.pixel_size > 1.5 else 3
    options = dict(seeds=args.seeds, invert=args.invert, area_min=args.area_min, area_max=args.area_max,
                   pixel_size=args.pixel_size, unit=args.unit)
    output = args.output or ROOT / "artifacts" / time.strftime("outline_fast_check_%Y%m%d_%H%M%S.json")
    results, started = [], time.perf_counter()
    for number, path in enumerate(images, 1):
        start = time.perf_counter()
        try:
            r = check_image(load_image(path), options, args.outline_blur, args.cap, same, args.device)
        except Exception as error:  # one bad image must not stop the folder
            r = {"fitted": False, "reason": str(error).replace(str(path), path.name) or type(error).__name__}
        r.update(image=path.name, seconds=time.perf_counter() - start)
        results.append(r)
        print(f"[{number}/{len(images)}] {image_line(path.name, r, digits)} ({r['seconds']:.1f} s)", flush=True)
    line = folder_line(results, same, digits)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"folder": str(args.folder.resolve()), "blur": args.outline_blur, "cap": args.cap,
                                  "same": same, "options": options, "line": line, "images": results}, indent=1),
                      encoding="utf-8")
    print(f"Details: {output} ({time.perf_counter() - started:.0f} s)", flush=True)
    print(line, flush=True)  # the last line: the one to type back from another machine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
