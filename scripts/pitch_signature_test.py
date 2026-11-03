"""Does each center method see a 1 nm row shift? Synthetic hole arrays with known centers, every CD method's centers
and the GPU center finders, the y pitch averaged over 1 to all columns, and an HTML report.

    python scripts/pitch_signature_test.py
    python scripts/pitch_signature_test.py --images circle100_square square300_square --output artifacts/ps_quick
    python scripts/pitch_signature_test.py --placement-sigma 0 --seed 2

Each test image is a page of the user's size (2048 x 1378 px, 21 nm pixels, 8-bit) rendered by
``subpx.synthetic_arrays.render_hole_array``: a square or staggered array of circles, squares or ellipses with
random CD (0.5 nm sd), placement (0.2 nm sd) and brightness (5 %) per hole, and rows about 1 um apart, but at least
5 rows apart, moved 1 nm in +y. Every method's centers are matched to the true holes; holes within one pitch of the
page edge are left out. Per method (y only):

- found: share of holes with a center;
- center error: sd of (measured - true) y, nm;
- pixel-position error: half the peak-to-peak of the mean y error against where the hole sits inside its pixel
  (16 bins), nm. With rows parallel to the pixel rows a whole row shares it, so it never averages away;
- y pitch per row interval, averaged like ``analyze_pitch`` (mean of the (r, k)-(r + 1, k) pairs of a band of N
  logical columns), for every disjoint band: curve noise against the true centers (the method alone) and against the
  design pitch + planted shift (what one would see: the mask's own placement noise included);
- spike: least-squares size of the planted pattern in the all-columns curve (true centers give about 1 nm);
- columns needed: fewest N whose curve noise is at most 1/3 of the planted shift.

Writes results.json, pitch_signature.csv, settings.json and report.html (scripts/pitch_signature_view.py) into the
output folder (default artifacts/pitch_signature).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx.synthetic_arrays import render_hole_array  # noqa: E402

PIXEL_NM = 21.0
# Ordered by shape, then size. pitch_nm = (x pitch, row pitch); staggered rows are offset by half the x pitch.
PATTERNS = {
    "circle250_square": dict(label="250 nm circles, square array", hole="circle", size_nm=(250.0, 250.0),
                             pitch_nm=(500.0, 500.0), layout="square"),
    "circle100_square": dict(label="100 nm circles, square array", hole="circle", size_nm=(100.0, 100.0),
                             pitch_nm=(200.0, 200.0), layout="square"),
    "circle100_stagger": dict(label="100 nm circles, staggered", hole="circle", size_nm=(100.0, 100.0),
                              pitch_nm=(200.0, 175.0), layout="staggered"),
    "circle80_square": dict(label="80 nm circles, square array", hole="circle", size_nm=(80.0, 80.0),
                            pitch_nm=(160.0, 160.0), layout="square"),
    "circle60_dense": dict(label="60 nm circles, dense staggered", hole="circle", size_nm=(60.0, 60.0),
                           pitch_nm=(140.0, 121.0), layout="staggered"),
    "circle50_dense": dict(label="50 nm circles, dense staggered", hole="circle", size_nm=(50.0, 50.0),
                           pitch_nm=(120.0, 104.0), layout="staggered"),
    "square300_square": dict(label="300 nm squares, square array", hole="square", size_nm=(300.0, 300.0),
                             pitch_nm=(600.0, 600.0), layout="square"),
    "square150_stagger": dict(label="150 nm squares, staggered", hole="square", size_nm=(150.0, 150.0),
                              pitch_nm=(300.0, 300.0), layout="staggered"),
    "ellipse160x80_square": dict(label="160 x 80 nm ellipses (wide), square array", hole="ellipse",
                                 size_nm=(160.0, 80.0), pitch_nm=(320.0, 200.0), layout="square"),
    "ellipse100x200_stagger": dict(label="100 x 200 nm ellipses (tall), staggered", hole="ellipse",
                                   size_nm=(100.0, 200.0), pitch_nm=(300.0, 300.0), layout="staggered"),
    "ellipse90x60_stagger": dict(label="90 x 60 nm ellipses (wide), staggered", hole="ellipse",
                                 size_nm=(90.0, 60.0), pitch_nm=(180.0, 120.0), layout="staggered"),
    "ellipse60x100_square": dict(label="60 x 100 nm ellipses (tall), square array", hole="ellipse",
                                 size_nm=(60.0, 100.0), pitch_nm=(140.0, 180.0), layout="square"),
    "ellipse70x45_dense": dict(label="70 x 45 nm ellipses (wide), dense staggered", hole="ellipse",
                               size_nm=(70.0, 45.0), pitch_nm=(140.0, 100.0), layout="staggered"),
}
COLUMNS = (1, 2, 4, 8, 16, 32, 64, 128)  # plus "all"
PHASE_BINS = 16
# Short names (the workbench's words; subpx.workbench.names needs Qt).
NAMES = {
    "gaussian": "Gaussian fit", "logquad": "Log fit", "gradient": "Gradient edge", "halfmax": "Half-height",
    "disk": "Blurred-hole fit", "flux_ratio": "Flux ratio", "projection": "Projection", "half_area": "Half area",
    "half_area_min": "Half area, min level", "half_area_min_avg": "Half area, min level, avg peak",
    "tail_area": "Tail area", "tail_area_norm": "Tail area, no level", "edge_area": "Edge area",
    "foot_area": "Foot area", "foot_area_ratio": "Foot area, image ratio",
    "local_projection": "Raw-cell projection", "local_half_area": "Raw-cell half area",
    "local_half_area_min_avg": "Raw-cell half area, min level, avg peak", "local_tail_area": "Raw-cell tail area",
    "gpu_logquad": "GPU log-quadratic", "gpu_weighted": "GPU weighted", "gpu_edge_gradmoment": "GPU edge-moment",
    "gpu_edge_erf": "GPU edge-erf", "gpu_radial_symmetry": "GPU radial symmetry",
    "gpu_isophote_curvature": "GPU isophote", "new_template": "Template fit (prototype)", "truth": "True centers",
}
NEEDS_BLUR = ("disk", "flux_ratio")  # given the true blur here; real pages do not have it


def rows_apart(pitch_nm, period_nm=1000.0, min_rows=5):
    """Rows between moved rows: the whole number of rows closest to ``period_nm``, but at least ``min_rows``, so
    each moved row shows as its own +/- pair in the y pitch instead of a zig-zag (1 um is only 2 rows at 500 nm)."""
    return max(1, int(min_rows), int(round(period_nm / float(pitch_nm[1]))))


# --------------------------------------------------------------------------------------------- matching
def spacing_px(page):
    """Nearest-neighbour distance of the design lattice (px)."""
    px, py = page.meta["pitch_px"]
    return min(px, math.hypot(px / 2, py)) if page.meta["layout"] == "staggered" else min(px, py)


def match_points(points, truth, tolerance):
    """(len(truth), 2) array: each true hole gets the nearest point within ``tolerance`` px; a point goes only to
    its nearest hole (as the workbench pairs GPU finder centers with holes). NaN where none."""
    from scipy.spatial import cKDTree

    out = np.full(truth.shape, np.nan)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return out
    distance, hole = cKDTree(truth).query(points, distance_upper_bound=tolerance)
    near = np.isfinite(distance)
    distance, hole, points = distance[near], hole[near], points[near]
    order = np.lexsort((distance, hole))
    holes, first = np.unique(hole[order], return_index=True)
    out[holes] = points[order[first]]
    return out


def seed_holes(seeds, truth, tolerance):
    """True hole index of each seed (-1 if none within ``tolerance``); a hole keeps only its nearest seed."""
    from scipy.spatial import cKDTree

    seeds = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    hole = np.full(len(seeds), -1, dtype=np.int64)
    if not len(seeds):
        return hole
    distance, nearest = cKDTree(truth).query(seeds, distance_upper_bound=tolerance)
    near = np.flatnonzero(np.isfinite(distance))
    order = near[np.lexsort((distance[near], nearest[near]))]
    _, first = np.unique(nearest[order], return_index=True)
    hole[order[first]] = nearest[order[first]]
    return hole


# ------------------------------------------------------------------------------------------- evaluation
def evaluation_set(page):
    """Holes on the page and at least one pitch from its edge."""
    h, w = page.image.shape
    margin = max(page.meta["pitch_px"])
    c = page.centers_xy
    return page.inside & (c[:, 0] >= margin) & (c[:, 0] <= w - 1 - margin) & (c[:, 1] >= margin) & (
        c[:, 1] <= h - 1 - margin)


def pitch_pairs(page, use):
    """Index pairs (r, k) -> (r + 1, k) of holes in ``use``: the y-pitch pairs of analyze_pitch."""
    rc = page.indices_rc
    lookup = {(int(r), int(k)): i for i, (r, k) in enumerate(rc) if use[i]}
    first, second = [], []
    for i in np.flatnonzero(use):
        j = lookup.get((int(rc[i, 0]) + 1, int(rc[i, 1])))
        if j is not None:
            first.append(i)
            second.append(j)
    return np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)


def evaluate(page, centers, use, pairs, columns=COLUMNS):
    """Center and y-pitch numbers of one method; ``centers`` (N, 2) aligned with ``page.centers_xy`` (NaN = none).

    Pitch noise is the root mean square over every (band of N columns, row interval) cell of the error after
    removing the band's own mean over its row intervals (a constant offset does not hide a spike). Cells need at
    least half of their N pairs measured.
    """
    truth = page.centers_xy
    shift_nm = float(page.meta["row_shift_nm"])
    found = np.isfinite(centers).all(axis=1)
    measured = use & found
    error = (centers - truth) * PIXEL_NM
    out = dict(found=float(measured.sum() / max(use.sum(), 1)), holes=int(use.sum()))
    if measured.sum() < 10:
        return {**out, "failed": True}
    ey = error[measured, 1] - error[measured, 1].mean()
    ex = error[measured, 0] - error[measured, 0].mean()
    out.update(center_sd_y=float(np.std(ey)), center_sd_x=float(np.std(ex)),
               center_robust_y=float(1.4826 * np.median(np.abs(ey - np.median(ey)))),
               center_worst_y=float(np.abs(ey).max()), center_offset_y=float(error[measured, 1].mean()))
    # Pixel-position error: mean y error against the true center's place inside its pixel.
    phase = truth[measured, 1] - np.rint(truth[measured, 1])
    bins = np.clip(((phase + 0.5) * PHASE_BINS).astype(int), 0, PHASE_BINS - 1)
    count = np.bincount(bins, minlength=PHASE_BINS)
    mean = np.bincount(bins, weights=ey, minlength=PHASE_BINS) / np.maximum(count, 1)
    curve = np.where(count >= 20, mean, np.nan)
    out["phase_curve"] = [None if not np.isfinite(v) else round(float(v), 4) for v in curve]
    out["phase_error"] = float(0.5 * (np.nanmax(curve) - np.nanmin(curve))) if np.isfinite(curve).sum() >= 2 else 0.0

    a, b = pairs
    rows = page.indices_rc[a, 0]
    cols = page.indices_rc[a, 1]
    shift_row = {}
    for i in np.flatnonzero(use):
        shift_row[int(page.indices_rc[i, 0])] = float(page.row_shift_nm[i])
    planted = np.array([shift_row.get(int(r) + 1, 0.0) - shift_row.get(int(r), 0.0) for r in rows])
    design = float(page.meta["pitch_nm"][1])
    ok = found[a] & found[b]
    m = (centers[b, 1] - centers[a, 1]) * PIXEL_NM
    t = (truth[b, 1] - truth[a, 1]) * PIXEL_NM
    r0, k0 = int(rows.min()), int(cols.min())
    n_rows = int(rows.max()) - r0 + 1
    n_cols = int(cols.max()) - k0 + 1
    out["columns_total"] = n_cols
    out["row_intervals"] = n_rows
    noise_method, noise_seen = {}, {}
    profiles, truth_profiles = {}, {}
    planted_row = np.zeros(n_rows)
    planted_row[rows - r0] = planted
    for n in [*[c for c in columns if c < n_cols], n_cols]:
        key = "all" if n == n_cols else str(n)
        band = (cols - k0) // n
        full = band < n_cols // n  # whole bands only
        cell = band * n_rows + (rows - r0)
        size = (n_cols // n) * n_rows
        sel = ok & full
        count = np.bincount(cell[sel], minlength=size)
        sum_m = np.bincount(cell[sel], weights=m[sel], minlength=size)
        sum_t = np.bincount(cell[sel], weights=t[sel], minlength=size)
        enough = count >= max(1, math.ceil(0.5 * n))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_m = np.where(enough, sum_m / np.maximum(count, 1), np.nan).reshape(-1, n_rows)
            mean_t = np.where(enough, sum_t / np.maximum(count, 1), np.nan).reshape(-1, n_rows)
        expected = design + planted_row[None, :]
        noise_method[key] = _band_rms(mean_m - mean_t)
        noise_seen[key] = _band_rms(mean_m - expected)
        # Displayed curve: the band nearest the middle of the page.
        middle = min(max((n_cols // 2) // n, 0), mean_m.shape[0] - 1)
        profiles[key] = _rounded(mean_m[middle] - design)
        truth_profiles[key] = _rounded(mean_t[middle] - design)
        if key == "all":
            out["spike"] = _spike(mean_m[0], planted_row / max(abs(shift_nm), 1e-12)) if shift_nm else 0.0
            out["spike_truth"] = _spike(mean_t[0], planted_row / max(abs(shift_nm), 1e-12)) if shift_nm else 0.0
    out["noise_method"] = noise_method
    out["noise_seen"] = noise_seen
    out["profiles"] = profiles
    out["truth_profiles"] = truth_profiles
    out["planted"] = _rounded(planted_row)
    limit = abs(shift_nm) / 3 if shift_nm else 0.0
    for name, noise in (("need_seen", noise_seen), ("need_method", noise_method)):
        need = None
        for key, value in noise.items():
            if np.isfinite(value) and value <= limit:
                need = n_cols if key == "all" else int(key)
                break
        out[name] = need
    return out


def _band_rms(error):
    centred = error - np.nanmean(error, axis=1, keepdims=True)
    values = centred[np.isfinite(centred)]
    return float(np.sqrt(np.mean(values ** 2))) if len(values) else float("nan")


def _spike(curve, template):
    keep = np.isfinite(curve)
    if keep.sum() < 3 or np.var(template[keep]) == 0:
        return float("nan")
    p, t = curve[keep] - curve[keep].mean(), template[keep] - template[keep].mean()
    return float((p * t).sum() / (t * t).sum())


def _rounded(values):
    return [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(values, dtype=float)]


# ------------------------------------------------------------------------------------------ measurement
def detection_limits(spec):
    """Component area limits (px) around the design hole area: 0.2x to 4x + 20 (at least 50)."""
    ax, ay = np.asarray(spec["size_nm"]) / PIXEL_NM
    area = ax * ay * (1.0 if spec["hole"] == "square" else math.pi / 4)
    return max(2, int(0.2 * area)), int(max(50, 4 * area + 20))


def measure_methods(page, spec, args, log):
    """{method: (centers aligned with page.centers_xy, seconds, note)} for the CD methods and GPU finders."""
    from subpx import detect_centers, measure_cds
    from subpx.cd import METHOD_LABELS
    from subpx._workbench_data import GPU_CENTER_FINDERS

    truth = page.centers_xy
    tolerance = 0.5 * spacing_px(page)
    area_min, area_max = detection_limits(spec)
    found = {}
    cd_methods = tuple(METHOD_LABELS) if args.cd_methods is None else tuple(m for m in METHOD_LABELS
                                                                            if m in args.cd_methods)
    if cd_methods:
        started = time.perf_counter()
        try:
            result = measure_cds(page.image, methods=cd_methods, backend=args.backend, pixel_size=PIXEL_NM, unit="nm",
                                 area_min=area_min, area_max=area_max, summary_only=True,
                                 psf_fwhm=args.blur if set(cd_methods) & set(NEEDS_BLUR) else None)
            seconds = time.perf_counter() - started
            hole = seed_holes(result.seeds_xy, truth, tolerance)
            meta = result.meta
            note = (f"one measure_cds call for all CD methods: {seconds:.1f} s, {len(result.seeds_xy):,} seeds, "
                    f"background {meta.get('background_source', '?')}, deblend converged "
                    f"{meta.get('deblend_converged', '?')}")
            log(f"  CD methods: {note}")
            for name in cd_methods:
                measurement = result.measurements[name]
                centers = np.full(truth.shape, np.nan)
                valid = np.asarray(measurement.valid, bool) & (hole >= 0)
                centers[hole[valid]] = np.asarray(measurement.centers_xy, float)[valid]
                found[name] = (centers, seconds, note)
        except Exception as exc:  # a failed call is a result too
            log(f"  CD methods failed: {exc!r}")
            for name in cd_methods:
                found[name] = (np.full(truth.shape, np.nan), time.perf_counter() - started, f"failed: {exc}")
    for finder in (args.finders if args.finders is not None else tuple(GPU_CENTER_FINDERS)):
        started = time.perf_counter()
        try:
            points = detect_centers(page.image, backend="gpu", threshold="otsu", invert=False, area_min=area_min,
                                    area_max=area_max, pad=3, refine=finder).centers_xy
            seconds = time.perf_counter() - started
            found[f"gpu_{finder}"] = (match_points(points, truth, tolerance), seconds, "")
            log(f"  GPU {finder}: {len(points):,} centers in {seconds:.2f} s")
        except Exception as exc:
            found[f"gpu_{finder}"] = (np.full(truth.shape, np.nan), time.perf_counter() - started, f"failed: {exc}")
            log(f"  GPU {finder} failed: {exc!r}")
    if args.template:
        from scripts.template_fit_prototype import template_fit_centers

        started = time.perf_counter()
        try:
            backend = "gpu" if args.backend == "gpu" else "cpu"
            start = detect_centers(page.image, backend=backend, threshold="otsu", invert=False, area_min=area_min,
                                   area_max=area_max, pad=3, refine="weighted").centers_xy
            points = template_fit_centers(page.image, start, backend=backend)
            seconds = time.perf_counter() - started
            found["new_template"] = (match_points(points, truth, tolerance), seconds,
                                     "prototype in scripts/template_fit_prototype.py, started from weighted centroids")
            log(f"  template fit (prototype): {np.isfinite(points).all(axis=1).sum():,} centers in {seconds:.1f} s")
        except Exception as exc:
            found["new_template"] = (np.full(truth.shape, np.nan), time.perf_counter() - started, f"failed: {exc}")
            log(f"  template fit failed: {exc!r}")
    found["truth"] = (truth.copy(), 0.0, "the true centers, as a check of the evaluation")
    return found


def precision_limit(page):
    """Best possible y-center sd (nm) for one hole alone: the Cramer-Rao bound with its brightness and background
    fitted too, for the design hole at the mean brightness, photon + read noise + 8-bit rounding, averaged over
    4 x 4 places inside the pixel. Neighbours' light and CD noise are left out."""
    from subpx.synthetic_arrays import LENS_FWHM_FACTOR, _render, blur_smear_sigma

    meta = page.meta
    size = np.asarray(meta["size_px"], dtype=float)
    blur, lens = meta["blur_nm"] / PIXEL_NM, meta["lens_nm"] / PIXEL_NM
    sigma = blur_smear_sigma(blur, lens)
    cutoff = LENS_FWHM_FACTOR / lens if lens > 0 else None
    patch = 2 * int(math.ceil(0.5 * float(np.hypot(*size)) + 4.0 * blur + 1))
    patch = max(patch, 64) if cutoff is not None else patch
    oversample = int(meta["oversample"])
    amplitude = meta["open_level"] - meta["background"]
    photons = meta["photons_per_level"]
    step = 0.01
    fisher = np.zeros((4, 4))
    for fx in np.arange(4) / 4:
        for fy in np.arange(4) / 4:
            center = np.array([[patch / 2 + fx, patch / 2 + fy]])

            def light(dx=0.0, dy=0.0):
                return _render((patch, patch), center + [dx, dy], size[None], np.array([amplitude]), meta["hole"],
                               0.0, patch, oversample, cutoff, sigma, "cpu")

            hole = light()
            level = meta["background"] + hole
            variance = (level / photons if photons else 0.0) + meta["read_noise"] ** 2 + (
                1 / 12 if meta["bits"] else 0.0)
            columns = [(light(step) - light(-step)) / (2 * step), (light(0, step) - light(0, -step)) / (2 * step),
                       hole / amplitude, np.ones_like(hole)]
            jac = np.stack([c.ravel() for c in columns], axis=1)
            fisher += jac.T @ (jac / variance.ravel()[:, None]) / 16
    return float(np.sqrt(np.linalg.inv(fisher)[1, 1]) * PIXEL_NM)


def group_of(name):
    if name == "truth":
        return "truth"
    if name.startswith("gpu_"):
        return "gpu"
    if name.startswith("new_"):
        return "new"
    return "cd"


def crop_png(page, size=72, scale=4):
    """A contrast-stretched crop of the page center, enlarged ``scale`` times, as PNG bytes."""
    import struct
    import zlib

    h, w = page.image.shape
    y0, x0 = (h - size) // 2, (w - size) // 2
    crop = page.image[y0:y0 + size, x0:x0 + size].astype(float)
    lo, hi = np.percentile(page.image, [0.5, 99.9])
    data = np.clip((crop - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    data = np.repeat(np.repeat(data, scale, axis=0), scale, axis=1)
    raw = b"".join(b"\x00" + row.tobytes() for row in data)

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", data.shape[1], data.shape[0], 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def run_image(name, spec, args, log):
    started = time.perf_counter()
    page = render_hole_array(
        tuple(args.shape), hole=spec["hole"], size_nm=spec["size_nm"], pitch_nm=spec["pitch_nm"],
        layout=spec["layout"], pixel_nm=PIXEL_NM, blur_nm=args.blur, lens_nm=args.lens,
        photons_per_level=args.photons, cd_sigma_nm=args.cd_sigma, placement_sigma_nm=args.placement_sigma,
        brightness_sigma=args.brightness_sigma, row_shift_nm=args.row_shift, row_shift_period_nm=args.row_period,
        row_shift_every=rows_apart(spec["pitch_nm"], args.row_period, args.min_rows_apart),
        rotation_deg=args.rotation, seed=args.seed + list(PATTERNS).index(name), backend=args.render_backend)
    log(f"{name}: {page.meta['holes_inside']:,} holes, rendered in {time.perf_counter() - started:.1f} s; every "
        f"{page.meta['row_shift_every']} rows moved {page.meta['row_shift_nm']} nm")
    use = evaluation_set(page)
    pairs = pitch_pairs(page, use)
    methods = measure_methods(page, spec, args, log)
    results = {}
    for key, (centers, seconds, note) in methods.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            numbers = evaluate(page, centers, use, pairs)
        numbers.update(seconds=round(float(seconds), 3), note=note, group=group_of(key),
                       name=NAMES.get(key, key), needs_blur=key in NEEDS_BLUR)
        results[key] = numbers
    import base64

    meta = {k: v for k, v in page.meta.items() if k not in ("origin_xy",)}
    limit = precision_limit(page)
    log(f"  precision limit (one hole alone): {limit:.3f} nm")
    return dict(key=name, label=spec["label"], spec={k: v for k, v in spec.items() if k != "label"}, meta=meta,
                limit_nm=round(limit, 4),
                crop_png=base64.b64encode(crop_png(page)).decode("ascii"), methods=results,
                evaluated_holes=int(use.sum()), pairs=int(len(pairs[0])),
                seconds=round(time.perf_counter() - started, 1))


def best_of(results, group):
    """Method of ``group`` that sees the spike with the fewest columns (ties: smaller method-alone noise at 1
    column, then smaller center error); methods given the true blur are left out, real pages lack it."""
    candidates = [(k, v) for k, v in results.items() if v["group"] == group and not v.get("failed")
                  and v["found"] >= 0.99 and not v.get("needs_blur")]

    def rank(item):
        value = item[1]
        need = value["need_seen"] if value["need_seen"] is not None else 10 ** 9
        return need, value["noise_method"].get("1", float("inf")), value["center_sd_y"]

    return min(candidates, key=rank)[0] if candidates else None


def finite(value):
    """Copy with NaN / inf as None, so results.json is standard JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


def write_csv(path, images):
    fields = ["image", "method", "name", "group", "found", "center_sd_y_nm", "center_sd_x_nm", "pixel_position_nm",
              "spike_nm", "spike_truth_nm", "columns_needed_seen", "columns_needed_method",
              *[f"noise_seen_{c}" for c in (*map(str, COLUMNS), "all")],
              *[f"noise_method_{c}" for c in (*map(str, COLUMNS), "all")], "seconds", "note"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for image in images:
            for key, v in image["methods"].items():
                row = dict(image=image["key"], method=key, name=v["name"], group=v["group"], found=v["found"],
                           seconds=v["seconds"], note=v["note"])
                if not v.get("failed"):
                    row.update(center_sd_y_nm=v["center_sd_y"], center_sd_x_nm=v["center_sd_x"],
                               pixel_position_nm=v["phase_error"], spike_nm=v["spike"],
                               spike_truth_nm=v["spike_truth"], columns_needed_seen=v["need_seen"],
                               columns_needed_method=v["need_method"])
                    for c in (*map(str, COLUMNS), "all"):
                        row[f"noise_seen_{c}"] = v["noise_seen"].get(c)
                        row[f"noise_method_{c}"] = v["noise_method"].get(c)
                writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", choices=tuple(PATTERNS), default=list(PATTERNS))
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "pitch_signature")
    parser.add_argument("--shape", type=int, nargs=2, default=(1378, 2048), metavar=("H", "W"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--blur", type=float, default=65.0, help="blur width at half height, nm")
    parser.add_argument("--lens", type=float, default=45.0, help="the lens's own blur width, nm (0: Gaussian blur)")
    parser.add_argument("--photons", type=float, default=30.0, help="photons per grey level")
    parser.add_argument("--cd-sigma", type=float, default=0.5, help="per-hole CD noise sd, nm")
    parser.add_argument("--placement-sigma", type=float, default=0.2, help="per-hole placement noise sd, nm")
    parser.add_argument("--brightness-sigma", type=float, default=0.05, help="per-hole brightness sd (fraction)")
    parser.add_argument("--row-shift", type=float, default=1.0, help="planted row shift, nm")
    parser.add_argument("--row-period", type=float, default=1000.0, help="distance between shifted rows, nm")
    parser.add_argument("--min-rows-apart", type=int, default=5,
                        help="shifted rows are at least this many rows apart (the period grows on large pitches)")
    parser.add_argument("--rotation", type=float, default=0.0, help="lattice rotation, degrees")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu", help="CD methods' backend")
    parser.add_argument("--render-backend", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--cd-methods", nargs="*", help="CD methods (default: all); none: skip them")
    parser.add_argument("--finders", nargs="*", help="GPU center finders (default: all)")
    parser.add_argument("--no-template", dest="template", action="store_false",
                        help="skip the template-fit prototype")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(text):
        print(text, flush=True)
        lines.append(text)

    started = time.perf_counter()
    images = [run_image(name, PATTERNS[name], args, log) for name in args.images]
    for image in images:
        image["best_gpu"] = best_of(image["methods"], "gpu")
        image["best_cd"] = best_of(image["methods"], "cd")
        image["best_new"] = best_of(image["methods"], "new")
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    settings.update(pixel_nm=PIXEL_NM, columns=list(COLUMNS), phase_bins=PHASE_BINS,
                    total_seconds=round(time.perf_counter() - started, 1))
    data = finite(dict(settings=settings, images=images, log=lines))
    (args.output / "results.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    write_csv(args.output / "pitch_signature.csv", images)
    from scripts.pitch_signature_view import write_report

    report = write_report(data, args.output / "report.html")
    log(f"done in {settings['total_seconds']} s: {report}")
    return data


if __name__ == "__main__":
    main()
