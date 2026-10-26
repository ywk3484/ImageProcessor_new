"""Candidate cures for deblend's slow settling, as switches on a copy of the GPU loop.

On the user's pages the deblending loop does not settle: after 30 rounds the 99th-percentile parameter step is still
above the tolerance, the parameters drift slowly one way, and the CDs depend on where the loop stops. ``deblend`` here
is a copy of ``subpx._gpu.deblend.deblend`` with each candidate cure behind a switch of ``Variant``; with every switch
off it returns the library's result bit for bit. scripts/cd_diagnose.py --part 2 patches it in at run time with
``Patch`` (measure_cds then runs it) to try the cures on the user's machine. Nothing here changes the library.

Switches of ``Variant`` (``Patch`` applies the middle five in the last pass only):
    lm_steps                LM steps of every fit (None: the library's max_iter, 30)
    tolerance               the stop rule's limit for the raw step, every call (None: the library's 1e-3);
                            TIGHT_TOLERANCE finds where the loop ends when it runs on, PRACTICAL_TOLERANCE is a
                            stop a cure could ship with
    freeze_background       keep the background map of the first round
    freeze_free             keep the free-pixel set of the first round; the map is still re-estimated over it
    anderson                Anderson acceleration of the rounds, with this many stored differences
    widen                   start with widths this fraction wider
    tile_free_px            a background tile is measured with this many free pixels (the library: max(16, 10 % of
                            the tile's pixels); below that its level is interpolated or inferred)
    continue_after_recheck  (``Patch``) the refit after a seed re-check starts kept holes from their previous fits
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from subpx._cpu import deblend as cpu
from subpx._gpu import deblend as gpu
from subpx._gpu.deblend import _CLEAN_SRC, cp, kernel, require_cupy

# Stops for the diagnostic (the library's is 1e-3). On a drifting test page (scripts/drift_page.py) the loop has one
# end point: the plain loop, Anderson and a 2 % wider start all reach it within 0.01 nm at TIGHT_TOLERANCE (242
# rounds for the plain loop), today's stop at 30 rounds is 1.3 nm away from it, and a PRACTICAL_TOLERANCE stop is
# within 0.05 nm (plain 110 rounds, Anderson 32).
TIGHT_TOLERANCE = 1e-5  # the reference: where the loop ends when it runs on
PRACTICAL_TOLERANCE = 1e-4  # a cure's stop: what a batch could afford


@dataclass(frozen=True)
class Variant:
    lm_steps: int | None = None        # inner LM steps per round and for the initial fit (None: the library's max_iter)
    tolerance: float | None = None     # the stop rule's limit for the raw step, every call (None: the library's 1e-3)
    freeze_background: bool = False    # the pass keeps the background map of its first round
    freeze_free: bool = False          # the pass keeps the free-pixel set of its first round (the map still updates)
    anderson: int = 0                  # Anderson acceleration with this many stored differences (0: the library's damped step)
    widen: float = 0.0                 # the pass starts with widths (1 + widen) wider than its start parameters
    tile_free_px: int | None = None    # free pixels that make a background tile measured (None: the library's rule)
    continue_after_recheck: bool = False  # a refit after the seed re-check starts kept holes from their previous fits


def _p99(step, rows, columns=slice(1, None)):
    """The stop rule's number: the 99th percentile over ``rows`` of each hole's largest |step| in ``columns`` (centre
    and widths); NaN without a row."""
    if not rows.any():
        return float("nan")
    return float(np.percentile(np.abs(step[rows][:, columns]).max(axis=1), 99))


def _background_tile_rule(residual, free, tile, usable, tile_free_px):
    """``gpu._background_device`` with a tile measured when it has at least ``tile_free_px`` free pixels instead of
    max(16, 10 % of its pixels); everything else as the library's background map."""
    shape = residual.shape
    rule = max(16, int(0.1 * tile * tile))  # cpu.background_tiles' threshold
    first = True

    def tile_medians(mask):
        nonlocal first
        medians, counts = gpu._tile_medians(residual, mask, tile)
        if first:
            # background_tiles' first call is over the free pixels, and it uses these counts only in `counts >= rule`:
            # rule where a tile has tile_free_px free pixels and 0 elsewhere makes exactly those tiles measured (and
            # info["measured_tiles"] counts them). A later call (all usable pixels, when none is measured) passes.
            counts, first = np.where(counts >= tile_free_px, rule, 0), False
        return medians, counts

    medians, info = cpu.background_tiles(tile_medians, free, usable, tile, shape)
    if medians is None:
        return cp.full(shape, float(np.median(cp.asnumpy(residual)))), info
    rows, j1, t = cpu._upsample_rows(medians, tile, shape)
    rows = cp.asarray(rows)
    if j1 is None:
        return cp.repeat(rows, shape[0], axis=0), info
    t, j1 = cp.asarray(t), cp.asarray(j1)
    return (1 - t) * rows[j1 - 1] + t * rows[j1], info


def _solve(matrix, rhs):
    """A small linear system by Gaussian elimination with partial pivoting in Python floats: the same bits on every
    machine (LAPACK's depend on its build and the CPU)."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    k = len(rows)
    for c in range(k):
        pivot = max(range(c, k), key=lambda r: abs(rows[r][c]))
        rows[c], rows[pivot] = rows[pivot], rows[c]
        for r in range(c + 1, k):
            factor = rows[r][c] / rows[c][c]
            for j in range(c, k + 1):
                rows[r][j] -= factor * rows[c][j]
    x = [0.0] * k
    for i in reversed(range(k)):
        x[i] = (rows[i][k] - sum(rows[i][j] * x[j] for j in range(i + 1, k))) / rows[i][i]
    return x


def deblend(image, seeds_xy, masks, origins_xy, *, iterations=30, tile=None, exclude=None, background_level=None,
            free_fraction=0.05, free_distance_fwhm=1.25, relaxation=0.6, tolerance=1e-3, max_iter=30, init=None,
            variant=Variant(), trace=None):
    """``subpx._gpu.deblend.deblend`` with the switches of ``variant`` (``continue_after_recheck`` is ``Patch``'s); with
    every switch off the same operations and the same result, bit for bit.

    The library's steps (LM fit, render, background map, cell parts) are called through their module, so that
    cd_diagnose's ``Probe``, which replaces them there, times and records this loop as it does the library's (with
    ``tile_free_px`` set the background map is this module's copy, which Probe records but does not time).

    ``trace``: a list that gets one dict per round: round (from 1); seconds; ok (holes fitted ok); step_p99 (the stop
    rule's number: 99th percentile over ok holes of the largest |fitted - params| of centre and widths; NaN without an
    ok hole); centre_p99, width_p99 (the same over the centre, the width columns only); edge_p99 (step_p99 of the holes
    whose seed is < 1, 1-2 and >= 2 pitches from the nearest image edge; NaN for an empty band); bg_mean (mean of the
    background map, image units); bg_max, bg_rms (largest and rms change of the background map since the previous
    round); free_px (free pixels; None for a fixed level); measured_tiles, source (of the background map); cos (cosine
    between this and the previous round's steps of centre and widths, over holes ok in both); update ("plain",
    "anderson" or "fallback"). Values that need a previous round are NaN in round 1. All plain Python values; the trace
    changes no number of the result.
    """
    from cupyx.scipy.ndimage import binary_erosion, distance_transform_edt

    require_cupy()
    steps = max_iter if variant.lm_steps is None else int(variant.lm_steps)
    if variant.tolerance is not None:
        tolerance = float(variant.tolerance)
    image = np.asarray(image, dtype=np.float64)
    shape = image.shape
    seeds_xy = np.asarray(seeds_xy, float).reshape(-1, 2)
    n = len(seeds_xy)
    masks = np.asarray(masks, dtype=bool)
    if n == 0:
        return cpu.deblend(image, seeds_xy, masks, origins_xy, iterations=iterations, tile=tile, exclude=exclude,
                           background_level=background_level, free_fraction=free_fraction,
                           free_distance_fwhm=free_distance_fwhm, relaxation=relaxation, tolerance=tolerance,
                           max_iter=steps, init=init)
    origins = np.asarray(origins_xy, dtype=np.int64).reshape(-1, 2)
    if tile is None:
        tile = cpu.auto_tile(seeds_xy, shape)[0]
    tile = int(tile)
    seed_pixels = np.ones(shape, bool)
    seed_pixels[np.clip(np.rint(seeds_xy[:, 1]).astype(int), 0, shape[0] - 1),
                np.clip(np.rint(seeds_xy[:, 0]).astype(int), 0, shape[1] - 1)] = False
    roi_shape = masks.shape[1:]
    radius_cap = int(max(roi_shape))
    parts = gpu._cell_parts(masks)

    image_d = cp.asarray(image)
    distance_d = distance_transform_edt(cp.asarray(seed_pixels))
    usable_d = cp.ones(shape, bool) if exclude is None else cp.asarray(~np.asarray(exclude, bool))
    part_masks = [cp.asarray(masks[:, :h, :w][rows], dtype=cp.uint8) for rows, h, w in parts]
    part_origins = [cp.asarray(origins[rows]) for rows, _, _ in parts]
    pixels = np.empty(n)
    for part, (rows, _, _) in enumerate(parts):
        pixels[rows] = cp.asnumpy(part_masks[part].reshape(len(rows), -1).sum(axis=1))
    caps = np.maximum(1.0, np.sqrt(pixels / np.pi))  # cpu.sigma_caps(masks)
    known = {}  # sigma_major of the current parameters: render radii and the background's FWHM
    frozen = {}  # what the call keeps from its first round: "free" (freeze_free), "background" (freeze_background)

    def sigmas(p):
        if known.get("params") is not p:
            known.update(params=p, sigma=gpu._sigma_major(p))
        return known["sigma"]

    def radii(p):  # cpu._render_radius(p, radius_cap)
        return np.clip(np.ceil(cpu.RENDER_SIGMAS * sigmas(p)), 2, radius_cap)

    def global_params(p, rows=slice(None)):
        g = p[rows].copy()
        g[:, 1:3] += origins[rows]
        return g

    def background_of(model_d, p, st):
        """(map, info, free pixels); free is None for a fixed level."""
        if "background" in frozen:
            return frozen["background"]
        if background_level is not None:
            return cp.full(shape, float(background_level)), {"source": "fixed"}, None
        free = frozen.get("free")
        if free is None:
            good = (st == "ok") & (p[:, 0] > 0)
            amplitude = float(np.median(p[good, 0])) if good.any() else 1.0
            fwhm = float(np.median(2 * cpu.FWHM_K * sigmas(p)[good])) if good.any() else 0.0
            free = (model_d < free_fraction * amplitude) & (distance_d >= free_distance_fwhm * fwhm)
        if variant.tile_free_px is None:
            return (*gpu._background_device(image_d - model_d, free, tile, usable_d), free)
        return (*_background_tile_rule(image_d - model_d, free, tile, usable_d, variant.tile_free_px), free)

    def clean(p, background_d, model_d, part):
        rows, h, w = parts[part]
        cells = part_masks[part]
        out = cp.empty(cells.shape, dtype=cp.float64)
        total = out.size
        kernel(_CLEAN_SRC, "clean_cells")(
            ((total + 255) // 256,), (256,),
            (image_d, background_d, model_d, cp.asarray(global_params(p, rows)),
             cp.asarray(radii(p)[rows].astype(np.int32)), cells, part_origins[part], out, np.int64(total), np.int32(h),
             np.int32(w), np.int32(shape[0]), np.int32(shape[1])))
        return out

    def fit(data_of, p):
        fitted = p.copy()
        for part, (rows, _, _) in enumerate(parts):
            fitted[rows] = gpu._fit_device(data_of(part), part_masks[part], p[rows], max_iter=steps)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(fitted, masks, caps, status, sigma=gpu._sigma_major(fitted))
        return fitted, status

    if init is None:
        # Initialize from the cell-border background; the iterations replace it.
        params = np.empty((n, 6))
        for part, (rows, h, w) in enumerate(parts):
            cells = part_masks[part].astype(bool)
            border = cells & ~binary_erosion(cells, structure=cp.ones((1, 3, 3), bool), border_value=0)
            raw = gpu._windows(image_d, part_origins[part], (h, w))
            bg0 = gpu.border_percentile(raw, border)
            if background_level is not None:
                bg0[:] = background_level
            data = cp.where(cells, raw - bg0[:, None, None], 0.0)
            start = cp.asnumpy(gpu._initial_params(data, cells, cp.asarray(seeds_xy[rows] - origins[rows])))
            params[rows] = gpu._fit_device(data, part_masks[part], start, max_iter=steps)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status, sigma=gpu._sigma_major(params))
    else:
        params, status = np.asarray(init, dtype=np.float64).copy(), np.full(n, "ok", dtype="U32")
    if variant.widen:
        # Widths (1 + widen) wider in every direction: L = [[e^u1, 0], [u2, e^u3]] divided by (1 + widen).
        params[:, 3] -= np.log(1 + variant.widen)
        params[:, 4] /= 1 + variant.widen
        params[:, 5] -= np.log(1 + variant.widen)
        status = np.full(n, "ok", dtype="U32")
        cpu.fit_status(params, masks, caps, status, sigma=gpu._sigma_major(params))

    model_d = gpu._render_device(global_params(params), shape, radius_cap, radii(params))
    if trace is not None:
        sync = cp.cuda.Device().synchronize
        sync()  # the initial render is not round 1's time
        pitch = cpu.auto_tile(seeds_xy, shape)[1]
        x, y = seeds_xy[:, 0], seeds_xy[:, 1]
        edge = np.minimum(np.minimum(x, shape[1] - 1 - x), np.minimum(y, shape[0] - 1 - y)) / pitch  # in pitches
        bands = (edge < 1, (edge >= 1) & (edge < 2), edge >= 2)
    low, high = -np.log(caps), -np.log(cpu._SIGMA_MIN)  # the LM fit's width bounds, for Anderson's update
    differences = []  # Anderson: the latest (params, step) differences between consecutive rounds
    previous = None  # the previous round: params, step, ok holes, Anderson's residual rms, background map
    used, converged = 0, False
    for used in range(1, int(iterations) + 1):
        started = time.perf_counter()
        background_d, info, free = background_of(model_d, params, status)
        if used == 1:
            if variant.freeze_free:
                frozen["free"] = free
            if variant.freeze_background:
                frozen["background"] = background_d, info, free
        fitted, status = fit(lambda part: clean(params, background_d, model_d, part), params)
        # All cells update at once against last iteration's neighbours, damped (see the CPU module).
        step = np.where(np.isfinite(fitted), fitted - params, 0.0)
        new, update, norm = params + relaxation * step, "plain", None
        ok = status == "ok"
        if variant.anderson:
            # Anderson acceleration (type II, Walker & Ni 2011) of params -> fitted: the stored differences of params
            # and steps of the latest rounds give a secant model of how the step changes with the params; the update
            # moves to where that model's step is least (least squares over the holes fitted ok this round), damped
            # like the plain step. The other holes take the plain step. Inner products with np.sum and the solve in
            # Python floats: BLAS sums depend on the thread count, and the diagnostic compares machines.
            mask = ok & np.isfinite(fitted).all(axis=1)
            # Amplitudes in grey levels would swamp centres (px) and log-widths in the least squares: relative to
            # the median amplitude, a 1 % amplitude change weighs like a 0.01 px centre move or a 1 % width change.
            amplitude = float(np.median(params[mask, 0])) if mask.any() else 0.0
            weight = np.array([1 / amplitude if amplitude > 0 else 1.0, 1, 1, 1, 1, 1])
            f = step[mask] * weight
            norm = math.sqrt(float(np.sum(f * f)) / max(len(f), 1))  # rms per hole: comparable as the holes change
            if previous is not None:
                differences.append((params - previous["params"], step - previous["step"]))
                differences = differences[-variant.anderson:]
                if norm > previous["norm"]:  # the residual grew: restart from the plain step
                    differences, update = [], "fallback"
            if differences:
                df = [d[mask] * weight for _, d in differences]
                gram = [[float(np.sum(a * b)) for b in df] for a in df]
                regular = 1e-10 * sum(gram[i][i] for i in range(len(df)))
                gamma = [0.0] * len(df)
                if regular > 0:
                    for i in range(len(df)):
                        gram[i][i] += regular
                    gamma = _solve(gram, [float(np.sum(a * f)) for a in df])
                mixed = new[mask]
                for g, (dx, ds) in zip(gamma, differences):
                    mixed -= g * (dx[mask] + relaxation * ds[mask])
                # Keep the extrapolated parameters where the fits can be: amplitude >= 0, widths inside the LM
                # fit's bounds, centres inside the cell stack.
                mixed[:, 0] = np.maximum(mixed[:, 0], 0.0)
                mixed[:, 1] = np.clip(mixed[:, 1], -0.5, roi_shape[1] - 0.5)
                mixed[:, 2] = np.clip(mixed[:, 2], -0.5, roi_shape[0] - 0.5)
                mixed[:, 3] = np.clip(mixed[:, 3], low[mask], high)
                mixed[:, 5] = np.clip(mixed[:, 5], low[mask], high)
                new[mask] = mixed
                update = "anderson"
        last_params, params = params, new
        model_d = gpu._render_device(global_params(params), shape, radius_cap, radii(params))
        p99 = _p99(step, ok)
        if trace is not None:
            sync()  # the round's GPU work counts in its time
            row = {"round": used, "seconds": time.perf_counter() - started, "ok": int(ok.sum()), "step_p99": p99,
                   "centre_p99": _p99(step, ok, slice(1, 3)), "width_p99": _p99(step, ok, slice(3, 6)),
                   "edge_p99": [_p99(step, ok & band) for band in bands], "bg_mean": float(background_d.mean()),
                   "bg_max": math.nan, "bg_rms": math.nan,
                   "free_px": None if free is None else int(free.sum()), "measured_tiles": info.get("measured_tiles"),
                   "source": info["source"], "cos": math.nan, "update": update}
            if previous is not None:
                delta = background_d - previous["background"]
                row["bg_max"] = float(cp.abs(delta).max())
                row["bg_rms"] = float(cp.sqrt(cp.mean(delta * delta)))
                both = ok & previous["ok"]
                a, b = step[both][:, 1:], previous["step"][both][:, 1:]
                norms = float(np.sum(a * a)) * float(np.sum(b * b))
                if norms > 0:
                    row["cos"] = float(np.sum(a * b)) / math.sqrt(norms)
            trace.append(row)
        previous = {"params": last_params, "step": step, "ok": ok, "norm": norm, "background": background_d}
        if not ok.any() or p99 < tolerance:
            converged = True
            break
    background_d, info, _ = background_of(model_d, params, status)
    cleaned = np.zeros(masks.shape)
    for part, (rows, h, w) in enumerate(parts):
        cleaned[rows, :h, :w] = cp.asnumpy(clean(params, background_d, model_d, part))
    return cpu.DeblendState(cp.asnumpy(background_d), cp.asnumpy(model_d), params, status, cleaned,
                            tile, used, info, converged)


class Patch:
    """Context manager: ``subpx._gpu.deblend.deblend`` (what measure_cds runs) becomes the variant loop; records
    every pass.

    measure_cds deblends in passes (subpx/cd.py): pass 1 asks for min(10, deblend_iterations) rounds; after the seed
    re-check either the cells are rebuilt and refitted from scratch, or one call with ``init`` finishes the fits with
    the remaining rounds; a third pass (after two re-checks that changed the seeds) asks for every round. A call is
    the last pass when it has ``init`` or asks for at least ``full_iterations`` rounds (pass the deblend_iterations
    given to measure_cds). freeze_background, freeze_free, anderson, widen and tile_free_px apply to the last pass
    only, so that the seed re-check sees the library's fits; lm_steps and tolerance apply to every call.

    continue_after_recheck: a call without ``init`` on the same image as the previous call but with other seeds is a
    refit after the re-check. It then starts from the library's start parameters (the initial fit of a call with no
    round), except the holes whose seed is exactly one of the previous call's: those start from their previous fit,
    moved to the new cell origin. A different image is a new measurement: nothing carries over.

    ``passes``: one dict per call, in order: asked, used (rounds), settled, continued (``init`` from the caller),
    continued_holes, last, holes, seconds, variant (the switches applied), rounds (the trace rows; None without
    ``trace``).
    """

    def __init__(self, variant=Variant(), *, full_iterations=30, trace=True):
        self.variant, self.full_iterations, self.trace = variant, int(full_iterations), bool(trace)
        self.passes = []

    def __enter__(self):
        self.passes, self.saved = [], gpu.deblend
        previous = {}  # the previous call: image, seeds, origins, fitted params

        def patched(image, seeds_xy, masks, origins_xy, **kwargs):
            start = time.perf_counter()
            given = kwargs.get("init") is not None
            last = given or kwargs.get("iterations", 30) >= self.full_iterations
            applied = self.variant if last else Variant(
                lm_steps=self.variant.lm_steps, tolerance=self.variant.tolerance,
                continue_after_recheck=self.variant.continue_after_recheck)
            seeds = np.asarray(seeds_xy, float).reshape(-1, 2)
            origins = np.asarray(origins_xy, dtype=np.int64).reshape(-1, 2)
            continued = 0
            if (applied.continue_after_recheck and not given and previous.get("image") is image
                    and not np.array_equal(seeds, previous["seeds"])):
                index = {key: i for i, key in enumerate(map(tuple, previous["seeds"].tolist()))}
                pairs = [(i, index[key]) for i, key in enumerate(map(tuple, seeds.tolist())) if key in index]
                if pairs:
                    kept, before = (np.array(column) for column in zip(*pairs))
                    init = deblend(image, seeds_xy, masks, origins_xy, **dict(kwargs, iterations=0),
                                   variant=Variant(lm_steps=self.variant.lm_steps)).params
                    init[kept] = previous["params"][before]
                    init[kept, 1:3] += previous["origins"][before] - origins[kept]
                    kwargs = dict(kwargs, init=init)
                    continued = len(pairs)
            rows = [] if self.trace else None
            state = deblend(image, seeds_xy, masks, origins_xy, **kwargs, variant=applied, trace=rows)
            previous.update(image=image, seeds=seeds, origins=origins, params=state.params)
            self.passes.append({
                "asked": int(kwargs.get("iterations", 30)), "used": int(state.iterations),
                "settled": bool(state.converged), "continued": given, "continued_holes": continued, "last": last,
                "holes": len(seeds), "seconds": time.perf_counter() - start, "variant": asdict(applied),
                "rounds": rows})
            return state

        gpu.deblend = patched
        return self

    def __exit__(self, *exc):
        gpu.deblend = self.saved
