"""Every step of outline_fit_fast on the images of a folder, as pictures on one page in the browser.

    python scripts/outline_fit_steps.py FOLDER --output OUT --pixel-size 21 --unit nm

Use the options the folder report was run with (--outline-blur, --invert, --seeds, --threshold, --area-min,
--area-max, --pad). Every .png of the folder is read (and one page of every .tif / .tiff: --page), or one image
file. Needs the GPU (CuPy), as the method does. The page is OUT/index.html (opened in the browser unless --no-open);
it works offline, from the folder it was written to.

What the page shows. The method's calculation is run unchanged, with a record of what every step produced
(``fit_outline_fast(trace=...)``); the page draws that record, step by step:

    1 image as read         4 first average hole          7 each hole fitted on the image    10 result
    2 holes found           5 each hole against it        8 second average hole
    3 spacing and window    6 first fit                   9 final fit

Beside every picture of the image is the same picture of a redrawn copy: an image drawn from the fit's own answer
(every hole with the image's one outline and one blur at its own position, brightness and level, plus the noise the
image has from pixel to pixel, rounded like the file) and put through the same steps. Where the two look alike, the
image is what the method assumes there; where they differ, it is not. The copy also says what a number would be on
this very image if the method's picture of it were right. A second view puts all images of the folder side by side.
How to read each step, and what planted faults look like: docs/outline_fit_steps.md.

Nothing in the library is changed by this script; it shows.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.image_values import SUFFIXES, file_values  # noqa: E402

PAGE = Path(__file__).with_suffix(".html")
YOU, COPY = "your image", "redrawn copy"
PIECE = 96  # px: side of the pieces shown at full size
SMALL = 320  # px: longest side of the whole image as shown
LIMIT = 30  # the library's cap on the final fit's iterations (subpx._gpu.outline_onepass.FINAL_ITERATIONS)
CENTRE = "px from the hole's centre"


# ------------------------------------------------------------------------------------------- running it
@contextmanager
def recording(box: dict):
    """While active, the fit that ``measure_cds`` runs for ``outline_fit_fast`` records its steps: ``box`` gets
    ``seeds`` (the full list it was handed), ``trace`` and ``fit``. The call and its result are otherwise the
    report's own."""
    import subpx.outline as outline

    original = outline.fit_outline_fast

    def recorded(image, seeds_xy=None, **kw):
        box["seeds"] = None if seeds_xy is None else np.array(seeds_xy, dtype=np.float64)
        box["trace"] = {}
        box["fit"] = original(image, seeds_xy, trace=box["trace"], redrawn=True, **kw)
        return box["fit"]

    outline.fit_outline_fast = recorded
    try:
        yield
    finally:
        outline.fit_outline_fast = original


def run(image, label: str, options: dict, blur: str, device: int = 0) -> dict:
    """One image through the report's ``outline_fit_fast`` with its steps recorded."""
    from subpx import measure_cds

    box = {}
    started = time.perf_counter()
    with recording(box):
        result = measure_cds(image, methods=("outline_fit_fast",), outline_blur=blur, backend="gpu", device=device,
                             **options)
    meta = result.meta.get("outline_fit_fast") or {}
    return dict(label=label, image=image, values=file_values(image), seeds=box.get("seeds"), fit=box.get("fit"),
                trace=box.get("trace") or {}, reason="" if meta.get("fitted") else str(meta.get("reason", "not fitted")),
                finder={k: result.meta.get(k) for k in ("seeds_added_from_residual", "seeds_dropped_as_insignificant")},
                seconds=time.perf_counter() - started)


def redrawn_copy(real: dict, seed: int = 0):
    """The image redrawn from the fit's answer, stored like the file: the method's picture of the image plus
    noise, turned back over for dark holes, rounded to the file's grey step and cut to its type's range. The noise
    is the image's pixel-to-pixel noise less what the rounding adds again (a twelfth of the squared grey step).
    Returns (image, the noise added), or (None, None) when the fit did not get that far."""
    drawn, info = real["trace"].get("redrawn"), real["trace"].get("image")
    if drawn is None:
        return None, None
    image = real["image"]
    whole = np.issubdtype(image.dtype, np.integer)
    step = float(real["values"]["step"] or 1) if whole else 0.0
    grain = math.sqrt(max(drawn["pixel_noise"] ** 2 - step * step / 12.0, 0.0))
    signal = drawn["picture"] + grain * np.random.default_rng(seed).standard_normal(drawn["picture"].shape, dtype=np.float32)
    values = info["top"] - signal if info["inverted"] else signal
    if not whole:
        return values.astype(image.dtype), grain
    low = float(image.min())  # the file's values lie whole grey steps from its lowest
    limits = np.iinfo(image.dtype)
    values = low + step * np.rint((values - low) / step)
    return np.clip(values, limits.min, limits.max).astype(image.dtype), grain


# --------------------------------------------------------------------------------------- small helpers
def fmt(value, digits=2) -> str:
    """A number for the page (thousands separated); '-' where there is none."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:,.{digits}f}"


def sig(value, figures=3) -> str:
    """A number for the page to ``figures`` significant figures, never as a power of ten."""
    if value is None or not np.isfinite(value):
        return "-"
    if value == 0:
        return "0"
    return f"{value:,.{max(figures - 1 - int(math.floor(math.log10(abs(value)))), 0)}f}"


def share(part, whole) -> str:
    return "-" if not whole else f"{100.0 * part / whole:.2f} %"


def b64(array) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def reduced(image, longest=SMALL):
    """The image as block means, at most ``longest`` px along its longer side: (array, block size)."""
    f = max(1, int(math.ceil(max(image.shape) / longest)))
    h, w = image.shape[0] // f * f, image.shape[1] // f * f
    return image[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3), dtype=np.float64).astype(np.float32), f


def blocks(xy, values, shape, size, how="median"):
    """Per block of ``size`` px of the image, from the holes whose centre lies in it: the median of their values
    (the lower of two middle ones), or their number (``how='count'``). NaN where a block holds no value."""
    ny, nx = int(math.ceil(shape[0] / size)), int(math.ceil(shape[1] / size))
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    index = (np.clip(xy[:, 1] // size, 0, ny - 1).astype(np.int64) * nx
             + np.clip(xy[:, 0] // size, 0, nx - 1).astype(np.int64))
    if how == "count":
        return np.bincount(index, minlength=ny * nx).reshape(ny, nx).astype(np.float32)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    index, values = index[ok], values[ok]
    out = np.full(ny * nx, np.nan, dtype=np.float32)
    if len(index):
        order = np.lexsort((values, index))
        index, values = index[order], values[order]
        first = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
        last = np.r_[first[1:], len(index)]
        out[index[first]] = values[(first + last - 1) // 2]
    return out.reshape(ny, nx)


def density(xy, shape, spacing, block):
    """How densely the holes lie over the image, as a multiple of the image's usual density: (map, x and y of its
    first value, step). The holes are counted in cells of half a spacing and smoothed over two spacings, so that a
    regular array reads 1 everywhere whatever its spacing (counting per block would show stripes of its own); the
    part of the smoothing that falls outside the frame is taken out."""
    from scipy.ndimage import gaussian_filter

    cell = max(2.0, spacing / 2.0)
    ny, nx = int(math.ceil(shape[0] / cell)), int(math.ceil(shape[1] / cell))
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    # every hole is shared between the four cells round it by its distance to their centres: putting it into the
    # one cell it falls in shifts whole rows of holes by half a cell at once and leaves stripes
    u, v = xy[:, 0] / cell - 0.5, xy[:, 1] / cell - 0.5
    iu, iv = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    fu, fv = u - iu, v - iv
    count = np.zeros(ny * nx)
    for dy, wy in ((0, 1.0 - fv), (1, fv)):
        for dx, wx in ((0, 1.0 - fu), (1, fu)):
            count += np.bincount(np.clip(iv + dy, 0, ny - 1) * nx + np.clip(iu + dx, 0, nx - 1), weights=wy * wx,
                                 minlength=ny * nx)
    count = count.reshape(ny, nx)
    sigma = 2.0 * spacing / cell
    smooth = gaussian_filter(count, sigma, mode="constant") / np.maximum(
        gaussian_filter(np.ones_like(count), sigma, mode="constant"), 1e-9)
    usual = float(np.median(smooth))
    step = max(1, int(round(block / cell)))
    first = step // 2
    out = (smooth[first::step, first::step] / usual if usual > 0 else smooth[first::step, first::step]).astype(np.float32)
    # within two spacings of the frame the smoothing is one-sided: no value there
    y = (first + step * np.arange(out.shape[0]) + 0.5) * cell
    x = (first + step * np.arange(out.shape[1]) + 0.5) * cell
    edge = 2.0 * spacing
    out[(y < edge) | (y > shape[0] - edge), :] = np.nan
    out[:, (x < edge) | (x > shape[1] - edge)] = np.nan
    return out, (first + 0.5) * cell - 0.5, step * cell


def parts_apart(parts: dict, whole, gap: int = 3):
    """The average hole of each part of the image minus the whole image's (brightness and level matched by least
    squares), laid side by side as the parts lie in the image, NaN between them; and the largest of the parts'
    rms differences (pixel-weighted)."""
    rows, cols, side = int(parts["rows"]), int(parts["cols"]), whole.shape[0]
    out = np.full((rows * side + (rows - 1) * gap, cols * side + (cols - 1) * gap), np.nan)
    worst = float("nan")
    for k in range(rows * cols):
        stack, count = parts["stacks"][k], parts["pixels"][k]
        ok = np.isfinite(stack) & np.isfinite(whole)
        if ok.sum() > 10:
            slope, level = np.polyfit(whole[ok], stack[ok], 1)
            r, c = divmod(k, cols)
            apart = np.where(ok, stack - (slope * whole + level), np.nan)
            out[r * (side + gap):r * (side + gap) + side, c * (side + gap):c * (side + gap) + side] = apart
            rms = math.sqrt(float(np.nansum(count * apart ** 2)) / max(float(count[ok].sum()), 1.0))
            worst = rms if not worst >= rms else worst
    return out, worst


def span(arrays, palette, center=None, robust=True):
    """The colour range for a set of pictures drawn on one scale: (low, high). 'diverging': the same distance to
    both sides of ``center`` (0 by default)."""
    values = [np.asarray(a, dtype=float).ravel() for a in arrays if a is not None]
    values = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.zeros(0)
    if values.size == 0:
        return 0.0, 1.0
    if values.size > 2_000_000:
        values = values[:: values.size // 1_000_000]
    if palette == "diverging":
        c = 0.0 if center is None else float(center)
        reach = float(np.percentile(np.abs(values - c), 99.5 if robust else 100.0))
        reach = reach if reach > 0 else 1.0
        return c - reach, c + reach
    low, high = (float(v) for v in np.percentile(values, [0.5, 99.5] if robust else [0.0, 100.0]))
    return (low, high) if high > low else (low, low + 1.0)


def panel(label, array, palette, low, high, unit, grid=None, **marks) -> dict:
    """One picture of the page: a 2-D array on a colour scale from ``low`` to ``high``. ``grid`` =
    (x of the first column, step, y of the first row, step, what x and y are); ``marks``: ``points`` (K, 2),
    ``lines`` (a list of (K, 2) polylines), ``rects`` (rows x0, y0, x1, y1), all in the grid's coordinates."""
    x0, dx, y0, dy, axis = grid or (0.0, 1.0, 0.0, 1.0, "px")
    array = np.asarray(array)
    out = dict(label=label, w=int(array.shape[1]), h=int(array.shape[0]), d=b64(array), palette=palette,
               lo=float(low), hi=float(high), unit=unit, x0=float(x0), dx=float(dx), y0=float(y0), dy=float(dy), axis=axis)
    if marks.get("points") is not None and len(marks["points"]):
        out["points"] = b64(np.asarray(marks["points"], dtype=float).reshape(-1, 2))
    if marks.get("lines"):
        out["lines"] = [b64(np.asarray(line, dtype=float).reshape(-1, 2)) for line in marks["lines"]]
    if marks.get("rects"):
        out["rects"] = [[float(v) for v in r] for r in marks["rects"]]
    return out


def heat(fid, title, sides, get, palette, unit, *, grid=None, center=None, minus=False, note="", robust=True,
         marks=None, cell=None, limits=None):
    """A figure of pictures, one per side, on one colour scale (``limits``, or the pictures' own range).
    ``get(side)``: its array or None; ``grid`` and ``marks`` may be functions of the side. ``minus``: a third
    picture, your image minus the copy, where the two have the same shape."""
    got = [(s, get(s)) for s in sides]
    got = [(s, np.asarray(a, dtype=float)) for s, a in got if a is not None]
    if not got:
        return None
    low, high = limits or span([a for _, a in got], palette, center, robust)

    def of(thing, s):
        return thing(s) if callable(thing) else thing

    panels = [panel(s["label"], a, palette, low, high, unit, of(grid, s), **(of(marks, s) or {})) for s, a in got]
    if minus and len(got) == 2 and got[0][1].shape == got[1][1].shape:
        d = got[0][1] - got[1][1]
        panels.append(panel(f"{YOU} minus {COPY}", d, "diverging", *span([d], "diverging", robust=robust), unit,
                            of(grid, got[0][0])))
    return dict(id=fid, kind="heat", title=title, note=note, panels=panels, cell=cell)


def lines(fid, title, x, y, series, *, vlines=(), note="", log=False):
    """A line graph on one y axis. ``series``: (name, slot, xs, ys) with slot 1 = your image, 2 = the copy."""
    series = [dict(name=name, slot=slot, x=[float(v) for v in xs],
                   y=[float(v) if np.isfinite(v) else None for v in ys]) for name, slot, xs, ys in series if len(xs)]
    if not series:
        return None
    return dict(id=fid, kind="lines", title=title, x=x, y=y, series=series, note=note, log=bool(log),
                vlines=[dict(x=float(v), label=text) for v, text in vlines])


def hist(fid, title, x, sides, get, *, bins=60, vlines=(), note="", log=True, low=None, high=None, count="holes"):
    """A histogram per side over the same bins (``count`` per bin; ``log``: on a logarithmic axis)."""
    got = [(s, get(s)) for s in sides]
    got = [(s, np.asarray(v, dtype=float).ravel()) for s, v in got if v is not None]
    got = [(s, v[np.isfinite(v)]) for s, v in got]
    got = [(s, v) for s, v in got if v.size]
    if not got:
        return None
    pooled = np.concatenate([v for _, v in got])
    a = float(np.percentile(pooled, 0.05)) if low is None else float(low)
    b = float(np.percentile(pooled, 99.95)) if high is None else float(high)
    marks = [float(v) for v, _ in vlines if np.isfinite(v)]
    a, b = min([a] + marks), max([b] + marks)
    if not b > a:
        b = a + 1.0
    pad = 0.02 * (b - a)
    edges = np.linspace(a - pad, b + pad, int(bins) + 1)
    series = [dict(name=s["label"], slot=1 + i, counts=np.histogram(v, edges)[0].tolist(),
                   outside=int(np.sum((v < edges[0]) | (v > edges[-1])))) for i, (s, v) in enumerate(got)]
    return dict(id=fid, kind="hist", title=title, x=x, edges=[float(e) for e in edges], series=series, note=note,
                log=bool(log), count=count, vlines=[dict(x=float(v), label=text) for v, text in vlines if np.isfinite(v)])


def table(fid, title, columns, rows, note=""):
    return dict(id=fid, kind="table", title=title, columns=list(columns), rows=[[str(c) for c in r] for r in rows],
                note=note)


def pooled(figures) -> list:
    """Every picture's array once: the pictures of ``figures`` get the place of their array in the returned list
    in its stead (the same piece of the image is drawn in several steps)."""
    places, store = {}, []
    for figure in figures:
        for p in figure.get("panels", ()):
            if p["d"] not in places:
                places[p["d"]] = len(store)
                store.append(p["d"])
            p["d"] = places[p["d"]]
    return store


# -------------------------------------------------------------------------------- what a side holds
_sharp: dict = {}
STAGES = {"round": "round outline", "super": "rounded box", "full": "every number free"}


def templates(blur):
    """(a template for readings, one with the coverage grid for the sharp outline's picture)."""
    from subpx._outline.model import Template
    from subpx.outline import BLUR_FAMILIES, UPSAMPLE, _reading_template

    if blur not in _sharp:
        _sharp[blur] = Template(upsample=UPSAMPLE, size_px=BLUR_FAMILIES[blur], blur=blur)
    return _reading_template(blur), _sharp[blur]


def fit_reading(record, template, blur, pixel) -> dict:
    """A recorded page fit in the page's words and units: the outline, the blur, and how far the best match is
    from the average hole (grey levels; pixel-weighted rms over the bins the fit used, and the largest)."""
    from subpx.outline import blur_second

    theta = record["theta"]
    d = template.describe(theta, pixel)
    used = record["fitted"]
    count = np.where(used, record["pixels"], 0.0)
    apart = np.where(used, record["average"] - record["match"], np.nan)
    rms = math.sqrt(float(np.nansum(count * apart ** 2)) / max(float(count.sum()), 1.0))
    alone = float(record["noise"]) / math.sqrt(max(float(count.sum()) / max(int(used.sum()), 1), 1e-12))
    inside = record["average"][used]
    # the same difference averaged round the centre: per ring of a quarter pixel, the pixel-weighted mean
    half = record["average"].shape[0] // 2
    offset = (np.arange(record["average"].shape[0]) - half) / record["bins_per_px"]
    ring = np.rint(np.hypot(offset[None, :], offset[:, None]) * 4).astype(int)
    weight = np.bincount(ring[used], weights=count[used])
    rings = np.bincount(ring[used], weights=(count * np.nan_to_num(apart))[used]) / np.maximum(weight, 1e-12)
    keep = (weight > 0) & (np.arange(len(weight)) <= 4 * half / record["bins_per_px"])
    course = []
    for r in record["iterations"]:
        width = template.blur_fwhm_px(r["theta"])
        course.append(dict(stage=r["stage"], iteration=r["iteration"], cd=template.cd_px(r["theta"]) * pixel,
                           blur=0.5 * (width[0] + width[1]) * pixel, accepted=r["accepted"],
                           apart=math.sqrt(max(r["cost"], 0.0) / max(float(count.sum()), 1.0)) * float(record["noise"])))
    return dict(theta=theta, cd=d["cd_nm"], extent=(d["extent_x_nm"], d["extent_y_nm"]), p=d["superellipse"]["p"],
                turn=d["superellipse"]["phi_deg"], blur=tuple(d["blur_fwhm_nm"]), second=blur_second(theta, blur),
                scale=record["scale"], level=record["const"], apart=apart, rms=rms, largest=float(np.nanmax(np.abs(apart))),
                hole=float(inside.max() - inside.min()), noise_alone=alone, times=rms / alone if alone > 0 else float("nan"),
                ring_px=np.flatnonzero(keep) / 4.0, ring_apart=rings[keep],
                stages=record["stages"], course=course, bins=int(used.sum()), per_bin=float(count.sum()) / max(int(used.sum()), 1))


def parameter_rows(theta, template, pixel, unit):
    """Every fitted number in plain words with its value and its limits: (name, value, lowest, highest, at a limit)."""
    lo, hi = template.bounds()
    j = template.n_shape
    names = [("outline: half width along x", "len"), ("outline: half width along y", "len"),
             ("outline: corner sharpness p (2 = ellipse, large = box)", "p"), ("outline: turned by", "deg")]
    names += [(f"outline: finer shape, term {k} ({part})", "plain") for k in template.ks for part in ("cos", "sin")]
    if template.blur == "gauss2":
        names += [("blur, first part: width along x (sigma)", "len"), ("blur, first part: width along y (sigma)", "len"),
                  ("blur, second part: width along x (sigma)", "len"), ("blur, second part: width along y (sigma)", "len"),
                  ("blur, second part: its share of the light", "share")]
    else:
        names += [("blur: Gaussian width along x (sigma)", "len"), ("blur: Gaussian width along y (sigma)", "len"),
                  ("blur: lens cut-off (cycles per pixel)", "exp")]

    def shown(v, kind):
        if kind == "len":
            return f"{math.exp(v) * pixel:.2f} {unit}"
        if kind == "p":
            return f"{1 + math.exp(v):.2f}"
        if kind == "deg":
            return f"{math.degrees(v):+.1f} deg"
        if kind == "share":
            return f"{100 / (1 + math.exp(-v)):.1f} %"
        if kind == "exp":
            return f"{math.exp(v):.3f}"
        return f"{v:+.4f}"

    rows = []
    for k, (name, kind) in enumerate(names[:len(theta)]):
        at = "at its lowest" if theta[k] <= lo[k] + 1e-4 else "at its highest" if theta[k] >= hi[k] - 1e-4 else ""
        rows.append((name, shown(theta[k], kind), shown(lo[k], kind), shown(hi[k], kind), at))
    assert j <= len(rows)
    return rows


def material(side: dict, blur: str, pixel: float, unit: str, places, block: int) -> dict:
    """Everything the page draws for one side (the image or its copy), cut down to what is drawn. A step the fit
    did not reach is missing."""
    t, image = side["trace"], side["image"]
    info = t.get("image") or {"inverted": False, "top": float(np.max(image))}
    signal = (info["top"] - image.astype(np.float32)) if info["inverted"] else image.astype(np.float32)
    shape = signal.shape
    m = dict(label=side["label"], reason=side["reason"], values=side["values"], inverted=info["inverted"], shape=shape,
             finder=side["finder"], seconds=side["seconds"], block=block)
    small, f = reduced(signal)
    m["small"], m["small_grid"] = small, ((f - 1) / 2, f, (f - 1) / 2, f, "image px")
    m["pieces"] = [signal[y:y + s, x:x + s] for _, y, x, s in places]
    stored = np.asarray(image)
    m["grey"] = stored.ravel()[:: max(1, stored.size // 2_000_000)].astype(np.float32)
    seeds = side["seeds"]
    if seeds is None or "window" not in t:
        return m
    w = t["window"]
    R = int(w["window_px"])
    m.update(seeds=seeds, holes=len(seeds), spacing=float(w["spacing_px"]), window=R, reach=float(w["reach_px"]),
             bins_per_px=int(w["bins_per_px"]))
    m["density"], first, step = density(seeds, shape, m["spacing"], block)
    m["density_grid"] = (first, step, first, step, "image px")
    m["piece_seeds"] = [seeds[(seeds[:, 0] >= x - 0.5) & (seeds[:, 0] < x + s - 0.5) & (seeds[:, 1] >= y - 0.5)
                              & (seeds[:, 1] < y + s - 0.5)] for _, y, x, s in places]
    near = w["nearest_px"]
    m["nearest"] = near[near < 1e10]
    m["alone"] = int(np.sum(near >= 1e10))
    _, y, x, s = places[0]
    middle = seeds[np.argmin(np.hypot(seeds[:, 0] - (x + s / 2), seeds[:, 1] - (y + s / 2)))]
    cx, cy = np.rint(middle)
    m["window_rects"] = [[cx - R - 0.5, cy - R - 0.5, cx + R + 0.5, cy + R + 0.5],
                         [cx - m["reach"], cy - m["reach"], cx + m["reach"], cy + m["reach"]]]
    half = (int(w["stack_bins"]) // 2) / m["bins_per_px"]
    m["stack_grid"] = (-half, 1.0 / m["bins_per_px"], -half, 1.0 / m["bins_per_px"], CENTRE)
    if "average_all" not in t:
        return m
    a = t["average_all"]
    inside, kept = a["inside"], a["kept"]
    low, high = a["brightness_limits"]
    dim = inside & ((a["brightness"] <= low) | (a["brightness"] >= high))
    loud = inside & ~dim & ~(a["leftover"] < a["leftover_limit"] * a["typical_leftover"])
    m.update(all_stack=a["stack"], all_pixels=a["pixels"], inside=int(inside.sum()), kept_1=int(kept.sum()),
             inside_mask=inside,
             near_frame=int((~inside).sum()), dim_1=int(dim.sum()), loud_1=int(loud.sum()),
             other_1=int((inside & ~kept & ~dim & ~loud).sum()),
             brightness_1=np.where(inside, a["brightness"], np.nan), level_1=np.where(inside, a["level"], np.nan),
             leftover_1=np.where(inside, a["leftover"], np.nan), limits_1=(low, high),
             leftover_limit_1=a["leftover_limit"] * a["typical_leftover"])
    for key in ("brightness_1", "level_1", "leftover_1"):
        m[key + "_map"] = blocks(seeds, m[key], shape, block)
    m["dropped_1_map"] = blocks(seeds[inside & ~kept], None, shape, block, "count")
    if "average_1" not in t:
        return m
    m.update(stack_1=t["average_1"]["stack"], pixels_1=t["average_1"]["pixels"], noise_1=float(t["average_1"]["noise"]))
    reading, sharp = templates(blur)
    nb_half = None
    for name, key in (("fit_1", "first"), ("fit_2", "final")):
        record = t.get(name)
        if key == "final" and record is not None and t.get("exchange"):
            m["exchange"] = dict(kept=bool(t["exchange"]["kept"]), cost_first=float(record["cost"]),
                                 cost_second=float(t["exchange"]["cost"]),
                                 iterations=int(t["exchange"]["stages"][-1][1]))
            m["first_end"] = fit_reading(record, reading, blur, pixel)
            if t["exchange"]["kept"]:
                record = t["exchange"]
        if record is None:
            break
        r = fit_reading(record, reading, blur, pixel)
        nb_half = (record["average"].shape[0] // 2) / record["bins_per_px"]
        r.update(average=record["average"], match=record["match"], alone=record["alone"],
                 neighbours=record["match"] - record["alone"] - record["const"],
                 pixels=np.where(record["fitted"], record["pixels"], np.nan), per_class=record["holes_per_class"],
                 grid=(-nb_half, 1.0 / record["bins_per_px"], -nb_half, 1.0 / record["bins_per_px"], CENTRE),
                 outline=np.column_stack(reading.outline_xy(record["theta"])), scan=record["grid"])
        # the blur by itself and the sharp outline, +-5 px round the centre at 8 samples per pixel
        c, k = reading.N // 2, 5 * reading.U
        r["kernel"] = reading.blur_kernel(record["theta"])[c - k:c + k + 1, c - k:c + k + 1]
        r["sharp"] = sharp.indicator(record["theta"])[c - k:c + k + 1, c - k:c + k + 1]
        r["fine_grid"] = (-5.0, 1.0 / reading.U, -5.0, 1.0 / reading.U, CENTRE)
        # where the neighbours stand: +-reach px round the stacked hole, 4 samples per pixel
        nm, per = record["neighbour_map"], int(record["neighbour_bins_per_px"])
        c, k = nm.shape[0] // 2, int(m["reach"] * per) // 2 * 2
        cut = nm[c - k:c + k, c - k:c + k]
        r["neighbour_map"] = cut.reshape(k, 2, k, 2).sum(axis=(1, 3))
        r["neighbour_grid"] = (-k / per + 1.0 / per, 2.0 / per, -k / per + 1.0 / per, 2.0 / per, "px from the averaged hole")
        m[key] = r
        if key == "first":
            if "holes" not in t:
                break
            h = t["holes"]
            # the holes that can be averaged (a window away from the frame): what the maps and numbers are about
            move = np.where(m["inside_mask"][:, None], h["centers_xy"] - seeds, np.nan)
            m.update(move_x=move[:, 0], move_y=move[:, 1], move=np.hypot(move[:, 0], move[:, 1]),
                     brightness_2=np.where(m["inside_mask"], h["brightness"], np.nan),
                     level_2=np.where(m["inside_mask"], h["level"], np.nan),
                     leftover_2=np.where(m["inside_mask"], h["leftover"], np.nan), passes=h["passes"])
            for field in ("move_x", "move_y", "brightness_2", "level_2", "leftover_2"):
                m[field + "_map"] = blocks(seeds, m[field], shape, block)
            m["picture_pieces"] = [h["picture"][y:y + s, x:x + s] for _, y, x, s in places]
            m["left_pieces"] = [signal[y:y + s, x:x + s] - h["picture"][y:y + s, x:x + s] for _, y, x, s in places]
            edge = int(math.ceil(m["reach"]))  # away from the frame, where the picture has every hole
            inner = (slice(edge, -edge), slice(edge, -edge)) if min(shape) > 4 * edge else (slice(None), slice(None))
            m["left_rms"] = float(np.sqrt(np.mean((signal - h["picture"])[inner] ** 2, dtype=np.float64)))
            if "kept_2" in t and "average_2" in t:
                k2 = t["kept_2"]
                good = t["average_2"]["kept"]
                dim2 = k2["inside"] & ~k2["usual_brightness"]
                loud2 = k2["inside"] & k2["usual_brightness"] & ~k2["usual_leftover"]
                m.update(kept_2=int(good.sum()), dim_2=int(dim2.sum()), loud_2=int(loud2.sum()),
                         dropped_2_map=blocks(seeds[k2["inside"] & ~good], None, shape, block, "count"),
                         stack_2=t["average_2"]["stack"], pixels_2=t["average_2"]["pixels"],
                         noise_2=float(t["average_2"]["noise"]),
                         leftover_limit_2=float(k2["leftover_limit"] * k2["typical_leftover"]))
                if "average_2_parts" in t:
                    m["parts"], m["parts_worst"] = parts_apart(t["average_2_parts"], t["average_2"]["stack"])
    fit = side["fit"]
    if fit is not None and "final" in m:
        m["result"] = dict(cd=fit.outline_cd_px * pixel, cd_x=fit.cd_x_px * pixel, cd_y=fit.cd_y_px * pixel,
                           converged=bool(fit.summary["converged"]), iterations=int(fit.summary["fit_iterations"]),
                           leftover=float(fit.summary["leftover_chi2"]), stacked=int(fit.summary["holes_stacked"]))
        m["parameters"] = parameter_rows(m["final"]["theta"], reading, pixel, unit)
    if "redrawn" in t:
        m["pixel_noise"], m["hole_leftover"] = float(t["redrawn"]["pixel_noise"]), float(t["redrawn"]["leftover"])
    return m


def piece_places(real: dict, side: int):
    """Where the three full-size pieces are cut: the middle, the top left corner, and the piece of the image where
    the most is left over after the method's picture is taken away (the bottom right corner without a picture).
    Rows (name, y, x, side)."""
    h, w = real["image"].shape
    side = int(min(side, h, w))
    out = [("the middle", (h - side) // 2, (w - side) // 2, side), ("the top left corner", 0, 0, side)]
    picture = (real["trace"].get("holes") or {}).get("picture")
    spot = (h - side, w - side)
    name = "the bottom right corner"
    if picture is not None and h >= 3 * side and w >= 3 * side:
        info = real["trace"]["image"]
        signal = (info["top"] - real["image"].astype(np.float32)) if info["inverted"] else real["image"].astype(np.float32)
        ny, nx = h // side - 2, w // side - 2  # tiles one piece away from the frame
        left = np.abs(signal - picture)[side:side + ny * side, side:side + nx * side]
        tiles = left.reshape(ny, side, nx, side).mean(axis=(1, 3), dtype=np.float64)
        ty, tx = np.unravel_index(int(np.argmax(tiles)), tiles.shape)
        spot, name = (side + ty * side, side + tx * side), "where the most is left over"
    out.append((name, int(spot[0]), int(spot[1]), side))
    return out


# ------------------------------------------------------------------------------------------ the page
def text_of(m, key, form, default="-"):
    return form(m[key]) if m is not None and key in m else default


def settled(stages):
    """'12, settled' or '30, cut off' for a fit's last stage."""
    _, count, stopped = stages[-1]
    return f"{count}, settled" if stopped else f"{count}, cut off at its limit"


def image_page(real: dict, copy: dict | None, places, pixel: float, unit: str, blur: str, grain) -> dict:
    """One image's page: the steps with their numbers and figures, your image beside its redrawn copy."""
    sides = [real] + ([copy] if copy is not None else [])
    steps, summary = [], []

    def has(key):
        return any(key in s for s in sides)

    def numbers(*rows):
        return [[label, text_of(real, key, form), text_of(copy, key, form) if copy is not None else ""]
                for label, key, form in rows]

    def add(n, title, does, look, rows, figures, note=""):
        steps.append(dict(n=n, title=title, does=does, look=look, numbers=rows, note=note,
                          figures=[f for f in figures if f is not None]))

    def line(n, title, key, form):
        summary.append([n, title, text_of(real, key, form, "not reached"),
                        text_of(copy, key, form, "not reached") if copy is not None else ""])

    def piece_grid(k):
        _, y, x, _ = places[k]
        return (x, 1.0, y, 1.0, "image px")

    grey = "grey levels"
    v = real["values"]
    turned = (" Dark holes: the image is turned over first (its highest value minus the image), so that holes are "
              "bright in everything below.") if real["inverted"] else ""
    # ---- 1
    rects = [[x - 0.5, y - 0.5, x + s - 0.5, y + s - 0.5] for _, y, x, s in places]
    add(1, "Image as read",
        "The image as the program reads it from the file, before anything is done to it." + turned
        + " The redrawn copy is drawn from the fit's answer at the end (step 10) and then goes through the same steps.",
        "Tops cut off flat (many pixels at the highest value the file can hold), few different grey values, areas "
        "without holes, a pattern that changes across the image.",
        numbers(("how the file stores it", "values", lambda q: q["dtype"]),
                ("lowest to highest value", "values", lambda q: f"{fmt(q['low'], 0)} to {fmt(q['high'], 0)}"),
                ("different grey values", "values", lambda q: fmt(q["levels"], 0)),
                ("grey step (values lie this far apart)", "values", lambda q: "-" if q["step"] is None else fmt(q["step"], 0)),
                ("pixels at the highest value the file type holds", "values", lambda q: f"{100 * q['top']:.3f} %"),
                ("pixels at zero", "values", lambda q: f"{100 * q['zero']:.3f} %")),
        [heat("1a", "The whole image, reduced (the squares are the three pieces shown at full size below)", sides,
              lambda s: s["small"], "grey", grey, grid=lambda s: s["small_grid"], marks=lambda s: dict(rects=rects))]
        + [heat(f"1{'bcd'[k]}", f"At full size: {places[k][0]}", sides, lambda s, k=k: s["pieces"][k], "grey", grey,
                grid=piece_grid(k), minus=True, robust=False) for k in range(3)]
        + [hist("1e", "How often each grey value occurs (values as stored in the file)", "grey value", sides,
                lambda s: s["grey"], bins=128, log=True, count="pixels")])
    line(1, "Image as read", "values",
         lambda q: f"{q['dtype']}, {fmt(q['levels'], 0)} grey values, {100 * q['top']:.2f} % cut off at the top")
    # ---- 2
    if has("seeds"):
        add(2, "Holes found",
            "The program looks for the holes, with the same search and options as the folder report. Every dot is a "
            "hole it found. Only these holes exist for the method: a hole without a dot is not in its picture of "
            "the image.",
            "Holes without a dot, dots where there is no hole, parts of the image with far fewer holes.",
            numbers(("holes found", "holes", lambda q: fmt(q, 0)),
                    ("of them at least a window away from the frame (these can be averaged)", "inside", lambda q: fmt(q, 0)),
                    ("found on a second look at what the first holes left over", "finder",
                     lambda q: fmt(q.get("seeds_added_from_residual"), 0)),
                    ("dropped again as too faint", "finder", lambda q: fmt(q.get("seeds_dropped_as_insignificant"), 0))),
            [heat(f"2{'abc'[k]}", f"Holes found: {places[k][0]}", sides, lambda s, k=k: s["pieces"][k], "grey", grey,
                  grid=piece_grid(k), robust=False,
                  marks=lambda s, k=k: dict(points=s["piece_seeds"][k]) if "piece_seeds" in s else {})
             for k in range(3)]
            + [heat("2d", "How densely the holes found lie over the whole image, as a multiple of the usual",
                    sides, lambda s: s.get("density"), "diverging", "x the usual", grid=lambda s: s["density_grid"],
                    limits=(0.8, 1.2),
                    note="1 = as everywhere else; lower = holes are missing there. Smoothed over two spacings; no "
                         "value within two spacings of the frame. An even picture means holes were found "
                         "everywhere alike.")])
        line(2, "Holes found", "holes", lambda q: fmt(q, 0))
    # ---- 3
    if has("spacing"):
        add(3, "Spacing and window",
            "The usual distance from a hole to its nearest neighbour is the spacing. Every hole gets a square window "
            "round it, 0.75 spacings to each side (rounded up to whole pixels). The window is what is averaged and "
            "fitted. Neighbours up to 13 px beyond the window are counted in as light that reaches into it.",
            "Two or more peaks in the distances (two patterns in one image), a window that cuts into the "
            "neighbouring holes more than it should.",
            numbers(("spacing", "spacing", lambda q: f"{q:.2f} px = {q * pixel:.1f} {unit}"),
                    ("window", "window", lambda q: f"+-{q} px ({2 * q + 1} x {2 * q + 1} px)"),
                    ("neighbours counted up to", "reach", lambda q: f"+-{q:.0f} px"),
                    ("holes without any neighbour nearby", "alone", lambda q: fmt(q, 0))),
            [hist("3a", "Distance from each hole to its nearest neighbour", "px", sides, lambda s: s.get("nearest"),
                  log=True, vlines=[(real.get("spacing", float("nan")), "spacing")]),
             heat("3b", "One hole's window (inner square) and how far its neighbours are counted (outer square)",
                  sides, lambda s: s["pieces"][0] if "window_rects" in s else None, "grey", grey, grid=piece_grid(0),
                  robust=False, marks=lambda s: dict(rects=s["window_rects"]))])
        line(3, "Spacing and window", "spacing", lambda q: f"{q:.2f} px")
    # ---- 4
    if has("all_stack"):
        mid = lambda s, axis: (s["all_stack"][s["all_stack"].shape[0] // 2, :] if axis == "x"  # noqa: E731
                               else s["all_stack"][:, s["all_stack"].shape[1] // 2])
        offsets = lambda s: s["stack_grid"][0] + s["stack_grid"][1] * np.arange(s["all_stack"].shape[0])  # noqa: E731
        add(4, "First average hole",
            "The windows of all holes that lie fully inside the image are laid on top of each other and averaged. "
            "A hole's centre rarely sits on a pixel's centre, so every pixel is put where it lies relative to its "
            "hole's centre, in steps of a quarter pixel: the average hole has 4 x 4 bins per pixel.",
            "An average hole that is not round or not in the middle, bins with very few pixels beside full ones "
            "(a spacing of a whole number of pixels fills only some bins).",
            numbers(("holes averaged", "inside", lambda q: fmt(q, 0)),
                    ("average hole, brightest minus darkest bin", "all_stack", lambda q: f"{np.nanmax(q) - np.nanmin(q):.2f} grey levels"),
                    ("pixels in a bin, fewest to most", "all_pixels",
                     lambda q: f"{fmt(float(q[q > 0].min()) if (q > 0).any() else 0, 0)} to {fmt(float(q.max()), 0)}"),
                    ("bins that got no pixel", "all_pixels", lambda q: share(int((q == 0).sum()), q.size))),
            [heat("4a", "Average hole", sides, lambda s: s.get("all_stack"), "grey", grey,
                  grid=lambda s: s["stack_grid"], minus=True, robust=False),
             heat("4b", "Pixels that went into each bin", sides, lambda s: s.get("all_pixels"), "sequential", "pixels",
                  grid=lambda s: s["stack_grid"], robust=False),
             lines("4c", "Average hole along x through its centre", CENTRE, grey,
                   [(s["label"], 1 + i, offsets(s), mid(s, "x")) for i, s in enumerate(sides) if "all_stack" in s],
                   note="Small steps in the curve come from the quarter-pixel bins (a bin holds pixels up to an "
                        "eighth of a pixel away from its place); the copy has them too."),
             lines("4d", "Average hole along y through its centre", CENTRE, grey,
                   [(s["label"], 1 + i, offsets(s), mid(s, "y")) for i, s in enumerate(sides) if "all_stack" in s])])
        line(4, "First average hole", "all_stack", lambda q: f"{np.nanmax(q) - np.nanmin(q):.1f} grey levels bright")
    # ---- 5
    by_construction = ("The copy is drawn with every hole's own position, brightness and level, so the brightness and "
                       "level maps look alike on both sides whatever the image holds. The leftover, the dropped holes "
                       "and the average holes are what can differ. The maps hold only the holes that can be averaged "
                       "(at least a window away from the frame).")
    at_frame = (" Right at the frame the method's picture is poor, in your image and in the copy alike: a hole cut by "
                "the frame is placed by the part of it that is visible, and holes beyond the frame, whose light "
                "reaches into the image, are not in the picture at all (they are in your image, not in the copy). "
                "The corner piece shows it.")
    block_grid = lambda s: ((s["block"] - 1) / 2, s["block"], (s["block"] - 1) / 2, s["block"], "image px")  # noqa: E731
    per_block = f"median of the holes in each block of {real['block']} x {real['block']} px"
    if has("brightness_1"):
        add(5, "Each hole against the average hole",
            "Each hole's window is compared with the average hole: how many times as bright it is (brightness), how "
            "much higher or lower it sits (level), and what is then still left (leftover: the rms difference over "
            "the window). Holes whose brightness is below 0.4 or above 2.5 times the usual one, or whose leftover is "
            "more than 2 times the usual one, are dropped. The rest are averaged again: that average is what the "
            "first fit works on.",
            "A leftover map that is not even (a blur or a hole size that changes across the image), stripes or "
            "steps in brightness or level, many dropped holes or dropped holes in one place.",
            numbers(("holes kept", "kept_1", lambda q: fmt(q, 0)),
                    ("dropped: less than a window from the frame", "near_frame", lambda q: fmt(q, 0)),
                    ("dropped: brightness below 0.4 or above 2.5 times the usual", "dim_1", lambda q: fmt(q, 0)),
                    ("dropped: leftover above 2 times the usual", "loud_1", lambda q: fmt(q, 0)),
                    ("usual leftover of a kept hole = the noise the fit takes", "noise_1", lambda q: f"{sig(q)} grey levels")),
            [heat("5a", f"Leftover of each hole ({per_block})", sides, lambda s: s.get("leftover_1_map"), "sequential",
                  grey, grid=block_grid),
             heat("5b", f"Brightness of each hole, as a multiple of the usual one ({per_block})", sides,
                  lambda s: s.get("brightness_1_map"), "diverging", "x the usual", grid=block_grid, center=1.0),
             heat("5c", f"Level of each hole ({per_block})", sides, lambda s: s.get("level_1_map"), "diverging", grey,
                  grid=block_grid, center=float(np.nanmedian(real["level_1"])) if "level_1" in real else 0.0),
             heat("5d", "Dropped holes per block (holes near the frame not counted)", sides,
                  lambda s: s.get("dropped_1_map"), "sequential", "holes", grid=block_grid, robust=False),
             hist("5e", "Leftover of each hole", grey, sides, lambda s: s.get("leftover_1"),
                  vlines=[(s["leftover_limit_1"], f"dropped above ({s['label']})") for s in sides if "leftover_limit_1" in s]),
             hist("5f", "Brightness of each hole, as a multiple of the usual one", "x the usual", sides,
                  lambda s: s.get("brightness_1"), vlines=[(0.4, "dropped below"), (2.5, "dropped above")]),
             heat("5g", "Average hole of the kept holes (the first fit's data)", sides, lambda s: s.get("stack_1"),
                  "grey", grey, grid=lambda s: s["stack_grid"], minus=True, robust=False)],
            note=by_construction)
        line(5, "Each hole against it", "kept_1", lambda q: f"{fmt(q, 0)} holes kept")

    def fit_figures(n, key, first):
        """The figures of a page fit (step 6 or 9)."""
        r = lambda s: s.get(key)  # noqa: E731
        got = [s for s in sides if key in s]
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        figs = []

        def fid():
            return f"{n}{next(letters)}"

        def course(field):
            out = []
            for i, s in enumerate(sides):
                if key in s:
                    rows = s[key]["course"]
                    out.append((s["label"], 1 + i, np.arange(len(rows)), [row[field] for row in rows]))
            return out

        marks = []
        if got:
            rows = got[0][key]["course"]
            starts = [k for k, row in enumerate(rows) if row["iteration"] == 0 and k]
            marks = [(k, f"{STAGES.get(rows[k]['stage'], rows[k]['stage'])} from here") for k in starts]
            if not first:
                marks.append((LIMIT, f"limit of {LIMIT}"))
        count = "iteration (0 = the start)" if not first else "iteration, both stages in a row (0 = the start)"
        figs.append(lines(fid(), f"CD after each iteration ({unit})", count, f"CD ({unit})", course("cd"), vlines=marks))
        figs.append(lines(fid(), f"Blur width at half height after each iteration, mean of x and y ({unit})", count,
                          f"blur width ({unit})", course("blur"), vlines=marks))
        figs.append(lines(fid(), "How far the best match is from the average hole after each iteration (grey levels)",
                          count, "rms distance (grey levels)", course("apart"), vlines=marks, log=True))
        figs.append(heat(fid(), "Average hole minus best match", sides, lambda s: r(s)["apart"] if r(s) else None,
                         "diverging", grey, grid=lambda s: r(s)["grid"],
                         note="The picture that decides. Noise alone leaves fine grain without a shape; rings, a "
                              "cross or light at the edges are what the outline and blur found cannot describe."))
        figs.append(lines(fid(), "Average hole minus best match, averaged round the centre (grey levels)",
                          "distance from the hole's centre (px)", grey,
                          [(s["label"], 1 + i, s[key]["ring_px"], s[key]["ring_apart"]) for i, s in enumerate(sides) if key in s],
                          note="The picture above, averaged over all directions: a ring that is hard to see in "
                               "the picture stands out here as a wave, where noise alone stays near zero."))
        figs.append(heat(fid(), "Average hole, with the outline found", sides, lambda s: r(s)["average"] if r(s) else None,
                         "grey", grey, grid=lambda s: r(s)["grid"], robust=False,
                         marks=lambda s: dict(lines=[np.vstack([r(s)["outline"], r(s)["outline"][:1]])])))
        figs.append(heat(fid(), "Best match = the hole by itself + light from the neighbours + a level", sides,
                         lambda s: r(s)["match"] if r(s) else None, "grey", grey, grid=lambda s: r(s)["grid"],
                         robust=False))
        figs.append(heat(fid(), "The hole by itself: the sharp outline seen through the blur", sides,
                         lambda s: r(s)["alone"] if r(s) else None, "grey", grey, grid=lambda s: r(s)["grid"],
                         robust=False))
        figs.append(heat(fid(), "Light from the neighbours inside the window", sides,
                         lambda s: r(s)["neighbours"] if r(s) else None, "grey", grey, grid=lambda s: r(s)["grid"],
                         robust=False))
        figs.append(heat(fid(), "The sharp outline (1 inside, 0 outside)", sides, lambda s: r(s)["sharp"] if r(s) else None,
                         "grey", "inside", grid=lambda s: r(s)["fine_grid"], robust=False, cell=3))
        figs.append(heat(fid(), "The blur by itself (how one point of light is spread)", sides,
                         lambda s: r(s)["kernel"] if r(s) else None, "sequential", "share of the light per sample",
                         grid=lambda s: r(s)["fine_grid"], robust=False, cell=3))
        figs.append(heat(fid(), "Where the neighbours of an averaged hole stand (the hole itself in the middle)", sides,
                         lambda s: r(s)["neighbour_map"] if r(s) else None, "sequential", "holes x brightness",
                         grid=lambda s: r(s)["neighbour_grid"], cell=2))
        figs.append(heat(fid(), "Averaged holes by where their centre lies inside a pixel (4 x 4 quarter-pixel places)",
                         sides, lambda s: r(s)["per_class"] if r(s) else None, "sequential", "holes", robust=False,
                         grid=(0.0, 0.25, 0.0, 0.25, "px inside the pixel"), cell=28))
        if first:
            def scan(s):
                if not r(s) or r(s)["scan"] is None:
                    return None
                g = r(s)["scan"]
                return np.sqrt(np.maximum(g["cost"], 0.0) / g["weight"])

            def scan_grid(s):  # x: the blur's sigma, y: the round hole's diameter, both in the page's unit
                g = r(s)["scan"]
                wide, size = g["widths_px"] * pixel, 2.0 * g["sizes_px"] * pixel
                return (wide[0], wide[1] - wide[0], size[0], size[1] - size[0],
                        f"x: blur sigma ({unit}), y: hole diameter ({unit})")

            def scan_marks(s):
                g = r(s)["scan"]
                return dict(points=[[g["widths_px"][g["pick"][1]] * pixel, 2.0 * g["sizes_px"][g["pick"][0]] * pixel]])

            figs.append(heat(fid(), "The 15 hole sizes x 12 blur widths tried first: rms distance to the average hole "
                                    "(the dot: the start taken). Down: larger hole; right: wider blur",
                             sides, scan, "sequential", grey, cell=18, marks=scan_marks, grid=scan_grid,
                             note="A valley of low values running across the table means that a larger hole with a "
                                  "narrower blur fits nearly as well as a smaller hole with a wider blur."))
        return figs

    def fit_numbers(key):
        return numbers(
            ("CD (diameter of the circle with the outline's area)", key, lambda q: f"{q['cd']:.2f} {unit}"),
            ("outline: width along x, along y", key, lambda q: f"{q['extent'][0]:.2f}, {q['extent'][1]:.2f} {unit}"),
            ("outline: corner sharpness p (2 = ellipse)", key, lambda q: f"{q['p']:.2f}"),
            ("blur width at half height: along x, along y", key, lambda q: f"{q['blur'][0]:.1f}, {q['blur'][1]:.1f} {unit}"),
            ("blur: second part", key, lambda q: q["second"] or "-"),
            ("iterations", key, lambda q: "; ".join(f"{STAGES.get(name, name)}: {count}, "
                                                    f"{'settled' if ok else 'cut off at its limit'}"
                                                    for name, count, ok in q["stages"])),
            ("brightness of an open area above the level", key, lambda q: f"{sig(q['scale'])} grey levels"),
            ("level", key, lambda q: f"{sig(q['level'])} grey levels"),
            ("average hole, brightest minus darkest bin", key, lambda q: f"{sig(q['hole'])} grey levels"),
            ("best match to average hole: rms distance", key,
             lambda q: f"{sig(q['rms'])} grey levels = {100 * q['rms'] / q['hole']:.3f} % of the hole"),
            ("... largest distance in one bin", key, lambda q: f"{sig(q['largest'])} grey levels"),
            ("... what noise alone would leave", key, lambda q: f"{sig(q['noise_alone'])} grey levels"),
            ("... rms distance as a multiple of that (1 = at the noise)", key, lambda q: f"{sig(q['times'])} x"))

    fit_look = ("Average hole minus best match: a shape (rings, a cross, light at the window's edge) instead of even "
                "grain. Curves that are still moving where they stop. A CD that moves one way while the blur moves "
                "the other. Numbers very different from the copy's.")
    # ---- 6
    if has("first"):
        add(6, "First fit",
            "One sharp outline and one blur are looked for such that the outline seen through the blur, with the "
            "light of the neighbouring holes and a level added, matches the average hole. First 15 sizes times 12 "
            "blur widths are tried and the closest is taken as the start. Then the match is improved with a round "
            "outline (at most 10 iterations), and then with a rounded box that may differ along x and y (at most 12).",
            fit_look, fit_numbers("first"), fit_figures(6, "first", True))
        line(6, "First fit", "first", lambda q: f"CD {q['cd']:.2f} {unit}, {sig(q['times'])} x the noise")
    # ---- 7
    if has("move"):
        last = lambda q: q[-1]  # noqa: E731
        add(7, "Each hole fitted on the image",
            "With the first fit's outline and blur, every hole's position, brightness and level are fitted on the "
            "image itself, twice over all holes. Each hole is fitted against the image with the light of all other "
            "holes taken away. The size is held: every hole has the image's one outline. A hole may move a quarter "
            "pixel at most per round.",
            "Image minus the method's picture: rings round the holes (their size or blur differs from the image's "
            "one), whole holes left over (holes that were not found), a pattern that changes across the image. "
            "Many holes that moved as far as they may.",
            numbers(("moved from where it was found: usual, largest 1 %", "move",
                     lambda q: f"{np.nanmedian(q):.3f}, {np.nanpercentile(q, 99):.3f} px"),
                    ("holes that moved as far as they may (0.5 px)", "move",
                     lambda q: share(int((q > 0.49).sum()), int(np.isfinite(q).sum()))),
                    ("brightness: usual, spread from hole to hole", "brightness_2",
                     lambda q: f"{sig(np.nanmedian(q))} grey levels, "
                               f"{100 * 1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q))) / np.nanmedian(q):.1f} %"),
                    ("level: usual, spread from hole to hole", "level_2",
                     lambda q: f"{sig(np.nanmedian(q))}, {sig(1.4826 * np.nanmedian(np.abs(q - np.nanmedian(q))))} grey levels"),
                    ("leftover of a hole's window: usual", "leftover_2", lambda q: f"{sig(np.nanmedian(q))} grey levels"),
                    ("image minus the method's picture, away from the frame (rms)", "left_rms",
                     lambda q: f"{sig(q)} grey levels"),
                    ("second round: largest 1 % of the moves, all holes", "passes", lambda q: f"{last(q)['move99']:.3f} px")),
            [heat(f"7{'abc'[k]}", f"Image minus the method's picture: {places[k][0]}", sides,
                  lambda s, k=k: s["left_pieces"][k] if "left_pieces" in s else None, "diverging", grey, grid=piece_grid(k))
             for k in range(3)]
            + [heat("7d", f"The method's picture of the image: {places[0][0]}", sides,
                    lambda s: s["picture_pieces"][0] if "picture_pieces" in s else None, "grey", grey, grid=piece_grid(0),
                    robust=False),
               heat("7e", f"Leftover of each hole's window ({per_block})", sides, lambda s: s.get("leftover_2_map"),
                    "sequential", grey, grid=block_grid),
               heat("7f", f"Move along x from where the hole was found ({per_block})", sides, lambda s: s.get("move_x_map"),
                    "diverging", "px", grid=block_grid),
               heat("7g", f"Move along y ({per_block})", sides, lambda s: s.get("move_y_map"), "diverging", "px",
                    grid=block_grid),
               heat("7h", f"Brightness of each hole ({per_block})", sides, lambda s: s.get("brightness_2_map"),
                    "diverging", grey, grid=block_grid,
                    center=float(np.nanmedian(real["brightness_2"])) if "brightness_2" in real else 0.0),
               heat("7i", f"Level of each hole ({per_block})", sides, lambda s: s.get("level_2_map"), "diverging", grey,
                    grid=block_grid, center=float(np.nanmedian(real["level_2"])) if "level_2" in real else 0.0),
               hist("7j", "Leftover of each hole's window", grey, sides, lambda s: s.get("leftover_2")),
               hist("7k", "How far each hole moved from where it was found", "px", sides, lambda s: s.get("move"),
                    vlines=[(0.5, "as far as a hole may move along x or y")])],
            note=by_construction + at_frame)
        line(7, "Each hole fitted on the image", "leftover_2", lambda q: f"leftover {sig(np.nanmedian(q))} grey levels")
    # ---- 8
    if has("stack_2"):
        def second_minus_first(s):
            if "stack_2" not in s or "stack_1" not in s:
                return None
            return s["stack_2"] - s["stack_1"]

        add(8, "Second average hole",
            "The holes are averaged again, now each at its fitted position instead of where it was found. Left out "
            "are, as before, holes near the frame and holes whose brightness or leftover (from step 7) is unusual. "
            "This average is what the final fit works on.",
            "An average hole that changed much from the first one, many holes dropped.",
            numbers(("holes averaged", "kept_2", lambda q: fmt(q, 0)),
                    ("dropped: brightness below 0.4 or above 2.5 times the usual", "dim_2", lambda q: fmt(q, 0)),
                    ("dropped: leftover above 2 times the usual", "loud_2", lambda q: fmt(q, 0)),
                    ("usual leftover of an averaged hole = the noise the final fit takes", "noise_2",
                     lambda q: f"{sig(q)} grey levels"),
                    ("pixels in a bin, fewest to most", "pixels_2",
                     lambda q: f"{fmt(float(q[q > 0].min()) if (q > 0).any() else 0, 0)} to {fmt(float(q.max()), 0)}"),
                    ("a ninth of the image against the whole image: largest rms difference of the average holes "
                     "(picture 8e)", "parts_worst", lambda q: f"{sig(q)} grey levels")),
            [heat("8a", "Second average hole", sides, lambda s: s.get("stack_2"), "grey", grey,
                  grid=lambda s: s["stack_grid"], minus=True, robust=False),
             heat("8b", "Second average hole minus the first", sides, second_minus_first, "diverging", grey,
                  grid=lambda s: s["stack_grid"]),
             heat("8c", "Pixels that went into each bin", sides, lambda s: s.get("pixels_2"), "sequential", "pixels",
                  grid=lambda s: s["stack_grid"], robust=False),
             heat("8d", "Dropped holes per block (holes near the frame not counted)", sides,
                  lambda s: s.get("dropped_2_map"), "sequential", "holes", grid=block_grid, robust=False),
             heat("8e", "Is the average hole the same everywhere? Each ninth of the image: its average hole minus "
                        "the whole image's", sides, lambda s: s.get("parts"), "diverging", grey, cell=2,
                  grid=(0.0, 1.0, 0.0, 1.0, "bins of a quarter pixel, nine pictures side by side"),
                  note="Not a step of the calculation: the method takes one outline and one blur for the whole "
                       "image, and this shows whether that holds. The nine pictures lie as the ninths lie in the "
                       "image; brightness and level are matched before the difference is taken. An even image "
                       "gives nine pictures of fine grain. A spot or a ring that changes from one ninth to the "
                       "next means that the hole size or the blur changes across the image.")])
        line(8, "Second average hole", "kept_2", lambda q: f"{fmt(q, 0)} holes averaged")
        for row, side in zip(summary[-1][2:], sides):  # and how far the ninths of the image are from the whole
            if "parts_worst" in side:
                summary[-1][2 + sides.index(side)] = row + f"; ninths of the image differ by up to {sig(side['parts_worst'])} grey levels"
    # ---- 9
    if has("final"):
        def exchange_text(q):
            return (f"run ({q['iterations']} iterations); its end was {'taken' if q['kept'] else 'not taken'}: cost "
                    f"{sig(q['cost_second'], 5)} against {sig(q['cost_first'], 5)}")

        rows = fit_numbers("final") + numbers(
            ("second run with the blur's two parts exchanged", "exchange", exchange_text),
            ("the report's line: leftover (1 = at the noise; the rms distance above, squared)", "result",
             lambda q: sig(q["leftover"])))
        figures = fit_figures(9, "final", False)
        if has("parameters"):
            names = [p[0] for p in real.get("parameters") or copy["parameters"]]
            cells = {s["label"]: s.get("parameters") for s in sides}
            figures.append(table(
                "9z", "Every number the final fit found, and the limits it may not pass",
                ["number"] + [f"{s['label']}" for s in sides] + ["lowest allowed", "highest allowed"]
                + [f"at a limit ({s['label']})" for s in sides],
                [[name] + [cells[s["label"]][k][1] if cells[s["label"]] else "-" for s in sides]
                 + list((real.get("parameters") or copy["parameters"])[k][2:4])
                 + [cells[s["label"]][k][4] if cells[s["label"]] else "" for s in sides] for k, name in enumerate(names)],
                note="A number resting on a limit was stopped there; the fit would have gone on."))
        add(9, "Final fit",
            "The fit of step 6 again, on the second average hole, starting from step 6's answer, now with every "
            f"number free (also the outline's finer shape), for at most {LIMIT} iterations. With the two-part blur, a "
            "fit that ends with the second part the narrower one is run once more with the two parts exchanged, and "
            "the better end is taken. What this fit ends with is the image's result.",
            fit_look + " A fit that was cut off at its limit was still moving: its CD is wherever it happened to be.",
            rows, figures)
        line(9, "Final fit", "final",
             lambda q: f"{settled(q['stages'])}; best match {sig(q['rms'])} grey levels off = {sig(q['times'])} x the noise")
    # ---- 10
    if has("result"):
        def circle(s):
            radius = s["result"]["cd"] / pixel / 2.0
            angle = np.linspace(0.0, 2 * np.pi, 181)
            return dict(lines=[np.vstack([s["final"]["outline"], s["final"]["outline"][:1]]),
                            np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])])

        rows = numbers(("CD of the image", "result", lambda q: f"{q['cd']:.3f} {unit}"),
                       ("width along x, along y", "result", lambda q: f"{q['cd_x']:.3f}, {q['cd_y']:.3f} {unit}"),
                       ("holes averaged for it", "result", lambda q: fmt(q["stacked"], 0)),
                       ("noise from pixel to pixel", "pixel_noise", lambda q: f"{sig(q)} grey levels"),
                       ("leftover of a hole's window under the final answer", "hole_leftover",
                        lambda q: f"{sig(q)} grey levels"))
        if copy is not None and "result" in copy and "result" in real:
            rows.append(["the copy was drawn with this CD, and read back", f"{real['result']['cd']:.3f} {unit}",
                         f"{copy['result']['cd']:.3f} {unit} ({copy['result']['cd'] - real['result']['cd']:+.3f})"])
        if grain is not None:
            rows.append(["noise added to the copy (the pixel noise less what rounding adds)", "", f"{sig(grain)} grey levels"])
        add(10, "Result",
            "The CD is the diameter of the circle that has the same area as the outline found by the final fit. The "
            "report gives this one value to every hole of the image (one colour per image on its map): the method "
            "measures no size per hole. The redrawn copy is drawn from this answer: every hole with this outline "
            "and blur, at its own position, brightness and level (fitted to your image under them), plus noise as "
            "your image has from pixel to pixel, rounded like your file.",
            "A copy that does not give back the CD it was drawn with, or does not settle either: then the method is "
            "unsteady on this kind of image even when its picture is right. A pixel noise far below the leftover "
            "of a hole: the holes differ from the method's picture by more than noise.",
            rows,
            [heat("10a", "The outline found (line) and the circle of the same area (its diameter is the CD)", sides,
                  lambda s: s["final"]["sharp"] if "result" in s else None, "grey", "inside",
                  grid=lambda s: s["final"]["fine_grid"], robust=False, cell=3, marks=circle)])
        line(10, "Result", "result", lambda q: f"CD {q['cd']:.3f} {unit}")
        if copy is not None and "result" in copy and "result" in real:
            summary[-1][3] += f" (drawn with {real['result']['cd']:.3f})"
    return dict(steps=steps, summary=summary, sides=[s["label"] for s in sides],
                store=pooled([f for step in steps for f in step["figures"]]),
                pieces=[dict(name=n, y=int(y), x=int(x), side=int(s)) for n, y, x, s in places])


def overview(name: str, real: dict, copy: dict | None, unit: str) -> dict:
    """One image's row and small pictures for the all-images view."""
    final, result = real.get("final"), real.get("result")
    row = dict(
        image=name, holes=text_of(real, "holes", lambda q: fmt(q, 0)), spacing=text_of(real, "spacing", lambda q: f"{q:.2f}"),
        window=text_of(real, "window", lambda q: f"+-{q}"), averaged=text_of(real, "kept_2", lambda q: fmt(q, 0)),
        brightness=text_of(real, "brightness_2", lambda q: sig(np.nanmedian(q))),
        noise=text_of(real, "pixel_noise", sig), leftover=text_of(real, "hole_leftover", sig),
        first=text_of(real, "first", lambda q: f"{q['cd']:.2f}"), cd=text_of(real, "result", lambda q: f"{q['cd']:.3f}"),
        blur=text_of(real, "final", lambda q: f"{q['blur'][0]:.1f}, {q['blur'][1]:.1f}"),
        iterations=text_of(real, "final", lambda q: settled(q["stages"])),
        apart=text_of(real, "final", lambda q: sig(q["rms"])), times=text_of(real, "final", lambda q: sig(q["times"])),
        copy_cd=text_of(copy, "result", lambda q: f"{q['cd']:.3f}") if copy is not None else "-",
        copy_iterations=text_of(copy, "final", lambda q: settled(q["stages"])) if copy is not None else "-",
        copy_times=text_of(copy, "final", lambda q: sig(q["times"])) if copy is not None else "-",
        reason=real["reason"])
    out = dict(row=row, cd=None if result is None else float(result["cd"]))
    if final is not None:
        out.update(average=final["average"], apart=final["apart"], grid=final["grid"],
                   course=[c["cd"] for c in final["course"]])
    if "leftover_2_map" in real:
        out.update(leftover_map=real["leftover_2_map"], block=real["block"])
    return out


def all_page(overviews: list, unit: str) -> dict:
    """The all-images view: one table, one graph of the final fits, and one row of small pictures per kind."""
    columns = [("image", "image"), ("holes", "holes found"), ("spacing", "spacing (px)"), ("window", "window (px)"),
               ("averaged", "holes averaged"), ("brightness", "brightness (grey levels)"),
               ("noise", "pixel noise (grey levels)"), ("leftover", "leftover of a hole (grey levels)"),
               ("first", f"first fit CD ({unit})"), ("cd", f"CD ({unit})"), ("blur", f"blur width x, y ({unit})"),
               ("iterations", "final fit: iterations"), ("apart", "best match off by (grey levels)"),
               ("times", "... x the noise"), ("copy_cd", f"copy: CD read back ({unit})"),
               ("copy_iterations", "copy: iterations"), ("copy_times", "copy: x the noise"), ("reason", "not fitted because")]
    figures = [table("A1", "The steps' numbers, one row per image", [c[1] for c in columns],
                     [[o["row"][c[0]] for c in columns] for o in overviews])]
    fitted = [o for o in overviews if "average" in o]
    figures.append(lines("A2", f"CD after each iteration of the final fit, one line per image ({unit})",
                         "iteration (0 = the start)", f"CD ({unit})",
                         [(o["row"]["image"], 1 + i, np.arange(len(o["course"])), o["course"]) for i, o in enumerate(fitted)],
                         vlines=[(LIMIT, f"limit of {LIMIT}")]))
    if figures[-1] is not None:
        figures[-1]["emphasis"] = True
    shapes = {o["average"].shape for o in fitted}
    if fitted:
        low, high = span([o["apart"] for o in fitted], "diverging")
        figures.append(dict(id="A3", kind="heat", title="Average hole minus best match, one colour scale for all images",
                            note="", cell=None,
                            panels=[panel(o["row"]["image"], o["apart"], "diverging", low, high, "grey levels", o["grid"])
                                    for o in fitted]))
    if len(fitted) > 1 and len(shapes) == 1:
        mean = np.nanmean([o["average"] for o in fitted], axis=0)
        apart = []
        for o in fitted:  # brightness and level matched to the mean before the difference is taken
            ok = np.isfinite(o["average"]) & np.isfinite(mean)
            slope, level = np.polyfit(mean[ok], o["average"][ok], 1)
            apart.append(np.where(ok, o["average"] - (slope * mean + level), np.nan))
        low, high = span(apart, "diverging")
        figures.append(dict(
            id="A4", kind="heat", cell=None,
            title="Each image's average hole minus the mean average hole of all images (brightness and level matched)",
            note="For images of the same area: what changed in the average hole from image to image, apart from "
                 "brightness and level. A ring means the hole came out wider or narrower in that image.",
            panels=[panel(o["row"]["image"], a, "diverging", low, high, "grey levels", o["grid"])
                    for o, a in zip(fitted, apart)]))
    mapped = [o for o in overviews if "leftover_map" in o]
    if mapped:
        low, high = span([o["leftover_map"] for o in mapped], "sequential")
        figures.append(dict(
            id="A5", kind="heat", cell=None, note="",
            title="Leftover of each hole's window over the image (median per block), one colour scale for all images",
            panels=[panel(o["row"]["image"], o["leftover_map"], "sequential", low, high, "grey levels",
                          ((o["block"] - 1) / 2, o["block"], (o["block"] - 1) / 2, o["block"], "image px")) for o in mapped]))
    cds = [o["cd"] for o in overviews if o["cd"] is not None]
    headline = ""
    if cds:
        headline = f"{len(cds)} of {len(overviews)} images fitted; CD {min(cds):.2f} to {max(cds):.2f} {unit}"
        if len(cds) > 1:
            headline += f" (3 x the standard deviation between images: {3 * float(np.std(cds, ddof=1)):.2f} {unit})"
    figures = [f for f in figures if f is not None]
    return dict(headline=headline, store=pooled(figures), figures=figures)


def process(image, name: str, options: dict, blur: str, pixel: float, unit: str, device: int = 0, seed: int = 0):
    """One image: (its page's data, its entry for the all-images view, one console line)."""
    real_run = run(image, YOU, options, blur, device)
    places = piece_places(real_run, PIECE)
    spacing = float(real_run["trace"]["window"]["spacing_px"]) if "window" in real_run["trace"] else 10.0
    block = max(8, int(round(3 * spacing)))
    stored, grain = redrawn_copy(real_run, seed)
    copy = None
    if stored is not None:
        try:
            copy = material(run(stored, COPY, options, blur, device), blur, pixel, unit, places, block)
        except Exception as error:  # the copy is a reference: without it the image's own steps are still shown
            copy = None
            real_run["copy_error"] = str(error) or type(error).__name__
    real = material(real_run, blur, pixel, unit, places, block)
    page = image_page(real, copy, places, pixel, unit, blur, grain)
    page.update(name=name, unit=unit, pixel=pixel, blur=blur, reason=real["reason"],
                copy_error=real_run.get("copy_error", ""), shape=[int(v) for v in image.shape])
    if real["reason"]:
        text = f"not fitted ({real['reason']})"
    else:
        r, f = real["result"], real["final"]
        text = (f"CD {r['cd']:.2f} {unit}, final fit {settled(f['stages'])}, best match {sig(f['rms'])} grey levels off "
                f"({sig(f['times'])} x the noise)")
        if copy is not None and "result" in copy:
            text += (f"; copy: CD {copy['result']['cd']:.2f} {unit}, {settled(copy['final']['stages'])}, "
                     f"{sig(copy['final']['times'])} x the noise")
    return page, overview(name, real, copy, unit), text


def write_page(output: Path, entries: list, everything: dict, title: str, info: str) -> Path:
    html = PAGE.read_text(encoding="utf-8")
    for key, value in (("__TITLE__", json.dumps(title)[1:-1]), ("__INFO__", json.dumps(info)),
                       ("__ENTRIES__", json.dumps(entries))):
        html = html.replace(key, value)
    (output / "data" / "all.js").write_text("window.OFS_ALL=" + json.dumps(everything, allow_nan=False) + ";\n",
                                            encoding="utf-8")
    path = output / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main(argv=None) -> int:
    from scripts.analyze_contact_holes import load_image
    from subpx.outline import _gpu_usable

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the images (.png, .tif, .tiff), or one image file")
    parser.add_argument("--output", type=Path, help="Folder for the page (default: artifacts/outline_fit_steps)")
    parser.add_argument("--page", type=int, default=0, help="Page of a TIFF file (from 0)")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--outline-blur", choices=("gauss2", "lens"), default="gauss2")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--threshold", choices=("otsu", "triangle"), default="otsu")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--pad", type=int, default=3)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-open", action="store_true", help="Do not open the page in the browser")
    args = parser.parse_args(argv)
    if args.folder.is_dir():
        images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in SUFFIXES)
    else:
        images = [args.folder] if args.folder.is_file() else []
    if not images:
        parser.error(f"No image (.png, .tif, .tiff) at {args.folder}")
    if not _gpu_usable():
        parser.error("outline_fit_fast needs the GPU (CuPy); no usable GPU was found")
    output = args.output or ROOT / "artifacts" / "outline_fit_steps"
    (output / "data").mkdir(parents=True, exist_ok=True)
    options = dict(seeds=args.seeds, threshold=args.threshold, invert=args.invert, area_min=args.area_min,
                   area_max=args.area_max, pad=args.pad, pixel_size=args.pixel_size, unit=args.unit)
    entries, overviews, used, started = [], [], set(), time.perf_counter()
    for number, path in enumerate(images, 1):
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "image"
        while ident.casefold() in used:
            ident += "_"
        used.add(ident.casefold())
        start = time.perf_counter()
        entry = dict(id=ident, name=path.name, data=f"data/{ident}.js", error="")
        try:
            image = load_image(path, page=args.page if path.suffix.lower() != ".png" else 0)
            page, row, text = process(image, path.name, options, args.outline_blur, args.pixel_size, args.unit,
                                      args.device)
            (output / entry["data"]).write_text(f"window.OFS_DATA[{json.dumps(ident)}]="
                                                + json.dumps(page, allow_nan=False) + ";\n", encoding="utf-8")
            overviews.append(row)
            entry.update(cd=row["row"]["cd"], reason=page["reason"])
        except Exception as error:  # one image that cannot be read or shown must not stop the folder
            text = "failed: " + (str(error).replace(str(path), path.name) or type(error).__name__)
            entry.update(error=text, data="")
        entries.append(entry)
        print(f"[{number}/{len(images)}] {path.name}: {text} ({time.perf_counter() - start:.1f} s)".encode(
            "ascii", "replace").decode("ascii"), flush=True)
    info = (f"{args.folder.resolve()} | blur form {args.outline_blur} | pixel {args.pixel_size:g} {args.unit} | holes: "
            f"--seeds {args.seeds} --threshold {args.threshold} --area-min {args.area_min} --area-max {args.area_max} "
            f"--pad {args.pad}{' --invert' if args.invert else ''} | {time.strftime('%Y-%m-%d %H:%M')}")
    path = write_page(output, entries, all_page(overviews, args.unit), f"Outline fit, step by step: {args.folder.name}", info)
    if not args.no_open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    print(f"Page: {path} ({time.perf_counter() - started:.0f} s)", flush=True)  # the last line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
