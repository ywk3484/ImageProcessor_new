"""What the images of a folder hold, in grey levels: how they are stored, and how far the outline fit's best match
is from the average hole.

    python scripts/image_values.py FOLDER

Every .png of the folder is read (and one page of every .tif / .tiff: --page). Use the options the report was run
with (--outline-blur, --seeds, --invert). The first half needs no GPU; the second half is outline_fit_fast as the
reports run it (GPU) and is left out with --no-fit or without a usable GPU. Nothing in the library is changed by this
script; it measures.

Why: the outline fit prints a leftover that has no unit (1 = the average hole is fitted to its noise; 430-526 on the
first real images). It is not a brightness, and by itself it does not say how far the fit's best match is from the
average hole, because three different things raise it:

    the best match misses the average hole      what the model assumes about a hole does not hold
    a quiet image, or many holes                the average hole is known so precisely that a small miss counts
    few grey levels for the image's noise       noise well below one grey step: rounding to whole grey steps does not
                                                average out, and no outline follows the steps it leaves

This script prints what tells them apart, everything in grey steps of the file. One line per image, the details in a
JSON file (--output), and as the last line the folder:

    values N=8 | uint16 3251-23544 | levels 17175-18217 step 1 | top 0.00% zero 0.00% | noise 299 hole 14767 | gap 6.10-9.81 (5.47) = 0.04-0.06%

    uint16 A-B          how the files are stored; lowest and highest pixel value of all images
    levels a-b          different grey values in one image, fewest and most over the images (256 or fewer in a
                        16 bit file: it holds 8 bit content)
    step s              the grey step: the values of an image lie a whole number of steps apart (1 normally; 257 or
                        256 = 8 bit content in a 16 bit file, 16 = 12 bit content)
    top p% zero q%      most pixels of one image at the highest value the file type can hold (255, 65535), and at
                        zero: more than about 0.01 % = the tops (or the dark areas) are cut off flat
    noise n             one pixel's noise in grey steps, as the fit takes it: how far a single hole lies from the
                        average hole (rms over its window, the median hole; median of the images). Real differences
                        between holes are in it
    hole h              brightness of the average hole in grey steps: its top above its darkest point (median)
    gap a-b (n) = p%    how far the fit's best match is from the average hole (rms over the hole's window), in grey
                        steps, lowest and highest image; in brackets what noise alone would leave (median); then the
                        gap in percent of the hole's brightness. gap / bracket, squared, is the leftover

How to read it. Generated images of 26,000-28,000 holes of 100 nm, 55-59 grey levels bright in 8 bit
(docs/outline_fit.md, "What the leftover is"):

                                                 noise   gap (noise alone)          % of hole   leftover   CD error
    the model fits, usual noise                  1.2     0.022-0.038 (0.019-0.030)  0.04-0.06   1.2-1.8    < 0.1 nm
    a blur the model cannot describe             1.2     0.034-0.054 (0.019)        0.06-0.10   3-8        1.2 nm
    8 bit, quiet image (noise 0.1 grey level)    0.39    0.058-0.069 (0.006-0.010)  0.10-0.12   49-95      0.3-0.4 nm
    the same quiet image in 16 bit               71      3.1-3.9 (1.2-1.8)          0.02-0.03   4.6-6.9    < 0.1 nm

    gap about the bracket                        the best match fits the average hole to its noise
    gap many times the bracket, thousands of     the best match misses the average hole by that many grey steps: the
    levels, noise far above 1                    model, not the storage. A gap of 0.1 % of the hole's brightness
                                                 already went with 1.2 nm of CD (second row)
    256 levels or fewer, noise below about 0.5   rounding to whole grey steps is part of the gap and moves the CD
                                                 (8 bit against 16 bit copies: the same CD down to a noise of 0.7,
                                                 0.06 nm apart at 0.5, 0.2 nm at 0.44, 0.4 nm at 0.39)
    top or zero above about 0.01 %               values cut off: an outline through a blur cannot follow a flat top
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUFFIXES = (".png", ".tif", ".tiff")


def file_values(image) -> dict:
    """How one image is stored: its type, lowest and highest value, how many different values it has, the grey
    step (integer images: the largest number that every difference between its values is a whole multiple of; None
    otherwise), and the shares of its pixels at the top of the file type's range and at zero."""
    a = np.asarray(image)
    values = np.unique(a)
    step, top = None, 0.0
    if np.issubdtype(a.dtype, np.integer):
        step = int(np.gcd.reduce(np.diff(values.astype(np.int64)))) if len(values) > 1 else 1
        # a 16 bit PNG may be read as int32: the ceiling is the one of the smallest file type that holds the values
        ceiling = 255 if a.dtype.itemsize == 1 else 65535 if values[-1] <= 65535 else np.iinfo(a.dtype).max
        top = float(np.mean(a == ceiling))
    return {"dtype": str(a.dtype), "low": float(values[0]), "high": float(values[-1]), "levels": int(len(values)),
            "step": step, "top": top, "zero": float(np.mean(a == 0))}


@contextmanager
def tapped_fit(box: dict):
    """While active, ``fit_outline_fast`` keeps its result (the average hole is not part of what ``measure_cds``
    returns) and runs unchanged."""
    import subpx.outline as outline

    original = outline.fit_outline_fast

    def tapped(image, seeds_xy=None, **kw):
        box["result"] = original(image, seeds_xy, **kw)
        return box["result"]

    outline.fit_outline_fast = tapped
    try:
        yield
    finally:
        outline.fit_outline_fast = original


def gap_values(stack, stack_weight, noise, leftover) -> dict:
    """From the fit's average hole (``stack``, with the number of pixels averaged into each of its bins), one
    pixel's noise and the leftover (chi-square per bin, weights = pixels / noise^2): the hole's brightness, what
    noise alone leaves between a matching model and the average hole, and how far the best match is (that x
    sqrt(leftover)). Both distances are rms over the bins, each bin counted by its pixels, in the image's values:
    sum(pixels x distance^2) = chi-square x noise^2, so rms^2 = leftover x noise^2 / (pixels per bin, the mean).
    The mean, not the median: where the pitch is a whole number of pixels a few bins hold nearly all pixels. (The
    fit leaves the outermost bins out; counting them moves the result by a few percent.)"""
    weight = np.asarray(stack_weight, dtype=float)
    stack = np.asarray(stack, dtype=float)
    per_bin = float(weight[weight > 0].mean())
    usual = weight >= 0.25 * per_bin  # the brightness from well-filled bins only
    alone = float(noise) / math.sqrt(per_bin)
    return {"per_bin": per_bin, "hole": float(stack[usual].max() - stack[usual].min()), "alone": alone,
            "gap": alone * math.sqrt(max(float(leftover), 0.0))}


def fit_values(image, options: dict, blur: str, device: int = 0) -> dict:
    """The one-pass outline fit of one image as the reports run it, in the image's own values (see ``gap_values``)."""
    from subpx import measure_cds

    box = {}
    with tapped_fit(box):
        result = measure_cds(image, methods=("outline_fit_fast",), outline_blur=blur, backend="gpu", device=device,
                             **options)
    meta = result.meta["outline_fit_fast"]
    if not meta.get("fitted") or "result" not in box:
        return {"fitted": False, "reason": meta.get("reason", "")}
    fit = box["result"]
    noise, leftover = float(fit.summary["noise_sd"]), float(fit.summary["leftover_chi2"])
    return {"fitted": True, "holes": int(fit.summary["holes_stacked"]), "noise": noise, "leftover": leftover,
            "converged": bool(fit.summary["converged"]), **gap_values(fit.stack, fit.stack_weight, noise, leftover)}


def num(value) -> str:
    """Three figures, never an exponent."""
    if value is None or not np.isfinite(value):
        return "-"
    a = abs(value)
    return f"{value:.0f}" if a >= 100 else f"{value:.1f}" if a >= 10 else f"{value:.2f}" if a >= 1 else f"{value:.3f}"


def in_steps(r: dict) -> dict:
    """The fit's values of one image in grey steps of its file."""
    step = r["step"] or 1
    return {k: r[k] / step for k in ("noise", "hole", "gap", "alone")}


def image_line(name: str, r: dict) -> str:
    """One ASCII line on an image."""
    if "dtype" not in r:
        return f"{name}: not read ({r.get('reason', '')})"
    text = (f"{name}: {r['dtype']} {r['low']:g}-{r['high']:g}, {r['levels']} levels, step {r['step'] or '-'}, "
            f"top {100 * r['top']:.2f}%, zero {100 * r['zero']:.2f}%")
    if "fitted" not in r:
        return text
    if not r["fitted"]:
        return f"{text} | not fitted ({r['reason']})"
    s = in_steps(r)
    return (f"{text} | noise {num(s['noise'])}, hole {num(s['hole'])}, gap {num(s['gap'])} (noise alone "
            f"{num(s['alone'])}) = {100 * r['gap'] / r['hole']:.2f}% of the hole; leftover {r['leftover']:.2f}, "
            f"{r['holes']} holes")


def folder_line(results: list) -> str:
    """The folder's line: short, ASCII, to be typed back (see the module docstring for its fields)."""
    read = [r for r in results if "dtype" in r]
    if not read:
        return "values N=0: no image was read"

    def span(values, text=str):
        lo, hi = min(values), max(values)
        return text(lo) if lo == hi else f"{text(lo)}-{text(hi)}"

    steps = [r["step"] for r in read if r["step"]]
    line = (f"values N={len(read)} | {'/'.join(sorted({r['dtype'] for r in read}))} "
            f"{min(r['low'] for r in read):g}-{max(r['high'] for r in read):g} | "
            f"levels {span([r['levels'] for r in read])} step {span(steps) if steps else '-'} | "
            f"top {100 * max(r['top'] for r in read):.2f}% zero {100 * max(r['zero'] for r in read):.2f}%")
    fitted = [r for r in read if r.get("fitted")]
    if not fitted:
        return line + (" | no fit" if any("fitted" in r for r in read) else "")
    s = [in_steps(r) for r in fitted]
    percent = [100 * r["gap"] / r["hole"] for r in fitted]
    return (f"{line} | noise {num(float(np.median([v['noise'] for v in s])))} "
            f"hole {num(float(np.median([v['hole'] for v in s])))} | "
            f"gap {num(min(v['gap'] for v in s))}-{num(max(v['gap'] for v in s))} "
            f"({num(float(np.median([v['alone'] for v in s])))}) = {min(percent):.2f}-{max(percent):.2f}%")


def main(argv=None) -> int:
    from scripts.analyze_contact_holes import load_image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the images (.png, .tif, .tiff), or one image file")
    parser.add_argument("--output", type=Path, help="JSON file for the details (default: under artifacts/)")
    parser.add_argument("--page", type=int, default=0, help="Page of a TIFF file (from 0)")
    parser.add_argument("--no-fit", action="store_true", help="Only how the files are stored (no GPU needed)")
    parser.add_argument("--outline-blur", choices=("gauss2", "lens"), default="gauss2")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.folder.is_dir():
        images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in SUFFIXES)
    else:
        images = [args.folder] if args.folder.is_file() else []
    if not images:
        parser.error(f"No image (.png, .tif, .tiff) at {args.folder}")
    fit = not args.no_fit
    if fit:
        from subpx.outline import _gpu_usable

        fit = _gpu_usable()
        if not fit:
            print("No usable GPU (CuPy): only how the files are stored; the fit's part is left out", flush=True)
    options = dict(seeds=args.seeds, invert=args.invert, area_min=args.area_min, area_max=args.area_max)
    output = args.output or ROOT / "artifacts" / time.strftime("image_values_%Y%m%d_%H%M%S.json")
    results, started = [], time.perf_counter()
    for number, path in enumerate(images, 1):
        start = time.perf_counter()
        r = {}
        try:
            image = load_image(path, page=args.page if path.suffix.lower() != ".png" else 0)
            r.update(file_values(image))
            if fit:
                try:
                    r.update(fit_values(image, options, args.outline_blur, args.device))
                except Exception as error:  # one image that cannot be fitted must not stop the folder
                    r.update(fitted=False, reason=str(error).replace(str(path), path.name) or type(error).__name__)
        except Exception as error:
            r["reason"] = str(error).replace(str(path), path.name) or type(error).__name__
        r.update(image=path.name, seconds=time.perf_counter() - start)
        results.append(r)
        print(f"[{number}/{len(images)}] {image_line(path.name, r)} ({r['seconds']:.1f} s)", flush=True)
    line = folder_line(results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"folder": str(args.folder.resolve()), "blur": args.outline_blur, "options": options,
                                  "page": args.page, "line": line, "images": results}, indent=1), encoding="utf-8")
    print(f"Details: {output} ({time.perf_counter() - started:.0f} s)", flush=True)
    print(line, flush=True)  # the last line: the one to type back from another machine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
