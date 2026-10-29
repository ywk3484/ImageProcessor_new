"""Build CD-center pitch and registration reports for an image or folder.

    python scripts/pitch_report.py IMAGE_OR_FOLDER --average-rows 5 --average-columns 7
    python scripts/pitch_report.py IMAGE.tif --pixel-size 21 --unit nm --backend gpu
    python scripts/pitch_report.py CD_RESULTS_FOLDER --from-cd --layout staggered
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_contact_holes import load_image
from subpx import load_cd_results, measure_cds, save_pitch_report
from subpx.cd import LOCAL_METHODS, METHOD_LABELS

DEFAULT_METHODS = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
                   "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio", *LOCAL_METHODS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--from-cd", action="store_true", help="Reuse a saved CD result folder or NPZ, without refitting")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--page", type=int, default=0, help="Zero-based TIFF page")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_LABELS), default=DEFAULT_METHODS)
    parser.add_argument("--layout", choices=("auto", "square", "staggered"), default="auto")
    parser.add_argument("--stagger-axis", choices=("row", "column"), default="row")
    parser.add_argument("--average-rows", type=int, help="Rows averaged for x pitch; default: all")
    parser.add_argument("--average-columns", type=int, help="Columns averaged for y pitch; default: all")
    parser.add_argument("--reference", type=float, nargs=2, metavar=("X_PX", "Y_PX"))
    parser.add_argument("--nominal-pitch", type=float, nargs=2, metavar=("X", "Y"), help="Nominal pitches in --unit; default: measured means")
    parser.add_argument("--pitch-hint", type=float, nargs=2, metavar=("X_PX", "Y_PX"))
    parser.add_argument("--line-tolerance", type=float, help="Grouping tolerance in pixels")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--threshold", choices=("otsu", "triangle"), default="otsu")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--pad", type=int, default=3)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--psf-fwhm", type=float, help="Blur in pixels for optional disk/flux_ratio methods")
    parser.add_argument("--half-level", type=float, default=0.5)
    parser.add_argument("--peak-pixels", type=float, default=4.0)
    parser.add_argument("--tail-level", type=float, default=0.2)
    parser.add_argument("--tail-window", type=float, default=1.2)
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input not found: {args.input}")
    output = args.output or ROOT / "artifacts" / time.strftime("pitch_report_%Y%m%d_%H%M%S")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be a new or empty folder")
    analysis = dict(layout=args.layout, stagger_axis=args.stagger_axis, average_rows=args.average_rows,
                    average_columns=args.average_columns, reference_xy=args.reference,
                    nominal_pitch_xy=args.nominal_pitch, line_tolerance=args.line_tolerance,
                    pitch_hint_xy=args.pitch_hint)
    options = dict(methods=tuple(args.methods), pixel_size=args.pixel_size, unit=args.unit,
                   backend=args.backend, device=args.device, seeds=args.seeds, threshold=args.threshold,
                   area_min=args.area_min, area_max=args.area_max, pad=args.pad, invert=args.invert,
                   psf_fwhm=args.psf_fwhm, half_area_level=args.half_level,
                   half_area_peak_pixels=args.peak_pixels, tail_area_level=args.tail_level,
                   tail_window=args.tail_window, outlines=False, metrics_only=True)
    if args.from_cd:
        found = load_cd_results(args.input)
        page = save_pitch_report(found, output, title=args.input.name, **analysis)
        print(f"Report: {page.resolve()}", flush=True)
        return 0
    batch = args.input.is_dir()
    paths = sorted(p for p in (args.input.rglob("*") if args.recursive else args.input.iterdir())
                   if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}
                   and not p.resolve().is_relative_to(output.resolve())) if batch else [args.input]
    if not paths:
        parser.error("No supported grayscale images found")
    # Fail on global measurement settings before creating a partial batch report.
    try:
        measure_cds(np.zeros((8, 8), np.uint8), **{**options, "backend": "cpu"})
        for value in (args.average_rows, args.average_columns):
            if value is not None and value < 1:
                raise ValueError("Averaging counts must be positive")
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=True)
    entries, used = [], set()
    started = time.perf_counter()
    print(f"Pitch report: {len(paths)} image(s), {len(args.methods)} CD methods", flush=True)
    for number, path in enumerate(paths, 1):
        name = path.relative_to(args.input).as_posix() if batch else path.name
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", str(Path(name).with_suffix(""))) or "image"
        while ident.casefold() in used:
            ident += "_"
        used.add(ident.casefold())
        target = output / ident if batch else output
        try:
            measured = measure_cds(load_image(path, page=args.page), **options)
            page = save_pitch_report(measured, target, title=name, **analysis)
            entry = dict(image=name, status="ok", report=page.relative_to(output).as_posix(), holes=len(measured.seeds_xy), error="")
            print(f"[{number}/{len(paths)}] {name}: {len(measured.seeds_xy):,} holes", flush=True)
        except Exception as error:
            entry = dict(image=name, status="failed", report="", holes="", error=str(error))
            print(f"[{number}/{len(paths)}] {name}: failed: {error}", flush=True)
        entries.append(entry)
    if batch:
        links = "".join(f"<li><a href='{html.escape(e['report'], quote=True)}'>{html.escape(e['image'])}</a> — {e['holes']} holes</li>"
                        if e["status"] == "ok" else f"<li>{html.escape(e['image'])}: failed: {html.escape(e['error'])}</li>" for e in entries)
        (output / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>Pitch reports</title>"
                                           "<h1>Pitch and registration reports</h1><ul>" + links + "</ul>", encoding="utf-8")
    with (output / "images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "status", "report", "holes", "error"))
        writer.writeheader()
        writer.writerows(entries)
    (output / "run.json").write_text(json.dumps(dict(measurement=options, analysis=analysis, images=entries,
                                                     elapsed_s=time.perf_counter() - started), indent=2), encoding="utf-8")
    if batch or entries[0]["status"] == "ok":
        print(f"Report: {(output / 'index.html').resolve()}", flush=True)
    else:
        print(f"No report generated; failure details: {(output / 'images.csv').resolve()}", flush=True)
    return int(any(e["status"] == "failed" for e in entries))


if __name__ == "__main__":
    raise SystemExit(main())
