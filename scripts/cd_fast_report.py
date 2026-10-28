"""Report every PNG with all original and fast methods in the original layout.

    python scripts/cd_fast_report.py FOLDER --pixel-size 21 --unit nm

Includes the ten original report methods, four raw-cell methods and three
native-pixel methods. Maps, measured outlines, hole details and statistics use
the original report. Open the printed index.html path directly in a browser.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cd_report as report
from scripts.analyze_contact_holes import load_image
from scripts.cd_batch import measurement_fingerprint
from scripts.cd_compare import finite_json, numeric_result
from subpx import measure_cds, save_cd_results
from subpx.cd import LOCAL_METHODS, _resolve_backends

ORIGINAL_METHODS = report.ALL_METHODS
FAST_METHODS = LOCAL_METHODS
ALL_METHODS = ORIGINAL_METHODS + FAST_METHODS


def write_csv(path, rows, fields=()):
    """Write even empty tables with an identifiable header."""
    names = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(finite_json(row) for row in rows)


@contextmanager
def measurement_runtime(backend, device):
    """Reuse the GPU pool across images and avoid excess CPU thread startup."""
    import cv2
    from scripts.benchmark_cd import _batch_gpu

    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with _batch_gpu(device) if backend == "gpu" else nullcontext():
            yield
    finally:
        cv2.setNumThreads(threads)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, help="New or empty report folder")
    # Keep the previously supplied command valid; native-pixel methods are now always included.
    parser.add_argument("--include-pixel", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recursive", action="store_true", help="Include PNGs in subfolders")
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
    parser.add_argument("--half-level", type=float, default=0.5)
    parser.add_argument("--peak-pixels", type=float, default=4.0)
    parser.add_argument("--tail-level", type=float, default=0.2)
    parser.add_argument("--tail-window", type=float, default=1.2)
    parser.add_argument("--deblend-iterations", type=int, default=30)
    parser.add_argument("--deblend-tolerance", type=float, default=1e-3)
    parser.add_argument("--no-hole-details", action="store_true", help="Skip the click panel's cell images and profiles")
    parser.add_argument("--save-results", action="store_true", help="Also save full per-hole results (compact NPZ + CSVs)")
    return parser


def _settings(args, backend):
    return dict(seeds=args.seeds, threshold=args.threshold, invert=args.invert,
                area_min=args.area_min, area_max=args.area_max, pad=args.pad,
                pixel_size=args.pixel_size, unit=args.unit, backend=backend, device=args.device,
                half_area_level=args.half_level, half_area_peak_pixels=args.peak_pixels,
                tail_area_level=args.tail_level, tail_window=args.tail_window,
                deblend_iterations=args.deblend_iterations, deblend_tolerance=args.deblend_tolerance)


def _write_data(path, namespace, key, value):
    path.write_text(f"window.{namespace}[{json.dumps(key)}]="
                    + json.dumps(value, allow_nan=False) + ";\n", encoding="utf-8")


def _process_image(path, name, ident, output, options, rng, *, hole_details=True, save_results=False):
    from PIL import Image

    started = time.perf_counter()
    image = load_image(path)
    source = dict(file=name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  shape=list(image.shape), dtype=str(image.dtype))
    read_s = time.perf_counter() - started
    started = time.perf_counter()
    # The original viewer aligns methods by hole ID. A mixed call retains that
    # common seed/cell list; each fast method still measures its raw-cell input.
    result = measure_cds(image, methods=ALL_METHODS, **options)
    measure_s = time.perf_counter() - started
    started = time.perf_counter()
    statistics = report.cd_statistics(result)
    rows = [dict(image=name, status="ok", method=method, holes=len(result.seeds_xy), **stats, unit=result.unit)
            for method, stats in statistics.items()]
    image_mean = float(np.mean(image, dtype=np.float64))
    intensity = report.cell_intensity(result)
    intensity = None if intensity is None else report._spread(intensity)
    layers = report.layer_values(result)
    samples = {}
    for layer, values in layers.items():
        values = values[np.isfinite(values)]
        samples[layer] = rng.choice(values, min(len(values), 50_000), replace=False)
    Image.fromarray(report.display_image(image)).save(output / "images" / f"{ident}.png")
    _write_data(output / "data" / f"{ident}.js", "CD_DATA", ident, report.hole_data(result, layers, ident))
    for method in result.measurements:
        _write_data(output / "data" / f"{ident}.{method}.rays.js", "CD_RAYS", ident + "/" + method,
                    report.rays_data(result, method))
    detail = tail = None
    if hole_details:
        detail = f"data/{ident}.detail.js"
        _write_data(output / detail, "CD_DETAIL", ident, report.detail_data(result))
        wide = report.tail_detail(result)
        if wide is not None:
            tail = f"data/{ident}.tail.js"
            _write_data(output / tail, "CD_TAIL", ident, wide)
    np.savez_compressed(output / "arrays" / f"{ident}.npz", **numeric_result(result))
    if save_results:
        save_cd_results(result, output / "results" / ident, compact=True)
    if intensity is not None:
        rows.append(dict(image=name, status="ok", method="cell_intensity", holes=len(result.seeds_xy),
                         cells=len(result.seeds_xy), **intensity, unit="counts"))
    rows.append(dict(image=name, status="ok", method="image_intensity", holes=len(result.seeds_xy),
                     valid=int(image.size), mean=image_mean, unit="counts"))
    writing_s = time.perf_counter() - started
    timing = dict(image=name, status="ok", read_s=read_s, measure_s=measure_s, export_s=writing_s)
    entry = dict(id=ident, name=name, holes=len(result.seeds_xy), stats=statistics, width=image.shape[1],
                 height=image.shape[0], seconds=measure_s, image=f"images/{ident}.png", data=f"data/{ident}.js",
                 detail=detail, tail=tail, agreement=report.agreement(result), intensity=intensity,
                 image_mean=image_mean, intensity_source="tail_area")
    source.update(id=ident, status="ok", meta=result.meta)
    return entry, rows, timing, source, samples


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"Folder not found: {args.folder}")
    output = args.output or ROOT / "artifacts" / time.strftime("cd_fast_report_%Y%m%d_%H%M%S")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be a new or empty folder")
    files = args.folder.rglob("*") if args.recursive else args.folder.iterdir()
    images = sorted((p for p in files if p.is_file() and p.suffix.lower() == ".png"
                     and not p.resolve().is_relative_to(output.resolve())), key=lambda p: str(p).casefold())
    if not images:
        parser.error(f"No PNG images in {args.folder}")
    backend, _ = _resolve_backends(args.backend, None, "model")
    options = _settings(args, backend)
    # Validate numerical options before creating output folders.
    try:
        measure_cds(np.zeros((8, 8), np.uint8), methods=ALL_METHODS, **{**options, "backend": "cpu"})
    except ValueError as error:
        parser.error(str(error))
    for folder in ("images", "data", "arrays"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    print(f"CD report: {len(images)} PNGs; {len(ORIGINAL_METHODS)} original + {len(FAST_METHODS)} fast methods; "
          f"{backend.upper()} device {args.device}.", flush=True)
    started = time.perf_counter()
    entries, rows, timings, sources = [], [], [], []
    samples, used = {}, set()
    rng = np.random.default_rng(0)
    with measurement_runtime(backend, args.device):
        for number, path in enumerate(images, 1):
            name = path.relative_to(args.folder).as_posix()
            ident = re.sub(r"[^A-Za-z0-9_-]", "_", str(Path(name).with_suffix(""))) or "image"
            while ident.casefold() in used:
                ident += "_"
            used.add(ident.casefold())
            try:
                entry, values, timing, source, values_by_layer = _process_image(
                    path, name, ident, output, options, rng,
                    hole_details=not args.no_hole_details, save_results=args.save_results)
            except Exception as error:
                message = str(error).replace(str(path), name) or type(error).__name__
                entries.append(dict(id=ident, name=name, error=message))
                rows.append(dict(image=name, status="failed", error=message))
                timings.append(dict(image=name, status="failed", error=message))
                sources.append(dict(file=name, status="failed", error=message))
                print(f"[{number}/{len(images)}] {name}: failed: {message}", flush=True)
                continue
            entries.append(entry)
            rows.extend(values)
            timings.append(timing)
            sources.append(source)
            for layer, values in values_by_layer.items():
                samples.setdefault(layer, []).append(values)
            print(f"[{number}/{len(images)}] {name}: {entry['holes']:,} holes; {timing['measure_s']:.2f} s", flush=True)
    fields = (["image", "status", "method", "holes", "cells"]
              + [key + suffix for suffix, _, _ in report.AXES
                 for key in ("valid", "mean", "std", "three_sigma_pct")] + ["unit", "error"])
    write_csv(output / "summary.csv", rows, fields)
    between = report.between_image_statistics(entries, ALL_METHODS)
    brightness = report.brightness_between(entries)
    fields = ["method"] + [key + suffix for suffix, _, _ in report.AXES
                           for key in ("images", "mean_of_means", "std_of_means", "three_sigma", "three_sigma_pct")] + ["unit"]
    write_csv(output / "between_images.csv", [dict(method=m, **values, unit=args.unit) for m, values in between.items()]
              + [dict(method=m, **values, unit="counts") for m, values in brightness.items()], fields)
    write_csv(output / "timings.csv", timings, ("image", "status"))
    ranges = {}
    for layer, parts in samples.items():
        values = np.concatenate(parts)
        if len(values):
            ranges[layer] = np.quantile(values, [0.02, 0.98]).tolist()
    page = report.write_report(output, entries, ALL_METHODS, [], args.unit, ranges)
    elapsed = time.perf_counter() - started
    failures = [e for e in entries if e.get("error")]
    manifest = dict(schema_version=2, source_folder=str(args.folder.resolve()),
                    measurement_fingerprint=measurement_fingerprint(), settings=options,
                    methods=ALL_METHODS, original_methods=ORIGINAL_METHODS, fast_methods=FAST_METHODS,
                    elapsed_s=elapsed, images=sources,
                    successful_images=len(entries) - len(failures), failed_images=len(failures),
                    note="All methods share the original model pipeline's seed/cell list. Fast methods measure "
                         "raw cells. Full display inputs and outlines are retained; this is not a fast-only batch timing.")
    (output / "run.json").write_text(json.dumps(finite_json(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {page} ({elapsed:.1f} s; {len(failures)} failed images).", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
