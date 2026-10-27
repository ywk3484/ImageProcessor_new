"""Report every PNG with fast CD methods and matched slow-method comparisons.

    python scripts/cd_fast_report.py FOLDER --output REPORT --pixel-size 21 --unit nm

Defaults to the three fused 4x local area methods. The corresponding original
model methods run separately on every image to check agreement and coverage.
--include-pixel also evaluates native-pixel counting; --fast-only skips the
reference calculation after validation. Open REPORT/index.html offline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
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

from scripts import cd_report as report
from scripts.analyze_contact_holes import load_image
from scripts.cd_batch import measurement_fingerprint
from scripts.cd_compare import finite_json, numeric_result
from scripts.cd_report_compare import aggregate_comparisons, compare_image
from scripts.cd_report_compare_view import comparison_html, write_comparison_plot
from subpx import measure_cds
from subpx.cd import PIXEL_METHODS, _resolve_backends

FAST_METHODS = ("local_half_area", "local_half_area_min_avg", "local_tail_area")
REFERENCE_METHODS = ("half_area", "half_area_min_avg", "tail_area")


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
    parser.add_argument("--include-pixel", action="store_true", help="Also compare the three native-pixel methods")
    parser.add_argument("--fast-only", action="store_true", help="Skip slow references; no agreement claim is made")
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
    parser.add_argument("--match-distance", type=float, default=1.0, help="Maximum seed matching distance in pixels")
    parser.add_argument("--max-difference", type=float,
                        help="Optional limit in --unit for the 95th percentile absolute paired-hole difference")
    return parser


def _settings(args, backend):
    return dict(seeds=args.seeds, threshold=args.threshold, invert=args.invert,
                area_min=args.area_min, area_max=args.area_max, pad=args.pad,
                pixel_size=args.pixel_size, unit=args.unit, backend=backend, device=args.device,
                half_area_level=args.half_level, half_area_peak_pixels=args.peak_pixels,
                tail_area_level=args.tail_level, tail_window=args.tail_window,
                deblend_iterations=args.deblend_iterations, deblend_tolerance=args.deblend_tolerance,
                summary_only=True, n_angles=16, outlines=False)


def _number_rows(name, result):
    return [dict(image=name, status="ok", method=method, holes=len(result.seeds_xy), **stats, unit=result.unit)
            for method, stats in report.cd_statistics(result).items()]


def _process_image(path, name, ident, output, fast_methods, reference_methods, options, max_distance, rng):
    from PIL import Image

    started = time.perf_counter()
    image = load_image(path)
    source = dict(file=name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  shape=list(image.shape), dtype=str(image.dtype))
    read_s = time.perf_counter() - started
    started = time.perf_counter()
    fast = measure_cds(image, methods=fast_methods, **options)
    fast_s = time.perf_counter() - started
    statistics = report.cd_statistics(fast)
    rows = _number_rows(name, fast)
    reference = None
    reference_s = None
    reference_error = None
    comparisons = []
    if reference_methods:
        started = time.perf_counter()
        try:
            reference = measure_cds(image, methods=reference_methods, **options)
        except Exception as error:
            reference_error = str(error) or type(error).__name__
        reference_s = time.perf_counter() - started
        if reference is not None:
            statistics.update(report.cd_statistics(reference))
            rows.extend(_number_rows(name, reference))
            comparisons = compare_image(reference, fast, image_id=name, max_distance=max_distance,
                                        reference_meta={"measure_s": reference_s},
                                        candidate_meta={"measure_s": fast_s})
        else:
            rows.extend(dict(image=name, status="failed", method=method, error=reference_error)
                        for method in reference_methods)

    started = time.perf_counter()
    image_mean = float(np.mean(image, dtype=np.float64))
    intensity = report.cell_intensity(fast)
    intensity = None if intensity is None else report._spread(intensity)
    layers = report.layer_values(fast)
    samples = {}
    for layer, values in layers.items():
        values = values[np.isfinite(values)]
        samples[layer] = rng.choice(values, min(len(values), 50_000), replace=False)
    Image.fromarray(report.display_image(image)).save(output / "images" / f"{ident}.png")
    data = report.hole_data(fast, layers, ident, image=image)
    (output / "data" / f"{ident}.js").write_text(
        f"window.CD_DATA[{json.dumps(ident)}]=" + json.dumps(data, allow_nan=False) + ";\n", encoding="utf-8")
    np.savez_compressed(output / "arrays" / f"{ident}.fast.npz", **numeric_result(fast))
    if reference is not None:
        np.savez_compressed(output / "arrays" / f"{ident}.reference.npz", **numeric_result(reference))
    if intensity is not None:
        rows.append(dict(image=name, status="ok", method="cell_intensity", holes=len(fast.seeds_xy),
                         cells=len(fast.seeds_xy), **intensity, unit="counts"))
    rows.append(dict(image=name, status="ok", method="image_intensity", holes=len(fast.seeds_xy),
                     valid=int(image.size), mean=image_mean, unit="counts"))
    writing_s = time.perf_counter() - started
    timing = dict(image=name, status="ok", read_s=read_s, fast_s=fast_s, reference_s=reference_s,
                  export_s=writing_s, reference_error=reference_error,
                  reference_converged=None if reference is None else reference.meta.get("deblend_converged"),
                  reference_iterations=None if reference is None else reference.meta.get("deblend_iterations_used"))
    entry = dict(id=ident, name=name, holes=len(fast.seeds_xy), stats=statistics, width=image.shape[1],
                 height=image.shape[0], seconds=fast_s, image=f"images/{ident}.png", data=f"data/{ident}.js",
                 detail=None, tail=None, agreement=report.agreement(fast), intensity=intensity, image_mean=image_mean,
                 intensity_source=report.intensity_source(fast.measurements))
    source.update(id=ident, status="ok", fast_meta=fast.meta,
                  reference_meta=None if reference is None else reference.meta, reference_error=reference_error)
    return entry, rows, comparisons, timing, source, samples


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"Folder not found: {args.folder}")
    if not np.isfinite(args.match_distance) or args.match_distance < 0:
        parser.error("--match-distance must be finite and nonnegative")
    if args.max_difference is not None and (not np.isfinite(args.max_difference) or args.max_difference < 0):
        parser.error("--max-difference must be finite and nonnegative")
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
    fast_methods = FAST_METHODS + (PIXEL_METHODS if args.include_pixel else ())
    references = () if args.fast_only else REFERENCE_METHODS
    methods = fast_methods + references
    # Validate numerical options before creating a partial report on a typo.
    measure_cds(np.zeros((8, 8), np.uint8), methods=fast_methods, **{**options, "backend": "cpu"})
    for folder in ("images", "data", "arrays"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    print(f"Fast CD report: {len(images)} PNGs; {backend.upper()} device {args.device}; "
          + ("slow references on every image." if references else "fast methods only."), flush=True)
    started = time.perf_counter()
    entries, rows, comparisons, timings, sources = [], [], [], [], []
    samples, used = {}, set()
    rng = np.random.default_rng(0)
    with measurement_runtime(backend, args.device):
        for number, path in enumerate(images, 1):
            name = path.relative_to(args.folder).as_posix()
            ident = re.sub(r"[^A-Za-z0-9_-]", "_", str(Path(name).with_suffix(""))) or "image"
            while ident in used:
                ident += "_"
            used.add(ident)
            try:
                entry, values, changes, timing, source, values_by_layer = _process_image(
                    path, name, ident, output, fast_methods, references, options, args.match_distance, rng)
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
            comparisons.extend(changes)
            timings.append(timing)
            sources.append(source)
            for layer, values in values_by_layer.items():
                samples.setdefault(layer, []).append(values)
            reference_text = ("off" if not references else "failed" if timing["reference_error"]
                              else f"{timing['reference_s']:.2f} s")
            print(f"[{number}/{len(images)}] {name}: fast {timing['fast_s']:.2f} s; reference {reference_text}", flush=True)
    aggregates = aggregate_comparisons(comparisons)
    write_csv(output / "summary.csv", rows, ("image", "status", "method"))
    between = report.between_image_statistics(entries, methods)
    brightness = report.brightness_between(entries)
    write_csv(output / "between_images.csv", [dict(method=m, **values, unit=args.unit) for m, values in between.items()]
              + [dict(method=m, **values, unit="counts") for m, values in brightness.items()], ("method",))
    write_csv(output / "comparison.csv", comparisons, ("image_id", "method_reference", "method_candidate", "axis"))
    write_csv(output / "comparison_between_images.csv", aggregates, ("method_reference", "method_candidate", "axis"))
    write_csv(output / "timings.csv", timings, ("image", "status"))
    plot = "comparison.png" if write_comparison_plot(output / "comparison.png", comparisons, args.unit) else None
    extra = comparison_html(comparisons, aggregates, args.unit, max_difference=args.max_difference, plot_file=plot)
    failures = [e for e in entries if e.get("error")]
    reference_failures = [t for t in timings if t.get("reference_error")]
    if reference_failures:
        extra += "<p class='error'>Reference calculation failed: " + "; ".join(
            html.escape(f"{t['image']}: {t['reference_error']}") for t in reference_failures) + "</p>"
    notes = ["Fast measurements run independently of the model reference; position matches use a mutual nearest "
             f"seed within {args.match_distance:g} px. Differences are fast minus reference in {args.unit}.",
             "Maps show fast-method area-equivalent circles and numerical hole values. Slow reference values "
             "are in the statistics and comparison tables. Cell intensity uses the raw-cell local tail region.",
             "Between-image spread describes these images; repeatability interpretation requires repeated "
             "acquisitions of the same target and settings. The paired valid-hole population can change between images."]
    if not references:
        notes.insert(0, "Slow references were not run (--fast-only); comparability has not been assessed by this report.")
    ranges = {}
    for layer, parts in samples.items():
        values = np.concatenate(parts)
        if len(values):
            ranges[layer] = np.quantile(values, [0.02, 0.98]).tolist()
    page = report.write_report(output, entries, methods, notes, args.unit, ranges, extra_summary=extra)
    elapsed = time.perf_counter() - started
    manifest = dict(schema_version=1, source_folder=str(args.folder.resolve()),
                    measurement_fingerprint=measurement_fingerprint(), settings=options,
                    fast_methods=fast_methods, reference_methods=references, max_distance_px=args.match_distance,
                    max_difference=args.max_difference, elapsed_s=elapsed, images=sources,
                    successful_images=len(entries) - len(failures), failed_images=len(failures),
                    reference_failures=len(reference_failures),
                    note="Timings include first-use compilation; fast/reference measurement and output are separate. "
                         "Comparison is against the selected model estimators, not physical ground truth.")
    (output / "run.json").write_text(json.dumps(finite_json(manifest), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {page} ({elapsed:.1f} s; {len(failures)} failed images, "
          f"{len(reference_failures)} failed references).", flush=True)
    return int(bool(failures or reference_failures))


if __name__ == "__main__":
    raise SystemExit(main())
