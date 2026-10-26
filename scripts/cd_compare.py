"""Compare model and fast local CD methods using an existing batch run's settings.

    python scripts/cd_compare.py D:\\results\\trial D:\\data\\scan.tif

Selects up to eight previously measured pages spanning the recorded runtimes.
Times each scheme separately after warm-up, checks a tighter model solve on the
median and slowest selected pages, and saves per-hole numerical arrays for later
comparison across repeated datasets. No dataset-wide repeatability is inferred.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cd_batch import FrameReader, PccCheck, measurement_fingerprint, read_image
from scripts.cd_report import ALL_METHODS, cd_statistics
from subpx import measure_cds
from subpx.cd import LOCAL_METHODS

PAIRS = {"projection": "local_projection", "half_area": "local_half_area",
         "half_area_min_avg": "local_half_area_min_avg", "tail_area": "local_tail_area"}
DEFAULT_LOCAL_METHODS = tuple(name for name in LOCAL_METHODS if name != "local_projection")


def rows_for(path, tiff, file_key=None):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if
                (r.get("file") == file_key if file_key else Path(r.get("file", "")).name.lower() == tiff.name.lower())]
    if len({r["file"] for r in rows}) > 1:
        raise ValueError("Several input files share this name; select their exact CSV path with --file-key")
    return rows


def select_pages(rows, count, explicit=None):
    """Deterministic runtime quantiles, including both extremes, without duplicates."""
    if count < 1:
        raise ValueError("page count must be positive")
    if explicit is not None:
        try:
            pages = list(dict.fromkeys(int(v) for v in explicit.split(",")))
        except ValueError:
            raise ValueError("--pages must be nonnegative comma-separated page numbers") from None
        if not pages or min(pages) < 0:
            raise ValueError("--pages must be nonnegative comma-separated page numbers")
        return pages
    measured = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            seconds, page = float(row["measure_s"]), int(row["page"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(seconds) and seconds >= 0 and page >= 0:
            measured[page] = seconds  # a resumed page may have more than one timing row
    ordered = sorted(measured, key=lambda page: (measured[page], page))
    if not ordered:
        raise ValueError("No measured pages found; give --pages with known hole pages")
    indices = np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    return [ordered[i] for i in indices]


def numeric_result(result):
    """Small independent arrays: enough to match physical holes and audit validity."""
    data = {"seeds_xy": result.seeds_xy.copy(), "pixel_size": np.asarray(result.pixel_size),
            "unit": np.asarray(result.unit), "methods": np.asarray(list(result.measurements))}
    for name, method in result.measurements.items():
        for key in ("centers_xy", "valid", "status"):
            data[f"{name}_{key}"] = getattr(method, key).copy()
        for key in ("cd_equivalent_px", "cd_x_px", "cd_y_px"):
            data[f"{name}_{key}"] = method.metrics[key].copy()
    return data


def matched_changes(a, b, pairs, max_distance=1.0):
    """Mutual nearest seed matches; changing validity is reported separately."""
    from scipy.spatial import cKDTree

    if str(a["unit"]) != str(b["unit"]):
        raise ValueError("Matched results must use the same unit")
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("Match distance must be nonnegative and finite")
    left, right = a["seeds_xy"], b["seeds_xy"]
    ia = ib = np.empty(0, dtype=int)
    if len(left) and len(right):
        distance, nearest = cKDTree(right).query(left)
        reverse = cKDTree(left).query(right)[1]
        ia = np.flatnonzero((distance <= max_distance) & (reverse[nearest] == np.arange(len(left))))
        ib = nearest[ia]
    scale_a, scale_b = float(a["pixel_size"]), float(b["pixel_size"])
    rows = []
    for old, new in pairs.items():
        if f"{old}_valid" not in a or f"{new}_valid" not in b:
            continue
        va, vb = a[f"{old}_valid"][ia], b[f"{new}_valid"][ib]
        all_a = a[f"{old}_cd_equivalent_px"] * scale_a
        all_b = b[f"{new}_cd_equivalent_px"] * scale_b
        valid_a = a[f"{old}_valid"] & np.isfinite(all_a)
        valid_b = b[f"{new}_valid"] & np.isfinite(all_b)
        mean_a = float(all_a[valid_a].mean()) if valid_a.any() else None
        mean_b = float(all_b[valid_b].mean()) if valid_b.any() else None
        av, bv = all_a[ia], all_b[ib]
        both = va & vb & np.isfinite(av) & np.isfinite(bv)
        delta = bv[both] - av[both]
        rows.append(dict(method_a=old, method_b=new, matched_positions=len(ia), valid_both=int(both.sum()),
                         only_a_valid=int((va & ~vb).sum()), only_b_valid=int((vb & ~va).sum()),
                         unmatched_a=len(left) - len(ia), unmatched_b=len(right) - len(ib),
                         valid_a=int(valid_a.sum()), valid_b=int(valid_b.sum()), mean_a=mean_a, mean_b=mean_b,
                         page_mean_change=None if mean_a is None or mean_b is None else mean_b - mean_a,
                         matched_mean_change=float(delta.mean()) if delta.size else None,
                         matched_p99_abs_change=float(np.percentile(np.abs(delta), 99)) if delta.size else None,
                         unit=str(a["unit"])))
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pages", help="Known hole pages, e.g. 94,46; default runtime quantiles of the prior run")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--file-key", help="Exact file column if several files have the same basename")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--local-methods", nargs="+", choices=LOCAL_METHODS, default=DEFAULT_LOCAL_METHODS,
                        help="Fast methods (default: the three local area methods; projection is optional)")
    parser.add_argument("--skip-stability", action="store_true", help="Skip tighter fits on two selected pages")
    args = parser.parse_args(argv)
    if args.count < 1 or args.repeats < 1:
        parser.error("--count and --repeats must be positive")
    if len(set(args.local_methods)) != len(args.local_methods):
        parser.error("--local-methods must be unique")
    try:
        previous = json.loads((args.run_folder / "run.json").read_text(encoding="utf-8"))
        timings = rows_for(args.run_folder / "timings.csv", args.tiff, args.file_key)
        pages = select_pages(timings, args.count, args.pages)
        previous_measure = previous["settings"]["measure"]
        if not isinstance(previous_measure, dict):
            raise ValueError("run.json settings.measure must be an object")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    options = {key: value for key, value in previous_measure.items()
               if key in inspect.signature(measure_cds).parameters}
    options.update(backend=args.backend, device=args.device, metrics_only=True, outlines=False, n_angles=16)
    model_methods = tuple(m for m in options.pop("methods", ALL_METHODS) if m not in LOCAL_METHODS)
    if not model_methods:
        model_methods = ALL_METHODS
    local_methods = tuple(args.local_methods)
    output = args.output or args.run_folder / time.strftime("comparison_%Y%m%d_%H%M%S")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output is not empty; use a new folder")

    # Read the selected pages once. This command times computation, not cold disk I/O.
    reader = FrameReader(args.tiff)
    try:
        images = {page: reader.read(page) for page in pages}
        old_pages = rows_for(args.run_folder / "pages.csv", args.tiff, args.file_key)
        blank_pages = list(dict.fromkeys(int(r["page"]) for r in old_pages if r.get("status") == "no_holes"))[:2]
        screen_images = {**images, **{page: reader.read(page) for page in blank_pages
                                     if page not in images and 0 <= page < reader.count}}
    finally:
        reader.close()
    check_options = previous["settings"].get("hole_check")
    reference = (read_image(Path(check_options["reference"]), check_options.get("reference_page", 0))
                 if check_options else None)
    output.mkdir(parents=True, exist_ok=True)
    arrays_dir = output / "holes"
    arrays_dir.mkdir()
    xp = np
    if args.backend == "gpu":
        import cupy as cp
        from subpx._gpu import _cuda
        cp.cuda.Device(args.device).use()
        _cuda.KEEP_MEMORY = True
        xp = cp
    print(f"Compare model and local methods on {len(pages)} pages; warm-up excluded; one device.", flush=True)
    results, changes, stability, screen, durations, stored = [], [], [], [], {"model": [], "local": []}, {}
    strict_pages = set() if args.skip_stability else {pages[len(pages) // 2], pages[-1]}
    for index, page in enumerate(pages):
        image = images[page]
        for mode, methods in (("model", model_methods), ("local", local_methods)):
            # Warm this page's cell sizes too: an outlier may compile different kernels.
            measure_cds(image, methods=methods, **options)
            times = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                result = measure_cds(image, methods=methods, **options)
                times.append(time.perf_counter() - started)
            elapsed = float(np.median(times))
            durations[mode].append(elapsed)
            data = numeric_result(result)
            stored[mode] = data
            np.savez_compressed(arrays_dir / f"page_{page}_{mode}.npz", **data)
            row = dict(page=page, pipeline=mode, measure_s=elapsed, holes=len(result.seeds_xy),
                       deblend_converged=result.meta.get("deblend_converged"),
                       deblend_iterations=result.meta.get("deblend_iterations_used", 0))
            for name, stats in cd_statistics(result).items():
                for field in ("valid", "mean", "mean_x", "mean_y"):
                    row[f"{name}_{field}"] = stats[field]
            results.append(row)
            del result
        changes.extend(dict(page=page, **row) for row in matched_changes(stored["model"], stored["local"], PAIRS))
        # A tighter solve can be slow. Preserve the completed comparison before it starts.
        write_csv(output / "pages.csv", results)
        write_csv(output / "changes.csv", changes)
        if page in strict_pages:
            strict = {**options, "deblend_iterations": max(60, 2 * options.get("deblend_iterations", 30)),
                      "deblend_tolerance": min(1e-4, options.get("deblend_tolerance", 1e-3) / 10)}
            result = measure_cds(image, methods=model_methods, **strict)
            data = numeric_result(result)
            np.savez_compressed(arrays_dir / f"page_{page}_tighter.npz", **data)
            records = matched_changes(stored["model"], data, {name: name for name in model_methods})
            stability.extend(dict(page=page, tighter_converged=result.meta["deblend_converged"],
                                  tighter_iterations=result.meta["deblend_iterations_used"], **row) for row in records)
            del result
        print(f"Page {page}: model {durations['model'][-1]:.2f} s, local {durations['local'][-1]:.2f} s "
              f"({index + 1}/{len(pages)})", flush=True)
        # Keep completed measurements if the user interrupts a later page.
        write_csv(output / "pages.csv", results)
        write_csv(output / "changes.csv", changes)
        write_csv(output / "stability.csv", stability)
    if reference is not None:
        raw, centered = PccCheck(reference, xp=xp), PccCheck(reference, xp=xp, mode="centered")
        prior_status = {int(row["page"]): row["status"] for row in old_pages}
        for page, image in screen_images.items():
            a, b = raw(image), centered(image)
            screen.append(dict(page=page, previous_status=prior_status.get(page, ""), raw_error=a, centered_error=b,
                               raw_default_threshold=0.3, centered_default_threshold=0.8,
                               raw_default_accept=a < 0.3, centered_default_accept=b < 0.8))
        write_csv(output / "screen.csv", screen)
    model_s, local_s = (float(np.median(durations[name])) for name in ("model", "local"))
    unsettled = sum(row.get("deblend_converged") is False for row in results if row["pipeline"] == "model")
    delta = max((abs(row["matched_mean_change"]) for row in stability if row["matched_mean_change"] is not None), default=None)
    page_delta = max((abs(row["page_mean_change"]) for row in stability if row["page_mean_change"] is not None), default=None)
    summary = dict(measurement_fingerprint=measurement_fingerprint(), tiff=str(args.tiff.resolve()), pages=pages,
                   source_run=str(args.run_folder.resolve()), source_git_commit=previous.get("git_commit"),
                   source_measurement_fingerprint=previous.get("measurement_fingerprint"), source_hole_check=check_options,
                   settings=options, model_methods=model_methods, local_methods=local_methods, backend=args.backend,
                   median_seconds={"model": model_s, "local": local_s}, speed_ratio=model_s / local_s,
                   unsettled_pages=unsettled, max_matched_mean_change_tighter=delta,
                   max_page_mean_change_tighter=page_delta,
                   note="One-device compute timing, selected prior hole pages; not dataset throughput or repeatability.")
    (output / "summary.json").write_text(json.dumps(finite_json(summary), indent=2) + "\n", encoding="utf-8")
    print(f"Model {model_s:.2f} s/page; local {local_s:.2f} s/page ({model_s / local_s:.1f}x).", flush=True)
    shift = "not checked" if page_delta is None else f"{page_delta:.3f} {options.get('unit', 'px')}"
    print(f"Unsettled {unsettled}/{len(pages)}; extra-round mean change {shift}.", flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
