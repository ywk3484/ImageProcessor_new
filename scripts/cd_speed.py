"""Measure original CD-method batch throughput using an earlier run's settings.

    python scripts/cd_speed.py D:\\results\\trial D:\\data\\scan.tif

Runs half_area, half_area_min_avg and tail_area once through cd_batch, retaining
Gaussian deblending, background correction and 4x cubic interpolation. Uses all
visible GPUs with one worker each. By default it selects up to 512 consecutive pages with
the most previously accepted hole pages. --pages A:B selects another interval.
The timer includes reference reading, worker startup, TIFF reading, screening,
measurement and output. No separate warm-up or stability measurements are run.
Disk cache state is uncontrolled; dataset projections assume the measured mix
of accepted/rejected pages and comparable I/O, and include this trial's startup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import cd_batch
from scripts.cd_compare import rows_for
from subpx.cd import LOCAL_METHODS

DEFAULT_METHODS = ("half_area", "half_area_min_avg", "tail_area")

MEASURE_FLAGS = {
    "seeds": "--seeds", "area_min": "--area-min", "area_max": "--area-max",
    "pixel_size": "--pixel-size", "unit": "--unit", "half_area_level": "--half-level",
    "half_area_peak_pixels": "--peak-pixels", "tail_area_level": "--tail-level",
    "tail_window": "--tail-window", "deblend_iterations": "--deblend-iterations",
    "deblend_tolerance": "--deblend-tolerance",
}


def page_window(rows, count=512, explicit=None):
    """A bounded contiguous interval, preferring the most prior accepted pages."""
    if count < 1:
        raise ValueError("--count must be positive")
    if explicit is not None:
        try:
            start, stop = (int(value) for value in explicit.split(":"))
        except ValueError:
            raise ValueError("--pages must be A:B with 0 <= A < B") from None
        if start < 0 or stop <= start:
            raise ValueError("--pages must be A:B with 0 <= A < B")
        return start, stop
    previous = {int(row["page"]): row.get("status") for row in rows if int(row["page"]) >= 0}
    if not previous or "ok" not in previous.values():
        raise ValueError("No prior accepted pages found for this TIFF; select a range with --pages A:B")
    stop = max(previous) + 1
    count = min(count, stop)
    prefix = [0]
    for page in range(stop):
        prefix.append(prefix[-1] + (previous.get(page) == "ok"))
    start = max(range(stop - count + 1), key=lambda p: prefix[p + count] - prefix[p])
    return start, start + count


def batch_arguments(args, settings, output, pages):
    """Pass saved numerical options to the existing batch command unchanged."""
    measure, check = settings["measure"], settings.get("hole_check")
    unsupported = set(measure) - set(MEASURE_FLAGS) - {
        "methods", "invert", "n_angles", "outlines", "metrics_only", "summary_only"}
    if unsupported:
        raise ValueError("Cannot reproduce saved settings: " + ", ".join(sorted(unsupported)))
    command = [str(args.tiff), "--output", str(output), "--pages", f"{pages[0]}:{pages[1]}",
               "--backend", args.backend, "--workers-per-gpu", str(args.workers_per_gpu),
               "--workers", str(args.workers), "--methods", *args.methods]
    for key, flag in MEASURE_FLAGS.items():
        if key in measure:
            command += [flag, str(measure[key])]
    if measure.get("invert"):
        command.append("--invert")
    if args.gpus:
        command += ["--gpus", *args.gpus]
    if check is None:
        command.append("--no-hole-check")
    else:
        command += ["--reference", check["reference"], "--reference-page", str(check.get("reference_page", 0)),
                    "--hole-check-mode", check.get("mode", "raw"), "--max-error", str(check["max_error"])]
        if check.get("error_function"):
            command += ["--error-function", check["error_function"]]
    return command


def result_summary(output, elapsed, invocation):
    """Count completed rows; failed or interrupted trials get no projection."""
    with (output / "pages.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = sum(row["status"] == "ok" for row in rows)
    rejected = sum(row["status"] == "no_holes" for row in rows)
    completed = accepted + rejected
    failed = invocation["counts"]["failed"]
    rate = completed / elapsed if elapsed > 0 else 0.0
    project = bool(invocation["complete"] and not failed and completed and rate > 0)
    return dict(elapsed_s=elapsed, completed_pages=completed, accepted_pages=accepted, rejected_pages=rejected,
                failed_pages=failed, complete=invocation["complete"], pages_per_s=rate,
                seconds_per_1500_pages=1500 / rate if project else None,
                hours_per_1000_files_of_1500_pages=1500000 / rate / 3600 if project else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_folder", type=Path)
    parser.add_argument("tiff", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pages", help="Consecutive page range A:B (B excluded)")
    parser.add_argument("--count", type=int, default=512, help="Automatic window size (default: 512 pages)")
    parser.add_argument("--file-key", help="Exact prior CSV file column if basenames are ambiguous")
    parser.add_argument("--methods", nargs="+", choices=(*cd_batch.ALL_METHODS, *LOCAL_METHODS),
                        default=DEFAULT_METHODS,
                        help="Default: half_area half_area_min_avg tail_area (deblended, 4x cubic)")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpus", nargs="+", help="Default: all visible GPUs")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="CPU workers (with --backend cpu)")
    args = parser.parse_args(argv)
    if args.workers_per_gpu < 1 or args.workers < 1:
        parser.error("worker counts must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("--methods must be unique")
    output = args.output or args.run_folder / time.strftime("speed_%Y%m%d_%H%M%S")
    if output.exists():
        parser.error("--output already exists; use a new folder for a fresh timing")
    try:
        if not args.tiff.is_file() or args.tiff.suffix.lower() not in (".tif", ".tiff"):
            raise ValueError(f"TIFF not found: {args.tiff}")
        previous = json.loads((args.run_folder / "run.json").read_text(encoding="utf-8"))
        rows = rows_for(args.run_folder / "pages.csv", args.tiff, args.file_key)
        pages = page_window(rows, args.count, args.pages)
        command = batch_arguments(args, previous["settings"], output, pages)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"CD batch throughput ({', '.join(args.methods)}): pages {pages[0]}:{pages[1]}; "
          "startup and I/O included; cache state unknown.",
          flush=True)
    started = time.perf_counter()
    code = cd_batch.main(command)
    elapsed = time.perf_counter() - started
    invocation = json.loads((output / "invocations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    summary = result_summary(output, elapsed, invocation)
    summary.update(source_run=str(args.run_folder.resolve()), source_git_commit=previous.get("git_commit"),
                   tiff=str(args.tiff.resolve()), pages=list(pages), methods=list(args.methods), backend=args.backend,
                   gpus=args.gpus, workers_per_gpu=args.workers_per_gpu, workers=args.workers,
                   hole_check=previous["settings"].get("hole_check"),
                   selection="explicit" if args.pages is not None else "most_prior_accepted_pages",
                   note="Projection assumes the same accepted/rejected page mix and comparable I/O. Disk cache is "
                        "uncontrolled. Startup is included and is not amortized over a full dataset.")
    (output / "speed.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Measured {summary['pages_per_s']:.2f} pages/s: {summary['completed_pages']} pages in {elapsed:.1f} s "
          f"({summary['accepted_pages']} accepted, {summary['rejected_pages']} rejected, "
          f"{summary['failed_pages']} failed).", flush=True)
    if summary["seconds_per_1500_pages"] is not None:
        print(f"Same mix/I/O: 1,500 pages ~{summary['seconds_per_1500_pages'] / 60:.1f} min; "
              f"1,000 such TIFFs ~{summary['hours_per_1000_files_of_1500_pages']:.1f} h. "
              "Selected sample; cache state unknown.", flush=True)
    else:
        print("No dataset projection: trial incomplete, failed, or measured no pages.", flush=True)
    print(f"Saved: {output}", flush=True)
    return code or (0 if summary["complete"] and summary["completed_pages"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
