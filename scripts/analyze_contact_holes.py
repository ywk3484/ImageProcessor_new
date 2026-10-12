"""Extract contact-hole cells, measure up to four CD definitions, and export maps/data.

Run from the repository root:
    python scripts/analyze_contact_holes.py
    python scripts/analyze_contact_holes.py image.tif --invert --pixel-size 2 --unit nm
    python scripts/analyze_contact_holes.py stack.tiff --page 10 --output artifacts/page_10
    python scripts/analyze_contact_holes.py dense.png --seeds peaks          # pitch/FWHM below ~1.6
    python scripts/analyze_contact_holes.py image.png --background border    # previous per-cell background
    python scripts/analyze_contact_holes.py data/*.png --backend gpu --outputs compact --output results
        # many large images: one subfolder per image plus batch_summary.csv; CSVs and figures are
        # written by a second process while the GPU measures the next image
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import load_cd_results, measure_cds, plot_cd_map, save_cd_results, save_cd_viewer
from subpx.cd import METHOD_LABELS, save_cd_arrays, save_cd_tables


def load_image(path: Path, *, page: int = 0) -> np.ndarray:
    """Read one grayscale image, decoding only the selected zero-based TIFF page."""
    if page < 0:
        raise ValueError("--page must be >= 0 (page indexes start at 0)")
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile
        with tifffile.TiffFile(path) as tif:
            count = len(tif.pages)
            if page >= count:
                raise ValueError(f"TIFF page {page} is out of range: {path.name} has {count} pages (indexes start at 0)")
            image = tif.pages[page].asarray()
    else:
        if page != 0:
            raise ValueError("Nonzero --page is only supported for TIFF files")
        from PIL import Image
        with Image.open(path) as loaded:
            image = np.asarray(loaded).copy()
    if image.ndim != 2:
        raise ValueError("Selected image must be 2D grayscale; export a grayscale plane before analysis.")
    return image


def _file_sha256(path: Path) -> str:
    """Hash the source file without loading a potentially large TIFF into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figures(result, output: Path, metric: str, *, shared_scale: bool = False, compact: bool = False) -> list[str]:
    """Overview figures; ``compact`` keeps the CD maps, background map and histograms only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = list(result.measurements)
    power = 2 if metric == "area_px2" else 1
    valid_values = np.concatenate([
        m.metrics[metric][m.valid] * result.pixel_size**power for m in result.measurements.values()
    ])
    low, high = (float(valid_values.min()), float(valid_values.max())) if valid_values.size else (0, 1)
    if not shared_scale:
        low, high = None, None
    files = []
    # Contours of ~1e5 holes are unreadable at figure resolution and slow to draw.
    for contours, name in ((False, "cd_maps.png"),) + (() if compact else ((True, "contours.png"),)):
        fig, axes = plt.subplots(len(methods), 1, figsize=(13, 3.6 * len(methods)), squeeze=False,
                                 constrained_layout=True)
        for ax, method in zip(axes[:, 0], methods):
            plot_cd_map(result, method=method, metric=metric, contours=contours, ax=ax, vmin=low, vmax=high)
        fig.suptitle("Contact-hole contours" if contours else "Contact-hole CD by Voronoi cell", fontsize=15)
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    if not compact:
        ax = plot_cd_map(result, metric="mean_intensity")
        name = "intensity_map.png"
        ax.figure.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(ax.figure)
        files.append(name)

    if result.background_map is not None:
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        artist = ax.imshow(result.background_map, cmap="magma", interpolation="nearest")
        fig.colorbar(artist, ax=ax, label="Fitted background (original intensity units)", pad=0.02)
        ax.set(title=f"Background map | source: {result.meta.get('background_source')}",
               xlabel="x (px)", ylabel="y (px)")
        name = "background_map.png"
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 3.5), squeeze=False,
                             constrained_layout=True)
    for ax, (method, measurement) in zip(axes[0], result.measurements.items()):
        values = measurement.metrics["cd_equivalent_px"][measurement.valid] * result.pixel_size
        # Nearly identical CDs can span fewer than 40 representable floats.
        # Give their single bin a visible width instead of plotting roundoff.
        bins, hist_range = 40, None
        if len(values) and np.isclose(values.min(), values.max(), rtol=1e-12, atol=0):
            center = float(values[0])
            half_width = max(0.5, 0.05 * abs(center))
            bins, hist_range = 1, (center - half_width, center + half_width)
        ax.hist(values, bins=bins, range=hist_range, color="#327da8", alpha=0.85)
        if len(values):
            ax.axvline(np.median(values), color="#d35333", ls="--", label=f"Median {np.median(values):.3f}")
            ax.legend(frameon=False)
        ax.set(title=METHOD_LABELS[method], xlabel=f"Area-equivalent CD ({result.unit})", ylabel="Cells")
    name = "cd_distributions.png"
    fig.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    files.append(name)

    reference = next((m for m in result.measurements.values() if m.valid.any()), None)
    if reference is not None and not compact:
        valid_ids = np.flatnonzero(reference.valid)
        ordered = valid_ids[np.argsort(reference.metrics["cd_equivalent_px"][valid_ids])]
        selections = list(dict.fromkeys(ordered[[0, len(ordered) // 2, len(ordered) - 1]].tolist()))
        fig, axes = plt.subplots(1, len(selections), figsize=(4.5 * len(selections), 4.3), squeeze=False,
                                 constrained_layout=True)
        colors = {"gaussian": "#7cfc00", "logquad": "#ffb000", "gradient": "#00cfec", "halfmax": "#ff5aa5"}
        for ax, i in zip(axes[0], selections):
            x0, y0 = result.origins_xy[i]
            h, w = result.masks[i].shape
            ax.imshow(result.image, cmap="gray", interpolation="nearest")
            ax.contour(np.arange(w) + x0, np.arange(h) + y0, result.masks[i], levels=[0.5],
                       colors=["#bbbbbb"], linewidths=0.7, linestyles="dashed")
            for method, measurement in result.measurements.items():
                if measurement.valid[i]:
                    ax.plot(*measurement.contours_xy[i].T, color=colors[method], lw=1.7)
            ax.set(xlim=(x0 - 0.5, min(x0 + w, result.image.shape[1]) - 0.5),
                   ylim=(min(y0 + h, result.image.shape[0]) - 0.5, y0 - 0.5),
                   xlabel="x (px)", ylabel="y (px)",
                   title=f"Cell {i} | {reference.metrics['cd_equivalent_px'][i] * result.pixel_size:.3f} {result.unit}")
        handles = [Line2D([], [], color=colors[m], label=METHOD_LABELS[m]) for m in methods]
        fig.legend(handles=handles, loc="outside upper center", ncol=len(methods), frameon=False)
        name = "cell_examples.png"
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)
    return files


def compare_ground_truth(result, path: Path, output: Path) -> dict:
    """Compare the synthetic generator's FWHM truth using unique spatial matches."""
    from scipy.spatial import cKDTree

    with np.load(path, allow_pickle=False) as truth:
        centers = np.asarray(truth["centers_xy"], dtype=float)
        widths = np.asarray(truth["fwhm_xy"], dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or widths.shape != centers.shape or len(centers) == 0:
        raise ValueError("Ground truth requires nonempty centers_xy and fwhm_xy arrays, both (N, 2)")
    distances, ids = cKDTree(centers).query(result.seeds_xy)
    matched = distances < 1.0
    matched_ids, counts = np.unique(ids[matched], return_counts=True)
    matched &= ~np.isin(ids, matched_ids[counts > 1])
    expected_fwhm = np.sqrt(np.prod(widths[ids], axis=1))
    summary = {"input": str(path.resolve()), "matched_cells": int(matched.sum()),
               "unmatched_cells": int((~matched).sum()), "methods": {},
               "note": "Native edge definitions: FWHM for logquad/halfmax; 2 sigma for gradient. Truth assumes Gaussian holes. Any smoothing bias remains in these errors."}
    with (output / "truth_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "truth_id", "method", "matched", "valid", "expected_cd_px", "measured_cd_px", "error_px"])
        for name, measured in result.measurements.items():
            expected = expected_fwhm / np.sqrt(2 * np.log(2)) if name == "gradient" else expected_fwhm
            errors = measured.metrics["cd_equivalent_px"] - expected
            usable = matched & measured.valid
            values = errors[usable]
            summary["methods"][name] = {
                "compared": int(usable.sum()),
                "bias_px": float(values.mean()) if values.size else None,
                "rmse_px": float(np.sqrt(np.mean(values**2))) if values.size else None,
                "p95_abs_error_px": float(np.percentile(abs(values), 95)) if values.size else None,
            }
            for i in range(len(result.seeds_xy)):
                writer.writerow([i, int(ids[i]) if matched[i] else -1, name, bool(matched[i]), bool(measured.valid[i]),
                                 expected[i] if matched[i] else np.nan, measured.metrics["cd_equivalent_px"][i],
                                 errors[i] if usable[i] else np.nan])
    (output / "accuracy.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def _expand_inputs(paths: list[Path]) -> list[Path]:
    """Expand wildcards (for shells that do not) and folders (their .png/.tif/.tiff files)."""
    import glob

    images = []
    for path in paths:
        if any(ch in str(path) for ch in "*?[") and not path.exists():
            images += [Path(p) for p in sorted(glob.glob(str(path)))]
        elif path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff"})
        else:
            images.append(path)
    return images


def _measure(path: Path, args):
    """Load and measure one image, printing its summary; returns (result, seconds, TIFF page or None)."""
    start = time.perf_counter()
    image = load_image(path, page=args.page)
    input_page = args.page if path.suffix.lower() in {".tif", ".tiff"} else None
    page_info = f", page={input_page}" if input_page is not None else ""
    print(f"Input: {path} | shape={image.shape}, dtype={image.dtype}{page_info}", flush=True)
    result = measure_cds(
        image, methods=None if args.methods is None else tuple(args.methods), background=args.background,
        seeds=args.seeds, peak_min_distance=args.peak_min_distance, background_level=args.background_level,
        background_tile=args.background_tile, deblend_iterations=args.deblend_iterations,
        threshold=args.threshold, invert=args.invert,
        area_min=args.area_min, area_max=args.area_max, pad=args.pad,
        n_angles=args.n_angles, radial_step=args.radial_step, smooth_sigma=args.smooth_sigma,
        fit_fraction=args.fit_fraction, pixel_size=args.pixel_size, unit=args.unit,
        backend=args.backend, cell_backend=args.cell_backend, device=args.device,
    )
    seconds = time.perf_counter() - start
    print(f"Extracted {len(result.seeds_xy)} cells in {seconds:.2f} s ({result.meta['measurement_backend']})", flush=True)
    if result.background_map is not None:
        m = result.meta
        print(f"  background: {m['background_source']} ({m['background_measured_tiles']}/{m['background_tiles']} tiles "
              f"measured), deblend iterations {m['deblend_iterations_used']}, seeds +{m['seeds_added_from_residual']}"
              f"/-{m['seeds_dropped_as_insignificant']}", flush=True)
        if m["background_source"] == "inferred":
            print("  WARNING: no hole-free area found; background inferred from the Gaussian model alone. "
                  "Pass --background-level if the background is known.", flush=True)
    for method, measured in result.measurements.items():
        values = measured.metrics["cd_equivalent_px"][measured.valid] * result.pixel_size
        median = f"{np.median(values):.4f}" if len(values) else "n/a"
        print(f"  {method}: {measured.valid.sum()}/{len(measured.valid)} valid; median CD={median} {result.unit}", flush=True)
    return result, seconds, input_page


def _options(args, path: Path, page, seconds: float, started: float) -> dict:
    """What the writing step needs besides the results (plain data: it may run in another process)."""
    return {
        "compact": args.outputs == "compact", "metric": args.metric, "shared_scale": args.shared_color_scale,
        "ground_truth": None if args.ground_truth is None else str(args.ground_truth), "started": started,
        "manifest": dict(input_image=str(path.resolve()), input_page=page, input_sha256=_file_sha256(path),
                         pipeline_source=str(Path(__file__).resolve()), measurement_seconds=seconds,
                         plot_metric=args.metric, shared_color_scale=args.shared_color_scale),
    }


def _finish(result, output: Path, options: dict) -> dict:
    """Figures, the viewer (full outputs only) and the manifest's run details; returns the manifest."""
    compact = options["compact"]
    figures = save_figures(result, output, options["metric"], shared_scale=options["shared_scale"], compact=compact)
    viewer = None if compact else save_cd_viewer(result, output / "contours.html", metric=options["metric"])
    path = output / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(options["manifest"], figures=figures, interactive_viewer=None if viewer is None else viewer.name)
    if options["ground_truth"]:
        manifest["accuracy"] = compare_ground_truth(result, Path(options["ground_truth"]), output)
        manifest["files"] += ["truth_comparison.csv", "accuracy.json"]
    manifest["files"] += figures + ([] if viewer is None else [viewer.name])
    manifest["elapsed_seconds"] = time.time() - options["started"]
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _write_in_background(output: str, options: dict) -> dict:
    """Writer-process job: CSVs, figures and manifest from the arrays the main process saved."""
    import matplotlib
    matplotlib.use("Agg")
    output = Path(output)
    result = load_cd_results(output, rois=False)  # tables and figures do not need the measured cells
    save_cd_tables(result, output, compact=options["compact"])
    return _finish(result, output, options)


def _summary_row(path: Path, output: Path, manifest: dict | None = None, error: str = "") -> dict:
    row = {"image": str(path), "output": str(output), "status": "failed" if error else "ok", "error": error,
           "height": "", "width": "", "cells": "", "measurement_seconds": "", "elapsed_seconds": ""}
    for method in METHOD_LABELS:
        row[f"{method}_valid"] = row[f"{method}_median_cd_px"] = ""
    if manifest is not None:
        row.update(height=manifest["shape_hw"][0], width=manifest["shape_hw"][1], cells=manifest["cells"],
                   measurement_seconds=round(manifest["measurement_seconds"], 3),
                   elapsed_seconds=round(manifest["elapsed_seconds"], 3))
        for method, summary in manifest["methods"].items():
            row[f"{method}_valid"] = summary["valid"]
            row[f"{method}_median_cd_px"] = "" if summary["median_cd_px"] is None else summary["median_cd_px"]
    return row


def _run_batch(images: list[Path], args) -> None:
    """Measure images one after another; a writer process saves image k while image k+1 is measured."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    names: dict[str, int] = {}
    outputs = []
    for path in images:
        count = names.get(path.stem, 0)
        names[path.stem] = count + 1
        outputs.append(args.output / (path.stem if count == 0 else f"{path.stem}_{count}"))
    rows: dict[int, dict] = {}

    def collect(job):
        index, future = job
        try:
            rows[index] = _summary_row(images[index], outputs[index], future.result())
        except Exception as exc:
            rows[index] = _summary_row(images[index], outputs[index], error=f"writing failed: {exc}")

    start = time.perf_counter()
    # Never fork a process that holds a CUDA context.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as writer:
        pending = None
        for index, (path, output) in enumerate(zip(images, outputs)):
            print(f"[{index + 1}/{len(images)}] ", end="", flush=True)
            started = time.time()
            try:
                result, seconds, page = _measure(path, args)
                save_cd_arrays(result, output, compact=args.outputs == "compact")
            except Exception as exc:  # keep going; the summary records the failure
                print(f"  FAILED: {exc}", flush=True)
                rows[index] = _summary_row(path, output, error=str(exc))
                continue
            options = _options(args, path, page, seconds, started)
            del result
            if pending is not None:
                collect(pending)  # at most one image waits for writing: bounded memory and disk lag
            pending = (index, writer.submit(_write_in_background, str(output), options))
        if pending is not None:
            collect(pending)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "batch_summary.csv"
    table = [rows[i] for i in sorted(rows)]
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    failed = sum(row["status"] != "ok" for row in table)
    print(f"Batch of {len(images)} images done in {time.perf_counter() - start:.1f} s ({failed} failed); "
          f"summary: {summary.resolve()}", flush=True)
    if args.outputs == "compact":
        print(f"Region viewer: python scripts/view_cd_region.py {outputs[0]} --region X Y W H", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="*",
                        help="Image files, wildcards or folders (default: the bundled synthetic example). With "
                             "several images, --output is a folder holding one subfolder per image.")
    parser.add_argument("--page", type=int, default=0, help="Zero-based TIFF page index (default: 0, the first image)")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/contact_holes_cd")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_LABELS), default=None,
                        help="Default: all four with --background model, all but gaussian with border")
    parser.add_argument("--background", choices=("model", "border"), default="model",
                        help="model: joint background map + neighbour-subtracted Gaussian fits (default); "
                             "border: previous per-cell 10th percentile")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold",
                        help="peaks: local maxima, for arrays too dense to threshold (pitch/FWHM below ~1.6)")
    parser.add_argument("--peak-min-distance", type=float, default=None, help="Default: half the estimated pitch")
    parser.add_argument("--background-level", type=float, default=None,
                        help="Known constant background (original intensity units) when no hole-free area exists")
    parser.add_argument("--background-tile", type=int, default=None, help="Background-map tile size in pixels")
    parser.add_argument("--deblend-iterations", type=int, default=30)
    parser.add_argument("--threshold", choices=("otsu", "triangle"), default="otsu")
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=50)
    parser.add_argument("--pad", type=int, default=3)
    parser.add_argument("--n-angles", type=int, default=128)
    parser.add_argument("--radial-step", type=float, default=0.1)
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    parser.add_argument("--fit-fraction", type=float, default=0.2)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="cpu",
                        help="gpu: CuPy cells, deblending and measurements (needs --background model); "
                             "auto: gpu when available")
    parser.add_argument("--cell-backend", choices=("cpu", "gpu"), default=None,
                        help="Voronoi partition only; default follows --backend")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--metric", choices=("cd_equivalent_px", "cd_x_px", "cd_y_px", "area_px2"), default="cd_equivalent_px")
    parser.add_argument("--shared-color-scale", action="store_true", help="Use the same color limits for all methods")
    parser.add_argument("--ground-truth", type=Path, help="Optional NPZ containing centers_xy and fwhm_xy (single image)")
    parser.add_argument("--outputs", choices=("full", "compact"), default="full",
                        help="compact: for large images/batches - measurements.csv, cells.csv, manifest, a lossless "
                             "uncompressed raw_data.npz and overview figures; no contours.csv, contours.png or "
                             "contours.html (write a region viewer with scripts/view_cd_region.py)")
    args = parser.parse_args(argv)
    images = _expand_inputs(args.images) if args.images else [ROOT / "examples/data/synthetic_ch/contact_holes.png"]
    if not images:
        parser.error("no images matched")
    if len(images) > 1 and args.ground_truth:
        parser.error("--ground-truth applies to a single image")

    import matplotlib
    matplotlib.use("Agg")

    if len(images) > 1:
        _run_batch(images, args)
        return
    path, started = images[0], time.time()
    try:
        result, seconds, page = _measure(path, args)
    except ValueError as exc:
        parser.error(str(exc))
    print("Writing raw data and figures...", flush=True)
    compact = args.outputs == "compact"
    save_cd_results(result, args.output, compact=compact)
    manifest = _finish(result, args.output, _options(args, path, page, seconds, started))
    print(f"Saved to {args.output.resolve()} ({manifest['elapsed_seconds']:.2f} s total)", flush=True)
    if compact:
        print(f"Region viewer: python scripts/view_cd_region.py {args.output} --region X Y W H", flush=True)


if __name__ == "__main__":
    main()
