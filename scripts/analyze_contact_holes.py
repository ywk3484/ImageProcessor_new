"""Extract contact-hole cells, measure three CD definitions, and export maps/data.

Run from the repository root:
    python scripts/analyze_contact_holes.py
    python scripts/analyze_contact_holes.py image.tif --invert --pixel-size 2 --unit nm
    python scripts/analyze_contact_holes.py stack.tiff --page 10 --output artifacts/page_10
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

from subpx import measure_cds, plot_cd_map, save_cd_results, save_cd_viewer
from subpx.cd import METHOD_LABELS


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


def save_figures(result, output: Path, metric: str, *, shared_scale: bool = False) -> list[str]:
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
    for contours, name in ((False, "cd_maps.png"), (True, "contours.png")):
        fig, axes = plt.subplots(len(methods), 1, figsize=(13, 3.6 * len(methods)), squeeze=False,
                                 constrained_layout=True)
        for ax, method in zip(axes[:, 0], methods):
            plot_cd_map(result, method=method, metric=metric, contours=contours, ax=ax, vmin=low, vmax=high)
        fig.suptitle("Contact-hole contours" if contours else "Contact-hole CD by Voronoi cell", fontsize=15)
        fig.savefig(output / name, dpi=180, bbox_inches="tight")
        plt.close(fig)
        files.append(name)

    ax = plot_cd_map(result, metric="mean_intensity")
    name = "intensity_map.png"
    ax.figure.savefig(output / name, dpi=180, bbox_inches="tight")
    plt.close(ax.figure)
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
    if reference is not None:
        valid_ids = np.flatnonzero(reference.valid)
        ordered = valid_ids[np.argsort(reference.metrics["cd_equivalent_px"][valid_ids])]
        selections = list(dict.fromkeys(ordered[[0, len(ordered) // 2, len(ordered) - 1]].tolist()))
        fig, axes = plt.subplots(1, len(selections), figsize=(4.5 * len(selections), 4.3), squeeze=False,
                                 constrained_layout=True)
        colors = {"logquad": "#ffb000", "gradient": "#00cfec", "halfmax": "#ff5aa5"}
        for ax, i in zip(axes[0], selections):
            x0, y0 = result.origins_xy[i]
            h, w = result.rois[i].shape
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, nargs="?", default=ROOT / "examples/data/synthetic_ch/contact_holes.png")
    parser.add_argument("--page", type=int, default=0, help="Zero-based TIFF page index (default: 0, the first image)")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/contact_holes_cd")
    parser.add_argument("--methods", nargs="+", choices=list(METHOD_LABELS), default=list(METHOD_LABELS))
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
    parser.add_argument("--cell-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--metric", choices=("cd_equivalent_px", "cd_x_px", "cd_y_px", "area_px2"), default="cd_equivalent_px")
    parser.add_argument("--shared-color-scale", action="store_true", help="Use the same color limits for all methods")
    parser.add_argument("--ground-truth", type=Path, help="Optional NPZ containing centers_xy and fwhm_xy")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    start = time.perf_counter()
    try:
        image = load_image(args.image, page=args.page)
    except ValueError as exc:
        parser.error(str(exc))
    input_page = args.page if args.image.suffix.lower() in {".tif", ".tiff"} else None
    page_info = f", page={input_page}" if input_page is not None else ""
    print(f"Input: {args.image} | shape={image.shape}, dtype={image.dtype}{page_info}", flush=True)
    result = measure_cds(
        image, methods=tuple(args.methods), threshold=args.threshold, invert=args.invert,
        area_min=args.area_min, area_max=args.area_max, pad=args.pad,
        n_angles=args.n_angles, radial_step=args.radial_step, smooth_sigma=args.smooth_sigma,
        fit_fraction=args.fit_fraction, pixel_size=args.pixel_size, unit=args.unit,
        cell_backend=args.cell_backend, device=args.device,
    )
    measurement_seconds = time.perf_counter() - start
    print(f"Extracted {len(result.seeds_xy)} cells in {measurement_seconds:.2f} s", flush=True)
    for method, measured in result.measurements.items():
        values = measured.metrics["cd_equivalent_px"][measured.valid] * result.pixel_size
        median = f"{np.median(values):.4f}" if len(values) else "n/a"
        print(f"  {method}: {measured.valid.sum()}/{len(measured.valid)} valid; median CD={median} {result.unit}", flush=True)
    print("Writing raw data and figures...", flush=True)
    paths = save_cd_results(result, args.output)
    figures = save_figures(result, args.output, args.metric, shared_scale=args.shared_color_scale)
    viewer = save_cd_viewer(result, args.output / "contours.html", metric=args.metric)
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    manifest.update(
        input_image=str(args.image.resolve()), input_page=input_page, input_sha256=_file_sha256(args.image),
        pipeline_source=str(Path(__file__).resolve()), measurement_seconds=measurement_seconds,
        plot_metric=args.metric, shared_color_scale=args.shared_color_scale, figures=figures,
        interactive_viewer=viewer.name,
    )
    if args.ground_truth:
        manifest["accuracy"] = compare_ground_truth(result, args.ground_truth, args.output)
        manifest["files"] += ["truth_comparison.csv", "accuracy.json"]
    manifest["files"] += figures + [viewer.name]
    manifest["elapsed_seconds"] = time.perf_counter() - start
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved to {args.output.resolve()} ({manifest['elapsed_seconds']:.2f} s total)", flush=True)


if __name__ == "__main__":
    main()
