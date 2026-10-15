"""Measure every PNG in a folder with all CD methods and write an offline HTML report.

    python scripts/cd_report.py FOLDER --output REPORT --nominal-cd 100 --pixel-size 21 --unit nm --backend gpu

Open REPORT/index.html in a browser (works from the folder, no server). It has a summary tab and one tab per image.

The summary tab first answers whether the images show the same CD on average. Per method, with m_i the mean CD
of image i (over the holes that method measured validly) and N the number of images:
    M = (m_1 + ... + m_N) / N
    s = sqrt( sum_i (m_i - M)^2 / (N - 1) )
    between-image 3 sigma % = 3 s / M x 100        (also given as 3 s in --unit)
Below that, per image and method: valid holes, mean CD and the within-image 3 sigma % = 3 sigma / mean x 100,
sigma being the standard deviation of that image's hole CDs (N - 1 in the denominator).

Each image tab starts with the CD map: the image with every detected hole drawn at its CD (drag to pan, wheel
to zoom; method, zoom and pan stay when switching tabs; colours on one scale for all images by default). Clicking
a hole shows how the selected method measured it (the cell it worked on, projection's x/y profiles and edges,
disk's radial fit and leftover, flux_ratio's numbers). Below the map, a band profile averages CD, brightness,
spot peak and sub-pixel position over bands of rows or columns, to tell where banding comes from. Then the
statistics, the per-hole agreement between methods and CD histograms. The statistics also go to
REPORT/summary.csv (per image) and REPORT/between_images.csv. --save-results also keeps the full per-hole
results in REPORT/results/<image>/ (``subpx.load_cd_results``; about 1 GB per 1e5-hole image with all methods).
Hole details take about 65 MB per 1e5-hole image in REPORT/data/<image>.detail.js (--no-hole-details skips them).

disk and flux_ratio need --nominal-cd or --psf-fwhm; without either they are skipped. With --nominal-cd they are
calibrated once, on the first image measured: the blur that makes its median disk diameter equal --nominal-cd and
the flux_ratio scale that does the same are reused for every later image, so image means stay comparable
(calibrating every image would force every image's median to --nominal-cd).
"""

from __future__ import annotations

import argparse
import base64
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

from subpx import measure_cds, save_cd_results
from subpx.cd import METHOD_LABELS

ALL_METHODS = ("gaussian", "logquad", "gradient", "halfmax", "disk", "flux_ratio", "projection")
DESCRIPTIONS = {
    "gaussian": "FWHM of a 2-D Gaussian fitted to the spot (deblended). Describes the blurred spot, not the hole.",
    "logquad": "FWHM of a Gaussian fitted to log intensity. Describes the blurred spot.",
    "gradient": "Contour through the steepest intensity drop along each ray. Spot edge.",
    "halfmax": "Contour at half the spot's peak height. Spot edge.",
    "disk": "Diameter of a round hole that, blurred by the optics and pixel-averaged, matches the spot. The hole itself; "
            "unaffected by the per-hole pulse strength. Absolute value follows --nominal-cd (or --psf-fwhm).",
    "flux_ratio": "Hole light divided by its own background, converted to a diameter and scaled so the median equals "
                  "--nominal-cd. Cancels the pulse when the background is pulse-lit absorber light (set --dark-level).",
    "projection": "Mean of the x and y widths between the steepest rise and fall of the cell's summed column and row profiles.",
}
# Extra map layers: (layer, metric, label) per method.
LAYERS = {"projection": (("projection_x", "cd_x_px", "projection, x width only"),
                         ("projection_y", "cd_y_px", "projection, y width only"))}


def cd_statistics(result) -> dict[str, dict]:
    """Per method: cells, valid holes, mean, sigma and 3 sigma % of the valid holes' CDs (in result.unit)."""
    stats = {}
    for name, measured in result.measurements.items():
        values = measured.metrics["cd_equivalent_px"][measured.valid] * result.pixel_size
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else float("nan")
        std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
        # The spread noise alone would give: the disk fit's own per-hole uncertainty (median).
        noise = measured.metrics.get("cd_uncertainty_px")
        noise = (float(np.nanmedian(noise[measured.valid]) * result.pixel_size) if noise is not None and values.size
                 else float("nan"))
        stats[name] = {"cells": int(len(measured.valid)), "valid": int(values.size), "mean": mean, "std": std,
                       "three_sigma_pct": 300 * std / mean if values.size > 1 and mean else float("nan"),
                       "noise_sigma": noise}
    return stats


def layer_values(result) -> dict[str, np.ndarray]:
    """Per map layer: each hole's CD in px, NaN where the method failed (projection adds its x and y widths)."""
    layers = {}
    for name, measured in result.measurements.items():
        layers[name] = np.where(measured.valid, measured.metrics["cd_equivalent_px"], np.nan)
        for layer, metric, _ in LAYERS.get(name, ()):
            layers[layer] = np.where(measured.valid, measured.metrics[metric], np.nan)
    return layers


def agreement(result) -> tuple[list[str], np.ndarray]:
    """Correlation of the per-hole CDs of every pair of methods, over the holes both measured."""
    names = list(result.measurements)
    r = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x = result.measurements[a].metrics["cd_equivalent_px"]
            y = result.measurements[b].metrics["cd_equivalent_px"]
            ok = result.measurements[a].valid & result.measurements[b].valid & np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0:
                r[i, j] = np.corrcoef(x[ok], y[ok])[0, 1]
    return names, r


def between_image_statistics(entries: list[dict], methods: tuple[str, ...]) -> dict[str, dict]:
    """Per method: spread of the per-image mean CDs over the images that method measured (see module docstring)."""
    stats = {}
    for name in methods:
        means = np.array([e["stats"][name]["mean"] for e in entries if not e.get("error") and name in e["stats"]])
        means = means[np.isfinite(means)]
        mean = float(means.mean()) if means.size else float("nan")
        std = float(means.std(ddof=1)) if means.size > 1 else float("nan")
        stats[name] = {"images": int(means.size), "mean_of_means": mean, "std_of_means": std, "three_sigma": 3 * std,
                       "three_sigma_pct": 300 * std / mean if means.size > 1 and mean else float("nan")}
    return stats


def display_image(image: np.ndarray) -> np.ndarray:
    """8-bit display copy, contrast-stretched between the 0.5 and 99.5 percentiles (measurements use the original)."""
    lo, hi = np.percentile(image, [0.5, 99.5])
    return np.clip(np.rint((image.astype(float) - lo) * (255 / max(hi - lo, 1e-12))), 0, 255).astype(np.uint8)


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def hole_data(result, layers: dict[str, np.ndarray]) -> dict:
    """Per map layer: hole centers, CD (px), validity and status as base64 arrays; plus what the band profile needs.

    Extra layers (projection's x/y widths) name their ``base`` method and share its centers and status.
    """
    seeds = result.seeds_xy
    methods = {}
    for name, measured in result.measurements.items():
        centers = np.where(np.isfinite(measured.centers_xy), measured.centers_xy, seeds)
        statuses, codes = np.unique(measured.status, return_inverse=True)
        methods[name] = {
            "label": METHOD_LABELS[name], "x": _b64(centers[:, 0], "<f4"), "y": _b64(centers[:, 1], "<f4"),
            "cd": _b64(layers[name], "<f4"),
            "valid": _b64(measured.valid, "u1"), "status": _b64(codes, "u1" if len(statuses) < 256 else "<u2"),
            "status_names": statuses.tolist(), "status_wide": len(statuses) >= 256,
        }
        for layer, _, label in LAYERS.get(name, ()):
            methods[layer] = {"label": label, "base": name, "cd": _b64(layers[layer], "<f4")}
    model = result.deblend_model
    # Spot peak above background: the deblend Gaussian's amplitude (mean cell brightness without a model).
    peak = model["params"][model["report"], 0] if model else result.mean_intensity
    image = result.image.astype(float)
    pitch = result.meta.get("pitch_px")
    return {"n": int(len(seeds)), "width": int(result.image.shape[1]), "height": int(result.image.shape[0]),
            "scale": float(result.pixel_size), "unit": result.unit, "methods": methods,
            # Background brightness of each pixel row / column: its 25th percentile, i.e. the absorber between holes.
            "peak": _b64(peak, "<f4"), "row_bg": _b64(np.percentile(image, 25, axis=1), "<f4"),
            "col_bg": _b64(np.percentile(image, 25, axis=0), "<f4"), "pitch": float(pitch) if pitch else None}


def _quantize(cells: np.ndarray, inside: np.ndarray) -> dict:
    """8-bit codes per cell pixel (255 = outside the cell) with each cell's offset and step, base64."""
    n = len(cells)
    flat = cells.reshape(n, -1).astype(float)
    inside = inside.reshape(n, -1) & np.isfinite(flat)
    lo = np.where(inside, flat, np.inf).min(axis=1)
    hi = np.where(inside, flat, -np.inf).max(axis=1)
    lo, hi = np.where(np.isfinite(lo), lo, 0.0), np.where(np.isfinite(hi), hi, 1.0)
    step = np.maximum(hi - lo, 1e-9) / 254
    codes = np.where(inside, np.rint((np.where(inside, flat, 0) - lo[:, None]) / step[:, None]), 255)
    return {"codes": _b64(codes, "u1"), "lo": _b64(lo, "<f4"), "step": _b64(step, "<f4")}


def detail_data(result, dark_level: float) -> dict:
    """What each method measured each hole on, for the click panel.

    ``clean``: the cell with background and neighbours' models removed (input of gaussian, logquad, gradient,
    halfmax, projection; zero outside the cell), 8-bit for display. ``prof_x``/``prof_y``: projection's column and
    row sums of that cell in full precision (8-bit sums move the steepest points visibly). ``disk``: the disk fit's own input cell, its fitted amplitude
    and background, and the blurred-hole profile table T(d, rho) of the fit (model = A T(d, rho) + B).
    """
    from subpx._cpu.invariant_cd import DiskTable

    n, h, w = result.masks.shape
    out = {"h": h, "w": w, "ox": _b64(result.origins_xy[:, 0], "<i4"), "oy": _b64(result.origins_xy[:, 1], "<i4"),
           "clean": _quantize(result.rois, result.masks)}
    if "projection" in result.measurements:
        cells = np.where(result.masks, result.rois, 0.0)
        out["prof_x"], out["prof_y"] = _b64(cells.sum(axis=1), "<f4"), _b64(cells.sum(axis=2), "<f4")
    fitted = result.measurements.get("disk") or result.measurements.get("flux_ratio")
    if fitted is not None and "cells" in fitted.inputs:
        out["disk"] = _quantize(fitted.inputs["cells"], np.isfinite(fitted.inputs["cells"]))
    if "disk" in result.measurements:
        m = result.measurements["disk"].metrics
        d = m["cd_equivalent_px"]
        d_max = float(np.nanmax(d)) * 1.2 + 1 if np.isfinite(d).any() else 3 * float(result.meta["psf_fwhm_px"])
        table = DiskTable(result.meta["psf_fwhm_px"], d_max, float(np.hypot(h, w)) + 1, dd=0.05, drho=0.05)
        out["disk_fit"] = {"amplitude": _b64(m["amplitude"], "<f4"), "background": _b64(m["local_background"], "<f4"),
                           "table": _b64(table.T, "<f4"), "nd": int(table.T.shape[0]), "nr": int(table.T.shape[1]),
                           "dd": table.dd, "drho": table.drho}
    if "flux_ratio" in result.measurements:
        m = result.measurements["flux_ratio"].metrics
        out["flux"] = {"flux": _b64(m["flux"], "<f4"), "background": _b64(m["local_background"], "<f4"),
                       "ratio": _b64(m["flux_ratio"], "<f4"), "scale": float(result.meta.get("flux_ratio_scale", 1.0)),
                       "dark": float(dark_level)}
    return out


def _fmt(value: float, digits: int = 2) -> str:
    return "–" if not np.isfinite(value) else f"{value:.{digits}f}"


def _plain(value: float, digits: int = 2) -> str:
    """``_fmt`` for the console: ASCII only, since e.g. a cp949 console cannot print an en dash."""
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def _stats_table(stats: dict, unit: str) -> str:
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(METHOD_LABELS[m])} "
        f"<code>{m}</code></td><td>{s['valid']:,} / {s['cells']:,}</td><td>{_fmt(s['mean'])}</td>"
        f"<td>{_fmt(s['std'])}</td><td><b>{_fmt(s['three_sigma_pct'])}</b></td><td>{_fmt(s['noise_sigma'])}</td></tr>"
        for m, s in stats.items())
    return (f"<table class='stats'><thead><tr><th>Method</th><th>Valid holes</th><th>Mean CD ({html.escape(unit)})</th>"
            f"<th>σ ({html.escape(unit)})</th><th>3σ (%)</th><th>σ from noise alone ({html.escape(unit)})</th></tr>"
            f"</thead><tbody>{rows}</tbody></table><p class='meta'>σ from noise alone (disk only): the median of the "
            "fit's own per-hole uncertainty. A measured σ much larger than this is real hole-to-hole variation or a "
            "systematic effect, not noise.</p>")


def _agreement_table(names: list[str], r: np.ndarray) -> str:
    head = "".join(f"<th>{html.escape(n)}</th>" for n in names)
    body = "".join(f"<tr><td>{html.escape(a)}</td>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
                   for a, row in zip(names, r))
    return ("<h3>Do the methods agree on which holes are big?</h3><p class='meta'>Correlation of the per-hole CDs of "
            "two methods, over the holes both measured: 1 = they rank the holes the same, 0 = unrelated. Two different "
            "methods that agree see the same thing in each hole (a real size difference, or something both take from "
            "the same input); low agreement means their hole-to-hole spread is mostly each method's own noise.</p>"
            f"<table class='stats wide'><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>")


COLORS = {"gaussian": "#5bbf2a", "logquad": "#e09a00", "gradient": "#00a9c4", "halfmax": "#e0468f",
          "disk": "#e03b3b", "flux_ratio": "#8f6ad8", "projection": "#7a7a7a", "projection_x": "#2b8cbe",
          "projection_y": "#e6550d"}


def write_report(output: Path, entries: list[dict], methods: tuple[str, ...], notes: list[str], unit: str,
                 ranges: dict | None = None) -> Path:
    """index.html: summary tab, one tab per image (data loaded on demand from data/<id>.js).

    ``ranges``: per map layer, the CD colour range (px) shared by all images.
    """
    def table(title, key):
        head = "".join(f"<th><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(m)}</th>"
                       for m in methods)
        body = []
        for e in entries:
            if e.get("error"):
                body.append(f"<tr><td>{html.escape(e['name'])}</td><td colspan='{1 + len(methods)}' class='error'>"
                            f"failed: {html.escape(e['error'])}</td></tr>")
                continue
            cells = "".join(f"<td title='{e['stats'][m]['valid']:,} valid holes'>{_fmt(e['stats'][m][key])}</td>"
                            if m in e["stats"] else "<td>–</td>" for m in methods)
            body.append(f"<tr><td><a href='#' data-tab='{e['id']}'>{html.escape(e['name'])}</a></td>"
                        f"<td>{e['holes']:,}</td>{cells}</tr>")
        return (f"<h3>{title}</h3><table class='stats wide'><thead><tr><th>Image</th><th>Holes</th>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")

    between = between_image_statistics(entries, methods)
    unit_html = html.escape(unit)
    rows = "".join(
        f"<tr><td><span class='swatch' style='background:{COLORS[m]}'></span>{html.escape(METHOD_LABELS[m])} "
        f"<code>{m}</code></td><td>{b['images']}</td><td>{_fmt(b['mean_of_means'])}</td><td>{_fmt(b['std_of_means'], 3)}</td>"
        f"<td>{_fmt(b['three_sigma'], 3)}</td><td><b>{_fmt(b['three_sigma_pct'], 3)}</b></td></tr>"
        for m, b in between.items())
    summary = (
        "<h3>Between images: do the images show the same mean CD?</h3>"
        "<p class='meta'>Per method, m<sub>i</sub> = mean CD of image i over its validly measured holes, N = number of "
        "images. M = (m<sub>1</sub> + … + m<sub>N</sub>) / N; s = √( Σ (m<sub>i</sub> − M)² / (N − 1) ); "
        "<b>between-image 3σ % = 3 s / M × 100</b>.</p>"
        f"<table class='stats'><thead><tr><th>Method</th><th>Images (N)</th><th>M ({unit_html})</th>"
        f"<th>s ({unit_html})</th><th>3s ({unit_html})</th><th>3s / M (%)</th></tr></thead><tbody>{rows}</tbody></table>"
        + table(f"Mean CD per image, m<sub>i</sub> ({unit_html})", "mean")
        + table("Within-image 3σ (%) = 3 × σ of that image's hole CDs / its mean × 100", "three_sigma_pct"))
    legend = "".join(f"<li><span class='swatch' style='background:{COLORS[m]}'></span><b>{m}</b> — "
                     f"{html.escape(DESCRIPTIONS[m])}</li>" for m in methods)
    tabs = "".join(f"<button class='tab' data-tab='{e['id']}'{' disabled' if e.get('error') else ''}>"
                   f"{html.escape(e['name'])}</button>" for e in entries)
    panels = "".join(
        f"<section class='panel' id='panel-{e['id']}' hidden><h2>{html.escape(e['name'])}: statistics</h2>"
        f"<p class='meta'>{e['holes']:,} holes · {e['width']} × {e['height']} px · {e['seconds']:.1f} s"
        f"{html.escape(e['extra'])}</p>{_stats_table(e['stats'], unit)}{_agreement_table(*e['agreement'])}"
        "<h3>CD histograms</h3><div class='hists'></div></section>"
        for e in entries if not e.get("error"))
    page = (_HTML.replace("__SUMMARY__", summary).replace("__LEGEND__", legend).replace("__TABS__", tabs)
            .replace("__PANELS__", panels)
            .replace("__NOTES__", "".join(f"<p class='note'>{html.escape(n)}</p>" for n in notes))
            .replace("__ENTRIES__", json.dumps({e["id"]: {"name": e["name"], "image": e["image"], "data": e["data"],
                                                          "detail": e.get("detail")}
                                                for e in entries if not e.get("error")}))
            .replace("__RANGES__", json.dumps(ranges or {}))
            .replace("__COLORS__", json.dumps(COLORS)))
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv=None) -> None:
    from PIL import Image
    from scripts.analyze_contact_holes import load_image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder holding the .png images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/cd_report")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--nominal-cd", type=float, default=None, help="Typical hole diameter in --unit (disk, flux_ratio)")
    parser.add_argument("--psf-fwhm", type=float, default=None, help="Known blur FWHM in --unit (overrides --nominal-cd)")
    parser.add_argument("--dark-level", type=float, default=0.0, help="Detector offset removed before flux_ratio divides")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--backend", choices=("cpu", "gpu", "auto"), default="auto")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold",
                        help="peaks: local maxima, for arrays too dense to threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--no-hole-details", action="store_true",
                        help="Skip the per-hole detail data (cells behind the click panel, about 65 MB per 1e5 holes)")
    parser.add_argument("--save-results", action="store_true",
                        help="Also save full per-hole results (compact NPZ + CSVs, ~1 GB per 1e5-hole image)")
    args = parser.parse_args(argv)

    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() == ".png") if args.folder.is_dir() else []
    if not images:
        raise SystemExit(f"No .png images in {args.folder}")
    methods, notes = tuple(args.methods), []
    if args.nominal_cd is None and args.psf_fwhm is None and {"disk", "flux_ratio"} & set(methods):
        notes.append("disk and flux_ratio were skipped: they need --nominal-cd or --psf-fwhm.")
        print(notes[-1], flush=True)
        methods = tuple(m for m in methods if m not in ("disk", "flux_ratio"))
    output = args.output
    for sub in ("images", "data"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    entries, rows, used = [], [], set()
    fixed = {}  # blur and flux_ratio scale from the first calibrated image, reused for all later images
    samples = {}  # per map layer: a sample of every image's CDs, for the shared colour range
    rng = np.random.default_rng(0)
    started = time.perf_counter()
    for number, path in enumerate(images, 1):
        ident = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem) or "image"
        while ident in used:
            ident += "_"
        used.add(ident)
        print(f"[{number}/{len(images)}] {path.name}", flush=True)
        start = time.perf_counter()
        try:
            image = load_image(path)
            result = measure_cds(
                image, methods=methods, seeds=args.seeds, invert=args.invert, area_min=args.area_min,
                area_max=args.area_max, pixel_size=args.pixel_size, unit=args.unit, backend=args.backend,
                nominal_cd=args.nominal_cd, psf_fwhm=fixed.get("psf_fwhm", args.psf_fwhm), dark_level=args.dark_level,
                flux_ratio_scale=fixed.get("flux_ratio_scale"),
            )
        except Exception as error:  # one bad image must not stop the batch
            message = str(error).replace(str(path), path.name) or type(error).__name__
            print(f"  failed: {message}", flush=True)
            entries.append({"id": ident, "name": path.name, "error": message})
            rows.append({"image": path.name, "status": "failed", "error": message})
            continue
        seconds = time.perf_counter() - start
        calibrated_here = args.nominal_cd is not None and not fixed and "psf_fwhm_px" in result.meta
        if calibrated_here:
            fixed = {"psf_fwhm": result.meta["psf_fwhm_px"] * args.pixel_size,
                     "flux_ratio_scale": result.meta.get("flux_ratio_scale"), "image": path.name}
            notes.append(f"disk and flux_ratio were calibrated once, on {path.name} (blur FWHM {fixed['psf_fwhm']:.4g} "
                         f"{args.unit}, flux_ratio scale {_plain(fixed['flux_ratio_scale'] or float('nan'), 4)}), and that "
                         "blur and scale were reused for every image, so their image means are on one scale.")
            print("  " + notes[-1], flush=True)
        stats = cd_statistics(result)
        layers = layer_values(result)
        for layer, values in layers.items():
            values = values[np.isfinite(values)]
            samples.setdefault(layer, []).append(rng.choice(values, min(values.size, 50_000), replace=False))
        Image.fromarray(display_image(image)).save(output / "images" / f"{ident}.png")
        (output / "data" / f"{ident}.js").write_text(
            f"window.CD_DATA[{json.dumps(ident)}]=" + json.dumps(hole_data(result, layers), allow_nan=False) + ";\n",
            encoding="utf-8")
        detail = None
        if not args.no_hole_details:
            detail = f"data/{ident}.detail.js"
            (output / detail).write_text(f"window.CD_DETAIL[{json.dumps(ident)}]="
                                         + json.dumps(detail_data(result, args.dark_level), allow_nan=False) + ";\n",
                                         encoding="utf-8")
        if args.save_results:
            save_cd_results(result, output / "results" / ident, compact=True)
        extra = ""
        if "psf_fwhm_px" in result.meta:
            source = result.meta["psf_source"] if not fixed or calibrated_here else f"calibrated on {fixed['image']}"
            extra = f" · blur FWHM {result.meta['psf_fwhm_px'] * args.pixel_size:.4g} {args.unit} ({source})"
        entries.append({"id": ident, "name": path.name, "holes": len(result.seeds_xy), "stats": stats,
                        "width": image.shape[1], "height": image.shape[0], "seconds": seconds, "extra": extra,
                        "image": f"images/{ident}.png", "data": f"data/{ident}.js", "detail": detail,
                        "agreement": agreement(result)})
        for method, s in stats.items():
            rows.append({"image": path.name, "status": "ok", "method": method, "holes": len(result.seeds_xy), **s,
                         "unit": args.unit})
            print(f"  {method:11s} {s['valid']:7d} valid  mean {_plain(s['mean'], 3)} {args.unit}  "
                  f"3sigma within image {_plain(s['three_sigma_pct'])} %", flush=True)
        print(f"  {seconds:.1f} s", flush=True)
    fields = ["image", "status", "method", "holes", "cells", "valid", "mean", "std", "three_sigma_pct", "noise_sigma",
              "unit", "error"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    between = between_image_statistics(entries, methods)
    with (output / "between_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "images", "mean_of_means", "std_of_means", "three_sigma",
                                                    "three_sigma_pct", "unit"])
        writer.writeheader()
        writer.writerows({"method": m, **b, "unit": args.unit} for m, b in between.items())
    print("Between images (3 s / M x 100, s = std of the per-image means, N - 1):", flush=True)
    for method, b in between.items():
        print(f"  {method:11s} N={b['images']}  M {_plain(b['mean_of_means'], 3)} {args.unit}  "
              f"3s {_plain(b['three_sigma'], 3)} {args.unit}  {_plain(b['three_sigma_pct'], 3)} %", flush=True)
    ranges = {}
    for layer, parts in samples.items():
        pooled = np.concatenate(parts)
        if pooled.size:
            ranges[layer] = [float(v) for v in np.quantile(pooled, [0.02, 0.98])]
    path = write_report(output, entries, methods, notes, args.unit, ranges)
    print(f"Report: {path} ({time.perf_counter() - started:.1f} s)", flush=True)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CD Report</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d2330;--muted:#5d6675;--card:#fff;--line:#d9dde3;--accent:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--fg:#e4e7ec;--muted:#9aa3b2;--card:#1e222a;--line:#343a46;--accent:#6e9ef0}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--fg)}
header{padding:14px 20px 0}h1{font-size:20px;margin:0 0 10px}h2{font-size:17px;margin:4px 0}
nav{display:flex;flex-wrap:wrap;gap:4px;padding:0 20px;border-bottom:1px solid var(--line)}
.tab{border:1px solid var(--line);border-bottom:none;background:transparent;color:var(--fg);padding:7px 12px;border-radius:6px 6px 0 0;cursor:pointer;font:inherit}
.tab.active{background:var(--card);font-weight:600;border-color:var(--accent)}.tab:disabled{opacity:.45;cursor:default}
main{padding:16px 20px}.panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
table.stats{border-collapse:collapse;margin:8px 0}table.stats th,table.stats td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:right;white-space:nowrap}
table.stats th:first-child,table.stats td:first-child{text-align:left}table.wide{display:block;overflow-x:auto;max-width:100%}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}
.meta,.note{color:var(--muted);margin:2px 0 6px}.error{color:#d33;text-align:left!important;white-space:normal!important;max-width:60ch}
h3{font-size:15px;margin:14px 0 2px}code{font-size:12px;color:var(--muted)}
ul.legend{padding-left:18px;color:var(--muted)}ul.legend b{color:var(--fg)}
.hists{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.hist{border:1px solid var(--line);border-radius:6px;padding:4px}
.hist div{font-size:12px;color:var(--muted)}a{color:var(--accent)}
#viewer{display:none}#viewer.on{display:block}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}
.controls select,.controls button,.controls input{font:inherit;padding:3px 8px}
.split{display:grid;grid-template-columns:minmax(0,1fr) 430px;gap:12px;align-items:start}
@media (max-width:1100px){.split{grid-template-columns:1fr}}
#stage{position:relative;height:70vh;min-height:380px;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:#000;cursor:grab}
#stage canvas{width:100%;height:100%;display:block}#bar{display:flex;align-items:center;gap:6px;font-size:12px}
#ramp{width:180px;height:10px;border-radius:2px}
#detail{font-size:13px;max-height:70vh;min-height:380px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:8px}
#detail table{border-collapse:collapse}#detail td{padding:1px 10px 1px 0}
.cells{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.cells figure{margin:0;font-size:12px;color:var(--muted)}
.cells canvas{display:block;image-rendering:pixelated;border:1px solid var(--line)}
canvas.plot{width:100%;display:block;margin:4px 0}
.explain{color:var(--muted);margin:4px 0 8px}
</style></head><body>
<header><h1>CD report</h1></header>
<nav><button class="tab active" data-tab="summary">Summary</button>__TABS__</nav>
<main>
<section class="panel" id="panel-summary"><h2>Summary</h2>
<p class="meta">All statistics use only the holes each method measured validly (hover a per-image value for the count). Standard deviations divide by N − 1. Click an image name to open it.</p>
__NOTES__ __SUMMARY__<h2>Methods</h2><ul class="legend">__LEGEND__</ul></section>
<section class="panel" id="viewer"><h2 id="vtitle"></h2><div class="controls">
<label>Map <select id="method"></select></label>
<label>Colours <select id="scale"><option value="shared">same for all images</option><option value="image">per image</option></select></label>
<button id="fit">Fit image</button><label><input type="checkbox" id="show" checked> Show holes</label>
<span id="bar"><span id="lo"></span><span id="ramp"></span><span id="hi"></span></span>
<span class="meta">Drag to pan · wheel to zoom · click a hole</span></div>
<div class="split"><div id="stage"><canvas id="canvas"></canvas></div><div id="detail"></div></div>
<h3>Band profile: where does banding come from?</h3>
<div class="controls"><label>Bands of <select id="axis"><option value="y">rows (along y)</option><option value="x">columns (along x)</option></select></label>
<label>Band width (px) <input id="bandw" type="number" min="1" step="1" style="width:6em"></label></div>
<p class="explain">Each band averages the holes whose centre falls in it (default width: one hole pitch, i.e. one row of holes per band). The top plot shows every map's band means, each minus its own image mean, so the maps can be compared directly; the table gives each map's band-to-band spread. How to read it:</p>
<ul class="explain"><li>Bands mainly in <b>projection_y</b> (or mainly in projection_x): the light changes from one pixel line to the next (scan or sensor lines), which distorts each hole's width across the lines.</li>
<li>Bands in the Deblended Gaussian but not in disk or projection: the light level leaks into the Gaussian CD (disk and projection ignore a change of light level).</li>
<li>The same bands in disk and in both projection directions: most likely real CD differences on the mask.</li>
<li>Bands that follow the sub-pixel position: the method is tied to the pixel grid.</li></ul>
<p class="explain">Background brightness and spot peak are shown for reference only: bigger holes are also brighter, and the blur spreads part of their light between the holes, so a CD that follows them does not by itself prove a light leak.</p>
<div id="bandinfo"></div>
<canvas class="plot" id="band-cd" style="height:170px"></canvas><canvas class="plot" id="band-br" style="height:100px"></canvas>
</section>
__PANELS__
</main>
<script>
window.CD_DATA={};window.CD_DETAIL={};
const ENTRIES=__ENTRIES__, COLORS=__COLORS__, RANGES=__RANGES__, $=id=>document.getElementById(id);
const HINT="<p class='meta'>Click a hole to see how the selected method measured it.</p>";
const VIRIDIS=[[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[110,206,88],[181,222,43],[253,231,37]];
function viridis(t){t=Math.min(1,Math.max(0,t))*9;const i=Math.min(8,Math.floor(t)),f=t-i,a=VIRIDIS[i],b=VIRIDIS[i+1];
  return `rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(",")})`;}
function unpack(s,T){const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}
function quantile(sorted,q){return sorted.length?sorted[Math.min(sorted.length-1,Math.floor(q*(sorted.length-1)))]:NaN;}
function fin(v){return Number.isFinite(v);}
const canvas=$("canvas"),ctx=canvas.getContext("2d");
let cur=null,curId=null,img=null,view=null,dims=null,sel=-1,bins=null,range=[0,1];
function prepare(d){if(d.ready)return;const ms=Object.values(d.methods);
  for(const m of ms)if(!m.base){m.x=unpack(m.x,Float32Array);m.y=unpack(m.y,Float32Array);m.valid=unpack(m.valid,Uint8Array);
    m.status=unpack(m.status,m.status_wide?Uint16Array:Uint8Array);}
  for(const m of ms){m.cd=unpack(m.cd,Float32Array);if(m.base){const b=d.methods[m.base];
    m.x=b.x;m.y=b.y;m.valid=b.valid;m.status=b.status;m.status_names=b.status_names;}
    m.sorted=Float32Array.from(m.cd.filter(fin)).sort();}
  d.peak=unpack(d.peak,Float32Array);d.row_bg=unpack(d.row_bg,Float32Array);d.col_bg=unpack(d.col_bg,Float32Array);d.ready=true;}
function show(id){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.tab===id));
  document.querySelectorAll(".panel").forEach(p=>p.hidden=p.id!=="panel-"+id&&p.id!=="viewer");
  $("viewer").classList.toggle("on",id!=="summary");if(id==="summary"){cur=null;curId=null;return;}
  const go=()=>{const d=window.CD_DATA[id];prepare(d);cur=d;curId=id;sel=-1;$("vtitle").textContent=ENTRIES[id].name;$("detail").innerHTML=HINT;
    const keep=$("method").value;$("method").innerHTML="";
    for(const [k,m] of Object.entries(d.methods)){const o=document.createElement("option");o.value=k;o.textContent=k+" — "+m.label;$("method").appendChild(o);}
    if(keep&&d.methods[keep])$("method").value=keep;
    if(!$("bandw").value)$("bandw").value=Math.max(1,Math.round(d.pitch||10));
    histograms(id,d);
    img=new Image();img.onload=()=>{
      // Same image size as the previous tab: keep zoom and pan so the same area can be compared.
      if(view&&dims&&dims[0]===d.width&&dims[1]===d.height)sizeCanvas();else fit();
      dims=[d.width,d.height];setMethod();};img.src=ENTRIES[id].image;};
  if(window.CD_DATA[id])go();else{$("detail").textContent="Loading…";const s=document.createElement("script");s.src=ENTRIES[id].data;s.onload=go;
    s.onerror=()=>$("detail").textContent="Could not load "+ENTRIES[id].data;document.body.appendChild(s);}}
function histograms(id,d){const box=document.querySelector(`#panel-${id} .hists`);if(box.dataset.done)return;box.dataset.done=1;
  for(const [k,m] of Object.entries(d.methods)){const v=m.sorted;if(!v.length)continue;
    const lo=quantile(v,.005),hi=quantile(v,.995),n=40,h=new Array(n).fill(0),w=(hi-lo)/n||1;
    for(const x of v){const i=Math.floor((x-lo)/w);if(i>=0&&i<n)h[i]++;}
    const wrap=document.createElement("div");wrap.className="hist";const c=document.createElement("canvas");c.width=240;c.height=110;
    const label=document.createElement("div");label.textContent=`${k} (${d.unit})`;wrap.append(label,c);box.appendChild(wrap);
    const g=c.getContext("2d"),top=Math.max(...h);g.fillStyle=COLORS[k]||"#888";
    h.forEach((y,i)=>g.fillRect(i*6,96-y/top*90,5,y/top*90));g.fillStyle=getComputedStyle(document.body).color;g.font="10px sans-serif";
    g.fillText((lo*d.scale).toPrecision(4),0,108);const t=(hi*d.scale).toPrecision(4);g.fillText(t,240-g.measureText(t).width,108);}}
function setMethod(){if(!cur)return;const k=$("method").value,m=cur.methods[k],v=m.sorted;
  range=$("scale").value==="shared"&&RANGES[k]?RANGES[k].slice():[quantile(v,.02),quantile(v,.98)];
  if(!(range[1]>range[0]))range=[range[0]-0.5,range[0]+0.5];
  // Colour bins: one fillStyle per bin keeps 1e5 holes fast to draw.
  bins=Array.from({length:32},()=>[]);for(let i=0;i<cur.n;i++)if(m.valid[i]&&fin(m.cd[i])){
    const b=Math.min(31,Math.max(0,Math.floor((m.cd[i]-range[0])/(range[1]-range[0])*32)));bins[b].push(i);}
  $("lo").textContent=(range[0]*cur.scale).toPrecision(4);$("hi").textContent=(range[1]*cur.scale).toPrecision(4)+" "+cur.unit;
  $("ramp").style.background="linear-gradient(to right,"+[0,.25,.5,.75,1].map(viridis).join(",")+")";draw();bandPlot();if(sel>=0)renderDetail();}
function sizeCanvas(){const r=canvas.getBoundingClientRect();canvas.width=r.width*devicePixelRatio;canvas.height=r.height*devicePixelRatio;}
function fit(){sizeCanvas();const k=Math.min(canvas.width/cur.width,canvas.height/cur.height);
  view={k,tx:(canvas.width-k*cur.width)/2,ty:(canvas.height-k*cur.height)/2};draw();}
function draw(){if(!cur||!img||!view)return;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle="#000";ctx.fillRect(0,0,canvas.width,canvas.height);
  const {k,tx,ty}=view;ctx.setTransform(k,0,0,k,tx,ty);ctx.imageSmoothingEnabled=k<2;ctx.drawImage(img,0,0);
  if(!$("show").checked||!bins)return;const m=cur.methods[$("method").value];
  const x0=-tx/k-20,y0=-ty/k-20,x1=(canvas.width-tx)/k+20,y1=(canvas.height-ty)/k+20,dots=k*4<6;
  ctx.lineWidth=1.5/k;
  bins.forEach((list,b)=>{const col=viridis((b+.5)/32);ctx.fillStyle=col;ctx.strokeStyle=col;if(!dots)ctx.beginPath();
    for(const i of list){const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
      if(dots){const s=Math.max(1.6/k,Math.min(m.cd[i],3));ctx.fillRect(x-s/2,y-s/2,s,s);}
      else{ctx.moveTo(x+m.cd[i]/2,y);ctx.arc(x,y,m.cd[i]/2,0,2*Math.PI);}}
    if(!dots)ctx.stroke();});
  if(!dots){ctx.strokeStyle="#9a9a9a";ctx.beginPath();const c=3;
    for(let i=0;i<cur.n;i++){if(m.valid[i]&&fin(m.cd[i]))continue;const x=m.x[i]+.5,y=m.y[i]+.5;if(x<x0||x>x1||y<y0||y>y1)continue;
      ctx.moveTo(x-c,y-c);ctx.lineTo(x+c,y+c);ctx.moveTo(x-c,y+c);ctx.lineTo(x+c,y-c);}ctx.stroke();}
  if(sel>=0){const x=m.x[sel]+.5,y=m.y[sel]+.5;ctx.strokeStyle="#ff3b3b";ctx.lineWidth=2.5/k;ctx.beginPath();
    ctx.arc(x,y,Math.max(fin(m.cd[sel])?m.cd[sel]/2+1:4,8/k),0,2*Math.PI);ctx.stroke();}}
function pick(px,py){const m=cur.methods[$("method").value],x=(px-view.tx)/view.k-.5,y=(py-view.ty)/view.k-.5;
  let best=-1,bd=Infinity;for(let i=0;i<cur.n;i++){const d=(m.x[i]-x)**2+(m.y[i]-y)**2;if(d<bd){bd=d;best=i;}}
  if(best<0||bd>36)return;sel=best;renderDetail();draw();}

// ---- small plotting helpers ----
function ticks(a,b,n){const raw=(b-a)/n;if(!(raw>0))return[a];const mag=10**Math.floor(Math.log10(raw)),e=raw/mag;
  const step=(e>=5?10:e>=2?5:e>=1.5?2:1)*mag,out=[];for(let t=Math.ceil(a/step)*step;t<=b+step*1e-6;t+=step)out.push(Math.abs(t)<step*1e-9?0:t);return out;}
function tick(t){return Math.abs(t)>=1e4||(Math.abs(t)<1e-3&&t!==0)?t.toExponential(1):String(+t.toPrecision(4));}
function plot(cv,o){const dpr=devicePixelRatio||1,W=cv.clientWidth||300,H=cv.clientHeight||120;cv.width=W*dpr;cv.height=H*dpr;
  const g=cv.getContext("2d");g.setTransform(dpr,0,0,dpr,0,0);const fg=getComputedStyle(document.body).color,L=50,R=10,T=16,B=o.xl?30:18;
  let [x0,x1]=o.x,[y0,y1]=o.y;if(!(x1>x0)){x0-=1;x1+=1;}if(!(y1>y0)){y0-=1;y1+=1;}const py=(y1-y0)*.08;y0-=py;y1+=py;
  const X=x=>L+(x-x0)/(x1-x0)*(W-L-R),Y=y=>H-B-(y-y0)/(y1-y0)*(H-T-B);
  g.clearRect(0,0,W,H);g.font="10px sans-serif";g.fillStyle=fg;g.strokeStyle=fg;g.lineWidth=1;g.globalAlpha=.35;g.strokeRect(L,T,W-L-R,H-T-B);g.globalAlpha=1;
  for(const t of ticks(x0,x1,6)){const s=tick(t);g.fillText(s,X(t)-g.measureText(s).width/2,H-B+11);}
  for(const t of ticks(y0,y1,3)){const s=tick(t);g.fillText(s,L-4-g.measureText(s).width,Y(t)+3);}
  if(o.title)g.fillText(o.title,L,11);if(o.xl)g.fillText(o.xl,(L+W-R)/2-g.measureText(o.xl).width/2,H-3);
  g.save();g.beginPath();g.rect(L,T,W-L-R,H-T-B);g.clip();
  for(const h of o.h||[]){g.strokeStyle=h.color;g.setLineDash([4,3]);g.beginPath();g.moveTo(L,Y(h.y));g.lineTo(W-R,Y(h.y));g.stroke();}
  for(const v of o.v||[]){g.strokeStyle=v.color;g.setLineDash([4,3]);g.beginPath();g.moveTo(X(v.x),T);g.lineTo(X(v.x),H-B);g.stroke();}
  g.setLineDash([]);
  for(const s of o.series||[]){g.strokeStyle=g.fillStyle=s.color||fg;g.lineWidth=s.width||1.5;
    if(s.dots){for(let i=0;i<s.xs.length;i++){if(!fin(s.ys[i]))continue;g.beginPath();g.arc(X(s.xs[i]),Y(s.ys[i]),s.r||2.5,0,7);g.fill();}}
    else{g.beginPath();let on=false;for(let i=0;i<s.xs.length;i++){if(!fin(s.ys[i])){on=false;continue;}
      if(on)g.lineTo(X(s.xs[i]),Y(s.ys[i]));else g.moveTo(X(s.xs[i]),Y(s.ys[i]));on=true;}g.stroke();}}
  g.restore();
  for(const v of o.v||[])if(v.label){g.fillStyle=v.color;g.fillText(v.label,Math.min(X(v.x)+3,W-R-g.measureText(v.label).width),T+10);}
  for(const h of o.h||[])if(h.label){g.fillStyle=h.color;g.fillText(h.label,W-R-g.measureText(h.label).width-2,Y(h.y)-3);}}
function extent(...arrays){let lo=Infinity,hi=-Infinity;for(const a of arrays)for(const v of a)if(fin(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}return[lo,hi];}
function solve(A,b){const n=b.length;A=A.map(r=>Float64Array.from(r));b=Float64Array.from(b);
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;[A[c],A[p]]=[A[p],A[c]];[b[c],b[p]]=[b[p],b[c]];
    if(Math.abs(A[c][c])<1e-300)return null;for(let r=c+1;r<n;r++){const f=A[r][c]/A[c][c];for(let k=c;k<n;k++)A[r][k]-=f*A[c][k];b[r]-=f*b[c];}}
  const x=new Float64Array(n);for(let r=n-1;r>=0;r--){let s=b[r];for(let k=r+1;k<n;k++)s-=A[r][k]*x[k];x[r]=s/A[r][r];}return x;}
// Cubic spline through y at x = 0..n-1 with scipy's default ("not-a-knot") ends, so the curve is the method's.
function spline(y){const n=y.length;if(n<4)return null;const A=[...Array(n)].map(()=>new Float64Array(n)),b=new Float64Array(n);
  A[0][0]=1;A[0][1]=-2;A[0][2]=1;A[n-1][n-3]=1;A[n-1][n-2]=-2;A[n-1][n-1]=1;
  for(let i=1;i<n-1;i++){A[i][i-1]=1;A[i][i]=4;A[i][i+1]=1;b[i]=6*(y[i+1]-2*y[i]+y[i-1]);}
  const M=solve(A,b);if(!M)return null;
  return x=>{const i=Math.min(n-2,Math.max(0,Math.floor(x))),t=x-i,a=y[i]-M[i]/6,c=y[i+1]-M[i+1]/6;
    return[M[i]*(1-t)**3/6+M[i+1]*t**3/6+a*(1-t)+c*t,-M[i]*(1-t)**2/2+M[i+1]*t*t/2-a+c];};}
function r2(y,cols){const n=y.length,k=cols.length+1;if(n<=k)return NaN;const A=[...Array(k)].map(()=>new Float64Array(k)),b=new Float64Array(k);
  for(let i=0;i<n;i++){const row=[1,...cols.map(c=>c[i])];for(let p=0;p<k;p++){b[p]+=row[p]*y[i];for(let q=0;q<k;q++)A[p][q]+=row[p]*row[q];}}
  const w=solve(A,b);if(!w)return NaN;const mean=y.reduce((s,v)=>s+v,0)/n;let res=0,tot=0;
  for(let i=0;i<n;i++){const f=w[0]+cols.reduce((s,c,j)=>s+w[j+1]*c[i],0);res+=(y[i]-f)**2;tot+=(y[i]-mean)**2;}return tot>0?1-res/tot:NaN;}

// ---- band profile ----
function bands(m,ax,W,size){const pos=ax==="y"?m.y:m.x,nb=Math.ceil(size/W);
  const n=new Float64Array(nb),cd=new Float64Array(nb),pk=new Float64Array(nb),cs=new Float64Array(nb),sn=new Float64Array(nb);
  for(let i=0;i<cur.n;i++){const v=m.cd[i];if(!m.valid[i]||!fin(v))continue;const p=pos[i],b=Math.floor(p/W);if(b<0||b>=nb)continue;
    const f=p-Math.floor(p);n[b]++;cd[b]+=v;pk[b]+=cur.peak[i];cs[b]+=Math.cos(2*Math.PI*f);sn[b]+=Math.sin(2*Math.PI*f);}
  const out={B:[],X:[],CD:[],PK:[],C:[],S:[]};
  for(let b=0;b<nb;b++){if(n[b]<3)continue;out.B.push(b);out.X.push((b+.5)*W);out.CD.push(cd[b]/n[b]*cur.scale);out.PK.push(pk[b]/n[b]);
    out.C.push(cs[b]/n[b]);out.S.push(sn[b]/n[b]);}return out;}
function bandPlot(){if(!cur)return;const k=$("method").value,ax=$("axis").value,W=Math.max(1,Math.round(+$("bandw").value||10));
  const size=ax==="y"?cur.height:cur.width,line=ax==="y"?cur.row_bg:cur.col_bg;
  const bandBg=b=>{let s=0,c=0;for(let j=b*W;j<Math.min((b+1)*W,size);j++){s+=line[j];c++;}return s/c;};
  const pct=v=>fin(v)?(100*v).toFixed(0)+" %":"–",series=[];let rows="",nb=0,chosen=null;
  for(const [kk,mm] of Object.entries(cur.methods)){const r=bands(mm,ax,W,size);if(r.CD.length<4)continue;
    const mean=r.CD.reduce((a,v)=>a+v,0)/r.CD.length,dev=r.CD.map(v=>v-mean),sd=Math.sqrt(dev.reduce((a,v)=>a+v*v,0)/(dev.length-1));
    const BG=r.B.map(bandBg);nb=Math.max(nb,r.CD.length);if(kk===k)chosen=r;
    series.push({xs:r.X,ys:dev,color:COLORS[kk]||"#888",width:kk===k?2.8:1.2});
    rows+=`<tr${kk===k?" style='font-weight:600'":""}><td><span class="swatch" style="background:${COLORS[kk]||"#888"}"></span>${kk}</td><td>${sd.toFixed(3)}</td>`+
      `<td>${pct(r2(r.CD,[BG]))}</td><td>${pct(r2(r.CD,[r.PK]))}</td><td>${pct(r2(r.CD,[r.C,r.S]))}</td></tr>`;}
  const xr=[0,size],xl=ax==="y"?"y (px)":"x (px)";
  plot($("band-cd"),{x:xr,y:extent(...series.map(s=>s.ys)),series,h:[{y:0,color:"#888"}],title:`band mean CD minus that map's own mean (${cur.unit}); bold: the map selected above`});
  const bx=[],by=[];if(chosen)for(const b of chosen.B){bx.push((b+.5)*W);by.push(bandBg(b));}
  plot($("band-br"),{x:xr,y:extent(by),series:[{xs:bx,ys:by,color:"#d08a00",width:2}],
    title:`background brightness per band (25th percentile of each pixel ${ax==="y"?"row":"column"}: the absorber between holes)`,xl});
  $("bandinfo").innerHTML=`<table class="stats"><thead><tr><th>Map</th><th>Band-to-band spread (${cur.unit})</th><th>R² with background</th><th>R² with spot peak</th><th>R² with sub-pixel position</th></tr></thead><tbody>${rows}</tbody></table>`+
    `<p class="meta">Band-to-band spread: standard deviation of the band means. R²: share of that band-to-band variation that goes along with the quantity (0–100 %). With ${nb} bands, chance alone gives about ${pct(1/Math.max(nb-1,1))} (sub-pixel position, fitted with two terms: ${pct(2/Math.max(nb-1,1))}).</p>`;}

// ---- hole detail ----
let detailLoading=null;
function loadDetail(id){if(detailLoading===id)return;detailLoading=id;const s=document.createElement("script");s.src=ENTRIES[id].detail;
  s.onload=()=>{detailLoading=null;if(curId===id&&sel>=0)renderDetail();};
  s.onerror=()=>{detailLoading=null;$("detail").insertAdjacentHTML("beforeend",`<p class='error'>Could not load ${ENTRIES[id].detail}</p>`);};document.body.appendChild(s);}
function prepareDetail(D){if(D.ready)return;D.ox=unpack(D.ox,Int32Array);D.oy=unpack(D.oy,Int32Array);
  for(const part of [D.clean,D.disk])if(part){part.codes=unpack(part.codes,Uint8Array);part.lo=unpack(part.lo,Float32Array);part.step=unpack(part.step,Float32Array);}
  if(D.disk_fit){const F=D.disk_fit;F.amplitude=unpack(F.amplitude,Float32Array);F.background=unpack(F.background,Float32Array);F.table=unpack(F.table,Float32Array);}
  if(D.prof_x){D.prof_x=unpack(D.prof_x,Float32Array);D.prof_y=unpack(D.prof_y,Float32Array);}
  if(D.flux){const F=D.flux;F.flux=unpack(F.flux,Float32Array);F.background=unpack(F.background,Float32Array);F.ratio=unpack(F.ratio,Float32Array);}D.ready=true;}
function cellValues(D,part,i){const n=D.h*D.w,out=new Float32Array(n);for(let p=0;p<n;p++){const c=part.codes[i*n+p];out[p]=c===255?NaN:part.lo[i]+c*part.step[i];}return out;}
function drawCell(id,D,vals,overlay,diverging){const cv=$(id),s=Math.max(6,Math.floor(170/Math.max(D.h,D.w)));cv.width=D.w*s;cv.height=D.h*s;
  const g=cv.getContext("2d");let [lo,hi]=extent(vals);const a=Math.max(Math.abs(lo),Math.abs(hi))||1;
  for(let r=0;r<D.h;r++)for(let c=0;c<D.w;c++){const v=vals[r*D.w+c];let col="#2a3340";
    if(fin(v)){if(diverging){const t=Math.max(-1,Math.min(1,v/a)),u=Math.round(255*(1-Math.abs(t)));col=t>0?`rgb(255,${u},${u})`:`rgb(${u},${u},255)`;}
      else{const u=Math.round(255*(v-lo)/((hi-lo)||1));col=`rgb(${u},${u},${u})`;}}
    g.fillStyle=col;g.fillRect(c*s,r*s,s,s);}
  if(overlay){g.lineWidth=2;overlay(g,s);}}
function outline(cx,cy,rx,ry,color){return(g,s)=>{if(!fin(rx)||!fin(ry))return;g.strokeStyle=color;g.beginPath();
  g.ellipse((cx+.5)*s,(cy+.5)*s,rx*s,ry*s,0,0,2*Math.PI);g.stroke();g.fillStyle=color;g.fillRect((cx+.5)*s-2,(cy+.5)*s-2,4,4);};}
function profilePlots(prefix,prof,edges,label){const sp=spline(prof),xs=[],vs=[],ss=[];const n=prof.length;
  if(sp)for(let x=0;x<=n-1+1e-9;x+=0.02){const [v,s]=sp(x);xs.push(x);vs.push(v);ss.push(s);}
  const idx=[...prof.keys()],w=edges[1]-edges[0],v=[{x:edges[0],color:"#e03b3b",label:"steepest rise"},{x:edges[1],color:"#e03b3b",label:"steepest fall"}];
  plot($(prefix+"-v"),{x:[0,n-1],y:extent(prof,vs),series:[{xs,ys:vs,color:"#7a7a7a"},{xs:idx,ys:prof,dots:true,color:getComputedStyle(document.body).color}],v,
    title:`${label}: width = ${fin(w)?w.toFixed(3)+" px = "+(w*cur.scale).toFixed(2)+" "+cur.unit:"–"}`});
  plot($(prefix+"-s"),{x:[0,n-1],y:extent(ss,[0]),series:[{xs,ys:ss,color:"#3a8fd8"}],v:v.map(e=>({x:e.x,color:e.color})),h:[{y:0,color:"#888"}],
    title:"slope of the curve (edges = its highest and lowest point)",xl:"pixel in the cell window"});}
function diskModel(F,d,rho){const fd=d/F.dd,i=Math.min(F.nd-2,Math.max(0,Math.floor(fd))),u=Math.min(1,Math.max(0,fd-i));
  const fr=rho/F.drho,j=Math.min(F.nr-2,Math.max(0,Math.floor(fr))),v=Math.min(1,Math.max(0,fr-j)),t=F.table,nr=F.nr,k=i*nr+j;
  return(1-u)*((1-v)*t[k]+v*t[k+1])+u*((1-v)*t[k+nr]+v*t[k+nr+1]);}
function renderDetail(){const i=sel;if(i<0||!cur)return;const k=$("method").value,m=cur.methods[k],base=m.base||k;
  let rows="";for(const [kk,mm] of Object.entries(cur.methods)){const v=mm.cd[i];
    rows+=`<tr><td><span class="swatch" style="background:${COLORS[kk]||"#888"}"></span>${kk}</td><td>${fin(v)?(v*cur.scale).toFixed(3)+" "+cur.unit:"–"}</td><td>${mm.status_names[mm.status[i]]}</td></tr>`;}
  let out=`<b>Hole ${i}</b> at x ${m.x[i].toFixed(2)}, y ${m.y[i].toFixed(2)} px<table>${rows}</table>`;
  const D=window.CD_DETAIL[curId];
  if(!ENTRIES[curId].detail){$("detail").innerHTML=out+"<p class='meta'>Hole details were not written (--no-hole-details).</p>";return;}
  if(!D){$("detail").innerHTML=out+"<p class='meta'>Loading hole details…</p>";loadDetail(curId);return;}
  prepareDetail(D);const ox=D.ox[i],oy=D.oy[i],cx=m.x[i]-ox,cy=m.y[i]-oy,cdv=m.cd[i];
  if(base==="projection"){const P=cur.methods.projection,px=cur.methods.projection_x.cd[i],py=cur.methods.projection_y.cd[i];
    const pcx=P.x[i]-ox,pcy=P.y[i]-oy,vals=cellValues(D,D.clean,i);
    const X=Array.from(D.prof_x.subarray(i*D.w,(i+1)*D.w)),Y=Array.from(D.prof_y.subarray(i*D.h,(i+1)*D.h));
    out+=`<h3>projection</h3><p class="explain">The cell with background and neighbours removed. Each column is summed into the x profile, each row into the y profile; a smooth curve goes through the sums, and the width is the distance between its steepest rise and steepest fall (red). Cell outline: dark pixels are outside the cell.</p>
      <div class="cells"><figure><canvas id="dc-a"></canvas>cell and the x/y widths</figure></div>
      <canvas class="plot" id="pp-x-v" style="height:150px"></canvas><canvas class="plot" id="pp-x-s" style="height:90px"></canvas>
      <canvas class="plot" id="pp-y-v" style="height:150px"></canvas><canvas class="plot" id="pp-y-s" style="height:90px"></canvas>`;
    $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(pcx,pcy,px/2,py/2,"#e03b3b"));
    profilePlots("pp-x",X,[pcx-px/2,pcx+px/2],"x profile (column sums)");profilePlots("pp-y",Y,[pcy-py/2,pcy+py/2],"y profile (row sums)");return;}
  if((base==="disk"||base==="flux_ratio")&&D.disk){const vals=cellValues(D,D.disk,i);
    if(base==="flux_ratio"&&D.flux){const F=D.flux,f=F.flux[i],b=F.background[i],r=F.ratio[i];
      out+=`<h3>flux_ratio</h3><p class="explain">The cell the disk fit used (neighbours removed, own background kept). Hole light = sum of every cell pixel above the fitted local background.</p>
        <div class="cells"><figure><canvas id="dc-a"></canvas>cell</figure></div>
        <table><tr><td>hole light (sum above background)</td><td>${f.toPrecision(5)}</td></tr><tr><td>local background B</td><td>${b.toPrecision(5)}</td></tr>
        <tr><td>dark level</td><td>${F.dark}</td></tr><tr><td>ratio = light / (B − dark)</td><td>${fin(r)?r.toPrecision(5):"–"}</td></tr>
        <tr><td>scale</td><td>${F.scale.toPrecision(5)}</td></tr>
        <tr><td>CD = scale × 2 √(ratio / π)</td><td>${fin(cdv)?(cdv).toFixed(3)+" px = "+(cdv*cur.scale).toFixed(2)+" "+cur.unit:"–"}</td></tr></table>`;
      $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(cx,cy,cdv/2,cdv/2,"#8f6ad8"));return;}
    if(base==="disk"&&D.disk_fit){const F=D.disk_fit,A=F.amplitude[i],B=F.background[i],dc=cur.methods.disk,d=dc.cd[i],dcx=dc.x[i]-ox,dcy=dc.y[i]-oy;
      out+=`<h3>disk (blurred-hole fit)</h3><p class="explain">The cell the fit used: neighbours' fitted holes removed, its own background kept. The model is a round hole of diameter d, blurred by the optics and averaged over each pixel, times a brightness A, plus a background B; d, A, B and the centre are fitted. The leftover is cell minus model (red = brighter than the model, blue = darker).</p>`;
      if(!fin(d)){$("detail").innerHTML=out+"<p class='meta'>No valid fit for this hole.</p>";return;}
      const rho=[],val=[],res=new Float32Array(vals.length).fill(NaN);
      for(let r=0;r<D.h;r++)for(let c=0;c<D.w;c++){const v=vals[r*D.w+c];if(!fin(v))continue;const q=Math.hypot(c-dcx,r-dcy);rho.push(q);val.push(v);res[r*D.w+c]=v-(A*diskModel(F,d,q)+B);}
      const xs=[],ms=[],top=Math.max(...rho,d);for(let q=0;q<=top+1e-9;q+=0.05){xs.push(q);ms.push(A*diskModel(F,d,q)+B);}
      out+=`<div class="cells"><figure><canvas id="dc-a"></canvas>cell, fitted hole edge</figure><figure><canvas id="dc-b"></canvas>leftover (cell − model)</figure></div>
        <canvas class="plot" id="dp" style="height:200px"></canvas><p class="meta">d = ${d.toFixed(3)} px = ${(d*cur.scale).toFixed(2)} ${cur.unit} · A = ${A.toPrecision(4)} · B = ${B.toPrecision(4)} · fit leftover RMS = ${Math.sqrt(res.filter(fin).reduce((s,v)=>s+v*v,0)/Math.max(1,rho.length)).toPrecision(3)}</p>`;
      $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(dcx,dcy,d/2,d/2,"#e03b3b"));drawCell("dc-b",D,res,null,true);
      plot($("dp"),{x:[0,top],y:extent(val,ms),series:[{xs,ys:ms,color:"#e03b3b"},{xs:rho,ys:val,dots:true,r:2,color:getComputedStyle(document.body).color}],
        v:[{x:d/2,color:"#e03b3b",label:"hole edge d/2"}],h:[{y:B,color:"#888",label:"background B"}],
        title:"each cell pixel vs its distance from the fitted centre; red: fitted model",xl:"distance from centre (px)"});return;}}
  const vals=cellValues(D,D.clean,i);
  out+=`<h3>${k}</h3><p class="explain">The cell this method measured (background and neighbours removed; dark pixels are outside the cell), with its outline at the CD.</p><div class="cells"><figure><canvas id="dc-a"></canvas>cell and outline</figure></div>`;
  $("detail").innerHTML=out;drawCell("dc-a",D,vals,outline(cx,cy,cdv/2,cdv/2,COLORS[k]||"#e03b3b"));}

let drag=null;
canvas.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY,tx:view.tx,ty:view.ty,moved:false};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=(e.clientX-drag.x)*devicePixelRatio,dy=(e.clientY-drag.y)*devicePixelRatio;
  if(Math.abs(dx)+Math.abs(dy)>4)drag.moved=true;view.tx=drag.tx+dx;view.ty=drag.ty+dy;draw();});
canvas.addEventListener("pointerup",e=>{if(drag&&!drag.moved){const r=canvas.getBoundingClientRect();pick((e.clientX-r.left)*devicePixelRatio,(e.clientY-r.top)*devicePixelRatio);}drag=null;});
canvas.addEventListener("wheel",e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),px=(e.clientX-r.left)*devicePixelRatio,py=(e.clientY-r.top)*devicePixelRatio;
  const f=Math.exp(-e.deltaY*0.0015),k=Math.min(80,Math.max(0.02,view.k*f)),g=k/view.k;view.tx=px-(px-view.tx)*g;view.ty=py-(py-view.ty)*g;view.k=k;draw();},{passive:false});
$("method").onchange=setMethod;$("scale").onchange=setMethod;$("show").onchange=draw;$("fit").onclick=fit;
$("axis").onchange=bandPlot;$("bandw").onchange=bandPlot;
window.addEventListener("resize",()=>{if(cur){sizeCanvas();draw();bandPlot();if(sel>=0)renderDetail();}});
document.querySelectorAll("[data-tab]").forEach(t=>t.addEventListener("click",e=>{e.preventDefault();show(t.dataset.tab);window.scrollTo(0,0);}));
</script></body></html>
"""


if __name__ == "__main__":
    main()
