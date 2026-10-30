"""Plot and export pattern pitch analysis; optional plotting imports stay lazy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .pattern import _neighbor_pairs, analyze_pitch
from .types import PatternPitchResult


def plot_pitch_report(image, result: PatternPitchResult):
    """Plot the selected bands, averaged x/y pitch, and registration error.

    Return a Matplotlib Figure. The input image and centers are in pixels;
    all profile lengths are in ``result.unit``. Orange denotes the x profile's
    selected rows; blue denotes the y profile's selected columns.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=(1.4, 1, 1))
    image_ax = fig.add_subplot(grid[0, :])
    image_ax.imshow(image, cmap="gray", interpolation="nearest")
    centers = result.centers_xy
    unindexed = (result.indices_rc < 0).any(axis=1) & np.isfinite(centers).all(axis=1)
    if np.any(unindexed):
        image_ax.scatter(*centers[unindexed].T, s=12, c="red", marker="x")
    for axis, color in (("x", "#c15c00"), ("y", "#1469b8")):
        profile = result.profiles[axis]
        selected = profile.selected_indices
        points = centers[selected]
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            margin = result.lattice_pitch_xy * 0.35
            low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
            image_ax.add_patch(Rectangle(low, *(high - low), facecolor=color, edgecolor=color,
                                        alpha=0.18, linewidth=1.5))
        image_ax.add_collection(LineCollection(centers[profile.pairs], colors=color, linewidths=0.5, alpha=0.7))
        good = selected[result.valid[selected]]
        image_ax.scatter(*centers[good].T, s=3, c=color)
        middle = (profile.position[:-1] + profile.position[1:]) / 2
        axes = [fig.add_subplot(grid[row, 0 if axis == "x" else 1]) for row in (1, 2)]
        axes[0].plot(middle, profile.pitch, ".-", color=color, markersize=3)
        axes[0].axhline(profile.nominal_pitch, color="gray", linestyle="--", linewidth=0.8)
        axes[1].plot(profile.position, profile.registration, ".-", color=color, markersize=3)
        axes[1].axhline(0, color="gray", linewidth=0.8)
        for ax, title, ylabel in zip(axes, ("Averaged pitch", "Registration error"),
                                     ("pitch", "sum(pitch - nominal)")):
            ax.set(title=f"{axis}: {title}", xlabel=f"{axis} position ({result.unit})",
                   ylabel=f"{ylabel} ({result.unit})")
            ax.grid(alpha=0.2)
    reference = centers[result.reference_index]
    if result.valid[result.reference_index]:
        image_ax.scatter(*reference, s=90, facecolors="none", edgecolors="#9828b9", linewidths=2)
    image_ax.set(title=f"{result.method} | {result.layout} | x: {result.meta['average_rows']} rows (orange), "
                       f"y: {result.meta['average_columns']} columns (blue) | purple: reference",
                 xlabel="x (px)", ylabel="y (px)")
    fig.suptitle("Pattern pitch and registration", fontsize=16)
    return fig


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {key: _json_value(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_value(row))


def save_pitch_report(result, output, *, title="Pattern pitch and registration", **analysis_options) -> Path:
    """Save an offline interactive report, a PNG, and reproducible CSV/JSON data.

    ``result`` is a full ``CDResult`` (its image is displayed). Keyword options
    are forwarded to ``analyze_pitch``. ``output`` must be new or empty. Browser
    controls re-average already measured centers immediately, with no refitting.
    CSVs/PNG describe the initial selection; the viewer exports the current CSV.
    """
    from PIL import Image
    import matplotlib.pyplot as plt

    from ._pitch_viewer import HTML
    from .cd import METHOD_LABELS

    found = analyze_pitch(result, **analysis_options)
    if not found:
        raise ValueError("The CD result contains no measurement methods")
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty folder")
    output.mkdir(parents=True, exist_ok=True)
    first = next(iter(found.values()))
    fields = ("method", "axis", "node", "position", "pitch_to_next", "sample_std", "pairs", "distance",
              "registration", "nominal_pitch", "unit", "reference_hole", "average_rows", "average_columns")

    def profile_rows():
        for name, measured in found.items():
            for axis, p in measured.profiles.items():
                for i in range(len(p.indices)):
                    row = dict(method=name, axis=axis, node=i, position=p.position[i], distance=p.distance[i],
                               registration=p.registration[i], nominal_pitch=p.nominal_pitch, unit=result.unit,
                               reference_hole=measured.reference_index, average_rows=measured.meta["average_rows"],
                               average_columns=measured.meta["average_columns"])
                    if i < len(p.pitch):
                        row.update(pitch_to_next=p.pitch[i], sample_std=p.std[i], pairs=p.counts[i])
                    yield row

    _write_csv(output / "profiles.csv", profile_rows(), fields)
    center_fields = ("method", "hole", "row", "column", "seed_x_px", "seed_y_px", "center_x_px", "center_y_px", "valid")
    _write_csv(output / "centers.csv", (
        dict(zip(center_fields, (name, i, *m.indices_rc[i], *result.seeds_xy[i], *m.centers_xy[i], bool(m.valid[i]))))
        for name, m in found.items() for i in range(len(result.seeds_xy))
    ), center_fields)
    summary = []
    for name, m in found.items():
        for axis, p in m.profiles.items():
            values = (m.centers_xy[p.pairs[:, 1], 0 if axis == "x" else 1]
                      - m.centers_xy[p.pairs[:, 0], 0 if axis == "x" else 1]) * result.pixel_size
            summary.append(dict(method=name, axis=axis, valid_centers=int(m.valid.sum()), pairs=len(values),
                                measured_mean=float(values.mean()) if len(values) else np.nan,
                                nominal_pitch=p.nominal_pitch, missing_intervals=int((p.counts == 0).sum()),
                                unit=result.unit, layout=m.layout, stagger_axis=m.stagger_axis))
    _write_csv(output / "summary.csv", summary, tuple(summary[0]))
    height, width = result.image.shape
    indexed = np.flatnonzero((first.indices_rc >= 0).all(axis=1))
    first_reference = indexed[np.lexsort((first.indices_rc[indexed, 1], first.indices_rc[indexed, 0]))[0]]
    data = dict(schema_version=1, title=title, width=width, height=height, seeds=result.seeds_xy,
                indices=first.indices_rc, shape=first.indices_rc.max(axis=0) + 1,
                layout=first.layout, stagger_axis=first.stagger_axis, pitch_xy=first.lattice_pitch_xy,
                reference=first.reference_index,
                first_reference=int(first_reference),
                pixel_size=result.pixel_size, unit=result.unit,
                average_rows=first.meta["average_rows"], average_columns=first.meta["average_columns"],
                nominal=analysis_options.get("nominal_pitch_xy"), settings=analysis_options,
                resolved_settings=first.meta, cd_meta=result.meta,
                pairs={axis: _neighbor_pairs(first.indices_rc, axis) for axis in ("x", "y")},
                methods={name: dict(label=METHOD_LABELS.get(name, name), centers=m.centers_xy, valid=m.valid)
                         for name, m in found.items()})
    raw = json.dumps(_json_value(data), allow_nan=False, separators=(",", ":"))
    (output / "data.json").write_text(raw + "\n", encoding="utf-8")
    (output / "data.js").write_text("window.PITCH_DATA=" + raw + ";\n", encoding="utf-8")
    image = result.image.astype(float)
    lo, hi = np.percentile(image, [0.5, 99.5])
    preview = np.clip((image - lo) * (255 / max(hi - lo, 1e-12)), 0, 255).astype(np.uint8)
    Image.fromarray(preview).save(output / "image.png")
    fig = plot_pitch_report(result.image, first)
    try:
        fig.savefig(output / "report.png", dpi=140)
    finally:
        plt.close(fig)
    (output / "index.html").write_text(HTML, encoding="utf-8")
    return output / "index.html"
