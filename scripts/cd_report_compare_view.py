"""HTML and a static image for position-matched CD comparisons.

Input rows come from ``cd_report_compare``. This module displays their
statistics without fitting, rescaling, or treating agreement as equivalence.
"""

from __future__ import annotations

from collections import defaultdict
import html
import math
from pathlib import Path


_AXES = (("cd", "CD"), ("x", "x width"), ("y", "y width"))
_FAMILIES = (("half_area", "Half area"),
             ("half_area_min_avg", "Minimum level, averaged peak"),
             ("tail_area", "Tail area"))
_COLORS = {"local_": "#2563a6", "pixel_": "#d96616"}


def _escape(value):
    return html.escape(str(value), quote=True)


def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _number(value):
    return f"{float(value):.5g}" if _finite(value) else "&mdash;"


def _count(value):
    return f"{int(value):,}" if _finite(value) else "&mdash;"


def _coverage(row):
    def percent(key):
        value = row.get(key)
        return f"{100 * float(value):.1f}%" if _finite(value) else "&mdash;"
    return percent("paired_coverage_reference") + " / " + percent("paired_coverage_candidate")


def _convergence(row, aggregate):
    if aggregate:
        total = row.get("images_total", 0)
        known = row.get("reference_convergence_known_images")
        failed = row.get("reference_unconverged_images", 0)
        return f"{_count(failed)} unsettled; {_count(known)} / {_count(total)} known"
    value = row.get("reference_converged")
    label = "unknown" if value is None else ("settled" if value else "unsettled")
    iterations = row.get("reference_iterations")
    return label + (f" ({_count(iterations)} rounds)" if _finite(iterations) else "")


def _table(rows, unit, *, aggregate, max_difference):
    u = _escape(unit)
    headers = ["Candidate / reference", "Axis", "Images" if aggregate else "Paired holes",
               "Paired / valid reference / valid candidate" if aggregate else "Valid reference / candidate",
               "Paired coverage<br>reference / candidate", f"Reference mean ({u})", f"Candidate mean ({u})"]
    if aggregate:
        headers.extend(("Reference 3s / mean (%)", "Candidate 3s / mean (%)"))
    headers.extend((f"Bias ({u})", f"MAE ({u})", f"RMSE ({u})", f"P95 |difference| ({u})", "Correlation",
                    "Reference fit"))
    if not aggregate and max_difference is not None:
        headers.append(f"P95 &le; {_number(max_difference)} {u}")
    body = []
    axis_names = dict(_AXES)
    for row in rows:
        note = _escape(row.get("definition_note", ""))
        pair = (f"<code>{_escape(row.get('method_candidate', ''))}</code><br>"
                f"<small>vs <code>{_escape(row.get('method_reference', ''))}</code></small>")
        count = (_count(row.get("n")) + " / " + _count(row.get("images_total"))
                 if aggregate else _count(row.get("n")))
        valid = _count(row.get("valid_reference")) + " / " + _count(row.get("valid_candidate"))
        if aggregate:
            valid = _count(row.get("paired_holes")) + " / " + valid
        cells = [pair, _escape(axis_names.get(row.get("axis"), row.get("axis", ""))), count,
                 valid, _coverage(row)]
        cells.extend(_number(row.get(key)) for key in ("mean_reference", "mean_candidate"))
        if aggregate:
            enough_images = _finite(row.get("n")) and float(row["n"]) >= 2
            cells.extend(_number(row.get(key) if enough_images else None)
                         for key in ("three_sigma_pct_reference", "three_sigma_pct_candidate"))
        cells.extend(_number(row.get(key)) for key in
                     ("bias", "mae", "rmse", "p95_abs_difference", "correlation"))
        cells.append(_convergence(row, aggregate))
        if not aggregate and max_difference is not None:
            p95 = row.get("p95_abs_difference")
            cells.append("&mdash;" if not _finite(p95) else
                         ("yes" if float(p95) <= max_difference else "no"))
        body.append(f"<tr title='{note}'>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    return ("<div class='cd-compare-scroll'><table class='stats cd-compare-table'><thead><tr>"
            + "".join(f"<th>{head}</th>" for head in headers) + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def comparison_html(rows, aggregates, unit, *, max_difference=None, plot_file=None) -> str:
    """Render paired-hole rows and equally weighted image-mean aggregates.

    ``max_difference`` checks only per-image paired-hole P95, in ``unit``;
    it never labels two methods equivalent or tests aggregate image means.
    Undefined statistics show a dash.
    ``plot_file`` is the report-relative path to an already written plot.
    """
    if max_difference is not None:
        if not _finite(max_difference) or float(max_difference) < 0:
            raise ValueError("max_difference must be finite and nonnegative")
        max_difference = float(max_difference)
    rows, aggregates = list(rows), list(aggregates)
    parts = ["<section class='cd-compare'><style>"
             ".cd-compare-scroll{overflow-x:auto;margin:12px 0}"
             ".cd-compare-table{width:100%;font-size:13px;border-collapse:collapse}"
             ".cd-compare-table th,.cd-compare-table td{padding:7px 9px;vertical-align:top}"
             ".cd-compare-table td:not(:first-child){white-space:nowrap}"
             ".cd-compare-table small{font-size:11px}"
             ".cd-compare details{margin:12px 0}.cd-compare summary{cursor:pointer}"
             ".cd-compare-plot{display:block;width:100%;max-width:1200px;height:auto}"
             ".cd-compare-note{padding:10px 12px;background:#fff5df;color:#4b3410;border-left:3px solid #b7791f}"
             "</style><h2>Comparison with model-based measurements</h2>"]
    parts.append("<p>Differences are candidate minus reference. Each image compares position-matched holes "
                 "that both methods measured. Coverage is the paired count divided by all valid holes of "
                 "each method, including unmatched holes. No fitting or rescaling is applied.</p>")
    if not rows:
        parts.append("<p>No comparison data are available.</p></section>")
        return "".join(parts)
    unsettled = sorted({str(r.get("image_id", "")) for r in rows if r.get("reference_converged") is False})
    if unsettled:
        parts.append(f"<p class='cd-compare-note'>The reference fit did not settle for {len(unsettled)} image(s): "
                     + ", ".join(_escape(name) for name in unsettled)
                     + ". Their comparisons use the values at the stopping iteration.</p>")
    low = sum(_finite(r.get("paired_coverage_reference")) and float(r["paired_coverage_reference"]) < 0.9
              for r in rows)
    if low:
        parts.append(f"<p class='cd-compare-note'>{low} image/method/axis comparison(s) pair fewer than 90% "
                     "of the valid reference holes. Their paired means describe that subset; inspect the "
                     "valid counts and coverage alongside the differences.</p>")
    definitions = list(dict.fromkeys(str(r["definition_note"]) for r in rows if r.get("definition_note")))
    if definitions:
        parts.append("<p><b>Boundary definitions:</b> " + "; ".join(_escape(n) for n in definitions) + ".</p>")
    if not any(_finite(r.get("n")) and float(r["n"]) > 0 for r in rows):
        parts.append("<p>No valid matched holes are available; comparison statistics are undefined.</p>")
    parts.append("<h3>Image means, one equal weight per image</h3><p>Each value entering these statistics is "
                 "an image's mean over its paired holes. Bias, MAE, RMSE, P95 and correlation below compare "
                 "those image means; they are not pooled hole statistics. Images shows usable / compared images; "
                 "failed reference calculations are listed separately and excluded from these aggregates. "
                 "Reference and candidate 3s / mean use the same paired image means: 100 times three times "
                 "their sample standard deviation divided by their mean. They require at least two images "
                 "and a nonzero mean. Hole counts and coverage are totals across images.</p>")
    parts.append(_table(aggregates, unit, aggregate=True, max_difference=max_difference) if aggregates
                 else "<p>No image-mean aggregates are available.</p>")
    if plot_file is not None:
        parts.append(f"<figure><img class='cd-compare-plot' src='{_escape(plot_file)}' "
                     "alt='Candidate versus reference paired image means, with identity lines'>"
                     "<figcaption>One point per image and candidate method, using that pair's shared valid holes. "
                     "The dashed line marks equal values. Blue: local 4x; orange: native pixel.</figcaption></figure>")
    by_image = defaultdict(list)
    for row in rows:
        by_image[str(row.get("image_id", ""))].append(row)
    parts.append("<h3>Per-image paired-hole differences</h3><p>Within each image, the statistics below compare "
                 "individual paired holes. A dash means undefined, including correlation for constant values.</p>")
    if max_difference is not None:
        parts.append(f"<p>The optional P95 &le; {_number(max_difference)} {_escape(unit)} column checks only "
                     "paired-hole differences within each image, not aggregate image-mean differences. "
                     "It does not establish measurement equivalence or repeatability.</p>")
    for name, group in by_image.items():
        parts.append(f"<details><summary>{_escape(name)} &mdash; {len(group)} method/axis comparisons</summary>"
                     + _table(group, unit, aggregate=False, max_difference=max_difference) + "</details>")
    parts.append("</section>")
    return "".join(parts)


def write_comparison_plot(output_path, rows, unit) -> bool:
    """Write a PNG of matched per-image means; return False if none are usable.

    Three family rows and three axis columns use identity lines and equal
    axis scales. No trend line, calibration, or weighting is applied.
    """
    rows = [r for r in rows if r.get("method_reference") in dict(_FAMILIES)
            and r.get("axis") in dict(_AXES) and _finite(r.get("n")) and float(r["n"]) > 0
            and _finite(r.get("mean_reference")) and _finite(r.get("mean_candidate"))
            and str(r.get("method_candidate", "")).startswith(tuple(_COLORS))]
    if not rows:
        return False
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 3, squeeze=False)
    for i, (family, family_label) in enumerate(_FAMILIES):
        for j, (axis, axis_label) in enumerate(_AXES):
            panel = axes[i, j]
            selected = [r for r in rows if r["method_reference"] == family and r["axis"] == axis]
            panel.set_title(f"{family_label}\n{axis_label}", fontsize=10)
            panel.set_xlabel(f"Reference mean ({unit})", fontsize=9)
            panel.set_ylabel(f"Candidate mean ({unit})", fontsize=9)
            if not selected:
                panel.text(0.5, 0.5, "No valid pairs", ha="center", va="center", transform=panel.transAxes)
                continue
            values = [float(r[key]) for r in selected for key in ("mean_reference", "mean_candidate")]
            low, high = min(values), max(values)
            padding = max((high - low) * 0.08, max(abs(low), abs(high)) * 0.01, 1e-6)
            low, high = low - padding, high + padding
            panel.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1, zorder=1)
            for prefix, color in _COLORS.items():
                group = [r for r in selected if r["method_candidate"].startswith(prefix)]
                if group:
                    panel.scatter([r["mean_reference"] for r in group], [r["mean_candidate"] for r in group],
                                  color=color, marker="o" if prefix == "local_" else "x", s=28, alpha=0.8,
                                  label="Local 4x" if prefix == "local_" else "Native pixel", zorder=2)
            panel.set_xlim(low, high)
            panel.set_ylim(low, high)
            panel.set_aspect("equal", adjustable="box")
            panel.grid(alpha=0.2)
            panel.legend(fontsize=8, loc="best")
            panel.tick_params(labelsize=8)
    figure.suptitle("Paired image means: candidate versus model-based reference", fontsize=14)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=150)
    return True
