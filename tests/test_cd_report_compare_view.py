"""Comparison rendering preserves statistical populations and handles missing data."""

import numpy as np
import pytest

from scripts.cd_report_compare_view import comparison_html, write_comparison_plot


def _row(**changes):
    row = dict(image_id="scan<script>.png", method_reference="half_area", method_candidate="local_half_area",
               axis="cd", n=8, valid_reference=10, valid_candidate=8, paired_coverage_reference=0.8,
               paired_coverage_candidate=1.0, mean_reference=4.0, mean_candidate=4.2,
               bias=0.2, mae=0.2, rmse=0.25, p95_abs_difference=0.3, correlation=None,
               reference_converged=False, reference_iterations=30,
               definition_note="raw cell < model & neighbour", unit="px")
    return {**row, **changes}


def test_comparison_html_escapes_text_and_distinguishes_populations():
    row = _row(correlation=np.nan, rmse=np.inf)
    aggregate = {**row, "n": 1, "images_total": 1, "paired_holes": 8,
                 "reference_unconverged_images": 1, "reference_convergence_known_images": 1}
    output = comparison_html([row], [aggregate], "px<", max_difference=0.4,
                             plot_file="plot' onload='bad.png")
    assert "scan&lt;script&gt;.png" in output and "<script>" not in output
    assert "raw cell &lt; model &amp; neighbour" in output
    assert "src='plot&#x27; onload=&#x27;bad.png'" in output
    assert "px&lt;" in output and "&mdash;" in output
    assert ">nan<" not in output and ">inf<" not in output
    assert "one equal weight per image" in output and "not pooled hole statistics" in output
    assert "reference fit did not settle" in output and "fewer than 90%" in output
    assert "does not establish measurement equivalence" in output
    assert "<details>" in output and "<td>yes</td>" in output


def test_comparison_view_empty_and_undefined_data(tmp_path):
    assert "No comparison data" in comparison_html([], [], "px")
    row = _row(n=0, mean_reference=None, mean_candidate=np.nan)
    assert "No valid matched holes" in comparison_html([row], [], "px")
    output = tmp_path / "empty.png"
    assert write_comparison_plot(output, [row], "px") is False
    assert not output.exists()
    with pytest.raises(ValueError, match="max_difference"):
        comparison_html([], [], "px", max_difference=np.nan)


def test_aggregate_shows_paired_image_spread_but_threshold_tests_only_holes():
    row = _row(p95_abs_difference=0.6)
    aggregate = {**row, "n": 2, "images_total": 2, "paired_holes": 16,
                 "p95_abs_difference": 0.01, "three_sigma_pct_reference": 9.1234,
                 "three_sigma_pct_candidate": 4.5678}
    output = comparison_html([row], [aggregate], "px", max_difference=0.4)
    aggregate_html, per_image_html = output.split("<h3>Per-image paired-hole differences</h3>")
    assert "Reference 3s / mean (%)" in aggregate_html
    assert "Candidate 3s / mean (%)" in aggregate_html
    assert "<td>9.1234</td>" in aggregate_html and "<td>4.5678</td>" in aggregate_html
    assert "at least two images" in aggregate_html
    assert "P95 &le;" not in aggregate_html
    assert "<td>yes</td>" not in output and "<td>no</td>" in per_image_html
    assert "color:#4b3410" in output
    aggregate["n"] = 1
    single_image = comparison_html([row], [aggregate], "px")
    assert "<td>9.1234</td>" not in single_image and "<td>4.5678</td>" not in single_image


def test_comparison_plot_writes_png_without_pyplot(tmp_path):
    from PIL import Image

    path = tmp_path / "figures" / "comparison.png"
    rows = [_row(), _row(image_id="b.png", method_candidate="pixel_half_area", mean_candidate=4.5),
            _row(axis="x", mean_candidate=4.1)]
    assert write_comparison_plot(path, rows, "px") is True
    with Image.open(path) as image:
        assert image.format == "PNG" and image.size == (1800, 1650)
