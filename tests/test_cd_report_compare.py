"""Paired report statistics distinguish holes, image means and validity changes."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cd_report_compare import (
    DEFAULT_PAIRS, aggregate_comparisons, compare_image, match_positions, paired_statistics,
)


def _numbers(seeds, values, valid=None, *, method="half_area", pixel_size=1.0, unit="nm", meta=None):
    values = np.asarray(values, dtype=float)
    return {
        "seeds_xy": np.asarray(seeds, dtype=float).reshape(-1, 2), "pixel_size": pixel_size, "unit": unit,
        f"{method}_valid": np.ones(len(values), bool) if valid is None else np.asarray(valid, dtype=bool),
        f"{method}_cd_equivalent_px": values,
        f"{method}_cd_x_px": values * 2, f"{method}_cd_y_px": values * 3, "meta": meta or {},
    }


def _cd_row(a, b, image_id="one", pair=("half_area", "local_half_area"), **kwargs):
    return next(r for r in compare_image(a, b, image_id=image_id, pairs=[pair], **kwargs) if r["axis"] == "cd")


def test_paired_statistics_known_differences_and_agreement_limits():
    result = paired_statistics([1, 2, 3], [2, 4, 2])
    assert result["n"] == 3
    assert result["mean_reference"] == 2
    assert result["mean_candidate"] == pytest.approx(8 / 3)
    assert result["bias"] == pytest.approx(2 / 3)
    assert result["median_bias"] == 1
    assert result["mae"] == pytest.approx(4 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(2))
    assert result["std_difference"] == pytest.approx(np.sqrt(7 / 3))
    assert result["p95_abs_difference"] == pytest.approx(1.9)
    assert result["correlation"] == pytest.approx(0)
    assert result["loa_low"] == pytest.approx(2 / 3 - 1.96 * np.sqrt(7 / 3))
    assert result["loa_high"] == pytest.approx(2 / 3 + 1.96 * np.sqrt(7 / 3))


def test_statistics_filter_pairs_jointly_and_leave_undefined_values_empty():
    result = paired_statistics([1, np.nan, 3, 4], [2, 4, np.inf, 6])
    assert result["n"] == 2 and result["bias"] == 1.5
    assert result["std_difference"] == pytest.approx(np.sqrt(0.5))
    assert paired_statistics([2, 2], [3, 4])["correlation"] is None
    single = paired_statistics([2], [3])
    assert single["bias"] == 1 and single["std_difference"] is None
    assert single["correlation"] is single["loa_low"] is single["loa_high"] is None
    empty = paired_statistics([], [])
    assert empty["n"] == 0 and all(v is None for k, v in empty.items() if k != "n")
    json.dumps([result, single, empty], allow_nan=False)
    with pytest.raises(ValueError, match="same length"):
        paired_statistics([1, 2], [1])


def test_mutual_nearest_matches_are_one_to_one_bounded_and_index_preserving():
    a = np.array([[0, 0], [0.3, 0], [5, 0], [10, 0], [np.nan, 0]])
    b = np.array([[0.1, 0], [5.2, 0], [11, 0], [30, 0]])
    ia, ib, distances = match_positions(a, b, max_distance=1)
    np.testing.assert_array_equal(ia, [0, 2, 3])
    np.testing.assert_array_equal(ib, [0, 1, 2])
    np.testing.assert_allclose(distances, [0.1, 0.2, 1])
    assert len(set(ia)) == len(ia) and len(set(ib)) == len(ib)
    assert len(match_positions(a, b, max_distance=0)[0]) == 0
    assert len(match_positions(np.empty((0, 2)), b)[0]) == 0
    duplicate_a = np.zeros((3, 2))
    duplicate_b = np.zeros((2, 2))
    ia, ib, _ = match_positions(duplicate_a, duplicate_b)
    assert len(ia) == len(ib) == 1
    for radius in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="max_distance"):
            match_positions(a, b, radius)
    with pytest.raises(ValueError, match="shape"):
        match_positions([0, 0], b)


def test_per_image_coverage_uses_all_valid_holes_and_exposes_changed_populations():
    a = _numbers([[0, 0], [0.3, 0], [5, 0], [10, 0], [20, 0]],
                 [1, 2, 3, np.nan, 10], pixel_size=21)
    b = _numbers([[0.1, 0], [5.2, 0], [10.1, 0], [30, 0]],
                 [1.1, 3.2, 4.1, 20], [True, False, True, True],
                 method="local_half_area", pixel_size=21)
    row = _cd_row(a, b)
    assert (row["matched_positions"], row["unmatched_reference"], row["unmatched_candidate"]) == (3, 2, 1)
    assert (row["n"], row["only_reference_valid"], row["only_candidate_valid"], row["both_invalid"]) == (1, 1, 1, 0)
    assert row["valid_reference"] == 4 and row["valid_candidate"] == 3
    assert row["paired_coverage_reference"] == 0.25
    assert row["paired_coverage_candidate"] == pytest.approx(1 / 3)
    assert row["bias"] == pytest.approx(2.1)
    assert row["mean_all_reference"] == 84
    assert row["mean_all_candidate"] == pytest.approx(176.4)
    assert row["unpaired_mean_difference"] == pytest.approx(92.4)
    assert row["population"] == "paired_holes"


def test_each_calibration_is_applied_to_all_three_axes():
    a = _numbers([[0, 0]], [2], pixel_size=10)
    b = _numbers([[0, 0]], [2], pixel_size=11, method="local_half_area")
    rows = compare_image(a, b, image_id="calibrated")
    assert [(r["axis"], r["bias"]) for r in rows] == [("cd", 2), ("x", 4), ("y", 6)]
    with pytest.raises(ValueError, match="same unit"):
        compare_image(a, dict(b, unit="px"), image_id="bad")
    with pytest.raises(ValueError, match="pixel_size"):
        compare_image(a, dict(b, pixel_size=0), image_id="bad")


def test_result_objects_metadata_and_tail_definition_are_preserved():
    def result(method, value, meta):
        measured = SimpleNamespace(valid=np.array([True]), metrics={
            "cd_equivalent_px": np.array([value]), "cd_x_px": np.array([value]),
            "cd_y_px": np.array([value]),
        })
        return SimpleNamespace(seeds_xy=np.array([[3.0, 4.0]]), pixel_size=21, unit="nm",
                               measurements={method: measured}, meta=meta)

    a = result("tail_area", 3, {"deblend_converged": False, "deblend_iterations_used": 30,
                                 "seconds": {"total": 2.0}})
    b = result("pixel_tail_area", 3.2, {"seconds": {"total": 0.1}})
    rows = compare_image(a, b, image_id="real.png", reference_meta={"measure_s": 2.1})
    assert len(rows) == 3
    for row in rows:
        assert row["different_window"] is True
        assert "wide window" in row["definition_note"] and "native pixels" in row["definition_note"]
        assert row["reference_converged"] is False and row["reference_iterations"] == 30
        assert row["reference_measure_s"] == 2.1 and row["candidate_measure_s"] == 0.1
    assert len(DEFAULT_PAIRS) == 6
    json.dumps(rows, allow_nan=False)


def test_missing_methods_and_empty_inputs_do_not_fabricate_pairs():
    a = _numbers([[0, 0]], [2])
    b = _numbers([], [], method="local_half_area")
    row = _cd_row(a, b)
    assert row["n"] == 0 and row["bias"] is None
    assert row["paired_coverage_reference"] == 0
    assert row["paired_coverage_candidate"] is None
    assert compare_image(a, b, image_id="none", pairs=[("tail_area", "pixel_tail_area")]) == []
    with pytest.raises(ValueError, match="align"):
        _cd_row(a, _numbers([[0, 0]], [1, 2], method="local_half_area"))
    with pytest.raises(ValueError, match="unique"):
        compare_image(a, b, image_id="duplicate", pairs=[("half_area", "local_half_area")] * 2)


def test_aggregate_weights_image_means_equally_instead_of_pooling_holes():
    # One image has two holes with differences 0 and 2. Another has one
    # hundred holes with difference 3. Their equally weighted image bias is 2.
    a1 = _numbers([[0, 0], [3, 0]], [10, 20])
    b1 = _numbers([[0, 0], [3, 0]], [10, 22], method="local_half_area")
    positions = np.column_stack((3 * np.arange(100), np.zeros(100)))
    a2 = _numbers(positions, np.full(100, 30), meta={"deblend_converged": False})
    b2 = _numbers(positions, np.full(100, 33), method="local_half_area")
    rows = [_cd_row(a1, b1, "one"), _cd_row(a2, b2, "two")]
    group, = aggregate_comparisons(rows)
    assert group["population"] == "image_means"
    assert group["n"] == group["images_total"] == 2 and group["paired_holes"] == 102
    assert group["mean_reference"] == 22.5 and group["mean_candidate"] == 24.5
    assert group["bias"] == 2 and group["std_difference"] == pytest.approx(np.sqrt(2))
    assert group["std_reference"] == pytest.approx(np.std([15, 30], ddof=1))
    assert group["std_candidate"] == pytest.approx(np.std([16, 33], ddof=1))
    assert group["three_sigma_reference"] == pytest.approx(3 * np.std([15, 30], ddof=1))
    assert group["three_sigma_pct_candidate"] == pytest.approx(300 * np.std([16, 33], ddof=1) / 24.5)
    delta = np.r_[0, 2, np.full(100, 3)]
    assert group["pooled_hole_bias"] == pytest.approx(delta.mean())
    assert group["pooled_hole_mae"] == pytest.approx(np.abs(delta).mean())
    assert group["pooled_hole_rmse"] == pytest.approx(np.sqrt((delta ** 2).mean()))
    assert group["pooled_hole_std_difference"] == pytest.approx(delta.std(ddof=1))
    assert group["reference_unconverged_images"] == 1
    assert group["reference_convergence_known_images"] == 1
    assert "pooled_hole_p95_abs_difference" not in group
    json.dumps(group, allow_nan=False)


def test_aggregate_keeps_no_pair_images_and_undefined_repeat_spread_visible():
    a = _numbers([[0, 0]], [0])
    b = _numbers([[0, 0]], [0], method="local_half_area")
    good = _cd_row(a, b, "one")
    empty = _cd_row(a, _numbers([], [], method="local_half_area"), "empty")
    group, = aggregate_comparisons([good, empty])
    assert group["images_total"] == 2 and group["n"] == 1
    assert group["bias"] == 0 and group["std_difference"] is None
    assert group["std_reference"] is group["three_sigma_pct_reference"] is None
    assert group["paired_coverage_reference"] == 0.5
    empty_group, = aggregate_comparisons([empty])
    assert empty_group["n"] == 0 and empty_group["paired_holes"] == 0
    assert empty_group["bias"] is None and empty_group["pooled_hole_bias"] is None
    assert aggregate_comparisons([]) == []
    with pytest.raises(ValueError, match="once"):
        aggregate_comparisons([good, good])


def test_aggregate_groups_axes_and_methods_without_mixing_units():
    a = _numbers([[0, 0]], [2])
    b = _numbers([[0, 0]], [3], method="local_half_area")
    rows = compare_image(a, b, image_id="one")
    groups = aggregate_comparisons(rows)
    assert [(g["axis"], g["bias"]) for g in groups] == [("cd", 1), ("x", 2), ("y", 3)]
    more = compare_image(dict(a, unit="px"), dict(b, unit="px"), image_id="one")
    assert len(aggregate_comparisons(rows + more)) == 6
