"""Remote comparison selection, matching, and a small complete CPU run."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import cd_compare as compare
from scripts.benchmark_cd import lattice
from subpx import measure_cds


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_page_selection_spans_runtime_and_honors_explicit_order():
    rows = [{"page": str(page), "status": "ok", "measure_s": str(seconds)}
            for page, seconds in ((0, 9), (1, 2), (2, 4), (3, 1), (4, 6))]
    rows += [{"page": "5", "status": "no_holes", "measure_s": "100"},
             {"page": "6", "status": "ok", "measure_s": "nan"},
             {"page": "7", "status": "ok", "measure_s": ""}]
    assert compare.select_pages(rows, 3) == [3, 2, 0]
    assert compare.select_pages(rows, 20) == [3, 1, 2, 4, 0]
    assert compare.select_pages(rows, 3, "4,1,4,8") == [4, 1, 8]
    # Latest completed timing for a resumed page, with only one selection per page.
    rows += [{"page": "3", "status": "ok", "measure_s": "12"}]
    assert compare.select_pages(rows, 3) == [1, 4, 3]
    for explicit in ("", "-1", "1,", "a,b"):
        with pytest.raises(ValueError, match="--pages"):
            compare.select_pages(rows, 3, explicit)
    with pytest.raises(ValueError, match="No measured pages"):
        compare.select_pages([], 2)
    with pytest.raises(ValueError, match="positive"):
        compare.select_pages(rows, 0)


def test_file_selection_requires_an_exact_key_for_duplicate_basenames(tmp_path):
    path = tmp_path / "timings.csv"
    path.write_text("file,page,status,measure_s\na/scan.tif,1,ok,1\nb/scan.tif,2,ok,2\nother.tif,3,ok,3\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="--file-key"):
        compare.rows_for(path, Path("scan.tif"))
    picked = compare.rows_for(path, Path("scan.tif"), "b/scan.tif")
    assert len(picked) == 1 and picked[0]["page"] == "2"
    assert compare.rows_for(tmp_path / "missing.csv", Path("scan.tif")) == []


def _numbers(seeds, values, valid, *, method="half_area", pixel_size=21.0, unit="nm"):
    return {"seeds_xy": np.asarray(seeds, dtype=float).reshape(-1, 2),
            "pixel_size": np.asarray(pixel_size), "unit": np.asarray(unit),
            f"{method}_cd_equivalent_px": np.asarray(values, dtype=float),
            f"{method}_valid": np.asarray(valid, dtype=bool)}


def test_mutual_matching_does_not_reuse_a_hole_and_keeps_validity_changes_separate():
    a = _numbers([[0, 0], [0.3, 0], [5, 0], [10, 0], [20, 0]],
                 [1, 2, 3, 4, 10], [True, True, True, False, True])
    b = _numbers([[0.1, 0], [5.2, 0], [10.1, 0], [30, 0]],
                 [1.1, 3.2, 4.1, 20], [True, False, True, True], method="local_half_area")
    row, = compare.matched_changes(a, b, {"half_area": "local_half_area"})
    assert (row["matched_positions"], row["unmatched_a"], row["unmatched_b"]) == (3, 2, 1)
    assert (row["valid_both"], row["only_a_valid"], row["only_b_valid"]) == (1, 1, 1)
    assert row["matched_mean_change"] == pytest.approx(2.1)
    assert row["matched_p99_abs_change"] == pytest.approx(2.1)
    assert row["valid_a"] == 4 and row["valid_b"] == 3
    assert row["mean_a"] == pytest.approx(np.mean([1, 2, 3, 10]) * 21)
    assert row["mean_b"] == pytest.approx(np.mean([1.1, 4.1, 20]) * 21)
    assert row["page_mean_change"] == pytest.approx(row["mean_b"] - row["mean_a"])


def test_matching_handles_empty_results_and_uses_each_calibration():
    a = _numbers([[0, 0]], [2], [True], pixel_size=10)
    b = _numbers([[0, 0]], [2], [True], pixel_size=11)
    row, = compare.matched_changes(a, b, {"half_area": "half_area"})
    assert row["matched_mean_change"] == 2
    empty = _numbers([], [], [])
    row, = compare.matched_changes(a, empty, {"half_area": "half_area"})
    assert row["matched_positions"] == 0 and row["unmatched_a"] == 1
    assert row["matched_mean_change"] is None and row["page_mean_change"] is None
    with pytest.raises(ValueError, match="same unit"):
        compare.matched_changes(a, dict(b, unit=np.asarray("px")), {"half_area": "half_area"})


@pytest.fixture()
def prior_run(tmp_path):
    import tifffile

    images = [lattice((64, 96), 10.0, 4.4, 30.0, seed=seed)[0] for seed in (1, 2)]
    images.append(np.full_like(images[0], 40))
    tiff = tmp_path / "scan.tif"
    tifffile.imwrite(tiff, np.stack(images), photometric="minisblack")
    run = tmp_path / "trial"
    run.mkdir()
    settings = {"measure": {"methods": ["projection", "half_area_min_avg"], "seeds": "threshold",
                             "area_min": 3, "area_max": 200, "pixel_size": 21.0, "unit": "nm",
                             "half_area_level": 0.45, "half_area_peak_pixels": 3.0, "tail_area_level": 0.25,
                             "deblend_iterations": 3, "deblend_tolerance": 0.002},
                "hole_check": {"reference": str(tiff), "reference_page": 0, "mode": "raw",
                               "max_error": 0.2, "error_function": None}}
    (run / "run.json").write_text(json.dumps({"git_commit": "prior-commit", "settings": settings}), encoding="utf-8")
    (run / "timings.csv").write_text("file,page,status,measure_s\nlot/scan.tif,0,ok,2\nlot/scan.tif,1,ok,1\n"
                                     "lot/scan.tif,2,no_holes,\n", encoding="utf-8")
    (run / "pages.csv").write_text("file,page,status\nlot/scan.tif,0,ok\nlot/scan.tif,1,ok\n"
                                   "lot/scan.tif,2,no_holes\n", encoding="utf-8")
    return run, tiff, settings


def test_complete_cpu_comparison_reuses_settings_and_writes_small_numeric_results(prior_run, tmp_path, monkeypatch, capsys):
    run, tiff, settings = prior_run
    output = tmp_path / "comparison"
    calls = []

    def measured(image, **options):
        calls.append(options.copy())
        return measure_cds(image, **options)

    # Keep the real signature for the script's filtering of saved settings.
    from functools import wraps
    monkeypatch.setattr(compare, "measure_cds", wraps(measure_cds)(measured))
    assert compare.main([str(run), str(tiff), "--output", str(output), "--backend", "cpu",
                         "--repeats", "1", "--count", "2"]) == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["pages"] == [1, 0]
    assert summary["source_git_commit"] == "prior-commit"
    assert summary["source_run"] == str(run.resolve())
    assert summary["source_hole_check"] == settings["hole_check"]
    assert summary["settings"]["pixel_size"] == 21 and summary["settings"]["unit"] == "nm"
    assert summary["settings"]["half_area_level"] == 0.45
    assert summary["settings"]["half_area_peak_pixels"] == 3
    assert summary["model_methods"] == settings["measure"]["methods"]
    assert summary["local_methods"] == list(compare.DEFAULT_LOCAL_METHODS)
    assert all(call["metrics_only"] is True and call["outlines"] is False for call in calls)
    assert any(call["deblend_iterations"] == 60 and call["deblend_tolerance"] == 1e-4 for call in calls)
    pages = _rows(output / "pages.csv")
    assert [(r["page"], r["pipeline"]) for r in pages] == [("1", "model"), ("1", "local"),
                                                          ("0", "model"), ("0", "local")]
    changes = _rows(output / "changes.csv")
    assert len(changes) == 2 and {row["method_b"] for row in changes} == {"local_half_area_min_avg"}
    stability = _rows(output / "stability.csv")
    assert len(stability) == 2 and {row["page"] for row in stability} == {"0"}
    assert all("page_mean_change" in row and "only_a_valid" in row for row in stability)
    screen = _rows(output / "screen.csv")
    assert {row["page"] for row in screen} == {"0", "1", "2"}
    assert all(float(row["raw_default_threshold"]) == 0.3 for row in screen)
    assert next(row for row in screen if row["page"] == "2")["centered_default_accept"] == "False"
    assert {p.name for p in (output / "holes").iterdir()} == {
        "page_0_model.npz", "page_0_local.npz", "page_0_tighter.npz", "page_1_model.npz", "page_1_local.npz"}
    with np.load(output / "holes" / "page_0_local.npz", allow_pickle=False) as data:
        assert set(data["methods"]) == set(compare.DEFAULT_LOCAL_METHODS)
        assert data["seeds_xy"].shape[1] == 2
        assert data["local_half_area_valid"].dtype == np.bool_
        assert not any("cells" in name or "contours" in name for name in data.files)
    text = capsys.readouterr().out
    assert "Model " in text and "Unsettled " in text and f"Saved: {output}" in text


def test_completed_page_is_saved_before_a_tighter_solve_is_interrupted(prior_run, tmp_path, monkeypatch):
    from functools import wraps

    run, tiff, _ = prior_run
    output = tmp_path / "interrupted"

    @wraps(measure_cds)
    def measured(image, **options):
        if options.get("deblend_iterations") == 60:
            raise KeyboardInterrupt
        return measure_cds(image, **options)

    monkeypatch.setattr(compare, "measure_cds", measured)
    with pytest.raises(KeyboardInterrupt):
        compare.main([str(run), str(tiff), "--output", str(output), "--backend", "cpu",
                      "--repeats", "1", "--pages", "0"])
    assert len(_rows(output / "pages.csv")) == 2
    assert len(_rows(output / "changes.csv")) == 1
    assert (output / "holes" / "page_0_model.npz").is_file()
    assert (output / "holes" / "page_0_local.npz").is_file()
