"""The throughput trial runs the batch with saved settings, not a substitute loop."""

import argparse
import csv
import json

import numpy as np
import pytest

from scripts import cd_batch, cd_speed
from scripts.benchmark_cd import lattice


def test_window_is_contiguous_and_contains_the_most_prior_accepted_pages():
    rows = [{"page": str(page), "status": "ok" if page in (1, 8, 9) else "no_holes"} for page in range(12)]
    assert cd_speed.page_window(rows, 4) == (6, 10)
    assert cd_speed.page_window(rows) == (0, 12)
    many = [{"page": str(page), "status": "ok"} for page in range(600)]
    assert cd_speed.page_window(many) == (0, 512)
    assert cd_speed.page_window([], explicit="100:228") == (100, 228)
    for text in ("-1:2", "2:2", "3:2", ":20", "20:", "1", "a:b", "1:2:3"):
        with pytest.raises(ValueError, match="--pages"):
            cd_speed.page_window(rows, explicit=text)
    with pytest.raises(ValueError, match="--count"):
        cd_speed.page_window(rows, 0)
    with pytest.raises(ValueError, match="No prior accepted"):
        cd_speed.page_window([{"page": "0", "status": "no_holes"}])


def test_old_run_preserves_raw_screen_and_numerical_settings(tmp_path):
    args = argparse.Namespace(tiff=tmp_path / "input.tif", backend="gpu", workers_per_gpu=1, workers=1,
                              methods=cd_speed.PIXEL_METHODS, gpus=["1", "3"])
    measure = {"methods": ["gaussian"], "invert": True, "half_area_level": 0.45, "half_area_peak_pixels": 3,
               "tail_area_level": 0.25, "pixel_size": 21, "unit": "nm", "seeds": "peaks",
               "n_angles": 16, "outlines": False, "metrics_only": True, "summary_only": True}
    check = {"reference": str(tmp_path / "ref.tif"), "reference_page": 12, "max_error": 0.27,
             "error_function": "D:/custom/screen.py:classify"}
    settings = {"measure": measure, "hole_check": check}
    command = cd_speed.batch_arguments(args, settings, tmp_path / "out", (10, 138))
    parsed = cd_batch.argument_parser().parse_args(command)
    assert parsed.hole_check_mode == "raw" and parsed.max_error == 0.27
    assert parsed.reference_page == 12 and parsed.error_function == check["error_function"]
    assert parsed.methods == list(cd_speed.PIXEL_METHODS) and parsed.gpus == ["1", "3"]
    assert parsed.pixel_size == 21 and parsed.unit == "nm" and parsed.seeds == "peaks" and parsed.invert
    assert parsed.half_level == 0.45 and parsed.peak_pixels == 3 and parsed.tail_level == 0.25
    assert parsed.pages == "10:138" and parsed.workers_per_gpu == 1
    settings["hole_check"] = dict(check, mode="centered", max_error=0.7)
    parsed = cd_batch.argument_parser().parse_args(cd_speed.batch_arguments(args, settings, tmp_path, (0, 1)))
    assert parsed.hole_check_mode == "centered" and parsed.max_error == 0.7
    settings["hole_check"] = None
    parsed = cd_batch.argument_parser().parse_args(cd_speed.batch_arguments(args, settings, tmp_path, (0, 1)))
    assert parsed.no_hole_check
    measure["unsupported_measurement_option"] = 1
    with pytest.raises(ValueError, match="Cannot reproduce"):
        cd_speed.batch_arguments(args, settings, tmp_path, (0, 1))


def test_projection_uses_completed_pages_and_is_withheld_for_failures_or_interrupts(tmp_path):
    (tmp_path / "pages.csv").write_text("file,page,status\na.tif,0,ok\na.tif,1,no_holes\n", encoding="utf-8")
    invocation = {"complete": True, "counts": {"ok": 1, "no_holes": 1, "failed": 0}}
    result = cd_speed.result_summary(tmp_path, 4.0, invocation)
    assert result["pages_per_s"] == 0.5 and result["accepted_pages"] == 1 and result["rejected_pages"] == 1
    assert result["seconds_per_1500_pages"] == 3000
    assert result["hours_per_1000_files_of_1500_pages"] == pytest.approx(3000000 / 3600)
    invocation["counts"]["failed"] = 1
    assert cd_speed.result_summary(tmp_path, 4.0, invocation)["seconds_per_1500_pages"] is None
    invocation.update(complete=False, counts={"failed": 0})
    assert cd_speed.result_summary(tmp_path, 4.0, invocation)["seconds_per_1500_pages"] is None


def test_complete_cpu_trial_uses_real_batch_and_saves_end_to_end_timing(tmp_path, capsys):
    import tifffile

    images = [lattice((64, 96), 10.0, 4.4, 30.0, seed=seed)[0] for seed in (1, 2)]
    images.append(np.full_like(images[0], 40))
    tiff = tmp_path / "scan.tif"
    tifffile.imwrite(tiff, np.stack(images), photometric="minisblack")
    run = tmp_path / "old"
    run.mkdir()
    settings = {"measure": {"methods": ["gaussian"], "area_min": 3, "area_max": 200,
                             "pixel_size": 21.0, "unit": "nm", "half_area_level": 0.45,
                             "half_area_peak_pixels": 3.0, "tail_area_level": 0.25},
                "hole_check": {"reference": str(tiff), "reference_page": 0, "max_error": 0.3,
                               "error_function": None}}
    (run / "run.json").write_text(json.dumps({"settings": settings, "git_commit": "prior"}), encoding="utf-8")
    (run / "pages.csv").write_text("file,page,status\nscan.tif,0,ok\nscan.tif,1,ok\nscan.tif,2,no_holes\n",
                                    encoding="utf-8")
    output = tmp_path / "speed"
    command = [str(run), str(tiff), "--output", str(output), "--backend", "cpu"]
    assert cd_speed.main(command) == 0
    result = json.loads((output / "speed.json").read_text(encoding="utf-8"))
    current = json.loads((output / "run.json").read_text(encoding="utf-8"))
    invocation = json.loads((output / "invocations.jsonl").read_text(encoding="utf-8"))
    assert result["complete"] and result["completed_pages"] == 3
    assert result["accepted_pages"] == 2 and result["rejected_pages"] == 1 and result["failed_pages"] == 0
    assert result["pages"] == [0, 3] and result["source_git_commit"] == "prior"
    assert result["elapsed_s"] >= invocation["elapsed_s"] > 0
    assert result["pages_per_s"] == 3 / result["elapsed_s"]
    assert current["settings"]["hole_check"]["mode"] == "raw"
    measured = current["settings"]["measure"]
    assert measured["methods"] == list(cd_speed.PIXEL_METHODS)
    assert measured["half_area_level"] == 0.45 and measured["tail_area_level"] == 0.25
    with (output / "pages.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["ok", "ok", "no_holes"]
    assert all(not row["deblend_iterations"] or int(row["deblend_iterations"]) == 0 for row in rows)
    text = capsys.readouterr().out
    assert "pages/s" in text and "1,000 such TIFFs" in text and "cache state unknown" in text
    with pytest.raises(SystemExit):
        cd_speed.main(command)
