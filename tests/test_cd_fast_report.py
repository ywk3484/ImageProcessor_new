"""The full PNG report keeps the original presentation and all measurement methods."""

import base64
import csv
import json

import numpy as np
from PIL import Image
import pytest

from scripts import cd_fast_report as fast_report
from scripts import cd_report as report
from scripts.benchmark_cd import lattice
from subpx import load_cd_results, measure_cds


ORIGINAL = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
            "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
FAST = ("local_projection", "local_half_area", "local_half_area_min_avg", "local_tail_area",
        "pixel_half_area", "pixel_half_area_min_avg", "pixel_tail_area")
METHODS = ORIGINAL + FAST


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _js(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text.split("]=", 1)[1].rstrip().rstrip(";"))


def _array(value, dtype):
    return np.frombuffer(base64.b64decode(value), dtype)


def test_report_keeps_original_statistics_and_adds_every_fast_method(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    images = {}
    for name, fwhm in (("a scan.png", 4.0), ("a_scan.PNG", 4.4)):
        images[name] = lattice((60, 80), 10, fwhm, 25)[0]
        Image.fromarray(images[name]).save(folder / name)
    (folder / "broken.png").write_bytes(b"not a PNG")
    (folder / "ignore.txt").write_text("unrelated")
    nested = folder / "subfolder"
    nested.mkdir()
    Image.fromarray(images["a scan.png"]).save(nested / "not_selected.png")
    output = tmp_path / "report"
    command = [str(folder), "--output", str(output), "--backend", "cpu", "--pixel-size", "21", "--unit", "nm"]
    assert fast_report.main(command) == 1
    summary = _rows(output / "summary.csv")
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["successful_images"] == 2 and manifest["failed_images"] == 1
    assert {r["image"] for r in summary} == {*images, "broken.png"}
    for name, image in images.items():
        original = measure_cds(image, methods=ORIGINAL, area_max=200, pixel_size=21, unit="nm", backend="cpu")
        full = measure_cds(image, methods=METHODS, area_max=200, pixel_size=21, unit="nm", backend="cpu")
        records = {r["method"]: r for r in summary if r["image"] == name}
        source = next(r for r in manifest["images"] if r["file"] == name)
        assert set(records) == set(METHODS) | {"cell_intensity", "image_intensity"}
        with np.load(output / "arrays" / f"{source['id']}.npz", allow_pickle=False) as arrays:
            assert arrays["methods"].tolist() == list(METHODS)
            for method in METHODS:
                measured = original.measurements[method] if method in ORIGINAL else full.measurements[method]
                np.testing.assert_array_equal(arrays[f"{method}_valid"], measured.valid)
                for suffix, metric, _ in report.AXES:
                    values = measured.metrics[metric][measured.valid] * 21
                    values = values[np.isfinite(values)]
                    assert int(records[method]["valid" + suffix]) == len(values)
                    assert float(records[method]["mean" + suffix]) == pytest.approx(values.mean(), nan_ok=True)
                    np.testing.assert_allclose(arrays[f"{method}_{metric}"], measured.metrics[metric], equal_nan=True)
        # Brightness keeps its original meaning, rather than switching to a raw-cell tail.
        light = report.cell_intensity(original)
        assert float(records["cell_intensity"]["mean"]) == pytest.approx(np.nanmean(light))
        data = _js(output / "data" / f"{source['id']}.js")
        assert set(data["methods"]) == {m + suffix for m in METHODS for suffix in ("", "_x", "_y")} | {"cell_intensity"}
        assert data["methods"]["cell_intensity"]["base"] == "tail_area"
        assert data["peak_label"] == "fitted spot amplitude"
        model = original.deblend_model
        np.testing.assert_allclose(_array(data["peak"], np.float32), model["params"][model["report"], 0], rtol=1e-6)
        for method in METHODS:
            layer = data["methods"][method]
            assert layer["outline_kind"] == "measured"
            rays = _js(output / layer["rays_file"])
            assert rays["n"] == 32
        assert (output / "data" / f"{source['id']}.detail.js").is_file()
        assert (output / "data" / f"{source['id']}.tail.js").is_file()
    between = {r["method"]: r for r in _rows(output / "between_images.csv")}
    assert set(between) == set(METHODS) | {"cell_intensity", "image_intensity"}
    for method in METHODS:
        means = np.array([float(r["mean"]) for r in summary if r.get("method") == method])
        means = means[np.isfinite(means)]
        assert int(between[method]["images"]) == len(means)
        assert float(between[method]["mean_of_means"]) == pytest.approx(means.mean(), nan_ok=True)
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "<h3>Between images: do the images show the same mean CD?</h3>" in page
    assert "Band profile: where does banding come from?" in page and "CD histograms" in page
    assert "Do the methods agree on which holes are big?" in page  # existed in the original report
    assert "broken.png" in page
    assert not list(output.glob("comparison*"))
    for added in ("Comparison with model-based measurements", "Per-image paired-hole differences",
                  "P95 |difference|", "Paired coverage", "RMSE", "max-difference"):
        assert added not in page
    assert len({r["id"] for r in manifest["images"] if r["status"] == "ok"}) == 2
    with pytest.raises(SystemExit):
        fast_report.main(command)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_all_methods_recursive_and_empty_images(tmp_path, backend):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        try:
            available = cp.cuda.runtime.getDeviceCount()
        except Exception:
            available = 0
        if not available:
            pytest.skip("CUDA unavailable")
    folder = tmp_path / "input"
    (folder / "nested").mkdir(parents=True)
    Image.fromarray(lattice((60, 80), 10, 4.4, 25)[0]).save(folder / "nested" / "holes.PNG")
    Image.fromarray(np.zeros((60, 80), np.uint8)).save(folder / "empty.png")
    output = tmp_path / "report"
    assert fast_report.main([str(folder), "--output", str(output), "--backend", backend,
                             "--recursive", "--include-pixel"]) == 0
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["successful_images"] == 2
    assert {r["file"] for r in manifest["images"]} == {"empty.png", "nested/holes.PNG"}
    rows = _rows(output / "summary.csv")
    for name in ("empty.png", "nested/holes.PNG"):
        records = {r["method"]: r for r in rows if r["image"] == name}
        assert set(records) == set(METHODS) | {"cell_intensity", "image_intensity"}
        if name == "empty.png":
            assert all(int(records[m]["valid"]) == 0 for m in METHODS)
    assert len(list((output / "arrays").glob("*.npz"))) == 2
    assert not list(output.glob("comparison*"))


def test_optional_original_report_exports_and_no_details(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    Image.fromarray(lattice((50, 60), 10, 4, 25)[0]).save(folder / "holes.png")
    output = tmp_path / "report"
    assert fast_report.main([str(folder), "--output", str(output), "--backend", "cpu",
                             "--no-hole-details", "--save-results"]) == 0
    assert not (output / "data" / "holes.detail.js").exists()
    assert not (output / "data" / "holes.tail.js").exists()
    saved = load_cd_results(output / "results" / "holes")
    assert set(saved.measurements) == set(METHODS)
    assert (output / "data" / "holes.local_projection.rays.js").is_file()


@pytest.mark.parametrize("option,value", [("--half-level", "0"), ("--pixel-size", "-1")])
def test_invalid_options_do_not_create_output(tmp_path, option, value):
    folder = tmp_path / "input"
    folder.mkdir()
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(folder / "x.png")
    output = tmp_path / "report"
    with pytest.raises((SystemExit, ValueError)):
        fast_report.main([str(folder), "--output", str(output), option, value])
    assert not output.exists()
