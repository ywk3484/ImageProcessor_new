"""PNG report integration: independent schemes, numerical exports and failure handling."""

import csv
import json

import numpy as np
from PIL import Image
import pytest

from scripts import cd_fast_report as fast_report
from scripts.benchmark_cd import lattice
from subpx import measure_cds


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_report_preserves_numeric_values_and_compares_each_png(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    images = {}
    for name, fwhm in (("a scan.png", 4.0), ("a_scan.PNG", 4.4)):
        images[name] = lattice((60, 80), 10, fwhm, 25)[0]
        Image.fromarray(images[name]).save(folder / name)
    (folder / "broken.png").write_bytes(b"not a PNG")
    (folder / "ignore.txt").write_text("unrelated")
    output = tmp_path / "report"
    command = [str(folder), "--output", str(output), "--backend", "cpu", "--pixel-size", "21", "--unit", "nm"]
    assert fast_report.main(command) == 1  # records the broken input and finishes the useful report
    summary = _rows(output / "summary.csv")
    changes = _rows(output / "comparison.csv")
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["successful_images"] == 2 and manifest["failed_images"] == 1
    assert manifest["reference_methods"] == list(fast_report.REFERENCE_METHODS)
    assert len(changes) == 18 and {r["image_id"] for r in changes} == set(images)
    assert {r["axis"] for r in changes} == {"cd", "x", "y"}
    assert {r["unit"] for r in changes} == {"nm"}
    assert all(not r["method_candidate"].startswith("pixel_") for r in changes)
    for name, image in images.items():
        direct = measure_cds(image, methods=fast_report.FAST_METHODS, summary_only=True,
                             area_max=200, pixel_size=21, unit="nm", backend="cpu", n_angles=16, outlines=False)
        records = {r["method"]: r for r in summary if r["image"] == name}
        source = next(r for r in manifest["images"] if r["file"] == name)
        assert source["fast_meta"]["deblend_iterations_used"] == 0
        assert source["reference_meta"]["deblend_iterations_used"] > 0
        with np.load(output / "arrays" / f"{source['id']}.fast.npz", allow_pickle=False) as arrays:
            for method in fast_report.FAST_METHODS:
                measured = direct.measurements[method]
                values = measured.metrics["cd_equivalent_px"][measured.valid] * 21
                assert float(records[method]["mean"]) == pytest.approx(values.mean())
                np.testing.assert_array_equal(arrays[f"{method}_valid"], measured.valid)
                np.testing.assert_allclose(arrays[f"{method}_cd_equivalent_px"], measured.metrics["cd_equivalent_px"])
        assert (output / "arrays" / f"{source['id']}.reference.npz").exists()
    assert (output / "comparison.png").is_file()
    html = (output / "index.html").read_text(encoding="utf-8")
    assert "Between images" in html and "local_half_area" in html and "broken.png" in html
    assert "Cell intensity: local_tail_area" in html or "local_tail_area" in html
    assert len({r["id"] for r in manifest["images"] if r["status"] == "ok"}) == 2
    assert all("rays_file" not in p.read_text(encoding="utf-8") for p in (output / "data").glob("*.js"))
    with pytest.raises(SystemExit):
        fast_report.main(command)  # a rerun cannot silently mix old and new report files


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_fast_only_with_pixel_methods_and_empty_image(tmp_path, backend):
    if backend == "gpu":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA unavailable")
    folder = tmp_path / "input"
    folder.mkdir()
    Image.fromarray(lattice((60, 80), 10, 4.4, 25)[0]).save(folder / "holes.png")
    Image.fromarray(np.zeros((60, 80), np.uint8)).save(folder / "empty.png")
    output = tmp_path / "report"
    assert fast_report.main([str(folder), "--output", str(output), "--backend", backend,
                             "--fast-only", "--include-pixel"]) == 0
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["reference_methods"] == [] and len(manifest["fast_methods"]) == 6
    assert manifest["successful_images"] == 2
    assert all(r["reference_meta"] is None for r in manifest["images"])
    assert not _rows(output / "comparison.csv")
    assert len(list((output / "arrays").glob("*.fast.npz"))) == 2
    assert not list((output / "arrays").glob("*.reference.npz"))
    if backend == "gpu":
        nonempty = next(r for r in manifest["images"] if r["file"] == "holes.png")
        assert nonempty["fast_meta"]["cell_storage"] == "fused"
    assert "comparability has not been assessed" in (output / "index.html").read_text(encoding="utf-8")


def test_failed_reference_retains_fast_report(tmp_path, monkeypatch):
    folder = tmp_path / "input"
    folder.mkdir()
    Image.fromarray(lattice((50, 60), 10, 4, 25)[0]).save(folder / "holes.png")
    original = fast_report.measure_cds

    def fail_reference(image, **options):
        if options["methods"] == fast_report.REFERENCE_METHODS:
            raise RuntimeError("reference failed <example>")
        return original(image, **options)

    monkeypatch.setattr(fast_report, "measure_cds", fail_reference)
    output = tmp_path / "report"
    assert fast_report.main([str(folder), "--output", str(output), "--backend", "cpu"]) == 1
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "reference failed &lt;example&gt;" in page
    rows = _rows(output / "summary.csv")
    assert {r["method"] for r in rows if r["status"] == "failed"} == set(fast_report.REFERENCE_METHODS)
    assert all(r["status"] == "ok" for r in rows if r["method"] in fast_report.FAST_METHODS)
    assert (output / "data" / "holes.js").is_file()


def test_invalid_options_do_not_create_output(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    Image.fromarray(np.zeros((20, 20), np.uint8)).save(folder / "x.png")
    output = tmp_path / "report"
    with pytest.raises(SystemExit):
        fast_report.main([str(folder), "--output", str(output), "--max-difference", "-1"])
    assert not output.exists()
