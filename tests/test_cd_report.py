import base64
import csv
import json

import numpy as np

from scripts.validate_invariant_cd import pulsed_lattice
from subpx import load_cd_results, measure_cds


def test_report_statistics_match_direct_measurement_and_failed_images_are_listed(tmp_path):
    from PIL import Image
    from scripts.cd_report import main

    folder = tmp_path / "in"
    folder.mkdir()
    images = {}
    for name, blur in (("a scan.png", 3.0), ("b.png", 3.4)):
        image = pulsed_lattice(psf_fwhm=blur, shape=(80, 100), seed=len(images))[0].astype(np.uint8)
        Image.fromarray(image).save(folder / name)
        images[name] = image
    (folder / "broken.png").write_text("not an image")
    (folder / "notes.txt").write_text("ignored")
    options = dict(pixel_size=21.0, unit="nm", area_max=200)
    main([str(folder), "--output", str(tmp_path / "out"), "--pixel-size", "21", "--unit", "nm", "--backend", "cpu",
          "--save-results"])
    out = tmp_path / "out"
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["image"] for r in rows if r["status"] == "failed"] == ["broken.png"]
    results = {}
    for name, image in images.items():
        result = measure_cds(image, methods=("gaussian", "projection", "half_area", "half_area_min",
                                             "half_area_min_avg"), **options)
        results[name] = result
        mine = {r["method"]: r for r in rows if r["image"] == name}
        assert set(mine) == set(result.measurements)
        for method, measured in result.measurements.items():
            # The CD and, separately, the x and the y widths.
            for suffix, metric in (("", "cd_equivalent_px"), ("_x", "cd_x_px"), ("_y", "cd_y_px")):
                cd = measured.metrics[metric][measured.valid] * 21.0
                assert int(mine[method]["valid" + suffix]) == cd.size
                assert float(mine[method]["mean" + suffix]) == np.float64(cd.mean())
                assert np.isclose(float(mine[method]["three_sigma_pct" + suffix]), 300 * cd.std(ddof=1) / cd.mean())
    with (out / "between_images.csv").open(encoding="utf-8") as handle:
        between = {r["method"]: r for r in csv.DictReader(handle)}
    for method in result.measurements:
        for suffix in ("", "_x", "_y"):
            means = np.array([float(r["mean" + suffix]) for r in rows if r.get("method") == method])
            assert int(between[method]["images" + suffix]) == 2
            assert np.isclose(float(between[method]["mean_of_means" + suffix]), means.mean())
            assert np.isclose(float(between[method]["three_sigma_pct" + suffix]),
                              300 * means.std(ddof=1) / means.mean())
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "Between images" in page and "half_area_min_avg" in page and "y width" in page
    assert "a_scan" in page and "broken.png" in page and "failed:" in page
    assert (out / "data" / "a_scan.js").read_text(encoding="utf-8").startswith('window.CD_DATA["a_scan"]=')
    assert (out / "images" / "b.png").exists()
    assert set(load_cd_results(out / "results" / "b").measurements) == set(result.measurements)
    # Hole details of image b: the cleaned cell within one 8-bit step, projection's sums exact, half_area's numbers.
    detail = _load_js(out / "data" / "b.detail.js", "CD_DETAIL")
    h, w = detail["h"], detail["w"]
    codes = _array(detail["clean"]["codes"], np.uint8).reshape(-1, h, w)
    lo, step = _array(detail["clean"]["lo"], np.float32), _array(detail["clean"]["step"], np.float32)
    np.testing.assert_array_equal(codes != 255, result.masks)
    cells = lo[:, None, None] + codes * step[:, None, None]
    assert np.all(np.abs(cells - result.rois)[result.masks] <= 0.51 * np.repeat(step, h * w).reshape(-1, h, w)[result.masks] + 1e-4)
    np.testing.assert_allclose(_array(detail["prof_x"], np.float32).reshape(-1, w),
                               np.where(result.masks, result.rois, 0).sum(axis=1), rtol=1e-5, atol=1e-4)
    for name, background in (("half_area", "median"), ("half_area_min", "minimum"), ("half_area_min_avg", "minimum")):
        half, panel = result.measurements[name], detail["half"][name]
        assert panel["background"] == background
        assert ("brightest 4" in panel["peak_rule"]) == name.endswith("_avg")
        for key, metric in (("level", "local_level"), ("peak", "peak_signal"), ("threshold", "threshold")):
            np.testing.assert_allclose(_array(panel[key], np.float32), half.metrics[metric], rtol=1e-6)
        rays = _array(panel["rays"], np.uint8).reshape(len(half.valid), -1) / 16
        ok = half.valid
        np.testing.assert_allclose(rays[ok], half.geometry["edge_px"][ok][:, ::4], atol=1 / 32 + 1e-6)
    # One colour range per map layer, shared by the images: 2nd-98th percentile of both images' CDs together.
    page_ranges = json.loads(page.split("RANGES=")[1].split(", $=")[0])
    pooled = np.concatenate([r.measurements["projection"].metrics["cd_y_px"][r.measurements["projection"].valid]
                             for r in results.values()])
    np.testing.assert_allclose(page_ranges["projection_y"], np.quantile(pooled, [0.02, 0.98]))
    data = _load_js(out / "data" / "b.js", "CD_DATA")
    assert set(data["methods"]) == {name + suffix for name in result.measurements for suffix in ("", "_x", "_y")}
    np.testing.assert_allclose(page_ranges["half_area_min_x"], np.quantile(np.concatenate(
        [r.measurements["half_area_min"].metrics["cd_x_px"][r.measurements["half_area_min"].valid]
         for r in results.values()]), [0.02, 0.98]))


def _array(text, dtype):
    return np.frombuffer(base64.b64decode(text), dtype)


def _load_js(path, name):
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("]=") + 2:].rstrip().rstrip(";"))


def test_default_report_runs_the_shape_free_methods_without_inputs(tmp_path):
    from PIL import Image
    from scripts.cd_report import main

    (tmp_path / "in").mkdir()
    image = pulsed_lattice(psf_fwhm=3.3, shape=(60, 80))[0].astype(np.uint8)
    Image.fromarray(image).save(tmp_path / "in" / "x.png")
    main([str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--backend", "cpu", "--no-hole-details"])
    with (tmp_path / "out" / "summary.csv").open(encoding="utf-8") as handle:
        methods = {r["method"] for r in csv.DictReader(handle)}
    assert methods == {"gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg"}
    assert not (tmp_path / "out" / "results").exists()
    assert not (tmp_path / "out" / "data" / "x.detail.js").exists()
