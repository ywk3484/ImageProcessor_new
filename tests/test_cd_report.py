import csv

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
    options = dict(nominal_cd=100.0, pixel_size=21.0, unit="nm", dark_level=6.0, area_max=200)
    main([str(folder), "--output", str(tmp_path / "out"), "--nominal-cd", "100", "--pixel-size", "21", "--unit", "nm",
          "--dark-level", "6", "--backend", "cpu", "--save-results"])
    out = tmp_path / "out"
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["image"] for r in rows if r["status"] == "failed"] == ["broken.png"]
    fixed = {}
    for name, image in images.items():
        result = measure_cds(image, methods=("gaussian", "logquad", "gradient", "halfmax", "disk", "flux_ratio",
                                             "projection"), **options, **fixed)
        # The first image calibrates the blur and the flux_ratio scale; later images reuse them.
        fixed = fixed or {"psf_fwhm": result.meta["psf_fwhm_px"] * 21.0,
                          "flux_ratio_scale": result.meta["flux_ratio_scale"]}
        mine = {r["method"]: r for r in rows if r["image"] == name}
        assert set(mine) == set(result.measurements)
        for method, measured in result.measurements.items():
            cd = measured.metrics["cd_equivalent_px"][measured.valid] * 21.0
            assert int(mine[method]["valid"]) == cd.size
            assert float(mine[method]["mean"]) == np.float64(cd.mean())
            assert np.isclose(float(mine[method]["three_sigma_pct"]), 300 * cd.std(ddof=1) / cd.mean())
    with (out / "between_images.csv").open(encoding="utf-8") as handle:
        between = {r["method"]: r for r in csv.DictReader(handle)}
    for method in result.measurements:
        means = np.array([float(r["mean"]) for r in rows if r.get("method") == method])
        assert int(between[method]["images"]) == 2
        assert np.isclose(float(between[method]["mean_of_means"]), means.mean())
        assert np.isclose(float(between[method]["three_sigma_pct"]), 300 * means.std(ddof=1) / means.mean())
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "Between images" in page and "calibrated once, on a scan.png" in page
    assert "a_scan" in page and "broken.png" in page and "failed:" in page
    assert (out / "data" / "a_scan.js").read_text(encoding="utf-8").startswith('window.CD_DATA["a_scan"]=')
    assert (out / "images" / "b.png").exists()
    assert set(load_cd_results(out / "results" / "b").measurements) == set(result.measurements)


def test_without_a_blur_the_hole_fit_methods_are_skipped(tmp_path):
    from PIL import Image
    from scripts.cd_report import main

    (tmp_path / "in").mkdir()
    image = pulsed_lattice(psf_fwhm=3.3, shape=(60, 80))[0].astype(np.uint8)
    Image.fromarray(image).save(tmp_path / "in" / "x.png")
    main([str(tmp_path / "in"), "--output", str(tmp_path / "out"), "--backend", "cpu"])
    with (tmp_path / "out" / "summary.csv").open(encoding="utf-8") as handle:
        methods = {r["method"] for r in csv.DictReader(handle)}
    assert methods == {"gaussian", "logquad", "gradient", "halfmax", "projection"}
    assert "skipped" in (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert not (tmp_path / "out" / "results").exists()
