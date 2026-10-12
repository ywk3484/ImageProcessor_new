import json
import re

import numpy as np
import pytest

from subpx import load_cd_results, measure_cds, save_cd_results, save_cd_viewer


def lattice(ratio, *, snr=20, shape=(120, 240), seed=0):
    """Staggered Gaussian holes (FWHM 4, amplitude 50 on 20) with a 15 px hole-free margin."""
    rng = np.random.default_rng(seed)
    pitch, sigma = ratio * 4.0, 4.0 / (2 * np.sqrt(2 * np.log(2)))
    xx, yy = np.meshgrid(np.arange(15, shape[1] - 15, pitch), np.arange(15, shape[0] - 15, pitch))
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    y, x = np.indices(shape, dtype=float)
    image = np.full(shape, 20.0)
    for cx, cy in centers:
        image += 50 * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / sigma**2)
    return image + rng.normal(0, 50 / snr, shape)


def assert_same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, label
    assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), label


def assert_same_result(loaded, original):
    for name in ("image", "binary", "component_labels", "component_ids", "boxes_xywh", "seeds_xy",
                 "cell_labels", "cell_areas_px2", "mean_intensity", "origins_xy", "rois", "masks"):
        assert_same(getattr(loaded, name), getattr(original, name), name)
    if original.background_map is None:
        assert loaded.background_map is None
    else:
        assert_same(loaded.background_map, original.background_map, "background_map")
    assert list(loaded.measurements) == list(original.measurements)
    for method, measured in original.measurements.items():
        other = loaded.measurements[method]
        for name in ("centers_xy", "contours_xy", "valid", "status"):
            assert_same(getattr(other, name), getattr(measured, name), f"{method}.{name}")
        assert set(other.metrics) == set(measured.metrics)
        for name, values in measured.metrics.items():
            assert_same(other.metrics[name], values, f"{method}.{name}")


@pytest.mark.parametrize("kwargs", [
    dict(area_max=400),
    dict(seeds="peaks", invert=True, background_level=80.0),
    dict(background="border", area_max=400),
], ids=["model-threshold", "model-peaks-dark-fixed", "border"])
def test_compact_export_rebuilds_every_array_exactly(tmp_path, kwargs):
    image = lattice(2.0)
    if kwargs.get("invert"):
        image = 100 - image
    result = measure_cds(image, **kwargs)
    full = save_cd_results(result, tmp_path / "full")
    compact = save_cd_results(result, tmp_path / "compact", compact=True)
    assert "contours.csv" not in compact and (tmp_path / "full" / "contours.csv").exists()
    assert compact["measurements.csv"].read_bytes() == full["measurements.csv"].read_bytes()
    with np.load(compact["raw_data.npz"]) as data:
        assert not {"rois", "component_labels"} & set(data.files)
        assert not any(name.endswith("_contours_xy") for name in data.files)
    manifest = json.loads(compact["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["compact"] is True and "contours.csv" not in manifest["files"]
    assert_same_result(load_cd_results(tmp_path / "compact"), result)
    assert_same_result(load_cd_results(tmp_path / "full" / "raw_data.npz"), result)


def test_cli_batch_writes_one_folder_per_image_and_a_summary(tmp_path):
    import csv

    from PIL import Image
    from scripts.analyze_contact_holes import main

    image = np.clip(np.rint(lattice(2.67)), 0, 255).astype(np.uint8)
    for name in ("a.png", "b.png"):
        Image.fromarray(image).save(tmp_path / name)
    (tmp_path / "broken.png").write_text("not an image")
    main([str(tmp_path / "*.png"), "--outputs", "compact", "--area-max", "400", "--output", str(tmp_path / "out")])
    with (tmp_path / "out" / "batch_summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["ok", "ok", "failed"]
    assert rows[0]["cells"] == rows[1]["cells"] != "" and rows[0]["gaussian_median_cd_px"] == rows[1]["gaussian_median_cd_px"]
    for name in ("a", "b"):
        folder = tmp_path / "out" / name
        assert {"measurements.csv", "cells.csv", "raw_data.npz", "manifest.json", "cd_maps.png"} <= {p.name for p in folder.iterdir()}
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["compact"] and "cd_maps.png" in manifest["figures"]
    assert (tmp_path / "out" / "a" / "measurements.csv").read_bytes() == (tmp_path / "out" / "b" / "measurements.csv").read_bytes()


def test_region_viewer_keeps_full_image_ids_and_coordinates(tmp_path):
    result = measure_cds(lattice(2.67), area_max=400)
    path = save_cd_viewer(result, tmp_path / "region.html", region=(60, 30, 80, 50))
    payload = json.loads(re.search(r'<script id="cd-data" type="application/json">(.*?)</script>',
                                   path.read_text(encoding="utf-8"), re.S).group(1))
    inside = np.flatnonzero((result.seeds_xy[:, 0] >= 60) & (result.seeds_xy[:, 0] < 140)
                            & (result.seeds_xy[:, 1] >= 30) & (result.seeds_xy[:, 1] < 80))
    assert payload["count"] == len(inside) > 0
    assert payload["shape"] == [50, 80] and payload["offset"] == [60, 30] and payload["full_shape"] == [120, 240]
    import base64
    ids = np.frombuffer(base64.b64decode(payload["ids"]), "<i4")
    labels = np.frombuffer(base64.b64decode(payload["labels"]), "<i4").reshape(50, 80)
    np.testing.assert_array_equal(ids, inside)
    crop = result.cell_labels[30:80, 60:140]
    np.testing.assert_array_equal(np.where(labels >= 0, ids[labels], -1), np.where(np.isin(crop, inside), crop, -1))
    with pytest.raises(ValueError, match="does not overlap"):
        save_cd_viewer(result, tmp_path / "none.html", region=(500, 0, 10, 10))
