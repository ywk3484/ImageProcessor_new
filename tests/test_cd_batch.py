import csv

import numpy as np
import pytest

from scripts.benchmark_cd import lattice
from scripts.cd_batch import FrameReader, file_keys, main, parse_pages, pcc_error
from scripts.cd_report import ALL_METHODS, cd_statistics
from subpx import measure_cds

H, W = 120, 160
OPTIONS = dict(area_min=3, area_max=200, pixel_size=21.0, unit="nm")
ARGS = ["--pixel-size", "21", "--unit", "nm", "--backend", "cpu"]


def _holes(seed, shift=(0, 0)):
    big, _ = lattice((H + 10, W + 10), 10.0, 4.4, 20.0, seed=seed)
    return big[shift[0]:shift[0] + H, shift[1]:shift[1] + W]


def _noise(seed):
    return np.clip(np.rint(40 + 8 * np.random.default_rng(seed).standard_normal((H, W))), 0, 255).astype(np.uint8)


def _lines(seed):
    from scipy.ndimage import gaussian_filter

    image = 40 + 110 * ((np.arange(W)[None] + np.zeros((H, 1))) % 20 < 8)
    noise = np.random.default_rng(seed).normal(0, 7, (H, W))
    return np.clip(np.rint(gaussian_filter(image, 1.9) + noise), 0, 255).astype(np.uint8)


@pytest.fixture()
def dataset(tmp_path):
    """Two TIFFs in a dataset folder (one in a subfolder) and a reference hole image outside it."""
    import tifffile

    data = tmp_path / "data"
    (data / "lot").mkdir(parents=True)
    pages = {"lot/a.tif": [_holes(1, (2, 3)), _noise(2), _lines(3), _holes(4, (5, 1))],
             "b.tif": [_lines(5), _holes(6, (7, 7))]}
    for name, images in pages.items():
        tifffile.imwrite(data / name, np.stack(images), photometric="minisblack")
    tifffile.imwrite(tmp_path / "reference.tif", _holes(99))
    return data, tmp_path / "reference.tif", pages


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_pcc_error_is_scikit_images_error_without_normalization():
    from skimage.registration import phase_cross_correlation

    reference = _holes(1)
    for image in (_holes(2, (3, 6)), _noise(3), _lines(4), _holes(5)[:100, :150]):
        ref = reference
        if image.shape != reference.shape:  # compared on the centre part both share
            y0, x0 = (H - image.shape[0]) // 2, (W - image.shape[1]) // 2
            ref = reference[y0:y0 + image.shape[0], x0:x0 + image.shape[1]]
        _, expected, _ = phase_cross_correlation(ref, image, normalization=None)
        assert np.isclose(pcc_error(reference, image), expected, rtol=1e-9, atol=1e-12)
    assert pcc_error(reference, _holes(2, (3, 6))) < 0.3 < pcc_error(reference, _noise(3))
    assert pcc_error(reference, _lines(4)) > 0.3
    assert np.isnan(pcc_error(reference, np.zeros((H, W))))


def test_batch_measures_the_hole_pages_like_the_report_and_skips_the_others(dataset, tmp_path):
    data, reference, pages = dataset
    out = tmp_path / "out"
    main([str(data), "--reference", str(reference), "--output", str(out), "--in-process", *ARGS])
    rows = _rows(out / "pages.csv")
    assert [(r["file"], int(r["page"]), r["status"]) for r in rows] == [
        ("b.tif", 0, "no_holes"), ("b.tif", 1, "ok"),
        ("lot/a.tif", 0, "ok"), ("lot/a.tif", 1, "no_holes"), ("lot/a.tif", 2, "no_holes"), ("lot/a.tif", 3, "ok")]
    for row in rows:
        image = pages[row["file"]][int(row["page"])]
        assert float(row["image_intensity"]) == float(np.mean(image, dtype=np.float64))
        if row["status"] != "ok":
            assert float(row["pcc_error"]) >= 0.3 and row["holes"] == "" and row["half_area_cd"] == ""
            continue
        assert float(row["pcc_error"]) < 0.3
        # The same numbers as measuring the page directly with the report's settings (outlines included).
        result = measure_cds(image, methods=ALL_METHODS, **OPTIONS)
        stats = cd_statistics(result)
        assert int(row["holes"]) == len(result.seeds_xy)
        for name in ALL_METHODS:
            assert int(row[f"{name}_valid"]) == stats[name]["valid"]
            for suffix in ("", "_x", "_y"):
                assert float(row[f"{name}_cd{suffix}"]) == stats[name]["mean" + suffix]
        tail = result.measurements["tail_area"]
        assert float(row["cell_intensity"]) == float(tail.metrics["mean_signal"][tail.valid].mean())
    timings = _rows(out / "timings.csv")
    assert len(timings) == len(rows) and all(float(t["measure_s"]) > 0 for t in timings if t["status"] == "ok")


def test_batch_resumes_and_refuses_other_settings(dataset, tmp_path):
    data, reference, _ = dataset
    out = tmp_path / "out"
    run = [str(data), "--reference", str(reference), "--output", str(out), "--in-process", *ARGS]
    main(run + ["--pages", "0:2"])  # a trial run on the first two pages of every file
    assert len(_rows(out / "pages.csv")) == 4
    main(run)  # the full run continues in the same folder
    first = (out / "pages.csv").read_text(encoding="utf-8")
    assert len(_rows(out / "pages.csv")) == 6
    main(run)  # nothing left to do
    assert (out / "pages.csv").read_text(encoding="utf-8") == first
    lines = first.splitlines(keepends=True)
    (out / "pages.csv").write_text("".join(lines[:3]) + "".join(lines[4:]) + "lot/a.tif,3,ok,0.1", encoding="utf-8")
    main(run)  # the removed row and the row cut off by a "crash" are measured again
    assert (out / "pages.csv").read_text(encoding="utf-8") == first
    with pytest.raises(SystemExit, match="other settings.*measure"):
        main(run + ["--tail-level", "0.3"])
    with pytest.raises(SystemExit, match="other settings.*hole_check"):
        main(run + ["--max-error", "0.25"])


def test_workers_survive_a_crash_and_a_hung_page(dataset, tmp_path):
    import tifffile

    data, reference, _ = dataset
    crash, hang = _holes(11), _holes(12)
    crash[0, 0], hang[0, 0] = 1, 2  # markers the error function below reacts to
    tifffile.imwrite(data / "c.tif", np.stack([crash, _holes(13), hang, _holes(14)]), photometric="minisblack")
    plugin = tmp_path / "my_error.py"
    plugin.write_text("import os, time\n"
                      "def error(reference, image):\n"
                      "    if image[0, 0] == 1:\n        os._exit(3)\n"
                      "    if image[0, 0] == 2:\n        time.sleep(120)\n"
                      "    return 0.0 if image.mean() > 55 else 1.0\n", encoding="utf-8")
    out = tmp_path / "out"
    code = main([str(data / "c.tif"), "--reference", str(reference), "--error-function", f"{plugin}:error",
                 "--output", str(out), "--workers", "1", "--page-timeout", "10", *ARGS])
    assert code == 1  # some pages failed
    failed = {int(r["page"]): r["error"] for r in _rows(out / "failed.csv")}
    assert set(failed) == {0, 2}
    assert "exit code 3" in failed[0] and "timeout" in failed[2]
    assert [(int(r["page"]), r["status"]) for r in _rows(out / "pages.csv")] == [(1, "ok"), (3, "ok")]


def test_ctrl_c_stops_after_the_running_pages_and_the_next_run_finishes(tmp_path):
    import signal
    import threading
    import time

    import tifffile

    data = tmp_path / "data"
    data.mkdir()
    tifffile.imwrite(data / "s.tif", np.stack([_holes(seed, (seed % 5, seed % 7)) for seed in range(8)]),
                     photometric="minisblack")
    out = tmp_path / "out"
    run = [str(data), "--no-hole-check", "--output", str(out), "--workers", "1", *ARGS]

    def press_ctrl_c_after_two_pages():
        deadline = time.time() + 120
        while time.time() < deadline:
            if (out / "pages.csv").exists() and len(_rows(out / "pages.csv")) >= 2:
                signal.raise_signal(signal.SIGINT)  # handled by the batch: no KeyboardInterrupt
                return
            time.sleep(0.05)

    presser = threading.Thread(target=press_ctrl_c_after_two_pages)
    presser.start()
    main(run)
    presser.join()
    stopped = len(_rows(out / "pages.csv"))
    assert 2 <= stopped < 8
    main(run)
    rows = _rows(out / "pages.csv")
    assert [int(r["page"]) for r in rows] == list(range(8)) and {r["status"] for r in rows} == {"ok"}


def test_frame_reader_reads_every_frame_of_large_imagej_stacks_and_compressed_files(tmp_path):
    import tifffile

    stack = np.stack([_holes(seed) for seed in range(5)])
    for name, options in (("imagej_over_4gb_layout.tif", {"imagej": True, "truncate": True}),
                          ("compressed.tif", {"compression": "zlib", "photometric": "minisblack"})):
        tifffile.imwrite(tmp_path / name, stack, **options)
        reader = FrameReader(tmp_path / name)
        try:
            assert reader.count == 5
            for index in range(5):
                np.testing.assert_array_equal(reader.read(index), stack[index])
        finally:
            reader.close()


def test_sequential_frames_equal_the_frame_reader_without_counting_pages(tmp_path):
    import tifffile

    from scripts.cd_batch import iter_frames

    stack = np.stack([_holes(seed) for seed in range(6)])
    for name, options in (("plain.tif", {"photometric": "minisblack", "metadata": None}),
                          ("bigtiff.tif", {"photometric": "minisblack", "bigtiff": True}),
                          ("imagej_over_4gb_layout.tif", {"imagej": True, "truncate": True}),
                          ("imagej.tif", {"imagej": True}),
                          ("compressed.tif", {"compression": "zlib", "photometric": "minisblack"})):
        tifffile.imwrite(tmp_path / name, stack, **options)
        frames = list(iter_frames(tmp_path / name))
        assert [index for index, _ in frames] == list(range(6)), name
        for index, image in frames:
            np.testing.assert_array_equal(image, stack[index], err_msg=name)
        picked = [index for index, _ in iter_frames(tmp_path / name, parse_pages("1:5"), skip={2})]
        assert picked == [1, 3, 4], name
    tifffile.imwrite(tmp_path / "rgb.tif", np.zeros((2, 8, 8, 3), np.uint8), photometric="rgb")
    assert all(isinstance(image, ValueError) for _, image in iter_frames(tmp_path / "rgb.tif"))
    with pytest.raises(SystemExit):
        parse_pages("-5:")


def test_outlines_off_and_fewer_angles_leave_every_number_unchanged():
    image = _holes(3)
    full = measure_cds(image, methods=ALL_METHODS, **OPTIONS)
    lean = measure_cds(image, methods=ALL_METHODS, n_angles=16, outlines=False, **OPTIONS)
    for name in ALL_METHODS:
        a, b = full.measurements[name], lean.measurements[name]
        np.testing.assert_array_equal(a.status, b.status)
        np.testing.assert_array_equal(a.centers_xy, b.centers_xy)
        for metric, values in a.metrics.items():
            if metric != "perimeter_px":
                np.testing.assert_array_equal(values, b.metrics[metric], err_msg=f"{name} {metric}")


def test_file_keys_and_page_selection(tmp_path):
    files = [tmp_path / "d" / "x" / "a.tif", tmp_path / "d" / "b.tif"]
    assert file_keys(files) == ["x/a.tif", "b.tif"]
    assert file_keys(files[:1]) == ["a.tif"]
    assert list(range(10)[parse_pages("2:5")]) == [2, 3, 4]
    assert list(range(10)[parse_pages("7:")]) == [7, 8, 9]
    assert list(range(10)[parse_pages("4")]) == [4]
    with pytest.raises(SystemExit):
        parse_pages("a:b")
