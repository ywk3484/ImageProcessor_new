import csv
import json
import sys

import numpy as np
import pytest

from scripts.benchmark_cd import lattice
from scripts.cd_batch import FrameReader, PccCheck, _sorted_rewrite, file_keys, main, parse_pages, pcc_error
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
            assert float(row["pcc_error"]) >= 0.8 and row["holes"] == "" and row["half_area_cd"] == ""
            continue
        assert float(row["pcc_error"]) < 0.8
        # The same numbers as measuring the page directly with the report's settings (outlines included).
        result = measure_cds(image, methods=ALL_METHODS, **OPTIONS)
        assert row["deblend_converged"] == str(result.meta["deblend_converged"])
        assert int(row["deblend_iterations"]) == result.meta["deblend_iterations_used"]
        assert int(row["deblend_passes"]) == result.meta["deblend_passes"]
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
    invocations = [json.loads(line) for line in (out / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [sum(entry["counts"].values()) for entry in invocations] == [4, 2, 0]
    assert all(entry["elapsed_s"] > 0 for entry in invocations)
    lines = first.splitlines(keepends=True)
    (out / "pages.csv").write_text("".join(lines[:3]) + "".join(lines[4:]) + "lot/a.tif,3,ok,0.1", encoding="utf-8")
    main(run)  # the removed row and the row cut off by a "crash" are measured again
    assert (out / "pages.csv").read_text(encoding="utf-8") == first
    with pytest.raises(SystemExit, match="other settings.*measure"):
        main(run + ["--tail-level", "0.3"])
    with pytest.raises(SystemExit, match="other settings.*hole_check"):
        main(run + ["--max-error", "0.25"])


def test_reference_is_one_page_of_a_tiff_even_one_of_the_dataset(dataset, tmp_path, capsys):
    data, _, _ = dataset
    out = tmp_path / "out"
    run = [str(data), "--reference", str(data / "lot" / "a.tif"), "--output", str(out), "--in-process", *ARGS]
    with pytest.raises(SystemExit, match=r"--reference-page 4 of .*a\.tif: IndexError: page 4 is out of range"):
        main(run + ["--reference-page", "4"])  # one past the last page
    assert not out.exists()  # nothing written, so the corrected command is not refused as other settings
    main(run + ["--reference-page", "3"])
    assert f"Reference: page 3 of {data / 'lot' / 'a.tif'} ({W} x {H} px)" in capsys.readouterr().out
    rows = {(r["file"], int(r["page"])): r for r in _rows(out / "pages.csv")}
    assert [rows[key]["status"] for key in sorted(rows)] == ["no_holes", "ok", "ok", "no_holes", "no_holes", "ok"]
    assert float(rows["lot/a.tif", 3]["pcc_error"]) < 1e-6  # the reference page itself, measured as usual


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file another program holds")
def test_a_results_file_open_in_another_program_at_the_end_keeps_all_rows(tmp_path, capsys):
    path = tmp_path / "pages.csv"
    path.write_text("file,page\nb.tif,1\na.tif,0\n", encoding="utf-8")
    order = lambda row: (row[0], int(row[1]))  # noqa: E731
    with path.open(encoding="utf-8"):  # held open, as Excel holds the files it opens
        _sorted_rewrite(path, ["file", "page"], order)
    assert "open in another program" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == "file,page\nb.tif,1\na.tif,0\n"
    assert [p.name for p in tmp_path.iterdir()] == ["pages.csv"]  # no temporary file left
    _sorted_rewrite(path, ["file", "page"], order)  # closed again: sorted
    assert path.read_text(encoding="utf-8") == "file,page\na.tif,0\nb.tif,1\n"


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
    with pytest.raises(SystemExit):
        parse_pages("-1")


def _array_module(backend):
    if backend == "cpu":
        return np
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False
    if not available:
        pytest.skip("CUDA device unavailable")
    return cp


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_centered_screen_rejects_bright_blank_and_noise_and_ignores_gain_and_offset(backend):
    reference, _ = lattice((256, 320), 10.0, 4.4, 20.0, amplitude=50, background=120)
    blank = np.full(reference.shape, reference.mean())
    noise = blank + np.random.default_rng(4).normal(0, 2, reference.shape)
    # The old screen accepts both because their constant background dominates.
    assert pcc_error(reference, blank) < 0.3
    assert pcc_error(reference, noise) < 0.3
    xp = _array_module(backend)
    check = PccCheck(reference, mode="centered", xp=xp)
    assert check(blank) == 1
    assert check(noise) > 0.99
    assert check(np.zeros(reference.shape)) == 1
    moved = np.roll(reference.astype(float), (3, 5), axis=(0, 1))
    assert check(2.3 * moved + 1500) < 1e-6
    assert PccCheck(blank, mode="centered", xp=xp)(reference) == 1


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_centered_screen_matches_centered_scikit_image_for_each_crop_size(backend):
    from skimage.registration import phase_cross_correlation
    from scripts.cd_batch import _common_centre

    reference = _holes(1)
    check = PccCheck(reference, mode="centered", xp=_array_module(backend))
    for image in (_holes(2), _holes(3)[:100, :150], _holes(4)):
        ref, mov = _common_centre(reference.astype(float), image.astype(float))
        _, expected, _ = phase_cross_correlation(ref - ref.mean(), mov - mov.mean(), normalization=None)
        assert check(image) == pytest.approx(expected, abs=1e-12)


def test_resume_checks_measurement_code_but_not_unrelated_git_commits(tmp_path, monkeypatch):
    import scripts.cd_batch as batch

    settings = {"measure": {"methods": ["gaussian"]}, "hole_check": None}
    monkeypatch.setattr(batch, "measurement_fingerprint", lambda: "algorithm-one")
    monkeypatch.setattr(batch, "_git_commit", lambda: "commit-one")
    batch.check_settings(tmp_path, settings)
    monkeypatch.setattr(batch, "_git_commit", lambda: "documentation-commit")
    batch.check_settings(tmp_path, settings)
    monkeypatch.setattr(batch, "measurement_fingerprint", lambda: "algorithm-two")
    with pytest.raises(SystemExit, match="different or unversioned measurement implementation"):
        batch.check_settings(tmp_path, settings)
    (tmp_path / "run.json").write_text(json.dumps({"settings": settings}), encoding="utf-8")
    with pytest.raises(SystemExit, match="new --output folder"):
        batch.check_settings(tmp_path, settings)


def test_measurement_fingerprint_tracks_measurement_source_only(tmp_path, monkeypatch):
    import scripts.cd_batch as batch

    (tmp_path / "subpx").mkdir()
    code = tmp_path / "subpx" / "cd.py"
    code.write_text("measurement = 1\n", encoding="utf-8")
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    before = batch.measurement_fingerprint()
    (tmp_path / "README.md").write_text("new guide", encoding="utf-8")
    (tmp_path / "subpx" / "viewer.py").write_text("viewer = 2\n", encoding="utf-8")
    assert batch.measurement_fingerprint() == before
    code.write_text("measurement = 2\n", encoding="utf-8")
    assert batch.measurement_fingerprint() != before


def test_resume_cannot_confuse_distinct_inputs_with_the_same_filename(tmp_path):
    import tifffile

    files = []
    for folder in ("first", "second"):
        (tmp_path / folder).mkdir()
        path = tmp_path / folder / "same.tif"
        tifffile.imwrite(path, _holes(1), photometric="minisblack")
        files.append(path)
    out = tmp_path / "out"
    options = ["--output", str(out), "--no-hole-check", "--in-process", "--pages", "0:0", *ARGS]
    assert main([str(files[0]), *options]) == 0
    with pytest.raises(SystemExit, match="other settings.*inputs"):
        main([str(files[1]), *options])


def test_batch_options_select_mode_specific_defaults_and_local_methods(tmp_path):
    import tifffile
    from scripts.cd_batch import argument_parser, build_settings

    parser = argument_parser()
    args = parser.parse_args(["input.tif", "--output", "out", "--no-hole-check", "--methods", "local_half_area",
                              "--deblend-iterations", "60", "--deblend-tolerance", "0.0001"])
    settings = build_settings(args)
    assert settings["measure"]["methods"] == ["local_half_area"]
    assert settings["measure"]["metrics_only"] is True
    assert settings["measure"]["deblend_iterations"] == 60
    assert settings["measure"]["deblend_tolerance"] == 1e-4
    reference = tmp_path / "reference.tif"
    tifffile.imwrite(reference, np.arange(64, dtype=np.uint8).reshape(8, 8))
    common = ["input.tif", "--output", "out", "--reference", str(reference)]
    assert build_settings(parser.parse_args(common))["hole_check"]["max_error"] == 0.8
    assert build_settings(parser.parse_args(common + ["--hole-check-mode", "raw"]))["hole_check"]["max_error"] == 0.3
    assert build_settings(parser.parse_args(common + ["--max-error", "0.37"]))["hole_check"]["max_error"] == 0.37


def test_batch_intensity_keeps_its_model_tail_definition():
    from scripts.benchmark_cd import lattice
    from scripts.cd_batch import page_values
    from scripts.cd_report import cell_intensity
    from subpx import measure_cds

    image = lattice((60, 80), 10, 4, 25)[0]
    for method in ("local_tail_area", "pixel_tail_area"):
        result = measure_cds(image, methods=(method,), summary_only=True, backend="cpu", area_max=200)
        assert cell_intensity(result) is not None  # the report labels this raw-cell intensity explicitly
        assert np.isnan(page_values(result, image, (method,))["cell_intensity"])
