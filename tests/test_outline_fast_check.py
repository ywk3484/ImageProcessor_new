"""scripts/outline_fast_check.py: outline_fit_fast's page CD against the start and the cap of its final fit.

The script measures; it changes nothing in the library. Its helpers are tested without a GPU, the whole check on a
folder of two small generated images with one.
"""

import json
import re

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from scripts import outline_fast_check as check
from subpx._outline.model import Template
from subpx.synthetic_arrays import render_hole_array


@pytest.mark.parametrize("blur, count, second", [("gauss2", 9, (2, 3, 4)), ("lens", 4, (2,))])
def test_other_starts_move_only_the_blurs_second_part(blur, count, second):
    t = Template(upsample=8, size_px=24, blur=blur)
    theta0 = t.start(2.4, 1.2)
    starts, labels = check.other_starts(t, theta0, blur)
    assert starts.shape == (count, t.n_params) and len(set(labels)) == count
    kept = [k for k in range(t.n_params) if k - t.n_shape not in second]
    np.testing.assert_allclose(starts[:, kept], np.broadcast_to(theta0[kept], (count, len(kept))))
    lo, hi = t.bounds()
    assert (starts > lo).all() and (starts < hi).all()
    assert len({tuple(np.round(s, 6)) for s in starts}) == count  # all different


def test_ends_are_grouped_by_page_cd_and_ordered_by_cost():
    cds = np.array([100.00, 99.20, 100.01, 100.015, 99.21])
    costs = np.array([50.0, 80.0, 49.0, 51.0, 79.0])
    ends = check.group_ends(cds, costs, same=0.02)
    assert [g["starts"] for g in ends] == [3, 2]
    assert ends[0]["cd"] == pytest.approx(np.mean([100.00, 100.01, 100.015])) and ends[0]["cost"] == 49.0
    assert ends[0]["best_start"] == 2 and ends[1]["best_start"] == 4 and ends[1]["cost"] == 79.0
    assert len(check.group_ends(cds, costs, same=1.0)) == 1


def _image(rep, best, converged=True, n_ends=1, gap=None, iterations=20):
    return {"fitted": True, "rep": rep, "end": rep, "best": best, "rep_converged": converged, "gap": gap,
            "end_iterations": iterations, "ends": [{}] * n_ends, "best_leftover": 1.3 + 0.01 * iterations,
            "best_blur_fwhm": 75.0 - (best - 100.0)}


def test_folder_line_is_short_ascii_and_counts_what_moved():
    results = [_image(100.05, 100.05), _image(100.04, 100.04), _image(99.98, 99.98), _image(100.02, 100.02, iterations=71),
               _image(99.24, 100.12, converged=False, n_ends=2, gap=710.4, iterations=34), _image(100.13, 100.13),
               _image(100.11, 100.11), _image(99.25, 100.18, converged=False, n_ends=2, gap=917.0, iterations=36),
               {"fitted": False, "reason": "fewer than 10 holes"}]
    line = check.folder_line(results, same=0.021, digits=2)
    assert line.isascii() and len(line) <= 150
    assert line.startswith("check N=8 | rep 99.24-100.13 3s 1.13 nc 2 | end 3s 1.13 it<=71 | best 99.98-100.18 3s 0.20 moved 2 | ")
    assert "| ends 1-2 gap 710 | left 1.5-2.0 | rblur -1.00" in line
    assert check.folder_line(results[-1:], 0.021, 2) == "check N=0: no image was fitted"
    one = check.folder_line(results[:1], 0.021, 2)  # one image: no spread, no correlation
    assert "3s - nc 0" in one and one.endswith("rblur -")


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")


def test_the_check_runs_on_a_folder_and_ends_with_its_line(gpu, tmp_path, capsys, monkeypatch):
    from PIL import Image

    import subpx
    import subpx.outline as outline

    folder = tmp_path / "in"
    folder.mkdir()
    for seed in (5, 6):
        page = render_hole_array((120, 170), hole="circle", size_nm=(100, 100), pitch_nm=(210, 210), layout="square",
                                 cd_sigma_nm=3.0, seed=seed, backend="cpu")
        Image.fromarray(page.image.astype(np.uint8)).save(folder / f"page{seed}.png")
    # what the script asks measure_cds for, and what it is told (a second call would not do: on an image of 135
    # holes two runs of the same call differ by up to 0.05 nm, the GPU seeds being summed in a varying order)
    calls, measure = [], subpx.measure_cds

    def recorded(image, **options):
        result = measure(image, **options)
        calls.append((options, result.meta["outline_fit_fast"]))
        return result

    monkeypatch.setattr(subpx, "measure_cds", recorded)
    original = outline.fit_outline_fast
    saved_to = tmp_path / "check.json"
    assert check.main([str(folder), "--pixel-size", "21", "--unit", "nm", "--output", str(saved_to)]) == 0
    assert outline.fit_outline_fast is original  # the tap on the seeds is taken off again
    printed = capsys.readouterr().out.strip().splitlines()
    last = printed[-1]
    assert last.isascii() and len(last) <= 150
    number = r"\d+\.\d\d"
    assert re.fullmatch(rf"check N=2 \| rep {number}-{number} 3s {number} nc \d \| end 3s {number} it<=\d+ \| "
                        rf"best {number}-{number} 3s {number} moved \d \| ends \d-\d gap (-|\d+) \| "
                        rf"left \d+\.\d-\d+\.\d \| rblur -", last), last
    assert all(line.isascii() for line in printed)
    saved = json.loads(saved_to.read_text(encoding="utf-8"))
    assert saved["line"] == last and [image["image"] for image in saved["images"]] == ["page5.png", "page6.png"]
    assert len(calls) == 2
    for image, (options, meta) in zip(saved["images"], calls):
        assert image["fitted"] and len(image["starts"]) == 10 and image["starts"][0]["start"] == "as the report starts"
        # "rep" is the report's own reading: measure_cds with the reports' options
        assert options["methods"] == ("outline_fit_fast",) and options["backend"] == "gpu"
        assert (options["seeds"], options["area_min"], options["area_max"]) == ("threshold", 3, 200)
        assert image["rep"] == pytest.approx(meta["outline_cd_px"] * 21, abs=1e-9)
        assert image["rep_converged"] == meta["converged"] and image["rep_iterations"] == meta["fit_iterations"]
        # the best end fits at least as well as the report's start run to its end, and is one of the listed ends
        assert image["best_leftover"] <= image["end_leftover"] + 1e-9
        assert image["ends"][0]["cd"] == image["best"] and sum(g["starts"] for g in image["ends"]) == 10
        if image["rep_converged"]:  # a fit that stopped by its tolerance is at its end
            assert abs(image["end"] - image["rep"]) < 0.05
