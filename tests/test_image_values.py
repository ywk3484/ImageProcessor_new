"""scripts/image_values.py: what the images of a folder hold, and the outline fit's gap in grey levels.

The script measures; it changes nothing in the library. Its helpers are tested without a GPU, the whole script on a
folder of two small generated images (the same image stored in 8 and in 16 bit) with one.
"""

import json
import re

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from scripts import image_values as values
from subpx.synthetic_arrays import render_hole_array


def test_file_values_tell_the_content_from_the_file_type():
    rng = np.random.default_rng(0)
    eight = rng.integers(10, 200, size=(60, 80)).astype(np.uint8)
    v = values.file_values(eight)
    assert (v["dtype"], v["low"], v["high"], v["step"]) == ("uint8", eight.min(), eight.max(), 1)
    assert v["levels"] == len(np.unique(eight)) and v["top"] == 0.0 and v["zero"] == 0.0
    # the same content in a 16 bit file: as many levels, 257 apart
    wide = values.file_values(eight.astype(np.uint16) * 257)
    assert (wide["dtype"], wide["levels"], wide["step"]) == ("uint16", v["levels"], 257)
    # a 12 bit camera in a 16 bit file, read as int32 (as some PNG readers do)
    assert values.file_values((eight.astype(np.int32) + 1000) * 16)["step"] == 16
    # values cut off at the top of the file type, and at zero
    cut = eight.copy()
    cut[:6] = 255
    cut[-3:] = 0
    v = values.file_values(cut)
    assert v["top"] == pytest.approx(6 / 60) and v["zero"] == pytest.approx(3 / 60)
    assert values.file_values(np.minimum(eight.astype(np.int32) * 400, 65535))["top"] > 0  # int32 holding 16 bit values
    assert values.file_values(np.full((4, 4), 7, np.uint8))["step"] == 1  # one value only
    floats = values.file_values(eight / 3.0)
    assert floats["step"] is None and floats["top"] == 0.0


def test_gap_is_the_pixel_weighted_distance_behind_the_leftover():
    rng = np.random.default_rng(1)
    count = np.zeros((9, 9))
    count[1:8, 1:8] = rng.integers(50, 4000, size=(7, 7))  # unequally filled bins, an empty rim
    model = np.where(count > 0, 20.0 + 60.0 * np.exp(-((np.indices((9, 9)) - 4.0) ** 2).sum(axis=0) / 6.0), np.nan)
    stack = model + np.where(count > 0, rng.normal(0.0, 0.3, (9, 9)), 0.0)
    noise = 2.5
    filled = count > 0
    chi2 = float(np.sum(count[filled] / noise ** 2 * (stack[filled] - model[filled]) ** 2))
    g = values.gap_values(stack, count, noise, chi2 / filled.sum())
    weighted = np.sqrt(np.sum(count[filled] * (stack[filled] - model[filled]) ** 2) / count[filled].sum())
    assert g["gap"] == pytest.approx(weighted, rel=1e-9)
    assert g["alone"] == pytest.approx(noise / np.sqrt(count[filled].mean()), rel=1e-12)
    assert g["hole"] == pytest.approx(np.nanmax(stack) - np.nanmin(stack))
    assert values.gap_values(stack, count, noise, 1.0)["gap"] == pytest.approx(g["alone"])  # leftover 1: at the noise
    # the image's unit cancels: 257 times the values, the same gap in percent of the hole
    wide = values.gap_values(stack * 257, count, noise * 257, chi2 / filled.sum())
    assert wide["gap"] / wide["hole"] == pytest.approx(g["gap"] / g["hole"], rel=1e-9)


def _image(dtype="uint16", low=3319.0, high=23412.0, levels=18205, step=1, top=0.0, zero=0.0, **fit):
    return {"dtype": dtype, "low": low, "high": high, "levels": levels, "step": step, "top": top, "zero": zero, **fit}


def test_folder_line_is_short_ascii_and_in_grey_steps():
    fit = dict(fitted=True, holes=27434, per_bin=1620.0, leftover=1.36)
    results = [_image(noise=299.0, hole=14963.0, gap=9.81, alone=7.41, **fit),
               _image(low=3251.0, high=23544.0, levels=17175, noise=294.0, hole=14767.0, gap=6.10, alone=5.47, **fit),
               _image(fitted=False, reason="fewer than 10 holes"), {"reason": "cannot identify image file"}]
    line = values.folder_line(results)
    assert line.isascii() and len(line) <= 150
    assert line == ("values N=3 | uint16 3251-23544 | levels 17175-18205 step 1 | top 0.00% zero 0.00% | "
                    "noise 296 hole 14865 | gap 6.10-9.81 (6.44) = 0.04-0.07%")
    # 8 bit content in a 16 bit file: everything in its grey steps of 257
    scaled = [_image(low=3341.0, high=23387.0, levels=79, step=257, noise=1.2 * 257, hole=58.2 * 257, gap=0.135 * 257,
                     alone=0.117 * 257, **fit)]
    assert values.folder_line(scaled) == ("values N=1 | uint16 3341-23387 | levels 79 step 257 | top 0.00% zero 0.00% | "
                                          "noise 1.20 hole 58.2 | gap 0.135-0.135 (0.117) = 0.23-0.23%")
    # the longest line that can occur stays short enough to type
    long = [_image(dtype=d, low=0.0, high=65535.0, levels=n, step=s, top=0.1234, zero=0.0567, noise=12345.6,
                   hole=54321.9, gap=g, alone=123.4, **fit) for d, n, s, g in (("uint16", 65536, 1, 1234.5), ("int32", 60000, 2, 2345.6))]
    assert len(values.folder_line(long)) <= 150 and values.folder_line(long).isascii()
    assert values.folder_line([_image(), _image(dtype="uint8", low=13.0, high=91.0, levels=79)]) == (
        "values N=2 | uint16/uint8 13-23412 | levels 79-18205 step 1 | top 0.00% zero 0.00%")  # --no-fit
    assert values.folder_line(results[2:3]).endswith("zero 0.00% | no fit")
    assert values.folder_line(results[3:]) == "values N=0: no image was read"
    assert values.image_line("a.png", results[3]) == "a.png: not read (cannot identify image file)"
    assert "not fitted (fewer than 10 holes)" in values.image_line("a.png", results[2])
    assert values.num(float("nan")) == "-" and [values.num(v) for v in (1234.5, 12.34, 1.234, 0.1234)] == ["1234", "12.3", "1.23", "0.123"]


def _folder(tmp_path):
    from PIL import Image

    folder = tmp_path / "in"
    folder.mkdir()
    page = render_hole_array((120, 170), hole="circle", size_nm=(100, 100), pitch_nm=(210, 210), layout="square",
                             cd_sigma_nm=3.0, seed=5, backend="cpu")
    eight = page.image.astype(np.uint8)
    Image.fromarray(eight).save(folder / "a_08bit.png")
    Image.fromarray(eight.astype(np.uint16) * 257).save(folder / "b_16bit.png")
    return folder, eight


def test_the_files_half_needs_no_gpu(tmp_path, capsys):
    folder, eight = _folder(tmp_path)
    saved_to = tmp_path / "values.json"
    assert values.main([str(folder), "--no-fit", "--output", str(saved_to)]) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    levels = len(np.unique(eight))
    assert last == (f"values N=2 | uint16/uint8 {eight.min()}-{int(eight.max()) * 257} | levels {levels} step 1-257 | "
                    f"top {100 * np.mean(eight == 255):.2f}% zero 0.00%")
    saved = json.loads(saved_to.read_text(encoding="utf-8"))
    assert saved["line"] == last and [image["image"] for image in saved["images"]] == ["a_08bit.png", "b_16bit.png"]
    assert values.main([str(folder / "a_08bit.png"), "--no-fit", "--output", str(saved_to)]) == 0  # one file
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("values N=1 | uint8 ")


@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")


def test_the_same_image_in_8_and_16_bit_reads_the_same(gpu, tmp_path, capsys):
    import subpx.outline as outline

    folder, _ = _folder(tmp_path)
    original = outline.fit_outline_fast
    saved_to = tmp_path / "values.json"
    assert values.main([str(folder), "--output", str(saved_to)]) == 0
    assert outline.fit_outline_fast is original  # the tap on the fit is taken off again
    printed = capsys.readouterr().out.strip().splitlines()
    last = printed[-1]
    assert all(line.isascii() for line in printed) and len(last) <= 150
    number = r"\d+(\.\d+)?"
    assert re.fullmatch(rf"values N=2 \| uint16/uint8 \d+-\d+ \| levels \d+ step 1-257 \| top \d+\.\d\d% zero 0\.00% \| "
                        rf"noise {number} hole {number} \| gap {number}-{number} \({number}\) = \d+\.\d\d-\d+\.\d\d%", last), last
    a, b = json.loads(saved_to.read_text(encoding="utf-8"))["images"]
    assert a["fitted"] and b["fitted"] and (a["step"], b["step"]) == (1, 257)
    assert a["gap"] >= 0.999 * a["alone"] * min(1.0, a["leftover"] ** 0.5) and a["hole"] > 20 * a["noise"] > 0
    # in grey steps the 16 bit copy of the 8 bit image is the same image (a small image: the fit repeats to a few percent)
    sa, sb = values.in_steps(a), values.in_steps(b)
    assert sb["hole"] == pytest.approx(sa["hole"], rel=0.01) and sb["noise"] == pytest.approx(sa["noise"], rel=0.05)
    assert sb["gap"] == pytest.approx(sa["gap"], rel=0.5)
