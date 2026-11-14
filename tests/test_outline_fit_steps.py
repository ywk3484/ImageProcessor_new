"""scripts/outline_fit_steps.py: every step of outline_fit_fast on a folder's images, as a page in the browser.

The script shows; it changes nothing in the library. What it cuts the record down to and how it builds the page's
figures are tested without a GPU; the whole page on a folder of generated images with one. The page itself (the
drawing in the browser) was looked at on generated folders with planted faults: docs/outline_fit_steps.md.
"""

import base64
import json

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("cv2")

from scripts import outline_fit_steps as steps
from subpx._outline.model import Template
from subpx.synthetic_arrays import render_hole_array


def array_of(panel, store=None):
    text = store[panel["d"]] if store is not None else panel["d"]
    return np.frombuffer(base64.b64decode(text), dtype="<f4").reshape(panel["h"], panel["w"])


# ------------------------------------------------------------------------------------------- no GPU needed
def test_numbers_are_written_plainly():
    assert steps.fmt(26412, 0) == "26,412" and steps.fmt(98.3149, 2) == "98.31" and steps.fmt(float("nan")) == "-"
    assert steps.sig(0.076412) == "0.0764" and steps.sig(1234.5) == "1,234" and steps.sig(1.0234) == "1.02"
    assert steps.sig(0) == "0" and steps.sig(None) == "-" and "e" not in steps.sig(1.2e-7)
    assert steps.share(3, 200) == "1.50 %" and steps.share(1, 0) == "-"
    assert steps.settled([("round", 10, False), ("full", 12, True)]) == "12, settled"
    assert steps.settled([("full", 30, False)]) == "30, cut off at its limit"
    assert steps.moved({"drift": -0.0001, "tail": 4}, "nm") == "+0.000 nm in its last 4 iterations"
    assert steps.moved({"drift": -0.4521, "tail": 1}, "nm") == "-0.452 nm in its last iteration"


def test_blocks_hold_the_median_of_their_holes():
    xy = np.array([[1.0, 1.0], [2.0, 3.0], [8.0, 1.0], [3.0, 2.0], [9.5, 9.5], [0.0, 9.0]])
    values = np.array([5.0, 1.0, 7.0, 3.0, np.nan, 2.0])
    got = steps.blocks(xy, values, (10, 10), 5)
    assert got.shape == (2, 2) and got[0, 0] == 3.0 and got[0, 1] == 7.0 and got[1, 0] == 2.0 and np.isnan(got[1, 1])
    np.testing.assert_array_equal(steps.blocks(xy, None, (10, 10), 5, "count"), [[3, 1], [1, 1]])
    assert steps.blocks(xy[:0], values[:0], (10, 10), 5).shape == (2, 2)


def test_the_density_of_a_regular_array_is_even_and_shows_missing_holes():
    pitch = 10.3  # not a whole number of pixels, and the spacing handed in is off, as the method's own is
    gx, gy = np.meshgrid(np.arange(3.0, 800, pitch), np.arange(4.0, 600, pitch))
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    full, first, step = steps.density(xy, (600, 800), 10.13, 30)
    # no stripes of its own [inside 0.0002; 0.007 with the values next to the frame; 0.04 when every hole is put
    # into the one cell it falls in]
    assert np.nanstd(full[3:-3, 3:-3]) < 0.002 and np.nanstd(full) < 0.015 and abs(np.nanmedian(full) - 1) < 0.005
    assert first == pytest.approx(3 * 5.065 + 5.065 / 2 - 0.5) and step == pytest.approx(6 * 5.065)
    # within two spacings of the frame the smoothing is one-sided: no value there, a value everywhere else
    y, x = first + 0.5 + step * np.arange(full.shape[0]), first + 0.5 + step * np.arange(full.shape[1])
    near = ((y < 20.26) | (y > 600 - 20.26))[:, None] | ((x < 20.26) | (x > 800 - 20.26))[None, :]
    np.testing.assert_array_equal(np.isnan(full), near)
    assert near[0].all() and not near[1:-1, 1:-1].any()
    gone = (np.abs(xy[:, 0] - 400) < 40) & (np.abs(xy[:, 1] - 300) < 40)  # an area of 8 x 8 holes not found
    holed, _, _ = steps.density(xy[~gone], (600, 800), 10.13, 30)
    row, col = np.unravel_index(np.nanargmin(holed), holed.shape)
    assert np.nanmin(holed) < 0.3 and abs(first + col * step - 400) < 30 and abs(first + row * step - 300) < 30


def test_pictures_side_by_side_share_one_colour_scale():
    you, copy = dict(label=steps.YOU), dict(label=steps.COPY)
    a = np.arange(12.0).reshape(3, 4)
    b = a * 0.5 + 1.0
    b[0, 0] = np.nan
    f = steps.heat("4a", "Average hole", [you, copy], lambda s: a if s is you else b, "grey", "grey levels",
                   grid=(-1.0, 0.25, -2.0, 0.25, "px"), minus=True, robust=False,
                   marks=lambda s: dict(points=[[0.0, 0.0]], rects=[[0, 0, 1, 1]], lines=[np.zeros((5, 2))]))
    assert [p["label"] for p in f["panels"]] == [steps.YOU, steps.COPY, f"{steps.YOU} minus {steps.COPY}"]
    one, two, minus = f["panels"]
    assert (one["lo"], one["hi"]) == (two["lo"], two["hi"]) == (0.0, 11.0)
    np.testing.assert_array_equal(array_of(one), a)
    np.testing.assert_array_equal(array_of(two), b.astype("<f4"))
    assert minus["palette"] == "diverging" and minus["lo"] == -minus["hi"] and np.isnan(array_of(minus)[0, 0])
    assert one["x0"] == -1.0 and one["dy"] == 0.25 and one["axis"] == "px"
    assert "points" in one and len(one["lines"]) == 1 and one["rects"] == [[0.0, 0.0, 1.0, 1.0]]
    json.dumps(f, allow_nan=False)  # nothing that is not a number goes into the page as a number
    # without the copy, or with another shape, there is no difference picture; nothing to show: no figure
    assert len(steps.heat("x", "", [you], lambda s: a, "grey", "", minus=True)["panels"]) == 1
    assert len(steps.heat("x", "", [you, copy], lambda s: a if s is you else a[:2], "grey", "", minus=True)["panels"]) == 2
    assert steps.heat("x", "", [you, copy], lambda s: None, "grey", "") is None
    assert steps.span([np.array([np.nan])], "grey") == (0.0, 1.0)
    assert steps.span([np.array([0.5, 1.5])], "diverging", center=1.0, robust=False) == (0.5, 1.5)
    assert steps.heat("x", "", [you], lambda s: a, "diverging", "", limits=(0.8, 1.2))["panels"][0]["lo"] == 0.8


def test_the_same_array_is_written_once():
    you = dict(label=steps.YOU)
    a, b = np.ones((2, 2)), np.zeros((2, 2))
    figures = [steps.heat("1b", "", [you], lambda s: a, "grey", ""), steps.heat("2a", "", [you], lambda s: a, "grey", ""),
               steps.heat("3b", "", [you], lambda s: b, "grey", ""), steps.table("9z", "", ["a"], [[1]])]
    store = steps.pooled(figures)
    assert len(store) == 2 and [f["panels"][0]["d"] for f in figures[:3]] == [0, 0, 1]
    np.testing.assert_array_equal(array_of(figures[2]["panels"][0], store), b)


def test_graphs_and_histograms_hold_both_sides():
    you, copy = dict(label=steps.YOU, v=np.array([1.0, 2.0, 2.0, np.nan])), dict(label=steps.COPY, v=np.array([2.0, 3.0]))
    h = steps.hist("5e", "Leftover", "grey levels", [you, copy], lambda s: s["v"], bins=4,
                   vlines=[(5.0, "dropped above"), (float("nan"), "no such line")], count="holes")
    assert [s["name"] for s in h["series"]] == [steps.YOU, steps.COPY] and len(h["edges"]) == 5
    assert sum(h["series"][0]["counts"]) == 3 and sum(h["series"][1]["counts"]) == 2
    assert h["edges"][-1] > 5.0 and [v["label"] for v in h["vlines"]] == ["dropped above"] and h["count"] == "holes"
    assert steps.hist("x", "", "", [you], lambda s: np.array([np.nan])) is None
    g = steps.lines("9a", "CD", "iteration", "CD (nm)", [(steps.YOU, 1, [0, 1, 2], [98.0, float("nan"), 98.4])],
                    vlines=[(30, "limit of 30")], log=True)
    assert g["series"][0]["y"] == [98.0, None, 98.4] and g["vlines"] == [{"x": 30.0, "label": "limit of 30"}] and g["log"]
    assert steps.lines("x", "", "", "", [(steps.YOU, 1, [], [])]) is None
    json.dumps([h, g], allow_nan=False)


def test_a_ninth_of_the_image_is_compared_after_matching_brightness_and_level():
    whole = np.random.default_rng(0).normal(50, 10, (7, 7))
    stacks = np.stack([1.2 * whole + 3.0] * 9)  # brighter and higher, the same hole: nothing is left
    stacks[4] = stacks[4] + np.where(np.arange(49).reshape(7, 7) == 24, 5.0, 0.0)  # the middle part: one bin off
    stacks[8, 0, 0] = np.nan
    pixels = np.full((9, 7, 7), 20.0)
    mosaic, worst = steps.parts_apart(dict(stacks=stacks, pixels=pixels, rows=3, cols=3), whole, gap=2)
    assert mosaic.shape == (3 * 7 + 4, 3 * 7 + 4) and np.isnan(mosaic[7, :]).all() and np.isnan(mosaic[-7, -7])
    assert np.nanmax(np.abs(mosaic[:7, :7])) < 1e-9  # the first part
    middle = mosaic[9:16, 9:16]
    assert middle[3, 3] == pytest.approx(5.0, abs=0.3) and worst == pytest.approx(np.sqrt(np.mean(middle ** 2)), rel=1e-9)


@pytest.mark.parametrize("blur, count", [("gauss2", 13), ("lens", 11)])
def test_every_fitted_number_is_named_with_its_limits(blur, count):
    t = Template(upsample=8, size_px=24, blur=blur)
    lo, hi = t.bounds()
    theta = t.start(2.4, 1.3)
    theta[2], theta[-1] = lo[2], hi[-1]
    rows = steps.parameter_rows(theta, t, 21.0, "nm")
    assert len(rows) == count and len({r[0] for r in rows}) == count
    assert rows[0][1] == "50.40 nm" and rows[0][2] == "4.20 nm" and rows[0][3] == "168.00 nm" and rows[0][4] == ""
    assert rows[2][4] == "at its lowest" and rows[2][1] == "1.30" and rows[-1][4] == "at its highest"
    assert rows[3][1] == "+0.0 deg" and rows[3][3] == "+45.0 deg"


def fake_run(image, picture, pixel_noise, inverted=False):
    return dict(image=image, values=steps.file_values(image),
                trace=dict(redrawn=dict(picture=picture.astype(np.float32), pixel_noise=pixel_noise),
                           image=dict(inverted=inverted, top=float(image.max()), own_seeds=False)))


def test_the_copy_is_stored_like_the_file():
    rng = np.random.default_rng(1)
    picture = rng.uniform(20, 200, (300, 400))
    image = np.clip(np.rint(picture + rng.normal(0, 3, picture.shape)), 0, 255).astype(np.uint8)
    copy, grain = steps.redrawn_copy(fake_run(image, picture, 3.0))
    assert copy.dtype == np.uint8 and copy.shape == image.shape
    assert grain == pytest.approx(np.sqrt(9 - 1 / 12))  # rounding to whole grey levels adds its twelfth again
    assert np.std(copy.astype(float) - picture) == pytest.approx(3.0, rel=0.02)
    # 8 bit content in a 16 bit file: the copy keeps the grey step and the type
    wide = image.astype(np.uint16) * 257
    copy16, grain16 = steps.redrawn_copy(fake_run(wide, picture * 257, 3.0 * 257))
    assert copy16.dtype == np.uint16 and steps.file_values(copy16)["step"] == 257
    assert grain16 == pytest.approx(257 * np.sqrt(9 - 1 / 12))
    # a quiet image: no noise is added below what the rounding makes itself; tops beyond the type's range are cut
    quiet, none = steps.redrawn_copy(fake_run(image, picture + 100.0, 0.1))
    assert none == 0.0 and quiet.max() == 255
    off = quiet.astype(float) - np.clip(np.rint(picture + 100.0), 0, 255)  # values half-way between two steps apart
    assert np.abs(off).max() <= 1 and np.mean(off != 0) < 1e-3
    # dark holes: the picture is of the image turned over; the copy is turned back
    dark, _ = steps.redrawn_copy(fake_run(255 - image, picture, 0.1, inverted=True))
    assert abs(float(np.mean(dark.astype(float) - (float((255 - image).max()) - picture)))) < 0.6
    # a float image is not rounded; a fit that did not get that far has no copy
    as_float, grain_f = steps.redrawn_copy(fake_run(image.astype(np.float32), picture, 3.0))
    assert as_float.dtype == np.float32 and grain_f == 3.0 and not np.allclose(as_float, np.rint(as_float))
    assert steps.redrawn_copy(dict(trace={}, image=image)) == (None, None)


# ------------------------------------------------------------------------------------------------- GPU
@pytest.fixture(scope="module")
def gpu():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except Exception:
        pytest.skip("CUDA unavailable")


def generated(seed, shape=(300, 400), **change):
    """A generated image that is what the method assumes (one size, a Gaussian blur, even noise of 1.2, 8 bit)."""
    kw = dict(hole="circle", size_nm=(100, 100), pitch_nm=(216.3, 216.3), layout="square", blur_nm=65, lens_nm=0,
              photons_per_level=None, read_noise=1.2, cd_sigma_nm=0, row_shift_nm=0, backend="cpu")
    kw.update(change)
    return render_hole_array(shape, seed=seed, **kw)


def page_of(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("=") + 1:text.rindex(";")])


@pytest.fixture(scope="module")
def folder(gpu, tmp_path_factory):
    """A folder of two generated images, a file that is no image and an image of three holes, and its page."""
    from PIL import Image

    source = tmp_path_factory.mktemp("images")
    Image.fromarray(generated(1).image).save(source / "a one.png")
    Image.fromarray(generated(2).image).save(source / "b.png")
    (source / "c broken.png").write_text("not an image")
    few = np.full((120, 160), 12, dtype=np.uint8)
    for y, x in ((30, 40), (60, 100), (90, 50)):
        few[y - 2:y + 3, x - 2:x + 3] = 90
    Image.fromarray(few).save(source / "d empty.png")
    output = tmp_path_factory.mktemp("page")
    return source, output


def test_the_page_of_a_folder(folder, capsys):
    source, output = folder
    assert steps.main([str(source), "--output", str(output), "--pixel-size", "21", "--unit", "nm", "--no-open"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert out.isascii() and lines[-1].startswith("Page: ") and "index.html" in lines[-1]
    assert lines[0].startswith("[1/4] a one.png: CD ") and "copy: CD " in lines[0] and "[3/4] c broken.png: failed: " in out
    assert "[4/4] d empty.png: not fitted (" in out

    html = (output / "index.html").read_text(encoding="utf-8")
    assert "__ENTRIES__" not in html and "__TITLE__" not in html and "__INFO__" not in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html  # works offline
    entries = json.loads(html[html.index("const ENTRIES=") + 14:html.index(",INFO=")])
    assert [e["id"] for e in entries] == ["a_one", "b", "c_broken", "d_empty"]
    assert entries[2]["error"].startswith("failed: ") and entries[2]["data"] == "" and not entries[0]["error"]
    assert entries[3]["reason"] and not (output / "data" / "c_broken.js").exists()

    page = page_of(output / "data" / "a_one.js")
    assert page["sides"] == [steps.YOU, steps.COPY] and page["name"] == "a one.png" and not page["reason"]
    assert [s["n"] for s in page["steps"]] == list(range(1, 11)) and [r[0] for r in page["summary"]] == list(range(1, 11))
    ids = [f["id"] for s in page["steps"] for f in s["figures"]]
    assert len(ids) == len(set(ids)) and {"1a", "2d", "4a", "5a", "6d", "7a", "8e", "9d", "9e", "9z", "10a"} <= set(ids)
    for step in page["steps"]:
        assert step["does"] and step["look"] and step["numbers"] and step["figures"]
        for row in step["numbers"]:
            assert len(row) == 3 and all(isinstance(c, str) for c in row)
        for f in step["figures"]:
            assert f["title"] and f["id"].startswith(str(step["n"]))
            for p in f.get("panels", ()):  # every picture: its array is in the store, finite limits, a label
                assert array_of(p, page["store"]).shape == (p["h"], p["w"]) and p["hi"] > p["lo"] and p["label"]
            if f["kind"] == "heat":  # your image first; pictures of the two sides on one scale
                labels = [p["label"] for p in f["panels"]]
                assert labels[0] == steps.YOU and (len(labels) == 1 or labels[1] == steps.COPY)
                if len(labels) > 1:
                    assert (f["panels"][0]["lo"], f["panels"][0]["hi"]) == (f["panels"][1]["lo"], f["panels"][1]["hi"])
            if f["kind"] in ("lines", "hist"):
                assert [s["slot"] for s in f["series"]] == [1, 2]
    numbers = {label: (real, copy) for s in page["steps"] for label, real, copy in s["numbers"]}
    assert numbers["how the file stores it"] == ("uint8", "uint8")
    assert numbers["window"] == ("+-8 px (17 x 17 px)",) * 2
    # an image that is what the method assumes: its copy looks the same in numbers [1.02 and 1.05 x the noise]
    final = [s for s in page["steps"] if s["n"] == 9][0]
    times = [r for r in final["numbers"] if "as a multiple of that" in r[0]][0]
    assert 0.8 < float(times[1].split()[0]) < 1.3 and 0.8 < float(times[2].split()[0]) < 1.3
    drawn = numbers["the copy was drawn with this CD, and read back"]
    assert abs(float(drawn[0].split()[0]) - float(drawn[1].split()[0])) < 0.6  # [0.1 to 0.4 nm with 1,000 holes]
    assert 1.15 < float(numbers["noise from pixel to pixel"][0].split()[0]) < 1.32  # [1.22: the 1.2 put in + rounding]

    empty = page_of(output / "data" / "d_empty.js")  # too few holes: the steps the calculation reached, no copy
    assert empty["reason"] and empty["sides"] == [steps.YOU] and [s["n"] for s in empty["steps"]] == [1]
    assert empty["summary"][0][2].startswith("uint8, ")

    everything = (output / "data" / "all.js").read_text(encoding="utf-8")
    everything = json.loads(everything[everything.index("=") + 1:everything.rindex(";")])
    assert everything["headline"].startswith("2 of 3 images fitted; CD ")
    table, graph = everything["figures"][0], everything["figures"][1]
    assert table["id"] == "A1" and [r[0] for r in table["rows"]] == ["a one.png", "b.png", "d empty.png"]
    assert graph["id"] == "A2" and graph["emphasis"] and len(graph["series"]) == 2
    assert {"A3", "A4", "A5"} <= {f["id"] for f in everything["figures"]}
    for f in everything["figures"]:
        for p in f.get("panels", ()):
            assert array_of(p, everything["store"]).shape == (p["h"], p["w"])


def test_a_planted_fault_shows_where_it_should(gpu):
    """Two hole sizes in one image (the left half 100 nm, the right half 106 nm): the final fit does not notice
    (the average of two sizes fits like one size), the ninths of the image do."""
    small, large = generated(3, (600, 800), bits=None, read_noise=0.0), generated(3, (600, 800), bits=None,
                                                                                    read_noise=0.0, size_nm=(106, 106))
    both = small.clean.copy()
    both[:, 400:] = large.clean[:, 400:]
    image = np.clip(np.rint(both + np.random.default_rng(5).normal(0, 1.2, both.shape)), 0, 255).astype(np.uint8)
    options = dict(seeds="threshold", threshold="otsu", invert=False, area_min=3, area_max=200, pad=3, pixel_size=21.0,
                   unit="nm")
    page, row, text = steps.process(image, "two sizes.png", options, "gauss2", 21.0, "nm")
    assert text.isascii() and text.startswith("CD 103.")  # [103.12 nm: between the two sizes]
    summary = {r[0]: r for r in page["summary"]}
    worst = [float(cell.split("up to ")[1].split()[0]) for cell in summary[8][2:]]
    assert worst[0] > 1.2 * worst[1]  # [0.43 against 0.32 grey levels; alike (0.30, 0.30) without the fault]
    final = [float(cell.split("= ")[1].split()[0]) for cell in summary[9][2:]]
    assert max(final) < 1.3  # [1.03 and 1.03 x the noise]
    assert row["row"]["cd"].startswith("103.") and row["cd"] == pytest.approx(103.1, abs=0.5)
