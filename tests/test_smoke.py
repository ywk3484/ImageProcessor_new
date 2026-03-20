import numpy as np

from maskproc.centers import filter_centers, dedupe_centers, detect_centers
from maskproc.pitch import estimate_pitch_lines
from maskproc.registration import estimate_shift


def test_filter_and_dedupe_smoke():
    pts = np.array([[1.0, 1.0], [1.2, 1.1], [20.0, 20.0]])
    f = filter_centers(pts, (100, 100), margin=0)
    d = dedupe_centers(f, eps=0.5)
    assert d.shape[1] == 2


def test_pitch_lines_smoke():
    pts = np.array([
        [0.0, 0.0], [10.0, 0.1], [20.0, -0.1],
        [0.0, 10.0], [10.0, 10.1], [20.0, 9.9],
    ])
    res = estimate_pitch_lines(pts, line_axis="row", tol=1.0)
    assert res.values.size >= 1


def test_detect_centers_cpu_smoke():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    img[30:34, 40:44] = 200
    res = detect_centers(img, backend="cpu", area_min=4, area_max=100, refine="weighted")
    assert res.centers_xy.shape[0] == 2


def test_estimate_shift_cpu_smoke():
    a = np.zeros((64, 64), dtype=np.float32)
    a[20:30, 25:35] = 1.0
    b = np.roll(np.roll(a, 3, axis=0), -4, axis=1)
    res = estimate_shift(a, b, backend="cpu")
    assert res.shift_yx.shape == (2,)
