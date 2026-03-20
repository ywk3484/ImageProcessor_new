import numpy as np

from maskproc import (
    detect_centers_tiled,
    connected_components_stats,
    fft_pitch_error,
    find_y_cluster_split_indices,
    residuals_vs_x_by_line,
)


def test_connected_components_cpu_smoke():
    m = np.zeros((8, 8), dtype=np.uint8)
    m[1:3, 1:3] = 1
    m[5:7, 5:7] = 1
    res = connected_components_stats(m, backend="cpu")
    assert res.num_labels == 3
    assert res.stats.shape[1] == 5


def test_fft_pitch_error_smoke():
    x = np.linspace(0, 100, 256)
    y = 0.5 * np.sin(2 * np.pi * x / 10.0)
    spec = fft_pitch_error(x, y)
    assert spec.frequency.size > 0
    assert spec.period.size == spec.frequency.size


def test_find_y_cluster_split_indices_smoke():
    y = np.array([0, 1, 2, 20, 21, 40], dtype=float)
    split_idx, bounds = find_y_cluster_split_indices(y, threshold=5)
    assert list(split_idx) == [3, 5]
    assert bounds == [(0, 3), (3, 5), (5, 6)]


def test_residuals_vs_x_by_line_smoke():
    centers = np.array([[0, 0], [10, 0], [20, 0], [0, 20], [10, 20], [20, 20]], dtype=float)
    cloud = residuals_vs_x_by_line(centers, line_axis="row", line_threshold=5, pitch=10)
    assert cloud["x"].shape[0] == 6


def test_detect_centers_tiled_smoke():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:13, 10:13] = 255
    img[40:43, 20:23] = 255
    res = detect_centers_tiled(img, backend="cpu", tile_h=24, overlap=4, area_min=1, area_max=30, refine="weighted")
    assert res.centers_xy.shape[1] == 2
