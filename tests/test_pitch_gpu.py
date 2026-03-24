import numpy as np
import pytest
from subpx.pitch import estimate_pitch_lines
from subpx.backends import is_gpu_available


def test_pitch_result_has_backend_field():
    """PitchResult should have a backend field set to 'cpu' for CPU path."""
    pts = np.array([
        [0.0, 0.0], [10.0, 0.1], [20.0, -0.1],
        [0.0, 10.0], [10.0, 10.1], [20.0, 9.9],
    ])
    res = estimate_pitch_lines(pts, line_axis="row", tol=1.0)
    assert hasattr(res, "backend")
    assert res.backend == "cpu"


def test_pitch_lines_cpu_explicit_backend():
    """CPU backend with explicit parameter works and sets backend field."""
    pts = np.array([
        [0.0, 0.0], [10.0, 0.1], [20.0, -0.1],
        [0.0, 10.0], [10.0, 10.1], [20.0, 9.9],
    ])
    res = estimate_pitch_lines(pts, line_axis="row", tol=1.0, backend="cpu")
    assert res.backend == "cpu"
    assert res.values.size >= 1


def test_pitch_lines_gpu_no_cupy_raises():
    """backend='gpu' without CuPy raises RuntimeError."""
    if is_gpu_available():
        pytest.skip("CuPy is available; cannot test missing-CuPy error")
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    with pytest.raises(RuntimeError, match="CuPy"):
        estimate_pitch_lines(pts, backend="gpu")


@pytest.mark.skipif(not is_gpu_available(), reason="CuPy not available")
class TestPitchLinesGPU:

    def test_smoke_regular_grid(self):
        """5x5 regular grid with pitch=10 in both axes."""
        xs = np.arange(5) * 10.0
        ys = np.arange(5) * 10.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        res = estimate_pitch_lines(pts, line_axis="row", tol=1.5, backend="gpu")
        assert res.backend == "gpu"
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 3
        np.testing.assert_allclose(valid, 10.0, atol=0.1)

    def test_smoke_with_noise(self):
        """Grid with small positional noise — pitch still accurate."""
        rng = np.random.default_rng(42)
        xs = np.arange(10) * 10.0
        ys = np.arange(10) * 10.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        pts += rng.normal(0, 0.2, pts.shape)
        res = estimate_pitch_lines(pts, line_axis="row", tol=2.0, backend="gpu")
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 5
        np.testing.assert_allclose(valid, 10.0, atol=0.5)

    def test_cpu_gpu_agreement(self):
        """CPU and GPU produce close pitch values on same input."""
        rng = np.random.default_rng(123)
        xs = np.arange(20) * 5.0
        ys = np.arange(15) * 5.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        pts += rng.normal(0, 0.1, pts.shape)

        cpu = estimate_pitch_lines(pts, line_axis="row", tol=2.0, backend="cpu")
        gpu = estimate_pitch_lines(pts, line_axis="row", tol=2.0, backend="gpu")

        cpu_valid = cpu.values[np.isfinite(cpu.values)]
        gpu_valid = gpu.values[np.isfinite(gpu.values)]
        assert abs(cpu_valid.size - gpu_valid.size) <= 2
        np.testing.assert_allclose(
            np.median(cpu_valid), np.median(gpu_valid), atol=0.1
        )

    def test_noisy_row_filtered(self):
        """A row with irregular spacing gets filtered by pitch_var_max."""
        pts_clean = []
        for row_y in [0.0, 10.0, 20.0]:
            for x in np.arange(10) * 5.0:
                pts_clean.append([x, row_y])
        pts_noisy = [[0.0, 30.0], [3.0, 30.0], [20.0, 30.0], [21.0, 30.0], [45.0, 30.0]]
        pts = np.array(pts_clean + pts_noisy)
        res = estimate_pitch_lines(
            pts, line_axis="row", tol=2.0, backend="gpu", pitch_var_max=0.25
        )
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 2
        np.testing.assert_allclose(valid, 5.0, atol=0.5)

    def test_column_axis(self):
        """line_axis='col' clusters by x, measures along y."""
        xs = np.arange(5) * 10.0
        ys = np.arange(8) * 7.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        res = estimate_pitch_lines(pts, line_axis="col", tol=1.5, backend="gpu")
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 3
        np.testing.assert_allclose(valid, 7.0, atol=0.1)

    def test_empty_input(self):
        """Empty input returns empty result."""
        pts = np.zeros((0, 2), dtype=np.float64)
        res = estimate_pitch_lines(pts, backend="gpu")
        assert res.values.size == 0
        assert res.backend == "gpu"

    def test_single_row(self):
        """Single row of points — returns one pitch value."""
        pts = np.array([[i * 8.0, 0.0] for i in range(10)])
        res = estimate_pitch_lines(pts, line_axis="row", tol=1.5, backend="gpu")
        valid = res.values[np.isfinite(res.values)]
        assert valid.size == 1
        np.testing.assert_allclose(valid[0], 8.0, atol=0.1)

    def test_single_point_per_row(self):
        """Rows with single points produce NaN pitch."""
        pts = np.array([
            [0.0, 0.0], [0.0, 10.0], [0.0, 20.0],
        ])
        res = estimate_pitch_lines(
            pts, line_axis="row", tol=1.5, backend="gpu", min_points_per_line=2
        )
        assert np.all(~np.isfinite(res.values)) or res.values.size == 0

    def test_meta_fields(self):
        """Result meta contains expected keys."""
        xs = np.arange(5) * 10.0
        ys = np.arange(3) * 10.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        res = estimate_pitch_lines(pts, line_axis="row", tol=1.5, backend="gpu")
        assert "line_centers_perp" in res.meta
        assert "line_counts" in res.meta
        assert "line_ids_per_point" in res.meta
        assert "valid_mask" in res.meta
        assert res.meta["valid_mask"].dtype == bool

    def test_centered_coordinates(self):
        """Grid centered at origin with negative coordinates."""
        xs = (np.arange(10) - 5) * 10.0  # -50 to 40
        ys = (np.arange(10) - 5) * 10.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        res = estimate_pitch_lines(pts, line_axis="row", tol=2.0, backend="gpu")
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 5
        np.testing.assert_allclose(valid, 10.0, atol=0.1)

    def test_large_grid_performance(self):
        """100x100 grid completes and produces correct results."""
        xs = np.arange(100) * 5.0
        ys = np.arange(100) * 5.0
        xx, yy = np.meshgrid(xs, ys)
        pts = np.column_stack([xx.ravel(), yy.ravel()])
        pts += np.random.default_rng(0).normal(0, 0.1, pts.shape)
        res = estimate_pitch_lines(pts, line_axis="row", tol=2.0, backend="gpu")
        valid = res.values[np.isfinite(res.values)]
        assert valid.size >= 50
        np.testing.assert_allclose(valid, 5.0, atol=0.5)
