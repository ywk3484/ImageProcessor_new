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
