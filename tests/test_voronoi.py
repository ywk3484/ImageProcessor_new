import numpy as np
import pytest


def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


@skipno_gpu
def test_voronoi_two_seeds():
    """Two seeds split a 20x20 image vertically."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[5.0, 10.0], [15.0, 10.0]])  # x, y
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    assert labels.shape == (20, 20)
    assert labels.dtype == np.int32
    # pixel (col=2, row=10) is closer to seed 0 (x=5)
    assert labels[10, 2] == 0
    # pixel (col=17, row=10) is closer to seed 1 (x=15)
    assert labels[10, 17] == 1


@skipno_gpu
def test_voronoi_staggered_grid():
    """Staggered 2x3 grid: Voronoi cells should tile correctly."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    # Row 0: x=10, 30, 50.  Row 1: x=20, 40 (staggered)
    seeds = np.array([
        [10.0, 10.0], [30.0, 10.0], [50.0, 10.0],
        [20.0, 30.0], [40.0, 30.0],
    ])
    labels = compute_voronoi_labels_gpu(seeds, (40, 60), device=0)
    assert labels.shape == (40, 60)
    # Each seed should own at least some pixels
    unique = np.unique(labels)
    assert len(unique) == 5


@skipno_gpu
def test_voronoi_single_seed():
    """Single seed: entire image assigned to it."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[10.0, 10.0]])
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    assert np.all(labels == 0)


@skipno_gpu
def test_voronoi_cell_areas():
    """Cell areas should sum to total image area."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    areas = np.bincount(labels.ravel(), minlength=4)
    assert areas.sum() == 20 * 20
    # Roughly equal areas for symmetric seeds
    assert np.all(areas > 50)  # each should be ~100
