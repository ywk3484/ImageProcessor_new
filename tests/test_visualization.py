# tests/test_visualization.py
import numpy as np
import pytest


def test_draw_voronoi_boundaries_basic():
    """Should draw boundaries where labels change between 4-neighbors."""
    from subpx.visualization import draw_voronoi_boundaries
    img = np.zeros((20, 20), dtype=np.float64)
    centers = np.array([[5.0, 10.0], [15.0, 10.0]])
    overlay = draw_voronoi_boundaries(img, centers, color=1.0)
    assert overlay.shape == (20, 20)
    # Boundary should exist somewhere near the midline (col ~10)
    boundary_cols = np.where(overlay[10, :] > 0)[0]
    assert len(boundary_cols) > 0
    assert np.abs(np.mean(boundary_cols) - 10) < 2


def test_draw_voronoi_boundaries_with_precomputed():
    """Should use pre-computed label_map when provided."""
    from subpx.visualization import draw_voronoi_boundaries
    img = np.zeros((10, 10), dtype=np.float64)
    label_map = np.zeros((10, 10), dtype=np.int32)
    label_map[:, 5:] = 1
    centers = np.array([[2.5, 5.0], [7.5, 5.0]])
    overlay = draw_voronoi_boundaries(img, centers, label_map=label_map, color=1.0)
    # Boundary at col=5 (where label changes)
    assert overlay[5, 4] > 0 or overlay[5, 5] > 0
