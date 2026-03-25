"""Visualization utilities for subpx."""
from __future__ import annotations

import numpy as np


def draw_voronoi_boundaries(
    image: np.ndarray,
    centers_xy: np.ndarray,
    label_map: np.ndarray | None = None,
    color: float = 1.0,
    thickness: int = 1,
    backend: str = "auto",
) -> np.ndarray:
    """Overlay Voronoi partition boundaries on an image.

    Boundaries are pixels where the Voronoi label differs from at least
    one 4-connected neighbor.

    Parameters
    ----------
    image : (H, W) ndarray
        Background image to overlay on.
    centers_xy : (N, 2) ndarray
        Seed center positions [x, y].
    label_map : (H, W) ndarray, optional
        Pre-computed Voronoi label map. Computed from centers_xy if None.
    color : float
        Intensity value for boundary pixels.
    thickness : int
        Boundary line width (morphological dilation iterations).
    backend : str
        "auto", "cpu", or "gpu". Only used for Voronoi computation if
        label_map is not provided.

    Returns
    -------
    overlay : (H, W) ndarray
        Copy of image with Voronoi boundaries drawn.
    """
    img = np.asarray(image)
    overlay = img.copy().astype(np.float64)
    H, W = overlay.shape[:2]

    if label_map is None:
        from .backends import resolve_backend
        b = resolve_backend(backend)
        if b == "gpu":
            from ._gpu.voronoi import compute_voronoi_labels_gpu
            label_map = compute_voronoi_labels_gpu(centers_xy, (H, W))
        else:
            # CPU fallback: per-pixel nearest-seed (tiled to avoid O(H*W*N) memory)
            centers = np.asarray(centers_xy, dtype=np.float64)
            label_map = np.zeros((H, W), dtype=np.int32)
            CHUNK = 256  # process rows in chunks to limit memory
            for r0 in range(0, H, CHUNK):
                r1 = min(H, r0 + CHUNK)
                yy, xx = np.mgrid[r0:r1, :W]
                coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (chunk*W, 2)
                dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
                label_map[r0:r1] = np.argmin(dists, axis=1).reshape(r1 - r0, W).astype(np.int32)

    labels = np.asarray(label_map, dtype=np.int32)

    # Find boundary pixels: where label differs from any 4-connected neighbor
    boundary = np.zeros((H, W), dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :]  |= labels[1:, :]  != labels[:-1, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:]  |= labels[:, 1:]  != labels[:, :-1]

    # Thicken boundary if requested
    if thickness > 1:
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(boundary, iterations=thickness - 1)

    overlay[boundary] = color
    return overlay
