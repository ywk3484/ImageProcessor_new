
from __future__ import annotations

import numpy as np


def image_hw(image_or_shape):
    if isinstance(image_or_shape, np.ndarray):
        h, w = image_or_shape.shape[:2]
        return int(h), int(w)
    if len(image_or_shape) < 2:
        raise ValueError("image_shape must have at least 2 elements (H, W).")
    return int(image_or_shape[0]), int(image_or_shape[1])


def crop_from_margin(image, margin: int):
    m = int(margin)
    if m < 0:
        raise ValueError("margin must be >= 0")
    if m == 0:
        return image
    return image[m:-m, m:-m]


def bounding_box(points_xy: np.ndarray):
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    return np.array([x0, y0, x1, y1], dtype=np.float64)
