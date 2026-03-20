
"""Notebook convenience helpers.

These helpers keep plotting or quick-look code out of the numerical modules.
"""

from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _require_matplotlib():
    if plt is None:
        raise ImportError("matplotlib is required for notebook display functions: pip install matplotlib")


def display_centers(image, centers_xy, *, ax=None, s: float = 8.0):
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    pts = np.asarray(centers_xy)
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], s=s)
    return ax


def display_pitch_map(x, pitch, *, ax=None):
    _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.asarray(x), np.asarray(pitch))
    ax.set_xlabel("x")
    ax.set_ylabel("pitch")
    return ax
