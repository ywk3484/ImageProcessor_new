
"""Notebook convenience helpers.

These helpers keep plotting or quick-look code out of the numerical modules.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def display_centers(image, centers_xy, *, ax=None, s: float = 8.0):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    pts = np.asarray(centers_xy)
    if pts.size:
        ax.scatter(pts[:, 0], pts[:, 1], s=s)
    return ax


def display_pitch_map(x, pitch, *, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.asarray(x), np.asarray(pitch))
    ax.set_xlabel("x")
    ax.set_ylabel("pitch")
    return ax
