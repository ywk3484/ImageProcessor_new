
"""Small I/O helpers for notebook workflows.

This file intentionally stays conservative and dependency-light. More specialized
TIFF/OASIS/domain loaders should be added as dedicated functions once stabilized.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def save_centers(path, centers_xy):
    path = Path(path)
    np.save(path, np.asarray(centers_xy, dtype=np.float64))
    return path


def load_centers(path):
    return np.asarray(np.load(Path(path)))
