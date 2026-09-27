from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class CenterResult:
    centers_xy: np.ndarray
    method: str
    backend: str = "cpu"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class PitchResult:
    values: np.ndarray
    axis: str
    method: str
    backend: str = "cpu"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShiftResult:
    shift_yx: np.ndarray
    error: float | None = None
    phasediff: float | None = None
    backend: str = "cpu"
    method: str = "phase_xcorr"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectrumResult:
    frequency: np.ndarray
    amplitude: np.ndarray
    period: np.ndarray
    method: str = "fft"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentStatsResult:
    labels: np.ndarray
    stats: np.ndarray
    centroids: np.ndarray
    num_labels: int
    backend: str = "cpu"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CDMethodResult:
    """One contour per cell; invalid measurements retain their cell index."""

    method: str
    centers_xy: np.ndarray
    contours_xy: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    metrics: dict[str, np.ndarray]


@dataclass
class CDResult:
    """Cell extraction, per-method CDs, and calibration for one image.

    Coordinates and geometric metrics are stored in pixels. ``pixel_size``
    converts lengths to ``unit``; areas use its square.
    """

    image: np.ndarray
    binary: np.ndarray
    component_labels: np.ndarray
    component_ids: np.ndarray
    boxes_xywh: np.ndarray
    seeds_xy: np.ndarray
    cell_labels: np.ndarray
    cell_areas_px2: np.ndarray
    origins_xy: np.ndarray
    rois: np.ndarray
    masks: np.ndarray
    measurements: dict[str, CDMethodResult]
    pixel_size: float = 1.0
    unit: str = "px"
    meta: dict[str, Any] = field(default_factory=dict)
    mean_intensity: np.ndarray = field(init=False)

    def __post_init__(self):
        """Mean original intensity over each full Voronoi cell, without fill."""
        assigned = self.cell_labels >= 0
        totals = np.bincount(
            self.cell_labels[assigned], weights=self.image[assigned].astype(np.float64),
            minlength=len(self.seeds_xy),
        )
        self.mean_intensity = np.divide(
            totals, self.cell_areas_px2,
            out=np.full(len(self.seeds_xy), np.nan), where=self.cell_areas_px2 > 0,
        )
