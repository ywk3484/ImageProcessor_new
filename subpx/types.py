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
class PitchProfile:
    """Averaged adjacent-hole pitch and its discrete integral, in ``unit``.

    Node arrays have length M; interval arrays (pitch, std, counts) have M-1
    entries. ``pairs`` contains original hole indices for the selected band.
    Missing intervals are NaN and interrupt integration from the reference.
    """

    axis: str
    indices: np.ndarray
    position: np.ndarray
    pitch: np.ndarray
    std: np.ndarray
    counts: np.ndarray
    distance: np.ndarray
    registration: np.ndarray
    selected_indices: np.ndarray
    pairs: np.ndarray
    nominal_pitch: float
    reference_node: int


@dataclass
class PatternPitchResult:
    """Pitch from one CD method; coordinates and lattice pitch remain in pixels.

    Profiles contain calibrated lengths. Lattice indices are [row, column],
    independent of which CD fits succeeded. Unindexed detections retain [-1, -1]
    indices and are excluded from ``valid``. Invalid fits never become seeds.
    """

    method: str
    centers_xy: np.ndarray
    valid: np.ndarray
    indices_rc: np.ndarray
    layout: str
    stagger_axis: str
    lattice_pitch_xy: np.ndarray
    reference_index: int
    profiles: dict[str, PitchProfile]
    pixel_size: float = 1.0
    unit: str = "px"
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
    """One contour per cell; invalid measurements retain their cell index.

    ``geometry`` holds the arrays each contour was built from (radial methods:
    ``center_local`` and per-ray ``edge_px``; ellipse methods: ``eigenvalues``
    and ``axes`` of the precision matrix, plus ``center_local`` for logquad),
    so compact exports can rebuild ``contours_xy`` exactly. ``inputs`` holds
    per-cell data the measurement was made on when it differs from the
    result's ``rois`` (``disk``/``flux_ratio``: ``cells``, the fit input with
    the neighbours' disk models removed, NaN outside the cell); in memory
    only, never saved.
    """

    method: str
    centers_xy: np.ndarray
    contours_xy: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    metrics: dict[str, np.ndarray]
    geometry: dict[str, np.ndarray] = field(default_factory=dict)
    inputs: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class CDSummary:
    """Numerical per-hole measurements without image, label maps or cell statistics.

    Used by ``measure_cds(summary_only=True)`` for batch page averages. Arrays
    are NumPy and use the same coordinates and units as ``CDResult``.
    """

    seeds_xy: np.ndarray
    measurements: dict[str, CDMethodResult]
    pixel_size: float = 1.0
    unit: str = "px"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CDResult:
    """Cell extraction, per-method CDs, and calibration for one image.

    Coordinates and geometric metrics are stored in pixels. ``pixel_size``
    converts lengths to ``unit``; areas use its square. ``deblend_model``
    (``background='model'`` only) holds the fitted global Gaussian ``params`` of
    every modelled hole, the ``report`` indices of the measured cells among
    them, and the background ``tile_medians`` / ``tile_px`` when available;
    with ``masks`` it rebuilds ``rois`` and ``background_map`` exactly.
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
    background_map: np.ndarray | None = None
    deblend_model: dict[str, Any] | None = None
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
