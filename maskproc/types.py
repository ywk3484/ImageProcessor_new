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
