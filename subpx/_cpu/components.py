from __future__ import annotations

import numpy as np

from ..types import ComponentStatsResult

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Install opencv-python.")


def connected_components_stats_cpu(mask: np.ndarray, *, connectivity: int = 8) -> ComponentStatsResult:
    _require_cv2()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("connected_components_stats expects a 2D mask.")
    bw = (m > 0).astype(np.uint8) * 255
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=int(connectivity))
    return ComponentStatsResult(
        labels=labels.astype(np.int32, copy=False),
        stats=stats.astype(np.int32, copy=False),
        centroids=centroids.astype(np.float64, copy=False),
        num_labels=int(num),
        backend="cpu",
        meta={"connectivity": int(connectivity), "implementation": "cv2.connectedComponentsWithStats"},
    )
