from __future__ import annotations

from typing import List
import numpy as np

from ..types import CenterResult
from .components import connected_components_stats_cpu

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


def _require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required. Install opencv-python.")




def choose_refine_method_for_bbox(width: float, height: float, *, small_feature_max: float = 12.0) -> str:
    """Choose a stable subpixel method from component bbox size.

    Heuristic derived from project usage:
    - small features (~5 px diameter, pitch ~2x feature size): logquadratic refinement
    - larger features (roughly > 12 px extent, 3-5 px edge blur): edge-gradmoment refinement
    """
    size = float(max(width, height))
    return "logquad" if size <= float(small_feature_max) else "edge_gradmoment"
def _bg_from_border(patch: np.ndarray, border: int = 2) -> float:
    g = np.asarray(patch)
    h, w = g.shape[:2]
    b = int(border)
    if h <= 2 * b or w <= 2 * b:
        return float(np.median(g))
    top = g[:b, :]
    bot = g[-b:, :]
    lef = g[:, :b]
    rig = g[:, -b:]
    border_pixels = np.concatenate([top.ravel(), bot.ravel(), lef.ravel(), rig.ravel()])
    return float(np.median(border_pixels))


def refine_weighted_centroid_cpu(gray_roi: np.ndarray, mask_roi: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gray_roi, dtype=np.float64)
    m = np.asarray(mask_roi, dtype=bool)
    if not np.any(m):
        return (np.nan, np.nan)

    bg = _bg_from_border(g, border=2)
    w = g - bg
    w[~m] = 0.0
    w[w < 0] = 0.0

    s = float(w.sum())
    if s <= 0:
        ys, xs = np.nonzero(m)
        return (float(xs.mean()), float(ys.mean()))

    ys, xs = np.indices(g.shape, dtype=np.float64)
    cx = float((xs * w).sum() / s)
    cy = float((ys * w).sum() / s)
    return (cx, cy)


def refine_logquadratic_cpu(gray_roi: np.ndarray, mask_roi: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gray_roi, dtype=np.float64)
    m = np.asarray(mask_roi, dtype=bool)

    if np.count_nonzero(m) < 6:
        return (np.nan, np.nan)

    bg = _bg_from_border(g, border=2)
    v = g - bg
    v[~m] = 0.0

    vpos = v[m]
    if vpos.size == 0 or np.max(vpos) <= 0:
        return (np.nan, np.nan)

    eps = 1e-9
    z = np.log(np.maximum(vpos, eps))
    ys, xs = np.nonzero(m)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)

    A = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c, d, e, _f = coef

    H = np.array([[2 * a, c], [c, 2 * b]], dtype=np.float64)
    gvec = np.array([-d, -e], dtype=np.float64)

    if np.linalg.cond(H) > 1e8:
        return (np.nan, np.nan)

    try:
        xy = np.linalg.solve(H, gvec)
    except np.linalg.LinAlgError:
        return (np.nan, np.nan)

    cx, cy = float(xy[0]), float(xy[1])
    h, w = g.shape
    if not (0 <= cx < w and 0 <= cy < h):
        return (np.nan, np.nan)
    return (cx, cy)



def _smooth_binom5_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True
    else:
        squeeze = False
    xm2 = np.concatenate([x[..., :1], x[..., :1], x[..., :-2]], axis=-1)
    xm1 = np.concatenate([x[..., :1], x[..., :-1]], axis=-1)
    xp1 = np.concatenate([x[..., 1:], x[..., -1:]], axis=-1)
    xp2 = np.concatenate([x[..., 2:], x[..., -1:], x[..., -1:]], axis=-1)
    out = (xm2 + 4 * xm1 + 6 * x + 4 * xp1 + xp2) * (1.0 / 16.0)
    return out[0] if squeeze else out


def _edge_from_profile_gradmoment_np(
    prof: np.ndarray,
    base_x: float,
    *,
    grad_power: float = 8.0,
    loc_rad: int = 3,
) -> tuple[float, bool]:
    prof = np.asarray(prof, dtype=np.float64)
    if prof.size < 3:
        return np.nan, False
    d = np.abs(np.diff(prof))
    if d.size == 0 or not np.any(np.isfinite(d)):
        return np.nan, False
    ip = int(np.nanargmax(d))
    ip0 = int(np.clip(ip, loc_rad, max(loc_rad, d.size - 1 - loc_rad)))
    jj = np.arange(ip0 - loc_rad, ip0 + loc_rad + 1, dtype=np.int32)
    jj = np.clip(jj, 0, d.size - 1)
    dloc = d[jj]
    w = np.power(np.maximum(dloc, 0.0), float(grad_power))
    sw = float(np.sum(w))
    if sw <= 1e-12:
        return np.nan, False
    xk = float(base_x) + (jj.astype(np.float64) + 0.5)
    x_edge = float(np.sum(w * xk) / sw)
    return x_edge, bool(np.isfinite(x_edge))


def refine_edge_moment_cpu(
    gray_roi: np.ndarray,
    mask_roi: np.ndarray,
    *,
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
) -> tuple[float, float]:
    g = np.asarray(gray_roi, dtype=np.float64)
    m = np.asarray(mask_roi, dtype=bool)
    if g.ndim != 2 or m.ndim != 2 or g.shape != m.shape or not np.any(m):
        return (np.nan, np.nan)

    ys, xs = np.nonzero(m)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w = float(x1 - x0 + 1)
    h = float(y1 - y0 + 1)
    xc = float(xs.mean())
    yc = float(ys.mean())
    H, W = g.shape

    def _profile_x(x_center: int, y_center: float):
        yy = np.clip(np.rint(y_center).astype(np.int32) + np.arange(-band_rad, band_rad + 1, dtype=np.int32), 0, H - 1)
        xx = np.clip(int(x_center) + np.arange(-edge_rad, edge_rad + 1, dtype=np.int32), 0, W - 1)
        patch = g[yy[:, None], xx[None, :]]
        prof = np.mean(patch, axis=0)
        for _ in range(int(smooth_passes)):
            prof = _smooth_binom5_np(prof)
        base = float(int(x_center) - edge_rad)
        return prof, base

    def _profile_y(y_center: int, x_center: float):
        xx = np.clip(np.rint(x_center).astype(np.int32) + np.arange(-band_rad, band_rad + 1, dtype=np.int32), 0, W - 1)
        yy = np.clip(int(y_center) + np.arange(-edge_rad, edge_rad + 1, dtype=np.int32), 0, H - 1)
        patch = g[yy[:, None], xx[None, :]]
        prof = np.mean(patch, axis=1)
        for _ in range(int(smooth_passes)):
            prof = _smooth_binom5_np(prof)
        base = float(int(y_center) - edge_rad)
        return prof, base

    ok_all = True
    for _ in range(int(iters)):
        xL0 = int(np.rint(xc - 0.5 * (w - 1.0)))
        xR0 = int(np.rint(xc + 0.5 * (w - 1.0)))
        yT0 = int(np.rint(yc - 0.5 * (h - 1.0)))
        yB0 = int(np.rint(yc + 0.5 * (h - 1.0)))

        profL, baseL = _profile_x(xL0, yc)
        profR, baseR = _profile_x(xR0, yc)
        profT, baseT = _profile_y(yT0, xc)
        profB, baseB = _profile_y(yB0, xc)

        xL, okL = _edge_from_profile_gradmoment_np(profL, baseL, grad_power=grad_power, loc_rad=loc_rad)
        xR, okR = _edge_from_profile_gradmoment_np(profR, baseR, grad_power=grad_power, loc_rad=loc_rad)
        yT, okT = _edge_from_profile_gradmoment_np(profT, baseT, grad_power=grad_power, loc_rad=loc_rad)
        yB, okB = _edge_from_profile_gradmoment_np(profB, baseB, grad_power=grad_power, loc_rad=loc_rad)
        ok_iter = okL and okR and okT and okB
        ok_all = ok_all and ok_iter
        if not ok_iter:
            break
        xc = 0.5 * (xL + xR)
        yc = 0.5 * (yT + yB)

    if not ok_all or not np.isfinite(xc) or not np.isfinite(yc):
        return (np.nan, np.nan)
    return float(xc), float(yc)

def refine_centers_cpu(gray_roi: np.ndarray, mask_roi: np.ndarray, *, method: str = "logquad", small_feature_max: float = 12.0) -> tuple[float, float]:
    if method == "auto":
        ys, xs = np.nonzero(mask_roi)
        if xs.size == 0:
            return (np.nan, np.nan)
        w = float(xs.max() - xs.min() + 1)
        h = float(ys.max() - ys.min() + 1)
        method = choose_refine_method_for_bbox(w, h, small_feature_max=small_feature_max)
    if method == "weighted":
        return refine_weighted_centroid_cpu(gray_roi, mask_roi)
    if method in {"logquad", "logquadratic"}:
        return refine_logquadratic_cpu(gray_roi, mask_roi)
    if method in {"edge_gradmoment", "edge_moment", "edges_centered"}:
        return refine_edge_moment_cpu(gray_roi, mask_roi)
    if method == "none":
        ys, xs = np.nonzero(mask_roi)
        if xs.size == 0:
            return (np.nan, np.nan)
        return float(xs.mean()), float(ys.mean())
    raise ValueError("method must be one of: 'none', 'weighted', 'logquad', 'edge_gradmoment', 'auto'.")


def _segment_binary(image: np.ndarray, *, threshold: str = "otsu", invert: bool = False, morph_open: int = 0, morph_close: int = 0):
    _require_cv2()
    g = np.asarray(image)
    if g.ndim != 2:
        raise ValueError("Expected 2D grayscale image.")
    if threshold != "otsu":
        raise ValueError("Currently supported threshold values: 'otsu'")

    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, bw = cv2.threshold(g, 0, 255, thr_type | cv2.THRESH_OTSU)

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=int(morph_open))
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=int(morph_close))
    return g, bw


def _segment_components(image: np.ndarray, *, threshold: str = "otsu", invert: bool = False, morph_open: int = 0, morph_close: int = 0, connectivity: int = 8):
    g, bw = _segment_binary(
        image,
        threshold=threshold,
        invert=invert,
        morph_open=morph_open,
        morph_close=morph_close,
    )
    comp = connected_components_stats_cpu(bw, connectivity=connectivity)
    return g, bw, comp.num_labels, comp.labels, comp.stats


def detect_centers_cpu(
    image: np.ndarray,
    *,
    threshold: str = "otsu",
    invert: bool = False,
    area_min: int = 1,
    area_max: int = 50,
    morph_open: int = 0,
    morph_close: int = 0,
    pad: int = 3,
    refine: str = "logquad",
    connectivity: int = 8,
    small_feature_max: float = 12.0,
) -> CenterResult:
    g, _bw, num, labels, stats = _segment_components(
        image,
        threshold=threshold,
        invert=invert,
        morph_open=morph_open,
        morph_close=morph_close,
        connectivity=connectivity,
    )

    H, W = g.shape
    centers: List[List[float]] = []
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area < int(area_min) or area > int(area_max):
            continue

        x0 = max(0, int(x) - int(pad))
        y0 = max(0, int(y) - int(pad))
        x1 = min(W, int(x + w) + int(pad))
        y1 = min(H, int(y + h) + int(pad))

        gray_roi = g[y0:y1, x0:x1]
        mask_roi = labels[y0:y1, x0:x1] == lab
        refine_local = choose_refine_method_for_bbox(w, h, small_feature_max=small_feature_max) if refine == "auto" else refine
        cx, cy = refine_centers_cpu(gray_roi, mask_roi, method=refine_local, small_feature_max=small_feature_max)
        if np.isfinite(cx) and np.isfinite(cy):
            centers.append([x0 + cx, y0 + cy])

    arr = np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 2), dtype=np.float64)
    return CenterResult(
        centers_xy=arr,
        method=f"threshold={threshold}, refine={refine}",
        backend="cpu",
        meta={
            "invert": bool(invert),
            "area_min": int(area_min),
            "area_max": int(area_max),
            "morph_open": int(morph_open),
            "morph_close": int(morph_close),
            "pad": int(pad),
            "connectivity": int(connectivity),
            "small_feature_max": float(small_feature_max),
        },
    )
