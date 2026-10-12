"""Center detection and subpixel refinement.

Public conventions:
- centers are (N, 2) arrays in [x, y]
- image shape is (H, W)
- public functions return NumPy arrays or CenterResult
- backend selects CPU or GPU internals while preserving one stable notebook API
"""

from __future__ import annotations

from typing import Sequence, Tuple, Union
import numpy as np

from .backends import resolve_backend, to_numpy
from .geometry import image_hw
from ._cpu.centers import (
    detect_centers_cpu,
    refine_centers_cpu,
    refine_weighted_centroid_cpu,
    refine_logquadratic_cpu,
    refine_edge_moment_cpu,
    choose_refine_method_for_bbox,
)
from ._gpu.centers import (
    detect_centers_gpu,
    refine_centers_gpu,
    refine_weighted_centroid_gpu,
    refine_logquadratic_gpu,
    refine_centers_edge_moment_gpu,
)
_VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}
# Methods that take an ROI stack, so they can run on deblended cells.
_DEBLEND_METHODS = {"weighted", "logquad", "logquadratic", "none"} | _VORONOI_METHODS


def recommend_refine_method(width_px: float, height_px: float, *, small_feature_max: float = 12.0) -> str:
    """Return the recommended center-refinement method from feature bbox size."""
    return choose_refine_method_for_bbox(width_px, height_px, small_feature_max=small_feature_max)
def refine_weighted_centroid(gray_roi: np.ndarray, mask_roi: np.ndarray, *, backend: str = "cpu", device: int = 0, use_float64: bool = True) -> tuple[float, float]:
    b = resolve_backend(backend)
    if b == "gpu":
        return refine_weighted_centroid_gpu(gray_roi, mask_roi, device=device, use_float64=use_float64)
    return refine_weighted_centroid_cpu(gray_roi, mask_roi)


def refine_logquadratic(gray_roi: np.ndarray, mask_roi: np.ndarray, *, backend: str = "cpu", device: int = 0, use_float64: bool = True) -> tuple[float, float]:
    b = resolve_backend(backend)
    if b == "gpu":
        return refine_logquadratic_gpu(gray_roi, mask_roi, device=device, use_float64=use_float64)
    return refine_logquadratic_cpu(gray_roi, mask_roi)



def refine_centers_edge_moment(
    gray,
    stats_xywh_cc: np.ndarray,
    *,
    backend: str = "gpu",
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    device: int = 0,
):
    """Refine component centers using center-anchored independent edge localization.

    Parameters
    ----------
    gray : ndarray
        Full grayscale image.
    stats_xywh_cc : (K,6) ndarray
        Per-component rows [x, y, w, h, cx, cy] where (cx,cy) are coarse component centers.
    """
    b = resolve_backend(backend)
    if b == "gpu":
        centers, ok = refine_centers_edge_moment_gpu(
            to_numpy(gray),
            np.asarray(stats_xywh_cc, dtype=np.float32),
            band_rad=band_rad,
            edge_rad=edge_rad,
            smooth_passes=smooth_passes,
            loc_rad=loc_rad,
            grad_power=grad_power,
            iters=iters,
            device=device,
        )
        return np.asarray(centers, dtype=np.float64), np.asarray(ok, dtype=bool)
    rows = np.asarray(stats_xywh_cc, dtype=np.float64)
    g = to_numpy(gray)
    out = []
    ok = []
    H, W = g.shape
    for x, y, w, h, cx, cy in rows:
        x0 = max(0, int(x) - 3)
        y0 = max(0, int(y) - 3)
        x1 = min(W, int(x + w) + 3)
        y1 = min(H, int(y + h) + 3)
        roi = g[y0:y1, x0:x1]
        mask = np.zeros_like(roi, dtype=bool)
        xs = int(round(x - x0))
        ys = int(round(y - y0))
        ww = int(round(w))
        hh = int(round(h))
        mask[max(0, ys):min(mask.shape[0], ys + hh), max(0, xs):min(mask.shape[1], xs + ww)] = True
        cx0, cy0 = refine_edge_moment_cpu(roi, mask, band_rad=band_rad, edge_rad=edge_rad, smooth_passes=smooth_passes, loc_rad=loc_rad, grad_power=grad_power, iters=iters)
        good = np.isfinite(cx0) and np.isfinite(cy0)
        ok.append(bool(good))
        out.append([x0 + cx0, y0 + cy0] if good else [np.nan, np.nan])
    return np.asarray(out, dtype=np.float64), np.asarray(ok, dtype=bool)

def refine_centers(
    gray_roi: np.ndarray,
    mask_roi: np.ndarray,
    *,
    method: str = "logquad",
    backend: str = "cpu",
    device: int = 0,
    use_float64: bool = True,
    small_feature_max: float = 12.0,
) -> tuple[float, float]:
    b = resolve_backend(backend)
    if b == "gpu":
        return refine_centers_gpu(gray_roi, mask_roi, method=method, device=device, use_float64=use_float64)
    return refine_centers_cpu(gray_roi, mask_roi, method=method, small_feature_max=small_feature_max)


def detect_centers(
    image: np.ndarray,
    *,
    backend: str = "auto",
    threshold: str = "triangle",
    invert: bool = False,
    area_min: int = 1,
    area_max: int = 50,
    morph_open: int = 0,
    morph_close: int = 0,
    pad: int = 3,
    refine: str = "logquad",
    connectivity: int = 8,
    gpu_batch: int = 4096,
    device: int = 0,
    use_float64: bool = True,
    components_backend: str = "cpu",
    small_feature_max: float = 12.0,
    upsample_factor: int = 4,
    # Tiling params
    tile_h: int | None = None,
    overlap: int = 128,
    threshold_mode: str = "auto",
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    dedupe_eps: float = 1.5,
    background: str | None = None,
    deblend_iterations: int = 30,
):
    """Detect and refine centers.

    ``background="model"`` (opt-in) re-refines the detected centers on
    deblended cells: a smooth background map and every neighbouring hole's
    fitted Gaussian are subtracted first, removing neighbour-tail and
    background bias. Supported for ROI-stack methods: weighted, logquad, none,
    radial_symmetry, isophote_curvature. The default (None) is unchanged.
    """
    b = resolve_backend(backend)
    img = to_numpy(image)
    if b == "cpu" and refine in _VORONOI_METHODS:
        raise NotImplementedError(
            f"refine='{refine}' requires GPU backend. Set backend='gpu' or install CuPy."
        )
    if background not in (None, "model"):
        raise ValueError("background must be None or 'model'")
    if background == "model" and refine not in _DEBLEND_METHODS:
        raise NotImplementedError(
            f"background='model' is not supported for refine='{refine}': edge methods read the full "
            "image rather than per-cell ROIs. Use one of " + ", ".join(sorted(_DEBLEND_METHODS)) + "."
        )
    if background == "model":
        result = detect_centers(
            img, backend=b, threshold=threshold, invert=invert, area_min=area_min, area_max=area_max,
            morph_open=morph_open, morph_close=morph_close, pad=pad, refine=refine,
            connectivity=connectivity, gpu_batch=gpu_batch, device=device, use_float64=use_float64,
            components_backend=components_backend, small_feature_max=small_feature_max,
            upsample_factor=upsample_factor, tile_h=tile_h, overlap=overlap,
            threshold_mode=threshold_mode, otsu_downsample=otsu_downsample,
            thr_scale=thr_scale, dedupe_eps=dedupe_eps,
        )
        return _refine_deblended(
            img, result, refine=refine, invert=invert, pad=pad, backend=b, device=device,
            use_float64=use_float64, upsample_factor=upsample_factor, iterations=deblend_iterations,
            tile_h=tile_h, overlap=overlap,
        )
    if b == "gpu":
        return detect_centers_gpu(
            img,
            threshold=threshold, invert=invert,
            area_min=area_min, area_max=area_max,
            morph_open=morph_open, morph_close=morph_close,
            pad=pad, refine=refine, connectivity=connectivity,
            gpu_batch=gpu_batch, device=device,
            use_float64=use_float64,
            components_backend=components_backend,
            small_feature_max=small_feature_max,
            upsample_factor=upsample_factor,
            tile_h=tile_h, overlap=overlap,
            threshold_mode=threshold_mode,
            otsu_downsample=otsu_downsample,
            thr_scale=thr_scale, dedupe_eps=dedupe_eps,
        )
    return detect_centers_cpu(
        img,
        threshold=threshold, invert=invert,
        area_min=area_min, area_max=area_max,
        morph_open=morph_open, morph_close=morph_close,
        pad=pad, refine=refine, connectivity=connectivity,
        small_feature_max=small_feature_max,
        tile_h=tile_h, overlap=overlap, dedupe_eps=dedupe_eps,
    )



def _deblended_cells_centers(signal, seeds, *, refine, pad, backend, device, use_float64,
                             upsample_factor, iterations):
    """Re-refine ``seeds`` on cells with background and neighbouring holes removed."""
    from ._cpu.cells import _extract_voronoi_rois, compute_voronoi_labels_cpu
    from ._cpu.deblend import auto_tile, deblend

    pitch = auto_tile(seeds, signal.shape)[1]
    half = int(np.ceil(0.6 * pitch)) if np.isfinite(pitch) else 16
    labels = compute_voronoi_labels_cpu(seeds, signal.shape)
    rows = [[x - half, y - half, 2 * half + 1, 2 * half + 1, x, y] for x, y in seeds]
    _, masks, origins = _extract_voronoi_rois(signal, labels, rows, pad=pad)
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
    state = deblend(signal, seeds, masks, origins, iterations=iterations)
    cleaned = state.cleaned
    quality = None
    if refine in _VORONOI_METHODS:
        from ._gpu.centers import _isophote_curvature_batch_gpu, _radial_symmetry_batch_gpu

        from ._cpu.deblend import FWHM_K, sigma_major

        # Limit gradients to about the window the thresholded path would use
        # (feature + pad); the whole cell adds noisy, signal-free gradients.
        reach = np.where(np.isfinite(state.params).all(axis=1), 2 * FWHM_K * sigma_major(state.params), np.inf)
        yy, xx = np.indices(masks.shape[1:])
        local_seeds = seeds - origins
        near = (np.hypot(xx[None] - local_seeds[:, 0, None, None], yy[None] - local_seeds[:, 1, None, None])
                <= (reach + pad)[:, None, None])
        window = masks & near
        batch = _radial_symmetry_batch_gpu if refine == "radial_symmetry" else _isophote_curvature_batch_gpu
        local, quality = batch(np.where(window, cleaned, 0.0), window, upsample_factor=upsample_factor, device=device)
    else:
        # The thresholded component these methods expect: the cell's bright core.
        peak = np.where(np.isfinite(state.params[:, 0]) & (state.params[:, 0] > 0), state.params[:, 0],
                        np.where(masks, cleaned, -np.inf).reshape(len(seeds), -1).max(axis=1))
        support = masks & (cleaned > 0.2 * peak[:, None, None])
        if backend == "gpu":
            from ._gpu.centers import _refine_batch_gpu

            local = _refine_batch_gpu(list(cleaned), list(support), method=refine,
                                      use_float64=use_float64, device=device)
        else:
            local = np.array([refine_centers_cpu(c, m, method=refine) for c, m in zip(cleaned, support)],
                             dtype=np.float64).reshape(-1, 2)
    return np.asarray(local, dtype=np.float64) + origins, quality, masks.sum(axis=(1, 2)), state


def _refine_deblended(img, result, *, refine, invert, pad, backend, device, use_float64,
                      upsample_factor, iterations, tile_h, overlap):
    from .types import CenterResult

    signal = np.asarray(img, dtype=np.float64) * (-1.0 if invert else 1.0)
    seeds = np.asarray(result.centers_xy, dtype=np.float64).reshape(-1, 2)
    height = signal.shape[0]
    bands = [(0, height)] if not tile_h else [(y0, min(height, y0 + int(tile_h))) for y0 in range(0, height, int(tile_h))]
    centers, qualities, areas, statuses, used, sources = [], [], [], [], 0, set()
    for y0, y1 in bands:
        # Strips overlap so holes near a strip edge still see all their neighbours.
        lo, hi = (0, height) if not tile_h else (max(0, y0 - int(overlap)), min(height, y1 + int(overlap)))
        own = (seeds[:, 1] >= y0) & (seeds[:, 1] < y1)
        if not own.any():
            continue
        inside = (seeds[:, 1] >= lo) & (seeds[:, 1] < hi)
        local_seeds = seeds[inside] - [0, lo]
        found, quality, area, state = _deblended_cells_centers(
            signal[lo:hi], local_seeds, refine=refine, pad=pad, backend=backend, device=device,
            use_float64=use_float64, upsample_factor=upsample_factor, iterations=iterations,
        )
        keep = own[inside]
        centers.append(found[keep] + [0, lo])
        areas.append(area[keep])
        statuses.append(state.status[keep])
        if quality is not None:
            qualities.append(np.asarray(quality, dtype=np.float64)[keep])
        used = max(used, state.iterations)
        sources.add(state.background_info.get("source"))
    centers = np.concatenate(centers) if centers else np.zeros((0, 2))
    finite = np.isfinite(centers).all(axis=1)
    status = np.concatenate(statuses) if statuses else np.zeros(0, "U32")
    names, counts = np.unique(status, return_counts=True)
    meta = dict(result.meta)
    meta.update(
        background="model", deblend_iterations=int(iterations), deblend_iterations_used=int(used),
        deblend_status_counts=dict(zip(names.tolist(), counts.tolist())),
        background_source=sorted(s for s in sources if s),
        deblend_dropped_nonfinite=int((~finite).sum()),
    )
    if refine in _VORONOI_METHODS:
        key = "radial_symmetry_residual" if refine == "radial_symmetry" else "isophote_curvature_spread"
        meta[key] = (np.concatenate(qualities) if qualities else np.zeros(0))[finite]
        meta["voronoi_cell_area"] = (np.concatenate(areas) if areas else np.zeros(0, np.int64))[finite].astype(np.int64)
    return CenterResult(centers_xy=centers[finite], method=result.method, backend=result.backend, meta=meta)


def filter_centers(
    centers_xy: np.ndarray,
    image_shape_hw: Union[np.ndarray, Tuple[int, int], Sequence[int]],
    *,
    margin: float = 5.0,
) -> np.ndarray:
    centers = np.asarray(centers_xy, dtype=np.float64)
    if centers.size == 0:
        return centers.reshape(0, 2)

    H, W = image_hw(image_shape_hw)
    m = float(margin)
    x = centers[:, 0]
    y = centers[:, 1]
    keep = (x >= m) & (x < (W - m)) & (y >= m) & (y < (H - m))
    return centers[keep]


def dedupe_centers(centers_xy: np.ndarray, *, eps: float = 1.5) -> np.ndarray:
    """Merge near-duplicate detections via radius-based union-find clustering."""
    pts = np.asarray(centers_xy, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return pts.reshape(0, 2)

    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int8)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    eps2 = float(eps) ** 2
    for i in range(n):
        d2 = np.sum((pts[i + 1:] - pts[i]) ** 2, axis=1)
        js = np.where(d2 <= eps2)[0] + (i + 1)
        for j in js:
            union(i, int(j))

    roots = np.array([find(i) for i in range(n)], dtype=np.int32)
    unique_roots, inv = np.unique(roots, return_inverse=True)

    K = unique_roots.size
    sums = np.zeros((K, 2), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int32)
    np.add.at(sums, inv, pts)
    np.add.at(counts, inv, 1)
    return sums / np.maximum(counts[:, None], 1)


