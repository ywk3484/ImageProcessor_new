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
from .types import CenterResult

_VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}


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
):
    b = resolve_backend(backend)
    img = to_numpy(image)
    if b == "cpu" and refine in _VORONOI_METHODS:
        raise NotImplementedError(
            f"refine='{refine}' requires GPU backend. Set backend='gpu' or install CuPy."
        )
    if b == "gpu":
        return detect_centers_gpu(
            img,
            threshold=threshold,
            invert=invert,
            area_min=area_min,
            area_max=area_max,
            morph_open=morph_open,
            morph_close=morph_close,
            pad=pad,
            refine=refine,
            connectivity=connectivity,
            gpu_batch=gpu_batch,
            device=device,
            use_float64=use_float64,
            components_backend=components_backend,
            small_feature_max=small_feature_max,
            upsample_factor=upsample_factor,
        )
    return detect_centers_cpu(
        img,
        threshold=threshold,
        invert=invert,
        area_min=area_min,
        area_max=area_max,
        morph_open=morph_open,
        morph_close=morph_close,
        pad=pad,
        refine=refine,
        connectivity=connectivity,
        small_feature_max=small_feature_max,
    )


def detect_centers_tiled(
    image: np.ndarray,
    *,
    backend: str = "gpu",
    tile_h: int = 8192,
    overlap: int = 128,
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    dedupe_eps: float = 1.5,
    **kwargs,
) -> CenterResult:
    """Detect centers on vertically tiled images.

    For GPU backend, uses a dedicated tiled pipeline with global Otsu threshold,
    GPU connected components, vectorized filtering, and band-based de-dup.
    For CPU backend, tiles are processed independently via detect_centers().
    """
    img = to_numpy(image)
    if img.ndim != 2:
        raise ValueError("detect_centers_tiled expects a 2D grayscale image.")
    H, W = img.shape
    tile_h = int(max(1, tile_h))
    overlap = int(max(0, overlap))

    b = resolve_backend(backend)

    # --- GPU path: dedicated tiled pipeline ---
    if b == "gpu":
        from ._gpu.centers import _detect_centers_tiled_gpu
        return _detect_centers_tiled_gpu(
            img,
            tile_h=tile_h,
            overlap=overlap,
            otsu_downsample=otsu_downsample,
            thr_scale=thr_scale,
            **kwargs,
        )

    # --- CPU path: tile-by-tile detect_centers ---
    if overlap >= tile_h:
        raise ValueError("overlap must be < tile_h")

    step = tile_h - overlap
    all_centers = []
    per_tile_counts = []

    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + tile_h)
        tile = img[y0:y1]
        res = detect_centers(tile, backend="cpu", **kwargs)
        pts = np.asarray(res.centers_xy, dtype=np.float64)
        if pts.size == 0:
            per_tile_counts.append(0)
        else:
            global_pts = pts.copy()
            global_pts[:, 1] += y0

            half = overlap // 2
            keep_lo = y0 if y0 == 0 else (y0 + half)
            keep_hi = y1 if y1 == H else (y1 - half)
            keep = (global_pts[:, 1] >= keep_lo) & (global_pts[:, 1] < keep_hi)
            kept = global_pts[keep]
            per_tile_counts.append(int(kept.shape[0]))
            if kept.size:
                all_centers.append(kept)

        if y1 == H:
            break
        y0 += step

    if all_centers:
        centers = np.vstack(all_centers)
        centers = dedupe_centers(centers, eps=float(dedupe_eps))
    else:
        centers = np.zeros((0, 2), dtype=np.float64)

    return CenterResult(
        centers_xy=centers,
        method=f"tiled({backend})",
        backend=b,
        meta={
            "tile_h": int(tile_h),
            "overlap": int(overlap),
            "otsu_downsample": int(max(1, otsu_downsample)),
            "dedupe_eps": float(dedupe_eps),
            "per_tile_counts": per_tile_counts,
            **kwargs,
        },
    )


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


def detect_centers_tiled_global_otsu(
    image: np.ndarray,
    *,
    backend: str = "gpu",
    tile_h: int = 8192,
    overlap: int = 128,
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    dedupe_eps: float = 1.5,
    **kwargs,
) -> CenterResult:
    """Tiled detection using one global Otsu threshold estimated on a downsampled image.

    This preserves the behavior of the historical notebook workflow where thresholding was
    computed once globally and then reused per tile to reduce tile-to-tile bias.
    """
    img = to_numpy(image)
    if img.ndim != 2:
        raise ValueError("detect_centers_tiled_global_otsu expects a 2D grayscale image.")
    if kwargs.get("refine") in _VORONOI_METHODS:
        raise NotImplementedError(
            "Voronoi-partitioned methods are not supported with tiled detection."
        )
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for global Otsu tiled detection.") from exc
    H, W = img.shape
    if overlap >= tile_h:
        raise ValueError("overlap must be < tile_h")
    ds = max(1, int(otsu_downsample))
    thr_type = cv2.THRESH_BINARY_INV if bool(kwargs.get("invert", False)) else cv2.THRESH_BINARY
    g_ds = img[::ds, ::ds]
    otsu_thresh, _ = cv2.threshold(g_ds, 0, 255, thr_type | cv2.THRESH_OTSU)
    thr = float(otsu_thresh) * float(thr_scale)
    # use the regular tiled implementation but pin a consistent threshold by pre-normalizing tiles
    # around the global threshold. This keeps the public API simple even though detect_centers
    # itself currently only exposes threshold='otsu'.
    all_centers = []
    step = tile_h - overlap
    for y0 in range(0, H, step):
        y1 = min(H, y0 + tile_h)
        tile = img[y0:y1]
        _, bw = cv2.threshold(tile, thr, 255, thr_type)
        # feed the binary tile through connected-components + public refinement by using the
        # detected component bounding boxes as ROIs on the original grayscale tile.
        comp_backend = kwargs.get("components_backend", "cpu")
        from .components import connected_components_stats
        comp = connected_components_stats(bw, backend=comp_backend, connectivity=int(kwargs.get("connectivity", 8)), device=int(kwargs.get("device", 0)))
        rows = []
        rois = []
        masks = []
        Ht, Wt = tile.shape
        area_min = int(kwargs.get("area_min", 1))
        area_max = int(kwargs.get("area_max", 50))
        pad = int(kwargs.get("pad", 3))
        for lab in range(1, comp.num_labels):
            x, y, w, h, area = comp.stats[lab]
            if area < area_min or area > area_max:
                continue
            x0r = max(0, int(x) - pad)
            y0r = max(0, int(y) - pad)
            x1r = min(Wt, int(x + w) + pad)
            y1r = min(Ht, int(y + h) + pad)
            rows.append((x0r, y0r))
            rois.append(tile[y0r:y1r, x0r:x1r])
            masks.append(comp.labels[y0r:y1r, x0r:x1r] == lab)
        refine = kwargs.get("refine", "logquad")
        device = int(kwargs.get("device", 0))
        use_float64 = bool(kwargs.get("use_float64", True))
        gpu_batch = int(kwargs.get("gpu_batch", 4096))
        pts = []
        if resolve_backend(backend) == "gpu":
            from ._gpu.centers import _refine_batch_gpu
            for i in range(0, len(rois), max(1, gpu_batch)):
                loc = _refine_batch_gpu(rois[i:i+gpu_batch], masks[i:i+gpu_batch], method=refine, use_float64=use_float64, device=device)
                for (x0r, y0r), (cx, cy) in zip(rows[i:i+gpu_batch], loc):
                    if np.isfinite(cx) and np.isfinite(cy):
                        pts.append([x0r + float(cx), y0r + float(cy)])
        else:
            for (x0r, y0r), roi, mask in zip(rows, rois, masks):
                cx, cy = refine_centers_cpu(roi, mask, method=refine)
                if np.isfinite(cx) and np.isfinite(cy):
                    pts.append([x0r + float(cx), y0r + float(cy)])
        if pts:
            pts = np.asarray(pts, dtype=np.float64)
            pts[:, 1] += y0
            half = overlap // 2
            keep_lo = y0 if y0 == 0 else (y0 + half)
            keep_hi = y1 if y1 == H else (y1 - half)
            m = (pts[:, 1] >= keep_lo) & (pts[:, 1] < keep_hi)
            if np.any(m):
                all_centers.append(pts[m])
        if y1 == H:
            break
    centers = np.vstack(all_centers) if all_centers else np.zeros((0, 2), dtype=np.float64)
    if centers.size:
        centers = dedupe_centers(centers, eps=float(dedupe_eps))
    meta = dict(kwargs)
    meta.update({"tile_h": int(tile_h), "overlap": int(overlap), "otsu_downsample": int(ds), "thr_scale": float(thr_scale), "dedupe_eps": float(dedupe_eps), "global_otsu_threshold": float(thr)})
    return CenterResult(centers_xy=centers, method=f"tiled_global_otsu({backend})", backend=resolve_backend(backend), meta=meta)
