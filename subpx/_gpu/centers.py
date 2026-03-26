from __future__ import annotations

import numpy as np

from ..types import CenterResult
from ..backends import gpu_device
from .._cpu.centers import _segment_binary, _bg_from_border, choose_refine_method_for_bbox
from .components import connected_components_stats_gpu, _connected_components_stats_gpu_core
from .._cpu.components import connected_components_stats_cpu

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

try:
    import cupyx.scipy.ndimage as cndi  # type: ignore
except Exception:  # pragma: no cover
    cndi = None

try:
    from cupyx.scipy.special import erf as cperf  # type: ignore
except Exception:  # pragma: no cover
    cperf = None


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for GPU center refinement.")


def _pad_rois(rois: list[np.ndarray], masks: list[np.ndarray]):
    B = len(rois)
    hs = np.asarray([r.shape[0] for r in rois], dtype=np.int32)
    ws = np.asarray([r.shape[1] for r in rois], dtype=np.int32)
    Hm = int(hs.max()) if B else 0
    Wm = int(ws.max()) if B else 0

    rois_pad = np.zeros((B, Hm, Wm), dtype=np.float64)
    masks_pad = np.zeros((B, Hm, Wm), dtype=bool)
    bgs = np.zeros((B,), dtype=np.float64)
    for i, (r, m) in enumerate(zip(rois, masks)):
        h, w = r.shape
        rois_pad[i, :h, :w] = r
        masks_pad[i, :h, :w] = m
        bgs[i] = _bg_from_border(r, border=2)
    return rois_pad, masks_pad, hs, ws, bgs


def _weighted_from_batch_gpu(g_batch, m_batch, hs, ws, bg_batch, *, dtype):
    yy = cp.arange(g_batch.shape[1], dtype=dtype)[None, :, None]
    xx = cp.arange(g_batch.shape[2], dtype=dtype)[None, None, :]

    w = g_batch - bg_batch[:, None, None]
    w = cp.where(m_batch, cp.maximum(w, 0), 0)
    s = w.sum(axis=(1, 2))

    cx = (w * xx).sum(axis=(1, 2)) / cp.maximum(s, dtype(1e-12))
    cy = (w * yy).sum(axis=(1, 2)) / cp.maximum(s, dtype(1e-12))

    msum = m_batch.sum(axis=(1, 2)).astype(dtype)
    cx_m = (m_batch.astype(dtype) * xx).sum(axis=(1, 2)) / cp.maximum(msum, dtype(1e-12))
    cy_m = (m_batch.astype(dtype) * yy).sum(axis=(1, 2)) / cp.maximum(msum, dtype(1e-12))

    bad = (s <= dtype(0)) | ~cp.isfinite(cx) | ~cp.isfinite(cy)
    cx = cp.where(bad, cx_m, cx)
    cy = cp.where(bad, cy_m, cy)

    valid_bounds = (cx >= 0) & (cy >= 0) & (cx < cp.asarray(ws, dtype=dtype)) & (cy < cp.asarray(hs, dtype=dtype))
    cx = cp.where(valid_bounds, cx, cp.nan)
    cy = cp.where(valid_bounds, cy, cp.nan)
    return cx, cy


def _logquadratic_from_batch_gpu(g_batch, m_batch, hs, ws, bg_batch, *, dtype):
    B, Hm, Wm = g_batch.shape
    yy = cp.arange(Hm, dtype=dtype)
    xx = cp.arange(Wm, dtype=dtype)
    Y, X = cp.meshgrid(yy, xx, indexing="ij")
    ones = cp.ones_like(X, dtype=dtype)
    A = cp.stack([X * X, Y * Y, X * Y, X, Y, ones], axis=-1)

    v = g_batch - bg_batch[:, None, None]
    valid = m_batch & (v > dtype(0))
    z = cp.where(valid, cp.log(cp.maximum(v, dtype(1e-12))), dtype(0))

    Av = A[None, ...] * valid[..., None].astype(dtype)
    G = cp.einsum('bhwk,bhwl->bkl', Av, A[None, ...])
    rhs = cp.einsum('bhwk,bhw->bk', Av, z)

    eye = cp.eye(6, dtype=dtype)[None, :, :]
    G = G + eye * dtype(1e-9)
    coef = cp.linalg.solve(G, rhs[..., None]).squeeze(-1)

    a = coef[:, 0]
    b = coef[:, 1]
    c = coef[:, 2]
    d = coef[:, 3]
    e = coef[:, 4]

    detH = 4 * a * b - c * c
    safe_det = cp.where(cp.abs(detH) > dtype(1e-12), detH, cp.nan)
    cx = (c * e - 2 * b * d) / safe_det
    cy = (c * d - 2 * a * e) / safe_det

    counts = valid.sum(axis=(1, 2))
    negdef = (a < dtype(-1e-12)) & (b < dtype(-1e-12)) & (detH > dtype(1e-12))
    in_bounds = (cx >= 0) & (cy >= 0) & (cx < cp.asarray(ws, dtype=dtype)) & (cy < cp.asarray(hs, dtype=dtype))
    ok = (counts >= 6) & negdef & in_bounds & cp.isfinite(cx) & cp.isfinite(cy)

    cx = cp.where(ok, cx, cp.nan)
    cy = cp.where(ok, cy, cp.nan)
    return cx, cy





_OFF_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _offsets_square(rad: int):
    """Return (dy, dx) CuPy offset arrays for a square window.

    Caches NumPy arrays to avoid holding GPU memory across calls.
    """
    key = ("sq", int(rad))
    if key in _OFF_CACHE:
        dy_np, dx_np = _OFF_CACHE[key]
        return cp.asarray(dy_np), cp.asarray(dx_np)
    r = int(rad)
    ys = np.arange(-r, r + 1, dtype=np.int32)
    xs = np.arange(-r, r + 1, dtype=np.int32)
    dy, dx = np.meshgrid(ys, xs, indexing="ij")
    _OFF_CACHE[key] = (dy.ravel(), dx.ravel())
    return cp.asarray(_OFF_CACHE[key][0]), cp.asarray(_OFF_CACHE[key][1])


def clear_offset_cache():
    """Free cached offset arrays."""
    _OFF_CACHE.clear()


def _masked_median_lower(vals: cp.ndarray, mask: cp.ndarray) -> cp.ndarray:
    """Masked lower-median per row. Returns +inf when the row has no valid points."""
    B, _M = vals.shape
    n = cp.sum(mask, axis=1).astype(cp.int32)
    v = cp.where(mask, vals, cp.inf)
    vs = cp.sort(v, axis=1)
    idx = cp.maximum((n - 1) // 2, 0)
    return vs[cp.arange(B, dtype=cp.int32), idx]


def _bg_from_border_gpu_batch(g_batch, hs, ws, border=2):
    """Per-ROI background from border pixels on GPU.

    g_batch: (B, Hmax, Wmax) CuPy float — padded gray ROIs
    hs, ws:  (B,) CuPy int32 — actual height/width per ROI
    border:  int — border width (default 2, matching CPU _bg_from_border)
    Returns: (B,) CuPy float — per-ROI background estimate
    """
    B, Hmax, Wmax = g_batch.shape
    b = int(border)
    dy = cp.arange(Hmax, dtype=cp.int32)[None, :, None]
    dx = cp.arange(Wmax, dtype=cp.int32)[None, None, :]

    valid = (dy < hs[:, None, None]) & (dx < ws[:, None, None])
    is_border = valid & (
        (dy < b) | (dy >= hs[:, None, None] - b) |
        (dx < b) | (dx >= ws[:, None, None] - b)
    )
    # Tiny ROIs (h or w <= 2*border): use all valid pixels
    tiny = (hs <= 2 * b) | (ws <= 2 * b)
    is_border = cp.where(tiny[:, None, None], valid, is_border)

    bg = _masked_median_lower(
        g_batch.reshape(B, Hmax * Wmax),
        is_border.reshape(B, Hmax * Wmax),
    )
    return cp.where(cp.isfinite(bg), bg, 0.0)


def _refine_weighted_core(tile_gpu, labels_gpu, stats_gpu, lab_ids_gpu, *,
                           pad=3, border=2, batch=4096, use_float64=True):
    """Weighted centroid — CuPy in, CuPy out, no device context.

    tile_gpu:    (H, W) float32 CuPy
    labels_gpu:  (H, W) int32 CuPy
    stats_gpu:   (K, 6) float32 CuPy [x, y, w, h, cx, cy]
    lab_ids_gpu: (K,) int32 CuPy
    Returns:     (centers_gpu, ok_gpu) — float64 (K,2), bool (K,)
    """
    dtype = cp.float64 if use_float64 else cp.float32
    K = int(stats_gpu.shape[0])
    H, W = tile_gpu.shape

    # Padded bounding boxes
    x0 = cp.maximum(stats_gpu[:, 0].astype(cp.int32) - pad, 0)
    y0 = cp.maximum(stats_gpu[:, 1].astype(cp.int32) - pad, 0)
    x1 = cp.minimum((stats_gpu[:, 0] + stats_gpu[:, 2]).astype(cp.int32) + pad, W)
    y1 = cp.minimum((stats_gpu[:, 1] + stats_gpu[:, 3]).astype(cp.int32) + pad, H)
    hs = y1 - y0
    ws = x1 - x0

    centers_out = cp.full((K, 2), cp.nan, dtype=cp.float64)
    ok_out = cp.zeros((K,), dtype=cp.bool_)

    for s in range(0, K, int(batch)):
        e = min(K, s + int(batch))
        B = e - s
        hs_b = hs[s:e]
        ws_b = ws[s:e]
        x0_b = x0[s:e]
        y0_b = y0[s:e]
        Hmax = int(hs_b.max()) if B > 0 else 0
        Wmax = int(ws_b.max()) if B > 0 else 0

        if Hmax == 0 or Wmax == 0:
            continue

        # Build index grids via broadcasting
        dy = cp.arange(Hmax, dtype=cp.int32)[None, :, None]  # (1, Hmax, 1)
        dx = cp.arange(Wmax, dtype=cp.int32)[None, None, :]  # (1, 1, Wmax)

        ys = y0_b[:, None, None] + dy  # (B, Hmax, Wmax)
        xs = x0_b[:, None, None] + dx  # (B, Hmax, Wmax)

        valid = (dy < hs_b[:, None, None]) & (dx < ws_b[:, None, None])

        # Clamp for safe indexing (out-of-valid will be zeroed)
        ys_c = cp.clip(ys, 0, H - 1)
        xs_c = cp.clip(xs, 0, W - 1)

        # Gather grayscale and labels
        g_batch = cp.where(valid, tile_gpu[ys_c, xs_c], cp.float32(0))
        l_batch = cp.where(valid, labels_gpu[ys_c, xs_c], cp.int32(0))

        # Build masks: pixel belongs to this component
        m_batch = (l_batch == lab_ids_gpu[s:e, None, None]) & valid

        # Background estimation
        bg_batch = _bg_from_border_gpu_batch(g_batch.astype(dtype), hs_b, ws_b, border=border)

        # Weighted centroid
        cx, cy = _weighted_from_batch_gpu(
            g_batch.astype(dtype), m_batch, hs_b, ws_b, bg_batch, dtype=dtype,
        )

        # Convert ROI-local → tile-global
        cx_global = cx + x0_b.astype(dtype)
        cy_global = cy + y0_b.astype(dtype)

        ok_b = cp.isfinite(cx_global) & cp.isfinite(cy_global)
        centers_out[s:e, 0] = cx_global
        centers_out[s:e, 1] = cy_global
        ok_out[s:e] = ok_b

    return centers_out, ok_out


def _plane_fit_batched(x: cp.ndarray, y: cp.ndarray, g: cp.ndarray, m: cp.ndarray, eps_det: float = 1e-18):
    mf = m.astype(cp.float64)
    S1 = cp.sum(mf, axis=1)
    Sx = cp.sum(mf * x, axis=1)
    Sy = cp.sum(mf * y, axis=1)
    Sxx = cp.sum(mf * x * x, axis=1)
    Syy = cp.sum(mf * y * y, axis=1)
    Sxy = cp.sum(mf * x * y, axis=1)
    Sv = cp.sum(mf * g, axis=1)
    Sxv = cp.sum(mf * x * g, axis=1)
    Syv = cp.sum(mf * y * g, axis=1)
    Mmat = cp.stack([
        cp.stack([Sxx, Sxy, Sx], axis=1),
        cp.stack([Sxy, Syy, Sy], axis=1),
        cp.stack([Sx, Sy, S1], axis=1),
    ], axis=1)
    bvec = cp.stack([Sxv, Syv, Sv], axis=1)[:, :, None]
    det = cp.linalg.det(Mmat)
    ok = cp.isfinite(det) & (cp.abs(det) > eps_det) & (S1 >= 3)
    eye3 = cp.eye(3, dtype=cp.float64)[None, :, :]
    M_safe = cp.where(ok[:, None, None], Mmat, eye3)
    b_safe = cp.where(ok[:, None, None], bvec, 0.0)
    coef = cp.linalg.solve(M_safe, b_safe).squeeze(-1)
    return coef[:, 0], coef[:, 1], coef[:, 2], ok


def refine_centers_logquad_gpu_match_cpu(
    g_gpu: cp.ndarray,
    lab_gpu: cp.ndarray,
    stats_gpu: cp.ndarray,
    lab_ids_gpu: cp.ndarray,
    *,
    pad: int = 3,
    win_rad: int = 4,
    core_frac: float = 0.20,
    weight_power: float = 1.0,
    bg_dilate: int = 2,
    bg_win_pad: int = 2,
    bg_tail_frac: float = 0.05,
    plane_bg: bool = True,
    min_bg_pts: int = 12,
    min_plane_pts: int = 20,
    max_cond: float = 1e10,
    negdef_eps: float = 1e-12,
    fit_dilate: int = 1,
    eps: float = 1e-12,
    batch: int = 50_000,
    use_float64: bool = True,
):
    if g_gpu.ndim != 2 or g_gpu.dtype != cp.float32:
        raise ValueError(f"g_gpu must be 2D float32, got shape={g_gpu.shape} dtype={g_gpu.dtype}")
    if lab_gpu.ndim != 2 or lab_gpu.dtype != cp.int32:
        raise ValueError(f"lab_gpu must be 2D int32, got shape={lab_gpu.shape} dtype={lab_gpu.dtype}")
    if stats_gpu.ndim != 2 or stats_gpu.shape[0] != lab_ids_gpu.shape[0]:
        raise ValueError(f"stats_gpu rows ({stats_gpu.shape[0]}) must match lab_ids_gpu length ({lab_ids_gpu.shape[0]})")
    if stats_gpu.shape[1] not in (4, 6):
        raise ValueError(f"stats_gpu must have 4 or 6 columns, got {stats_gpu.shape[1]}")

    H, W = g_gpu.shape
    N = int(lab_ids_gpu.size)
    g = g_gpu.astype(cp.float64) if use_float64 else g_gpu.astype(cp.float32)
    labimg = lab_gpu
    lab_ids = lab_ids_gpu.astype(cp.int32, copy=False)

    x = stats_gpu[:, 0].astype(cp.int32)
    y = stats_gpu[:, 1].astype(cp.int32)
    w = stats_gpu[:, 2].astype(cp.int32)
    h = stats_gpu[:, 3].astype(cp.int32)
    x0 = cp.maximum(0, x - int(pad))
    y0 = cp.maximum(0, y - int(pad))
    x1 = cp.minimum(W, x + w + int(pad))
    y1 = cp.minimum(H, y + h + int(pad))

    if stats_gpu.shape[1] == 6:
        cxf = stats_gpu[:, 4].astype(cp.float64)
        cyf = stats_gpu[:, 5].astype(cp.float64)
    else:
        cxf = (x + (w // 2)).astype(cp.float64)
        cyf = (y + (h // 2)).astype(cp.float64)

    cxi = cp.rint(cxf).astype(cp.int32)
    cyi = cp.rint(cyf).astype(cp.int32)

    Rout = int(win_rad + bg_win_pad)
    dy_out, dx_out = _offsets_square(Rout)
    dy_w, dx_w = _offsets_square(int(win_rad))
    Mout = int(dx_out.size)
    K = int(dx_w.size)
    Pout = 2 * Rout + 1
    Pwin = 2 * int(win_rad) + 1

    PAD = int(Rout + 2)
    gp = cp.pad(g, pad_width=PAD, mode="constant", constant_values=0.0)
    lp = cp.pad(labimg, pad_width=PAD, mode="constant", constant_values=0)

    refined = cp.empty((N, 2), dtype=cp.float64)
    ok_all = cp.empty((N,), dtype=cp.bool_)

    for s in range(0, N, int(batch)):
        e = min(N, s + int(batch))
        B = e - s
        lab = lab_ids[s:e]
        cx = cxi[s:e]
        cy = cyi[s:e]
        cxp = (cx + PAD).astype(cp.int32)
        cyp = (cy + PAD).astype(cp.int32)

        xs_out = cxp[:, None] + dx_out[None, :]
        ys_out = cyp[:, None] + dy_out[None, :]
        g_out = gp[ys_out, xs_out]
        l_out = lp[ys_out, xs_out]

        xg_out = (cx[:, None] + dx_out[None, :]).astype(cp.int32)
        yg_out = (cy[:, None] + dy_out[None, :]).astype(cp.int32)
        roi_out = (xg_out >= x0[s:e][:, None]) & (xg_out < x1[s:e][:, None]) & (yg_out >= y0[s:e][:, None]) & (yg_out < y1[s:e][:, None])

        m0 = (l_out == lab[:, None]).reshape(B, Pout, Pout)
        if int(bg_dilate) > 0 and cndi is not None:
            k = int(2 * int(bg_dilate) + 1)
            dil = (cndi.maximum_filter(m0.astype(cp.uint8), size=(1, k, k)) > 0)
        else:
            dil = m0
        dil_vec = dil.reshape(B, Mout)

        bg_mask = roi_out & (~dil_vec)
        n_bg = cp.sum(bg_mask, axis=1)
        bg0 = _masked_median_lower(g_out, bg_mask)
        ok_bg = cp.isfinite(bg0) & (n_bg >= int(min_bg_pts))

        xs_w = cxp[:, None] + dx_w[None, :]
        ys_w = cyp[:, None] + dy_w[None, :]
        g_win = gp[ys_w, xs_w]
        l_win = lp[ys_w, xs_w]
        xg_win = (cx[:, None] + dx_w[None, :]).astype(cp.int32)
        yg_win = (cy[:, None] + dy_w[None, :]).astype(cp.int32)
        roi_win = (xg_win >= x0[s:e][:, None]) & (xg_win < x1[s:e][:, None]) & (yg_win >= y0[s:e][:, None]) & (yg_win < y1[s:e][:, None])

        v0_win = g_win - bg0[:, None]
        vmax0 = cp.max(cp.where(roi_win, v0_win, -cp.inf), axis=1)
        if float(bg_tail_frac) > 0.0:
            v0_bg = g_out - bg0[:, None]
            thr = (float(bg_tail_frac) * vmax0)[:, None]
            bg_mask2 = bg_mask & (v0_bg <= thr)
            n_bg2 = cp.sum(bg_mask2, axis=1)
            bg0_2 = _masked_median_lower(g_out, bg_mask2)
            use2 = cp.isfinite(bg0_2) & (n_bg2 >= int(min_bg_pts))
            bg0 = cp.where(use2, bg0_2, bg0)
            bg_mask_used = cp.where(use2[:, None], bg_mask2, bg_mask)
        else:
            bg_mask_used = bg_mask

        if plane_bg:
            n_plane = cp.sum(bg_mask_used, axis=1)
            ok_plane = ok_bg & (n_plane >= int(min_plane_pts))
            a_bg, b_bg, c_bg, ok_p = _plane_fit_batched(
                xg_out.astype(cp.float64),
                yg_out.astype(cp.float64),
                g_out.astype(cp.float64),
                bg_mask_used,
            )
            ok_plane = ok_plane & ok_p
            plane_win = a_bg[:, None] * xg_win.astype(cp.float64) + b_bg[:, None] * yg_win.astype(cp.float64) + c_bg[:, None]
            v_win = g_win.astype(cp.float64) - plane_win
            v_win = cp.where(ok_plane[:, None], v_win, (g_win.astype(cp.float64) - bg0[:, None]))
            ok_bg2 = ok_bg
        else:
            v_win = g_win.astype(cp.float64) - bg0[:, None]
            ok_bg2 = ok_bg

        in_blob = (l_win == lab[:, None]) & roi_win
        if int(fit_dilate) > 0 and cndi is not None:
            m = in_blob.reshape(B, Pwin, Pwin).astype(cp.uint8)
            k = int(2 * int(fit_dilate) + 1)
            md = (cndi.maximum_filter(m, size=(1, k, k)) > 0)
            in_blob = (md.reshape(B, K)) & roi_win

        fit = in_blob & (v_win > 0) & cp.isfinite(v_win)
        vmax = cp.max(cp.where(fit, v_win, -cp.inf), axis=1)
        ok_fit = ok_bg2 & cp.isfinite(vmax) & (vmax > 0)
        core_thr = float(core_frac) * vmax
        core = fit & (v_win >= core_thr[:, None])
        n_core = cp.sum(core, axis=1)
        sel = cp.where((n_core >= 6)[:, None], core, fit)

        n_sel = cp.sum(sel, axis=1)
        ok_sel = ok_fit & (n_sel >= 6)

        vv = cp.maximum(v_win, float(eps))
        z = cp.log(vv)
        wts = (vv ** float(weight_power)) * sel.astype(cp.float64)

        wpad = cp.where(sel, wts, cp.inf)
        wsort = cp.sort(wpad, axis=1)
        idx99 = cp.maximum((cp.floor(0.99 * (n_sel.astype(cp.float64) - 1.0))).astype(cp.int32), 0)
        thr99 = wsort[cp.arange(B, dtype=cp.int32), idx99]
        thr99 = cp.where(cp.isfinite(thr99), thr99, cp.max(cp.where(sel, wts, 0.0), axis=1))
        wts = cp.minimum(wts, thr99[:, None])

        sel_f = sel.astype(cp.float64)
        n = cp.maximum(cp.sum(sel_f, axis=1), 1.0)
        xg = xg_win.astype(cp.float64)
        yg = yg_win.astype(cp.float64)
        xm = cp.sum(sel_f * xg, axis=1) / n
        ym = cp.sum(sel_f * yg, axis=1) / n
        ex2 = cp.sum(sel_f * xg * xg, axis=1) / n
        ey2 = cp.sum(sel_f * yg * yg, axis=1) / n
        varx = cp.maximum(ex2 - xm * xm, 0.0)
        vary = cp.maximum(ey2 - ym * ym, 0.0)
        scls = cp.maximum(cp.maximum(cp.sqrt(varx), cp.sqrt(vary)), 1.0)
        xn = (xg - xm[:, None]) / scls[:, None]
        yn = (yg - ym[:, None]) / scls[:, None]

        p0 = xn * xn
        p1 = yn * yn
        p2 = xn * yn
        p3 = xn
        p4 = yn
        p5 = cp.ones_like(p0)
        P = cp.stack([p0, p1, p2, p3, p4, p5], axis=1)
        wz = wts * z
        bvec = cp.sum(P * wz[:, None, :], axis=2)
        Pw = P * wts[:, None, :]
        Mmat = Pw @ cp.transpose(P, (0, 2, 1))

        eye6 = cp.eye(6, dtype=cp.float64)[None, :, :]
        M_safe = cp.where(ok_sel[:, None, None], Mmat, eye6)
        b_safe = cp.where(ok_sel[:, None], bvec, 0.0)
        coef = cp.linalg.solve(M_safe, b_safe[..., None]).squeeze(-1)
        a, bq, c, d, e1, _f = [coef[:, i] for i in range(6)]

        H00 = 2.0 * a
        H11 = 2.0 * bq
        H01 = c
        tr = H00 + H11
        disc = cp.sqrt(cp.maximum((H00 - H11) * (H00 - H11) + 4.0 * H01 * H01, 0.0))
        lam1 = 0.5 * (tr - disc)
        lam2 = 0.5 * (tr + disc)
        ok_peak = ok_sel & (lam1 < -float(negdef_eps)) & (lam2 < -float(negdef_eps))
        condH = cp.abs(lam2) / cp.maximum(cp.abs(lam1), 1e-30)
        ok_peak = ok_peak & (condH <= float(max_cond))

        det = H00 * H11 - H01 * H01
        ok_peak = ok_peak & (cp.abs(det) > 1e-20)
        cxn = (-d * H11 + e1 * H01) / det
        cyn = (d * H01 - e1 * H00) / det
        cxg = xm + scls * cxn
        cyg = ym + scls * cyn
        ok_in = (cxg >= (cxf[s:e] - float(win_rad))) & (cxg <= (cxf[s:e] + float(win_rad))) & (cyg >= (cyf[s:e] - float(win_rad))) & (cyg <= (cyf[s:e] + float(win_rad)))
        ok = ok_peak & ok_in
        rx = cp.where(ok, cxg, cxf[s:e])
        ry = cp.where(ok, cyg, cyf[s:e])
        refined[s:e, 0] = rx
        refined[s:e, 1] = ry
        ok_all[s:e] = ok

    return refined, ok_all
def _smooth_binom5_gpu(x):
    xm2 = cp.concatenate([x[..., :1], x[..., :1], x[..., :-2]], axis=-1)
    xm1 = cp.concatenate([x[..., :1], x[..., :-1]], axis=-1)
    xp1 = cp.concatenate([x[..., 1:], x[..., -1:]], axis=-1)
    xp2 = cp.concatenate([x[..., 2:], x[..., -1:], x[..., -1:]], axis=-1)
    return (xm2 + 4 * xm1 + 6 * x + 4 * xp1 + xp2) * (1.0 / 16.0)


def _edge_from_profile_gradmoment_gpu(prof, base_x, *, grad_power: float = 8.0, loc_rad: int = 3):
    d = cp.abs(prof[:, 1:] - prof[:, :-1])
    B, Lm1 = d.shape
    if Lm1 <= 0:
        return cp.full((B,), cp.nan, dtype=cp.float64), cp.zeros((B,), dtype=bool)
    ip = cp.argmax(d, axis=1).astype(cp.int32)
    ip0 = cp.clip(ip, loc_rad, max(loc_rad, Lm1 - 1 - loc_rad))
    offs = cp.arange(-loc_rad, loc_rad + 1, dtype=cp.int32)
    jj = ip0[:, None] + offs[None, :]
    jj = cp.clip(jj, 0, Lm1 - 1)
    row = cp.arange(B, dtype=cp.int32)[:, None]
    dloc = d[row, jj].astype(cp.float32)
    w = dloc ** float(grad_power)
    sw = cp.sum(w, axis=1) + 1e-20
    k = jj.astype(cp.float32)
    xk = base_x[:, None] + (k + 0.5)
    x_edge = cp.sum(w * xk, axis=1) / sw
    ok = sw > 1e-12
    return x_edge, ok


def _refine_edge_moment_core(gray_gpu, stats_gpu, *,
                              band_rad=4, edge_rad=10, smooth_passes=2,
                              loc_rad=3, grad_power=8.0, iters=2):
    """CuPy in, CuPy out, no device context.

    gray_gpu:  (H, W) float32 CuPy
    stats_gpu: (K, 6) float32 CuPy [x, y, w, h, cx, cy]
    Returns:   (centers_gpu, ok_gpu) — CuPy float64 (K,2), CuPy bool (K,)
    """
    H, W = map(int, gray_gpu.shape)
    K = int(stats_gpu.shape[0])
    w = stats_gpu[:, 2].astype(cp.float32)
    h = stats_gpu[:, 3].astype(cp.float32)
    xc = stats_gpu[:, 4].astype(cp.float32)
    yc = stats_gpu[:, 5].astype(cp.float32)
    By = 2 * band_rad + 1
    L = 2 * edge_rad + 1
    dy = cp.arange(-band_rad, band_rad + 1, dtype=cp.int32)
    dx = cp.arange(-edge_rad, edge_rad + 1, dtype=cp.int32)
    gflat = gray_gpu.ravel()
    ok_all = cp.ones((K,), dtype=bool)
    for _ in range(int(iters)):
        xL0 = cp.rint(xc - 0.5 * (w - 1.0)).astype(cp.int32)
        xR0 = cp.rint(xc + 0.5 * (w - 1.0)).astype(cp.int32)
        yT0 = cp.rint(yc - 0.5 * (h - 1.0)).astype(cp.int32)
        yB0 = cp.rint(yc + 0.5 * (h - 1.0)).astype(cp.int32)
        ys = cp.clip(cp.rint(yc).astype(cp.int32)[:, None] + dy[None, :], 0, H - 1)
        def prof_at_x(x0_int):
            xs = cp.clip(x0_int[:, None] + dx[None, :], 0, W - 1)
            idx = (ys[:, :, None] * W + xs[:, None, :]).astype(cp.int64)
            patch = cp.take(gflat, idx.ravel()).reshape(K, By, L)
            prof = cp.mean(patch, axis=1)
            for _ in range(int(smooth_passes)):
                prof = _smooth_binom5_gpu(prof)
            base = x0_int.astype(cp.float32) - edge_rad
            return prof, base
        profL, baseL = prof_at_x(xL0)
        profR, baseR = prof_at_x(xR0)
        xL, okL = _edge_from_profile_gradmoment_gpu(profL, baseL, grad_power=grad_power, loc_rad=loc_rad)
        xR, okR = _edge_from_profile_gradmoment_gpu(profR, baseR, grad_power=grad_power, loc_rad=loc_rad)
        xs_band = cp.clip(cp.rint(xc).astype(cp.int32)[:, None] + dy[None, :], 0, W - 1)
        def prof_at_y(y0_int):
            ys2 = cp.clip(y0_int[:, None] + dx[None, :], 0, H - 1)
            idx2 = (ys2[:, None, :] * W + xs_band[:, :, None]).astype(cp.int64)
            patch2 = cp.take(gflat, idx2.ravel()).reshape(K, By, L)
            prof = cp.mean(patch2, axis=1)
            for _ in range(int(smooth_passes)):
                prof = _smooth_binom5_gpu(prof)
            base = y0_int.astype(cp.float32) - edge_rad
            return prof, base
        profT, baseT = prof_at_y(yT0)
        profB, baseB = prof_at_y(yB0)
        yT, okT = _edge_from_profile_gradmoment_gpu(profT, baseT, grad_power=grad_power, loc_rad=loc_rad)
        yB, okB = _edge_from_profile_gradmoment_gpu(profB, baseB, grad_power=grad_power, loc_rad=loc_rad)
        ok_iter = okL & okR & okT & okB
        ok_all &= ok_iter
        xc = 0.5 * (xL + xR)
        yc = 0.5 * (yT + yB)
    centers = cp.stack([xc, yc], axis=1).astype(cp.float64)
    return centers, ok_all


def refine_centers_edge_moment_gpu(
    gray: np.ndarray,
    stats_xywh_cc: np.ndarray,
    *,
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    device: int = 0,
):
    _require_cupy()
    g_np = np.asarray(gray, dtype=np.float32)
    stats_np = np.asarray(stats_xywh_cc, dtype=np.float32)
    if g_np.ndim != 2:
        raise ValueError("gray must be a 2D grayscale image")
    if stats_np.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    with gpu_device(device):
        centers, ok = _refine_edge_moment_core(
            cp.asarray(g_np, dtype=cp.float32),
            cp.asarray(stats_np, dtype=cp.float32),
            band_rad=band_rad, edge_rad=edge_rad,
            smooth_passes=smooth_passes, loc_rad=loc_rad,
            grad_power=grad_power, iters=iters,
        )
        return cp.asnumpy(centers), cp.asnumpy(ok)



def _erf_gpu(x):
    if cperf is not None:
        return cperf(x)
    if cp is not None and hasattr(cp, "erf"):
        return cp.erf(x)
    raise RuntimeError("CuPy erf support is required for edge_erf refinement.")


def _edge_polarity_from_profile_gpu(prof, base_x, x_seed, *, loc_rad: int = 2):
    prof = prof.astype(cp.float64)
    d = prof[:, 1:] - prof[:, :-1]
    B, Lm1 = d.shape
    if Lm1 <= 0:
        return cp.ones((prof.shape[0],), dtype=cp.float64)
    xk = base_x[:, None].astype(cp.float64) + (cp.arange(Lm1, dtype=cp.float64)[None, :] + 0.5)
    ip = cp.argmin(cp.abs(xk - x_seed[:, None].astype(cp.float64)), axis=1).astype(cp.int32)
    ip = cp.clip(ip, 0, max(0, Lm1 - 1))
    offs = cp.arange(-int(loc_rad), int(loc_rad) + 1, dtype=cp.int32)
    jj = cp.clip(ip[:, None] + offs[None, :], 0, max(0, Lm1 - 1))
    row = cp.arange(B, dtype=cp.int32)[:, None]
    sgn = cp.sum(d[row, jj], axis=1)
    pol = cp.where(sgn >= 0, 1.0, -1.0).astype(cp.float64)
    return pol


def _edge_from_profile_erf_gpu(
    prof,
    base_x,
    x_seed,
    *,
    fit_rad: int = 5,
    gn_iters: int = 5,
    init_sigma: float = 1.25,
    sigma_min: float = 0.35,
    sigma_max: float = 6.0,
    max_shift: float = 2.5,
    damp: float = 1e-4,
    cond_max: float = 1e8,
    loc_rad: int = 2,
):
    prof = prof.astype(cp.float64)
    base_x = base_x.astype(cp.float64)
    x_seed = x_seed.astype(cp.float64)
    B, L = prof.shape
    if L < 5:
        return x_seed.astype(cp.float64), cp.zeros((B,), dtype=bool)

    xs = base_x[:, None] + cp.arange(L, dtype=cp.float64)[None, :]
    pol = _edge_polarity_from_profile_gpu(prof, base_x, x_seed, loc_rad=loc_rad)

    ymin = cp.min(prof, axis=1)
    ymax = cp.max(prof, axis=1)
    amp_floor = cp.maximum((ymax - ymin) * 1e-6, 1e-9)

    a = ymin.copy()
    b = cp.zeros((B,), dtype=cp.float64)
    c = cp.maximum(ymax - ymin, amp_floor)
    x0 = x_seed.copy()
    lsig = cp.full((B,), float(np.log(max(float(init_sigma), 1e-6))), dtype=cp.float64)

    fit_rad_f = float(max(2, int(fit_rad)))
    max_shift_f = float(max(0.25, max_shift))
    eye5 = cp.eye(5, dtype=cp.float64)[None, :, :]
    sqrt_2 = float(np.sqrt(2.0))
    sqrt_pi = float(np.sqrt(np.pi))
    sqrt_2pi = float(np.sqrt(2.0 * np.pi))

    ok = cp.ones((B,), dtype=bool)

    for _ in range(int(gn_iters)):
        sig = cp.clip(cp.exp(lsig), float(sigma_min), float(sigma_max))
        z = (xs - x0[:, None]) / (sqrt_2 * sig[:, None])
        s = 0.5 * (1.0 + pol[:, None] * _erf_gpu(z))
        model = a[:, None] + b[:, None] * xs + c[:, None] * s

        w_fit = (cp.abs(xs - x0[:, None]) <= fit_rad_f).astype(cp.float64)
        if L >= 7:
            w_fit += 0.15 * (cp.abs(xs - x0[:, None]) <= (fit_rad_f + 1.0)).astype(cp.float64)
        sw = cp.sqrt(w_fit)
        r = (prof - model) * sw

        ez = cp.exp(-(z * z))
        ds_dx0 = (-pol[:, None] * ez / (sqrt_2pi * sig[:, None]))
        ds_dlsig = (-pol[:, None] * z * ez / sqrt_pi)

        J = cp.stack([
            sw,
            sw * xs,
            sw * s,
            sw * (c[:, None] * ds_dx0),
            sw * (c[:, None] * ds_dlsig),
        ], axis=2)

        JTJ = cp.einsum('bik,bij->bkj', J, J)
        JTr = cp.einsum('bik,bi->bk', J, r)

        diag = cp.maximum(cp.diagonal(JTJ, axis1=1, axis2=2), 1e-9)
        JTJ = JTJ + float(damp) * diag[:, :, None] * eye5

        try:
            delta = cp.linalg.solve(JTJ, JTr[..., None]).squeeze(-1)
        except Exception:
            delta = cp.zeros((B, 5), dtype=cp.float64)
        bad_delta = ~cp.all(cp.isfinite(delta), axis=1)
        delta = cp.where(bad_delta[:, None], 0.0, delta)
        ok &= ~bad_delta

        delta[:, 3] = cp.clip(delta[:, 3], -0.75, 0.75)
        delta[:, 4] = cp.clip(delta[:, 4], -0.35, 0.35)

        a = a + delta[:, 0]
        b = b + delta[:, 1]
        c = cp.maximum(c + delta[:, 2], amp_floor)
        x0 = cp.clip(x0 + delta[:, 3], x_seed - max_shift_f, x_seed + max_shift_f)
        lsig = cp.clip(lsig + delta[:, 4], float(np.log(sigma_min)), float(np.log(sigma_max)))

    # Final condition check (only on last iteration's JTJ)
    cond = cp.linalg.cond(JTJ)
    ok &= cp.isfinite(cond) & (cond < float(cond_max))

    sig = cp.clip(cp.exp(lsig), float(sigma_min), float(sigma_max))
    z = (xs - x0[:, None]) / (sqrt_2 * sig[:, None])
    s = 0.5 * (1.0 + pol[:, None] * _erf_gpu(z))
    model = a[:, None] + b[:, None] * xs + c[:, None] * s
    w_fit = (cp.abs(xs - x0[:, None]) <= fit_rad_f).astype(cp.float64)
    n_fit = cp.maximum(cp.sum(w_fit, axis=1), 1.0)
    rms = cp.sqrt(cp.sum(w_fit * (prof - model) ** 2, axis=1) / n_fit)

    dyn = cp.maximum(ymax - ymin, 1e-9)
    ok &= cp.isfinite(x0) & cp.isfinite(sig) & cp.isfinite(rms)
    ok &= (cp.abs(x0 - x_seed) <= (max_shift_f + 1e-6))
    ok &= (sig >= float(sigma_min)) & (sig <= float(sigma_max))
    ok &= (c > amp_floor)
    ok &= (rms <= (0.35 * dyn + 1e-6))

    x_out = cp.where(ok, x0, cp.nan)
    return x_out.astype(cp.float64), ok


def _refine_edge_erf_core(gray_gpu, stats_gpu, *,
                           band_rad=4, edge_rad=10, smooth_passes=2,
                           loc_rad=3, grad_power=8.0, iters=2,
                           erf_fit_rad=5, erf_gn_iters=5,
                           erf_init_sigma=1.25, erf_sigma_min=0.35,
                           erf_sigma_max=6.0, erf_max_shift=2.5,
                           erf_damp=1e-4, erf_cond_max=1e8):
    """CuPy in, CuPy out, no device context.

    gray_gpu:  (H, W) float32 CuPy
    stats_gpu: (K, 6) float32 CuPy [x, y, w, h, cx, cy]
    Returns:   (centers_gpu, ok_gpu) — CuPy float64 (K,2), CuPy bool (K,)
    """
    H, W = map(int, gray_gpu.shape)
    K = int(stats_gpu.shape[0])
    w = stats_gpu[:, 2].astype(cp.float32)
    h = stats_gpu[:, 3].astype(cp.float32)
    xc = stats_gpu[:, 4].astype(cp.float64)
    yc = stats_gpu[:, 5].astype(cp.float64)
    By = 2 * int(band_rad) + 1
    L = 2 * int(edge_rad) + 1
    dy = cp.arange(-int(band_rad), int(band_rad) + 1, dtype=cp.int32)
    dx = cp.arange(-int(edge_rad), int(edge_rad) + 1, dtype=cp.int32)
    gflat = gray_gpu.ravel()
    ok_all = cp.ones((K,), dtype=bool)

    for _ in range(int(iters)):
        xL0 = cp.rint(xc - 0.5 * (w.astype(cp.float64) - 1.0)).astype(cp.int32)
        xR0 = cp.rint(xc + 0.5 * (w.astype(cp.float64) - 1.0)).astype(cp.int32)
        yT0 = cp.rint(yc - 0.5 * (h.astype(cp.float64) - 1.0)).astype(cp.int32)
        yB0 = cp.rint(yc + 0.5 * (h.astype(cp.float64) - 1.0)).astype(cp.int32)

        ys = cp.clip(cp.rint(yc).astype(cp.int32)[:, None] + dy[None, :], 0, H - 1)

        def prof_at_x(x0_int):
            xs = cp.clip(x0_int[:, None] + dx[None, :], 0, W - 1)
            idx = (ys[:, :, None] * W + xs[:, None, :]).astype(cp.int64)
            patch = cp.take(gflat, idx.ravel()).reshape(K, By, L)
            prof = cp.mean(patch, axis=1)
            for _ in range(int(smooth_passes)):
                prof = _smooth_binom5_gpu(prof)
            base = x0_int.astype(cp.float64) - float(edge_rad)
            return prof.astype(cp.float64), base

        profL, baseL = prof_at_x(xL0)
        profR, baseR = prof_at_x(xR0)
        xL_seed, okL0 = _edge_from_profile_gradmoment_gpu(profL, baseL.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
        xR_seed, okR0 = _edge_from_profile_gradmoment_gpu(profR, baseR.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
        xL, okL1 = _edge_from_profile_erf_gpu(
            profL, baseL, xL_seed,
            fit_rad=erf_fit_rad, gn_iters=erf_gn_iters, init_sigma=erf_init_sigma,
            sigma_min=erf_sigma_min, sigma_max=erf_sigma_max, max_shift=erf_max_shift,
            damp=erf_damp, cond_max=erf_cond_max, loc_rad=max(1, int(loc_rad) - 1),
        )
        xR, okR1 = _edge_from_profile_erf_gpu(
            profR, baseR, xR_seed,
            fit_rad=erf_fit_rad, gn_iters=erf_gn_iters, init_sigma=erf_init_sigma,
            sigma_min=erf_sigma_min, sigma_max=erf_sigma_max, max_shift=erf_max_shift,
            damp=erf_damp, cond_max=erf_cond_max, loc_rad=max(1, int(loc_rad) - 1),
        )

        xs_band = cp.clip(cp.rint(xc).astype(cp.int32)[:, None] + dy[None, :], 0, W - 1)

        def prof_at_y(y0_int):
            ys2 = cp.clip(y0_int[:, None] + dx[None, :], 0, H - 1)
            idx2 = (ys2[:, None, :] * W + xs_band[:, :, None]).astype(cp.int64)
            patch2 = cp.take(gflat, idx2.ravel()).reshape(K, By, L)
            prof = cp.mean(patch2, axis=1)
            for _ in range(int(smooth_passes)):
                prof = _smooth_binom5_gpu(prof)
            base = y0_int.astype(cp.float64) - float(edge_rad)
            return prof.astype(cp.float64), base

        profT, baseT = prof_at_y(yT0)
        profB, baseB = prof_at_y(yB0)
        yT_seed, okT0 = _edge_from_profile_gradmoment_gpu(profT, baseT.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
        yB_seed, okB0 = _edge_from_profile_gradmoment_gpu(profB, baseB.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
        yT, okT1 = _edge_from_profile_erf_gpu(
            profT, baseT, yT_seed,
            fit_rad=erf_fit_rad, gn_iters=erf_gn_iters, init_sigma=erf_init_sigma,
            sigma_min=erf_sigma_min, sigma_max=erf_sigma_max, max_shift=erf_max_shift,
            damp=erf_damp, cond_max=erf_cond_max, loc_rad=max(1, int(loc_rad) - 1),
        )
        yB, okB1 = _edge_from_profile_erf_gpu(
            profB, baseB, yB_seed,
            fit_rad=erf_fit_rad, gn_iters=erf_gn_iters, init_sigma=erf_init_sigma,
            sigma_min=erf_sigma_min, sigma_max=erf_sigma_max, max_shift=erf_max_shift,
            damp=erf_damp, cond_max=erf_cond_max, loc_rad=max(1, int(loc_rad) - 1),
        )

        ok_iter = okL0 & okR0 & okT0 & okB0 & okL1 & okR1 & okT1 & okB1
        ok_iter &= (xR > xL) & (yB > yT)
        ok_all &= ok_iter
        xc = 0.5 * (xL + xR)
        yc = 0.5 * (yT + yB)

    centers = cp.stack([xc, yc], axis=1).astype(cp.float64)
    return centers, ok_all


def refine_centers_edge_erf_gpu(
    gray: np.ndarray,
    stats_xywh_cc: np.ndarray,
    *,
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    erf_fit_rad: int = 5,
    erf_gn_iters: int = 5,
    erf_init_sigma: float = 1.25,
    erf_sigma_min: float = 0.35,
    erf_sigma_max: float = 6.0,
    erf_max_shift: float = 2.5,
    erf_damp: float = 1e-4,
    erf_cond_max: float = 1e8,
    device: int = 0,
):
    """Large-feature center refinement using gradient-moment seeds followed by 1D erf-edge fitting.

    This is a drop-in replacement for ``refine_centers_edge_moment_gpu(...)`` when you want
    more accurate edge positions for center / registration testing. Input ``stats_xywh_cc``
    must be ``(N, 6)`` or compatible rows ``[x, y, w, h, cx, cy]``.
    """
    _require_cupy()
    g_np = np.asarray(gray, dtype=np.float32)
    stats_np = np.asarray(stats_xywh_cc, dtype=np.float32)
    if g_np.ndim != 2:
        raise ValueError("gray must be a 2D grayscale image")
    if stats_np.size == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    if stats_np.ndim != 2 or stats_np.shape[1] < 6:
        raise ValueError("stats_xywh_cc must have shape (N, >=6) with columns [x, y, w, h, cx, cy]")
    with gpu_device(device):
        centers, ok = _refine_edge_erf_core(
            cp.asarray(g_np, dtype=cp.float32),
            cp.asarray(stats_np[:, :6], dtype=cp.float32),
            band_rad=band_rad, edge_rad=edge_rad,
            smooth_passes=smooth_passes, loc_rad=loc_rad,
            grad_power=grad_power, iters=iters,
            erf_fit_rad=erf_fit_rad, erf_gn_iters=erf_gn_iters,
            erf_init_sigma=erf_init_sigma, erf_sigma_min=erf_sigma_min,
            erf_sigma_max=erf_sigma_max, erf_max_shift=erf_max_shift,
            erf_damp=erf_damp, erf_cond_max=erf_cond_max,
        )
        return cp.asnumpy(centers), cp.asnumpy(ok)

def _refine_batch_gpu(rois: list[np.ndarray], masks: list[np.ndarray], *, method: str, use_float64: bool, device: int):
    _require_cupy()
    if not rois:
        return np.zeros((0, 2), dtype=np.float64)

    rois_pad, masks_pad, hs, ws, bgs = _pad_rois(rois, masks)
    dtype = cp.float64 if use_float64 else cp.float32

    with gpu_device(device):
        g_batch = cp.asarray(rois_pad, dtype=dtype)
        m_batch = cp.asarray(masks_pad)
        bg_batch = cp.asarray(bgs, dtype=dtype)

        if method in {"edge_gradmoment", "edge_moment", "edges_centered"}:
            raise ValueError("edge_gradmoment is only supported through detect_centers_gpu(...) or refine_centers_edge_moment_gpu(...)")
        if method == "weighted":
            cx, cy = _weighted_from_batch_gpu(g_batch, m_batch, hs, ws, bg_batch, dtype=dtype)
        elif method in {"logquad", "logquadratic"}:
            cx, cy = _logquadratic_from_batch_gpu(g_batch, m_batch, hs, ws, bg_batch, dtype=dtype)
        elif method == "none":
            cx, cy = _weighted_from_batch_gpu(g_batch, m_batch, hs, ws, bg_batch, dtype=dtype)
        else:
            raise ValueError("method must be one of: 'none', 'weighted', 'logquad'.")

        out = cp.stack([cx, cy], axis=1)
        return cp.asnumpy(out)


def refine_weighted_centroid_gpu(gray_roi: np.ndarray, mask_roi: np.ndarray, *, device: int = 0, use_float64: bool = True) -> tuple[float, float]:
    out = _refine_batch_gpu([np.asarray(gray_roi)], [np.asarray(mask_roi, dtype=bool)], method="weighted", use_float64=use_float64, device=device)
    return float(out[0, 0]), float(out[0, 1])


def refine_logquadratic_gpu(gray_roi: np.ndarray, mask_roi: np.ndarray, *, device: int = 0, use_float64: bool = True) -> tuple[float, float]:
    out = _refine_batch_gpu([np.asarray(gray_roi)], [np.asarray(mask_roi, dtype=bool)], method="logquad", use_float64=use_float64, device=device)
    return float(out[0, 0]), float(out[0, 1])


def refine_centers_gpu(gray_roi: np.ndarray, mask_roi: np.ndarray, *, method: str = "logquad", device: int = 0, use_float64: bool = True) -> tuple[float, float]:
    out = _refine_batch_gpu([np.asarray(gray_roi)], [np.asarray(mask_roi, dtype=bool)], method=method, use_float64=use_float64, device=device)
    return float(out[0, 0]), float(out[0, 1])


# ---------------------------------------------------------------------------
# Parthasarathy radial-symmetry batch GPU kernel
# ---------------------------------------------------------------------------

def _radial_symmetry_batch_gpu(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    upsample_factor: int = 4,
    boundary_margin: int | None = None,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch Parthasarathy radial symmetry center estimation on GPU.

    Parameters
    ----------
    rois : (N, H, W) float64 array -- stacked ROIs (background-filled outside mask)
    masks : (N, H, W) bool array -- Voronoi cell masks (True = valid pixel)
    upsample_factor : int -- bicubic upsampling factor (1 = disabled)
    boundary_margin : int or None -- gradient margin near Voronoi boundary
    device : int -- GPU device

    Returns
    -------
    centers : (N, 2) float64 array -- centers in ROI pixel coordinates [x, y]
    residuals : (N,) float64 array -- goodness-of-fit metric per blob
    """
    _require_cupy()

    rois = np.asarray(rois, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    N = rois.shape[0]

    if N == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    if boundary_margin is None:
        boundary_margin = upsample_factor

    with gpu_device(device):
        # -- Step 1: Track per-ROI original dimensions (from mask extent) --
        orig_hs = np.array([rois.shape[1]] * N, dtype=np.int32)
        orig_ws = np.array([rois.shape[2]] * N, dtype=np.int32)

        # -- Step 2 & 3: Upsample (single batch kernel launch) --
        factor = int(upsample_factor)
        if factor > 1:
            from .upsample import batch_bicubic_upsample, batch_nn_upsample
            rois_gpu = cp.asarray(rois, dtype=cp.float64)
            masks_gpu = cp.asarray(masks.astype(np.float64))
            g_batch = batch_bicubic_upsample(rois_gpu, factor)
            m_batch = batch_nn_upsample(masks_gpu, factor) > 0.5
            up_hs = np.full(N, g_batch.shape[1], dtype=np.int32)
            up_ws = np.full(N, g_batch.shape[2], dtype=np.int32)
            Hmax = g_batch.shape[1]
            Wmax = g_batch.shape[2]
        else:
            g_batch = cp.asarray(rois)
            m_batch = cp.asarray(masks)
            up_hs = orig_hs.copy()
            up_ws = orig_ws.copy()
            Hmax = g_batch.shape[1]
            Wmax = g_batch.shape[2]

        # -- Step 4: Erode mask by boundary_margin --
        # Adaptively cap the margin so small ROIs retain enough valid
        # midpoints.  After erosion of margin m the remaining valid region
        # is roughly (H - 2m) x (W - 2m) and the midpoint grid is one
        # smaller in each dimension.  We require the eroded region to be
        # large enough to yield at least a ~10x10 midpoint grid for
        # robust fitting, so cap at (min_dim - 11) // 2.  For very small
        # ROIs this naturally reduces to 0.
        min_up_dim = int(min(int(up_hs.min()), int(up_ws.min())))
        max_margin = max(0, (min_up_dim - 11) // 2)
        eff_margin = min(boundary_margin, max_margin)
        if eff_margin > 0:
            struct = cp.ones((1, 2 * eff_margin + 1, 2 * eff_margin + 1), dtype=bool)
            m_eroded = cndi.binary_erosion(m_batch, structure=struct)
        else:
            m_eroded = m_batch.copy()

        # -- Step 5: Diagonal gradients at midpoints --
        # I[i,j] notation: i=row, j=col.  Midpoint between (i,j), (i+1,j), (i,j+1), (i+1,j+1)
        # is at (i+0.5, j+0.5).
        # dIdu = I[i, j+1] - I[i+1, j]  (diagonal u)
        # dIdv = I[i, j]   - I[i+1, j+1] (diagonal v)
        I = g_batch
        Hp = Hmax - 1  # midpoint grid height
        Wp = Wmax - 1  # midpoint grid width

        if Hp < 1 or Wp < 1:
            # Degenerate case: ROIs too small for gradients
            centers = np.full((N, 2), np.nan, dtype=np.float64)
            residuals = np.full((N,), np.inf, dtype=np.float64)
            return centers, residuals

        dIdu = I[:, :Hp, 1:Wp+1] - I[:, 1:Hp+1, :Wp]       # (N, Hp, Wp)
        dIdv = I[:, :Hp, :Wp]    - I[:, 1:Hp+1, 1:Wp+1]     # (N, Hp, Wp)

        # -- Step 6: Midpoint validity mask: all 4 surrounding pixels inside eroded mask --
        m00 = m_eroded[:, :Hp, :Wp]
        m01 = m_eroded[:, :Hp, 1:Wp+1]
        m10 = m_eroded[:, 1:Hp+1, :Wp]
        m11 = m_eroded[:, 1:Hp+1, 1:Wp+1]
        valid = m00 & m01 & m10 & m11  # (N, Hp, Wp)

        # -- Step 7: Slope m = -(dIdv + dIdu) / (dIdu - dIdv), clamp near-vertical --
        denom = dIdu - dIdv
        near_zero = cp.abs(denom) < 1e-12
        safe_denom = cp.where(near_zero, cp.float64(1.0), denom)
        slope = -(dIdv + dIdu) / safe_denom
        # Clamp near-vertical slopes
        slope = cp.where(near_zero, cp.where(-(dIdv + dIdu) >= 0, cp.float64(1e9), cp.float64(-1e9)), slope)

        # -- Step 8: ROI-centered midpoint coordinates --
        # Midpoint (i+0.5, j+0.5) in pixel coords. We want coordinates centered
        # on the ROI center: xm, ym relative to center of each ROI's upsampled grid.
        # Per-ROI centering: for ROI i, center_x = (up_ws[i]-1)/2, center_y = (up_hs[i]-1)/2
        # Midpoint pixel coords: col = j + 0.5, row = i_row + 0.5
        col_idx = cp.arange(Wp, dtype=cp.float64)[None, None, :] + 0.5  # (1, 1, Wp)
        row_idx = cp.arange(Hp, dtype=cp.float64)[None, :, None] + 0.5  # (1, Hp, 1)

        # Per-ROI center offsets
        up_ws_g = cp.asarray(up_ws, dtype=cp.float64)
        up_hs_g = cp.asarray(up_hs, dtype=cp.float64)
        cx_off = (up_ws_g - 1.0) / 2.0  # (N,)
        cy_off = (up_hs_g - 1.0) / 2.0  # (N,)

        xm = col_idx - cx_off[:, None, None]  # (N, Hp, Wp)
        ym = row_idx - cy_off[:, None, None]  # (N, Hp, Wp)

        # -- Step 9: Line intercepts b = ym - slope * xm --
        b = ym - slope * xm  # (N, Hp, Wp)

        # -- Step 10: Weights w = |grad|^2 / dist_to_gradient_centroid --
        grad_mag_sq = dIdu**2 + dIdv**2  # (N, Hp, Wp)

        # Gradient centroid per ROI (weighted by grad_mag_sq within valid mask)
        w_base = cp.where(valid, grad_mag_sq, cp.float64(0.0))
        w_sum = w_base.sum(axis=(1, 2))  # (N,)
        w_sum_safe = cp.maximum(w_sum, cp.float64(1e-30))
        gc_x = (w_base * xm).sum(axis=(1, 2)) / w_sum_safe  # (N,)
        gc_y = (w_base * ym).sum(axis=(1, 2)) / w_sum_safe  # (N,)

        dist_sq = (xm - gc_x[:, None, None])**2 + (ym - gc_y[:, None, None])**2
        dist = cp.sqrt(cp.maximum(dist_sq, cp.float64(1e-30)))

        w = cp.where(valid, grad_mag_sq / dist, cp.float64(0.0))  # (N, Hp, Wp)

        # -- Step 11: Analytic 2x2 solve --
        # Weighted least squares for intersection of lines y = m*x + b
        # Minimize sum_k w_k * (y_k - m_k * x_k - b_k)^2 over (xc, yc)
        # where each line constraint is: yc = slope_k * xc + b_k
        # Rearranging: slope_k * xc - yc + b_k = 0
        # Normal equations:
        #   [sum(m^2*w)  -sum(m*w)] [xc]   [-sum(m*b*w)]
        #   [-sum(m*w)    sum(w)  ] [yc] = [ sum(b*w)  ]
        sw = w.sum(axis=(1, 2))       # (N,)
        smmw = (slope**2 * w).sum(axis=(1, 2))  # (N,)
        smw = (slope * w).sum(axis=(1, 2))       # (N,)
        smbw = (slope * b * w).sum(axis=(1, 2))  # (N,)
        sbw = (b * w).sum(axis=(1, 2))           # (N,)

        det = smmw * sw - smw * smw  # (N,)
        det_safe = cp.where(cp.abs(det) < 1e-30, cp.float64(1e-30), det)
        xc = (-smbw * sw + smw * sbw) / det_safe   # (N,)
        yc = (smmw * sbw - smbw * smw) / det_safe   # (N,)

        # -- Step 12: Residual: weighted mean perpendicular distance squared --
        # perpendicular distance from point (xm, ym) to line yc = slope*xc + b
        # For each midpoint line: distance of (xc, yc) to line y = m*x + b
        # d = |m*xc - yc + b| / sqrt(m^2 + 1)
        perp_d_sq = (slope * xc[:, None, None] - yc[:, None, None] + b)**2 / (slope**2 + 1.0)
        residual = (w * perp_d_sq).sum(axis=(1, 2)) / cp.maximum(sw, cp.float64(1e-30))

        # -- Step 13: Per-ROI coordinate transform --
        # Transfer to CPU
        xc_cpu = cp.asnumpy(xc)  # (N,) centered coords in upsampled grid
        yc_cpu = cp.asnumpy(yc)
        residual_cpu = cp.asnumpy(residual)

    # Convert from centered upsampled coords to original ROI pixel coords
    # xc is relative to center of upsampled grid: xc_pix_up = xc + (w_up - 1) / 2
    # Then map back: xc_orig = xc_pix_up * (w_orig - 1) / (w_up - 1)
    centers = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        w_up_i = float(up_ws[i])
        h_up_i = float(up_hs[i])
        w_orig_i = float(orig_ws[i])
        h_orig_i = float(orig_hs[i])

        xc_pix_up = xc_cpu[i] + (w_up_i - 1.0) / 2.0
        yc_pix_up = yc_cpu[i] + (h_up_i - 1.0) / 2.0

        if factor > 1 and w_up_i > 1 and h_up_i > 1:
            xc_orig = xc_pix_up * (w_orig_i - 1.0) / (w_up_i - 1.0)
            yc_orig = yc_pix_up * (h_orig_i - 1.0) / (h_up_i - 1.0)
        else:
            xc_orig = xc_pix_up
            yc_orig = yc_pix_up

        # -- Quality gate 1: in-bounds check --
        if not (0 <= xc_orig < w_orig_i and 0 <= yc_orig < h_orig_i):
            xc_orig = np.nan
            yc_orig = np.nan
            residual_cpu[i] = np.inf

        centers[i, 0] = xc_orig
        centers[i, 1] = yc_orig

    return centers, residual_cpu


def _isophote_curvature_batch_gpu(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    upsample_factor: int = 4,
    boundary_margin: int | None = None,
    pre_smooth_sigma: float = 0.5,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch isophote curvature center estimation on GPU.

    Each pixel's gradient defines a displacement vector toward the local
    curvature center.  The weighted average of these voted centers (weighted
    by curvedness) gives the subpixel center estimate.

    Parameters
    ----------
    rois : (N, H, W) float64 array -- stacked ROIs
    masks : (N, H, W) bool array -- validity masks (True = valid pixel)
    upsample_factor : int -- bicubic upsampling factor (1 = disabled)
    boundary_margin : int or None -- erosion margin near mask boundary
    pre_smooth_sigma : float -- Gaussian smoothing sigma (in upsampled pixels)
        applied before computing second derivatives.  Set to 0 to disable.
    device : int -- GPU device

    Returns
    -------
    centers : (N, 2) float64 array -- centers in ROI pixel coordinates [x, y]
    spreads : (N,) float64 array -- weighted spread of voted centers per blob
    """
    _require_cupy()

    rois = np.asarray(rois, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    N = rois.shape[0]

    if N == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    if boundary_margin is None:
        boundary_margin = upsample_factor

    with gpu_device(device):
        # -- Step 1: Track per-ROI original dimensions --
        orig_hs = np.array([rois.shape[1]] * N, dtype=np.int32)
        orig_ws = np.array([rois.shape[2]] * N, dtype=np.int32)

        # -- Step 2 & 3: Upsample (single batch kernel launch) --
        factor = int(upsample_factor)
        if factor > 1:
            from .upsample import batch_bicubic_upsample, batch_nn_upsample
            rois_gpu = cp.asarray(rois, dtype=cp.float64)
            masks_gpu = cp.asarray(masks.astype(np.float64))
            g_batch = batch_bicubic_upsample(rois_gpu, factor)
            m_batch = batch_nn_upsample(masks_gpu, factor) > 0.5
            up_hs = np.full(N, g_batch.shape[1], dtype=np.int32)
            up_ws = np.full(N, g_batch.shape[2], dtype=np.int32)
            Hmax = g_batch.shape[1]
            Wmax = g_batch.shape[2]
        else:
            g_batch = cp.asarray(rois)
            m_batch = cp.asarray(masks)
            up_hs = orig_hs.copy()
            up_ws = orig_ws.copy()
            Hmax = g_batch.shape[1]
            Wmax = g_batch.shape[2]

        # -- Step 4: Optional Gaussian pre-smoothing --
        if pre_smooth_sigma > 0:
            # Apply per-slice smoothing; zero-padded areas are outside mask anyway
            cndi.gaussian_filter(g_batch, sigma=(0, pre_smooth_sigma, pre_smooth_sigma),
                                 output=g_batch)

        # -- Step 5: Erode mask by boundary_margin --
        min_up_dim = int(min(int(up_hs.min()), int(up_ws.min())))
        # Need at least a 3x3 interior for second derivatives (the derivative
        # grid is 2 pixels smaller), so require eroded region >= 5.
        max_margin = max(0, (min_up_dim - 5) // 2)
        eff_margin = min(boundary_margin, max_margin)
        if eff_margin > 0:
            struct = cp.ones((1, 2 * eff_margin + 1, 2 * eff_margin + 1), dtype=bool)
            m_eroded = cndi.binary_erosion(m_batch, structure=struct)
        else:
            m_eroded = m_batch.copy()

        # -- Step 6: First derivatives via central differences --
        # Operating on interior pixels [1:-1, 1:-1], producing a grid of size (Hmax-2, Wmax-2)
        I = g_batch
        Hd = Hmax - 2  # derivative grid height
        Wd = Wmax - 2  # derivative grid width

        if Hd < 1 or Wd < 1:
            # Degenerate case: ROIs too small for second derivatives
            centers = np.full((N, 2), np.nan, dtype=np.float64)
            spreads = np.full((N,), np.inf, dtype=np.float64)
            return centers, spreads

        Ix = (I[:, 1:-1, 2:] - I[:, 1:-1, :-2]) / 2.0   # (N, Hd, Wd)
        Iy = (I[:, 2:, 1:-1] - I[:, :-2, 1:-1]) / 2.0   # (N, Hd, Wd)

        # -- Step 7: Second derivatives --
        Ixx = I[:, 1:-1, 2:] - 2.0 * I[:, 1:-1, 1:-1] + I[:, 1:-1, :-2]   # (N, Hd, Wd)
        Iyy = I[:, 2:, 1:-1] - 2.0 * I[:, 1:-1, 1:-1] + I[:, :-2, 1:-1]   # (N, Hd, Wd)
        Ixy = (I[:, 2:, 2:] - I[:, 2:, :-2] - I[:, :-2, 2:] + I[:, :-2, :-2]) / 4.0  # (N, Hd, Wd)

        # -- Step 8: Validity mask --
        eps = cp.float64(1e-12)
        grad_sq = Ix**2 + Iy**2                              # (N, Hd, Wd)
        denom = Iy**2 * Ixx - 2.0 * Ix * Ixy * Iy + Ix**2 * Iyy  # (N, Hd, Wd)

        m_interior = m_eroded[:, 1:-1, 1:-1]                  # (N, Hd, Wd)
        valid = m_interior & (cp.abs(denom) > eps) & (grad_sq > eps)

        # -- Step 9 & 10: Displacement to curvature center --
        safe_denom = cp.where(cp.abs(denom) > eps, denom, cp.float64(1.0))
        Dx = -Ix * grad_sq / safe_denom   # (N, Hd, Wd)
        Dy = -Iy * grad_sq / safe_denom   # (N, Hd, Wd)

        # -- Step 11: Curvedness weight --
        curvedness = cp.sqrt(Ixx**2 + 2.0 * Ixy**2 + Iyy**2)  # (N, Hd, Wd)

        # -- Step 12: ROI-centered pixel coordinates for derivative grid --
        # The derivative grid pixels correspond to pixel positions [1, 2, ..., Hmax-2] in y
        # and [1, 2, ..., Wmax-2] in x of the upsampled image.
        # Per-ROI centering: center_x = (up_ws[i]-1)/2, center_y = (up_hs[i]-1)/2
        col_idx = cp.arange(Wd, dtype=cp.float64)[None, None, :] + 1.0  # pixel col in upsampled
        row_idx = cp.arange(Hd, dtype=cp.float64)[None, :, None] + 1.0  # pixel row in upsampled

        up_ws_g = cp.asarray(up_ws, dtype=cp.float64)
        up_hs_g = cp.asarray(up_hs, dtype=cp.float64)
        cx_off = (up_ws_g - 1.0) / 2.0  # (N,)
        cy_off = (up_hs_g - 1.0) / 2.0  # (N,)

        xp = col_idx - cx_off[:, None, None]  # (N, Hd, Wd) centered x
        yp = row_idx - cy_off[:, None, None]  # (N, Hd, Wd) centered y

        # -- Step 13: Voted center positions --
        vote_x = xp + Dx  # (N, Hd, Wd)
        vote_y = yp + Dy  # (N, Hd, Wd)

        # -- Step 14: Outlier rejection -- skip votes with large displacement
        disp_sq = Dx**2 + Dy**2
        # ROI half-size: use max of up_ws, up_hs per ROI
        roi_half = cp.maximum(up_ws_g, up_hs_g)[:, None, None] / 2.0
        within_range = disp_sq < roi_half**2
        valid = valid & within_range

        # Apply validity: zero out invalid votes in weight
        w = cp.where(valid, curvedness, cp.float64(0.0))  # (N, Hd, Wd)

        # -- Step 15: Weighted average center --
        w_sum = w.sum(axis=(1, 2))  # (N,)
        w_sum_safe = cp.maximum(w_sum, cp.float64(1e-30))
        xc = (w * vote_x).sum(axis=(1, 2)) / w_sum_safe  # (N,)
        yc = (w * vote_y).sum(axis=(1, 2)) / w_sum_safe  # (N,)

        # -- Step 16: Spread metric --
        spread_sq = (w * ((vote_x - xc[:, None, None])**2 +
                          (vote_y - yc[:, None, None])**2)).sum(axis=(1, 2)) / w_sum_safe
        spread = cp.sqrt(spread_sq)  # (N,)

        # -- Step 17: Per-ROI coordinate transform (centered upsampled -> original) --
        xc_cpu = cp.asnumpy(xc)   # (N,) centered coords in upsampled grid
        yc_cpu = cp.asnumpy(yc)
        spread_cpu = cp.asnumpy(spread)

    # Convert from centered upsampled coords to original ROI pixel coords
    centers = np.empty((N, 2), dtype=np.float64)
    for i in range(N):
        w_up_i = float(up_ws[i])
        h_up_i = float(up_hs[i])
        w_orig_i = float(orig_ws[i])
        h_orig_i = float(orig_hs[i])

        # xc is relative to center of upsampled grid: xc_pix_up = xc + (w_up - 1) / 2
        xc_pix_up = xc_cpu[i] + (w_up_i - 1.0) / 2.0
        yc_pix_up = yc_cpu[i] + (h_up_i - 1.0) / 2.0

        if factor > 1 and w_up_i > 1 and h_up_i > 1:
            xc_orig = xc_pix_up * (w_orig_i - 1.0) / (w_up_i - 1.0)
            yc_orig = yc_pix_up * (h_orig_i - 1.0) / (h_up_i - 1.0)
        else:
            xc_orig = xc_pix_up
            yc_orig = yc_pix_up

        centers[i, 0] = xc_orig
        centers[i, 1] = yc_orig

    return centers, spread_cpu


def _extract_voronoi_rois(
    gray: np.ndarray,
    voronoi_labels: np.ndarray,
    rows: list,
    pad: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Extract Voronoi-masked ROIs with border-pixel background fill.

    Parameters
    ----------
    gray : (H, W) float64 grayscale image
    voronoi_labels : (H, W) int32 Voronoi label map
    rows : list of (x, y, w, h, cx, cy) per component
    pad : border padding in pixels

    Returns
    -------
    rois_stack : (N, Hmax, Wmax) float64 — padded ROIs
    masks_stack : (N, Hmax, Wmax) bool — Voronoi masks
    origins : list of (x0, y0) top-left corners
    """
    H, W = gray.shape
    rois_list = []
    masks_list = []
    origins = []

    for j, (x, y, w, h, cx, cy) in enumerate(rows):
        x0 = max(0, int(x) - int(pad))
        y0 = max(0, int(y) - int(pad))
        x1 = min(W, int(x + w) + int(pad))
        y1 = min(H, int(y + h) + int(pad))
        roi = gray[y0:y1, x0:x1].astype(np.float64)
        vmask = voronoi_labels[y0:y1, x0:x1] == j

        # Background-fill: median of border pixels (numpy-only erosion)
        inner = vmask.copy()
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        inner[1:, :] &= vmask[:-1, :]
        inner[:-1, :] &= vmask[1:, :]
        inner[:, 1:] &= vmask[:, :-1]
        inner[:, :-1] &= vmask[:, 1:]
        border_mask = vmask & ~inner
        border_vals = roi[border_mask]
        bg = float(np.median(border_vals)) if border_vals.size > 0 else 0.0
        roi_filled = np.where(vmask, roi, bg)

        rois_list.append(roi_filled)
        masks_list.append(vmask)
        origins.append((x0, y0))

    # Pad to uniform size and stack
    hs = [r.shape[0] for r in rois_list]
    ws = [r.shape[1] for r in rois_list]
    Hm, Wm = max(hs), max(ws)
    rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
    for j, (roi, vmask) in enumerate(zip(rois_list, masks_list)):
        h_r, w_r = roi.shape
        rois_stack[j, :h_r, :w_r] = roi
        masks_stack[j, :h_r, :w_r] = vmask

    return rois_stack, masks_stack, origins


def detect_centers_gpu(
    image: np.ndarray,
    *,
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
) -> CenterResult:
    g, bw = _segment_binary(
        image,
        threshold=threshold,
        invert=invert,
        morph_open=morph_open,
        morph_close=morph_close,
    )
    if components_backend == "gpu":
        comp = connected_components_stats_gpu(bw, connectivity=connectivity, device=device)
    else:
        comp = connected_components_stats_cpu(bw, connectivity=connectivity)
    labels = comp.labels
    stats = comp.stats
    num = comp.num_labels

    # ── Voronoi-partitioned methods: early dispatch ──────────────────────
    # Must branch BEFORE method_rows loop which rejects unknown method names.
    _VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}
    if refine in _VORONOI_METHODS:
        # Collect component stats (same filtering as method_rows path)
        H_img, W_img = g.shape
        rows = []
        for lab in range(1, num):
            x, y, w, h, area = stats[lab]
            if area < int(area_min) or area > int(area_max):
                continue
            # Exclude blobs whose bbox touches the image boundary
            # (partial blobs produce unreliable radial symmetry estimates)
            if x <= 0 or y <= 0 or x + w >= W_img or y + h >= H_img:
                continue
            cx, cy = comp.centroids[lab]
            rows.append([x, y, w, h, cx, cy])

        if not rows:
            arr = np.zeros((0, 2), dtype=np.float64)
            meta = {
                "invert": bool(invert), "area_min": int(area_min), "area_max": int(area_max),
                "morph_open": int(morph_open), "morph_close": int(morph_close),
                "pad": int(pad), "connectivity": int(connectivity),
                "gpu_batch": int(gpu_batch), "device": int(device),
                "use_float64": bool(use_float64),
                "components_backend": str(components_backend),
                "small_feature_max": float(small_feature_max),
                "upsample_factor": int(upsample_factor),
            }
            return CenterResult(centers_xy=arr, method=f"threshold={threshold}, refine={refine}", backend="gpu", meta=meta)

        rows_arr = np.asarray(rows, dtype=np.float64)
        coarse_centers = rows_arr[:, 4:6]  # [cx, cy] = [x, y] order

        # 1. Voronoi partition
        from .voronoi import compute_voronoi_labels_gpu
        voronoi_labels = compute_voronoi_labels_gpu(
            coarse_centers,
            g.shape,
            device=device,
        )
        cell_areas = np.bincount(voronoi_labels.ravel(), minlength=len(rows))

        # 2-3. Extract Voronoi-masked ROIs, pad to uniform size
        rois_stack, masks_stack, origins = _extract_voronoi_rois(
            g, voronoi_labels, rows, pad=pad,
        )

        # 4. Batch refinement
        if refine == "radial_symmetry":
            local_centers, quality = _radial_symmetry_batch_gpu(
                rois_stack, masks_stack, upsample_factor=upsample_factor, device=device,
            )
            quality_key = "radial_symmetry_residual"
        else:
            local_centers, quality = _isophote_curvature_batch_gpu(
                rois_stack, masks_stack, upsample_factor=upsample_factor, device=device,
            )
            quality_key = "isophote_curvature_spread"

        # 5. Transform to image coordinates and filter
        centers_out = []
        quality_kept = []
        areas_kept = []
        for j, (lc, q) in enumerate(zip(local_centers, quality)):
            x0, y0 = origins[j]
            gx = x0 + float(lc[0])
            gy = y0 + float(lc[1])
            if np.isfinite(gx) and np.isfinite(gy):
                centers_out.append([gx, gy])
                quality_kept.append(float(q))
                areas_kept.append(int(cell_areas[j]) if j < len(cell_areas) else 0)

        arr = np.asarray(centers_out, dtype=np.float64) if centers_out else np.zeros((0, 2), dtype=np.float64)
        meta = {
            "invert": bool(invert), "area_min": int(area_min), "area_max": int(area_max),
            "morph_open": int(morph_open), "morph_close": int(morph_close),
            "pad": int(pad), "connectivity": int(connectivity),
            "gpu_batch": int(gpu_batch), "device": int(device),
            "use_float64": bool(use_float64),
            "components_backend": str(components_backend),
            "small_feature_max": float(small_feature_max),
            "upsample_factor": int(upsample_factor),
            quality_key: np.asarray(quality_kept, dtype=np.float64),
            "voronoi_cell_area": np.asarray(areas_kept, dtype=np.int64),
        }
        return CenterResult(centers_xy=arr, method=f"threshold={threshold}, refine={refine}", backend="gpu", meta=meta)

    rows = []
    lab_ids = []
    method_rows = {"logquad": [], "weighted": [], "none": [], "edge_gradmoment": [], "edge_erf": []}
    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area < int(area_min) or area > int(area_max):
            continue
        cx, cy = comp.centroids[lab]
        method_local = choose_refine_method_for_bbox(w, h, small_feature_max=small_feature_max) if refine == "auto" else refine
        if method_local in {"edge_moment", "edges_centered"}:
            method_local = "edge_gradmoment"
        if method_local in {"edge_gradmoment_erf", "edge_erf_centered"}:
            method_local = "edge_erf"
        if method_local not in method_rows:
            raise ValueError(f"Unsupported refine method: {method_local}")
        rows.append([x, y, w, h, cx, cy])
        lab_ids.append(lab)
        method_rows[method_local].append(len(rows) - 1)

    centers = []
    if method_rows["logquad"]:
        ids = np.asarray(method_rows["logquad"], dtype=np.int32)
        with gpu_device(device):
            g_gpu = cp.asarray(np.asarray(g, dtype=np.float32), dtype=cp.float32)
            lab_gpu = cp.asarray(np.asarray(labels, dtype=np.int32), dtype=cp.int32)
            stats_gpu = cp.asarray(np.asarray(rows, dtype=np.float32)[ids], dtype=cp.float32)
            lab_ids_gpu = cp.asarray(np.asarray(lab_ids, dtype=np.int32)[ids], dtype=cp.int32)
            refined_gpu, ok_gpu = refine_centers_logquad_gpu_match_cpu(
                g_gpu,
                lab_gpu,
                stats_gpu,
                lab_ids_gpu,
                pad=pad,
                batch=max(1, int(gpu_batch)),
                use_float64=use_float64,
            )
            refined = cp.asnumpy(refined_gpu)
            ok = cp.asnumpy(ok_gpu).astype(bool)
        for pt, good in zip(refined, ok):
            if good and np.all(np.isfinite(pt)):
                centers.append([float(pt[0]), float(pt[1])])

    if method_rows["edge_gradmoment"]:
        rows_edge = np.asarray(rows, dtype=np.float32)[np.asarray(method_rows["edge_gradmoment"], dtype=np.int32)]
        centers_arr, ok = refine_centers_edge_moment_gpu(g, rows_edge, device=device)
        for pt, good in zip(centers_arr, np.asarray(ok, dtype=bool)):
            if good and np.all(np.isfinite(pt)):
                centers.append([float(pt[0]), float(pt[1])])

    if method_rows["edge_erf"]:
        rows_edge = np.asarray(rows, dtype=np.float32)[np.asarray(method_rows["edge_erf"], dtype=np.int32)]
        centers_arr, ok = refine_centers_edge_erf_gpu(g, rows_edge, device=device)
        for pt, good in zip(centers_arr, np.asarray(ok, dtype=bool)):
            if good and np.all(np.isfinite(pt)):
                centers.append([float(pt[0]), float(pt[1])])

    for method_name in ("weighted", "none"):
        idxs = method_rows[method_name]
        if not idxs:
            continue
        boxes = []
        rois = []
        masks = []
        H, W = g.shape
        for j in idxs:
            lab = int(lab_ids[j])
            x, y, w, h = [int(v) for v in rows[j][:4]]
            x0 = max(0, x - int(pad))
            y0 = max(0, y - int(pad))
            x1 = min(W, x + w + int(pad))
            y1 = min(H, y + h + int(pad))
            boxes.append((x0, y0))
            rois.append(g[y0:y1, x0:x1])
            masks.append(labels[y0:y1, x0:x1] == lab)
        loc = _refine_batch_gpu(rois, masks, method=method_name, use_float64=use_float64, device=device)
        for (x0, y0), (cx, cy) in zip(boxes, loc):
            if np.isfinite(cx) and np.isfinite(cy):
                centers.append([x0 + float(cx), y0 + float(cy)])

    arr = np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 2), dtype=np.float64)
    return CenterResult(
        centers_xy=arr,
        method=f"threshold={threshold}, refine={refine}",
        backend="gpu",
        meta={
            "invert": bool(invert),
            "area_min": int(area_min),
            "area_max": int(area_max),
            "morph_open": int(morph_open),
            "morph_close": int(morph_close),
            "pad": int(pad),
            "connectivity": int(connectivity),
            "gpu_batch": int(gpu_batch),
            "device": int(device),
            "use_float64": bool(use_float64),
            "components_backend": str(components_backend),
            "small_feature_max": float(small_feature_max),
        },
    )


def _detect_centers_tiled_gpu(
    image: np.ndarray,
    *,
    invert: bool = False,
    area_min: int = 1,
    area_max: int = 50,
    morph_open: int = 0,
    morph_close: int = 0,
    tile_h: int = 8192,
    overlap: int = 128,
    otsu_downsample: int = 4,
    thr_scale: float = 0.8,
    refine: str = "edge_gradmoment",
    connectivity: int = 8,
    device: int = 0,
    # edge_gradmoment / edge_erf shared params
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    # edge_erf-specific params
    erf_fit_rad: int = 5,
    erf_gn_iters: int = 5,
    erf_init_sigma: float = 1.25,
    erf_sigma_min: float = 0.35,
    erf_sigma_max: float = 6.0,
    erf_max_shift: float = 2.5,
    erf_damp: float = 1e-4,
    erf_cond_max: float = 1e8,
    # logquad/weighted params
    pad: int = 3,
    gpu_batch: int = 4096,
    use_float64: bool = True,
    small_feature_max: float = 12.0,
    # Voronoi method params
    upsample_factor: int = 4,
) -> CenterResult:
    """Dedicated tiled GPU center detection pipeline.

    Optimized: single device context, RawKernel CC stats, GPU-side filtering,
    core refinement functions called directly (no GPU↔CPU transfers in the loop).
    """
    _require_cupy()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for tiled GPU center detection.") from exc

    g = np.asarray(image)
    if g.ndim != 2:
        raise ValueError("Expected 2D grayscale image")
    H, W = g.shape

    if overlap >= tile_h:
        raise ValueError("overlap must be < tile_h")

    # Validate refine method before entering tile loop
    _supported_refine = {
        "edge_gradmoment", "edge_erf", "logquad", "logquadratic",
        "weighted", "auto", "radial_symmetry", "isophote_curvature",
    }
    if refine not in _supported_refine:
        raise ValueError(f"Unknown refine method: {refine}")

    # --- Step 0: Global threshold (once, on downsampled image) ---
    ds = max(1, int(otsu_downsample))
    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    g_ds = g[::ds, ::ds]
    otsu_thresh, _ = cv2.threshold(g_ds, 0, 255, thr_type | cv2.THRESH_OTSU)
    thr = float(otsu_thresh) * float(thr_scale)

    step = tile_h - overlap
    all_centers = []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) if (morph_open > 0 or morph_close > 0) else None

    # Single device context for the entire tile loop
    with cp.cuda.Device(int(device)):
        y0 = 0
        while y0 < H:
            y1 = min(H, y0 + tile_h)
            tile = g[y0:y1]

            # --- Per-tile threshold + morphology (CPU) ---
            _, bw = cv2.threshold(tile, thr, 255, thr_type)
            if morph_open > 0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=int(morph_open))
            if morph_close > 0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=int(morph_close))

            # --- GPU CC via core (stays on GPU, no roundtrip) ---
            bw_gpu = cp.asarray(bw != 0)
            labels_gpu, stats_gpu, centroids_gpu, num = (
                _connected_components_stats_gpu_core(bw_gpu, connectivity=int(connectivity))
            )

            # num includes background (label 0). num == 1 means no foreground.
            if num > 1:
                # --- Vectorized area filter (CuPy, on GPU) ---
                area = stats_gpu[1:, 4]
                mask = (area >= int(area_min)) & (area <= int(area_max))

                if cp.any(mask):
                    # Build (K, 6) stats array on GPU: [x, y, w, h, cx, cy]
                    filtered_stats = stats_gpu[1:][mask]
                    filtered_cents = centroids_gpu[1:][mask]
                    stats_xywh_cc = cp.concatenate([
                        filtered_stats[:, :4].astype(cp.float32),
                        filtered_cents.astype(cp.float32),
                    ], axis=1)

                    # --- Refinement dispatch ---
                    if refine in ("radial_symmetry", "isophote_curvature"):
                        # Voronoi path: per-tile Voronoi → ROI extraction → batch GPU
                        stats_np = cp.asnumpy(filtered_stats)
                        cents_np = cp.asnumpy(filtered_cents)
                        tile_local_h = y1 - y0

                        # Exclude blobs whose bbox touches the image boundary
                        keep = np.ones(stats_np.shape[0], dtype=bool)
                        bx, by = stats_np[:, 0], stats_np[:, 1]
                        bw, bh = stats_np[:, 2], stats_np[:, 3]
                        keep &= (bx > 0) & (bx + bw < W)
                        if y0 == 0:
                            keep &= (by > 0)
                        if y1 == H:
                            keep &= (by + bh < tile_local_h)
                        stats_np = stats_np[keep]
                        cents_np = cents_np[keep]

                        if stats_np.shape[0] == 0:
                            refined = np.zeros((0, 2), dtype=np.float64)
                        else:
                            rows_v = np.column_stack([
                                stats_np[:, :4], cents_np,
                            ]).tolist()
                            coarse_centers = cents_np.astype(np.float64)

                            from .voronoi import compute_voronoi_labels_gpu
                            voronoi_labels = compute_voronoi_labels_gpu(
                                coarse_centers, (tile_local_h, W), device=device,
                            )

                            tile_f64 = tile.astype(np.float64)
                            rois_stack, masks_stack, origins = _extract_voronoi_rois(
                                tile_f64, voronoi_labels, rows_v, pad=pad,
                            )

                            if refine == "radial_symmetry":
                                local_centers, quality = _radial_symmetry_batch_gpu(
                                    rois_stack, masks_stack,
                                    upsample_factor=upsample_factor, device=device,
                                )
                            else:
                                local_centers, quality = _isophote_curvature_batch_gpu(
                                    rois_stack, masks_stack,
                                    upsample_factor=upsample_factor, device=device,
                                )

                            # Transform ROI-local → tile-local coordinates
                            K_v = local_centers.shape[0]
                            refined = np.full((K_v, 2), np.nan, dtype=np.float64)
                            for jj in range(K_v):
                                x0_r, y0_r = origins[jj]
                                refined[jj, 0] = x0_r + local_centers[jj, 0]
                                refined[jj, 1] = y0_r + local_centers[jj, 1]
                            good = np.all(np.isfinite(refined), axis=1)
                            refined = refined[good]
                    else:
                        # Non-Voronoi path: all on GPU via core functions
                        tile_gpu = cp.asarray(tile, dtype=cp.float32)

                        if refine == "edge_gradmoment":
                            refined_gpu, ok_gpu = _refine_edge_moment_core(
                                tile_gpu, stats_xywh_cc,
                                band_rad=band_rad, edge_rad=edge_rad,
                                smooth_passes=smooth_passes, loc_rad=loc_rad,
                                grad_power=grad_power, iters=iters,
                            )
                        elif refine == "edge_erf":
                            refined_gpu, ok_gpu = _refine_edge_erf_core(
                                tile_gpu, stats_xywh_cc,
                                band_rad=band_rad, edge_rad=edge_rad,
                                smooth_passes=smooth_passes, loc_rad=loc_rad,
                                grad_power=grad_power, iters=iters,
                                erf_fit_rad=erf_fit_rad, erf_gn_iters=erf_gn_iters,
                                erf_init_sigma=erf_init_sigma,
                                erf_sigma_min=erf_sigma_min,
                                erf_sigma_max=erf_sigma_max,
                                erf_max_shift=erf_max_shift,
                                erf_damp=erf_damp, erf_cond_max=erf_cond_max,
                            )
                        elif refine in ("logquad", "logquadratic"):
                            lab_ids_gpu = (cp.nonzero(mask)[0] + 1).astype(cp.int32)
                            refined_gpu, ok_gpu = refine_centers_logquad_gpu_match_cpu(
                                tile_gpu,
                                labels_gpu.astype(cp.int32),
                                stats_xywh_cc,
                                lab_ids_gpu,
                                pad=pad, batch=max(1, int(gpu_batch)),
                                use_float64=use_float64,
                            )
                        elif refine == "weighted":
                            lab_ids_gpu = (cp.nonzero(mask)[0] + 1).astype(cp.int32)
                            refined_gpu, ok_gpu = _refine_weighted_core(
                                tile_gpu, labels_gpu.astype(cp.int32),
                                stats_xywh_cc, lab_ids_gpu,
                                pad=pad, batch=max(1, int(gpu_batch)),
                                use_float64=use_float64,
                            )
                        elif refine == "auto":
                            lab_ids_gpu = (cp.nonzero(mask)[0] + 1).astype(cp.int32)
                            max_dim = cp.maximum(stats_xywh_cc[:, 2], stats_xywh_cc[:, 3])
                            is_small = max_dim <= float(small_feature_max)
                            is_large = ~is_small

                            K_total = int(stats_xywh_cc.shape[0])
                            refined_gpu = cp.full((K_total, 2), cp.nan, dtype=cp.float64)
                            ok_gpu = cp.zeros((K_total,), dtype=cp.bool_)

                            if cp.any(is_small):
                                si = cp.nonzero(is_small)[0]
                                r, o = refine_centers_logquad_gpu_match_cpu(
                                    tile_gpu, labels_gpu.astype(cp.int32),
                                    stats_xywh_cc[si], lab_ids_gpu[si],
                                    pad=pad, batch=max(1, int(gpu_batch)),
                                    use_float64=use_float64,
                                )
                                refined_gpu[si] = r
                                ok_gpu[si] = o

                            if cp.any(is_large):
                                li = cp.nonzero(is_large)[0]
                                r, o = _refine_edge_moment_core(
                                    tile_gpu, stats_xywh_cc[li],
                                    band_rad=band_rad, edge_rad=edge_rad,
                                    smooth_passes=smooth_passes, loc_rad=loc_rad,
                                    grad_power=grad_power, iters=iters,
                                )
                                refined_gpu[li] = r
                                ok_gpu[li] = o

                        refined = cp.asnumpy(refined_gpu).astype(np.float64)
                        ok = cp.asnumpy(ok_gpu).astype(bool)
                        good = ok & np.all(np.isfinite(refined), axis=1)
                        refined = refined[good]

                    # --- Band-based de-dup (common for all methods) ---
                    if refined.shape[0] > 0:
                        refined[:, 1] += y0  # tile-local → global

                        half = overlap // 2
                        keep_lo = y0 if y0 == 0 else (y0 + half)
                        keep_hi = y1 if y1 == H else (y1 - half)
                        m = (refined[:, 1] >= keep_lo) & (refined[:, 1] < keep_hi)
                        if np.any(m):
                            all_centers.append(refined[m])

            if y1 == H:
                break
            y0 += step

    if all_centers:
        centers = np.vstack(all_centers)
    else:
        centers = np.zeros((0, 2), dtype=np.float64)

    return CenterResult(
        centers_xy=centers,
        method=f"tiled_gpu(refine={refine})",
        backend="gpu",
        meta={
            "tile_h": int(tile_h),
            "overlap": int(overlap),
            "otsu_downsample": int(ds),
            "thr_scale": float(thr_scale),
            "global_otsu_threshold": float(thr),
            "refine": str(refine),
            "invert": bool(invert),
            "area_min": int(area_min),
            "area_max": int(area_max),
            "device": int(device),
            "small_feature_max": float(small_feature_max),
            "upsample_factor": int(upsample_factor),
            "implementation": "rawkernel_cc+core_refine",
        },
    )
