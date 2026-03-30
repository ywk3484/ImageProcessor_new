# Center Detection API Unification & GPU Memory Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge three overlapping center detection functions into one unified `detect_centers()`, add GPU batching to Voronoi methods to fix OOM, and use float32 internals for memory efficiency.

**Architecture:** The public `detect_centers()` gains optional `tile_h` parameter for spatial tiling. Internally, both GPU and CPU backends extract a shared `_detect_single_tile_*` core function called once (no tiling) or in a loop (tiling). Voronoi-based methods (`radial_symmetry`, `isophote_curvature`) get batched processing and float32 intermediates to cut GPU memory ~60×.

**Tech Stack:** Python 3.10+, NumPy, CuPy (GPU), OpenCV (segmentation), CUDA RawKernels (upsampling)

**Spec:** `docs/superpowers/specs/2026-03-30-center-detection-unification-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `subpx/_gpu/upsample.py` | CUDA upsample kernels | Add float32 kernel variants |
| `subpx/_gpu/centers.py` | GPU center detection & refinement | Merge `detect_centers_gpu` + `_detect_centers_tiled_gpu` → unified function with `_detect_single_tile_gpu` core; add batching + float32 to Voronoi methods; vectorize Python loops |
| `subpx/_cpu/centers.py` | CPU center detection & refinement | Add `tile_h` tiling loop to `detect_centers_cpu` |
| `subpx/centers.py` | Public API dispatch | Remove `detect_centers_tiled` + `detect_centers_tiled_global_otsu`; add tiling params to `detect_centers` |
| `subpx/__init__.py` | Package exports | Update imports for removed functions |
| `subpx/legacy.py` | Deprecated name wrappers | Add `detect_centers_tiled` + `detect_centers_tiled_global_otsu` aliases |
| `subpx/batch.py` | Batch processing helpers | Update `detect_centers_tiled_batch` |
| `tests/test_unification.py` | Tests for unified API + GPU memory | New test file |

---

### Task 1: Add float32 CUDA upsample kernels

**Files:**
- Modify: `subpx/_gpu/upsample.py`
- Test: `tests/test_unification.py` (new file)

The existing bicubic and NN upsample kernels hardcode `double`. We need float32 variants for the memory optimization.

- [ ] **Step 1: Write failing test for float32 bicubic upsample**

Create `tests/test_unification.py`:

```python
import numpy as np
import pytest

def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False

skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


@skipno_gpu
def test_bicubic_upsample_f32_matches_f64():
    """Float32 bicubic upsample must match float64 within 1e-4."""
    import cupy as cp
    from subpx._gpu.upsample import batch_bicubic_upsample, batch_bicubic_upsample_f32

    rng = np.random.RandomState(42)
    rois = rng.rand(5, 10, 12).astype(np.float64)
    factor = 4

    with cp.cuda.Device(0):
        ref = batch_bicubic_upsample(cp.asarray(rois, dtype=cp.float64), factor)
        out = batch_bicubic_upsample_f32(cp.asarray(rois, dtype=cp.float32), factor)
        diff = float(cp.max(cp.abs(ref - out.astype(cp.float64))))
    assert diff < 1e-4, f"f32 vs f64 max diff: {diff}"


@skipno_gpu
def test_nn_upsample_f32_matches_f64():
    """Float32 NN upsample must match float64 exactly (no interpolation)."""
    import cupy as cp
    from subpx._gpu.upsample import batch_nn_upsample, batch_nn_upsample_f32

    rng = np.random.RandomState(42)
    rois = rng.rand(5, 8, 10).astype(np.float64)
    factor = 4

    with cp.cuda.Device(0):
        ref = batch_nn_upsample(cp.asarray(rois, dtype=cp.float64), factor)
        out = batch_nn_upsample_f32(cp.asarray(rois, dtype=cp.float32), factor)
        diff = float(cp.max(cp.abs(ref - out.astype(cp.float64))))
    assert diff < 1e-6, f"f32 NN vs f64 max diff: {diff}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_bicubic_upsample_f32_matches_f64 -v`
Expected: FAIL with `ImportError` — `batch_bicubic_upsample_f32` does not exist.

- [ ] **Step 3: Implement float32 upsample kernels**

In `subpx/_gpu/upsample.py`, add after the existing `_NN_KERNEL_SRC`:

```python
# ── Float32 batch bicubic upsample (Catmull-Rom, a=-0.5) ────────────────
_BICUBIC_F32_KERNEL_SRC = r"""
extern "C" __global__
void batch_bicubic_upsample_f32(
    const float* __restrict__ src,    // (N, Hin, Win) contiguous
    float* __restrict__ dst,          // (N, Hout, Wout) contiguous
    const int N, const int Hin, const int Win,
    const int Hout, const int Wout, const int factor
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)N * Hout * Wout;
    if (idx >= total) return;

    int n  = (int)(idx / ((long long)Hout * Wout));
    int rem = (int)(idx % ((long long)Hout * Wout));
    int oy = rem / Wout;
    int ox = rem % Wout;

    float scale_y = (Hout > 1) ? ((float)Hin - 1.0f) / ((float)Hout - 1.0f) : 0.0f;
    float scale_x = (Wout > 1) ? ((float)Win - 1.0f) / ((float)Wout - 1.0f) : 0.0f;
    float sy = (float)oy * scale_y;
    float sx = (float)ox * scale_x;

    int iy = (int)floorf(sy);
    int ix = (int)floorf(sx);
    float fy = sy - (float)iy;
    float fx = sx - (float)ix;

    float wy[4], wx[4];
    {
        float t;
        t = 1.0f + fy;
        wy[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t = fy;
        wy[1] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 1.0f - fy;
        wy[2] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 2.0f - fy;
        wy[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }
    {
        float t;
        t = 1.0f + fx;
        wx[0] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
        t = fx;
        wx[1] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 1.0f - fx;
        wx[2] = ((1.5f*t - 2.5f)*t)*t + 1.0f;
        t = 2.0f - fx;
        wx[3] = ((-0.5f*t + 2.5f)*t - 4.0f)*t + 2.0f;
    }

    const float* src_n = src + (long long)n * Hin * Win;
    float val = 0.0f;
    for (int dy = 0; dy < 4; dy++) {
        int yy = iy - 1 + dy;
        if (yy < 0) yy = 0;
        if (yy >= Hin) yy = Hin - 1;
        for (int dx = 0; dx < 4; dx++) {
            int xx = ix - 1 + dx;
            if (xx < 0) xx = 0;
            if (xx >= Win) xx = Win - 1;
            val += wy[dy] * wx[dx] * src_n[yy * Win + xx];
        }
    }

    dst[idx] = val;
}
"""

# ── Float32 batch nearest-neighbor upsample ──────────────────────────────
_NN_F32_KERNEL_SRC = r"""
extern "C" __global__
void batch_nn_upsample_f32(
    const float* __restrict__ src,    // (N, Hin, Win) contiguous
    float* __restrict__ dst,          // (N, Hout, Wout) contiguous
    const int N, const int Hin, const int Win,
    const int Hout, const int Wout, const int factor
) {
    long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    long long total = (long long)N * Hout * Wout;
    if (idx >= total) return;

    int n  = (int)(idx / ((long long)Hout * Wout));
    int rem = (int)(idx % ((long long)Hout * Wout));
    int oy = rem / Wout;
    int ox = rem % Wout;

    float scale_y = (Hout > 1) ? ((float)Hin - 1.0f) / ((float)Hout - 1.0f) : 0.0f;
    float scale_x = (Wout > 1) ? ((float)Win - 1.0f) / ((float)Wout - 1.0f) : 0.0f;
    int sy = (int)((float)oy * scale_y + 0.5f);
    int sx = (int)((float)ox * scale_x + 0.5f);
    if (sy >= Hin) sy = Hin - 1;
    if (sx >= Win) sx = Win - 1;

    dst[idx] = src[(long long)n * Hin * Win + sy * Win + sx];
}
"""
```

Update `_get_kernel` to include the new kernel sources:

```python
def _get_kernel(device: int, name: str):
    key = (device, name)
    if key not in _kernel_cache:
        src = {
            "batch_bicubic_upsample": _BICUBIC_KERNEL_SRC,
            "batch_nn_upsample": _NN_KERNEL_SRC,
            "batch_bicubic_upsample_f32": _BICUBIC_F32_KERNEL_SRC,
            "batch_nn_upsample_f32": _NN_F32_KERNEL_SRC,
        }[name]
        _kernel_cache[key] = cp.RawKernel(src, name)
    return _kernel_cache[key]
```

Add the two new Python functions:

```python
def batch_bicubic_upsample_f32(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) float32 batch by *factor* using Catmull-Rom bicubic."""
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float32)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_bicubic_upsample_f32")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst


def batch_nn_upsample_f32(rois_gpu: "cp.ndarray", factor: int) -> "cp.ndarray":
    """Upsample (N, H, W) float32 batch by *factor* using nearest-neighbor."""
    N, Hin, Win = rois_gpu.shape
    Hout, Wout = Hin * factor, Win * factor
    dst = cp.empty((N, Hout, Wout), dtype=cp.float32)

    total = N * Hout * Wout
    block = 256
    grid = (total + block - 1) // block

    device = int(rois_gpu.device.id)
    kernel = _get_kernel(device, "batch_nn_upsample_f32")
    kernel(
        (grid,), (block,),
        (cp.ascontiguousarray(rois_gpu), dst,
         np.int32(N), np.int32(Hin), np.int32(Win),
         np.int32(Hout), np.int32(Wout), np.int32(factor)),
    )
    return dst
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unification.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/upsample.py tests/test_unification.py
git commit -m "feat: add float32 CUDA upsample kernels for GPU memory optimization"
```

---

### Task 2: Add GPU batching + float32 to `_radial_symmetry_batch_gpu`

**Files:**
- Modify: `subpx/_gpu/centers.py` (function `_radial_symmetry_batch_gpu`, lines 974-1201)
- Test: `tests/test_unification.py`

The current function processes ALL N ROIs at once, causing OOM. Wrap the core computation in a batch loop and use float32 for intermediates.

- [ ] **Step 1: Write failing test for batched radial symmetry**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_radial_symmetry_batched_matches_unbatched():
    """Batched processing (gpu_batch=3) must produce same results as full batch."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu

    rng = np.random.RandomState(42)
    N = 10
    size = 11
    rois = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        yy, xx = np.mgrid[:size, :size]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.2**2))
        rois.append(blob)
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)

    # Full batch (default gpu_batch is large enough for N=10)
    c_full, r_full = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=10000,
    )
    # Small batches
    c_batched, r_batched = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=3,
    )
    # Results must be identical (same computation, just chunked)
    np.testing.assert_allclose(c_full, c_batched, atol=1e-10)
    np.testing.assert_allclose(r_full, r_batched, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_radial_symmetry_batched_matches_unbatched -v`
Expected: FAIL — `_radial_symmetry_batch_gpu` does not accept `gpu_batch` parameter.

- [ ] **Step 3: Implement batched radial symmetry with float32 internals**

Refactor `_radial_symmetry_batch_gpu` in `subpx/_gpu/centers.py`. The strategy:

1. Add `gpu_batch` parameter (default 2048).
2. Extract the current body (lines 999-1201) into `_radial_symmetry_inner_gpu()` that processes a single batch.
3. The outer function loops over batches, calling the inner function, and collects results.
4. Change the inner function to use `cp.float32` for upsampling, gradients, slopes, weights, and intercepts. Keep the 2×2 WLS solve and final center coords in float64.
5. Vectorize the coordinate-transform loop (lines 1169-1194): replace the Python `for i in range(N)` with vectorized numpy.
6. Import `batch_bicubic_upsample_f32` and `batch_nn_upsample_f32` from `.upsample`.

The refactored function structure:

```python
def _radial_symmetry_batch_gpu(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    upsample_factor: int = 4,
    boundary_margin: int | None = None,
    residual_max: float | None = None,
    gpu_batch: int = 2048,
    device: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    _require_cupy()
    rois = np.asarray(rois, dtype=np.float64)
    masks = np.asarray(masks, dtype=bool)
    N = rois.shape[0]
    if N == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)
    if boundary_margin is None:
        boundary_margin = max(1, upsample_factor // 2)

    factor = int(upsample_factor)
    centers_all = np.empty((N, 2), dtype=np.float64)
    residuals_all = np.empty((N,), dtype=np.float64)

    for s in range(0, N, gpu_batch):
        e = min(N, s + gpu_batch)
        c, r = _radial_symmetry_inner_gpu(
            rois[s:e], masks[s:e],
            factor=factor, boundary_margin=boundary_margin, device=device,
        )
        centers_all[s:e] = c
        residuals_all[s:e] = r

    if residual_max is not None:
        bad = residuals_all > residual_max
        centers_all[bad] = np.nan
        residuals_all[bad] = np.inf

    return centers_all, residuals_all
```

The inner function `_radial_symmetry_inner_gpu` contains the existing GPU computation but uses float32 for intermediates:

```python
def _radial_symmetry_inner_gpu(
    rois: np.ndarray,     # (B, H, W) float64
    masks: np.ndarray,    # (B, H, W) bool
    *,
    factor: int,
    boundary_margin: int,
    device: int,
) -> tuple[np.ndarray, np.ndarray]:
    B = rois.shape[0]
    orig_hs = np.full(B, rois.shape[1], dtype=np.int32)
    orig_ws = np.full(B, rois.shape[2], dtype=np.int32)

    with gpu_device(device):
        # Float32 upsampling
        if factor > 1:
            from .upsample import batch_bicubic_upsample_f32, batch_nn_upsample_f32
            rois_gpu = cp.asarray(rois, dtype=cp.float32)
            masks_gpu = cp.asarray(masks.astype(np.float32))
            g_batch = batch_bicubic_upsample_f32(rois_gpu, factor)
            m_batch = batch_nn_upsample_f32(masks_gpu, factor) > 0.5
            del rois_gpu, masks_gpu  # free input copies early
        else:
            g_batch = cp.asarray(rois, dtype=cp.float32)
            m_batch = cp.asarray(masks)

        Hmax = g_batch.shape[1]
        Wmax = g_batch.shape[2]
        up_hs = np.full(B, Hmax, dtype=np.int32)
        up_ws = np.full(B, Wmax, dtype=np.int32)

        # Erode mask (same logic as before)
        min_up_dim = int(min(Hmax, Wmax))
        max_margin = max(0, (min_up_dim - 11) // 2)
        eff_margin = min(boundary_margin, max_margin)
        if eff_margin > 0:
            struct = cp.ones((1, 2 * eff_margin + 1, 2 * eff_margin + 1), dtype=bool)
            m_eroded = cndi.binary_erosion(m_batch, structure=struct)
        else:
            m_eroded = m_batch.copy()

        # Gradients at midpoints — all in float32
        I = g_batch
        Hp = Hmax - 1
        Wp = Wmax - 1
        if Hp < 1 or Wp < 1:
            return (np.full((B, 2), np.nan, dtype=np.float64),
                    np.full((B,), np.inf, dtype=np.float64))

        dIdu = I[:, :Hp, 1:Wp+1] - I[:, 1:Hp+1, :Wp]
        dIdv = I[:, :Hp, :Wp]    - I[:, 1:Hp+1, 1:Wp+1]

        if Hp >= 3 and Wp >= 3:
            def _smooth3x3(arr):
                p = cp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
                return (
                    p[:, 0:-2, 0:-2] + p[:, 0:-2, 1:-1] + p[:, 0:-2, 2:]
                    + p[:, 1:-1, 0:-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:]
                    + p[:, 2:,   0:-2] + p[:, 2:,   1:-1] + p[:, 2:,   2:]
                ) / cp.float32(9.0)
            dIdu = _smooth3x3(dIdu)
            dIdv = _smooth3x3(dIdv)

        # Validity mask
        m00 = m_eroded[:, :Hp, :Wp]
        m01 = m_eroded[:, :Hp, 1:Wp+1]
        m10 = m_eroded[:, 1:Hp+1, :Wp]
        m11 = m_eroded[:, 1:Hp+1, 1:Wp+1]
        valid = m00 & m01 & m10 & m11

        # Slope — float32
        denom = dIdu - dIdv
        near_zero = cp.abs(denom) < cp.float32(1e-6)
        safe_denom = cp.where(near_zero, cp.float32(1.0), denom)
        slope = -(dIdv + dIdu) / safe_denom
        slope = cp.where(near_zero, cp.where(
            -(dIdv + dIdu) >= 0, cp.float32(1e9), cp.float32(-1e9)
        ), slope)

        # Midpoint coordinates — float32
        up_ws_g = cp.asarray(up_ws, dtype=cp.float32)
        up_hs_g = cp.asarray(up_hs, dtype=cp.float32)
        col_idx = cp.arange(Wp, dtype=cp.float32)[None, None, :] + 0.5
        row_idx = cp.arange(Hp, dtype=cp.float32)[None, :, None] + 0.5
        cx_off = (up_ws_g - 1.0) / 2.0
        cy_off = (up_hs_g - 1.0) / 2.0
        xm = col_idx - cx_off[:, None, None]
        ym = row_idx - cy_off[:, None, None]

        # Intercepts, weights — float32
        b = ym - slope * xm
        grad_mag_sq = dIdu**2 + dIdv**2
        w_base = cp.where(valid, grad_mag_sq, cp.float32(0.0))
        w_sum = w_base.sum(axis=(1, 2))
        w_sum_safe = cp.maximum(w_sum, cp.float32(1e-30))
        gc_x = (w_base * xm).sum(axis=(1, 2)) / w_sum_safe
        gc_y = (w_base * ym).sum(axis=(1, 2)) / w_sum_safe
        dist_sq = (xm - gc_x[:, None, None])**2 + (ym - gc_y[:, None, None])**2
        dist = cp.sqrt(cp.maximum(dist_sq, cp.float32(1e-30)))
        w = cp.where(valid, grad_mag_sq / dist, cp.float32(0.0))

        # WLS solve — promote to float64 for precision
        slope_f64 = slope.astype(cp.float64)
        b_f64 = b.astype(cp.float64)
        w_f64 = w.astype(cp.float64)

        wm2p1 = w_f64 / (slope_f64**2 + 1.0)
        sw = wm2p1.sum(axis=(1, 2))
        smmw = (slope_f64**2 * wm2p1).sum(axis=(1, 2))
        smw = (slope_f64 * wm2p1).sum(axis=(1, 2))
        smbw = (slope_f64 * b_f64 * wm2p1).sum(axis=(1, 2))
        sbw = (b_f64 * wm2p1).sum(axis=(1, 2))

        det = smmw * sw - smw * smw
        bad_det = cp.abs(det) < cp.float64(1e-30)
        det_safe = cp.where(bad_det, cp.float64(1.0), det)
        xc = cp.where(bad_det, cp.nan, (-smbw * sw + smw * sbw) / det_safe)
        yc = cp.where(bad_det, cp.nan, (smmw * sbw - smbw * smw) / det_safe)

        # Residual
        perp_d_sq = (slope_f64 * xc[:, None, None] - yc[:, None, None] + b_f64)**2 / (slope_f64**2 + 1.0)
        residual = (w_f64 * perp_d_sq).sum(axis=(1, 2)) / cp.maximum(sw, cp.float64(1e-30))

        xc_cpu = cp.asnumpy(xc)
        yc_cpu = cp.asnumpy(yc)
        residual_cpu = cp.asnumpy(residual)

    # Vectorized coordinate transform (no Python loop)
    up_ws_f = up_ws.astype(np.float64)
    up_hs_f = up_hs.astype(np.float64)
    orig_ws_f = orig_ws.astype(np.float64)
    orig_hs_f = orig_hs.astype(np.float64)

    xc_pix_up = xc_cpu + (up_ws_f - 1.0) / 2.0
    yc_pix_up = yc_cpu + (up_hs_f - 1.0) / 2.0

    if factor > 1:
        centers_x = xc_pix_up * (orig_ws_f - 1.0) / np.maximum(up_ws_f - 1.0, 1.0)
        centers_y = yc_pix_up * (orig_hs_f - 1.0) / np.maximum(up_hs_f - 1.0, 1.0)
    else:
        centers_x = xc_pix_up
        centers_y = yc_pix_up

    centers = np.column_stack([centers_x, centers_y])

    # Quality gate: in-bounds
    oob = (centers[:, 0] < 0) | (centers[:, 0] >= orig_ws_f) | \
          (centers[:, 1] < 0) | (centers[:, 1] >= orig_hs_f)
    centers[oob] = np.nan
    residual_cpu[oob] = np.inf

    return centers, residual_cpu
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unification.py tests/test_radial_symmetry.py -v`
Expected: All PASS. The existing radial symmetry tests must still pass since the API is unchanged (just gains an optional `gpu_batch` param).

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "feat: add GPU batching + float32 internals to radial symmetry"
```

---

### Task 3: Add GPU batching + float32 to `_isophote_curvature_batch_gpu`

**Files:**
- Modify: `subpx/_gpu/centers.py` (function `_isophote_curvature_batch_gpu`, lines 1204-1398)
- Test: `tests/test_unification.py`

Same treatment as radial symmetry: batch loop + float32 intermediates + vectorized coordinate transform.

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_isophote_batched_matches_unbatched():
    """Batched isophote curvature (gpu_batch=3) matches full batch."""
    from subpx._gpu.centers import _isophote_curvature_batch_gpu

    rng = np.random.RandomState(42)
    N = 10
    size = 15
    rois = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        yy, xx = np.mgrid[:size, :size]
        blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 1.5**2))
        rois.append(blob)
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)

    c_full, s_full = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=10000,
    )
    c_batched, s_batched = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0, gpu_batch=3,
    )
    np.testing.assert_allclose(c_full, c_batched, atol=1e-10)
    np.testing.assert_allclose(s_full, s_batched, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_isophote_batched_matches_unbatched -v`
Expected: FAIL — `gpu_batch` parameter not accepted.

- [ ] **Step 3: Implement batched isophote curvature with float32 internals**

Apply the same pattern as Task 2:

1. Add `gpu_batch` parameter to `_isophote_curvature_batch_gpu`.
2. Extract the core computation into `_isophote_curvature_inner_gpu(rois, masks, *, factor, boundary_margin, pre_smooth_sigma, device)`.
3. The outer function loops over batches of `gpu_batch` ROIs.
4. Change the inner function to use float32 for the image, first derivatives (`Ix`, `Iy`), second derivatives (`Ixx`, `Iyy`, `Ixy`), displacements (`Dx`, `Dy`), curvedness weights, and vote coordinates. Keep the weighted average center and spread in float64.
5. Vectorize the coordinate-transform loop (lines 1372-1397).

The changes mirror Task 2 exactly in structure. Key differences:
- Isophote uses central-difference derivatives instead of diagonal midpoint gradients
- The "solve" is a weighted average (no matrix solve), but still promote to float64 for the accumulation
- Gaussian pre-smoothing (`cndi.gaussian_filter`) should operate on float32 data

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unification.py tests/test_isophote_curvature.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "feat: add GPU batching + float32 internals to isophote curvature"
```

---

### Task 4: Extract `_detect_single_tile_gpu` core function

**Files:**
- Modify: `subpx/_gpu/centers.py`
- Test: `tests/test_unification.py`

Extract the non-tiled detection logic from `detect_centers_gpu()` into a reusable `_detect_single_tile_gpu()` core. This function receives a single image/tile, performs segmentation, connected components, area filtering, and refinement dispatch. It accepts an optional pre-computed binary threshold.

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_detect_single_tile_gpu_exists():
    """_detect_single_tile_gpu must be importable and produce centers."""
    from subpx._gpu.centers import _detect_single_tile_gpu

    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    img[30:34, 40:44] = 200

    centers, meta = _detect_single_tile_gpu(
        img, threshold="triangle", invert=False,
        area_min=4, area_max=100, morph_open=0, morph_close=0,
        pad=3, refine="weighted", connectivity=8,
        gpu_batch=4096, device=0, use_float64=True,
        components_backend="cpu", small_feature_max=12.0,
        upsample_factor=4, pre_threshold=None,
    )
    assert centers.shape == (2, 2)
    assert isinstance(meta, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_detect_single_tile_gpu_exists -v`
Expected: FAIL — `_detect_single_tile_gpu` does not exist.

- [ ] **Step 3: Extract `_detect_single_tile_gpu` from `detect_centers_gpu`**

Create `_detect_single_tile_gpu()` by moving the body of `detect_centers_gpu()` into it. The function:

```python
def _detect_single_tile_gpu(
    image: np.ndarray,
    *,
    threshold: str,
    invert: bool,
    area_min: int,
    area_max: int,
    morph_open: int,
    morph_close: int,
    pad: int,
    refine: str,
    connectivity: int,
    gpu_batch: int,
    device: int,
    use_float64: bool,
    components_backend: str,
    small_feature_max: float,
    upsample_factor: int,
    pre_threshold: float | None,   # If set, use this fixed threshold instead of auto-computing
) -> tuple[np.ndarray, dict]:
    """Process a single image/tile through segmentation + refinement.

    Returns (centers_xy, meta_dict) where centers are in tile-local coordinates.
    """
    # Segmentation: use pre_threshold if provided, else auto-threshold
    ...
    # Connected components
    ...
    # Area filtering
    ...
    # Refinement dispatch (all existing method branches)
    ...
    return centers_xy, meta
```

Then rewrite `detect_centers_gpu()` to call it:

```python
def detect_centers_gpu(image, *, tile_h=None, threshold_mode="auto", overlap=128,
                       otsu_downsample=4, thr_scale=0.8, dedupe_eps=1.5, **kwargs):
    img = np.asarray(image)
    if tile_h is None:
        centers, meta = _detect_single_tile_gpu(img, pre_threshold=None, **kwargs)
        return CenterResult(centers_xy=centers, method=..., backend="gpu", meta=meta)
    else:
        # Tiling path (covered in Task 5)
        ...
```

- [ ] **Step 4: Run tests to verify existing behavior preserved**

Run: `pytest tests/test_radial_symmetry.py tests/test_smoke.py tests/test_isophote_curvature.py tests/test_unification.py -v`
Expected: All PASS. The public `detect_centers(..., backend="gpu")` call path is unchanged.

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "refactor: extract _detect_single_tile_gpu core from detect_centers_gpu"
```

---

### Task 5: Merge tiled GPU path into `detect_centers_gpu`

**Files:**
- Modify: `subpx/_gpu/centers.py`
- Test: `tests/test_unification.py`

Add the tiling loop to `detect_centers_gpu()`, replacing the standalone `_detect_centers_tiled_gpu()`. The tiling loop computes a global threshold (when `threshold_mode` is `"global"` or `"auto"`), iterates tiles, calls `_detect_single_tile_gpu()` per tile, offsets coordinates, and deduplicates at boundaries.

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_detect_centers_gpu_tiled_basic():
    """detect_centers_gpu with tile_h should produce centers from a tall image."""
    from subpx._gpu.centers import detect_centers_gpu

    # 10x4 grid of dots
    def _make_dot_grid(rows, cols, spacing, dot_size, margin):
        H = 2 * margin + (rows - 1) * spacing + dot_size
        W = 2 * margin + (cols - 1) * spacing + dot_size
        img = np.zeros((H, W), dtype=np.uint8)
        expected = []
        for r in range(rows):
            for c in range(cols):
                y0 = margin + r * spacing
                x0 = margin + c * spacing
                img[y0:y0+dot_size, x0:x0+dot_size] = 255
                expected.append([x0 + (dot_size-1)/2.0, y0 + (dot_size-1)/2.0])
        return img, np.array(expected, dtype=np.float64)

    img, expected = _make_dot_grid(rows=10, cols=4, spacing=20, dot_size=3, margin=10)
    H = img.shape[0]
    result = detect_centers_gpu(
        img, tile_h=H // 3, overlap=30, threshold_mode="global",
        area_min=1, area_max=50, refine="weighted",
    )
    assert isinstance(result, CenterResult)
    assert abs(result.centers_xy.shape[0] - expected.shape[0]) <= 2


@skipno_gpu
def test_detect_centers_gpu_tiled_threshold_modes():
    """Both 'global' and 'per_tile' threshold modes should work."""
    from subpx._gpu.centers import detect_centers_gpu

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    for mode in ("global", "per_tile"):
        result = detect_centers_gpu(
            img, tile_h=100, overlap=20, threshold_mode=mode,
            area_min=4, area_max=100, refine="weighted",
        )
        assert result.centers_xy.shape[0] == 2, f"mode={mode}: expected 2 centers"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_detect_centers_gpu_tiled_basic -v`
Expected: FAIL — `detect_centers_gpu` does not accept `tile_h`.

- [ ] **Step 3: Implement tiling loop in `detect_centers_gpu`**

Add `tile_h`, `overlap`, `threshold_mode`, `otsu_downsample`, `thr_scale`, `dedupe_eps` parameters to `detect_centers_gpu()`. When `tile_h is not None`:

```python
# In detect_centers_gpu, after the non-tiled path:
if tile_h is not None:
    import cv2
    H, W = img.shape
    if overlap >= tile_h:
        raise ValueError("overlap must be < tile_h")

    # Global threshold if needed
    pre_threshold = None
    if threshold_mode in ("global", "auto"):
        ds = max(1, int(otsu_downsample))
        thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        g_ds = img[::ds, ::ds]
        otsu_thresh, _ = cv2.threshold(g_ds, 0, 255, thr_type | cv2.THRESH_OTSU)
        pre_threshold = float(otsu_thresh) * float(thr_scale)

    step = tile_h - overlap
    all_centers = []
    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + tile_h)
        tile = img[y0:y1]

        centers, _ = _detect_single_tile_gpu(
            tile, pre_threshold=pre_threshold, **detection_kwargs,
        )

        if centers.shape[0] > 0:
            centers[:, 1] += y0  # tile-local → global
            half = overlap // 2
            keep_lo = y0 if y0 == 0 else (y0 + half)
            keep_hi = y1 if y1 == H else (y1 - half)
            m = (centers[:, 1] >= keep_lo) & (centers[:, 1] < keep_hi)
            if np.any(m):
                all_centers.append(centers[m])

        if y1 == H:
            break
        y0 += step

    if all_centers:
        centers = np.vstack(all_centers)
        from ..centers import dedupe_centers
        centers = dedupe_centers(centers, eps=float(dedupe_eps))
    else:
        centers = np.zeros((0, 2), dtype=np.float64)

    return CenterResult(centers_xy=centers, ...)
```

After this, delete `_detect_centers_tiled_gpu()` entirely since its logic is now absorbed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unification.py tests/test_tiled_gpu_pipeline.py tests/test_radial_symmetry.py -v`

Note: Tests in `test_tiled_gpu_pipeline.py` that import `_detect_centers_tiled_gpu` directly will fail. This is expected — they'll be updated in Task 8.

Expected: `test_unification.py` tests PASS. Some `test_tiled_gpu_pipeline.py` tests may fail (those importing deleted internal function).

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "feat: merge tiled GPU path into detect_centers_gpu with tile_h parameter"
```

---

### Task 6: Add tiling support to `detect_centers_cpu`

**Files:**
- Modify: `subpx/_cpu/centers.py`
- Test: `tests/test_unification.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
def test_detect_centers_cpu_tiled():
    """CPU detect_centers_cpu with tile_h should tile and dedupe."""
    from subpx._cpu.centers import detect_centers_cpu

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    result = detect_centers_cpu(
        img, tile_h=100, overlap=20, dedupe_eps=1.5,
        area_min=4, area_max=100, refine="weighted",
    )
    assert result.centers_xy.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_detect_centers_cpu_tiled -v`
Expected: FAIL — `detect_centers_cpu` does not accept `tile_h`.

- [ ] **Step 3: Implement tiling in `detect_centers_cpu`**

Add `tile_h=None`, `overlap=128`, `dedupe_eps=1.5` parameters to `detect_centers_cpu()`. When `tile_h` is set, loop over vertical tiles, call the existing untiled detection per tile, offset coordinates, and dedupe at boundaries. This uses the same pattern as the current CPU path in the old `detect_centers_tiled`.

```python
def detect_centers_cpu(image, *, tile_h=None, overlap=128, dedupe_eps=1.5, **existing_params):
    if tile_h is None:
        # Existing code path (unchanged)
        ...
    else:
        # Tiling loop
        from ..centers import dedupe_centers
        H, W = image.shape
        step = tile_h - overlap
        all_centers = []
        y0 = 0
        while y0 < H:
            y1 = min(H, y0 + tile_h)
            tile = image[y0:y1]
            res = detect_centers_cpu(tile, **existing_params)  # recurse without tiling
            pts = np.asarray(res.centers_xy, dtype=np.float64)
            if pts.size > 0:
                pts[:, 1] += y0
                half = overlap // 2
                keep_lo = y0 if y0 == 0 else (y0 + half)
                keep_hi = y1 if y1 == H else (y1 - half)
                m = (pts[:, 1] >= keep_lo) & (pts[:, 1] < keep_hi)
                if np.any(m):
                    all_centers.append(pts[m])
            if y1 == H:
                break
            y0 += step
        centers = np.vstack(all_centers) if all_centers else np.zeros((0, 2), dtype=np.float64)
        if centers.size:
            centers = dedupe_centers(centers, eps=float(dedupe_eps))
        return CenterResult(centers_xy=centers, method=f"tiled_cpu(...)", backend="cpu", meta={...})
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_unification.py::test_detect_centers_cpu_tiled tests/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/_cpu/centers.py tests/test_unification.py
git commit -m "feat: add tile_h tiling support to detect_centers_cpu"
```

---

### Task 7: Unify public `detect_centers()` in `centers.py`

**Files:**
- Modify: `subpx/centers.py`
- Test: `tests/test_unification.py`

Merge the three public functions into one. Remove `detect_centers_tiled` and `detect_centers_tiled_global_otsu` from this file.

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
def test_unified_detect_centers_with_tile_h_cpu():
    """Public detect_centers with tile_h should work on CPU."""
    from subpx.centers import detect_centers

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    result = detect_centers(
        img, backend="cpu", tile_h=100, overlap=20,
        area_min=4, area_max=100, refine="weighted",
    )
    assert result.centers_xy.shape[0] == 2


@skipno_gpu
def test_unified_detect_centers_with_tile_h_gpu():
    """Public detect_centers with tile_h should work on GPU."""
    from subpx.centers import detect_centers

    img = np.zeros((200, 100), dtype=np.uint8)
    img[20:24, 20:24] = 200
    img[120:124, 60:64] = 200

    result = detect_centers(
        img, backend="gpu", tile_h=100, overlap=20,
        area_min=4, area_max=100, refine="weighted",
    )
    assert result.centers_xy.shape[0] == 2


def test_unified_no_tile_h_unchanged():
    """detect_centers without tile_h must behave exactly as before."""
    from subpx.centers import detect_centers

    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    img[30:34, 40:44] = 200

    result = detect_centers(img, backend="cpu", area_min=4, area_max=100, refine="weighted")
    assert result.centers_xy.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_unified_detect_centers_with_tile_h_cpu -v`
Expected: FAIL — `detect_centers` does not accept `tile_h`.

- [ ] **Step 3: Unify `detect_centers` in `centers.py`**

1. Add `tile_h`, `overlap`, `threshold_mode`, `otsu_downsample`, `thr_scale`, `dedupe_eps` parameters to `detect_centers()`.
2. Pass them through to the backend functions: `detect_centers_gpu(... tile_h=tile_h, ...)` and `detect_centers_cpu(... tile_h=tile_h, ...)`.
3. Delete `detect_centers_tiled()` (lines 190-279) and `detect_centers_tiled_global_otsu()` (lines 346-446) from this file.

Updated `detect_centers`:

```python
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
):
    b = resolve_backend(backend)
    img = to_numpy(image)
    if b == "cpu" and refine in _VORONOI_METHODS:
        raise NotImplementedError(
            f"refine='{refine}' requires GPU backend."
        )
    if b == "gpu":
        return detect_centers_gpu(
            img,
            threshold=threshold, invert=invert,
            area_min=area_min, area_max=area_max,
            morph_open=morph_open, morph_close=morph_close,
            pad=pad, refine=refine, connectivity=connectivity,
            gpu_batch=gpu_batch, device=device, use_float64=use_float64,
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_unification.py tests/test_smoke.py tests/test_radial_symmetry.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/centers.py tests/test_unification.py
git commit -m "feat: unify detect_centers with tile_h parameter, remove detect_centers_tiled"
```

---

### Task 8: Update `__init__.py`, `legacy.py`, `batch.py`, and tests

**Files:**
- Modify: `subpx/__init__.py`
- Modify: `subpx/legacy.py`
- Modify: `subpx/batch.py`
- Modify: `tests/test_tiled_gpu_pipeline.py`
- Modify: `tests/test_radial_symmetry.py`
- Test: `tests/test_unification.py`

- [ ] **Step 1: Write failing test for deprecated aliases**

Append to `tests/test_unification.py`:

```python
def test_deprecated_detect_centers_tiled_warns():
    """Deprecated detect_centers_tiled should warn and still work."""
    import warnings
    from subpx.legacy import detect_centers_tiled

    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = detect_centers_tiled(img, backend="cpu", area_min=4, area_max=100, refine="weighted")
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "detect_centers_tiled" in str(w[0].message)
    assert result.centers_xy.shape[0] == 1


def test_deprecated_detect_centers_tiled_global_otsu_warns():
    """Deprecated detect_centers_tiled_global_otsu should warn and still work."""
    import warnings
    from subpx.legacy import detect_centers_tiled_global_otsu

    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = detect_centers_tiled_global_otsu(
            img, backend="cpu", area_min=4, area_max=100, refine="weighted",
        )
        assert len(w) >= 1
        assert issubclass(w[0].category, DeprecationWarning)
    assert result.centers_xy.shape[0] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unification.py::test_deprecated_detect_centers_tiled_warns -v`
Expected: FAIL — `detect_centers_tiled` not in `legacy.py`.

- [ ] **Step 3: Update `legacy.py`**

Add deprecated aliases. Update the import at top to use the new unified `detect_centers`:

```python
# At top of legacy.py, update import:
from .centers import detect_centers, refine_weighted_centroid, refine_logquadratic, filter_centers, dedupe_centers

# Add new deprecated wrappers:
def detect_centers_tiled(image, *, tile_h=8192, **kwargs):
    _warn("detect_centers_tiled", "detect_centers(image, tile_h=...)")
    return detect_centers(image, tile_h=tile_h, **kwargs)


def detect_centers_tiled_global_otsu(image, *, tile_h=8192, threshold_mode="global", **kwargs):
    _warn("detect_centers_tiled_global_otsu", "detect_centers(image, tile_h=..., threshold_mode='global')")
    return detect_centers(image, tile_h=tile_h, threshold_mode=threshold_mode, **kwargs)
```

Also update the existing `find_subpixel_centers_tiled_logquad_gpu` to use the new unified function.

- [ ] **Step 4: Update `__init__.py`**

Change the centers import to remove the deleted functions and re-export from legacy instead:

```python
from .centers import (
    recommend_refine_method,
    detect_centers,
    refine_centers,
    refine_weighted_centroid,
    refine_logquadratic,
    refine_centers_edge_moment,
    filter_centers,
    dedupe_centers,
)
# Backward compat — deprecated, will be removed
from .legacy import detect_centers_tiled, detect_centers_tiled_global_otsu
```

Keep `detect_centers_tiled` and `detect_centers_tiled_global_otsu` in `__all__` for now (they're deprecated, not removed from the public surface).

- [ ] **Step 5: Update `batch.py`**

Change `detect_centers_tiled_batch` to use unified API:

```python
def detect_centers_tiled_batch(frames: Iterable, /, *args, tile_h: int = 8192, progress: bool = False, **kwargs):
    from .centers import detect_centers
    def _tiled(frame, *a, **kw):
        return detect_centers(frame, *a, tile_h=tile_h, **kw)
    return process_frames(frames, _tiled, *args, progress=progress, **kwargs)
```

- [ ] **Step 6: Update existing tests**

In `tests/test_tiled_gpu_pipeline.py`:
- Tests that import `_detect_centers_tiled_gpu` directly need to be updated to call `detect_centers_gpu` with `tile_h=` parameter or call `detect_centers` with `tile_h=`.
- Tests that import `detect_centers_tiled` from `subpx` continue to work (deprecated alias still exports it).

In `tests/test_radial_symmetry.py`:
- `test_detect_centers_tiled_radial_symmetry` (line 135): uses `from subpx.centers import detect_centers_tiled` — change to `from subpx import detect_centers_tiled` (which now comes from legacy) or update to use `detect_centers(... tile_h=64, ...)`.
- `test_detect_centers_tiled_global_otsu_radial_symmetry_raises` (line 153): same treatment.

- [ ] **Step 7: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 8: Commit**

```bash
git add subpx/__init__.py subpx/legacy.py subpx/batch.py tests/test_tiled_gpu_pipeline.py tests/test_radial_symmetry.py tests/test_unification.py
git commit -m "feat: add deprecated aliases, update exports and tests for unified API"
```

---

### Task 9: Vectorize `_extract_voronoi_rois`

**Files:**
- Modify: `subpx/_gpu/centers.py` (function `_extract_voronoi_rois`, lines 1401-1465)
- Test: `tests/test_unification.py`

The current implementation loops over every blob in Python. Vectorize the bounding box computation, background estimation, and ROI extraction.

- [ ] **Step 1: Write failing test**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_extract_voronoi_rois_vectorized_matches():
    """Vectorized _extract_voronoi_rois must match original loop output."""
    from subpx._gpu.centers import _extract_voronoi_rois
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu

    # Build a simple image with 4 blobs
    img = np.zeros((64, 64), dtype=np.float64)
    yy, xx = np.mgrid[:64, :64]
    rows = []
    for cy, cx in [(15, 15), (15, 45), (45, 15), (45, 45)]:
        img += 200 * np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * 2.0**2))
        rows.append([cx - 4, cy - 4, 8, 8, float(cx), float(cy)])

    seeds = np.array([[r[4], r[5]] for r in rows], dtype=np.float64)
    voronoi_labels = compute_voronoi_labels_gpu(seeds, (64, 64), device=0)

    rois, masks, origins = _extract_voronoi_rois(img, voronoi_labels, rows, pad=3)
    assert rois.shape[0] == 4
    assert masks.shape[0] == 4
    assert len(origins) == 4
    # Each ROI should have some non-zero pixels
    for i in range(4):
        assert np.any(masks[i])
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `pytest tests/test_unification.py::test_extract_voronoi_rois_vectorized_matches -v`
Expected: PASS (function exists, but is slow with Python loop).

- [ ] **Step 3: Vectorize `_extract_voronoi_rois`**

Replace the Python loop with vectorized numpy:

```python
def _extract_voronoi_rois(gray, voronoi_labels, rows, pad=3):
    H, W = gray.shape
    N = len(rows)
    rows_arr = np.asarray(rows, dtype=np.float64)

    # Vectorized bounding boxes
    x0 = np.maximum(0, rows_arr[:, 0].astype(np.int32) - pad)
    y0 = np.maximum(0, rows_arr[:, 1].astype(np.int32) - pad)
    x1 = np.minimum(W, (rows_arr[:, 0] + rows_arr[:, 2]).astype(np.int32) + pad)
    y1 = np.minimum(H, (rows_arr[:, 1] + rows_arr[:, 3]).astype(np.int32) + pad)
    hs = y1 - y0
    ws = x1 - x0
    Hm = int(hs.max()) if N else 0
    Wm = int(ws.max()) if N else 0

    rois_stack = np.zeros((N, Hm, Wm), dtype=np.float64)
    masks_stack = np.zeros((N, Hm, Wm), dtype=bool)
    origins = list(zip(x0.tolist(), y0.tolist()))

    # This loop is unavoidable for variable-size ROI extraction,
    # but the per-iteration work is minimal (slicing + mask)
    for j in range(N):
        h_r = int(hs[j])
        w_r = int(ws[j])
        roi = gray[int(y0[j]):int(y1[j]), int(x0[j]):int(x1[j])].astype(np.float64)
        vmask = voronoi_labels[int(y0[j]):int(y1[j]), int(x0[j]):int(x1[j])] == j

        # Background: 10th percentile of border pixels (vectorized erosion)
        inner = vmask.copy()
        inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
        inner[1:, :] &= vmask[:-1, :]
        inner[:-1, :] &= vmask[1:, :]
        inner[:, 1:] &= vmask[:, :-1]
        inner[:, :-1] &= vmask[:, 1:]
        border_vals = roi[vmask & ~inner]
        bg = float(np.percentile(border_vals, 10)) if border_vals.size > 0 else 0.0

        rois_stack[j, :, :] = bg  # fill entire ROI with bg first
        rois_stack[j, :h_r, :w_r] = np.where(vmask, roi, bg)
        masks_stack[j, :h_r, :w_r] = vmask

    return rois_stack, masks_stack, origins
```

Note: The inner loop over N blobs remains because each ROI has different dimensions and a different region of the image to extract. The improvement is that bounding box computation is vectorized and per-ROI work is minimal. For truly eliminating the loop, a CuPy kernel would be needed, but that's out of scope for this task.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_unification.py tests/test_radial_symmetry.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "refactor: vectorize bounding box computation in _extract_voronoi_rois"
```

---

### Task 10: Final integration test and cleanup

**Files:**
- Modify: `tests/test_unification.py`
- Modify: `subpx/_gpu/centers.py` (remove dead code)

- [ ] **Step 1: Write integration test**

Append to `tests/test_unification.py`:

```python
@skipno_gpu
def test_full_pipeline_tiled_radial_symmetry():
    """End-to-end: detect_centers(tile_h=..., refine='radial_symmetry') on GPU."""
    from subpx.centers import detect_centers

    img = np.zeros((128, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:128, :64]
    img += (200 * np.exp(-((xx - 20)**2 + (yy - 20)**2) / (2 * 2.0**2))).astype(np.uint8)
    img += (200 * np.exp(-((xx - 40)**2 + (yy - 80)**2) / (2 * 2.0**2))).astype(np.uint8)

    result = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        tile_h=64, overlap=16, area_min=4, area_max=200,
        upsample_factor=4, gpu_batch=1024,
    )
    assert result.centers_xy.shape[0] == 2
    ys = sorted(result.centers_xy[:, 1])
    assert 15 < ys[0] < 30
    assert 75 < ys[1] < 90


@skipno_gpu
def test_gpu_batch_param_propagates_to_voronoi():
    """gpu_batch parameter should be used by Voronoi methods."""
    from subpx.centers import detect_centers

    img = np.zeros((64, 64), dtype=np.uint8)
    yy, xx = np.mgrid[:64, :64]
    for cy, cx in [(15, 15), (15, 45), (45, 15), (45, 45)]:
        img += (200 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*2.0**2))).astype(np.uint8)

    # Very small gpu_batch to force multiple batches
    result = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=200, upsample_factor=4, gpu_batch=2,
    )
    assert result.centers_xy.shape[0] == 4
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_unification.py::test_full_pipeline_tiled_radial_symmetry tests/test_unification.py::test_gpu_batch_param_propagates_to_voronoi -v`
Expected: PASS.

- [ ] **Step 3: Remove dead code from `_gpu/centers.py`**

Delete the old `_detect_centers_tiled_gpu` function if it wasn't already removed in Task 5. Verify no other code references it.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_unification.py
git commit -m "test: add integration tests for unified API; remove dead code"
```
