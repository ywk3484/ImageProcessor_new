# Radial Symmetry & Isophote Curvature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two GPU-accelerated refinement methods (`radial_symmetry`, `isophote_curvature`) with Voronoi geometric partitioning and bicubic upsampling for small, closely-packed PSF-like blobs.

**Architecture:** Voronoi label map computed once from coarse CC centroids, then each blob's ROI is Voronoi-masked, background-filled, bicubic-upsampled 4×, and refined via either Parthasarathy radial symmetry (closed-form gradient intersection) or isophote curvature center (derivative-based center voting). Both methods return `CenterResult` with per-blob quality metrics.

**Tech Stack:** CuPy (GPU compute), cupyx.scipy.ndimage.zoom (bicubic upsampling), NumPy (output arrays), pytest (tests)

**Spec:** `docs/superpowers/specs/2026-03-25-radial-symmetry-isophote-curvature-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `subpx/_gpu/voronoi.py` | Create | Voronoi partitioning on GPU (grid-accelerated nearest neighbor) |
| `subpx/_gpu/centers.py` | Modify | Add `_refine_voronoi_batch_gpu()`, radial symmetry kernel, isophote curvature kernel; modify `detect_centers_gpu()` dispatch |
| `subpx/centers.py` | Modify | Thread `upsample_factor` through public API; add `NotImplementedError` for CPU; guard tiled detection |
| `subpx/visualization.py` | Create | `draw_voronoi_boundaries()` function |
| `tests/test_voronoi.py` | Create | Tests for Voronoi partitioning |
| `tests/test_radial_symmetry.py` | Create | Tests for radial symmetry refinement |
| `tests/test_isophote_curvature.py` | Create | Tests for isophote curvature refinement |
| `tests/test_visualization.py` | Create | Tests for Voronoi visualization |

---

### Task 1: GPU Voronoi Partitioning

**Files:**
- Create: `subpx/_gpu/voronoi.py`
- Create: `tests/test_voronoi.py`

- [ ] **Step 1: Write failing tests for Voronoi partitioning**

```python
# tests/test_voronoi.py
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
def test_voronoi_two_seeds():
    """Two seeds split a 20x20 image vertically."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[5.0, 10.0], [15.0, 10.0]])  # x, y
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    assert labels.shape == (20, 20)
    assert labels.dtype == np.int32
    # pixel (col=2, row=10) is closer to seed 0 (x=5)
    assert labels[10, 2] == 0
    # pixel (col=17, row=10) is closer to seed 1 (x=15)
    assert labels[10, 17] == 1


@skipno_gpu
def test_voronoi_staggered_grid():
    """Staggered 2x3 grid: Voronoi cells should tile correctly."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    # Row 0: x=10, 30, 50.  Row 1: x=20, 40 (staggered)
    seeds = np.array([
        [10.0, 10.0], [30.0, 10.0], [50.0, 10.0],
        [20.0, 30.0], [40.0, 30.0],
    ])
    labels = compute_voronoi_labels_gpu(seeds, (40, 60), device=0)
    assert labels.shape == (40, 60)
    # Each seed should own at least some pixels
    unique = np.unique(labels)
    assert len(unique) == 5


@skipno_gpu
def test_voronoi_single_seed():
    """Single seed: entire image assigned to it."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[10.0, 10.0]])
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    assert np.all(labels == 0)


@skipno_gpu
def test_voronoi_cell_areas():
    """Cell areas should sum to total image area."""
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    seeds = np.array([[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]])
    labels = compute_voronoi_labels_gpu(seeds, (20, 20), device=0)
    areas = np.bincount(labels.ravel(), minlength=4)
    assert areas.sum() == 20 * 20
    # Roughly equal areas for symmetric seeds
    assert np.all(areas > 50)  # each should be ~100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_voronoi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'subpx._gpu.voronoi'`

- [ ] **Step 3: Implement `compute_voronoi_labels_gpu`**

Create `subpx/_gpu/voronoi.py`:

```python
"""GPU-accelerated Voronoi partitioning via grid-accelerated nearest neighbor."""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from ..backends import gpu_device


def _require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for GPU Voronoi partitioning.")


def compute_voronoi_labels_gpu(
    seeds_xy: np.ndarray,
    image_shape: tuple[int, int],
    *,
    device: int = 0,
) -> np.ndarray:
    """Assign each pixel to the nearest seed via Voronoi partitioning.

    Parameters
    ----------
    seeds_xy : (N, 2) float array
        Seed center positions in [x, y] order.
    image_shape : (H, W)
        Image dimensions.
    device : int
        GPU device index.

    Returns
    -------
    label_map : (H, W) int32 ndarray (NumPy)
        Each pixel assigned to the index of its nearest seed.
    """
    _require_cupy()
    seeds = np.asarray(seeds_xy, dtype=np.float64)
    N = seeds.shape[0]
    H, W = image_shape

    if N == 0:
        return np.full((H, W), -1, dtype=np.int32)
    if N == 1:
        return np.zeros((H, W), dtype=np.int32)

    with gpu_device(device):
        seeds_gpu = cp.asarray(seeds)  # (N, 2) [x, y]

        # Grid-accelerated nearest neighbor
        # Compute cell size from median nearest-neighbor distance
        if N > 1:
            # Pairwise distances between seeds (N x N)
            # For large N, tile this. For typical N < 50k, fits in memory.
            sx = seeds_gpu[:, 0]  # (N,)
            sy = seeds_gpu[:, 1]  # (N,)
            dx = sx[:, None] - sx[None, :]  # (N, N)
            dy = sy[:, None] - sy[None, :]  # (N, N)
            d2 = dx * dx + dy * dy
            # Set diagonal to inf to exclude self-distance
            d2[cp.arange(N), cp.arange(N)] = cp.inf
            nn_dists = cp.sqrt(cp.min(d2, axis=1))  # (N,)
            cell_size = float(cp.median(nn_dists).get()) * 1.5
        else:
            cell_size = float(max(H, W))

        cell_size = max(cell_size, 1.0)  # safety floor

        # Assign seeds to grid cells
        grid_cols = int(np.ceil(W / cell_size)) + 1
        grid_rows = int(np.ceil(H / cell_size)) + 1
        seed_gc = cp.floor(seeds_gpu / cell_size).astype(cp.int32)  # (N, 2) [gx, gy]

        # Build cell-to-seed mapping
        # For each grid cell, collect seed indices
        # Use a flat array approach: hash grid cell to 1D index
        cell_idx = seed_gc[:, 1] * grid_cols + seed_gc[:, 0]  # (N,)

        # For each pixel, find nearest seed by checking 9 neighboring cells
        # Process in tiles to limit memory
        TILE = 512
        label_map = cp.full((H, W), 0, dtype=cp.int32)
        best_d2 = cp.full((H, W), cp.inf, dtype=cp.float64)

        # Pre-transfer to CPU to avoid per-seed .get() calls (GPU→CPU sync)
        seed_gc_cpu = cp.asnumpy(seed_gc)  # (N, 2)
        seeds_cpu = np.asarray(seeds, dtype=np.float64)  # already numpy but ensure

        # Iterate over seeds: for each seed, update pixels in its 3x3 grid neighborhood
        for i in range(N):
            sx_i = float(seeds_cpu[i, 0])
            sy_i = float(seeds_cpu[i, 1])
            gx_i = int(seed_gc_cpu[i, 0])
            gy_i = int(seed_gc_cpu[i, 1])

            # Bounding box of pixels that could be nearest to this seed:
            # pixels in the 3x3 grid cells around this seed's cell
            px0 = max(0, int((gx_i - 1) * cell_size))
            py0 = max(0, int((gy_i - 1) * cell_size))
            px1 = min(W, int((gx_i + 2) * cell_size) + 1)
            py1 = min(H, int((gy_i + 2) * cell_size) + 1)

            if px1 <= px0 or py1 <= py0:
                continue

            # Compute distances for this rectangular region on GPU
            cols = cp.arange(px0, px1, dtype=cp.float64)
            rows_r = cp.arange(py0, py1, dtype=cp.float64)
            dc = cols[None, :] - sx_i  # (1, W_region)
            dr = rows_r[:, None] - sy_i  # (H_region, 1)
            dist2 = dc * dc + dr * dr  # (H_region, W_region)

            region_best = best_d2[py0:py1, px0:px1]
            closer = dist2 < region_best
            label_map[py0:py1, px0:px1] = cp.where(closer, i, label_map[py0:py1, px0:px1])
            best_d2[py0:py1, px0:px1] = cp.minimum(region_best, dist2)

        return cp.asnumpy(label_map)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_voronoi.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/voronoi.py tests/test_voronoi.py
git commit -m "feat: add GPU Voronoi partitioning for center refinement"
```

---

### Task 2: Parthasarathy Radial Symmetry — Core GPU Kernel

**Files:**
- Modify: `subpx/_gpu/centers.py`
- Create: `tests/test_radial_symmetry.py`

- [ ] **Step 1: Write failing tests for radial symmetry**

```python
# tests/test_radial_symmetry.py
import numpy as np
import pytest


def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


def _make_gaussian_blob(cx, cy, sigma, shape):
    """Create a 2D Gaussian blob centered at (cx, cy) in an image of given shape."""
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    blob = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
    return blob.astype(np.float64)


@skipno_gpu
def test_radial_symmetry_isolated_blob_accuracy():
    """Single Gaussian blob: center should be accurate to < 0.05 px."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])          # (1, 15, 15)
    masks = np.ones_like(rois, dtype=bool)
    centers, residuals = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0
    )
    assert centers.shape == (1, 2)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.05, f"Center error {err:.4f} px exceeds 0.05 px"


@skipno_gpu
def test_radial_symmetry_subpixel_positions():
    """Test multiple subpixel offsets to check for pixel-locking bias."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    offsets = [0.1, 0.25, 0.5, 0.75, 0.9]
    errors = []
    for dx in offsets:
        cx, cy = 7.0 + dx, 7.0 + dx
        blob = _make_gaussian_blob(cx, cy, sigma=1.5, shape=(15, 15))
        rois = np.stack([blob])
        masks = np.ones_like(rois, dtype=bool)
        centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
        err = np.sqrt((centers[0, 0] - cx)**2 + (centers[0, 1] - cy)**2)
        errors.append(err)
    # No single offset should have dramatically worse accuracy (pixel-locking)
    assert max(errors) < 0.1, f"Pixel-locking detected: max error {max(errors):.4f}"
    assert np.std(errors) < 0.03, f"Pixel-locking bias: std of errors {np.std(errors):.4f}"


@skipno_gpu
def test_radial_symmetry_small_blob_3px():
    """3px blob with 4x upsampling should still give reasonable accuracy."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 1.4, 1.6
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=0.8, shape=(3, 3))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, residuals = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0
    )
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.2, f"3px blob error {err:.4f} px exceeds 0.2 px"


@skipno_gpu
def test_radial_symmetry_batch():
    """Batch of 10 blobs should all be localized correctly."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    rng = np.random.RandomState(42)
    N = 10
    size = 11
    rois = []
    true_centers = []
    for _ in range(N):
        cx = size / 2 + rng.uniform(-1, 1)
        cy = size / 2 + rng.uniform(-1, 1)
        blob = _make_gaussian_blob(cx, cy, sigma=1.2, shape=(size, size))
        rois.append(blob)
        true_centers.append([cx, cy])
    rois = np.stack(rois)
    masks = np.ones_like(rois, dtype=bool)
    true_centers = np.array(true_centers)
    centers, residuals = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=4, device=0
    )
    assert centers.shape == (N, 2)
    assert residuals.shape == (N,)
    errors = np.sqrt(np.sum((centers - true_centers)**2, axis=1))
    assert np.all(errors < 0.1), f"Max batch error: {errors.max():.4f}"


@skipno_gpu
def test_radial_symmetry_voronoi_mask():
    """Masked ROI (simulating Voronoi boundary) should still give good center."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 7.3
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    mask = np.ones((15, 15), dtype=bool)
    mask[:, 12:] = False  # right side masked (simulating Voronoi crop)
    rois = np.stack([blob])
    masks = np.stack([mask])
    centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.15, f"Masked ROI error {err:.4f} px"


@skipno_gpu
def test_radial_symmetry_no_upsample():
    """upsample_factor=1 should still work (no upsampling)."""
    from subpx._gpu.centers import _radial_symmetry_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, _ = _radial_symmetry_batch_gpu(
        rois, masks, upsample_factor=1, device=0
    )
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.1, f"No-upsample error {err:.4f} px"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: FAIL — `ImportError: cannot import name '_radial_symmetry_batch_gpu'`

- [ ] **Step 3: Implement `_radial_symmetry_batch_gpu` in `subpx/_gpu/centers.py`**

Add the following function to `subpx/_gpu/centers.py` (before `detect_centers_gpu`):

```python
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
    rois : (N, H, W) float64 array
        Stacked ROIs. Out-of-mask pixels should already be background-filled.
    masks : (N, H, W) bool array
        Voronoi cell masks (True = valid pixel).
    upsample_factor : int
        Bicubic upsampling factor (1 = disabled).
    boundary_margin : int or None
        Gradient margin in upsampled pixels near Voronoi boundary.
        Defaults to upsample_factor.
    device : int
        GPU device.

    Returns
    -------
    centers : (N, 2) float64 array
        Centers in ROI pixel coordinates [x, y].
    residuals : (N,) float64 array
        Goodness-of-fit metric per blob.
    """
    _require_cupy()
    if boundary_margin is None:
        boundary_margin = upsample_factor

    N = rois.shape[0]
    # Track per-ROI original dimensions for correct coordinate transform
    orig_sizes = []  # list of (H_i, W_i) per ROI

    with gpu_device(device):
        # --- Upsample ---
        if upsample_factor > 1:
            import cupyx.scipy.ndimage as cndi_loc
            rois_up_list = []
            masks_up_list = []
            for i in range(N):
                h_i, w_i = rois[i].shape[-2], rois[i].shape[-1]
                # Find actual (non-padded) extent: last row/col with mask True
                m_i = masks[i]
                rows_valid = np.any(m_i, axis=1)
                cols_valid = np.any(m_i, axis=0)
                if rows_valid.any() and cols_valid.any():
                    h_eff = int(np.max(np.where(rows_valid))) + 1
                    w_eff = int(np.max(np.where(cols_valid))) + 1
                else:
                    h_eff, w_eff = h_i, w_i
                orig_sizes.append((h_eff, w_eff))

                roi_gpu = cp.asarray(rois[i, :h_eff, :w_eff])
                roi_up = cndi_loc.zoom(roi_gpu, upsample_factor, order=3)
                rois_up_list.append(roi_up)
                # Upsample mask with nearest-neighbor (order=0) to preserve sharp boundary
                mask_gpu = cp.asarray(masks[i, :h_eff, :w_eff].astype(np.float64))
                mask_up = cndi_loc.zoom(mask_gpu, upsample_factor, order=0) > 0.5
                masks_up_list.append(mask_up)

            # Pad to uniform upsampled size
            H_up = max(r.shape[0] for r in rois_up_list)
            W_up = max(r.shape[1] for r in rois_up_list)
            g_batch = cp.zeros((N, H_up, W_up), dtype=cp.float64)
            m_batch = cp.zeros((N, H_up, W_up), dtype=cp.bool_)
            up_sizes = []  # actual upsampled sizes per ROI
            for i in range(N):
                h, w = rois_up_list[i].shape
                g_batch[i, :h, :w] = rois_up_list[i]
                m_batch[i, :h, :w] = masks_up_list[i]
                up_sizes.append((h, w))
        else:
            g_batch = cp.asarray(rois, dtype=cp.float64)
            m_batch = cp.asarray(masks)
            H_up, W_up = rois.shape[1], rois.shape[2]
            for i in range(N):
                orig_sizes.append((rois.shape[1], rois.shape[2]))
            up_sizes = [(H_up, W_up)] * N

        # --- Erode mask by boundary_margin for gradient masking ---
        if boundary_margin > 0 and upsample_factor > 1:
            import cupyx.scipy.ndimage as cndi_loc
            struct = cp.ones((1, 3, 3), dtype=cp.bool_)
            m_eroded = m_batch.copy()
            for _ in range(boundary_margin):
                m_eroded = cndi_loc.binary_erosion(m_eroded, structure=struct)
        else:
            m_eroded = m_batch

        # --- Diagonal gradients at midpoints ---
        # dIdu[i,j] = I[i, j+1] - I[i+1, j]
        dIdu = g_batch[:, :-1, 1:] - g_batch[:, 1:, :-1]   # (N, H_up-1, W_up-1)
        dIdv = g_batch[:, :-1, :-1] - g_batch[:, 1:, 1:]   # (N, H_up-1, W_up-1)

        grad_mag2 = dIdu * dIdu + dIdv * dIdv  # (N, H_up-1, W_up-1)

        # --- Gradient validity mask at midpoint grid ---
        # A midpoint is valid only if all 4 surrounding pixels are inside the eroded mask
        mid_mask = (
            m_eroded[:, :-1, :-1] &
            m_eroded[:, :-1, 1:] &
            m_eroded[:, 1:, :-1] &
            m_eroded[:, 1:, 1:]
        )

        # --- Slope m = -(dIdv + dIdu) / (dIdu - dIdv) ---
        denom_slope = dIdu - dIdv
        eps = 1e-12
        near_zero_denom = cp.abs(denom_slope) < eps
        near_zero_grad = grad_mag2 < eps * eps

        m_slope = cp.where(
            near_zero_denom,
            cp.sign(-(dIdv + dIdu)) * 1e9,
            -(dIdv + dIdu) / denom_slope,
        )

        # Weight = 0 for zero-gradient midpoints or outside mask
        valid = mid_mask & (~near_zero_grad)

        # --- Midpoint coordinates (centered on ROI) ---
        Hm = dIdu.shape[1]  # H_up - 1
        Wm = dIdu.shape[2]  # W_up - 1
        xm = cp.arange(Wm, dtype=cp.float64) - (Wm - 1) / 2.0  # centered
        ym = cp.arange(Hm, dtype=cp.float64) - (Hm - 1) / 2.0
        xm_2d = xm[None, None, :]  # (1, 1, Wm)
        ym_2d = ym[None, :, None]  # (1, Hm, 1)

        # --- Line intercepts b = ym - m * xm ---
        b_val = ym_2d - m_slope * xm_2d  # (N, Hm, Wm)

        # --- Weights: |grad|^2 / dist_to_gradient_centroid ---
        gm2_valid = cp.where(valid, grad_mag2, 0.0)
        sum_gm2 = gm2_valid.sum(axis=(1, 2), keepdims=True)  # (N, 1, 1)
        sum_gm2 = cp.maximum(sum_gm2, eps)

        x_gc = (gm2_valid * xm_2d).sum(axis=(1, 2), keepdims=True) / sum_gm2
        y_gc = (gm2_valid * ym_2d).sum(axis=(1, 2), keepdims=True) / sum_gm2

        dist_to_gc = cp.sqrt((xm_2d - x_gc)**2 + (ym_2d - y_gc)**2)
        dist_to_gc = cp.maximum(dist_to_gc, eps)

        w = cp.where(valid, gm2_valid / dist_to_gc, 0.0)

        # --- Analytic least-squares solve ---
        m2p1 = m_slope * m_slope + 1.0
        wp = w / m2p1  # w' = w / (m^2 + 1)

        sw   = wp.sum(axis=(1, 2))                          # (N,)
        smmw = (m_slope * m_slope * wp).sum(axis=(1, 2))    # (N,)
        smw  = (m_slope * wp).sum(axis=(1, 2))              # (N,)
        smbw = (m_slope * b_val * wp).sum(axis=(1, 2))      # (N,)
        sbw  = (b_val * wp).sum(axis=(1, 2))                # (N,)

        det = smw * smw - smmw * sw
        det = cp.where(cp.abs(det) < eps, cp.full_like(det, cp.nan), det)

        xc = (smbw * sw - smw * sbw) / det
        yc = (smbw * smw - smmw * sbw) / det

        # --- Goodness-of-fit residual ---
        d2 = (b_val - (yc[:, None, None] - m_slope * xc[:, None, None]))**2 / m2p1
        residual = (d2 * gm2_valid).sum(axis=(1, 2)) / sum_gm2.squeeze()

        # --- Per-ROI coordinate transform: upsampled ROI-centered → ROI pixel coords ---
        # zoom with grid_mode=False (default): center-aligned
        # output pixel i maps to input pixel i * (N_in - 1) / (N_out - 1)
        # IMPORTANT: Each ROI may have different original/upsampled size due to padding.
        # Pre-transfer to CPU to avoid per-ROI .get() calls.
        xc_cpu = cp.asnumpy(xc)
        yc_cpu = cp.asnumpy(yc)
        xc_orig = np.empty(N, dtype=np.float64)
        yc_orig = np.empty(N, dtype=np.float64)
        for i in range(N):
            h_orig_i, w_orig_i = orig_sizes[i]
            h_up_i, w_up_i = up_sizes[i]
            if upsample_factor > 1:
                xc_pix_up = xc_cpu[i] + (w_up_i - 1) / 2.0
                yc_pix_up = yc_cpu[i] + (h_up_i - 1) / 2.0
                scale_x = (w_orig_i - 1) / max(w_up_i - 1, 1)
                scale_y = (h_orig_i - 1) / max(h_up_i - 1, 1)
                xc_orig[i] = xc_pix_up * scale_x
                yc_orig[i] = yc_pix_up * scale_y
            else:
                xc_orig[i] = xc_cpu[i] + (w_orig_i - 1) / 2.0
                yc_orig[i] = yc_cpu[i] + (h_orig_i - 1) / 2.0

        centers_out = np.stack([xc_orig, yc_orig], axis=1)  # (N, 2)
        residual_cpu = cp.asnumpy(residual)
        return centers_out, residual_cpu
```

Also add `from .voronoi import compute_voronoi_labels_gpu` at the top of `_gpu/centers.py` (inside the cupy try/except block or at module level with a guard).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run existing tests to verify no regression**

Run: `pytest tests/test_smoke.py -v`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_radial_symmetry.py
git commit -m "feat: add Parthasarathy radial symmetry batch GPU kernel"
```

---

### Task 3: Isophote Curvature Center — Core GPU Kernel

**Files:**
- Modify: `subpx/_gpu/centers.py`
- Create: `tests/test_isophote_curvature.py`

- [ ] **Step 1: Write failing tests for isophote curvature**

```python
# tests/test_isophote_curvature.py
import numpy as np
import pytest


def _has_cupy():
    try:
        import cupy
        return True
    except Exception:
        return False


skipno_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


def _make_gaussian_blob(cx, cy, sigma, shape):
    H, W = shape
    yy, xx = np.mgrid[:H, :W]
    return np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2)).astype(np.float64)


@skipno_gpu
def test_isophote_curvature_isolated_blob():
    """Single Gaussian blob: isophote center should agree with true center."""
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    centers, spreads = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0
    )
    err = np.sqrt((centers[0, 0] - true_cx)**2 + (centers[0, 1] - true_cy)**2)
    assert err < 0.15, f"Isophote center error {err:.4f} px"


@skipno_gpu
def test_isophote_curvature_spread_low_for_symmetric():
    """Spread should be low for a perfectly symmetric blob."""
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    blob = _make_gaussian_blob(7.0, 7.0, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    _, spreads = _isophote_curvature_batch_gpu(
        rois, masks, upsample_factor=4, device=0
    )
    assert spreads[0] < 1.0, f"Spread {spreads[0]:.4f} too high for symmetric blob"


@skipno_gpu
def test_isophote_agrees_with_radial_symmetry():
    """Isophote and radial symmetry should agree for isolated symmetric blob."""
    from subpx._gpu.centers import (
        _radial_symmetry_batch_gpu,
        _isophote_curvature_batch_gpu,
    )
    true_cx, true_cy = 7.3, 6.7
    blob = _make_gaussian_blob(true_cx, true_cy, sigma=1.5, shape=(15, 15))
    rois = np.stack([blob])
    masks = np.ones_like(rois, dtype=bool)
    rs_centers, _ = _radial_symmetry_batch_gpu(rois, masks, upsample_factor=4, device=0)
    ic_centers, _ = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    dist = np.sqrt(np.sum((rs_centers - ic_centers)**2))
    assert dist < 0.15, f"Methods disagree by {dist:.4f} px"


@skipno_gpu
def test_isophote_curvature_batch():
    """Batch of blobs should all produce valid results."""
    from subpx._gpu.centers import _isophote_curvature_batch_gpu
    rng = np.random.RandomState(42)
    N = 5
    rois = np.stack([
        _make_gaussian_blob(7 + rng.uniform(-1, 1), 7 + rng.uniform(-1, 1), 1.2, (15, 15))
        for _ in range(N)
    ])
    masks = np.ones_like(rois, dtype=bool)
    centers, spreads = _isophote_curvature_batch_gpu(rois, masks, upsample_factor=4, device=0)
    assert centers.shape == (N, 2)
    assert spreads.shape == (N,)
    assert np.all(np.isfinite(centers))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_isophote_curvature.py -v`
Expected: FAIL — `ImportError: cannot import name '_isophote_curvature_batch_gpu'`

- [ ] **Step 3: Implement `_isophote_curvature_batch_gpu` in `subpx/_gpu/centers.py`**

```python
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

    Parameters
    ----------
    rois : (N, H, W) float64 array
        Stacked ROIs. Out-of-mask pixels should already be background-filled.
    masks : (N, H, W) bool array
        Voronoi cell masks (True = valid pixel).
    upsample_factor : int
        Bicubic upsampling factor (1 = disabled).
    boundary_margin : int or None
        Gradient margin in upsampled pixels. Defaults to upsample_factor.
    pre_smooth_sigma : float
        Gaussian sigma for pre-smoothing before second derivatives.
    device : int
        GPU device.

    Returns
    -------
    centers : (N, 2) float64 array — centers in ROI pixel coords [x, y].
    spreads : (N,) float64 array — curvature-consistency spread per blob.
    """
    _require_cupy()
    if boundary_margin is None:
        boundary_margin = upsample_factor

    N = rois.shape[0]
    orig_sizes = []

    with gpu_device(device):
        # --- Upsample (same logic as radial symmetry) ---
        if upsample_factor > 1:
            import cupyx.scipy.ndimage as cndi_loc
            rois_up_list = []
            masks_up_list = []
            for i in range(N):
                m_i = masks[i]
                rows_valid = np.any(m_i, axis=1)
                cols_valid = np.any(m_i, axis=0)
                if rows_valid.any() and cols_valid.any():
                    h_eff = int(np.max(np.where(rows_valid))) + 1
                    w_eff = int(np.max(np.where(cols_valid))) + 1
                else:
                    h_eff, w_eff = rois[i].shape[-2], rois[i].shape[-1]
                orig_sizes.append((h_eff, w_eff))
                roi_gpu = cp.asarray(rois[i, :h_eff, :w_eff])
                roi_up = cndi_loc.zoom(roi_gpu, upsample_factor, order=3)
                rois_up_list.append(roi_up)
                mask_gpu = cp.asarray(masks[i, :h_eff, :w_eff].astype(np.float64))
                mask_up = cndi_loc.zoom(mask_gpu, upsample_factor, order=0) > 0.5
                masks_up_list.append(mask_up)

            H_up = max(r.shape[0] for r in rois_up_list)
            W_up = max(r.shape[1] for r in rois_up_list)
            g_batch = cp.zeros((N, H_up, W_up), dtype=cp.float64)
            m_batch = cp.zeros((N, H_up, W_up), dtype=cp.bool_)
            up_sizes = []
            for i in range(N):
                h, w = rois_up_list[i].shape
                g_batch[i, :h, :w] = rois_up_list[i]
                m_batch[i, :h, :w] = masks_up_list[i]
                up_sizes.append((h, w))
        else:
            g_batch = cp.asarray(rois, dtype=cp.float64)
            m_batch = cp.asarray(masks)
            H_up, W_up = rois.shape[1], rois.shape[2]
            for i in range(N):
                orig_sizes.append((rois.shape[1], rois.shape[2]))
            up_sizes = [(H_up, W_up)] * N

        # --- Optional Gaussian pre-smoothing for second derivatives ---
        if pre_smooth_sigma > 0:
            import cupyx.scipy.ndimage as cndi_loc
            for i in range(N):
                g_batch[i] = cndi_loc.gaussian_filter(g_batch[i], sigma=pre_smooth_sigma)

        # --- Erode mask by boundary_margin ---
        if boundary_margin > 0 and upsample_factor > 1:
            import cupyx.scipy.ndimage as cndi_loc
            struct = cp.ones((1, 3, 3), dtype=cp.bool_)
            m_eroded = m_batch.copy()
            for _ in range(boundary_margin):
                m_eroded = cndi_loc.binary_erosion(m_eroded, structure=struct)
        else:
            m_eroded = m_batch

        eps = 1e-12

        # --- First derivatives via central finite differences ---
        # Interior pixels only (1:-1 in both dims)
        Ix = (g_batch[:, 1:-1, 2:] - g_batch[:, 1:-1, :-2]) / 2.0  # (N, H-2, W-2)
        Iy = (g_batch[:, 2:, 1:-1] - g_batch[:, :-2, 1:-1]) / 2.0

        # --- Second derivatives ---
        Ixx = g_batch[:, 1:-1, 2:] - 2*g_batch[:, 1:-1, 1:-1] + g_batch[:, 1:-1, :-2]
        Iyy = g_batch[:, 2:, 1:-1] - 2*g_batch[:, 1:-1, 1:-1] + g_batch[:, :-2, 1:-1]
        Ixy = (g_batch[:, 2:, 2:] - g_batch[:, 2:, :-2]
               - g_batch[:, :-2, 2:] + g_batch[:, :-2, :-2]) / 4.0

        # --- Validity mask (interior pixels of eroded mask) ---
        valid = m_eroded[:, 1:-1, 1:-1]

        # --- Curvature denominator ---
        denom = Iy**2 * Ixx - 2*Ix*Ixy*Iy + Ix**2 * Iyy
        grad_sq = Ix**2 + Iy**2

        # Skip flat regions and inflection points
        valid = valid & (cp.abs(denom) > eps) & (grad_sq > eps)

        # --- Displacement to curvature center ---
        Dx = cp.where(valid, -Ix * grad_sq / denom, 0.0)
        Dy = cp.where(valid, -Iy * grad_sq / denom, 0.0)

        # --- Curvedness weight ---
        curvedness = cp.sqrt(Ixx**2 + 2*Ixy**2 + Iyy**2)

        # --- Pixel coordinates (centered on ROI) ---
        Hd, Wd = Ix.shape[1], Ix.shape[2]  # derivative grid size
        xp = cp.arange(Wd, dtype=cp.float64) - (Wd - 1) / 2.0
        yp = cp.arange(Hd, dtype=cp.float64) - (Hd - 1) / 2.0
        xp_2d = xp[None, None, :]  # (1, 1, Wd)
        yp_2d = yp[None, :, None]  # (1, Hd, 1)

        # --- Voted center from each pixel ---
        vote_x = xp_2d + Dx
        vote_y = yp_2d + Dy

        # --- Outlier rejection: displacement > ROI_size / 2 ---
        disp_mag = cp.sqrt(Dx**2 + Dy**2)
        roi_half = max(H_up, W_up) / 2.0
        outlier = disp_mag > roi_half
        valid = valid & (~outlier)

        w = cp.where(valid, curvedness, 0.0)

        # --- Weighted average of votes ---
        sum_w = w.sum(axis=(1, 2))  # (N,)
        sum_w = cp.maximum(sum_w, eps)
        xc = (w * vote_x).sum(axis=(1, 2)) / sum_w
        yc = (w * vote_y).sum(axis=(1, 2)) / sum_w

        # --- Curvature-consistency spread ---
        spread = cp.sqrt(
            (w * ((vote_x - xc[:, None, None])**2 + (vote_y - yc[:, None, None])**2)).sum(axis=(1, 2)) / sum_w
        )

        # --- Per-ROI coordinate transform ---
        # Pre-transfer to CPU to avoid per-ROI .get() calls.
        xc_cpu = cp.asnumpy(xc)
        yc_cpu = cp.asnumpy(yc)
        spread_cpu = cp.asnumpy(spread)
        xc_orig = np.empty(N, dtype=np.float64)
        yc_orig = np.empty(N, dtype=np.float64)
        for i in range(N):
            h_orig_i, w_orig_i = orig_sizes[i]
            h_up_i, w_up_i = up_sizes[i]
            # Derivative grid (central differences) is 2 pixels smaller than upsampled grid.
            # Centered coords in derivative grid map to upsampled pixel = coord + (W_up-1)/2.
            # This is because: derivative grid center = (Wd-1)/2 where Wd = W_up-2,
            # and the +1 offset from border trimming cancels out:
            # upsampled_pixel = coord + (Wd-1)/2 + 1 = coord + (W_up-2-1)/2 + 1 = coord + (W_up-1)/2
            if upsample_factor > 1:
                xc_pix_up = xc_cpu[i] + (w_up_i - 1) / 2.0
                yc_pix_up = yc_cpu[i] + (h_up_i - 1) / 2.0
                scale_x = (w_orig_i - 1) / max(w_up_i - 1, 1)
                scale_y = (h_orig_i - 1) / max(h_up_i - 1, 1)
                xc_orig[i] = xc_pix_up * scale_x
                yc_orig[i] = yc_pix_up * scale_y
            else:
                xc_orig[i] = xc_cpu[i] + (w_orig_i - 1) / 2.0
                yc_orig[i] = yc_cpu[i] + (h_orig_i - 1) / 2.0

        centers_out = np.stack([xc_orig, yc_orig], axis=1)
        return centers_out, spread_cpu
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_isophote_curvature.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (no regression)

- [ ] **Step 6: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_isophote_curvature.py
git commit -m "feat: add isophote curvature center diagnostic GPU kernel"
```

---

### Task 4: Integrate into `detect_centers` Public API

**Files:**
- Modify: `subpx/centers.py` (lines 130-181 — `detect_centers` function, lines 184-273 — `detect_centers_tiled`)
- Modify: `subpx/_gpu/centers.py` (lines 970-1101 — `detect_centers_gpu` function)

- [ ] **Step 1: Write failing integration tests**

```python
# Add to tests/test_radial_symmetry.py

@skipno_gpu
def test_detect_centers_radial_symmetry_integration():
    """Full pipeline: detect_centers with refine='radial_symmetry'."""
    from subpx.centers import detect_centers
    img = np.zeros((64, 64), dtype=np.uint8)
    # Two small blobs
    for cy, cx in [(15, 15), (15, 45)]:
        yy, xx = np.mgrid[:64, :64]
        img += (200 * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2*2.0**2))).astype(np.uint8)
    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=200, upsample_factor=4,
    )
    assert res.centers_xy.shape[0] == 2
    assert "radial_symmetry_residual" in res.meta
    assert "voronoi_cell_area" in res.meta
    assert res.meta["upsample_factor"] == 4


def test_detect_centers_radial_symmetry_cpu_raises():
    """CPU backend must raise NotImplementedError (no GPU needed for this test)."""
    from subpx.centers import detect_centers
    img = np.zeros((32, 32), dtype=np.uint8)
    img[10:14, 10:14] = 200
    with pytest.raises(NotImplementedError):
        detect_centers(img, backend="cpu", refine="radial_symmetry")


def test_detect_centers_tiled_radial_symmetry_raises():
    """Tiled detection must raise NotImplementedError for new methods (no GPU needed)."""
    from subpx.centers import detect_centers_tiled
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    with pytest.raises(NotImplementedError):
        detect_centers_tiled(img, backend="gpu", refine="radial_symmetry")


def test_detect_centers_tiled_global_otsu_radial_symmetry_raises():
    """tiled_global_otsu must also raise NotImplementedError."""
    from subpx.centers import detect_centers_tiled_global_otsu
    img = np.zeros((64, 64), dtype=np.uint8)
    img[10:14, 10:14] = 200
    with pytest.raises(NotImplementedError):
        detect_centers_tiled_global_otsu(img, backend="gpu", refine="radial_symmetry")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_radial_symmetry.py::test_detect_centers_radial_symmetry_integration -v`
Expected: FAIL

- [ ] **Step 3: Modify `subpx/centers.py`**

Changes to `detect_centers()` (around line 130):
1. Add `upsample_factor: int = 4` parameter to signature
2. Add early guard for CPU backend + new methods:
   ```python
   _VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}
   if b == "cpu" and refine in _VORONOI_METHODS:
       raise NotImplementedError(
           f"refine='{refine}' requires GPU backend. Set backend='gpu' or install CuPy."
       )
   ```
3. Pass `upsample_factor=upsample_factor` to `detect_centers_gpu()` call

Changes to `detect_centers_tiled()` (around line 184):
1. Add guard after `refine = kwargs.get("refine", ...)`:
   ```python
   if kwargs.get("refine") in _VORONOI_METHODS:
       raise NotImplementedError(
           "Voronoi-partitioned methods are not supported with tiled detection."
       )
   ```

Add same guard at the top of `detect_centers_tiled_global_otsu()` (around line 352, after `img.ndim != 2` check):
   ```python
   _VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}
   if kwargs.get("refine") in _VORONOI_METHODS:
       raise NotImplementedError(
           "Voronoi-partitioned methods are not supported with tiled detection."
       )
   ```

- [ ] **Step 4: Modify `subpx/_gpu/centers.py` — `detect_centers_gpu`**

Add `upsample_factor: int = 4` to `detect_centers_gpu()` signature (line 970).

**CRITICAL: The Voronoi dispatch must go BEFORE the `method_rows` loop** (line 1003), because the existing loop validates `method_local not in method_rows` and would raise `ValueError` for the new methods. Insert the following block immediately after the component extraction (line 1001, after `num = comp.num_labels`):

```python
    # ── Voronoi-partitioned methods: early dispatch ──────────────────────
    # Must branch BEFORE method_rows loop which rejects unknown method names.
    _VORONOI_METHODS = {"radial_symmetry", "isophote_curvature"}
    if refine in _VORONOI_METHODS:
        # Collect component stats (same filtering as method_rows path, without method validation)
        rows = []
        for lab in range(1, num):
            x, y, w, h, area = stats[lab]
            if area < int(area_min) or area > int(area_max):
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
        # rows are [x, y, w, h, cx, cy] where cx=column (x), cy=row (y)
        coarse_centers = rows_arr[:, 4:6]  # already [x, y] order

        # 1. Voronoi partition
        from .voronoi import compute_voronoi_labels_gpu
        voronoi_labels = compute_voronoi_labels_gpu(
            coarse_centers,
            g.shape,
            device=device,
        )
        cell_areas = np.bincount(voronoi_labels.ravel(), minlength=len(rows))

        # 2. Extract Voronoi-masked ROIs with border-pixel background fill
        rois_list = []
        masks_list = []
        H, W = g.shape
        for j, (x, y, w, h, cx, cy) in enumerate(rows):
            x0 = max(0, int(x) - int(pad))
            y0 = max(0, int(y) - int(pad))
            x1 = min(W, int(x + w) + int(pad))
            y1 = min(H, int(y + h) + int(pad))
            roi = g[y0:y1, x0:x1].astype(np.float64)
            vmask = voronoi_labels[y0:y1, x0:x1] == j
            # Background-fill: use median of cell-interior BORDER pixels
            # (pixels inside mask but adjacent to outside-mask pixels)
            # NumPy-only binary erosion (no scipy dependency)
            inner = vmask.copy()
            inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
            inner[1:, :] &= vmask[:-1, :]
            inner[:-1, :] &= vmask[1:, :]
            inner[:, 1:] &= vmask[:, :-1]
            inner[:, :-1] &= vmask[:, 1:]
            border_mask = vmask & ~inner  # pixels at edge of Voronoi cell interior
            border_vals = roi[border_mask]
            bg = float(np.median(border_vals)) if border_vals.size > 0 else 0.0
            roi_filled = np.where(vmask, roi, bg)
            rois_list.append(roi_filled)
            masks_list.append(vmask)

        # 3. Pad to uniform size and stack
        hs = [r.shape[0] for r in rois_list]
        ws = [r.shape[1] for r in rois_list]
        Hm, Wm = max(hs), max(ws)
        rois_stack = np.zeros((len(rois_list), Hm, Wm), dtype=np.float64)
        masks_stack = np.zeros((len(rois_list), Hm, Wm), dtype=bool)
        origins = []
        for j, (roi, vmask) in enumerate(zip(rois_list, masks_list)):
            h, w = roi.shape
            rois_stack[j, :h, :w] = roi
            masks_stack[j, :h, :w] = vmask
            x0 = max(0, int(rows[j][0]) - int(pad))
            y0 = max(0, int(rows[j][1]) - int(pad))
            origins.append((x0, y0))

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
        centers = []
        quality_kept = []
        areas_kept = []
        for j, (lc, q) in enumerate(zip(local_centers, quality)):
            x0, y0 = origins[j]
            gx = x0 + float(lc[0])
            gy = y0 + float(lc[1])
            if np.isfinite(gx) and np.isfinite(gy):
                centers.append([gx, gy])
                quality_kept.append(float(q))
                areas_kept.append(int(cell_areas[j]) if j < len(cell_areas) else 0)

        arr = np.asarray(centers, dtype=np.float64) if centers else np.zeros((0, 2), dtype=np.float64)
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

    # ── Existing method dispatch (unchanged from here) ──────────────────
    # The existing method_rows loop (lines 1003-1101) continues unmodified.
```

**No `else` wrapping needed** — the Voronoi path returns early. The existing `method_rows` loop runs only for non-Voronoi methods.

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/test_radial_symmetry.py -v`
Expected: All tests PASS including the integration tests

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add subpx/centers.py subpx/_gpu/centers.py tests/test_radial_symmetry.py
git commit -m "feat: integrate radial symmetry and isophote curvature into detect_centers API"
```

---

### Task 5: Voronoi Visualization

**Files:**
- Create: `subpx/visualization.py`
- Create: `tests/test_visualization.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_visualization.py
import numpy as np
import pytest


def test_draw_voronoi_boundaries_basic():
    """Should draw boundaries where labels change between 4-neighbors."""
    from subpx.visualization import draw_voronoi_boundaries
    img = np.zeros((20, 20), dtype=np.float64)
    centers = np.array([[5.0, 10.0], [15.0, 10.0]])
    overlay = draw_voronoi_boundaries(img, centers, color=1.0)
    assert overlay.shape == (20, 20)
    # Boundary should exist somewhere near the midline (col ~10)
    boundary_cols = np.where(overlay[10, :] > 0)[0]
    assert len(boundary_cols) > 0
    assert np.abs(np.mean(boundary_cols) - 10) < 2


def test_draw_voronoi_boundaries_with_precomputed():
    """Should use pre-computed label_map when provided."""
    from subpx.visualization import draw_voronoi_boundaries
    img = np.zeros((10, 10), dtype=np.float64)
    label_map = np.zeros((10, 10), dtype=np.int32)
    label_map[:, 5:] = 1
    centers = np.array([[2.5, 5.0], [7.5, 5.0]])
    overlay = draw_voronoi_boundaries(img, centers, label_map=label_map, color=1.0)
    # Boundary at col=5 (where label changes)
    assert overlay[5, 4] > 0 or overlay[5, 5] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_visualization.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `draw_voronoi_boundaries`**

Create `subpx/visualization.py`:

```python
"""Visualization utilities for subpx."""
from __future__ import annotations

import numpy as np


def draw_voronoi_boundaries(
    image: np.ndarray,
    centers_xy: np.ndarray,
    label_map: np.ndarray | None = None,
    color: float = 1.0,
    thickness: int = 1,
    backend: str = "auto",
) -> np.ndarray:
    """Overlay Voronoi partition boundaries on an image.

    Boundaries are pixels where the Voronoi label differs from at least
    one 4-connected neighbor.

    Parameters
    ----------
    image : (H, W) ndarray
        Background image to overlay on.
    centers_xy : (N, 2) ndarray
        Seed center positions [x, y].
    label_map : (H, W) ndarray, optional
        Pre-computed Voronoi label map. Computed from centers_xy if None.
    color : float
        Intensity value for boundary pixels.
    thickness : int
        Boundary line width (morphological dilation iterations).
    backend : str
        "auto", "cpu", or "gpu". Only used for Voronoi computation if
        label_map is not provided.

    Returns
    -------
    overlay : (H, W) ndarray
        Copy of image with Voronoi boundaries drawn.
    """
    img = np.asarray(image)
    overlay = img.copy().astype(np.float64)
    H, W = overlay.shape[:2]

    if label_map is None:
        from .backends import resolve_backend
        b = resolve_backend(backend)
        if b == "gpu":
            from ._gpu.voronoi import compute_voronoi_labels_gpu
            label_map = compute_voronoi_labels_gpu(centers_xy, (H, W))
        else:
            # CPU fallback: per-pixel nearest-seed (tiled to avoid O(H*W*N) memory)
            centers = np.asarray(centers_xy, dtype=np.float64)
            N_seeds = centers.shape[0]
            label_map = np.zeros((H, W), dtype=np.int32)
            CHUNK = 256  # process rows in chunks to limit memory
            for r0 in range(0, H, CHUNK):
                r1 = min(H, r0 + CHUNK)
                yy, xx = np.mgrid[r0:r1, :W]
                coords = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (chunk*W, 2)
                # For small N, broadcast is fine per-chunk
                dists = np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2)
                label_map[r0:r1] = np.argmin(dists, axis=1).reshape(r1 - r0, W).astype(np.int32)

    labels = np.asarray(label_map, dtype=np.int32)

    # Find boundary pixels: where label differs from any 4-connected neighbor
    boundary = np.zeros((H, W), dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :]  |= labels[1:, :]  != labels[:-1, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:]  |= labels[:, 1:]  != labels[:, :-1]

    # Thicken boundary if requested
    if thickness > 1:
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(boundary, iterations=thickness - 1)

    overlay[boundary] = color
    return overlay
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_visualization.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add subpx/visualization.py tests/test_visualization.py
git commit -m "feat: add Voronoi boundary visualization"
```

---

### Task 6: Neighbor Contamination Regression Test

**Files:**
- Modify: `tests/test_radial_symmetry.py`

This is the critical test: verify that radial symmetry + Voronoi partitioning actually solves the contamination problem that motivated the whole design.

- [ ] **Step 1: Write the contamination test**

```python
# Add to tests/test_radial_symmetry.py

@skipno_gpu
def test_radial_symmetry_no_neighbor_bias():
    """Closely-packed blobs: radial symmetry should not have neighbor-induced bias.

    Create two Gaussian blobs separated by only 2*sigma. Without Voronoi
    partitioning, a centroid method would be pulled toward the neighbor.
    With Voronoi partitioning + radial symmetry, the center should be unbiased.
    """
    from subpx.centers import detect_centers
    sigma = 2.0
    sep = 2.5 * sigma  # 5 px separation — closely packed
    cx1, cy1 = 20.0, 20.0
    cx2, cy2 = 20.0 + sep, 20.0

    img = np.zeros((40, 50), dtype=np.float64)
    yy, xx = np.mgrid[:40, :50]
    img += 200 * np.exp(-((xx - cx1)**2 + (yy - cy1)**2) / (2 * sigma**2))
    img += 200 * np.exp(-((xx - cx2)**2 + (yy - cy2)**2) / (2 * sigma**2))
    img = np.clip(img, 0, 255).astype(np.uint8)

    res = detect_centers(
        img, backend="gpu", refine="radial_symmetry",
        area_min=4, area_max=500, upsample_factor=4,
    )
    assert res.centers_xy.shape[0] == 2

    # Sort by x to identify blob 1 and blob 2
    pts = res.centers_xy[np.argsort(res.centers_xy[:, 0])]

    err1 = np.sqrt((pts[0, 0] - cx1)**2 + (pts[0, 1] - cy1)**2)
    err2 = np.sqrt((pts[1, 0] - cx2)**2 + (pts[1, 1] - cy2)**2)

    # The key assertion: no systematic bias toward the neighbor
    # Both errors should be small (< 0.3 px) despite close packing
    assert err1 < 0.3, f"Blob 1 error {err1:.4f} px — possible neighbor bias"
    assert err2 < 0.3, f"Blob 2 error {err2:.4f} px — possible neighbor bias"
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_radial_symmetry.py::test_radial_symmetry_no_neighbor_bias -v`
Expected: PASS (validates the core design goal)

- [ ] **Step 3: Commit**

```bash
git add tests/test_radial_symmetry.py
git commit -m "test: add neighbor contamination regression test for radial symmetry"
```

---

### Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from subpx.centers import detect_centers; from subpx.visualization import draw_voronoi_boundaries; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Verify GPU methods raise on CPU**

Run: `python -c "from subpx.centers import detect_centers; import numpy as np; detect_centers(np.zeros((32,32), dtype='uint8'), backend='cpu', refine='radial_symmetry')"`
Expected: `NotImplementedError`

- [ ] **Step 4: Final commit with any cleanup**

```bash
git status
# Only commit if there are changes. Stage specific files:
git add subpx/_gpu/voronoi.py subpx/_gpu/centers.py subpx/centers.py subpx/visualization.py tests/test_voronoi.py tests/test_radial_symmetry.py tests/test_isophote_curvature.py tests/test_visualization.py
git commit -m "chore: final cleanup for radial symmetry and isophote curvature"
```
