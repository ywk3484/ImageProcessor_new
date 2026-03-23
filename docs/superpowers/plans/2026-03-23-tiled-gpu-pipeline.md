# Tiled GPU Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `_detect_centers_tiled_gpu` function that matches the reference `find_subpixel_centers_tiled_hybrid_gpu` performance and results, then wire `detect_centers_tiled` to dispatch to it for GPU backend.

**Architecture:** New internal function in `_gpu/centers.py` implements a single-pass tiled pipeline: global Otsu threshold (once), per-tile fixed threshold + GPU connected components + vectorized area filter + batched `refine_centers_edge_moment_gpu` + band-based de-dup. Public `detect_centers_tiled` dispatches to it when `backend="gpu"`.

**Tech Stack:** NumPy, CuPy, OpenCV (cv2), existing `connected_components_stats_gpu` and `refine_centers_edge_moment_gpu`.

**Spec:** `docs/superpowers/specs/2026-03-23-tiled-gpu-pipeline-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `subpx/_gpu/centers.py` | Modify (append) | Add `_detect_centers_tiled_gpu` function |
| `subpx/centers.py` | Modify (lines 184-251) | Update `detect_centers_tiled` to dispatch GPU path, add `thr_scale` param, update CPU path overlap |
| `tests/test_tiled_gpu_pipeline.py` | Create | All tests for the new tiled pipeline |

---

### Task 1: Write `_detect_centers_tiled_gpu` with tests

**Files:**
- Modify: `subpx/_gpu/centers.py` (append after line 950)
- Create: `tests/test_tiled_gpu_pipeline.py`

This is the core implementation — the entire new tiled pipeline function.

- [ ] **Step 1: Write the test file with a CPU-only smoke test**

Create `tests/test_tiled_gpu_pipeline.py`. Start with a test that can validate the basic structure even without a GPU, using a mock or by testing the CPU fallback path. Then add the GPU-specific test.

```python
"""Tests for the dedicated tiled GPU pipeline."""
import numpy as np
import pytest

from subpx.types import CenterResult


def _make_dot_grid(rows=4, cols=4, spacing=20, dot_size=3, margin=10):
    """Create a synthetic image with a grid of bright dots on black background."""
    H = 2 * margin + (rows - 1) * spacing + dot_size
    W = 2 * margin + (cols - 1) * spacing + dot_size
    img = np.zeros((H, W), dtype=np.uint8)
    expected = []
    for r in range(rows):
        for c in range(cols):
            y0 = margin + r * spacing
            x0 = margin + c * spacing
            img[y0:y0 + dot_size, x0:x0 + dot_size] = 255
            expected.append([x0 + (dot_size - 1) / 2.0, y0 + (dot_size - 1) / 2.0])
    return img, np.array(expected, dtype=np.float64)


# --- GPU tests (skip if no CuPy) ---

try:
    import cupy as _cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


@gpu
def test_tiled_gpu_pipeline_basic():
    """Basic: small image, single tile covers everything."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = _detect_centers_tiled_gpu(
        img,
        area_min=1,
        area_max=50,
        tile_h=8192,  # larger than image → single tile
        overlap=16,
        refine="edge_gradmoment",
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert result.centers_xy.shape[0] == expected.shape[0]


@gpu
def test_tiled_gpu_pipeline_multi_tile():
    """Multi-tile: image taller than tile_h, tests tiling + band de-dup."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, expected = _make_dot_grid(rows=10, cols=4, spacing=20, dot_size=3, margin=10)
    H = img.shape[0]
    tile_h = H // 3  # force ~3 tiles
    result = _detect_centers_tiled_gpu(
        img,
        area_min=1,
        area_max=50,
        tile_h=tile_h,
        overlap=30,
        refine="edge_gradmoment",
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    # Should find all dots — allow ±2 for edge effects
    assert abs(result.centers_xy.shape[0] - expected.shape[0]) <= 2


@gpu
def test_tiled_gpu_pipeline_empty_image():
    """All-black image should return zero centers."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img = np.zeros((100, 100), dtype=np.uint8)
    result = _detect_centers_tiled_gpu(img, area_min=1, area_max=50)
    assert result.centers_xy.shape == (0, 2)


@gpu
def test_tiled_gpu_pipeline_overlap_validation():
    """overlap >= tile_h should raise ValueError."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must be < tile_h"):
        _detect_centers_tiled_gpu(img, tile_h=64, overlap=64)


@gpu
@pytest.mark.parametrize("method", ["logquad", "weighted", "edge_erf", "auto"])
def test_tiled_gpu_pipeline_unimplemented_refine(method):
    """Unimplemented refine methods should raise NotImplementedError."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, _ = _make_dot_grid(rows=2, cols=2, spacing=15, dot_size=3, margin=10)
    with pytest.raises(NotImplementedError):
        _detect_centers_tiled_gpu(img, refine=method, area_min=1, area_max=50)


@gpu
def test_tiled_gpu_pipeline_unknown_refine():
    """Unknown refine method should raise ValueError."""
    from subpx._gpu.centers import _detect_centers_tiled_gpu

    img, _ = _make_dot_grid(rows=2, cols=2, spacing=15, dot_size=3, margin=10)
    with pytest.raises(ValueError, match="Unknown refine method"):
        _detect_centers_tiled_gpu(img, refine="nonexistent", area_min=1, area_max=50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tiled_gpu_pipeline.py -v`
Expected: `ImportError` or `AttributeError` — `_detect_centers_tiled_gpu` does not exist yet.

- [ ] **Step 3: Implement `_detect_centers_tiled_gpu`**

Append to `subpx/_gpu/centers.py` after the existing `detect_centers_gpu` function (after line 950):

```python
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
    band_rad: int = 4,
    edge_rad: int = 10,
    smooth_passes: int = 2,
    loc_rad: int = 3,
    grad_power: float = 8.0,
    iters: int = 2,
    pad: int = 3,
    gpu_batch: int = 4096,
    use_float64: bool = True,
) -> CenterResult:
    """Dedicated tiled GPU center detection pipeline.

    Mirrors the reference find_subpixel_centers_tiled_hybrid_gpu:
    one global Otsu threshold, per-tile fixed threshold + GPU CC +
    vectorized area filter + batched refinement + band-based de-dup.
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

    # --- Step 0: Global threshold (once, on downsampled image) ---
    ds = max(1, int(otsu_downsample))
    thr_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    g_ds = g[::ds, ::ds]
    otsu_thresh, _ = cv2.threshold(g_ds, 0, 255, thr_type | cv2.THRESH_OTSU)
    thr = float(otsu_thresh) * float(thr_scale)

    step = tile_h - overlap
    all_centers = []

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) if (morph_open > 0 or morph_close > 0) else None

    y0 = 0
    while y0 < H:
        y1 = min(H, y0 + tile_h)
        tile = g[y0:y1]

        # --- Step 2: Per-tile threshold + morphology (CPU) ---
        _, bw = cv2.threshold(tile, thr, 255, thr_type)
        if morph_open > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=int(morph_open))
        if morph_close > 0:
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=int(morph_close))

        # --- Step 3: GPU connected components ---
        comp = connected_components_stats_gpu(bw, connectivity=int(connectivity), device=int(device))
        num = comp.num_labels

        # num_labels includes background (label 0). num_labels == 1 means no foreground.
        if num > 1:
            stats = comp.stats          # (num_labels, 5) int32: [min_x, min_y, w, h, area]
            centroids = comp.centroids  # (num_labels, 2) float64: [cx, cy]

            # --- Step 4: Vectorized area filtering (no Python loop) ---
            area = stats[1:, 4]
            mask = (area >= int(area_min)) & (area <= int(area_max))

            if np.any(mask):
                filtered_stats = stats[1:][mask]
                filtered_cents = centroids[1:][mask]

                # Build (K, 6): [x, y, w, h, cx, cy] — drop area column via [:, :4]
                stats_xywh_cc = np.concatenate([
                    filtered_stats[:, :4].astype(np.float32),
                    filtered_cents.astype(np.float32),
                ], axis=1)

                # --- Step 5: Refinement dispatch ---
                if refine == "edge_gradmoment":
                    refined, ok = refine_centers_edge_moment_gpu(
                        tile, stats_xywh_cc,
                        band_rad=int(band_rad), edge_rad=int(edge_rad),
                        smooth_passes=int(smooth_passes), loc_rad=int(loc_rad),
                        grad_power=float(grad_power), iters=int(iters),
                        device=int(device),
                    )
                elif refine in ("logquad", "logquadratic"):
                    raise NotImplementedError(
                        "refine='logquad' is not yet supported in the tiled GPU pipeline. "
                        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
                    )
                elif refine == "weighted":
                    raise NotImplementedError(
                        "refine='weighted' is not yet supported in the tiled GPU pipeline. "
                        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
                    )
                elif refine == "edge_erf":
                    raise NotImplementedError(
                        "refine='edge_erf' is not yet supported in the tiled GPU pipeline. "
                        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
                    )
                elif refine == "auto":
                    raise NotImplementedError(
                        "refine='auto' is not yet supported in the tiled GPU pipeline. "
                        "Use refine='edge_gradmoment' or the non-tiled detect_centers_gpu()."
                    )
                else:
                    raise ValueError(f"Unknown refine method: {refine}")

                # --- Step 6: Global coordinate offset + band-based de-dup ---
                refined = np.asarray(refined, dtype=np.float64)
                ok = np.asarray(ok, dtype=bool)
                good = ok & np.all(np.isfinite(refined), axis=1)
                refined = refined[good]

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
        },
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tiled_gpu_pipeline.py -v`
Expected: All GPU tests PASS (or SKIP if no CuPy).

- [ ] **Step 5: Commit**

```bash
git add subpx/_gpu/centers.py tests/test_tiled_gpu_pipeline.py
git commit -m "feat: add _detect_centers_tiled_gpu dedicated pipeline"
```

---

### Task 2: Update `detect_centers_tiled` to dispatch to new GPU pipeline

**Files:**
- Modify: `subpx/centers.py` (lines 184-251)
- Modify: `tests/test_tiled_gpu_pipeline.py` (add integration test)

- [ ] **Step 1: Write integration test through public API**

Append to `tests/test_tiled_gpu_pipeline.py`:

```python
@gpu
def test_detect_centers_tiled_dispatches_to_gpu_pipeline():
    """detect_centers_tiled(backend='gpu') should use the new tiled GPU pipeline."""
    from subpx import detect_centers_tiled

    img, expected = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = detect_centers_tiled(
        img,
        backend="gpu",
        tile_h=8192,
        overlap=16,
        refine="edge_gradmoment",
        area_min=1,
        area_max=50,
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert result.centers_xy.shape[0] == expected.shape[0]
    # Verify it used the new pipeline (check method string)
    assert "tiled_gpu" in result.method


def test_detect_centers_tiled_cpu_still_works():
    """CPU path should be unchanged."""
    from subpx import detect_centers_tiled

    img, _ = _make_dot_grid(rows=3, cols=3, spacing=15, dot_size=3, margin=10)
    result = detect_centers_tiled(
        img,
        backend="cpu",
        tile_h=8192,
        overlap=16,
        refine="weighted",
        area_min=1,
        area_max=50,
    )
    assert isinstance(result, CenterResult)
    assert result.centers_xy.shape[1] == 2
    assert "tiled(cpu)" in result.method


def test_detect_centers_tiled_cpu_overlap_validation():
    """CPU path: overlap >= tile_h should raise ValueError."""
    from subpx import detect_centers_tiled

    img = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap must be < tile_h"):
        detect_centers_tiled(img, backend="cpu", tile_h=64, overlap=64)
```

Note on overlap validation: The old `detect_centers_tiled` rejected `tile_h <= 2 * overlap` (e.g. `tile_h=9, overlap=4` would fail). The new validation relaxes this to `overlap >= tile_h`, matching the reference function's behavior. Inputs where `tile_h > overlap` but `tile_h <= 2 * overlap` now succeed — this is correct since `step = tile_h - overlap` is positive in those cases.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tiled_gpu_pipeline.py::test_detect_centers_tiled_dispatches_to_gpu_pipeline -v`
Expected: FAIL — `detect_centers_tiled` doesn't dispatch to GPU pipeline yet, `result.method` won't contain `"tiled_gpu"`.

- [ ] **Step 3: Update `detect_centers_tiled` in `subpx/centers.py`**

Replace lines 184-251 with the updated function. Key changes:
1. Add `thr_scale` parameter
2. Dispatch to `_detect_centers_tiled_gpu` when GPU backend
3. Update CPU path to use `step = tile_h - overlap` with band-based de-dup
4. Remove O(N^2) `dedupe_centers` from GPU path

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/test_tiled_gpu_pipeline.py tests/test_added_smoke.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add subpx/centers.py tests/test_tiled_gpu_pipeline.py
git commit -m "feat: wire detect_centers_tiled to dedicated GPU pipeline"
```

---

### Task 3: Run full test suite and verify no regressions

**Files:**
- None modified — verification only

- [ ] **Step 1: Run entire test suite**

Run: `pytest tests/ -v`
Expected: All existing tests PASS (some may SKIP if no GPU).

- [ ] **Step 2: Run the existing tiled smoke test specifically**

Run: `pytest tests/test_added_smoke.py::test_detect_centers_tiled_smoke -v`
Expected: PASS — this test uses `backend="cpu"` so it exercises the updated CPU path.

- [ ] **Step 3: Commit if any fixes were needed**

Only if regressions were found and fixed:
```bash
git add -u
git commit -m "fix: resolve test regressions from tiled pipeline refactor"
```
