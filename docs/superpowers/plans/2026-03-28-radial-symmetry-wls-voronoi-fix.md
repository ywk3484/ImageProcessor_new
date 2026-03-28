# Radial Symmetry WLS + Voronoi Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix WLS normalization bug, add gradient smoothing, reduce erosion, fix diagnostic colormap

**Architecture:** Three fixes to `_radial_symmetry_batch_gpu` (WLS normalization, gradient smoothing, erosion), two fixes to diagnostic script (colormap, mirror algorithm fixes)

**Tech Stack:** CuPy (GPU), NumPy, matplotlib

---

### Task 1: Fix WLS perpendicular-distance normalization

**Files:**
- Modify: `subpx/_gpu/centers.py:1126-1135`
- Test: `pytest tests/ -v`

- [ ] **Step 1: Read the current WLS solve block**

Read `subpx/_gpu/centers.py` lines 1118-1142. Current code (lines 1126-1135):

```python
sw = w.sum(axis=(1, 2))
smmw = (slope**2 * w).sum(axis=(1, 2))
smw = (slope * w).sum(axis=(1, 2))
smbw = (slope * b * w).sum(axis=(1, 2))
sbw = (b * w).sum(axis=(1, 2))

det = smmw * sw - smw * smw
det_safe = cp.where(cp.abs(det) < 1e-30, cp.float64(1e-30), det)
xc = (-smbw * sw + smw * sbw) / det_safe
yc = (smmw * sbw - smbw * smw) / det_safe
```

- [ ] **Step 2: Apply the fix**

Replace lines 1126-1130 with normalized weights. The key change is to divide `w` by `(slope**2 + 1.0)` before computing all sums, matching the Parthasarathy reference MATLAB implementation (`wm2p1 = w./(m.*m+1)`):

```python
# -- Step 11: Analytic 2x2 solve --
# Weighted least squares for intersection of lines y = m*x + b
# Minimize sum_k w_k * (m_k*xc - yc + b_k)^2 / (m_k^2 + 1) over (xc, yc)
# The 1/(m^2+1) factor converts from vertical to perpendicular distance.
# Reference: Parthasarathy, Nature Methods 9:724 (2012), radialcenter.m
wm2p1 = w / (slope**2 + 1.0)    # (N, Hp, Wp) — perpendicular-distance normalized
sw = wm2p1.sum(axis=(1, 2))       # (N,)
smmw = (slope**2 * wm2p1).sum(axis=(1, 2))  # (N,)
smw = (slope * wm2p1).sum(axis=(1, 2))       # (N,)
smbw = (slope * b * wm2p1).sum(axis=(1, 2))  # (N,)
sbw = (b * wm2p1).sum(axis=(1, 2))           # (N,)
```

Lines 1132-1135 (det, det_safe, xc, yc) remain unchanged.

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All existing tests pass. The normalization change improves accuracy but doesn't change the API.

- [ ] **Step 4: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "fix: add 1/(m²+1) perpendicular-distance normalization to radial symmetry WLS"
```

---

### Task 2: Add 3×3 gradient smoothing

**Files:**
- Modify: `subpx/_gpu/centers.py:1065-1066` (insert after these lines)
- Test: `pytest tests/ -v`

- [ ] **Step 1: Read the current gradient computation**

Read `subpx/_gpu/centers.py` lines 1055-1066. Current code:

```python
I = g_batch
Hp = Hmax - 1
Wp = Wmax - 1
# ...
dIdu = I[:, :Hp, 1:Wp+1] - I[:, 1:Hp+1, :Wp]
dIdv = I[:, :Hp, :Wp]    - I[:, 1:Hp+1, 1:Wp+1]
```

- [ ] **Step 2: Add smoothing after gradient computation**

Insert after line 1066 (after `dIdv = ...`). Use a manual 3×3 averaging stencil via neighbor indexing (avoids FFT overhead, more efficient for a 3×3 kernel on GPU):

```python
# -- Step 5b: Smooth gradients with 3x3 averaging (Parthasarathy reference) --
# Reduces noise in gradient field; original paper reports error reduction
# from 0.04 to 0.027 pixels with this smoothing.
if Hp >= 3 and Wp >= 3:
    def _smooth3x3(arr):
        """3x3 mean filter via neighbor sum, replicate-padded."""
        # Pad by 1 on each side (replicate boundary)
        p = cp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
        out = (
            p[:, 0:-2, 0:-2] + p[:, 0:-2, 1:-1] + p[:, 0:-2, 2:]
            + p[:, 1:-1, 0:-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:]
            + p[:, 2:,   0:-2] + p[:, 2:,   1:-1] + p[:, 2:,   2:]
        ) / 9.0
        return out
    dIdu = _smooth3x3(dIdu)
    dIdv = _smooth3x3(dIdv)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "feat: add 3x3 gradient smoothing to radial symmetry (per Parthasarathy reference)"
```

---

### Task 3: Reduce erosion margin

**Files:**
- Modify: `subpx/_gpu/centers.py:1006-1007`
- Test: `pytest tests/ -v`

- [ ] **Step 1: Read the current default**

Read `subpx/_gpu/centers.py` lines 1006-1007:

```python
if boundary_margin is None:
    boundary_margin = upsample_factor
```

- [ ] **Step 2: Reduce the default**

```python
if boundary_margin is None:
    boundary_margin = max(1, upsample_factor // 2)
```

For 4× upsample: erosion goes from 4 to 2 upsampled pixels. Combined with the 3×3 gradient smoothing from Task 2, boundary artifacts are sufficiently attenuated.

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add subpx/_gpu/centers.py
git commit -m "fix: reduce radial symmetry erosion margin from factor to factor//2"
```

---

### Task 4: Fix diagnostic Voronoi colormap

**Files:**
- Modify: `scripts/diagnose_radial_symmetry.py` (in `inspect_blob`, Voronoi label visualization)

- [ ] **Step 1: Read the current colormap code**

In `inspect_blob`, find the Voronoi label imshow (around line 327):

```python
vlabels = voronoi_labels[vy0:vy1, vx0:vx1]
ax.imshow(vlabels, cmap="tab20", origin="upper")
```

- [ ] **Step 2: Replace with shuffled colormap**

```python
vlabels = voronoi_labels[vy0:vy1, vx0:vx1]
# Use shuffled random colors to avoid tab20 color collisions
# (tab20 has only 20 colors — cells with indices differing by 20 look identical)
n_labels = int(voronoi_labels.max()) + 1
rng_colors = np.random.default_rng(0)
label_colors = rng_colors.random((max(n_labels, 1), 3))
from matplotlib.colors import ListedColormap
voronoi_cmap = ListedColormap(label_colors)
ax.imshow(vlabels, cmap=voronoi_cmap, origin="upper", interpolation="nearest")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/diagnose_radial_symmetry.py
git commit -m "fix: use shuffled colormap for Voronoi labels in diagnostic"
```

---

### Task 5: Update diagnostic `_inspect_radial_symmetry_steps`

**Files:**
- Modify: `scripts/diagnose_radial_symmetry.py`, function `_inspect_radial_symmetry_steps`

This function manually re-implements the radial symmetry algorithm for per-blob visualization. It must mirror the production code fixes from Tasks 1-3.

- [ ] **Step 1: Add gradient smoothing (matching Task 2)**

After line 467 (`dIdv = I[:, :Hp, :Wp] - I[:, 1:Hp+1, 1:Wp+1]`), add:

```python
# Smooth gradients with 3x3 averaging (matching production pipeline)
if Hp >= 3 and Wp >= 3:
    def _smooth3x3(arr):
        p = cp.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
        return (
            p[:, 0:-2, 0:-2] + p[:, 0:-2, 1:-1] + p[:, 0:-2, 2:]
            + p[:, 1:-1, 0:-2] + p[:, 1:-1, 1:-1] + p[:, 1:-1, 2:]
            + p[:, 2:,   0:-2] + p[:, 2:,   1:-1] + p[:, 2:,   2:]
        ) / 9.0
    dIdu = _smooth3x3(dIdu)
    dIdv = _smooth3x3(dIdv)
```

- [ ] **Step 2: Fix WLS normalization (matching Task 1)**

Replace lines 508-516 (the WLS sums):

```python
# From:
sw = w.sum(axis=(1, 2))
smmw = (slope**2 * w).sum(axis=(1, 2))
smw = (slope * w).sum(axis=(1, 2))
smbw = (slope * b * w).sum(axis=(1, 2))
sbw = (b * w).sum(axis=(1, 2))

# To:
wm2p1 = w / (slope**2 + 1.0)
sw = wm2p1.sum(axis=(1, 2))
smmw = (slope**2 * wm2p1).sum(axis=(1, 2))
smw = (slope * wm2p1).sum(axis=(1, 2))
smbw = (slope * b * wm2p1).sum(axis=(1, 2))
sbw = (b * wm2p1).sum(axis=(1, 2))
```

- [ ] **Step 3: Reduce erosion margin (matching Task 3)**

Change line 453 from `eff_margin = min(boundary_margin, max_margin)` to use the reduced default:

```python
boundary_margin = max(1, factor // 2)  # reduced from factor
# ... rest of erosion logic unchanged
```

- [ ] **Step 4: Commit**

```bash
git add scripts/diagnose_radial_symmetry.py
git commit -m "fix: mirror WLS, smoothing, erosion fixes in diagnostic inspect_blob"
```

---

### Task 6: Run full verification

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify with diagnostic script**

In Jupyter, run:
```python
diag = run_diagnosis(img, invert=True)
plot_residual_histogram(diag)
inspect_blob(diag, blob_idx=diag["worst_blob_ids"][0])
```

Expected:
- Distance median and 95th percentile should be smaller than before
- No centers outside features (max distance should be reasonable)
- Voronoi labels show distinct colors per cell
- Gradient mask covers more of the circle (less clipping)

- [ ] **Step 3: Final commit if any cleanup needed**
