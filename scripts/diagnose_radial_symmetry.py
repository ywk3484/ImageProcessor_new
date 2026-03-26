"""Diagnostic script for radial symmetry pipeline.

Run in Jupyter or as a script. Visualizes each pipeline step for a
selected set of blobs to identify where center estimation goes wrong.

Usage (Jupyter):
    %run scripts/diagnose_radial_symmetry.py

Then call:
    diag = run_diagnosis(IMAGE_PATH, invert=True, ...)
    inspect_blob(diag, blob_idx=42)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# 1. Run both methods and find the discrepant blobs
# ---------------------------------------------------------------------------

def run_diagnosis(
    image,
    *,
    invert: bool = True,
    area_min: int = 1,
    area_max: int = 50,
    pad: int = 3,
    upsample_factor: int = 4,
    device: int = 0,
    n_worst: int = 10,
    n_random: int = 5,
    crop_region: tuple[int, int, int, int] | None = None,
) -> dict:
    """Run both logquad and radial_symmetry, collect all intermediates.

    Parameters
    ----------
    image : numpy array or path to grayscale image
    crop_region : (y0, y1, x0, x1) to work on a sub-region (faster)
    n_worst : number of worst-discrepancy blobs to select
    n_random : number of random blobs to select

    Returns
    -------
    dict with all intermediate data for inspection
    """
    import cupy as cp
    from subpx._gpu.centers import (
        _segment_binary,
        _extract_voronoi_rois,
        _radial_symmetry_batch_gpu,
    )
    from subpx._gpu.voronoi import compute_voronoi_labels_gpu
    from subpx.components import connected_components_stats_cpu

    # -- Load image --
    if isinstance(image, np.ndarray):
        img = image
    else:
        import tifffile
        img = tifffile.imread(image) if str(image).endswith(('.tif', '.tiff')) else plt.imread(image)
    if img.ndim == 3:
        img = img[:, :, 0]
    img = np.asarray(img)

    if crop_region is not None:
        y0, y1, x0, x1 = crop_region
        img = img[y0:y1, x0:x1].copy()

    print(f"Image shape: {img.shape}, dtype: {img.dtype}")

    # -- Step A: Segmentation --
    g, bw = _segment_binary(img, threshold="otsu", invert=invert)
    print(f"Grayscale range: [{g.min()}, {g.max()}]")
    print(f"Binary: {bw.sum()} foreground pixels out of {bw.size}")

    # -- Step B: Connected components --
    comp = connected_components_stats_cpu(bw, connectivity=8)
    num = comp.num_labels
    print(f"Connected components: {num - 1} (excluding background)")

    # -- Step C: Filter by area --
    rows = []
    for lab in range(1, num):
        x, y, w, h, area = comp.stats[lab]
        if area < area_min or area > area_max:
            continue
        cx, cy = comp.centroids[lab]
        rows.append([x, y, w, h, cx, cy])
    print(f"After area filter: {len(rows)} components")

    if len(rows) == 0:
        print("No components found! Check area_min/area_max.")
        return {}

    rows_arr = np.asarray(rows, dtype=np.float64)
    coarse_centers = rows_arr[:, 4:6]  # [cx, cy]

    # -- Step D: Voronoi partition --
    print("Computing Voronoi labels...")
    voronoi_labels = compute_voronoi_labels_gpu(
        coarse_centers, g.shape, device=device,
    )
    cell_areas = np.bincount(voronoi_labels.ravel(), minlength=len(rows))
    print(f"Voronoi labels: min={voronoi_labels.min()}, max={voronoi_labels.max()}")
    print(f"Cell areas: min={cell_areas.min()}, max={cell_areas.max()}, "
          f"mean={cell_areas.mean():.1f}")

    # -- Step E: Extract ROIs --
    print("Extracting Voronoi-masked ROIs...")
    rois_stack, masks_stack, origins = _extract_voronoi_rois(
        g, voronoi_labels, rows, pad=pad,
    )
    print(f"ROI stack shape: {rois_stack.shape}")
    print(f"Mask stack shape: {masks_stack.shape}")

    # -- Step F: Run radial symmetry --
    print("Running radial symmetry refinement...")
    local_centers, residuals = _radial_symmetry_batch_gpu(
        rois_stack, masks_stack, upsample_factor=upsample_factor, device=device,
    )
    print(f"Local centers shape: {local_centers.shape}")
    print(f"Residuals: min={residuals.min():.6f}, max={residuals.max():.6f}, "
          f"median={np.median(residuals):.6f}")

    # -- Step G: Transform to image coordinates --
    N = len(rows)
    global_centers = np.empty((N, 2), dtype=np.float64)
    for j in range(N):
        x0, y0 = origins[j]
        global_centers[j, 0] = x0 + local_centers[j, 0]
        global_centers[j, 1] = y0 + local_centers[j, 1]

    valid_mask = np.all(np.isfinite(global_centers), axis=1)
    print(f"Valid centers: {valid_mask.sum()} / {N}")

    # -- Step H: Run logquad for comparison --
    print("Running logquad for comparison...")
    from subpx.centers import detect_centers
    res_lq = detect_centers(
        img, backend="gpu", refine="logquad", invert=invert,
        area_min=area_min, area_max=area_max, device=device,
    )
    print(f"Logquad found {res_lq.centers_xy.shape[0]} centers")

    # -- Step I: Match centers between methods --
    # For each radial_symmetry center, find nearest logquad center
    from scipy.spatial import cKDTree
    if res_lq.centers_xy.shape[0] > 0 and valid_mask.sum() > 0:
        tree_lq = cKDTree(res_lq.centers_xy)
        dists, idxs = tree_lq.query(global_centers[valid_mask])
    else:
        dists = np.array([])
        idxs = np.array([])

    print(f"Distance to nearest logquad center: "
          f"median={np.median(dists):.3f}, 95th={np.percentile(dists, 95):.3f}, "
          f"max={dists.max():.3f}")

    # -- Select sample blobs --
    valid_indices = np.where(valid_mask)[0]
    # Map dists back to full-array indices
    dist_full = np.full(N, np.inf)
    dist_full[valid_indices] = dists

    # Worst: largest distance to logquad
    worst_idx = np.argsort(dist_full[valid_indices])[::-1][:n_worst]
    worst_blob_ids = valid_indices[worst_idx]

    # Random
    rng = np.random.default_rng(42)
    rand_pool = valid_indices[dist_full[valid_indices] < np.median(dists) * 3]
    if len(rand_pool) > n_random:
        random_blob_ids = rng.choice(rand_pool, n_random, replace=False)
    else:
        random_blob_ids = rand_pool[:n_random]

    # Good: closest to logquad (sanity check)
    good_idx = np.argsort(dist_full[valid_indices])[:5]
    good_blob_ids = valid_indices[good_idx]

    sample_ids = np.unique(np.concatenate([worst_blob_ids, random_blob_ids, good_blob_ids]))

    print(f"\nSelected {len(sample_ids)} sample blobs for inspection:")
    print(f"  Worst {n_worst}: blob ids {worst_blob_ids.tolist()}")
    print(f"  Random {n_random}: blob ids {random_blob_ids.tolist()}")
    print(f"  Best 5: blob ids {good_blob_ids.tolist()}")

    diag = {
        "img": img,
        "g": g,
        "bw": bw,
        "rows": rows,
        "rows_arr": rows_arr,
        "coarse_centers": coarse_centers,
        "voronoi_labels": voronoi_labels,
        "cell_areas": cell_areas,
        "rois_stack": rois_stack,
        "masks_stack": masks_stack,
        "origins": origins,
        "local_centers": local_centers,
        "residuals": residuals,
        "global_centers": global_centers,
        "valid_mask": valid_mask,
        "logquad_centers": res_lq.centers_xy,
        "dist_to_logquad": dist_full,
        "sample_ids": sample_ids,
        "worst_blob_ids": worst_blob_ids,
        "good_blob_ids": good_blob_ids,
        "upsample_factor": upsample_factor,
        "pad": pad,
        "device": device,
    }
    return diag


# ---------------------------------------------------------------------------
# 2. Per-blob step-by-step inspection
# ---------------------------------------------------------------------------

def inspect_blob(diag: dict, blob_idx: int):
    """Visualize every pipeline step for a single blob.

    Parameters
    ----------
    diag : dict from run_diagnosis()
    blob_idx : index into rows/rois_stack (0-based)
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    from subpx._gpu.upsample import batch_bicubic_upsample, batch_nn_upsample
    from subpx.backends import gpu_device

    g = diag["g"]
    rows = diag["rows"]
    voronoi_labels = diag["voronoi_labels"]
    rois_stack = diag["rois_stack"]
    masks_stack = diag["masks_stack"]
    origins = diag["origins"]
    local_centers = diag["local_centers"]
    global_centers = diag["global_centers"]
    logquad_centers = diag["logquad_centers"]
    factor = diag["upsample_factor"]
    pad = diag["pad"]
    device = diag["device"]
    dist_to_lq = diag["dist_to_logquad"]

    j = blob_idx
    x, y, w, h, cx, cy = rows[j]
    x0, y0 = origins[j]
    roi = rois_stack[j]
    mask = masks_stack[j]
    lc = local_centers[j]
    gc = global_centers[j]

    H_img, W_img = g.shape
    x0_bb = max(0, int(x) - pad)
    y0_bb = max(0, int(y) - pad)
    x1_bb = min(W_img, int(x + w) + pad)
    y1_bb = min(H_img, int(y + h) + pad)
    roi_h = y1_bb - y0_bb
    roi_w = x1_bb - x0_bb

    print(f"=== Blob {j} ===")
    print(f"  CC bbox: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}")
    print(f"  Coarse centroid: cx={cx:.2f}, cy={cy:.2f}")
    print(f"  ROI origin: x0={x0}, y0={y0}")
    print(f"  ROI shape (in stack): {roi.shape}")
    print(f"  Mask true pixels: {mask.sum()}")
    print(f"  Local center: x={lc[0]:.3f}, y={lc[1]:.3f}")
    print(f"  Global center: x={gc[0]:.3f}, y={gc[1]:.3f}")
    print(f"  Distance to logquad: {dist_to_lq[j]:.3f}")

    # ---- Figure 1: Overview in image context ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1a: Original image crop around blob (20px context)
    ctx = 20
    vy0 = max(0, int(cy) - ctx)
    vy1 = min(H_img, int(cy) + ctx)
    vx0 = max(0, int(cx) - ctx)
    vx1 = min(W_img, int(cx) + ctx)
    ax = axes[0]
    ax.imshow(g[vy0:vy1, vx0:vx1], cmap="gray", origin="upper",
              extent=[vx0, vx1, vy1, vy0])
    ax.plot(gc[0], gc[1], "r+", ms=12, mew=2, label="radial_sym")
    ax.plot(cx, cy, "bx", ms=10, mew=2, label="coarse")
    # Find nearest logquad center
    if logquad_centers.shape[0] > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(logquad_centers)
        d, idx = tree.query([gc[0], gc[1]])
        lq_pt = logquad_centers[idx]
        ax.plot(lq_pt[0], lq_pt[1], "g+", ms=12, mew=2, label=f"logquad (d={d:.2f})")
    rect = mpatches.Rectangle((x0_bb, y0_bb), roi_w, roi_h,
                                linewidth=1, edgecolor="yellow", facecolor="none")
    ax.add_patch(rect)
    ax.legend(fontsize=8)
    ax.set_title(f"Blob {j}: Image context")

    # 1b: Voronoi labels around blob
    ax = axes[1]
    vlabels = voronoi_labels[vy0:vy1, vx0:vx1]
    ax.imshow(vlabels, cmap="tab20", origin="upper",
              extent=[vx0, vx1, vy1, vy0])
    ax.plot(gc[0], gc[1], "r+", ms=12, mew=2)
    ax.set_title(f"Voronoi labels (this blob = {j})")

    # 1c: ROI with mask overlay
    ax = axes[2]
    # Show actual ROI dimensions (not zero-padded)
    ax.imshow(roi[:roi_h, :roi_w], cmap="gray", origin="upper")
    mask_vis = mask[:roi_h, :roi_w].astype(float)
    ax.imshow(mask_vis, cmap="Reds", alpha=0.3, origin="upper")
    ax.plot(lc[0], lc[1], "r+", ms=12, mew=2, label="local center")
    # Expected local center = coarse - origin
    expected_lx = cx - x0
    expected_ly = cy - y0
    ax.plot(expected_lx, expected_ly, "bx", ms=10, mew=2, label="coarse (local)")
    ax.legend(fontsize=8)
    ax.set_title(f"ROI [{roi_h}x{roi_w}] + Voronoi mask")

    plt.tight_layout()
    plt.show()

    # ---- Figure 2: Upsample comparison ----
    with gpu_device(device):
        # Batch kernel upsample
        roi_3d = rois_stack[j:j+1]  # (1, H, W)
        mask_3d = masks_stack[j:j+1]
        roi_gpu = cp.asarray(roi_3d, dtype=cp.float64)
        mask_gpu = cp.asarray(mask_3d.astype(np.float64))

        up_batch = cp.asnumpy(batch_bicubic_upsample(roi_gpu, factor))[0]
        up_mask_batch = cp.asnumpy(batch_nn_upsample(mask_gpu, factor))[0] > 0.5

        # Scipy reference upsample
        roi_g = cp.asarray(roi_3d[0])
        mask_g = cp.asarray(mask_3d[0].astype(np.float64))
        up_scipy = cp.asnumpy(cndi.zoom(roi_g, factor, order=3))
        up_mask_scipy = cp.asnumpy(cndi.zoom(mask_g, factor, order=0)) > 0.5

    print(f"\n  Upsample factor: {factor}")
    print(f"  Batch kernel output: {up_batch.shape}")
    print(f"  Scipy output: {up_scipy.shape}")
    diff = up_batch - up_scipy
    print(f"  Max abs diff (batch vs scipy): {np.abs(diff).max():.6f}")
    print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")
    print(f"  Mask agreement: {(up_mask_batch == up_mask_scipy).mean()*100:.2f}%")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    up_h, up_w = up_batch.shape

    ax = axes[0, 0]
    ax.imshow(up_batch[:roi_h*factor, :roi_w*factor], cmap="gray", origin="upper")
    ax.set_title(f"Batch upsample [{up_h}x{up_w}]")

    ax = axes[0, 1]
    ax.imshow(up_scipy[:roi_h*factor, :roi_w*factor], cmap="gray", origin="upper")
    ax.set_title(f"Scipy upsample [{up_scipy.shape[0]}x{up_scipy.shape[1]}]")

    # Clip to common shape for diff
    ch = min(up_batch.shape[0], up_scipy.shape[0])
    cw = min(up_batch.shape[1], up_scipy.shape[1])
    diff_clip = up_batch[:ch, :cw] - up_scipy[:ch, :cw]
    ax = axes[0, 2]
    im = ax.imshow(diff_clip[:roi_h*factor, :roi_w*factor], cmap="RdBu_r", origin="upper")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Diff (batch - scipy), max={np.abs(diff_clip).max():.4f}")

    # Mask comparison
    ax = axes[1, 0]
    ax.imshow(up_mask_batch[:roi_h*factor, :roi_w*factor], cmap="gray", origin="upper")
    ax.set_title("Mask: batch NN upsample")

    ax = axes[1, 1]
    ax.imshow(up_mask_scipy[:roi_h*factor, :roi_w*factor], cmap="gray", origin="upper")
    ax.set_title("Mask: scipy NN upsample")

    ax = axes[1, 2]
    mask_diff = up_mask_batch[:ch, :cw].astype(int) - up_mask_scipy[:ch, :cw].astype(int)
    ax.imshow(mask_diff[:roi_h*factor, :roi_w*factor], cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax.set_title(f"Mask diff (batch - scipy)")

    plt.tight_layout()
    plt.show()

    # ---- Figure 3: Radial symmetry internals ----
    # Re-run the algorithm step by step for this single ROI
    _inspect_radial_symmetry_steps(
        roi_3d, mask_3d, factor=factor, device=device, blob_idx=j,
    )


def _inspect_radial_symmetry_steps(
    rois: np.ndarray,
    masks: np.ndarray,
    *,
    factor: int = 4,
    device: int = 0,
    blob_idx: int = 0,
):
    """Re-run radial symmetry steps for a single ROI and visualize."""
    import cupy as cp
    import cupyx.scipy.ndimage as cndi
    from subpx._gpu.upsample import batch_bicubic_upsample, batch_nn_upsample
    from subpx.backends import gpu_device

    N = 1  # single ROI
    boundary_margin = factor

    with gpu_device(device):
        orig_h, orig_w = rois.shape[1], rois.shape[2]

        # Step 2-3: Upsample
        if factor > 1:
            rois_gpu = cp.asarray(rois, dtype=cp.float64)
            masks_gpu = cp.asarray(masks.astype(np.float64))
            g_batch = batch_bicubic_upsample(rois_gpu, factor)
            m_batch = batch_nn_upsample(masks_gpu, factor) > 0.5
        else:
            g_batch = cp.asarray(rois)
            m_batch = cp.asarray(masks)

        up_h, up_w = g_batch.shape[1], g_batch.shape[2]

        # Step 4: Erode mask
        min_up_dim = min(up_h, up_w)
        max_margin = max(0, (min_up_dim - 11) // 2)
        eff_margin = min(boundary_margin, max_margin)
        if eff_margin > 0:
            struct = cp.ones((1, 2 * eff_margin + 1, 2 * eff_margin + 1), dtype=bool)
            m_eroded = cndi.binary_erosion(m_batch, structure=struct)
        else:
            m_eroded = m_batch.copy()

        # Step 5: Diagonal gradients
        I = g_batch
        Hp = up_h - 1
        Wp = up_w - 1

        dIdu = I[:, :Hp, 1:Wp+1] - I[:, 1:Hp+1, :Wp]
        dIdv = I[:, :Hp, :Wp]    - I[:, 1:Hp+1, 1:Wp+1]

        # Step 6: Valid midpoints
        m00 = m_eroded[:, :Hp, :Wp]
        m01 = m_eroded[:, :Hp, 1:Wp+1]
        m10 = m_eroded[:, 1:Hp+1, :Wp]
        m11 = m_eroded[:, 1:Hp+1, 1:Wp+1]
        valid = m00 & m01 & m10 & m11

        # Step 7: Slope
        denom = dIdu - dIdv
        near_zero = cp.abs(denom) < 1e-12
        safe_denom = cp.where(near_zero, cp.float64(1.0), denom)
        slope = -(dIdv + dIdu) / safe_denom
        slope = cp.where(near_zero,
                         cp.where(-(dIdv + dIdu) >= 0, cp.float64(1e9), cp.float64(-1e9)),
                         slope)

        # Step 8: Centered coordinates
        col_idx = cp.arange(Wp, dtype=cp.float64)[None, None, :] + 0.5
        row_idx = cp.arange(Hp, dtype=cp.float64)[None, :, None] + 0.5

        cx_off = (up_w - 1.0) / 2.0
        cy_off = (up_h - 1.0) / 2.0
        xm = col_idx - cx_off
        ym = row_idx - cy_off

        # Step 9: Intercepts
        b = ym - slope * xm

        # Step 10: Weights
        grad_mag_sq = dIdu**2 + dIdv**2
        w_base = cp.where(valid, grad_mag_sq, cp.float64(0.0))
        w_sum = w_base.sum(axis=(1, 2))
        w_sum_safe = cp.maximum(w_sum, cp.float64(1e-30))
        gc_x = (w_base * xm).sum(axis=(1, 2)) / w_sum_safe
        gc_y = (w_base * ym).sum(axis=(1, 2)) / w_sum_safe
        dist_sq = (xm - gc_x[:, None, None])**2 + (ym - gc_y[:, None, None])**2
        dist = cp.sqrt(cp.maximum(dist_sq, cp.float64(1e-30)))
        w = cp.where(valid, grad_mag_sq / dist, cp.float64(0.0))

        # Step 11: Solve
        sw = w.sum(axis=(1, 2))
        smmw = (slope**2 * w).sum(axis=(1, 2))
        smw = (slope * w).sum(axis=(1, 2))
        smbw = (slope * b * w).sum(axis=(1, 2))
        sbw = (b * w).sum(axis=(1, 2))
        det = smmw * sw - smw * smw
        det_safe = cp.where(cp.abs(det) < 1e-30, cp.float64(1e-30), det)
        xc = (-smbw * sw + smw * sbw) / det_safe
        yc = (smmw * sbw - smbw * smw) / det_safe

        # Step 12: Residual
        perp_d_sq = (slope * xc[:, None, None] - yc[:, None, None] + b)**2 / (slope**2 + 1.0)
        residual = (w * perp_d_sq).sum(axis=(1, 2)) / cp.maximum(sw, cp.float64(1e-30))

        # Download everything
        g_np = cp.asnumpy(g_batch[0])
        m_np = cp.asnumpy(m_batch[0])
        m_eroded_np = cp.asnumpy(m_eroded[0])
        dIdu_np = cp.asnumpy(dIdu[0])
        dIdv_np = cp.asnumpy(dIdv[0])
        valid_np = cp.asnumpy(valid[0])
        slope_np = cp.asnumpy(slope[0])
        grad_mag = cp.asnumpy(cp.sqrt(grad_mag_sq[0]))
        w_np = cp.asnumpy(w[0])
        b_np = cp.asnumpy(b[0])
        xm_full = cp.broadcast_to(xm, (1, Hp, Wp))
        ym_full = cp.broadcast_to(ym, (1, Hp, Wp))
        xm_np = cp.asnumpy(xm_full[0])
        ym_np = cp.asnumpy(ym_full[0])
        xc_val = float(cp.asnumpy(xc[0]))
        yc_val = float(cp.asnumpy(yc[0]))
        residual_val = float(cp.asnumpy(residual[0]))

    # Center in upsampled pixel coordinates
    xc_pix = xc_val + (up_w - 1) / 2
    yc_pix = yc_val + (up_h - 1) / 2

    # Center in original ROI coordinates
    if factor > 1 and up_w > 1 and up_h > 1:
        xc_orig = xc_pix * (orig_w - 1) / (up_w - 1)
        yc_orig = yc_pix * (orig_h - 1) / (up_h - 1)
    else:
        xc_orig = xc_pix
        yc_orig = yc_pix

    print(f"\n  --- Radial Symmetry Internals (blob {blob_idx}) ---")
    print(f"  Upsampled size: {up_h} x {up_w}")
    print(f"  Effective erosion margin: {eff_margin}")
    print(f"  Valid midpoints: {valid_np.sum()} / {Hp * Wp}")
    print(f"  Center (centered coords): xc={xc_val:.4f}, yc={yc_val:.4f}")
    print(f"  Center (upsampled px): xc={xc_pix:.4f}, yc={yc_pix:.4f}")
    print(f"  Center (original px): xc={xc_orig:.4f}, yc={yc_orig:.4f}")
    print(f"  Residual: {residual_val:.6f}")

    # ---- Figure 3: Internal steps ----
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # 3a: Upsampled ROI with center
    ax = axes[0, 0]
    ax.imshow(g_np, cmap="gray", origin="upper")
    ax.plot(xc_pix, yc_pix, "r+", ms=15, mew=2, label="estimated")
    ax.plot((up_w - 1) / 2, (up_h - 1) / 2, "bx", ms=10, mew=2, label="ROI center")
    ax.legend(fontsize=8)
    ax.set_title(f"Upsampled ROI + center")

    # 3b: Mask (upsampled) + eroded mask
    ax = axes[0, 1]
    ax.imshow(m_np, cmap="gray", origin="upper")
    # Overlay eroded mask boundary
    ax.contour(m_eroded_np, levels=[0.5], colors="red", linewidths=1)
    ax.set_title(f"Mask + eroded boundary (margin={eff_margin})")

    # 3c: Gradient magnitude at midpoints
    ax = axes[0, 2]
    im = ax.imshow(grad_mag, cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax)
    ax.set_title("Gradient magnitude (midpoints)")

    # 3d: Valid midpoints
    ax = axes[0, 3]
    ax.imshow(valid_np, cmap="gray", origin="upper")
    ax.set_title(f"Valid midpoints ({valid_np.sum()})")

    # 3e: Slope field (clamped for visualization)
    ax = axes[1, 0]
    slope_vis = np.clip(slope_np, -10, 10)
    im = ax.imshow(slope_vis, cmap="RdBu_r", origin="upper", vmin=-5, vmax=5)
    plt.colorbar(im, ax=ax)
    ax.set_title("Slope (clamped to [-5,5])")

    # 3f: Weights
    ax = axes[1, 1]
    im = ax.imshow(np.where(valid_np, w_np, 0), cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax)
    ax.set_title("Weights (grad^2 / dist)")

    # 3g: Symmetry lines (subsample for clarity)
    ax = axes[1, 2]
    ax.imshow(g_np, cmap="gray", origin="upper")
    # Draw a subset of symmetry lines through the estimated center
    rows_mp, cols_mp = np.where(valid_np & (grad_mag > np.percentile(grad_mag[valid_np], 70)))
    step = max(1, len(rows_mp) // 40)
    for ii in range(0, len(rows_mp), step):
        r, c = rows_mp[ii], cols_mp[ii]
        mx = xm_np[r, c]
        my = ym_np[r, c]
        s = slope_np[r, c]
        bi = b_np[r, c]
        # Line: y = s * x + bi (in centered coords)
        # Convert to pixel coords for plotting
        x_line = np.linspace(mx - 5, mx + 5, 20)
        y_line = s * x_line + bi
        x_line_px = x_line + (up_w - 1) / 2
        y_line_px = y_line + (up_h - 1) / 2
        ax.plot(x_line_px, y_line_px, "c-", alpha=0.3, linewidth=0.5)
    ax.plot(xc_pix, yc_pix, "r+", ms=15, mew=2)
    ax.set_xlim(0, up_w)
    ax.set_ylim(up_h, 0)
    ax.set_title("Symmetry lines → center")

    # 3h: Center on original-scale ROI
    ax = axes[1, 3]
    ax.imshow(rois[0], cmap="gray", origin="upper")
    ax.plot(xc_orig, yc_orig, "r+", ms=15, mew=2, label="radial_sym")
    ax.plot((orig_w - 1) / 2, (orig_h - 1) / 2, "bx", ms=10, mew=2, label="ROI center")
    ax.legend(fontsize=8)
    ax.set_title(f"Original ROI + center (x={xc_orig:.2f}, y={yc_orig:.2f})")

    plt.suptitle(f"Blob {blob_idx}: Radial Symmetry Steps  |  residual={residual_val:.6f}",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Quick overview plots
# ---------------------------------------------------------------------------

def plot_overview(diag: dict, region: tuple[int, int, int, int] | None = None):
    """Show global overview comparing logquad vs radial symmetry centers.

    Parameters
    ----------
    region : (y0, y1, x0, x1) zoom region in image coords
    """
    g = diag["g"]
    gc = diag["global_centers"]
    lq = diag["logquad_centers"]
    valid = diag["valid_mask"]

    if region is None:
        # Auto-pick a region around the worst blob
        worst = diag["worst_blob_ids"]
        if len(worst) > 0:
            j = worst[0]
            cy, cx = gc[j, 1], gc[j, 0]
            region = (int(cy) - 50, int(cy) + 50, int(cx) - 50, int(cx) + 50)
        else:
            region = (0, min(200, g.shape[0]), 0, min(200, g.shape[1]))

    y0, y1, x0, x1 = region
    y0, y1 = max(0, y0), min(g.shape[0], y1)
    x0, x1 = max(0, x0), min(g.shape[1], x1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, title, centers, color, marker in [
        (axes[0], "Logquad", lq, "lime", "+"),
        (axes[1], "Radial symmetry", gc[valid], "red", "+"),
    ]:
        ax.imshow(g[y0:y1, x0:x1], cmap="gray", origin="upper",
                  extent=[x0, x1, y1, y0])
        in_region = (
            (centers[:, 0] >= x0) & (centers[:, 0] < x1) &
            (centers[:, 1] >= y0) & (centers[:, 1] < y1)
        )
        pts = centers[in_region]
        ax.plot(pts[:, 0], pts[:, 1], marker, color=color, ms=8, mew=1.5)
        ax.set_title(f"{title} ({pts.shape[0]} centers in view)")

    plt.suptitle(f"Region [{y0}:{y1}, {x0}:{x1}]", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_residual_histogram(diag: dict):
    """Plot histogram of residuals and distance-to-logquad."""
    residuals = diag["residuals"]
    dists = diag["dist_to_logquad"]
    valid = diag["valid_mask"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(residuals[valid], bins=100, log=True)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Radial symmetry residual distribution")

    ax = axes[1]
    d = dists[valid & np.isfinite(dists)]
    ax.hist(d, bins=100, log=True)
    ax.axvline(np.median(d), color="red", label=f"median={np.median(d):.2f}")
    ax.axvline(np.percentile(d, 95), color="orange", label=f"95th={np.percentile(d, 95):.2f}")
    ax.legend()
    ax.set_xlabel("Distance to nearest logquad center (px)")
    ax.set_ylabel("Count")
    ax.set_title("Distance: radial_symmetry vs logquad")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Convenience: inspect all sample blobs
# ---------------------------------------------------------------------------

def inspect_all_samples(diag: dict):
    """Run inspect_blob for all selected sample blobs."""
    for j in diag["sample_ids"]:
        inspect_blob(diag, j)
        print("\n" + "=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Diagnostic script loaded.")
    print()
    print("Usage in Jupyter:")
    print("  from scripts.diagnose_radial_symmetry import *")
    print()
    print("  # Run diagnosis (adjust path and params)")
    print('  diag = run_diagnosis(img_array, invert=True)  # or "path/to/image.tif"')
    print()
    print("  # Overview plots")
    print("  plot_overview(diag)")
    print("  plot_residual_histogram(diag)")
    print()
    print("  # Inspect specific blob")
    print("  inspect_blob(diag, blob_idx=42)")
    print()
    print("  # Inspect all worst + random samples")
    print("  inspect_all_samples(diag)")
    print()
    print("  # Inspect a specific region")
    print("  plot_overview(diag, region=(100, 300, 200, 400))")
