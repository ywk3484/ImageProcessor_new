#!/usr/bin/env python3
"""Diagnostic B — Per-check failure breakdown for edge_erf on a synthetic grid image.

Creates a grid of large bright rectangles, runs both edge_gradmoment and edge_erf,
then instruments the erf validation to report which specific checks cause rejection.

Run:  python tests/diag_erf_integration.py
"""
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("CuPy required — run on GPU workstation")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from subpx._gpu.centers import (
    _edge_from_profile_erf_gpu,
    _edge_from_profile_gradmoment_gpu,
    _edge_polarity_from_profile_gpu,
    _smooth_binom5_gpu,
    _erf_gpu,
)


# ── Synthetic image: grid of bright rectangles ──────────────────────

def make_grid_image(n_rows=10, n_cols=10, feat_h=25, feat_w=25, pitch=50,
                    bg=30.0, fg=220.0, margin=50):
    """Create a synthetic image with a grid of bright rectangles."""
    H = 2 * margin + n_rows * pitch
    W = 2 * margin + n_cols * pitch
    img = np.full((H, W), bg, dtype=np.float32)
    centers_true = []
    stats = []  # [x, y, w, h, cx, cy]

    for r in range(n_rows):
        for c in range(n_cols):
            cy = margin + r * pitch + pitch // 2
            cx = margin + c * pitch + pitch // 2
            y0 = cy - feat_h // 2
            x0 = cx - feat_w // 2
            img[y0:y0 + feat_h, x0:x0 + feat_w] = fg
            centers_true.append([cx, cy])
            stats.append([x0, y0, feat_w, feat_h, cx, cy])

    return img, np.array(centers_true, dtype=np.float64), np.array(stats, dtype=np.float32)


# ── Instrumented erf validation ─────────────────────────────────────

def instrumented_erf_edge(prof, base_x, x_seed, label="edge"):
    """Run _edge_from_profile_erf_gpu and report per-check pass/fail rates."""
    prof_g = cp.asarray(prof, dtype=cp.float64)
    base_g = cp.asarray(base_x, dtype=cp.float64)
    seed_g = cp.asarray(x_seed, dtype=cp.float64)
    B, L = prof_g.shape

    # Get the actual fitter result
    x_out, ok = _edge_from_profile_erf_gpu(prof_g, base_g, seed_g)

    # Now re-compute validation checks to see which ones fail
    # We need to run through the same logic to get the final parameters
    # The simplest approach: replicate the post-validation checks

    pol = _edge_polarity_from_profile_gpu(prof_g, base_g, seed_g, loc_rad=2)
    ymin = cp.min(prof_g, axis=1)
    ymax = cp.max(prof_g, axis=1)
    amp_floor = cp.maximum((ymax - ymin) * 1e-6, 1e-9)
    dyn = cp.maximum(ymax - ymin, 1e-9)

    sigma_min, sigma_max = 0.35, 6.0
    max_shift_f = 2.5
    init_sigma = 1.25
    fit_rad_f = 5.0
    damp = 1e-4
    cond_max = 1e8
    sqrt_2 = float(np.sqrt(2.0))
    sqrt_pi = float(np.sqrt(np.pi))
    sqrt_2pi = float(np.sqrt(2.0 * np.pi))

    # Re-run the Gauss-Newton to capture final state
    xs = base_g[:, None] + cp.arange(L, dtype=cp.float64)[None, :]
    a = ymin.copy()
    b = cp.zeros((B,), dtype=cp.float64)
    c = cp.maximum(ymax - ymin, amp_floor)
    x0 = seed_g.copy()
    lsig = cp.full((B,), float(np.log(init_sigma)), dtype=cp.float64)
    eye5 = cp.eye(5, dtype=cp.float64)[None, :, :]
    ok_gn = cp.ones((B,), dtype=bool)

    for it in range(5):
        sig = cp.clip(cp.exp(lsig), sigma_min, sigma_max)
        z = (xs - x0[:, None]) / (sqrt_2 * sig[:, None])
        s = 0.5 * (1.0 + pol[:, None] * _erf_gpu(z))
        model = a[:, None] + b[:, None] * xs + c[:, None] * s
        w_fit = (cp.abs(xs - x0[:, None]) <= fit_rad_f).astype(cp.float64)
        if L >= 7:
            w_fit += 0.15 * (cp.abs(xs - x0[:, None]) <= (fit_rad_f + 1.0)).astype(cp.float64)
        sw = cp.sqrt(w_fit)
        r = (prof_g - model) * sw
        ez = cp.exp(-(z * z))
        ds_dx0 = (-pol[:, None] * ez / (sqrt_2pi * sig[:, None]))
        ds_dlsig = (-pol[:, None] * z * ez / sqrt_pi)
        J = cp.stack([sw, sw * xs, sw * s, sw * (c[:, None] * ds_dx0),
                       sw * (c[:, None] * ds_dlsig)], axis=2)
        JTJ = cp.einsum('bik,bij->bkj', J, J)
        JTr = cp.einsum('bik,bi->bk', J, r)
        diag = cp.maximum(cp.diagonal(JTJ, axis1=1, axis2=2), 1e-9)
        JTJ_d = JTJ + damp * diag[:, :, None] * eye5
        try:
            delta = cp.linalg.solve(JTJ_d, JTr[..., None]).squeeze(-1)
        except Exception:
            delta = cp.zeros((B, 5), dtype=cp.float64)
        bad = ~cp.all(cp.isfinite(delta), axis=1)
        delta = cp.where(bad[:, None], 0.0, delta)
        ok_gn &= ~bad
        delta[:, 3] = cp.clip(delta[:, 3], -0.75, 0.75)
        delta[:, 4] = cp.clip(delta[:, 4], -0.35, 0.35)
        a = a + delta[:, 0]
        b = b + delta[:, 1]
        c = cp.maximum(c + delta[:, 2], amp_floor)
        x0 = cp.clip(x0 + delta[:, 3], seed_g - max_shift_f, seed_g + max_shift_f)
        lsig = cp.clip(lsig + delta[:, 4], float(np.log(sigma_min)), float(np.log(sigma_max)))

    # Post-fit checks
    cond = cp.linalg.cond(JTJ_d)
    sig = cp.clip(cp.exp(lsig), sigma_min, sigma_max)
    z = (xs - x0[:, None]) / (sqrt_2 * sig[:, None])
    s = 0.5 * (1.0 + pol[:, None] * _erf_gpu(z))
    model = a[:, None] + b[:, None] * xs + c[:, None] * s
    w_fit = (cp.abs(xs - x0[:, None]) <= fit_rad_f).astype(cp.float64)
    n_fit = cp.maximum(cp.sum(w_fit, axis=1), 1.0)
    rms = cp.sqrt(cp.sum(w_fit * (prof_g - model) ** 2, axis=1) / n_fit)

    # Individual checks
    checks = {
        "gn_solve":     ok_gn,
        "cond_finite":  cp.isfinite(cond),
        "cond_ok":      cond < cond_max,
        "x0_finite":    cp.isfinite(x0),
        "sig_finite":   cp.isfinite(sig),
        "rms_finite":   cp.isfinite(rms),
        "shift_ok":     cp.abs(x0 - seed_g) <= (max_shift_f + 1e-6),
        "sigma_lo":     sig >= sigma_min,
        "sigma_hi":     sig <= sigma_max,
        "amp_ok":       c > amp_floor,
        "rms_ok":       rms <= (0.35 * dyn + 1e-6),
    }

    print(f"  {label} ({B} profiles):")
    for name, chk in checks.items():
        n_pass = int(cp.sum(chk).get())
        pct = 100.0 * n_pass / B
        flag = "  " if n_pass == B else "**"
        print(f"    {flag}{name:15s}: {n_pass:4d}/{B} ({pct:5.1f}%)")

    # Print summary stats for key values
    print(f"    -- rms  : median={float(cp.median(rms).get()):.4f}  "
          f"max={float(cp.max(rms).get()):.4f}  "
          f"threshold={float((0.35 * cp.median(dyn) + 1e-6).get()):.4f}")
    print(f"    -- sigma: median={float(cp.median(sig).get()):.4f}  "
          f"range=[{float(cp.min(sig).get()):.4f}, {float(cp.max(sig).get()):.4f}]")
    print(f"    -- shift: median={float(cp.median(cp.abs(x0 - seed_g)).get()):.4f}  "
          f"max={float(cp.max(cp.abs(x0 - seed_g)).get()):.4f}")
    print(f"    -- cond : median={float(cp.median(cond).get()):.2e}  "
          f"max={float(cp.max(cond).get()):.2e}")

    return cp.asnumpy(ok)


# ── Instrumented orchestration ──────────────────────────────────────

def instrumented_pipeline(img, stats):
    """Run full edge_erf pipeline with per-edge and per-check instrumentation."""
    K = stats.shape[0]
    gray_gpu = cp.asarray(img, dtype=cp.float32)
    stats_gpu = cp.asarray(stats, dtype=cp.float32)
    H, W = gray_gpu.shape

    band_rad, edge_rad, smooth_passes = 4, 10, 2
    loc_rad, grad_power = 3, 8.0

    w = stats_gpu[:, 2].astype(cp.float32)
    h = stats_gpu[:, 3].astype(cp.float32)
    xc = stats_gpu[:, 4].astype(cp.float64)
    yc = stats_gpu[:, 5].astype(cp.float64)
    By = 2 * band_rad + 1
    L = 2 * edge_rad + 1
    dy = cp.arange(-band_rad, band_rad + 1, dtype=cp.int32)
    dx = cp.arange(-edge_rad, edge_rad + 1, dtype=cp.int32)
    gflat = gray_gpu.ravel()

    # Compute edge positions
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
        for _ in range(smooth_passes):
            prof = _smooth_binom5_gpu(prof)
        base = x0_int.astype(cp.float64) - float(edge_rad)
        return prof.astype(cp.float64), base

    xs_band = cp.clip(cp.rint(xc).astype(cp.int32)[:, None] + dy[None, :], 0, W - 1)

    def prof_at_y(y0_int):
        ys2 = cp.clip(y0_int[:, None] + dx[None, :], 0, H - 1)
        idx2 = (ys2[:, None, :] * W + xs_band[:, :, None]).astype(cp.int64)
        patch2 = cp.take(gflat, idx2.ravel()).reshape(K, By, L)
        prof = cp.mean(patch2, axis=1)
        for _ in range(smooth_passes):
            prof = _smooth_binom5_gpu(prof)
        base = y0_int.astype(cp.float64) - float(edge_rad)
        return prof.astype(cp.float64), base

    # Extract profiles
    profL, baseL = prof_at_x(xL0)
    profR, baseR = prof_at_x(xR0)
    profT, baseT = prof_at_y(yT0)
    profB, baseB = prof_at_y(yB0)

    # Gradient-moment seeds
    xL_seed, okL0 = _edge_from_profile_gradmoment_gpu(profL, baseL.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
    xR_seed, okR0 = _edge_from_profile_gradmoment_gpu(profR, baseR.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
    yT_seed, okT0 = _edge_from_profile_gradmoment_gpu(profT, baseT.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)
    yB_seed, okB0 = _edge_from_profile_gradmoment_gpu(profB, baseB.astype(cp.float32), grad_power=grad_power, loc_rad=loc_rad)

    print("Gradient-moment seed pass rates:")
    for name, ok in [("L seed", okL0), ("R seed", okR0), ("T seed", okT0), ("B seed", okB0)]:
        n = int(cp.sum(ok).get())
        print(f"  {name}: {n}/{K} ({100*n/K:.1f}%)")

    # Instrumented erf fitting per edge
    print("\nErf fitting per-check breakdown:")
    ok_erf_L = instrumented_erf_edge(profL, baseL, xL_seed, label="Left edge")
    ok_erf_R = instrumented_erf_edge(profR, baseR, xR_seed, label="Right edge")
    ok_erf_T = instrumented_erf_edge(profT, baseT, yT_seed, label="Top edge")
    ok_erf_B = instrumented_erf_edge(profB, baseB, yB_seed, label="Bottom edge")

    # Combined
    okL0_np = cp.asnumpy(okL0)
    okR0_np = cp.asnumpy(okR0)
    okT0_np = cp.asnumpy(okT0)
    okB0_np = cp.asnumpy(okB0)
    ok_all = okL0_np & okR0_np & okT0_np & okB0_np & ok_erf_L & ok_erf_R & ok_erf_T & ok_erf_B

    print(f"\nCombined: {np.sum(ok_all)}/{K} centers pass all checks ({100*np.sum(ok_all)/K:.1f}%)")


# ── Compare with edge_gradmoment via public API ────────────────────

def compare_methods(img):
    """Run detect_centers with both methods and compare counts."""
    from subpx.centers import detect_centers

    r_gm = detect_centers(img, refine="edge_gradmoment", backend="gpu")
    r_erf = detect_centers(img, refine="edge_erf", backend="gpu")

    print(f"  edge_gradmoment: {len(r_gm.centers_xy)} centers")
    print(f"  edge_erf:        {len(r_erf.centers_xy)} centers")
    return r_gm, r_erf


if __name__ == "__main__":
    print("=== Diagnostic B: edge_erf integration test ===\n")

    img, centers_true, stats = make_grid_image()
    print(f"Synthetic image: {img.shape}, {len(centers_true)} features "
          f"(25x25 bright rects, pitch=50)\n")

    print("--- Method comparison (full pipeline) ---")
    compare_methods(img)

    print("\n--- Per-edge failure breakdown (iteration 1) ---")
    instrumented_pipeline(img, stats)

    print("\nDone.")
