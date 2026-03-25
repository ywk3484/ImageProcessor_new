#!/usr/bin/env python3
"""Diagnostic A — Unit test of _edge_from_profile_erf_gpu on synthetic erf profiles.

Tests the fitter directly with known-good inputs. If this fails, the bug is
in the fitting function itself. If it passes, look at orchestration/validation.

Run:  python tests/diag_erf_unit.py
"""
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise SystemExit("CuPy required — run on GPU workstation")

from scipy.special import erf as scipy_erf

# --- Import internals ---
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from subpx._gpu.centers import (
    _edge_from_profile_erf_gpu,
    _edge_polarity_from_profile_gpu,
)


def make_erf_profile(edge_pos, sigma, amp, baseline, n_pts=21, x_start=0.0):
    """Create a perfect 1D erf edge profile (rising)."""
    xs = np.arange(n_pts, dtype=np.float64) + x_start
    z = (xs - edge_pos) / (np.sqrt(2.0) * sigma)
    prof = baseline + amp * 0.5 * (1.0 + scipy_erf(z))
    return prof, x_start


def run_test(label, edge_pos, sigma, amp, baseline, n_pts=21, x_start=0.0, seed_offset=0.0):
    """Run the erf fitter on one synthetic profile and report results."""
    prof_np, base = make_erf_profile(edge_pos, sigma, amp, baseline, n_pts, x_start)

    # Shape: (1, L) batch of 1
    prof_gpu = cp.asarray(prof_np[None, :], dtype=cp.float64)
    base_gpu = cp.asarray(np.array([base]), dtype=cp.float64)
    seed_gpu = cp.asarray(np.array([edge_pos + seed_offset]), dtype=cp.float64)

    x_out, ok = _edge_from_profile_erf_gpu(prof_gpu, base_gpu, seed_gpu)
    x_val = float(cp.asnumpy(x_out)[0])
    ok_val = bool(cp.asnumpy(ok)[0])
    err = abs(x_val - edge_pos) if ok_val else float('nan')

    status = "PASS" if ok_val else "FAIL"
    print(f"  {label}: {status}  x_fit={x_val:.4f}  x_true={edge_pos:.4f}  err={err:.6f}  ok={ok_val}")

    # If failed, run detailed breakdown
    if not ok_val:
        _detailed_breakdown(prof_gpu, base_gpu, seed_gpu)

    return ok_val


def _detailed_breakdown(prof, base_x, x_seed):
    """Re-run key validation checks and report which ones fail."""
    from subpx._gpu.centers import _erf_gpu

    prof = prof.astype(cp.float64)
    base_x = base_x.astype(cp.float64)
    x_seed = x_seed.astype(cp.float64)
    B, L = prof.shape
    xs = base_x[:, None] + cp.arange(L, dtype=cp.float64)[None, :]
    pol = _edge_polarity_from_profile_gpu(prof, base_x, x_seed, loc_rad=2)

    ymin = cp.min(prof, axis=1)
    ymax = cp.max(prof, axis=1)
    amp_floor = cp.maximum((ymax - ymin) * 1e-6, 1e-9)
    dyn = cp.maximum(ymax - ymin, 1e-9)

    sigma_min, sigma_max = 0.35, 6.0
    max_shift = 2.5
    init_sigma = 1.25
    fit_rad = 5.0
    sqrt_2 = float(np.sqrt(2.0))

    # Run the fitter to get final params (just re-do a simpler version)
    a = ymin.copy()
    b = cp.zeros((B,), dtype=cp.float64)
    c = cp.maximum(ymax - ymin, amp_floor)
    x0 = x_seed.copy()
    lsig = cp.full((B,), float(np.log(init_sigma)), dtype=cp.float64)

    # After 5 iterations using full fitter — just call it and inspect
    x_out, ok = _edge_from_profile_erf_gpu(prof, base_x, x_seed)

    # Re-compute final state params manually for diagnostics
    # We can't easily extract intermediate state, so compute post-fit metrics
    # by calling the fitter then checking what the outputs give us
    x0_val = float(cp.asnumpy(x_out)[0])
    seed_val = float(cp.asnumpy(x_seed)[0])

    print(f"    polarity: {float(cp.asnumpy(pol)[0]):.1f}")
    print(f"    ymin={float(cp.asnumpy(ymin)[0]):.2f}  ymax={float(cp.asnumpy(ymax)[0]):.2f}  "
          f"dyn={float(cp.asnumpy(dyn)[0]):.2f}")
    print(f"    x0={x0_val:.4f}  seed={seed_val:.4f}  shift={abs(x0_val - seed_val):.4f}  "
          f"max_shift={max_shift}")
    print(f"    x0_finite={np.isfinite(x0_val)}  "
          f"shift_ok={abs(x0_val - seed_val) <= max_shift + 1e-6}")


if __name__ == "__main__":
    print("=== Diagnostic A: erf fitter unit test ===\n")

    print("1. Ideal erf profiles (should all PASS):")
    run_test("rising, σ=1.5, center", edge_pos=10.0, sigma=1.5, amp=100.0, baseline=50.0)
    run_test("rising, σ=2.5, center", edge_pos=10.0, sigma=2.5, amp=100.0, baseline=50.0)
    run_test("rising, σ=0.8, sharp",  edge_pos=10.0, sigma=0.8, amp=200.0, baseline=20.0)
    run_test("falling (low→high→low)", edge_pos=10.0, sigma=1.5, amp=-100.0, baseline=150.0)

    print("\n2. Seed offset tests (robustness to imperfect seeds):")
    run_test("seed +0.5px off", edge_pos=10.0, sigma=1.5, amp=100.0, baseline=50.0, seed_offset=0.5)
    run_test("seed +1.5px off", edge_pos=10.0, sigma=1.5, amp=100.0, baseline=50.0, seed_offset=1.5)
    run_test("seed -1.0px off", edge_pos=10.0, sigma=1.5, amp=100.0, baseline=50.0, seed_offset=-1.0)

    print("\n3. Edge cases:")
    run_test("low contrast (amp=5)",  edge_pos=10.0, sigma=1.5, amp=5.0, baseline=50.0)
    run_test("wide sigma (σ=5.0)",    edge_pos=10.0, sigma=5.0, amp=100.0, baseline=50.0)
    run_test("narrow (σ=0.4)",        edge_pos=10.0, sigma=0.4, amp=100.0, baseline=50.0)

    print("\nDone.")
