"""Scratch: the fast seed finder on every generated pattern: found against true holes, duplicates, misses."""
import sys, time
import numpy as np, cupy as cp
from scipy.spatial import cKDTree
from gpufit import find_seeds, sync, ROOT
sys.path.insert(0, str(ROOT))
from scripts.pitch_signature_test import PATTERNS
from truth_gpu import render, DEFAULTS
for name in PATTERNS:
    page, image, spec = render(name, (1378, 2048), dict(DEFAULTS))
    img = cp.asarray(image).astype(cp.float32)
    find_seeds(img); sync(); t0 = time.perf_counter()
    s = cp.asnumpy(find_seeds(img)); sync(); dt = time.perf_counter() - t0
    truth = page.centers_xy
    h, w = image.shape
    inside = (truth[:, 0] > 3) & (truth[:, 0] < w - 4) & (truth[:, 1] > 3) & (truth[:, 1] < h - 4)
    d, j = cKDTree(s).query(truth[inside]) if len(s) else (np.full(inside.sum(), np.inf), None)
    pitch = min(spec["pitch_nm"]) / 21.0
    missed = int((d > 0.4 * pitch).sum())
    d2, _ = cKDTree(truth).query(s) if len(s) else (np.zeros(0), None)
    stray = int((d2 > 0.4 * pitch).sum())
    nn = cKDTree(s).query(s, k=2)[0][:, 1] if len(s) > 1 else np.zeros(0)
    dup = int((nn < 0.5 * pitch).sum())
    print(f"{name:<24s} true inside {inside.sum():6d}, found {len(s):6d}, missed {missed:5d}, stray {stray:5d}, closer than half a pitch {dup:5d}, "
          f"error sd {np.std(d[d < 0.4 * pitch]) * 21:.2f} nm, {dt * 1e3:.1f} ms")
