"""Example: GPU-accelerated pitch estimation on a photomask stripe image.

Assumes `stripe` is a 2D NumPy array (grayscale) already loaded in memory.
Adjust detect_centers parameters (area_min, area_max, refine) to match your feature size.
"""

import numpy as np
from subpx.centers import detect_centers
from subpx.pitch import estimate_pitch_lines

# ---------------------------------------------------------------------------
# 1. Detect centers
# ---------------------------------------------------------------------------
# Adapt these parameters to your image:
#   area_min / area_max : expected blob area range in pixels
#   refine              : "weighted" (fast), "logquad" (accurate), "edge_gradmoment" (large features)
#   backend             : "auto" picks GPU if CuPy is available, "cpu" forces CPU
centers = detect_centers(
    stripe,
    backend="auto",
    area_min=4,
    area_max=200,
    refine="logquad",
)
print(f"Detected {centers.centers_xy.shape[0]} centers (backend={centers.backend})")

# ---------------------------------------------------------------------------
# 2. Estimate pitch per row (GPU-accelerated)
# ---------------------------------------------------------------------------
# tol            : how close (in pixels) points must be in y to belong to the same row
# pitch_var_max  : reject rows where spacing varies by more than this fraction (0.25 = 25%)
# min_points_per_line : rows with fewer points are skipped
pitch = estimate_pitch_lines(
    centers.centers_xy,
    line_axis="row",
    tol=2.0,
    min_points_per_line=5,
    pitch_var_max=0.25,
    backend="auto",
)

valid = pitch.values[np.isfinite(pitch.values)]
print(f"Rows found: {pitch.values.size}, valid: {valid.size} (backend={pitch.backend})")
print(f"Median pitch: {np.median(valid):.4f} px")
print(f"Std across rows: {np.std(valid):.4f} px")

# ---------------------------------------------------------------------------
# 3. Inspect per-row details
# ---------------------------------------------------------------------------
row_centers = pitch.meta["line_centers_perp"]    # y-position of each row
row_counts = pitch.meta["line_counts"]           # number of points per row
valid_mask = pitch.meta["valid_mask"]            # True if row passed filters

print(f"\n{'Row':>4}  {'Y pos':>8}  {'Count':>5}  {'Pitch':>8}  {'Valid'}")
for i in range(pitch.values.size):
    print(
        f"{i:4d}  {row_centers[i]:8.2f}  {row_counts[i]:5d}  "
        f"{pitch.values[i]:8.4f}  {'yes' if valid_mask[i] else 'no'}"
    )

# ---------------------------------------------------------------------------
# 4. Column pitch (optional)
# ---------------------------------------------------------------------------
# To measure pitch along columns instead of rows:
pitch_col = estimate_pitch_lines(
    centers.centers_xy,
    line_axis="col",
    tol=2.0,
    backend="auto",
)
valid_col = pitch_col.values[np.isfinite(pitch_col.values)]
if valid_col.size > 0:
    print(f"\nColumn pitch: {np.median(valid_col):.4f} px")
