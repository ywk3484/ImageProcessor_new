"""Example: Interactive GL viewer with overlays in a Jupyter notebook.

Run this in a Jupyter notebook (or IPython) with the Qt event loop enabled:

    %gui qt
    %run examples/viewer_overlay_example.py

Assumes `img` is a 2D grayscale NumPy array (or memmap) already loaded.
Replace the synthetic image below with your own data.
"""

import numpy as np
import subpx

# ---------------------------------------------------------------------------
# 0. Synthetic test image (replace with your real image)
# ---------------------------------------------------------------------------
H, W = 512, 1024
img = np.zeros((H, W), dtype=np.uint8)
rng = np.random.default_rng(42)

# Grid of bright squares to simulate photomask features
pitch_x, pitch_y = 32, 32
for cy in range(pitch_y // 2, H, pitch_y):
    for cx in range(pitch_x // 2, W, pitch_x):
        # Add slight jitter
        dx, dy = rng.integers(-1, 2, size=2)
        y0, y1 = max(0, cy + dy - 3), min(H, cy + dy + 4)
        x0, x1 = max(0, cx + dx - 3), min(W, cx + dx + 4)
        img[y0:y1, x0:x1] = 200

img += rng.integers(0, 15, size=img.shape, dtype=np.uint8)  # noise

# ---------------------------------------------------------------------------
# 1. Open the viewer
# ---------------------------------------------------------------------------
v = subpx.imshow(img, cmap="gray", divider_step=64, title="Overlay Demo")

# ---------------------------------------------------------------------------
# 2. Detect centers and overlay as scatter markers
# ---------------------------------------------------------------------------
res = subpx.detect_centers(
    img, backend="cpu", area_min=4, area_max=200, refine="weighted"
)
print(f"Detected {res.centers_xy.shape[0]} centers")

v.add_points(res.centers_xy, color="red", size=4, name="centers")

# ---------------------------------------------------------------------------
# 3. Add bounding boxes around a subset of centers
# ---------------------------------------------------------------------------
# Draw 10x10 pixel boxes centered on the first 20 detected features
subset = res.centers_xy[:20]
box_size = 10.0
rects = np.column_stack([
    subset[:, 0] - box_size / 2,  # x
    subset[:, 1] - box_size / 2,  # y
    np.full(len(subset), box_size),  # w
    np.full(len(subset), box_size),  # h
])
v.add_rects(rects, color="cyan", width=1.0, name="roi-boxes")

# ---------------------------------------------------------------------------
# 4. Add line segments (e.g., connecting nearest-neighbor pairs)
# ---------------------------------------------------------------------------
# Draw horizontal lines connecting consecutive centers in the first row
first_row = res.centers_xy[res.centers_xy[:, 1] < pitch_y]
first_row = first_row[first_row[:, 0].argsort()]  # sort by x

if len(first_row) >= 2:
    segments = np.stack([
        first_row[:-1],  # start points
        first_row[1:],   # end points
    ], axis=1)  # shape (M, 2, 2)
    v.add_lines(segments, color="yellow", width=1.5, name="row-links")

print("Overlays added. Pan/zoom the viewer to inspect.")

# ---------------------------------------------------------------------------
# 5. Managing overlays
# ---------------------------------------------------------------------------
# Remove a specific overlay by name:
#   v.remove_overlay("roi-boxes")
#
# Remove all overlays:
#   v.clear_overlays()
#
# Replace an overlay (same name auto-removes the old one):
#   v.add_points(new_centers, color="green", size=6, name="centers")
#
# Use custom RGBA colors:
#   v.add_points(xy, color=(0.2, 0.8, 0.2, 0.5))  # semi-transparent green
