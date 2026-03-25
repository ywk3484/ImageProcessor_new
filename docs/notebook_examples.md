
# Notebook Examples

## Huge image viewer with overlays
```python
%gui qt
import subpx

v = subpx.imshow(img, cmap="gray", divider_step=64, title="my image")

# Scatter markers from detected centers
v.add_points(centers_xy, color="red", size=4)

# Bounding boxes: each row is [x, y, w, h]
v.add_rects(bboxes, color="cyan", width=1.0, name="roi")

# Line segments: shape (M, 2, 2), each [[x0,y0], [x1,y1]]
v.add_lines(segments, color="yellow", width=1.5)

# Remove overlays
v.remove_overlay("roi")   # by name
v.clear_overlays()         # all
```

See `examples/viewer_overlay_example.py` for a complete runnable workflow.

## Center detection
```python
from subpx import detect_centers
res = detect_centers(img, refine="logquad", area_min=1, area_max=50)
centers = res.centers_xy
```

## Global pitch
```python
from subpx import estimate_pitch
pitch = estimate_pitch(centers, k=8, theta_deg=15)
pitch_x = pitch["refined"]["pitch_x"] if pitch["refined"] else pitch["initial"]["pitch_x"]
```

## Registration
```python
from subpx import estimate_shift, apply_shift
shift = estimate_shift(ref, mov, upsample_factor=20)
mov_aligned = apply_shift(mov, shift.shift_yx)
```
