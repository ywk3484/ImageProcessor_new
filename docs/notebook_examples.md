
# Notebook Examples

## Huge image viewer
```python
from subpx import imshow_huge
viewer = imshow_huge(img, divider_step=64, stripe_width=64)
```

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
