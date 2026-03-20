
# Migration Map

## Old names -> New names
- `find_subpixel_centers_otsu_opencv` -> `detect_centers(...).centers_xy`
- `refine_center_weighted_centroid` -> `refine_centers(..., method="weighted")`
- `refine_center_log_quadratic` -> `refine_centers(..., method="logquad")`
- `filter_centers_by_margin` -> `filter_centers(...)`
- `cluster_centers_radius` -> `dedupe_centers(...)`
- `estimate_pitch_knn` -> `estimate_pitch(...)["initial"]`
- `pitch_per_line` -> `estimate_pitch_lines(...)`

## Import policy for old notebooks
During migration, old notebooks may temporarily use:
```python
from maskproc.legacy import *
```

New notebooks should instead use:
```python
from maskproc import detect_centers, estimate_pitch, estimate_shift, imshow_huge
```


## Added GPU legacy wrappers
- `find_centers_hybrid_gpu_cpu_logquad` -> `detect_centers(..., backend="gpu", refine="logquad")`
- `find_subpixel_centers_tiled_logquad_gpu` -> `detect_centers(..., backend="gpu", refine="logquad")`
