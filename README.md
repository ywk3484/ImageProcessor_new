# (ADD A SUITABLE TITLE FOR PROJECT)

A cleaned notebook-facing package for photomask image processing with a stable
public API and internal CPU/GPU backend dispatch.

## Goals
- short task-based public API names
- stable import paths
- NumPy-first outputs for notebook use
- legacy wrappers for old function names during migration
- private `_cpu` / `_gpu` modules behind one public API

## Current scope
Implemented from the current working source plus the project chat history summary:
- viewer module migrated from `gl_image_viewer.py`
- center detection with CPU connected-components and optional GPU refinement
- pitch estimation (global and per-line)
- registration API with CPU or CuPy FFT backend
- lightweight batch helpers and batch convenience wrappers

## Important note on GPU migration
The historical GPU functions were not available here as verbatim source files, so the
GPU layer was rebuilt under the new structure rather than copied verbatim. The public
API is stable; internal kernels can keep evolving without changing notebook code.

## Install locally
```bash
pip install -e /path/to/maskproc_package
```

## Example
```python
from maskproc import detect_centers, estimate_pitch, imshow_huge

viewer = imshow_huge(img)
res = detect_centers(img, backend="gpu", refine="logquad")
centers = res.centers_xy
pitch = estimate_pitch(centers)
```
