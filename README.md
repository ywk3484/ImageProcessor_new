# subpx

Notebook-friendly subpixel image-processing library with transparent CPU/GPU backend dispatch.

Python 3.10+ | NumPy-only core | Optional GPU acceleration via CuPy

## Features

| Module | What it does |
|---|---|
| `detect_centers` | Otsu threshold, connected components, subpixel refinement (weighted / log-quadratic / edge gradient-moment) |
| `measure_cds` | Voronoi cell extraction, Gaussian/gradient/half-height contours, CD maps, and CSV/NPZ exports |
| `estimate_pitch` | KNN or index-regression pitch estimation, global and per-line |
| `estimate_shift` | Phase cross-correlation image registration |
| `fft_pitch_error` | FFT-based periodic error / spectral analysis |
| `fit_rowwise_distortion_field` | Row/column clustering, residual maps, distortion field fitting |
| `build_tiff_index` | TIFF stripe/board parsing with global coordinate mapping |
| `imshow_huge` | OpenGL tiled viewer for large images (PyQt5/6 + pyqtgraph) |

All functions return dataclass results (`CenterResult`, `PitchResult`, `ShiftResult`, etc.) with NumPy arrays and metadata. Coordinates are `(N, 2)` in `[x, y]` order.

## Install

```bash
pip install -e .
```

Optional dependencies for full functionality:

```bash
pip install opencv-python scikit-image scipy tifffile matplotlib
pip install cupy-cuda12x   # for GPU acceleration
pip install PyQt5 pyqtgraph pyopengl  # for the viewer
```

### Contact-hole contours and CD maps

Run the contour/CD pipeline on a grayscale image:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_contact_holes.py image.tif
```

Results in `artifacts/contact_holes_cd` include CD and mean-intensity colorbar
maps, an offline interactive `contours.html` viewer, per-cell measurements,
every contour vertex, and the extracted cell arrays.
The three methods are Gaussian FWHM (`logquad`), radial gradient maxima, and
half-height contours. See [CD pipeline usage and definitions](docs/contact_holes_cd.md)
for physical calibration, other images, notebook usage, and accuracy measurements.
For your own images on a remote Windows server, see the
[remote pipeline guide](docs/remote_server.md).

## Quick start

```python
from subpx import detect_centers, estimate_pitch, estimate_shift

# Detect photomask feature centers with GPU-accelerated subpixel refinement
result = detect_centers(img, backend="gpu", refine="logquad")
centers = result.centers_xy          # (N, 2) array in [x, y] order

# Estimate pitch along each row
pitch = estimate_pitch(centers)

# Register two images
shift = estimate_shift(ref_img, mov_img)
print(shift.shift_yx)                # [dy, dx] in pixels
```

### Tiled processing for large images

```python
from subpx import detect_centers_tiled

result = detect_centers_tiled(huge_image, tile_h=8192, overlap=128)
```

### Backend selection

Functions with CPU/GPU implementations accept `backend=`:
- `"gpu"` — CuPy-accelerated (default for center detection and tiled functions)
- `"cpu"` — OpenCV / scikit-image
- `"auto"` — GPU if CuPy is available, else CPU

CD measurements run on the CPU; `measure_cds(..., cell_backend="gpu")` selects
GPU acceleration for the Voronoi partition.

## Tests

```bash
pytest tests/ -v
```

## Project structure

```
subpx/
  __init__.py          # Public API re-exports
  centers.py           # Center detection (public API)
  pitch.py             # Pitch estimation
  registration.py      # Image registration
  spectra.py           # FFT spectral analysis
  calibration.py       # Distortion field fitting
  mosaic.py            # TIFF stripe/board parsing
  batch.py             # Stack processing wrappers
  viewer.py            # OpenGL tiled image viewer
  types.py             # Result dataclasses
  backends.py          # CPU/GPU dispatch logic
  legacy.py            # Deprecation wrappers for old API
  _cpu/                # CPU implementations
  _gpu/                # GPU implementations (hybrid CPU segmentation + GPU refinement)
```
