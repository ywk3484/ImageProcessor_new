# Contact-hole contours and CD maps

`scripts/analyze_contact_holes.py` extends the workflow in
`artifacts/contact_holes_diagnosis`: extract each cell once, measure it with
several edge definitions, and export spatial maps, contours, and numerical data.
The reusable API is `measure_cds`, `plot_cd_map`, `save_cd_results`, and
`save_cd_viewer`.

## Run the pipeline

From the repository root, using the existing Windows environment:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_contact_holes.py
```

The default image is `examples/data/synthetic_ch/contact_holes.png`; results
go to `artifacts/contact_holes_cd`. OpenCV and SciPy perform the analysis;
Matplotlib and Pillow/tifffile handle plotting and image loading. For a fresh
environment, install `pip install -e ".[analysis]"`.

To use the original diagnostic's GPU Voronoi partition and compare against the
supplied synthetic ground truth:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_contact_holes.py --cell-backend gpu --ground-truth examples/data/synthetic_ch/ground_truth.npz
```

For another image, including an isotropic physical calibration:

```powershell
.\.venv\Scripts\python.exe scripts/analyze_contact_holes.py image.tif --output artifacts/my_cd --invert --area-max 500 --pad 6 --pixel-size 2.5 --unit nm
```

Use `--invert` only for dark holes on a bright background. The script accepts
single-channel images and single TIFF pages. To select methods, pass e.g.
`--methods logquad gradient`. The map quantity can be `--metric cd_equivalent_px`,
`cd_x_px`, `cd_y_px`, or `area_px2`. Each method normally has its own color range
to reveal spatial variation; `--shared-color-scale` uses a common range.

The pipeline also writes a separate `intensity_map.png`, `cells.csv`, and the
offline interactive viewer `contours.html` automatically.

## Interactive contour inspection

Open `artifacts/contact_holes_cd/contours.html` in a browser. It contains its
own data and JavaScript and needs no server, internet connection, or additional
Python plotting package.

- Scroll to zoom around the pointer, drag to pan, and use **Fit image** to reset.
- Select **Contour overlay**, **CD cell map**, or **Mean intensity map** in View.
- Choose a method and CD metric, or enable **Compare all contours** to overlay
  the three methods with the same colors as `cell_examples.png`.
- Click a cell to see its intensity and all method measurements. Enter a cell
  ID and choose **Zoom to selected cell** to inspect a specific example.
- Enable **Show measured vertices** to see the individual contour samples when
  sufficiently zoomed in. **Smooth background display** affects the raster's
  appearance only; it does not recalculate gradients or CDs.

The viewer redraws contour paths at the current zoom, so enlarging it does not
enlarge a previously rasterized contour image. The source image still has its
original pixel resolution. Packed local float32 drawing coordinates keep the
HTML size manageable (about 14 MB for this sample); CSV/NPZ retain the original
float64 measurement coordinates. The viewer always scales each method's CD
colors separately. No source-image data is sent to an external service.

## Mean cell intensity

`result.mean_intensity` is the arithmetic mean of **original input pixels in
the full Voronoi cell**, including its background. This matches the displayed
cell partition. It excludes ROI padding/fill and is independent of CD fit
validity, polarity inversion, smoothing, background subtraction, and physical
pixel calibration. Values keep the input image's original intensity units.

`intensity_map.png` uses the same spatial layout and colorbar format as the CD
maps. Larger means are brighter/yellower. It has its own intensity color scale.
The values are saved once per cell in `cells.csv`, repeated for convenience in
`measurements.csv`, and included as `mean_intensity` in `raw_data.npz`.

Outer Voronoi cells extend to the image edge and may contain more background.
Their mean intensity can therefore be lower even for equally bright holes.
This quantity is neither the peak intensity nor an average over the hole contour
or the cropped measurement ROI.

## Cell extraction

1. Apply the same Otsu/triangle segmentation and 8-connected labeling used by
   `scripts/diagnose_radial_symmetry.py`.
2. Keep components within the inclusive area limits, default 3–50 pixels, and
   apply the diagnostic's same image-edge exclusion.
3. Partition the image using the coarse component centroids as Voronoi seeds.
4. Extract the component bounding box plus `pad=3` pixels, restrict it to its
   Voronoi cell, and fill outside the mask with the tenth percentile of the
   cell-border intensity. The helper is shared with the existing diagnostic.

The default CPU partition uses an exact nearest-neighbor tree with deterministic
ties. `--cell-backend gpu` uses the existing GPU implementation. Its float32
distance/tie behavior can differ at boundaries; its previously documented
sparse/clustered-seed limitation remains described in [validation](validation.md).
Contour fitting runs on the CPU for both choices.

Measurements retain original image intensities. For dark holes the working
intensity is negated so all methods operate on positive contrast above a local
background. Float inputs (and uint16 with triangle thresholding) are scaled to
uint8 **only for segmentation**. uint8 and uint16 Otsu inputs retain their native
segmentation behavior.

## Methods and definitions

| Method | Contour definition | Useful outputs |
|---|---|---|
| `logquad` | Half-height ellipse of a fitted 2D Gaussian above local background | Principal-axis sigma and FWHM, orientation, fit residual, ellipse area |
| `gradient` | Connected subpixel maxima of the outward radial intensity decrease | Measured edge polygon, area, horizontal/vertical extent, ray coverage |
| `halfmax` | First outward crossing of half the interpolated central signal above background along each ray | Measured half-height polygon, area, horizontal/vertical extent, ray coverage |

For `logquad`, fit `log(I - background)` to a quadratic with an `xy` term.
Only pixels above `fit_fraction=0.2` of the measured peak enter the fit, and
intensity weights reduce sensitivity to noisy tails. The inverse negative
Hessian gives the Gaussian covariance. Its eigenvalues give squared principal
sigmas, and **FWHM = 2 sqrt(2 ln 2) sigma**. The ellipse may be rotated.
`angle_deg` is the major-axis angle modulo 180 degrees, measured from +x toward
+y in image coordinates; it is unstable for nearly circular spots.

The radial methods start at an intensity-weighted center, sample cubic-interpolated
profiles at `radial_step=0.1` pixels along `n_angles=128` directions, and connect
the resulting edge points in angular order. Gradient maxima receive parabolic
subpixel refinement; half-height crossings receive linear refinement between
ray samples. The half-height peak is the interpolated signal at that weighted
center. A one-pixel mask erosion excludes the background-filled cell boundary
from the search. `--smooth-sigma` optionally smooths the source of the radial
profiles; the default is zero. Smoothing changes measured widths.

The gradient method evaluates the cubic-interpolated image directly at the ray
coordinates; it does not first create a fixed 4x-upsampled image. It computes
the outward **radial derivative**, finds each ray's strongest decrease, refines
that maximum, and joins the vertices. There is no angular smoothing or ellipse
constraint on those vertices. Independent maxima can produce uneven or flattened
segments when small changes in direction select different derivative peaks.

The supplied holes have only about three pixels across their FWHM. Noise,
pixel-grid sampling, and cubic interpolation therefore affect gradient maxima
noticeably. In example cells 2521, 644, and 1335, changing the ray step from
0.1 to 0.025 px moved vertices by at most 0.0164, 0.0125, and 0.0198 px,
respectively: finer radial sampling preserves the unusual shapes. In cell 1335,
the gradient CD is 2.9606 px on the noisy image versus 2.5830 px on the
noise-free source. `artifacts/contact_holes_cd/gradient_inspection.json` records
these checks. No gradient estimator or smoothing default was changed for the
intensity-map/viewer update. `--smooth-sigma 0.5` is available for comparison,
but changes the measured edge and does not guarantee a more accurate CD.

For an ideal Gaussian, the radial gradient peaks at **one sigma**, giving a
diameter of **2 sigma**, smaller than FWHM. These methods define different CDs;
their absolute values should not be treated as interchangeable. The Gaussian
fit assumes a Gaussian profile; the radial methods assume a single feature whose
boundary can be reached from its center along every ray. Neither is a general
segmentation method for disconnected or strongly concave shapes.

The common metrics are:

- `area_px2`: analytic half-height ellipse area for `logquad`, closed-polygon
  area for the radial methods.
- `cd_equivalent_px`: diameter of a circle with the same contour area,
  `2 sqrt(area / pi)`. This is the default map quantity.
- `cd_x_px`, `cd_y_px`: horizontal and vertical contour extents. These are
  bounding widths, not central chords or principal-axis widths.
- `perimeter_px`: length of the sampled closed polygon.

`logquad` also exports `sigma_major_px`, `sigma_minor_px`, `fwhm_major_px`,
`fwhm_minor_px`, `angle_deg`, and the intensity-weighted `log_fit_rmse`.
The radial methods export `ray_coverage`. All methods export `background` and
`peak_signal` in the working polarity's intensity units.

Cells keep stable zero-based IDs shared by all methods. The original
connected-component ID is retained separately. Degenerate fits, contours that
leave the cell, and rays without an observable edge produce `valid=False` and
an explicit status. Common geometry metrics are NaN for failed measurements;
partial/candidate contours and fit diagnostics can remain for inspection.
Validity indicates a completed numerical measurement, not a calibrated accuracy
guarantee. Low signal, background estimation, sampling, or non-Gaussian profiles
can bias a valid result. Increasing angular/radial sampling does not add optical
resolution. For clipped contours, increasing `--pad` may help if the neighboring
Voronoi boundary permits it.

## Files and coordinates

| File | Contents |
|---|---|
| `cd_maps.png` | One colorbar map per method; each Voronoi cell carries its hole's CD/area |
| `intensity_map.png` | Separate colorbar map of the original mean intensity of each full Voronoi cell |
| `contours.png` | Subpixel contours over the source image, colored by CD/area |
| `contours.html` | Offline interactive contours, CD/intensity maps, zoom/pan, and per-cell inspection |
| `cd_distributions.png` | Distributions of area-equivalent CD and their medians |
| `cell_examples.png` | Smallest, median, and largest valid cells from the first successful method, with contours overlaid |
| `measurements.csv` | One row per cell and method, including failures, metrics, IDs, centers, and calibration |
| `cells.csv` | One row per cell: ID, centroid, Voronoi area, and mean intensity |
| `contours.csv` | One row per contour vertex with cell ID, method, validity, and coordinates; the last vertex repeats the first |
| `raw_data.npz` | Source image, binary/CC/Voronoi labels, component IDs/bboxes, seeds, mean intensities, ROI origins, masked ROI stacks, all contours/metrics/status arrays |
| `manifest.json` | Input path/hash, parameters, definitions, counts, summary statistics, timing, and file inventory |
| `accuracy.json`, `truth_comparison.csv` | Optional comparisons to synthetic Gaussian ground truth |

Maps use the full Voronoi partition as a spatial display, including the outer
cells extending to the image edge. The colored region is the cell, not the hole's
measured area. Invalid measurements are gray in maps and marked with crosses in
overlays. `cell_areas_px2` records Voronoi areas; hole areas are per-method
`area_px2` arrays.

All NumPy coordinates and geometric metrics are in original pixels. Points are
`[x, y]`, images are indexed `[y, x]`, and ROI origins are also `[x, y]`. Contour
arrays have shape `(N, n_angles + 1, 2)`. To recover a local contour, subtract
`origins_xy[:, None, :]`. Masked ROI values are polarity-normalized; the original
source image is also saved.

`pixel_size` is an isotropic length-per-pixel calibration. CSV columns suffixed
`_px`/`_px2` retain pixels; unsuffixed coordinates/CDs/perimeters use `unit`, and
unsuffixed `area` uses `unit²`. Plotted colorbars use the physical calibration;
plot axes remain in pixels. With no calibration, both are pixels. NPZ archives
contain numeric/string arrays and load without pickle:

```python
import numpy as np

with np.load("artifacts/contact_holes_cd/raw_data.npz", allow_pickle=False) as raw:
    valid = raw["logquad_valid"]
    cd_px = raw["logquad_cd_equivalent_px"][valid]
    contours_xy = raw["logquad_contours_xy"][valid]
```

For interactive notebook plots and selective exports:

```python
from subpx import measure_cds, plot_cd_map, save_cd_results, save_cd_viewer

result = measure_cds(img, methods=("logquad", "gradient", "halfmax"))
ax = plot_cd_map(result, method="gradient", contours=True)
ax.set_xlim(200, 240)
ax.set_ylim(140, 100)
save_cd_results(result, "artifacts/my_cd")
plot_cd_map(result, metric="mean_intensity")
save_cd_viewer(result, "artifacts/my_cd/contours.html")
```

## Validation on the supplied image

On `examples/data/synthetic_ch/contact_holes.png`, with the original diagnostic's
GPU cell partition and default measurement parameters, all three methods
completed **2,805 / 2,805** cells. All cells matched the supplied ground truth
uniquely within one pixel.

The extracted ROI values, masks, and origins were compared with the original
implementation and were identical. On this image the CPU and GPU partitions
and all measured CDs also agreed exactly. Export validation confirmed 8,415
measurement rows and 1,085,535 contour-vertex rows.

| Method | Median equivalent CD (px) | Bias (px) | CD RMSE (px) | 95th percentile absolute error (px) |
|---|---:|---:|---:|---:|
| Gaussian FWHM | 3.0337 | +0.0338 | 0.0515 | 0.0967 |
| Radial gradient | 2.5985 | +0.0512 | 0.1171 | 0.2254 |
| Half height | 3.0413 | +0.0414 | 0.0638 | 0.1195 |

The comparison uses Gaussian FWHM ground truth for `logquad`/`halfmax` and
Gaussian `2 sigma` ground truth for `gradient`, with geometric means for
elliptical holes. These results describe this noisy synthetic sample; the small
positive bias includes the low-percentile background estimate and interpolation.

`tests/test_cd.py` checks rotated Gaussian widths/orientation, analytic edge
definitions, coordinate offsets, polarity/intensity invariance, spatial size
variation, blurred disks, incomplete contours, calibrated exports, empty images,
CPU/GPU cell parity, parameter validation, and the command-line workflow.

Validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cd.py tests/test_synthetic.py tests/test_visualization.py tests/test_smoke.py -q
```

This completed with **36 passed**, including the GPU parity check, with no
skipped tests. The earlier full-suite run completed with **126 passed, 10 failed,
1 skipped**; its failing test names match the existing [baseline](validation.md).
During that run the known failing tiled-GPU logquad test encountered a
`MemoryError`. After the interruption, validation and artifact generation were
run sequentially. The entire suite was not repeated.

The added tests check means against original pixels (including dark-hole
polarity and failed CD fits), intensity exports, and reconstruction of packed
HTML contours. A dependency-free JavaScript diagnostic exercises the actual
viewer code with DOM/canvas test doubles:

```powershell
node tests/diag_cd_viewer.cjs artifacts/contact_holes_cd/contours.html
```

It checks contour paths, intensity map colors, metric switching, method
comparison, pan/zoom/reset, selection, and vertex display. Automated opening of
the local HTML file was blocked by the browser URL policy, so this is code-level
interaction validation rather than a rendered-browser check.
