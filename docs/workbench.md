# Desktop inspection workbench

The PyQt6 application runs the repository's existing analysis methods and uses
pyqtgraph for every image, overlay and diagnostic plot. It is separate from the
OpenGL tiled notebook viewer in `subpx.viewer`.

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m subpx.workbench --demo
.\.venv\Scripts\python.exe -m subpx.workbench path\to\image.tif
```

For a fresh installation:

```powershell
python -m pip install -e ".[analysis,viewer]"
subpx-workbench
```

![Workbench showing Voronoi cells, CD contours and a selected radial sample](images/workbench.png)

The left panel holds settings; the center holds the image and diagnostic tabs;
the right panel compares methods for the selected feature. Drag the splitters
to give an image or diagnostic more room, or use **Expand** beside the diagnostic
tabs. The toolbar's **Guide** explains the
workflow inside the application.

## Image review

- Open TIFF, PNG, JPEG, BMP or a two-dimensional `.npy` array. TIFF pages are
  selected individually with the one-based **Page** control. A multichannel
  scientific stack must first be reduced to a two-dimensional image.
- Grayscale data retain their original precision. RGB is converted to luminance
  with `0.2126 R + 0.7152 G + 0.0722 B`; alpha is ignored. The conversion is
  displayed beside the image name. Nonfinite and complex images are rejected.
  Boolean arrays are converted to uint8 with their 0/1 intensities preserved.
- Wheel to zoom, drag to pan, press **F** to fit and **Z** to zoom to the selected
  feature. Hover reports image coordinates and the original pixel intensity.
  The histogram to the right adjusts display contrast without changing data or
  measurements. Layer toggles preserve the chosen contrast.
- Toggle **Analysis region**, then move or resize the gold rectangle. The next
  analysis uses only that crop. Its boundaries can clip features, affect seed
  detection and exclude cells; review the region edges. Pixel centers have
  integer coordinates, x is the column and y increases down the image.
- Set **Length / pixel** and **Length unit** together before measurement. The UI
  displays calibrated lengths and areas; original repository result arrays
  retain pixel units. CPU is the default; Auto and GPU use existing dispatch.

## Find patterns

Set threshold, polarity, component area limits and center refinement, then click
**Find patterns**. Switch the image view to **Threshold mask** or **Component
labels** and enable **Rejected boxes** to see components outside the area limits.
Click the image or a measurement-table row to select a feature. The inspector
shows seed and refined coordinates; the lower tab shows original pixels,
segmentation and the center-row intensity.

Detection input is rescaled to uint8 when required by OpenCV (uint16 Otsu is
retained). The Activity tab records this conversion. This rescaling does not
replace the original image used for CD.

## CD and intermediate steps

Select methods in the **CD** settings and click **Measure CD**. All methods in
`subpx.cd.METHOD_LABELS` are available. Each has its existing prerequisites:
Gaussian and invariant methods require model background; disk/flux ratio require
nominal CD or PSF calibration. The Activity tab reports invalid combinations
without discarding the previous results. Local-only methods bypass deblending.
The detection refinement control applies to **Find patterns**; CD independently
constructs its own seed list using its selected threshold or peak pipeline.

Use **Contours**, **Voronoi edges**, **CD colors**, and the stage selector to
review the measurements on the image. Boundaries come from the actual raster
cell labels. CD colors run from the displayed purple minimum to yellow maximum
of the selected method's valid equivalent CDs. Red crosses identify failed
measurements at their seed positions; they remain selectable.

The inspector compares calibrated CD and status across methods. It also shows
the selected method's metrics, full cell area and mean original cell intensity.
**Next failed** and **Show failed measurements only** help review problems.
Failed measurements retain their status and NaN values, with no seed fallback.

The **Feature / CD steps** tab shows original pixels, the method's input and
overlaid method contours. The emphasized contour is the selected method.

- **Gaussian:** input row, fitted Gaussian on that same row and the row residual.
- **Log-quadratic:** gold pixels show the actual fit-fraction support; the
  horizontal dashed line shows its intensity cutoff. Fit error is in the metric
  table. The row is contextual, not a reconstruction of the two-dimensional fit.
- **Gradient / half-height:** choose a ray. The input image highlights that ray
  and its measured edge. Plots use the same cubic sampling, smoothing, support
  mask and radial step as the CPU measurement. Signal is divided by center
  signal; falling slope is divided by its maximum absolute value. Dashed lines
  show the measured radius and, for half-height, the half-height level. Missing
  rays are shown without an invented edge. GPU measurements are diagnosed with
  CPU sampling of the returned NumPy input, so rounding can differ.
- **Projection / local projection:** choose x or y. Plots show the summed
  samples, the same zero-padded cubic spline and its derivative, and the picked
  steepest rising/falling edges. These edge distances define the reported width.
- **Area methods:** compare the saved outline with the input and reported
  threshold/local level. The row is contextual; the measured region uses 4x
  spline samples. Wide-window methods store only the center of their full
  measurement window for display; this is explicitly labelled. Display windows
  use their own stored origins rather than assuming a standard cell origin.

Method definitions differ. A Gaussian FWHM, a gradient edge, a thresholded area
and a calibrated hole diameter should not be expected to have identical CDs.
Convergence and a successful status do not establish measurement accuracy.

## Pitch and placement error

Choose **CD method centers** or **Detection centers** in the **Pitch** settings,
then click **Pitch + placement**. The workbench reuses matching measurements
when only pitch settings changed; otherwise it calculates the required source.
It supports square/rectangular and staggered indexing, row/column averaging,
pitch hints, a selected reference feature and a nominal x/y pitch.

Gold links are the actual x pairs; purple links are the y pairs in the selected
averaging bands. A star marks the calculation's reference. The **Pitch &
placement** tab shows interval pitch and cumulative placement error for both
axes, with pair counts, selected bands and the baseline source.

Placement error is the discrete sum of `pitch - baseline`, zero at the reference
node. It is distinct from two-image alignment. Zero nominal x/y controls select
the measured means; set both positive to compare against a design pitch. Hints
are in pixels and affect indexing only; nominal pitch is in the chosen length
unit. Missing intervals break cumulative curves. Pitch error bars show the
spread across contributing pairs within an interval, not repeated-image
precision or uncertainty of the mean. Irregular patterns can fail lattice
indexing; the CD results and diagnostic notes remain available.

## Image registration

In **Register**, open a moving image of the same dimensions as the loaded fixed
reference, select its page if needed, then click **Register images**. Registration
always uses the full loaded images, independently of the analysis region.

The CPU phase-correlation result is the shift **to apply** to the moving image,
in `[dy, dx]` pixels. The lower tab shows before/after differences on the same
intensity scale and the integer-lag phase correlation. Switch to the aligned
moving image or a 50% blend to inspect alignment directly. RMSE is calculated
before and after on the same valid overlap; padding is excluded. The normalized
phase-correlation error is reported separately. Repetitive patterns may have
ambiguous correlation peaks.

## Results and responsiveness

**Export results** creates a uniquely named directory under the chosen parent:

- Native CD arrays, tables and manifest using `save_cd_results`, or detection
  labels, area decisions and global center coordinates.
- Per-method, per-axis pitch CSVs, including node positions, pitch, pair spread,
  counts, integrated distance and placement error.
- `workbench.json` with analysis parameters, elapsed time, notes and crop origin;
  `source.json` with image paths/pages and registration metadata when present.
- `registration.npz` with shift, aligned image, valid overlap, difference images
  and correlation when image registration has been calculated.

CD exports use crop-local `[x, y]` coordinates. Add `origin_xy` from
`workbench.json` for coordinates in the loaded image. The detection-only center
file is explicitly named `centers_global_xy.npy` and already includes the origin.
**Save screenshot** saves the current window as a PNG.

Loading, measurement, registration and exports run in a single background Qt
worker. Image navigation and diagnostics remain available while a job runs;
input changes and new jobs are disabled until it finishes. Closing waits for
completion by keeping the window open; there is no forced worker termination.
This workbench holds one image/page and its diagnostic arrays in memory. It uses
pyqtgraph display downsampling, not the huge-image viewer's tiled storage path.

Changes to parameters are labelled as pending until the next run. Results on
screen continue to describe the last completed calculation. A new image/page
clears all dependent measurements and registration results.

## NumPy / notebook use

```python
import subpx

# In IPython/Jupyter, first execute: %gui qt
window = subpx.launch_workbench(image, name="Inspection image", block=False)
```

In a normal Python script, omit `block=False` so the function starts the Qt event
loop if necessary. Importing `subpx` does not import this optional UI module.

## Validation

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest tests/test_workbench_data.py tests/test_workbench.py -q
```

Tests cover scientific image precision, pages, crops and coordinate origins,
exports, radial/projection diagnostic agreement, raw and wide method inputs,
known registration shifts, actual Qt workers, method/stage switching, selection,
contrast preservation, reuse, error recovery and state reset. Qt tests skip if
PyQt6 or pyqtgraph is not installed.
