# Pattern pitch and registration reports

This pipeline measures pitch from the centers already calculated by the CD
methods. It produces an offline HTML report with method selection, adjustable
averaging counts, an image showing the selected regions, x/y pitch profiles,
and registration error profiles integrating pitch minus its mean. It also exports CSVs,
raw JSON, and a PNG of the initial selection for the first method.

## Run on an image or folder

Use the existing Windows environment from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/pitch_report.py image.tif --output artifacts/pitch_report --average-rows 5 --average-columns 7 --backend gpu
```

The same command accepts a folder containing grayscale PNG, TIFF, BMP, or JPEG
images. Each image gets its own report and the top-level `index.html` links them.
Add `--recursive` for subfolders, or `--page 10` to select TIFF page 10 (zero-based).
Images that cannot be measured are listed in `images.csv`; a batch continues
and returns a nonzero exit code if any image failed. Choose a new or empty
output folder for every run.

Open the printed `index.html` in a browser. It works directly from disk without
a server or internet connection. Change the method, rows/columns averaged, or
click a hole to move the reference. The highlighted regions and all four curves
update immediately using the measured centers. The **Download current CSV**
button exports the displayed selection.

The default includes all 17 methods in `cd_fast_report.py`: ten original methods,
four raw-cell methods, and three native-pixel methods. To run a smaller set:

```powershell
.\.venv\Scripts\python.exe scripts/pitch_report.py image.png --methods gaussian projection half_area local_projection pixel_half_area --layout staggered --reference 384 128 --average-rows 5 --average-columns 7
```

All other `measure_cds` methods are selectable too. The optional `disk` and
`flux_ratio` methods require `--psf-fwhm` in pixels. The CLI exposes segmentation,
inversion, area limits, local thresholds, and the tail window; use the Python API
for other CD settings. `--backend auto` uses the existing CPU/GPU dispatch.
CD evaluation uses `metrics_only=True, outlines=False`: centers are retained
without constructing unnecessary display contours or full cell stacks.

Reuse existing saved CD results without remeasuring the image:

```powershell
.\.venv\Scripts\python.exe scripts/pitch_report.py artifacts/cd_results --from-cd --output artifacts/pitch_from_cd --average-rows 5 --average-columns 7
```

`--from-cd` uses all saved methods, calibration, and image data. Measurement
options such as `--methods`, `--backend`, and `--pixel-size` apply to image input.

## Centers, lattice, and averaging

Gaussian methods provide fit centers. Projection methods use the midpoint
between the two steepest edges of each x/y projection. Area methods use the
centroid of their thresholded region (interpolated or native pixels, according
to the method). Radial methods use their existing measurement origin; they do
not acquire a separate contour-centroid refinement in this pipeline. Centers
are read directly from `CDMethodResult.centers_xy`. Invalid CD measurements
and nonfinite centers are excluded, with no replacement by segmentation seeds.

A single lattice is indexed from the shared detection seeds and reused across
methods. `--layout auto` distinguishes rectangular/square and half-pitch
staggered arrays. With the default `--stagger-axis row`, alternate rows shift
in x by half a pitch. Use `--stagger-axis column` when alternate columns shift
in y. `square` allows unequal x/y spacings.

Let `C[r,c] = [x[r,c], y[r,c]]` be a method's measured center. Local pitch is:

```text
Px[r,c] = x[r,c+1] - x[r,c]
Py[r,c] = y[r+1,c] - y[r,c]
```

For a row-staggered grid, the y neighbour lies in the next row despite its
horizontal offset. `Py` is the projected adjacent-row spacing, not the diagonal
distance or the distance between every other row. Column staggering uses the
corresponding definition for `Px`.

- `--average-rows N` averages `Px[r,c]` over N contiguous logical rows at every
  column interval, producing x pitch versus x position.
- `--average-columns N` averages `Py[r,c]` over N contiguous logical columns at
  every row interval, producing y pitch versus y position.
- Omitting either count uses all rows or columns. Counts must fit the lattice.
- Bands surround the reference and shift inward at image edges. For even
  counts, one extra line lies toward increasing row/column index.
- Orange marks the selected rows; blue marks the selected columns. Lines and
  center markers identify the actual valid pairs inside the highlighted areas.
  Staggered bands follow logical columns/rows and may zigzag at their edges.
- Each interval reports its actual valid pair count and sample standard
  deviation. Missing/invalid holes reduce counts. A missing lattice neighbour
  never generates a doubled pitch. Zero-pair intervals are unavailable.

Profile positions are the mean measured x or y coordinate of the selected
holes at each lattice node; interval points are plotted halfway between nodes.
Seed coordinates supply plot positions only when a node has no valid centers.
They never supply pitch measurements. All image coordinates and pitch hints
are in pixels. Profile lengths are multiplied by `pixel_size` into `unit`.

## Integrated pitch error and registration

The default reference is the first lattice hole. `--reference X Y` chooses the
nearest detected hole to an image point in pixels; clicking the image changes
it interactively. Registration is zero at that reference's column or row:

```text
registration[k] = signed sum of (averaged pitch - nominal pitch) over those intervals
```

These are discrete sums over hole intervals, accumulated in opposite directions
on either side of the reference. Integrating against physical x/y position would have
units of length squared, so the code sums by lattice index. Missing intervals
interrupt the integral; the pipeline does not interpolate across them.

By default each method's nominal x/y pitches are the means of all contributing
pairs in the selected bands. This shows accumulated relative placement error
after removing the mean pitch. It cannot establish absolute placement error
against a design. To retain a known mean pitch offset, provide design pitches:

```powershell
.\.venv\Scripts\python.exe scripts/pitch_report.py image.tif --pixel-size 21 --unit nm --nominal-pitch 168 168
```

Nominal pitches use the calibrated unit (`nm` above), and can also be entered
in the report. Blank fields restore the selected measured means. The curves
are one-dimensional averaged placement profiles; they are not an image-to-image
registration transform or a full 2-D displacement field. A reference hole whose
CD fit failed is identified in the report; its lattice index still anchors the
integrals, but no measured center is substituted for it.

## Python API

```python
from subpx import measure_cds, analyze_pitch, plot_pitch_report, save_pitch_report

cd = measure_cds(image, methods=("gaussian", "projection", "half_area"),
                 area_max=200, pixel_size=21, unit="nm", backend="auto",
                 metrics_only=True, outlines=False)
options = dict(layout="auto", average_rows=5, average_columns=7,
               reference_xy=(384, 128))
methods = analyze_pitch(cd, **options)
p = methods["projection"].profiles["y"]
print(p.pitch, p.counts, p.registration)
fig = plot_pitch_report(image, methods["projection"])
fig.savefig("projection_pitch.png", dpi=180)
save_pitch_report(cd, "artifacts/my_pitch_report", **options)
```

`analyze_pitch` also accepts `CDSummary` from `measure_cds(summary_only=True)`.
`save_pitch_report` needs a full `CDResult` because it displays the image. All
array fields are NumPy arrays. `PitchProfile.position`, `distance`, and
`registration` contain one entry per node; `pitch`, `std`, and `counts` contain
one entry per interval. `pairs` stores original hole indices for the selected
valid pairs. `PatternPitchResult.indices_rc` stores `[row, column]` lattice
indices; `centers_xy` and `lattice_pitch_xy` remain in pixels. The raw cumulative
`distance` field is retained in Python and CSV exports, but is not plotted.

Each image report contains:

| File | Contents |
| --- | --- |
| `index.html`, `data.js`, `image.png` | Offline interactive report |
| `report.png` | Initial selection, first method, all four curves and overlay |
| `profiles.csv` | Initial x/y profiles for every method, counts, spread, both integrals |
| `centers.csv` | Every method center, validity, seed, and shared lattice index |
| `summary.csv` | Mean pair pitch, valid center/pair counts, missing intervals per method/axis |
| `data.json` | All centers, lattice, neighbour pairs, settings, and CD metadata |

Keep the report files together. Browser changes do not overwrite initial
artifacts; export the current CSV for a changed selection.

## Scope and validation

The indexer targets nearly axis-aligned regular arrays with small drift. It
does not estimate rotation, arbitrary oblique lattices, or non-half-pitch row
offsets. Grouping tolerance defaults to 20% of the typical nearest seed spacing;
adjust `--line-tolerance` in pixels if needed. With missing entire lines or
uncertain spacing, use `--pitch-hint X_PX Y_PX` to preserve skipped lattice indices.
Severely sparse or ambiguous arrays require an explicit hint or correction of
the detection settings. Lattice indexing and aggregation run on the CPU;
the CD methods retain their existing GPU acceleration.

Run the deterministic numerical and report tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pattern.py tests/test_pitch_report.py -q
```

The tests check both stagger orientations, square layouts, analytic variable
pitch and cumulative error, calibrated units, band edges, missing fits/lines,
CD centers on synthetic Gaussian images (pitch tolerance 0.08 px), and CPU/GPU
pitch parity (2e-5 px tolerance). Node.js, when available, executes the offline
viewer and compares six changed selections to Python to 1e-9 in calibrated
units. GPU and Node tests skip with explicit reasons when unavailable.

Local validation on 2026-10-29: all 23 new tests passed, including GPU parity
and the Node viewer checks. The full suite returned 455 passed, 1 skipped, and
9 failures; all nine reproduced on the unchanged `72aa659` commit in an isolated
checkout (older GPU isophote/radial/tiled detection tests and an auto-backend
expectation). The skipped test exercises missing CuPy and skips because CUDA
is available; no new GPU/report tests were skipped.

The included 768 × 256 staggered sample was also processed with all 17 methods,
five rows/seven columns averaged, and 21 nm/px calibration. The run measured
2,805 holes and generated its report in 2.58 s, excluding interpreter startup.
The largest selected mean-pitch deviation from the sample's nominal 8 px was
0.00191 px across the methods and axes. This is an end-to-end sample check,
not an accuracy guarantee for experimental images. Browser interaction and
the exported PNG were visually checked.
