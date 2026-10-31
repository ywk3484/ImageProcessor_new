# Desktop inspection workbench

The PyQt6 application runs the repository's existing analysis methods on sample
images and uses pyqtgraph for every image, overlay and diagnostic plot. It is
separate from the OpenGL tiled notebook viewer in `subpx.viewer`.

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m subpx.workbench --demo
.\.venv\Scripts\python.exe -m subpx.workbench path\to\image.tif [--page 5]
```

For a fresh installation:

```powershell
python -m pip install -e ".[analysis,viewer]"
subpx-workbench
```

![Workbench: Setup on the left, the image with failed cells, the hole inspector on the right, the holes table below](images/workbench.png)

## Layout

```
 File Edit View Run Holes Help
 [Measure|Align]  file.tif   ◀ Page 5 ▶ of 48   2048 x 1378 px | uint8        Compute: GPU (auto)
 ┌ Setup | Pages ┐ ┌ (error / "results are from earlier settings" banner) ┐ ┌ Hole inspector ─┐
 │ 1 Image     ● │ │ Show [Image]  Method [Gaussian fit]  Layers          │ │ Hole 812 of 39k │
 │ 2 Holes     ● │ │                                                      │ │ every method's  │
 │ 3 CD methods ●│ │            the active page (or the stack)            │ │ CD and status,  │
 │ 4 Pitch     ● │ │                                                      │ │ pixels, profile │
 │ [Measure CD]  │ ├ Holes | CD distribution | Pitch | Log ───────────────┤ │                 │
 └───────────────┘ └──────────────────────────────────────────────────────┘ └─────────────────┘
 status | hover position | job + elapsed + Cancel | holes, mean CD, failures of every method
```

- **Setup** (left) holds the steps in pipeline order: **Image** (pixel size,
  unit, polarity, compute device), **Holes** (seeds, threshold, area limits with
  a histogram of the component areas and draggable limits), **CD methods**
  (preset, methods in groups with All / None, a *Parameters of the ticked
  methods* fold, calibration only when a calibrated method is ticked) and
  **Pitch** (source, lattice, averaging, design pitch, an *Advanced* fold with
  indexing hints). One step is open at a time; its **Run** button stays at the
  bottom. A disabled Run button says why directly under it (for example "Area
  min (300) is above area max (200)"), and the field at fault gets a red border.
  Each step has a mark: grey not run, green up to date, amber **changed** since
  the result on screen.
- **Pages** (the tab beside Setup) lists the file's pages, counted from 0 as in
  the batch's `pages.csv`, with thumbnails made as they scroll into view and a
  line on what was measured there ("CD 39,042 holes, most failed: Projection
  2,048"). **Stack vertically** is here and in the View menu.
- The **image** shows the result chosen in the **Method** box: a CD method or
  **Previewed holes**. Running one step never hides another step's result: the
  box switches between them.
- The **hole inspector** (right) compares the selected hole in every method:
  CD, center and status per method with the method's colour, original pixels and
  the method's input with all contours (current one thick), the profile, the
  metrics and the definition. Click a method there to show it everywhere.
- The **bottom tabs** hold the **Holes** table, the **CD distribution**, the
  **Pitch** result and the **Log**. The Log never opens by itself.
- **Ctrl+1 / Ctrl+2 / Ctrl+3** show or hide Setup, the inspector and the bottom
  panel. Panel sizes, positions and the open tab are remembered. On a small
  screen (below 1300 x 800) Setup starts collapsed.
- The **status bar** shows the last message, the position under the mouse, the
  running job with its elapsed time and **Cancel**, and the result summary with
  the **failed holes of every method** ("Failed: Projection 2,048 (5 %) · Foot
  area 471 (1 %)"), not only of the method shown.

**The defaults are the batch's defaults** (`scripts/cd_batch.py`): the **Batch
default** preset (its ten methods: Gaussian fit, Projection, Half area, Half
area min level, Half area min level avg peak, Tail area, Tail area no level,
Edge area, Foot area, Foot area image ratio), area 3-200 px², padding 3,
deblend iterations 30 and stop tolerance 0.001, and **Compute: Auto** (the GPU
when CuPy works). A test pins them to `cd_batch`, so the workbench measures a
page as the batch does. Two more presets compare **Spot widths** (Gaussian, log
fit, gradient, half-height) and **Raw-cell widths**. The calibrated methods
(blurred hole fit, flux ratio) need a nominal CD or PSF and are only in the
full list. Every method has one display name everywhere; its library key and
label are in tooltips and exports.

**Settings are remembered** between sessions (every Setup control, layers,
locks, the table filter, the window and panel layout, the last folder and
recent files). **Edit > Reset settings** restores the defaults above; the image
and results stay.

## Keys

| Key | Action |
| --- | --- |
| Ctrl+O, or drop a file | open (File > Open recent lists the last 8) |
| Ctrl+Enter (either Enter key) | run the open Setup step |
| F / Shift+F / Z | fit the page / fit the whole stack / zoom to the selected hole |
| J / K, Down / Up in the table | next / previous hole in the table's order (J = next, as in vim) |
| N / Shift+N | next / previous hole the current method failed on (wraps) |
| Page Down / Page Up | next / previous page, while the image or the page list has focus |
| Esc | cancel the running job |
| Ctrl+C | copy the selected table rows, or the selected hole's row (tab-separated) |
| Space | Align view: swap the reference and the aligned page |
| Ctrl+1 / Ctrl+2 / Ctrl+3 | show or hide Setup / Inspector / bottom panel |
| Ctrl+E | export |
| F1 | keys and workflow |

Every menu entry and button shows its key. Typing in a number or text field is
never taken by these keys. In the table, Page keys scroll the table.

## Errors, out-of-date results and jobs

- Errors appear as one plain line in a red banner over the image; the results
  stay visible. **Details** opens the full text in the **Log** tab. Unexpected
  errors in the window itself go to the same banner.
- When a setting differs from the one the result on screen was made with, an
  amber banner says what changed ("Results are from earlier settings: area max
  200 -> 150, threshold Otsu -> Triangle (+1 more)") with **Re-run** (the
  out-of-date step with the current settings) and **Revert** (the settings, and
  the analysis region, of the result on screen).
- Opening, measuring, aligning and exporting run on one background worker; the
  status bar shows the job and its elapsed time. **Cancel** (Esc) stops waiting
  and discards the result; the library call cannot be interrupted, so the worker
  finishes it in the background. Zooming, panning and inspecting stay available
  while a job runs.
- Opening another file and closing ask before discarding results. Closing while
  a job runs is refused with a hint to press Esc first.

## Image review

- Open TIFF, PNG, JPEG, BMP or a two-dimensional `.npy` array. A multipage file
  shows a page bar in the header (◀, the page number from 0, ▶) and the page
  list. A TIFF is opened once: its page table is read and the file stays open
  for page changes, the stack, thumbnails and moving pages of the same file.
- Changing the page keeps the zoom, the analysis region, the bottom tab, the
  image view, the selected position (after measuring, the hole at that position
  is selected) and the moving page, so one area can be followed through the
  pages. Tick **Lock** under the histogram to keep the contrast too; otherwise
  each page is stretched to its own 1-99.5 %. When a panel beside the image
  appears or hides (a banner, the CD scale), the image keeps its center and
  zoom and shows more or less of the page.
- **Stack vertically** displays all pages together, in file order. Click any
  page to make it active; its outline turns teal and the page list follows.
  Pages may have different dimensions or numeric types. They are displayed
  separately, with space between them, rather than concatenated into one
  measurement image. Analysis, regions, hole selection, alignment and export use
  the active page only. Overlays appear on that page. Hover reports the page
  number and coordinates local to that page (the pixel value when that page is
  in memory). **F** fits the active page, **Shift+F** the whole stack. Page
  labels appear while at most 12 pages are in view.
- Stacking reads nothing first: the file is open. Pages are decoded when they
  come into view, nearest the middle of the view first, a page at a time between
  other events. Zoomed out (3 or more image pixels per screen pixel) a page is
  drawn from a preview at most 256 pixels wide (block means); zoomed in, from
  the full page. Previews are kept for every page seen (about 44 kB for a 2048 x
  1378 8-bit page); full pages are kept up to 128 MB, the least recently used
  dropped first. Pages out of view are not drawn.
- Grayscale data retain their original precision. RGB is converted to luminance
  with `0.2126 R + 0.7152 G + 0.0722 B`; alpha is ignored. The conversion is
  displayed beside the image size. Nonfinite and complex images are rejected.
- Wheel to zoom, drag to pan. The histogram adjusts display levels without
  changing data or measurements.
- **Layers > Analysis region** shows a gold rectangle; drag it or its corners.
  The next run uses only that crop. Its boundaries can clip holes, affect seed
  detection and exclude cells. Pixel centers have integer coordinates, x is the
  column and y increases down the image.
- In **Image**, set the pixel size and its unit together before measuring. The
  UI displays calibrated lengths and areas; result arrays keep pixel units.

![TIFF pages stacked vertically: the active page outlined, the page list beside it](images/workbench_stack.png)

## Previewing holes

Open the **Holes** step, adjust seeds, threshold, area limits and refinement,
then **Preview holes**. The area histogram shows every component's area
against the two limits (drag a limit line to set it) and counts how many are
inside, smaller and larger. Switch **Show** to **Threshold mask** or
**Component labels** and turn on **Layers > Rejected boxes** to see components
outside the limits. The Method box then shows **Previewed holes**; the holes
table lists their positions and areas.

Detection input is rescaled to uint8 when required by OpenCV (uint16 Otsu is
retained); the Log records this. The rescaling does not replace the original
image used for CD.

## CD and intermediate steps

Open **CD methods**, choose a preset or tick methods, then **Measure CD**. All
methods in `subpx.cd.METHOD_LABELS` are available, with their prerequisites:
Gaussian and invariant methods require the model background; disk and flux
ratio require a nominal CD or PSF. Local-only methods bypass deblending. The
refinement of the Holes step applies to previewed holes; CD builds its own seeds
with its threshold or peak pipeline.

The image shows pages of 40,000 holes without slowing down:

- **Raster maps** are drawn at every zoom and made once per result and method:
  **CD colors** fills each Voronoi cell with its hole's CD (failed holes stay
  clear), **Failed cells** (on by default) tints the cells of the holes the
  shown method failed on in red, **Voronoi edges** draws the cell boundaries.
- **Centers, contours, all methods' contours, pitch pairs and rejected boxes**
  are drawn for the holes in view only, and only while at most 2,000 holes (or
  contours) are in view; farther out the line above the image says "zoom in to
  see centers and contours". Red crosses mark failed measurements at their seed
  positions; they remain selectable.
- Selecting a hole moves only its white ring; nothing else is redrawn.
- The **CD color scale** beside the image runs over 1-99 % of the shown
  method's valid CDs, so one outlier does not wash out the rest. Drag its ends
  to set a range; tick **Lock** under it to keep that range for every method
  and page.

The **holes table** keeps one column order: ID, x, y, status, CD x and CD y of
the shown method, then the CD of every method in the preset's order (the shown
one in bold). A sort (click a header) stays when the method or the filter
changes, and the selected hole keeps its row; J / K and N follow the table's
order. Filters: all holes, failed by the current method, failed by any method.
Type a hole ID and press Enter to select it. Click another method's CD column
to show that method. Right-click to copy a value, the selected rows or all rows
shown.

The **CD distribution** tab shows the current method's histogram with its mean
and spread. **Pick a range** puts a range on it: the holes in that range are
shaded on the image and the table lists only them (**Show all** ends it).

The hole inspector's crops and profile:

- **Gaussian:** input row, fitted Gaussian on that same row and the row residual.
- **Log-quadratic:** gold pixels show the actual fit-fraction support; the
  horizontal dashed line shows its intensity cutoff. The row is contextual, not
  a reconstruction of the two-dimensional fit.
- **Gradient / half-height:** a plot of the edge radius against the ray angle
  (failed rays as red crosses); click a ray there or on the input crop. The
  profile uses the same cubic sampling, smoothing, support mask and radial step
  as the CPU measurement. GPU measurements are diagnosed with CPU sampling of
  the returned NumPy input, so rounding can differ.
- **Projection / local projection:** choose x or y. Plots show the summed
  samples, the same zero-padded cubic spline and its derivative, and the picked
  steepest rising/falling edges.
- **Area methods:** compare the saved outline with the input and the reported
  threshold/local level. Wide-window methods store only the center of their
  measurement window for display; this is labelled.

Method definitions differ. A Gaussian FWHM, a gradient edge, a thresholded area
and a calibrated hole diameter should not be expected to have identical CDs.
Convergence and a successful status do not establish measurement accuracy.

## Pitch and placement error

Open **Pitch**, choose the source (**CD measurement** or **Previewed holes**),
then **Measure pitch**. Pitch uses that result of the active page as it is on
screen; it never measures again ("Measure CD first" otherwise). Averaging,
lattice, design pitch and a reference at the selected hole are in the step;
**Advanced** holds the stagger direction and pitch hints.

The **Pitch** tab shows, for the chosen method, the x and y pitch in plain words
(mean, pairs, design, placement range, lattice, reference, averaging) and two
plots: **Pitch** per interval with the spread across pairs, or **Placement
error** per row/column. Click a point: an interval picks the holes of its pairs,
a placement point the holes of that column or row; they are shaded on the
image, listed in the table, and the hole nearest their middle is selected. The
**Pitch pairs** layer draws gold x pairs, purple y pairs and the reference star.

Placement error is the discrete sum of `pitch - baseline`, zero at the
reference node. It is distinct from two-image alignment. Zero design pitches
select the measured means; set both to compare against a design pitch. Hints
are in pixels and affect indexing only. Missing intervals break cumulative
curves. Error bars show the spread across contributing pairs, not repeated-image
precision. Irregular patterns can fail lattice indexing; the CD results stay.

## Align

Switch the header to **Align**. Setup then lists the pages of the open file
(**This file**) or of another file (**Other file...**); click a page: it is read
and aligned to the active page (the reference) at once. The moving page stays
when the reference page changes, so several pages can be aligned to one moving
page, or one reference page to several (click another page in the list), with
**Ctrl+Enter** to align again. The pages must have matching dimensions.
Alignment uses the full pages, independently of the analysis region or the
stack.

The comparison fills the main image area: **Flicker** (Space swaps the
reference and the aligned page), **Difference before** and **after** on one
symmetric colour scale, **Checkerboard** (block size adjustable) and **Blend**.
The panel on the right gives the shift to apply to the moving page in `[dy,
dx]` pixels and in words, the RMSE before and after on the same overlap, the
batch's **hole check** (`cd_batch.py`, same function, pinned by a test:
**centered** passes below 0.8, **raw** below 0.3; when it fails the batch would
skip the page as "no holes" and the shift may mean nothing), and the phase
correlation around its peak with the runner-up marked. A runner-up above 0.8
of the best warns that on a repeating pattern the shift may be off by whole
pitches.

## Export

**File > Export** (Ctrl+E) asks for a folder (default: a new
`inspection_<date>_<time>` folder beside the opened file) and what to write:

- **Holes table as shown** (filter and sort) as `holes_table.csv`.
- **All result arrays**: native CD arrays, tables and manifest from
  `save_cd_results`, or detection labels, area decisions and global center
  coordinates; per-method, per-axis pitch CSVs (in `pitch/` when the pitch was
  measured on another result than the one shown); `workbench.json` with the
  parameters, time, notes, warnings and crop origin.
- **Alignment**: `registration.npz` with the shift, aligned page, valid overlap,
  difference images and correlation.

`source.json` is always written: file, zero-based active page, page count,
display mode (`page` or `stack`), the method shown, the moving page and the
alignment's shift, RMSE and hole check. CD exports use crop-local `[x, y]`
coordinates; add `origin_xy` from `workbench.json` for page coordinates. The
detection-only `centers_global_xy.npy` already includes the origin. **Save
screenshot** saves the window as a PNG.

## Memory

Results are kept per page (previewed holes, CD and pitch together) for the open
page and the most recently measured other pages: at most 3 pages and 3 GB (one
39,042-hole page measured with the ten batch methods holds about 1.25 GB, so
then the open page and the one before). Pages without results take no place.
Older pages' results are dropped, with a line in the Log; the page list keeps
saying what was measured there. Opening another file clears all results.

## NumPy / notebook use

```python
import subpx

# In IPython/Jupyter, first execute: %gui qt
window = subpx.launch_workbench(image, name="Inspection image", block=False)
```

In a normal Python script, omit `block=False` so the function starts the Qt
event loop if necessary. Importing `subpx` does not import this optional UI.

## Code

`subpx/workbench/` is a package: `session.py` holds the state (open file, page,
results per page, method, selection, moving page, alignment, settings) and runs
the jobs (`jobs.py`); `window.py` builds menus, header, docks, banners and the
status bar; `canvas.py` the image view, its layers and the stack;
`panels/` the Setup steps, page list, hole inspector, holes table, CD
distribution, pitch, Align view and log. Panels listen to the session's signals
and call its methods, never each other. `names.py` and `settings.py` (method
names, settings labels, run checks) do not import Qt. The Qt-free data layer
stays in `subpx/_workbench_data.py`.

## Validation

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest tests/test_workbench_data.py tests/test_workbench.py -q
```

The tests use their own settings file, never your saved settings.
`scripts/workbench_timing.py` times clicks, method switches, layer toggles and
repaints on a page of your size (default: `artifacts/cd_test/data/stripe1_board1.tif`),
the memory over the following pages, and the stack of that 48-page file.
On this PC (RTX 4060 Ti, 39,042 holes, ten methods): Measure CD 7.4 s, click
0.03-0.05 s, J 0.03 s, N 0.001 s, method switch 0.04-0.06 s, layer toggle
0.004-0.02 s, repaint 0.003 s at full view and 0.02 s zoomed in with 1,225
holes drawn; memory 1.9 -> 3.0 -> 4.2 -> 4.4 -> 4.3 GB over five measured
pages; the 48-page stack turns on in 0.12 s and fills its view in 0.26 s.

Tests cover the batch defaults, remembered settings and layout, Reset, disabled
styles and Run reasons, the Setup steps and their Run button, docks and
Ctrl+1/2/3, the stale banner with Revert and Re-run, J / K / N / Shift+N in the
table's order, Ctrl+C, picking holes from a CD range and from a pitch point,
asking before discarding results, Cancel, refusing to close while busy, Page
keys scoped to the image and the page list, the page list and the stack
following each other, the CD map, its scale and lock, the selection redrawing
nothing else, centers and contours for the holes in view only, the table's
fixed columns and kept sort, results kept for the last pages only, stack
previews, full pages in view and their memory cap, F / Shift+F, per-method
failure counts, the state kept across pages, the Align view and the batch's
hole check, the error banner, unreadable pages and files, page-local
overlays/crops/exports, compatible presets, calibration guidance, method
comparison, crops and coordinate origins, export, raw and wide method inputs,
known registration shifts and actual Qt workers. Qt tests skip if PyQt6 or
pyqtgraph is not installed.

## Known gaps

- Opening a `cd_batch` run (its settings and page list) is planned (P3).
- Progress shows the job and its elapsed time, not the stage inside a CD run:
  the library call reports no progress.
- The analysis region is set by dragging only (no typed x0 y0 x1 y1).
- Previewed holes list their kept components; rejected components are counted
  in the Holes step and drawn with **Rejected boxes**, not listed in the table.
