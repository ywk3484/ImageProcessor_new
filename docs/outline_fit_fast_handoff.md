# `outline_fit_fast`: handoff for the build

**Status, 2026-11-12.** The user approved building the one-pass outline fit as a new method beside `outline_fit` and
asked for this handoff instead of an implementation in the session that proposed it. Nothing in `subpx/` or
`scripts/` has been changed. A fresh session builds from this file.

Read in this order:

1. this file (what to build, where, how to check it);
2. `docs/outline_fit_one_pass.md` (the method exactly, its settings, every measurement, what was tried and dropped);
3. the prototype, `artifacts/outline_fit/fast/` (tracked in git although `artifacts/` is ignored; `README.txt` lists
   the scripts). It is the reference implementation: the build is a port, not a redesign.

`docs/outline_fit_one_pass.html` is the page the user read before deciding.

## 1. What the user decided (2026-11-12)

- "Yes implement the new method beside outline_fit. Make it available in the report pipeline so that I can test it on
  my real images within a folder. I need to check whether the new method actually is comparable to outline fit in
  real images."
- Output: "For now, page mean. In the future, possibly every hole, but that's only when this method is proven to be
  good enough."
- The one minute of reading a TIFF is measured on the remote. The remote has 64 CPU threads (4 x RTX 6000 Ada).
- How the remote selects pattern areas today (for the later batch work, section 8): all pages of a TIFF are loaded
  as one (N, H, W) array, every page is cut into m tiles of (h, w), giving (N, m, h, w); the reference page is cut
  the same way; phase cross-correlation runs over the whole 4-D array; the error values give the valid (N, m)
  indices. "This does not mean that you have to follow this procedure and you may propose a better method."

Standing rules that apply (CLAUDE.md, memory): add, never replace or rename a method or a UI feature; commit and
push to `main` after every verified change, staging only your own files; the GPU is shared with the user's own jobs
(look at `nvidia-smi` first); console output ASCII only; everything on the remote is typed by hand (one command,
about 150 characters back).

## 2. Scope

**Phase 1, this build.** `outline_fit_fast` in the library and in both folder reports, GPU only, page values only.

- "Report pipeline" is read as the folder reports `scripts/cd_report.py` and `scripts/cd_fast_report.py` (a folder
  of PNG images). They are the only place where `outline_fit` runs on a folder, so they are where the two can be
  compared. **Asked, not yet answered:** whether the user's real test images are PNGs in a folder as before; if
  they are TIFF pages, the reports need a way to read pages of a TIFF (ask before building that).
- Page values only: the image's outline CD, its x / y widths and its blur. Every reported hole reads the image's CD
  (status `page_size`, the report's existing convention for "no size per hole").

**Not in phase 1:** the TIFF batch (`scripts/cd_batch.py`), measuring only part of a page (tiles), a size per hole,
a CPU path, the workbench. Sections 8 and 9 say what is known about them.

## 3. What the method is, in ten lines

Per image, with the same model, parameter vector, bounds and CD definition as `subpx._outline`:

    A  seeds -> plain stack -> pair maps -> first page fit (grid of starts, round + superellipse)     theta_0
    B  hole pass with the size held (centre, brightness, level of every hole), two sweeps, template theta_0
    C  plain stack again with B's centres -> pair maps -> final page fit from theta_0, all free         theta_1

- Plain stack: every stacked hole's window binned at its sub-pixel offset on a quarter-pixel grid; the plain mean.
- Pair maps: for each of the 16 sub-pixel classes, where the neighbours (and the hole itself) stand; this puts the
  neighbours' light into the model, so no alternation is needed.
- Page fit: outline and blur by least squares on the stack; the template's spectrum is analytic, so model and
  Jacobian come from one pass over the outline; scale and constant projected out; solver on the host.
- The page's CD is `Template.cd_px(theta_1)`. Step D (a size per hole, three sweeps) exists in the prototype and is
  not part of this build.

Derivation, settings and numbers: `docs/outline_fit_one_pass.md` sections 3 to 5.

## 4. The prototype (reference implementation)

`artifacts/outline_fit/fast/`, run from that folder with the repo's venv (`README.txt` has the commands):

| File | Holds | Goes to |
|---|---|---|
| `spec.py` | `Spec` (template spectrum, polygon form, any edge offset), `Band` (frequencies and the way back to a grid) | library, for the hole pass's templates and the far kernel |
| `shapefit.py` | `ShapeFitter`: outline integrals (kernel `outline_hat`), blur transfer, synthesis, `evaluate`, `grid_start`, `default_prior`, `fit` (Levenberg-Marquardt on the host, pages in lockstep); runs under NumPy or CuPy | library, as it is |
| `gpufit.py` | the CUDA kernels (`stack_acc`, `hole_vs_stack`, `pair_splat`, `nearest`, `hole_normal`, `hole_update`, `render_scatter`, `noise_hist`, `far_splat`), `find_seeds`, `CellList`, `GpuPage` (stack, pair maps, templates, hole pass) | library, GPU module |
| `pipeline.py` | `fit_pages` / `fit_page`: steps A to D for a batch of pages | library: the driver behind the public function |
| `truth_gpu.py`, `truth_matrix.py`, `small_pages.py`, `port_checks.py`, `run_file.py`, `seed_test.py`, `checks.py`, `check_spec.py` | the measurements (generated pages with known sizes, small images with the report's seeds, the numbers quoted for the tests, the test file) | stay scratch; reuse them to check the port |

Prototype defaults that the numbers were made with: 4 bins per pixel, pair grid 8 per pixel, band `|f| <= 0.6`,
256 outline points, plain ("raw") stack, single-precision templates, first fit `round` (10 iterations at most) and
`super` (12) with `ftol 1e-6`, final fit `full` (30, `ftol 1e-7`), two centring sweeps.

Leave out when porting (prototype leftovers): `weighting="uniform"` / `"a2"`, `build_stack(recentre=True)`
(biased, notes 6.4), `Band.from_grid`, the double-precision template switch unless a test wants it.

## 5. Phase 1: the build

Work in plan mode first (CLAUDE.md), then in this order. Each step ends with its check.

### 5.1 Library

- [ ] **Page fit module** (suggested `subpx/_outline/onepass.py`): `Spec`, `Band`, `ShapeFitter` from the
  prototype, unchanged in behaviour, NumPy or CuPy. Check: tests 1 to 3 of section 5.4 (no GPU needed).
- [ ] **GPU module** (suggested `subpx/_gpu/outline_onepass.py`): the kernels and `GpuPage`. Compile through one
  cached `cp.RawModule` per device and template precision (`subpx/_gpu/_cuda.py` has `kernel()` for single
  kernels and `release_memory()`); options `--std=c++14`. Port `find_seeds` too (used when no seeds are given).
- [ ] **Driver**: steps A to C for one image (`pipeline.fit_pages` with `level=1`), returning the page fit and the
  holes' centres, brightness, level, leftover and stacked mask after B. Keep the batch form (a list of images, the
  page fits in lockstep): phase 2 needs it.
- [ ] **Public function** in `subpx/outline.py`, exported from `subpx/__init__.py` beside `fit_outline`:

      fit_outline_fast(image, seeds_xy=None, *, pixel_size=1.0, unit="px", blur="gauss2", invert=False,
                       n_rays=128, backend="auto", device=0, log=None) -> OutlineFastResult

  - `seeds_xy=None`: the method's own seed finder; given: used as they are (the reports pass theirs).
  - Input checks as `fit_outline` (2-D finite image, finite seeds, `blur` in `BLUR_FAMILIES`, at least `MIN_HOLES`
    seeds, positive `pixel_size`). `invert` as `_fit` does it (`top - image`; report the level back in the image's
    grey levels).
  - GPU only: `backend="auto"` takes the GPU when CuPy has a device; otherwise, and for `backend="cpu"`, raise
    `RuntimeError` with one plain sentence ("fit_outline_fast needs the GPU (CuPy); ..."). Run inside
    `gpu_device(device)`; `release_memory()` in a `finally`.
  - `OutlineFastResult` (new dataclass in `subpx/types.py`, NumPy arrays, lengths in pixels): per hole
    `centers_xy` (B's centres plus the outline's centroid, as `PageFit.results` does), `brightness`, `level`,
    `rms`, `good` (stacked); per page `outline_cd_px`, `cd_x_px`, `cd_y_px` (`ellipse_widths(template, theta,
    zeros(1))`), `rays_px` (n_rays,) of the page's outline (`outline_rays` with a zero offset), `blur_fwhm_px`,
    `theta`, `stack`, `stack_weight`, `upsample`, `blur`, `summary`, `pixel_size`, `unit`.
  - `summary` (JSON-ready, same key names as `fit_outline`'s where the meaning is the same): `blur_family`,
    `outline_cd_px`, `outline_extent_x_px`, `outline_extent_y_px`, `superellipse_p`, `outline_turn_deg`,
    `harmonics`, `blur_fwhm_px`, `blur_rms_px`, `holes_fitted`, `holes_stacked`, `spacing_px`, `window_px`,
    `noise_sd`, `leftover_chi2` (chi-square per bin of the final page fit: 1 = the stack is fitted to its noise),
    `first_cd_px` (the first page fit's CD), `fit_iterations`, `converged` (the final fit stopped by its tolerance,
    not by its iteration cap), `per_hole_size: False`, `seconds`, `backend`, `theta`, `lengths: "px"`.
- [ ] **`measure_cds` method** `outline_fit_fast` (`subpx/cd.py`; the existing `outline_fit` wiring is at the lines
  given, mirror it):
  - registry dict near line 35: add the name with a one-line description;
  - option check near line 301: same `background='model'` and `outline_blur` checks (no rounds);
  - line 643: add the name to the methods that `measure_cells` does not handle;
  - after the `outline_fit` block (line 687): call the adapter below with `all_seeds`, `report`,
    `origins[report]`, `blur=outline_blur`, `invert`, `n_rays=n_angles`, `backend`, `device`; time it into
    `seconds["outline_fit_fast"]`;
  - line 704: its `background` metric is its own fitted level, like `outline_fit`'s.
- [ ] **Adapter** in `subpx/outline.py`, beside `measure_outline`:
  `measure_outline_fast(image, seeds_xy, rows, origins_xy, *, pixel_size, unit, blur, invert, n_rays, backend,
  device) -> (CDMethodResult, summary)`.
  - Fewer than `MIN_HOLES` seeds: every row invalid, status `too_few_holes`, summary `{"fitted": False,
    "reason": ...}` (as `measure_outline`).
  - No GPU in this run: every row invalid, status `needs_gpu`, summary `{"fitted": False, "reason": "needs the GPU
    (CuPy); this run uses the CPU"}`. No exception: the other methods of the image must not be lost.
    (`cd_fast_report.py` validates its options with an 8 x 8 image on the CPU; that call must pass.)
  - Otherwise: `CDMethodResult(method="outline_fit_fast", ...)` with every reported hole reading the page:
    `cd_equivalent_px` = the page's CD, `cd_x_px`, `cd_y_px`, `area_px2`, `perimeter_px` from the page's outline;
    `brightness`, `background` (the hole's level), `leftover_rms` per hole; `valid` where the centre is finite;
    `status` `page_size` for valid holes, `not_finite` otherwise; `geometry` `center_local` and `edge_px` (the
    page's rays repeated per hole) and contours through `radial_contours`, exactly as `outline_measurement` does.
    Summary `{"fitted": True, **fit.summary}`.
  - `fast_summary(meta, pixel_size, unit)` and `fast_line(summary, ascii_only)` beside `fit_summary` /
    `fit_line`. The line, in the report's unit: outline CD, blur FWHM x and y with the family, exponent p, the
    leftover against the noise, the first fit's CD, holes stacked of holes fitted, seconds; and "page size for
    every hole" so that nobody reads a spread into it.

### 5.2 Folder reports

`scripts/cd_report.py` (existing `outline_fit` wiring at lines 22-25, 122-128, 150-164, 434-444, 503, 613, 632-643,
677-682, 711-718, 738, 776-784) and `scripts/cd_fast_report.py` (lines 6-7, 88-93, 104, 165, 185, 194-196,
229-231). Mirror each place; do not change what `--outline-fit` does.

- [ ] Flag `--outline-fit-fast` in both scripts; `--outline-blur` applies to both methods (say so in its help).
  `OPTIONAL_METHODS` is used as "what `--outline-fit` adds" in two places (`cd_report.py` line 711,
  `cd_fast_report.py` line 185): select the two names by their own flags.
- [ ] `cd_statistics`: the rule "every hole was given the image's size: the mean is a reading, the spread is not"
  (lines 150-155) applies to `outline_fit_fast` on every image (`per_hole_size` is always False): its `std` and
  `three_sigma_pct` are NaN, shown as "-".
- [ ] Description (dict near line 122), colour (line 503), the console line per image and the line on the image
  tab ("Outline fit (fast): ..."), a note when it was not fitted. One general note for the report, once, not per
  image: `outline_fit_fast` gives every hole the image's size (no within-image sigma).
- [ ] **One comparison line** when both methods ran, on the console at the end and among the report's notes:

      outline_fit_fast - outline_fit: mean +0.08, largest 0.14 nm over 12 images; 0.3 s against 95 s per image

  (difference of the two page CDs per image; ASCII; at most about 150 characters: this is the line the user types
  back from the remote). No other statistics.
- [ ] The hole panel of `index.html`: nothing special; the generic branch (the cell with a circle at the CD) is
  enough. Check that the map and the click panel do not break for the new method.

### 5.3 Docs

- [ ] `docs/outline_fit.md`: a section for `outline_fit_fast` (what it is in three sentences, the command, what the
  line means, GPU only, page values only, its limits from the notes' section 7, time). The module docstrings of the
  two report scripts; `measure_cds`' docstring.
- [ ] `tasks/todo.md`: tick the items of the section "Outline fit in one pass"; add the review.

### 5.4 Tests (`tests/test_outline_fit_fast.py`)

GPU tests skip without a CUDA device, with the pattern of `test_gpu_fit_matches_the_cpu_fit`
(`tests/test_outline_fit.py` line 139). Numbers in brackets are the prototype's.

No GPU needed:

1. The analytic template against `Template.render` on the template's own box, for `gauss2` and `lens`, with and
   without an edge offset and the bin box: largest difference below 3e-4 of the peak [1.2e-4;
   `check_spec.py`].
2. `ShapeFitter` under NumPy: the model's derivatives against central differences at a clearly non-round theta
   (the turn angle has no effect on a round outline): relative error below 1e-3 per parameter [2e-4 for the turn
   angle, 5e-5 for the exponent, 5e-6 and smaller for the rest; `port_checks.py`].
3. `ShapeFitter.fit` under NumPy on a stack synthesised from the model (a hole without neighbours: pair maps of
   ones; known theta; noise of 0.16 grey levels per bin on a hole of 180): the outline CD within 0.05 nm of the
   true one [-0.016 gauss2, -0.020 lens], the same from the grid of starts and from a start at the truth (1e-3
   nm), chi-square per bin 1.0; 0.3 s on the CPU [`port_checks.py`].
4. `measure_cds(..., methods=("outline_fit_fast",))` on a page without holes (16 x 16 zeros) and on the CPU with
   holes: no error, `meta["outline_fit_fast"]["fitted"] is False`, the reason names the cause, the other methods'
   numbers and the seeds are unchanged by adding the method.
5. Input checks of `fit_outline_fast` (too few seeds, unknown blur, `backend="cpu"`).

GPU:

6. The library's own test page (`render_hole_array((120, 170), circle 100 nm, pitch 210 nm, cd_sigma 3, seed 5)`:
   pitch exactly 10 px, 187 seeds from `measure_cds`, 135 stacked): page CD within 1.5 nm of the mean true CD
   for both blur forms [+0.02 nm gauss2, +0.31 nm lens], leftover chi-square per bin between 0.5 and 2 [0.9];
   statuses all `page_size`; valid share above 0.95; the centres within 1.5 nm rms of the truth.
7. A 300 x 400 px page of the same pattern (seed 6; 999 stacked): within 0.8 nm [-0.28 gauss2, +0.16 lens]. A
   600 x 800 px page, pitch 200 nm (seed 7; 4,941 stacked): within 0.5 nm [+0.09, -0.06]. With the method's own
   seeds as well as with `measure_cds`' seeds.
8. The same image fitted twice: page CD equal to 1e-3 px.
9. `cd_statistics`: mean finite, `std` and `three_sigma_pct` NaN; `fast_line(..., ascii_only=True).isascii()`.
10. Both reports with `--outline-fit --outline-fit-fast` on one PNG (the pattern of
    `test_reports_measure_the_outline_fit_on_request`, line 176; `--outline-rounds 2` keeps `outline_fit` short):
    `summary.csv` and `between_images.csv` hold both methods, `index.html` holds both lines and the comparison
    line.
11. `save_cd_results` / `load_cd_results` round trip with the new method (compact export rebuilds its contours).

### 5.5 Check the port against the prototype (before calling it done)

- [ ] Full-size generated pages (`truth_gpu.py` shows how to make them; point it at `subpx.fit_outline_fast`):
  the table in `docs/outline_fit_one_pass.md` section 5.2. Lens rows within 0.01 nm of the table. Gauss2 rows
  within 0.01 nm too, except the two that depend on the start (80 nm circles, and 100 nm circles at blur 72 nm:
  0.1 nm; notes 6.13).
- [ ] The 40 hole pages of the test file with the method's own seeds: page CD within 0.005 nm of the prototype's
  (`artifacts/outline_fit/fast/runs/file_l2_v4.json`, on this PC only; mean 100.80 nm gauss2, 100.74 nm lens).
- [ ] Time per image for steps A to C alone on a full page (39,000 holes), kernels compiled, one image per call:
  about 0.23 s with gauss2 and 0.12 s with lens [`port_checks.py`] (0.08 s per page when 20 pages are fitted
  together, which the reports do not do).
- [ ] Both reports on a folder of three or four pages of the test file saved as PNG, with both flags: read the
  console as the user will.
- [ ] `pytest tests/` (all of it), then commit and push.

### 5.6 What to hand the user when it is built

One command for the remote and what to type back:

    python scripts\cd_fast_report.py D:\data\png_images --output D:\data\report_fast_cmp --pixel-size 21 --unit nm --outline-fit --outline-fit-fast

- A folder of about ten images first: `outline_fit` takes 1.5 to 4.5 minutes per image.
- Type back the last line (`outline_fit_fast - outline_fit: ...`). Everything else is in `summary.csv`,
  `between_images.csv` and `index.html` on the remote.
- Tell the user what to expect: on generated pages the new method read 0.05 to 0.3 nm above `outline_fit` with the
  same blur form, and was the closer one to the truth; the first run compiles the GPU routines (about half a
  minute, once per machine).
- If the two disagree by more than that on real images: the line of each image shows the leftover against the
  noise (near 1 on generated pages) and the first fit's CD; a leftover far above 1 means the model does not fit
  the real average hole, which is the thing to look at next (with the stack saved by `--save-results`).

## 6. Things that will bite (all met in the prototype)

1. **A pitch of a whole number of pixels leaves most sub-pixel classes empty** (the library's test page: 10 px).
   The grid of starts must be compared on each page's most populated class, and the first guess of the size must
   come from the stack's second moment over its filled bins; otherwise the first fit lands on a 300 nm outline.
   Both are in the prototype now (`ShapeFitter.grid_start`, `GpuPage.start_theta`).
2. **Few holes.** With 21 stacked holes the page CD is about 1 nm off, with 135 it is within 0.6 nm. That is
   statistics, not a defect; keep `MIN_HOLES` seeds as the limit, report `holes_stacked`, and do not promise more.
3. **gauss2 depends on the start at the 0.1 nm level** on some pages (80 nm circles, large blur): its second
   Gaussian is not determined by a blur that has none. The lens form is stable to 0.002 nm. Compare the two
   methods with the same `--outline-blur`; the default stays `gauss2` because `outline_fit`'s is.
4. **The weak priors on the blur's second part need the stack's weights in 1 / variance units** (the noise
   estimate is the median leftover of a window against the scaled stack; it is about 30 % high, which is fine).
5. **Do not centre holes against the stack** (+0.38 nm) and **do not divide a window by its hole's fitted
   brightness** (-0.09 nm). Plain mean, centring by the hole pass.
6. **The outermost bins of the stack are filled by part of a class**: only bins with `|n| <= floor((R - 0.5) U)`
   enter the fit.
7. **Bounds as an active set.** Clipping a solved Levenberg-Marquardt step made the fit stall; a parameter with no
   curvature (the turn angle of a round outline) needs a floor on its damping diagonal.
8. **The pair reach, the fit's box and the template box follow the window** (`reach = R + 13`,
   `L = 8 ceil((2 reach + 8) / 8)`, template box `max(32, 2 (R + 6))` px); with fixed values a 250 nm pattern lost
   its neighbours from the model.
9. **Holes cut by an area boundary** bias the page if they are modelled as whole holes (-0.25 nm in an artificial
   test). Not a phase 1 case (whole images); it is one for the tiles of phase 2.
10. **The method does not know a page without the pattern**: it fits noise and reports CD 50 nm, blur 14 nm with a
    good leftover. In the reports the image is the user's choice; in the batch the hole check stays in front.
11. **First use compiles the kernels** (tens of seconds, cached in `%USERPROFILE%\.cupy`): time a second call.
12. **The test file is generated**, like every page used so far. Nothing here is evidence about real images.

## 7. Evidence so far (one paragraph; details in the notes)

Prototype on this PC (RTX 4060 Ti): 80 ms per page for the page values with 20 pages per call, on the 40 hole
pages of the test file. On 17 generated full pages of 80 nm and larger with known sizes, the page CD is closer to
the truth than `outline_fit`'s on 15 and equal on 2 (lens form: +0.01, -0.23, +0.09, +0.12, +0.14, -0.13, 0.00 nm
against -0.20, -0.23, +0.79, +0.19, +0.14, -2.01, -0.11). On the test file it reads 0.11 +- 0.02 nm above
`outline_fit`. Small generated images with the report's seeds: +0.02 / +0.31 nm (135 holes), -0.28 / +0.16 (999),
+0.09 / -0.06 (4,941) for gauss2 / lens, where the library's `outline_fit` reads +0.08, -0.16, -0.52.

## 8. Phase 2, after the user's verdict on real images: the TIFF batch

Not to be built before the user says the method is good enough. What is known:

- **Throughput.** 80 ms per page on one RTX 4060 Ti with 20 pages per call; one CPU core per GPU process is busy
  all the time and the GPU about 70 % of it. The remote has 64 threads, so two processes per GPU are affordable if
  one leaves the GPU idle. Target: 1,500 pages in under a minute on 4 GPUs = 160 ms per page and GPU.
- **Shape.** One reader (the image is read front to back, or loaded whole as the remote does now), the hole
  check, then per GPU a process that takes 20 pages per call (`fit_pages`), then one row per page. No pool of
  workers per GPU (Windows gives them turns; measured).
- **The area of a page.** Seeds are found on the whole page (light from outside the area must be modelled); only
  holes whose window lies inside the area are stacked and counted; holes cut by the area's border are a problem
  (section 6.9), so an area made of tiles needs its border holes left out of the stack, not out of the model.
- **Which areas: two options to put to the user.**
  1. Keep the remote's tile check as it is and hand its valid (N, m) table to the fit as the area. No risk, no new
     validation; the fit only needs the tile grid.
  2. Let the fit select holes instead of tiles, with what it computes anyway: fit the reference page once (its
     average hole with the neighbours in it, at the reference's pitch), then for every page test each found hole's
     window against that average hole (brightness and level free; the kernel `hole_vs_stack` does this in a few
     milliseconds per page). A hole belongs to the pattern if its leftover is within a few noise levels; a
     different pitch, shape or a blank area fails the same test. It works hole by hole (no tile edges, partial
     tiles count), needs no 4-D array and no FFT per tile, and is not thrown by brightness flicker. It is
     untested on real pages: the threshold has to be set on the remote, and it should run beside the tile check
     for a while (both selections logged per page) before replacing it.
- **Not known:** whether a kept tile always holds the reference's pattern only (asked twice, not answered); the
  remote's time per page.

## 9. Phase 3, only if the method proves itself: a size per hole

Step D exists in the prototype (`fit_holes(sweeps=3, size=True, relax=1.3)` after `fit_noise()`; `results()`):
response 1.00 on generated pages, 137 ms per page. To carry over from the library when it is built: the rule that
switches per-hole sizes off when a hole's offset is not readable (median standard error above 0.3 outline radii),
the offset bound and the `edge_offset_at_clip` status, `cd_se_px`, the mean offset taken out, x / y widths per
hole.
