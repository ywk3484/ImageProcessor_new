# `outline_fit_fast`: handoff for the build

**Status, 2026-11-12.** The user approved building the one-pass outline fit as a new method beside `outline_fit` and
asked for this handoff instead of an implementation in the session that proposed it. Nothing in `subpx/` or
`scripts/` has been changed. A fresh session builds from this file.

**Updated later the same day**, after the prototype was run the way the reports will use it (their seeds, small
images, many starts): the fit can end in more than one place on some pages (section 6.3), the prototype gained one
step for the gauss2 form (`pipeline.in_order`, section 6.4), and what the neighbours' reach does is measured
(section 6.13). Sections 3, 4, 5.1, 5.2, 5.4, 5.6, 6 and 7 changed with it.

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
       gauss2 only: a fit that ends with its two Gaussians exchanged is run once more from the exchanged labels

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
| `pipeline.py` | `fit_pages` / `fit_page`: steps A to D for a batch of pages; `in_order`, `exchanged`, `second_is_narrower`: the gauss2 exchange step after C | library: the driver behind the public function |
| `truth_gpu.py`, `truth_matrix.py`, `small_pages.py`, `port_checks.py`, `run_file.py`, `seed_test.py`, `checks.py`, `check_spec.py` | the measurements (generated pages with known sizes, small images with the report's seeds, the numbers quoted for the tests, the test file) | stay scratch; reuse them to check the port |
| `basins.py`, `reach.py`, `ring.py` | where the final fit can end from 81 starts; the pair reach (ends and time); holes near the image border against inner holes | stay scratch; `basins.py` is the check for any change to the fit's starts |

Prototype defaults that the numbers were made with: 4 bins per pixel, pair grid 8 per pixel, band `|f| <= 0.6`,
256 outline points, plain ("raw") stack, single-precision templates, first fit `round` (10 iterations at most) and
`super` (12) with `ftol 1e-6`, final fit `full` (30, `ftol 1e-7`), two centring sweeps, pair reach `R + 13` px, the
gauss2 exchange step on.

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
  page fits in lockstep): phase 2 needs it. With it:
  - the gauss2 exchange step after C (`pipeline.in_order`; section 6.4). Check: tests 6 and 9 of section 5.4;
  - the pair reach as one named setting, `R + 13` px (section 6.13 has what 30 px does; do not change the default
    in this build);
  - the prototype's two caches: the library `Template` used for the parameter bookkeeping, one per blur form
    (`pipeline.lib_template`; 14 ms to build, against 0.12 s for a whole image with the lens form), and the
    `ShapeFitter` per window size (`pipeline.fitter_for`).
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
    `centers_xy` (B's centres plus the centroid of the final outline, `Template.centroid_px(theta_1)`, as
    `PageFit.results` does; the prototype's `results()` leaves it out, it is below 1e-3 px on its pages),
    `brightness`, `level`, `rms`, `good` (stacked); per page `outline_cd_px`, `cd_x_px`, `cd_y_px`
    (`ellipse_widths(template, theta, zeros(1))`), `rays_px` (n_rays,) of the page's outline (`outline_rays` with
    a zero offset), `blur_fwhm_px`, `theta`, `stack`, `stack_weight`, `upsample`, `blur`, `summary`, `pixel_size`,
    `unit`.
  - `summary` (JSON-ready, same key names as `fit_outline`'s where the meaning is the same): `blur_family`,
    `outline_cd_px`, `outline_extent_x_px`, `outline_extent_y_px`, `superellipse_p`, `outline_turn_deg`,
    `harmonics`, `blur_fwhm_px`, `blur_rms_px`, `holes_fitted`, `holes_stacked`, `spacing_px`, `window_px`,
    `noise_sd`, `leftover_chi2` (chi-square per bin of the final page fit: 1 = the stack is fitted to its noise),
    `first_cd_px` (the first page fit's CD), `fit_iterations`, `converged` (the final fit stopped by its tolerance,
    not by its iteration cap), `exchanged` (gauss2: the exchange step replaced the final fit), `per_hole_size:
    False`, `seconds`, `backend`, `theta`, `lengths: "px"`.
  - `blur_second(theta, blur) -> str` beside it: the blur's second part of a `theta` of either method in words
    ("pupil cut-off 0.49 cycles/px" for lens; "second Gaussian 2.2 x 2.2 px, 12 % of the light" for gauss2; empty
    for a form without a second part). It is what tells two end points of the fit apart (section 6.3); `fast_line`
    and the reports' comparison use it.
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
    `fit_line`. The line, in the report's unit: outline CD, blur FWHM x and y with the family, the blur's second
    part (`blur_second`), exponent p, the leftover against the noise, the first fit's CD, holes stacked of holes
    fitted, seconds; and "page size for every hole" so that nobody reads a spread into it.

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
- [ ] **Per image, when both methods ran:** one note under the two methods' lines (console and image tab) with the
  blur's second part of each, from `blur_second` on the two `meta[...]["theta"]`:

      blur second part: outline_fit second Gaussian 2.0 x 2.0 px, 19 %; outline_fit_fast 3.5 x 3.6 px, 25 %

  It is a new line; what `--outline-fit` prints stays as it is. It is how a difference that comes from the fit's
  two possible end points is told from a real disagreement (section 6.3).
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
6. The gauss2 exchange (`exchanged`): for a theta with the second Gaussian the narrower one and a weight of 0.45,
   the exchanged theta has it the wider one, the template's spectrum is the same [difference 0], and exchanging
   twice gives theta back [0]; a weight of 0.05 becomes 0.6, the bound [`port_checks.py`, part 4].

GPU:

7. The library's own test page (`render_hole_array((120, 170), circle 100 nm, pitch 210 nm, cd_sigma 3, seed 5)`:
   pitch exactly 10 px, 187 seeds from `measure_cds`, 135 stacked): page CD within 1.5 nm of the mean true CD
   for both blur forms [+0.63 nm gauss2, +0.31 nm lens], leftover chi-square per bin between 0.5 and 2 [0.9];
   statuses all `page_size`; valid share above 0.95; the stacked holes' centres within 1.5 nm rms of the truth,
   both axes together [0.53 nm, from seeds that are 3.1 nm off; `small_pages.py`].
8. A 300 x 400 px page of the same pattern (seed 6; 999 stacked): within 0.8 nm [+0.34 gauss2, +0.16 lens]. A
   600 x 800 px page, pitch 200 nm (seed 7; 4,941 stacked): within 0.5 nm [+0.09, -0.06].
9. The same three images with `measure_cds`' seeds and with the method's own seeds (`seeds_xy=None`): the gauss2
   page CD of the two within 0.05 nm [0.005, 0.005, 0.004 nm; without the exchange step 0.62, 0.62, 0.40 nm: this
   is the test of that step]; the lens page CD within 0.05 nm [0.02 nm at most].
10. The same image fitted twice: page CD equal to 1e-3 px.
11. `cd_statistics`: mean finite, `std` and `three_sigma_pct` NaN; `fast_line(..., ascii_only=True).isascii()`.
12. Both reports with `--outline-fit --outline-fit-fast` on one PNG (the pattern of
    `test_reports_measure_the_outline_fit_on_request`, line 176; `--outline-rounds 2` keeps `outline_fit` short):
    `summary.csv` and `between_images.csv` hold both methods, `index.html` holds both lines, the note with the
    blur's second part of both, and the comparison line.
13. `save_cd_results` / `load_cd_results` round trip with the new method (compact export rebuilds its contours).

The brackets of 7 to 9 are errors of the page CD on generated images, not targets: with 135 holes the fit's own
standard error is about 0.25 nm (section 6.2).

### 5.5 Check the port against the prototype (before calling it done)

- [ ] Full-size generated pages (`truth_gpu.py` shows how to make them; point it at `subpx.fit_outline_fast`):
  the table in `docs/outline_fit_one_pass.md` section 5.2. Rows of well determined pages within 0.01 nm of the
  table for both blur forms. The pages with more than one end (section 6.3: 80 nm circles, 100 nm circles at blur
  72 nm) within 0.01 nm of one of the ends that `basins.py` lists for them; the small dense patterns (60 and 50 nm
  holes, 90 x 60 nm ellipses) within 0.1 nm (they repeat no better: section 6.15).
- [ ] The 40 hole pages of the test file with the method's own seeds: page CD within 0.005 nm of the prototype's
  (`artifacts/outline_fit/fast/runs/file_l2_v5.json` and `file_l2_lens_v5.json`, on this PC only; mean 100.80 nm
  gauss2, 100.74 nm lens).
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
- Tell the user what to expect: on generated full pages the new method read 0.05 to 0.3 nm above `outline_fit`
  with the same blur form, and was the closer one to the truth; on small generated images (135 to 5,000 holes) up
  to 0.6 nm above it; the first run compiles the GPU routines (about half a minute, once per machine).
- Compare with the default blur form (gauss2) first: it has one end point on most pages (section 6.3).
- If the two disagree by more than that on real images, three things to read on each image, in this order:
  1. the note with the blur's second part of both methods: clearly different second parts mean the two fits sit
     at different end points of the same model (section 6.3); that is a question about the model, not about the
     new procedure;
  2. the new method's leftover against the noise (near 1 on generated pages): far above 1 means the model does
     not fit the real average hole (look at the stack saved by `--save-results`);
  3. the first fit's CD against the final one.

## 6. Things that will bite (all met in the prototype)

1. **A pitch of a whole number of pixels leaves most sub-pixel classes empty** (the library's test page: 10 px).
   The grid of starts must be compared on each page's most populated class, and the first guess of the size must
   come from the stack's second moment over its filled bins; otherwise the first fit lands on a 300 nm outline.
   Both are in the prototype now (`ShapeFitter.grid_start`, `GpuPage.start_theta`).
2. **Few holes.** Generated images, error of the page CD, gauss2 / lens: 21 stacked holes +0.95 / +1.17 nm, 135
   holes +0.63 / +0.31, 999 holes +0.34 / +0.16, 4,941 holes +0.09 / -0.06. The fit's own standard error of the
   page CD (from its curvature; `small_pages.py`) is 0.5 nm at 21 holes, 0.24 at 135, 0.08 at 999, 0.04 at 4,941:
   the errors are 1 to 4 of those. Holes near the image border do not read differently from inner ones beyond
   that scatter (`ring.py`), so the border is not the cause; nothing else was pinned down. Keep `MIN_HOLES` seeds
   as the limit, report `holes_stacked`, and do not promise more than this.
3. **On some pages the final fit has more than one end point, and its start decides** (`basins.py`: 81 starts
   round the pipeline's end on 10 full pages, the final stage run to its end; table in the notes, 6.13).
   - lens: one end on 3 pages (150 nm squares, 160 x 80 nm ellipses, 250 nm circles), two on 7: one with the
     pupil's cut-off near 0.5 cycles/px and one with the pupil outside the band (cut-off 0.75 to 8) and a wider
     smear. Apart in page CD: 0.02 nm (100 nm circles, staggered), 0.04 (test file), 0.19 and 0.21 (100 nm circles
     at blur 65 and 72 nm), 0.57 and 0.65 (80 nm circles).
   - gauss2, with the step of 4: one end on 7 pages; several on 3 (100 nm circles at blur 72 nm: +0.09 and -0.16
     nm; 80 nm circles: -0.82 to -0.74; 80 nm at blur 72 nm: -0.73 to -0.52).
   - The pipeline starts where the library's fit starts (cut-off 0.5; second Gaussian twice as wide, weight 0.05):
     keep that, so that the two methods are compared like for like. The generator's cut-off is 0.480, so on
     generated pages the lens form starts next to the generator's own end: **the lens errors quoted in section 7
     and on the page are flattered**; at the other end the 100 nm circles read -0.17 instead of +0.01 nm.
   - This belongs to the model (one outline, one free blur shape), not to the one-pass procedure; whether
     `outline_fit` ends at the same end on a given page was not tested. Nothing to build for it in phase 1 except
     `blur_second` in the lines (section 5.2), which makes it visible. Choosing between ends (two starts, lower
     cost) is not built and would pick the wrong one at the default reach (13).
4. **gauss2: the exchanged half.** The two Gaussians can be exchanged without changing the blur, but the bounds
   (second weight at most 0.6) and the priors are written for a wide, weak second one. A final fit that drifts to
   a narrower second Gaussian stops on the weight's bound, short of the minimum (cost 23 to 96 above it on small
   images), with a page CD 0.4 to 0.6 nm away. Whether it drifts depends on the first fit, and that on the seeds:
   with `measure_cds`' seeds against the method's own, the page CD of the small images differed by 0.62, 0.62 and
   0.40 nm. The step (`pipeline.in_order`): after C, a page with `log s2x + log s2y < log s1x + log s1y` is fitted
   again (`full`, 30 iterations) from the exchanged labels (`s1 <-> s2`, `logit w2 -> -logit w2`, clipped to the
   bounds), pages together; the end with the lower cost is kept. After it: 0.005, 0.005, 0.004 nm. On full pages
   2 of 58 ended exchanged (one moved by 0.02 nm); the test file's page CDs changed by 0.0004 nm at most and the
   time not at all. The library's `outline_fit` does not end exchanged on these images (checked). It ends with a
   different second Gaussian (1.6 to 2.3 px wide with 14 to 42 % of the light, where the new method's minimum has
   3.3 to 3.5 px and 21 to 25 %) and reads 0.5 to 0.6 nm below the new method there; whether that is an earlier
   stop on the way to the same minimum was not tested.
5. **The weak priors on the blur's second part need the stack's weights in 1 / variance units** (the noise
   estimate is the median leftover of a window against the scaled stack; it is about 30 % high, which is fine).
6. **Do not centre holes against the stack** (+0.38 nm) and **do not divide a window by its hole's fitted
   brightness** (-0.09 nm). Plain mean, centring by the hole pass.
7. **The outermost bins of the stack are filled by part of a class**: only bins with `|n| <= floor((R - 0.5) U)`
   enter the fit.
8. **Bounds as an active set.** Clipping a solved Levenberg-Marquardt step made the fit stall; a parameter with no
   curvature (the turn angle of a round outline) needs a floor on its damping diagonal.
9. **The pair reach, the fit's box and the template box follow the window** (`reach = R + 13`,
   `L = 8 ceil((2 reach + 8) / 8)`, template box `max(32, 2 (R + 6))` px); with fixed values a 250 nm pattern lost
   its neighbours from the model.
10. **Holes cut by an area boundary** bias the page if they are modelled as whole holes (-0.25 nm in an artificial
    test). Not a phase 1 case (whole images); it is one for the tiles of phase 2.
11. **The method does not know a page without the pattern**: it fits noise and reports CD 50 nm, blur 14 nm with a
    good leftover. In the reports the image is the user's choice; in the batch the hole check stays in front.
12. **First use compiles the kernels** (tens of seconds, cached in `%USERPROFILE%\.cupy`): time a second call.
13. **The pair reach decides which lens end fits better** (`reach.py`). The lens blur has faint far rings; a
    neighbour's rings beyond the reach are missing from the stack's model and the blur's shape makes up for them.
    At `R + 13` (21 px) the end with the generator's own pupil has the higher cost on all 4 generated pages tried
    (by 22 to 533); at 30 px it has the lower cost on 3 of them (by 25 to 39), and at 40 px the fitted blur of the
    staggered page is the generator's within 0.4 % (smear 1.007 / 1.011 px for 1.0072, cut-off 0.4799 for 0.4802).
    Inside one end the page CD moves by 0.02 nm at most with the reach; gauss2 does not depend on it (0.002 nm).
    Time for the page values, 20 pages per call: gauss2 80.5 / 92.5 / 105.0 ms per page at 20 / 30 / 40 px, lens
    73.4 / 78.6 / 85.5. The generator draws every hole's light on a 64 px patch, so what a real page needs is not
    known.
    **Not adopted:** the default stays `R + 13` in phase 1 (the numbers of section 7 were made with it); put the
    choice to the user with the result on real images.
14. **The test file is generated**, like every page used so far. Nothing here is evidence about real images.
15. **Poorly determined pages repeat worse.** Two runs of the same page agree to 0.0003 nm where the page has one
    end; to 0.007 nm on the pages with several (100 nm at blur 72 nm, 80 nm at blur 72 nm); to 0.02 to 0.07 nm on
    the small dense patterns (60 and 50 nm holes, 90 x 60 nm ellipses). The hole kernels add in an order that
    differs from run to run.

## 7. Evidence so far (one paragraph; details in the notes)

Prototype on this PC (RTX 4060 Ti): 80 ms per page for the page values with 20 pages per call, on the 40 hole
pages of the test file. On 17 generated full pages of 80 nm and larger with known sizes, the page CD is closer to
the truth than `outline_fit`'s on 15 and equal on 2 (lens form: +0.01, -0.23, +0.09, +0.12, +0.14, -0.13, 0.00 nm
against -0.20, -0.23, +0.79, +0.19, +0.14, -2.01, -0.11). On the test file it reads 0.11 +- 0.02 nm above
`outline_fit`. Small generated images with the report's seeds: +0.63 / +0.31 nm (135 holes), +0.34 / +0.16 (999),
+0.09 / -0.06 (4,941) for gauss2 / lens, where the library's `outline_fit` reads +0.08, -0.16, -0.52.

Read those numbers with two limits. The lens errors are flattered by the generated pages, whose pupil is where the
fit starts; with gauss2 (not flattered: the generator's blur is not of that form) the full pages read -0.14 (100 nm
circles), -0.82 (80 nm), +0.11 (150 nm squares), +0.06 (250 nm), -0.26 (160 x 80 nm ellipses) nm. And on 7 of 10
full pages the lens form has a second end point 0.02 to 0.65 nm away (section 6.3).

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
  (section 6.10), so an area made of tiles needs its border holes left out of the stack, not out of the model.
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
- **To decide with the result on real images** (sections 6.3 and 6.13): the pair reach (30 px costs 5 to 12 ms
  per page), and whether the fit should try both ends and keep the one that fits better. On generated pages the
  second is right only with the longer reach.
- **Not known:** whether a kept tile always holds the reference's pattern only (asked twice, not answered); the
  remote's time per page.

## 9. Phase 3, only if the method proves itself: a size per hole

Step D exists in the prototype (`fit_holes(sweeps=3, size=True, relax=1.3)` after `fit_noise()`; `results()`):
response 1.00 on generated pages, 137 ms per page. To carry over from the library when it is built: the rule that
switches per-hole sizes off when a hole's offset is not readable (median standard error above 0.3 outline radii),
the offset bound and the `edge_offset_at_clip` status, `cd_se_px`, the mean offset taken out, x / y widths per
hole.
