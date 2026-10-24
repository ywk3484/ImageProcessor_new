# One-command CD report for a folder of images

Plan: `scripts/cd_report.py FOLDER --output REPORT` measures every PNG with all 7 methods and writes an offline
report: summary tab (per image x method: valid holes, mean CD, sigma, 3 sigma %), one tab per image with stats,
histograms and a pan/zoom canvas of the image with every hole drawn at its CD (per-image data lazy-loaded as JS).

- [x] `scripts/cd_report.py` (statistics, display PNG, per-image data JS, index.html with tabs + viewer)
- [x] `tests/test_cd_report.py` (summary matches direct statistics; failed image row; skipped hole-fit methods)
- [x] Browser check (Playwright): tabs, zoom, hole click, alignment
- [x] Full-size timing and report size; full test suite

## Review

Deviations from the approved plan:
- Per-hole results are opt-in (`--save-results`): the compact export with all 7 methods is ~1.2 GB per
  110k-hole image (five methods store 128 contour radii per hole).
- Summary split into a "Mean CD" and a "3σ (%)" table (images x methods); one wide table needed scrolling.
- Display images are contrast-stretched (0.5-99.5 %) for every bit depth, display only: ACTIS images span
  ~15-77 counts and looked black.

Results: 110k-hole image (GPU deblend) 71 s; report adds a 14 MB data file + 0.8 MB PNG; tab loads in 0.2 s,
full-view redraw 56 ms, zoomed 1.5 ms. Full suite: 181 passed, 9 failed = the pre-existing failures.

# Pulse-invariant CD methods: divide by background, x/y projections, blurred-hole fit

Plan: on ACTIS images every hole's brightness carries a random light-pulse fluctuation the reference cannot
remove. Three opt-in `measure_cds` methods (model background only): `flux_ratio` (hole light / own cell
background), `projection` (x/y projected profiles, width between steepest points), `disk` (fit a round hole
of diameter d imaged through an Airy blur and pixel-integrated, free brightness and background per hole;
blur from `psf_fwhm` or calibrated so the median d equals `nominal_cd`).

- [x] `subpx/_cpu/invariant_cd.py`: pixel-integrated disk table, batched LM disk fit (2 neighbour rounds),
      disk rendering + far-tail field, blur calibration, flux ratio, projections
- [x] `measure_cds(nominal_cd=, psf_fwhm=, dark_level=)`, method labels/validation, manifest definitions
- [x] CLI flags, viewer colours
- [x] `tests/test_invariant_cd.py` (exact Fourier-rendered synthetic images with per-hole pulse)
- [x] `scripts/validate_invariant_cd.py` (old vs new methods at blur 50/75/100 nm)
- [x] Full test suite, timing at full image size

## Review

Deviations from the approved plan:
- Far-tail field added (`far_field`, FFT): ~4 % of each hole's light lies beyond the 16 px rendering radius
  (slow Airy tail); summed over the lattice it read as background that the hole's own pulse does not light,
  biasing the fitted local background +8.5 % and leaking the pulse into `flux_ratio` (0.10 -> 0.07 per unit
  pulse after the fix; background bias 1.8 %, rest from holes outside the small test image).
- `projection` cd_equivalent = mean of cd_x and cd_y (the user's definition), not the area equivalent.
- More neighbour rounds than 2 changed nothing (tested 2/3/4).

Results (`scripts/validate_invariant_cd.py`, 100 nm hole, 21 nm px, 5 % pulse, calibrated by nominal CD):
disk follows true CD with slope 0.98-1.06 and 1.2/2.1/4.1 nm per-hole scatter at 50/75/100 nm blur; a 10 %
stronger pulse moves it 0.3-1.2 nm (gaussian 3.4-7.8 nm). flux_ratio similar with dark level known.
projection: 6/14/159 nm scatter. Full image (110k holes): +47 s on the CPU for disk + flux_ratio + projection.
Full suite: 179 passed, 9 failed = the pre-existing failures.

# GPU backend for CD measurement at full-image scale

Plan: ~109k holes per 2048×5408 image make the CPU pipeline take ~14–28 min/image (deblend fits, per-cell
radial methods, and ~6 min of output writing). Compact lossless outputs + an on-demand region viewer, then a
GPU backend for the `background="model"` path (FP64, NumPy at function boundaries, CPU path unchanged).

- [x] Step 1: compact outputs (`save_cd_results(compact=True)`, CLI `--outputs`), lossless contour/background
      construction data, `load_cd_results`, region viewer (`scripts/view_cd_region.py`), faster CSV writer
- [x] Step 2: `subpx/_gpu/_cuda.py` + `subpx/_gpu/deblend.py` (fused LM kernel with LU, fixed-point render,
      batched init/clean/background), `measure_cds(backend=...)`, CLI `--backend`
- [x] Step 3: `subpx/_gpu/cd.py` (ray kernel for halfmax/gradient, logquad sums, gaussian), host ellipses
- [x] Step 4: fix `subpx/_gpu/voronoi.py` grid kernel; `subpx/_gpu/cells.py` masks
- [x] Step 5: multi-image CLI, writer process, `batch_summary.csv`
- [x] `tests/test_cd_gpu.py`, `tests/test_cd_compact.py`; `scripts/benchmark_cd.py`;
      `scripts/validate_cd_accuracy.py --backend gpu`; docs

## Review

Deviations from the approved plan:
- Region viewer: `save_cd_viewer(..., region=)` crops inside the viewer writer (keeps full-image cell IDs,
  coordinates, means and areas exact) instead of a `crop_cd_result` that would re-derive a cropped `CDResult`.
- Compact NPZ keeps `masks` (small) so `load_cd_results` rebuilds `rois` exactly from `deblend_*` params.
- CPU module refactors with bit-identical results (verified against HEAD on 4 configurations, CSVs byte-for-byte):
  `clean_cells`/`roi_grids`, `background_tiles`/`_upsample_rows`, `fit_status`/`sigma_caps`; `measure_cells`
  records each contour's construction data (`CDMethodResult.geometry`); `CDResult.deblend_model`.
- `save_cd_results` = `save_cd_tables` + `save_cd_arrays`, so the batch writer process builds tables from the NPZ.
- Fixed-point render scale from the simple bound n·max|A| (no extra coverage pass): resolution ~1e-12 of the
  brightest hole.
- logquad on GPU: GPU sums of the normal equations, host LAPACK 6×6 solves, rank test λmin ≤ 1e-12·λmax.
- Full-size (1.1e5 holes) runs not done here: this Windows box had ~3–4 GB commit left (a node.exe holds 52 GB);
  measured at 1/4 size instead (cost is linear in hole count).

Results (RTX 4060 Ti): 26,750 holes, measurement 118.7 s → 3.0 s (pitch/FWHM 2), dense 1.43 ≈330 s → 4.5 s;
compact writing 1.7 s; peak GPU memory 0.63/0.93 GiB. CPU vs GPU: statuses identical, CDs ≤ 3e-8 px except
threshold flips on values within rounding of a threshold — logquad fit-fraction pixels (2/26,750 holes,
≤ 0.018 px) and, at SNR 10 dense with peak seeds, the seed revision's marginal noise seeds (507 vs 519 dropped),
which changes a few neighbours; accuracy table statistics unchanged. GPU deterministic run to run.
Full suite: 172 passed, 9 failed = the pre-existing failures minus the fixed GPU Voronoi clustered test.

# Deblended background + dense-array CD measurement (done, commit 8157790)

Plan: neighbour tails and noise bias the per-cell border background; dense arrays (pitch/FWHM ≲ 2)
have no tail-free pixels and break threshold seeding. Model every hole as a Gaussian, subtract
neighbours, estimate a smooth background map from the residual, iterate.

- [x] `subpx/_cpu/deblend.py`: batched LM Gaussian fit, model rendering, background map, `deblend()`
- [x] `detect_peaks()` with autocorrelation pitch estimate
- [x] `measure_cds`: `background="model"` (default) / `"border"`, `seeds="threshold"|"peaks"`, `gaussian` method
- [x] `measure_cells`: optional per-cell backgrounds
- [x] `CDResult.background_map`, NPZ + manifest
- [x] CLI flags, colours, ground truth, background map figure
- [x] `detect_centers(background="model")` opt-in post-pass (CPU + GPU)
- [x] `tests/test_deblend.py`; existing tests unchanged (one count assertion updated for 4 default methods)
- [x] `scripts/validate_cd_accuracy.py` + docs
- [x] End-to-end CLI run + timing

## Review

Deviations from the approved plan, found during implementation:
- Background map: the plan's "median of all residual pixels in tiles without free pixels"
  converged to *background too high, holes too narrow* at pitch/FWHM ≤ 1.6 (+14–30 % background).
  Inside a dense array the sum of overlapping Gaussians is nearly flat, so background and tails are
  not separable there. Now: background only from pixels ≥ 1.25 FWHM from every hole, interpolated
  across dense tiles; residual inference only as a flagged last resort; `background_level` override.
- Simultaneous updates oscillated at pitch/FWHM 1.4 → damped updates (0.6) with convergence stop.
- One missed/spurious seed cascaded through neighbours (2200/3142 fits diverged at 1.4) → width cap
  at the cell's equivalent radius + seed revision (drop low-amplitude, add residual maxima).
- No per-hole offset parameter (6 instead of 7): it is degenerate with width inside small cells.
- `detect_centers` edge methods raise `NotImplementedError` with `background="model"` instead of
  using `image - B`: their GPU path needs component stats that `CenterResult` does not carry.

Results: see final report in conversation; numbers reproducible with `scripts/validate_cd_accuracy.py`.
Pre-existing failures (10, unchanged): GPU tiling/radial-symmetry/isophote/voronoi/pitch tests.

# Projection cell-outline fix and between-image 3σ (2026-10-15)

- [x] Diagnose projection scatter / row alternation on real data (no code change)
- [x] Projection: sum, not mean, over the cell's columns/rows
- [x] Report: between-image 3σ % = 3 s / M × 100 (s = std of image means, N − 1), equation shown
- [x] Report: calibrate disk / flux_ratio once (first image) and reuse; new `measure_cds(flux_ratio_scale=)`
- [x] Tests, validation, browser check

## Review
- Cause: the per-column mean divided by the cell's pixel count, which changes by one pixel along the ragged
  Voronoi outline (10 vs 11): a ~10 % step in the profile moved the steepest point. Identical noise-free holes:
  std 3.1 → 1.55 nm (ideal hole in its cell, summed: 0.55 nm; the rest is imperfect neighbour removal).
  Validation at 75 nm blur: pulse +10 % moved projection −1.7 → −0.2 nm; scatter 14 → 13.5 nm (noise-dominated).
- Found while testing: per-image nominal_cd calibration forced every image's disk/flux_ratio median to the
  nominal, so their between-image 3σ was fake (0.08 % vs 0.84 % true on synthetic images; truth 0.87 nm).
- Suite: 182 passed, same 9 pre-existing failures.

# Report: hole details, map first, band diagnosis (2026-10-15)

- [x] Map at the top; method, zoom and pan kept across tabs; colour scale shared by all images (or per image)
- [x] Click panel: projection x/y profiles + curve + slope + edges; disk cell, leftover, radial fit; flux_ratio numbers
- [x] `CDMethodResult.inputs` (in memory only): disk/flux_ratio fit input cells
- [x] Band profile: every map's band means, band-to-band spread, R² with background / peak / sub-pixel position
- [x] Disk "σ from noise alone"; per-hole agreement between methods
- [x] Tests, browser check, planted-cause synthetic check, full-size timing

## Review
- 8-bit cells moved projection's steepest points (up to 1.46 px in a few holes): the detail file now carries
  projection's column/row sums in float32; the browser's spline then finds the method's edges to 1e-5 px (864/864).
- Planted causes (420x420 px, 70 nm blur, band = one hole row), band-to-band spread in nm:
  gaussian / disk / projection_x / projection_y — none 0.4/0.4/1.6/1.5; slow light over rows 2.0/0.4/1.4/1.4;
  3 % random light per pixel line 1.0/1.9/1.5/10.3; real CD bands ±4 % 1.7/3.0/2.6/2.2.
  Brightness/peak R² cannot tell real CD from a light leak (bigger holes are brighter: R² 71-92 % for real CD
  bands); the method/direction comparison can. The guide in the report says so.
- Found: console print of "–" crashed on cp949 consoles (single image → NaN between-image values); fixed (`_plain`).
- Full size (99.7k holes, GPU): 79-89 s; data 14.7 MB + detail 65 MB; tab 0.25 s, first click 0.5 s, next 3 ms.
- Suite: 183 passed, same 9 pre-existing failures.

# Shape-free half_area method, faster report (2026-10-19)

- [x] Diagnose the lopsided projection profiles (spline exists; edges come from ~2 dots per side: sampling phase, ragged cell, noise)
- [x] Compare shape-free, pulse-invariant size measures on mixed rotated shapes (circles, rectangles, triangles, L, blobs)
- [x] New method `half_area` (local level = median of the cell's outer ring; 4x cubic zoom; region connected to the peak above b + 0.5 x height; CD = 2 sqrt(area/pi))
- [x] Projection spline and half_area run on NumPy or CuPy (spline / zoom as fixed matrices, flood fill instead of labelling)
- [x] Report: only gaussian, projection, half_area; disk/flux_ratio/logquad/gradient/halfmax and their inputs/panels removed
- [x] Tests, planted-cause band check, browser check, full-size timing

## Review
- Rejected on the way (user): a round-hole profile fit and anything built on the disk fit or needing psf/nominal CD.
- Benchmark (210x210 px, mixed shapes, 5 % pulse lighting feature and surroundings), per-feature error in % of size:
  70 nm blur projection 9.9 / half_area 3.4; 90 nm blur 12.5 / 5.8. Pulse leak half_area 0.01 / 0.03.
  +20 % global light: +0.11 / +0.20 %. +2 % true size reads +1.7 / +1.0 % (half the real change at 90 nm blur).
- GPU and CPU results identical (projection exact; half_area CDs and contours equal).
- Planted causes (band = 10 px rows, spread in nm): half_area none 0.44, line light 2.0, real CD bands 1.9;
  sub-pixel R^2 at chance (not pixel-locked). projection_y still singles out line-wise light (10.3).
- Full size (99.7k features, 2048x5408): report 10.8-11.5 s per image on GPU (was 79-89 s); half_area alone
  2.2 s GPU / 17.5 s CPU; data 7.2 MB + detail 45 MB per image.
- Suite: 187 passed, 1 skipped, same 9 pre-existing failures.

# half_area_min, half_area_min_avg, x/y statistics, GPU setup (2026-10-20)
- [x] half_area_min: local level = minimum of the cell edge (user: better repeatability on real data)
- [x] half_area_min_avg: peak = mean of the brightest 4 px' worth of the enlarged cell (--peak-pixels)
- [x] half_area family cd_x/cd_y: equal-area ellipse diameters from the region's x/y second moments (were 1/4-px extents)
- [x] Report: CD, x and y statistics in every table, summary.csv, between_images.csv; x/y map layers for every method
- [x] Report prints the backend used and why not the GPU; docs/gpu.md setup section

## Review
- Synthetic (90 nm blur, 0.3 % pixels spiked +15..40): CD error half_area_min 10.2 % -> half_area_min_avg 6.3 %
  (peak_pixels 1/2/4/8: 8.3/7.1/6.3/6.1 %; 8 lowers the size response, so 4 is the default). No spikes, disks: 4.6 -> 3.7 %.
- Extra 10 nm blur along x only: half_area_min_avg x width +5.2 %, y width -0.2 %.
- Full size GPU: 12-13 s per image (5 methods); data 15 MB + detail 57 MB per image.
- Suite: 190 passed, 1 skipped, same 9 pre-existing failures. Browser: map, layers, panel, x/y tables OK.

# tail_area and cell intensity (2026-10-20)
- [x] tail_area: low threshold (default 0.3 of the height), local level = mean of the lowest quarter of the cell edge, averaged peak
- [x] Cell intensity = tail_area's mean light above the local level over its region: report statistics, map (counts), CSV rows, correlation with every CD
- [x] Tests, full-size run, browser check

## Review
- Synthetic, 90 nm blur, noise proportional to brightness (10 %): per-feature error, threshold 0.5 / 0.3 / 0.2 with the
  lowest-quarter level: 9.2 / 6.7 / 6.5 %; valid 100 / 99.4 / 89.5 % (at 0.2 the region reaches the cell edge at
  10.5 px pitch). Minimum level at 0.3: 8.1 % (the level weighs 0.7 in the threshold, one pixel is too noisy).
  Mild noise only: 0.3 4.8 % vs 0.5 4.8 % (no gain). Real size +2 % reads +0.9 % at 0.3 (vs +1.0 % at 0.5).
- Full size GPU: 15 s per image (6 methods + intensity); data 18 MB + detail 66 MB per image.
- Suite: 191 passed, 1 skipped, same 9 pre-existing failures.

# Wide tail_area window, outlines on the map (2026-10-20)
- [x] tail_area on a wider window (0.8 pitch half-width) of image - background - all other holes' models; region may
      pass the cell; local level = median of the hole's own territory beyond 0.6 pitch; default level 0.2
- [x] Map: each hole's measured outline (32 directions, lazily loaded per method) coloured by size when zoomed in
- [x] Tests, full-size run, browser check

## Review
- Cause of the old failures (measured): the hole's own light at 20 % height was as wide as its cell, and the level
  taken from the cell edge held that light. Wide window: 100 % measured at levels 0.1-0.5.
- Synthetic, 10 % brightness noise: level 0.2 error 5.3 % (old cell: 6.5 %, 10 % of holes lost). Pulse leak 0.15 with
  each hole's surroundings lit by its own pulse as a sharp patch; ~0.02 without that patch.
- Full size GPU: 17 s per image; outline files 4 MB per method per image, loaded on demand.
- Suite: 194 passed, 1 skipped, same 9 pre-existing failures.

# tail_area: light sharing instead of neighbour subtraction (2026-10-20)
- [x] Cause of mass failure (reproduced): subtracting Gaussian neighbour models leaves 40-80 % of a neighbour's peak
      for big / square spots; at a low threshold the region spills over it to the window edge
- [x] Tried a learned spot shape (empirical PSF): no better; on a regular array the split of light between
      neighbours is not determined by the image (any shape needs an assumption)
- [x] Fix: each pixel's light shared among holes in proportion to their models; window 1.2 pitch half-width
- [x] Panel shows the wider window with the ordinary cell dashed

## Review
- Stress set, level 0.2, share measured (subtract -> share, window 0.8 -> 1.2): mixed 150 nm 62 % -> 100 %,
  squares 150 nm 56 % -> 97 %, mixed 100 nm blur 130 nm 94 % -> 100 %, mixed 100 nm 100 %.
- Pulse leak ~0.02 without per-hole lit patches; 0.23 with the synthetic sharp lit patches.
- Full size GPU: 21 s per image; tail window file 39 MB per image (loaded on opening its panel).
- Suite: 194 passed, 1 skipped, same 9 pre-existing failures.

# tail_area: no value on staggered arrays (2026-10-20)
- Cause: the local level came from the hole's territory beyond 0.6 pitch; on a staggered array the territory
  reaches only ~0.56 pitch, so it was empty and the hole got no value (2562 of 2805 on examples/data/synthetic_ch).
- Fix: level = median of the farthest tenth (at least 4 px) of the hole's own territory. synthetic_ch: 2805/2805
  measured at levels 0.1-0.5; stress set 96-100 %. Regression test on a staggered array added.

# tail_area_norm (2026-10-20)
- [x] Each pattern divided by its averaged peak (as half_area_min_avg), area where the normalized pattern exceeds
      --tail-level; same wider shared-light window as tail_area; no background step (user's choice: identical to
      tail_area otherwise)
- Stress set at 0.2: error per feature below tail_area in all 5 cases (e.g. mixed 150 nm 8.6 vs 14.5 %), measured
  90-100 % (tail_area 96-100 %). synthetic_ch: 2805/2805, 3 sigma 2.99 % (tail_area 3.21 %).

# edge_area: boundary without the peak (2026-10-20)
- [x] Measured first how much of the noise the peak causes (synthetic_ch, tail_area_norm): with the true,
      noise-free peak the per-feature noise only drops 0.86 -> 0.81 %; the boundary pixels make the rest
- [x] edge_area: threshold = mean brightness weighted by steepness (|grad|, central difference +-1 px) on
      tail_area's window; no peak, no local level; scale- and offset-invariant
- [x] Tried and dropped: squared-steepness weights (noisier, 1.57 %), noise-floor subtraction from the far
      pixels (1.29 % on synthetic_ch), threshold scaled by each hole's total light (low noise but follows size x0.4)
- [x] Report: method, colour, panel shows the wider window and the threshold rule

## Review
- Repeat test (same features, new noise + pulse, 6 shots, 100 nm, lit surroundings): repeat 0.92 % (tail_area
  0.99, tail_area_norm 0.86); pulse leak 0.14 (0.09, 0.18).
- synthetic_ch: 2805/2805, noise per feature 0.95 % (tail_area_norm 0.86 %), follows true size x0.81 (0.84).
- Bright top 30 % brighter on a flat-topped spot: CD moves 1.8 % (tail_area_norm at 0.5: 6.1 %).
- So on synthetic data it is level with the peak-based methods, not better; it helps only if the real bright
  top fluctuates more than simple noise.
- Suite: 198 passed, 1 skipped, same 9 pre-existing failures.

# foot_area / foot_area_ratio: tail boundary without the peak (2026-10-20)
- [x] User correction: edge_area sits at ~0.4 (steepest slope, what projection already reads); a TAIL boundary
      was asked for. edge_area kept as is; two methods added.
- [x] foot_area: threshold = the foot of the edge (below the steepest level, mean brightness weighted by the
      signed Laplacian, within the hole's own territory) -> ~0.09 of the peak; no peak, no local level
- [x] foot_area_ratio: r = image median of foot / steepest level, threshold = r x each hole's steepest level
- Repeat test (6 shots, mixed 100 nm): foot_area_ratio 0.82 % (tail_area_norm 0.86, tail_area 0.99);
  foot_area 2.4 % but pulse leak 0.02 with lit surroundings (ratio 0.21, tail_area_norm 0.18).
- synthetic_ch: both 2805/2805; at the same tail height the noise is set by the height, not by the reference
  (true peak vs estimated peak vs steepest level: 1.04-1.07 %).
- Suite: 201 passed, 1 skipped, same 9 pre-existing failures.

# Brightness between images (2026-10-21)
- [x] image_intensity: simple mean of all pixels per image; between-image 3 s / M x 100 (s with N - 1)
- [x] cell_intensity: the same over the per-image mean cell intensities (was computed but only as a small line in
      the HTML, not in the console); both now in a "Between images: brightness" table, the console and
      between_images.csv; image intensity also per image in summary.csv and the image tab
- Check: 3 copies of synthetic_ch scaled x1.00 / 1.05 / 0.97 -> image intensity 12.2 %, gains alone 12.0 %.

# Batch CD over a whole dataset: hundreds of TIFFs x up to 1500 pages, 4 GPUs (2026-10-23)

Plan: `scripts/cd_batch.py` runs the report's 10 methods on every page of every TIFF of one dataset and writes the
average CD per page and method (no statistics: repeatability is computed later by the user, between 3 datasets).
Pages without holes (other patterns) are skipped by phase cross-correlation against a reference hole image, error as
scikit-image's (`normalization=None`) < 0.3; the user can plug in their own function. Workers: 2 per GPU (spawn,
`CUDA_VISIBLE_DEVICES`), one page at a time each, all holes of a page batched on the GPU; resume from `pages.csv`.
Speed-ups in `_half_area` that leave every result unchanged.

- [x] `subpx/_gpu/regions.py`: region-growing kernel (bit-packed, same steps and limit), top-n selection kernel
- [x] `_half_area`: kernels on CuPy, mask/territory enlargement by index gather; exactness tests
- [x] `measure_cds`: stage times in `meta["seconds"]`; opt-in `outlines=False`
- [x] `scripts/cd_batch.py`: inputs, run.json, frame reader (incl. ImageJ > 4 GB), PCC check, workers, watchdog,
      pages.csv / timings.csv / failed.csv, resume, progress
- [x] `tests/test_cd_batch.py`, `tests/test_half_area_speedups.py`; full suite
- [x] Timing before/after (29k-hole page); 2-worker run; Ctrl+C + resume
- [x] `docs/remote_server.md`: how to run it

## Review

Deviations from the approved plan:
- Added `measure_cds(outlines=False)` (opt-in): skips the half_area family's display outlines, the last 11 % of a
  batch page. The report keeps outlines.
- `--pages` is not part of the resume check (it only selects pages): a trial run and the full run share a folder.
- Found by the Ctrl+C test: on Windows a signal during the wait on the worker pipes raises InterruptedError; handled.

Results (synthetic page 1730 x 1730 px, 28,900 holes, RTX 4060 Ti):
- Per page: 16.2 s (HEAD) -> 8.4 s (report, 128 outline directions) -> 4.7 s (batch: 16 directions, no outlines).
  Every per-hole value (CDs, x/y, centres, statuses, all metrics but the outline perimeter) bit-identical to HEAD.
  Per 1,284-cell chunk: flood fill 24 -> 0.2 ms, averaged peak 42 -> 1.2 ms (cupy.partition sorts whole rows).
- 1 vs 2 workers on this one GPU: same throughput (GPU-bound), identical pages.csv.
- Hole check, scikit-image error with normalization=None: hole pages 0.16-0.26, noise / flat / gradient 0.47-0.50,
  lines / pads 0.58-0.61, other pitch 0.63. The default normalization="phase" gives 1.000 for every pair.
- Found, not changed (it would change the report's numbers; user to decide): the cell stack is the page's largest
  component box. One 48 px diagonal streak grew it 13 -> 19 px: page time +40 %, and holes far from the streak
  changed in the 3 cell-based half_area methods and projection (half_area_min_avg page mean +0.21 nm, single holes up
  to 2.6 nm), because their 4x enlargement grid and averaged-peak area follow the stack size. gaussian and the
  tail-window methods did not change.
- Suite: 212 passed, 1 skipped, same 9 pre-existing failures.

# Batch CD at disk speed: same numbers, at most 1 s per page (2026-10-24)

Plan: the remote's data (pages 2048 x 1378 8-bit, ~400 TIFFs of ~2 GB on one HDD, 4x RTX 6000 Ada) need <= 0.34 s
per page per GPU to be limited by the disk (~7 h per 400 files); the current code takes 5.2 s per page here (15.2 s
with one 42-px blob, which grows every hole's square to 51 x 49). Phase A keeps every number (to 1e-6 px, same
statuses), checked against golden outputs of 6 test pages saved before any change; Phase B fixes the square-size bug
in place (user's decision): the 9 methods' numbers change once.

- [x] Golden outputs of the current code (user-size page, blob page, synthetic_ch, border/merged page, outlines,
      peak seeds); reproducible run to run
- [x] A1: wide-window methods: `_wide_cells` kernel, fused `_half_area` post-processing, one enlargement for the
      4 methods without a local level, arrays kept on the GPU (user page 5.13 -> 2.55 s, all 6 pages same)
- [x] A2: projection + cell-stack half_area family computed in each hole's own box (page-max operators' sub-blocks)
      (blob page: half_area family 4.3 -> 0.19 s, projection 0.94 -> 0.33 s; all 6 pages same)
- [x] A3: deblend: warp-per-cell LM over own boxes, GPU distance map, radii once per iteration, exact tile-median
      kernel (fit status stays on the host). The LM adds its sums in another order: fits differ by <= 1.6e-7
      relative, CDs by <= 3.5e-8 px, no status changes on the 6 pages
- [x] A4: CPU steps: isin lookup + OpenCV dilation (identical mask), GPU seed-revision filters
      (user page 5.13 -> 2.05 s, blob page 14.5 -> 4.1 s)
- [ ] A5: batch: one sequential reader, no page counting, light resume; workers per GPU re-measured; docs
- [ ] Phase A done: <= 1 s per page on this 4060 Ti on both test pages, numbers unchanged; tests
- [ ] B: grid fix in place (fixed 1/4-px grid, zero-extended spline), square-size independence test, before/after
      table, docs
