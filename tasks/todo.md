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
