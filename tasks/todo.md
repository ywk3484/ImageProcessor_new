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
