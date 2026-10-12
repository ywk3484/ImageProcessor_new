# Deblended background + dense-array CD measurement

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
