# Spectral analysis

## Public functions
- `fft_pitch_error(x, y, ...)`
- `top_periodic_errors(spec, n=5)`

## Purpose
Compute the spatial-frequency spectrum of pitch or residual error measured as `y(x)`.

- `frequency` is returned in inverse position units, e.g. `1/um` if `x` is in `um`
- `period` is returned in the original `x` units, e.g. `um`
- `amplitude` remains in the original `y` units

This matches the notebook use case where you want to detect periodic pitch error while keeping the magnitude interpretable in the original physical units.


## Uniform and resampled FFT helpers

Added `fft_periodicity_uniform(...)` and `fft_periodicity_resample(...)` for notebook workflows that explicitly want outputs in cycles/µm and period in µm.


## Irregular sampling

For uneven `x` spacing, use `periodicity_lombscargle(...)` or `periodicity_irregular(...)`. These return frequency in cycles/um and period in um, but the ordinate is Lomb-Scargle power rather than direct pitch-error amplitude. Use this path when resample-then-FFT would smear narrow peaks or when missing spans are substantial.
