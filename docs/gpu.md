# GPU backend notes

## Public contract
The stable public API stays short and task-based:
- `detect_centers(..., backend="gpu")`
- `estimate_shift(..., backend="gpu")`

Public functions return NumPy outputs even when the internal implementation uses CuPy.
That keeps notebook workflows predictable.

## Current GPU design
### Center detection
Current GPU center detection is intentionally **hybrid**:
- thresholding / morphology / connected-components: CPU (OpenCV)
- subpixel refinement for each component ROI: GPU (CuPy)

This matches the historical direction of the project where the heavy per-blob
subpixel math moved to GPU while keeping segmentation stable.

### Registration
GPU phase cross correlation uses:
- CuPy FFT
- cross-power spectrum normalization
- inverse FFT correlation surface
- local 3-point parabolic peak refinement for subpixel estimation

## Why hybrid instead of fully GPU today?
The verbatim historical GPU implementations are not available here as source files.
So the new package keeps the public API stable and rebuilds the GPU internals in a
clean structure. This avoids freezing the package around old chat-specific names.


## CPU-matched logquadratic refinement

The GPU detector path now includes `refine_centers_logquad_gpu_match_cpu(...)`, which mirrors the more complete CPU logic for background selection, optional plane-background removal, fit dilation, and batched quadratic fitting. The stable public entry point remains `detect_centers(..., backend="gpu", refine="logquad")`.
