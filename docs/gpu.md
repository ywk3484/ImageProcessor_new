# GPU backend notes

## Setup: what to install

`backend="auto"` uses the GPU only when CuPy imports **and** finds a CUDA device;
otherwise it silently runs on the CPU. `scripts/cd_report.py` prints the backend
it will use, and the reason when it is the CPU, as its first line:
`Backend: GPU (NVIDIA ...)` or `Backend: CPU, because ...`.

1. **NVIDIA driver.** `nvidia-smi` must work and show `CUDA Version: 12.x` or
   newer (driver >= 525.60.13 on Linux, >= 527.41 on Windows). A newer driver
   (e.g. CUDA 13) also runs CUDA 12 programs. No system CUDA Toolkit is needed.
2. **CuPy with its CUDA libraries, in the same Python environment that runs the
   scripts** (check with `python -c "import sys; print(sys.executable)"`):

   ```bash
   python -m pip install "cupy-cuda12x[ctk]"
   ```

   `cupy-cuda12x` alone is not enough: it does not bundle the CUDA runtime,
   NVRTC (the kernel compiler), cuBLAS, cuFFT, cuRAND, cuSOLVER and cuSPARSE.
   The `[ctk]` extra installs them as pip packages (`cuda-toolkit[...]==12.*`).
   Tested here: CuPy 14.2.0 on an RTX 4060 Ti. Do not install two CuPy
   packages side by side (`pip list | grep -i cupy` must show exactly one).
3. **Check:**

   ```bash
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), cupy.arange(4).sum())"
   ```

   It prints `1 6` (or more devices). The first call compiles kernels and takes
   a few seconds; compiled kernels are cached in `~/.cupy/kernel_cache`.
4. **Docker only:** the host needs the NVIDIA Container Toolkit, and the
   container must be started with `--gpus all` (compose: a `deploy.resources`
   GPU reservation).

Typical failures:

| Message | Cause | Fix |
|---|---|---|
| `CuPy cannot be imported (ModuleNotFoundError ...)` | CuPy not installed in this environment | step 2 |
| `... libnvrtc.so ... / nvrtc64_*.dll ... not found` | CuPy without its CUDA libraries | step 2 with `[ctk]` |
| `CuPy finds no CUDA device` / `cudaErrorNoDevice` / `cudaErrorInsufficientDriver` | no driver, too old a driver, or a container without `--gpus all` | steps 1 and 4 |
| `CUDA path could not be detected` warning (Windows) | harmless with `[ctk]` | none |

Run with `--backend gpu` instead of `auto` to get an error instead of a silent
CPU fallback.

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
