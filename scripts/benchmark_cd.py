"""Time CD measurement at full-image scale on a synthetic contact-hole lattice.

Defaults use the batch's ten report methods, metrics-only output, and a
1378x2048 px page with a 10 px pitch. Run it on the processing machine:

    python scripts/benchmark_cd.py                                   # cpu and gpu, pitch/FWHM 2
    python scripts/benchmark_cd.py --backends gpu --fwhm 7 --seeds peaks   # dense array
    python scripts/benchmark_cd.py --shape 256 384 --backends cpu gpu     # quick parity check
    python scripts/benchmark_cd.py --backends gpu --full-results         # include display data and export

Reports measurement time, peak live GPU memory, and CPU/GPU agreement.
Writing compact output is timed only with --full-results. This clean Gaussian
lattice can converge much faster than real pages; it is not a dataset-time estimate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import measure_cds, save_cd_results
from scripts.cd_report import ALL_METHODS


@contextmanager
def _batch_gpu(device):
    """Use the requested GPU and the batch worker's reusable memory pool."""
    import cupy as cp
    from subpx._gpu import _cuda

    with cp.cuda.Device(device):
        saved = _cuda.KEEP_MEMORY
        _cuda.KEEP_MEMORY = True
        try:
            yield
        finally:
            _cuda.KEEP_MEMORY = saved
            _cuda.release_memory()


def lattice(shape, pitch: float, fwhm: float, snr: float, *, amplitude=150.0, background=40.0, seed=0):
    """Staggered Gaussian holes rendered locally (fast for 1e5 holes), as uint8 like a PNG."""
    rng = np.random.default_rng(seed)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    xx, yy = np.meshgrid(np.arange(15, shape[1] - 15, pitch), np.arange(15, shape[0] - 15, pitch))
    xx = xx.astype(float)
    xx[1::2] += pitch / 2
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    centers = centers[centers[:, 0] < shape[1] - 15]
    centers = centers + rng.uniform(-0.3, 0.3, centers.shape)
    image = np.full(shape, background, dtype=float).ravel()
    radius = int(np.ceil(5 * sigma))
    ox, oy = (a.ravel() for a in np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)))
    for chunk in np.array_split(np.arange(len(centers)), max(1, len(centers) // 20000)):
        c = centers[chunk]
        ix = np.floor(c[:, 0]).astype(int)[:, None] + ox[None]
        iy = np.floor(c[:, 1]).astype(int)[:, None] + oy[None]
        values = amplitude * np.exp(-0.5 * ((ix - c[:, :1]) ** 2 + (iy - c[:, 1:]) ** 2) / sigma**2)
        ok = (ix >= 0) & (ix < shape[1]) & (iy >= 0) & (iy < shape[0])
        image += np.bincount((iy * shape[1] + ix)[ok], values[ok], minlength=image.size)
    image = image.reshape(shape) + rng.normal(0, amplitude / snr, shape)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), centers


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=int, nargs=2, default=(1378, 2048), metavar=("H", "W"))
    parser.add_argument("--pitch", type=float, default=10.0)
    parser.add_argument("--fwhm", type=float, default=5.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, help="Default: the batch's ten report methods")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=("cpu", "gpu"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--full-results", action="store_true", help="Keep display data and time compact export")
    args = parser.parse_args(argv)

    image, centers = lattice(tuple(args.shape), args.pitch, args.fwhm, args.snr)
    print(f"Synthetic {image.shape[0]}x{image.shape[1]}: {len(centers)} holes, pitch {args.pitch:g} px, "
          f"FWHM {args.fwhm:g} px (pitch/FWHM {args.pitch / args.fwhm:.2f}), SNR {args.snr:g}, seeds={args.seeds}", flush=True)
    options = dict(seeds=args.seeds, area_max=200, device=args.device, methods=tuple(args.methods),
                   n_angles=16, outlines=args.full_results, metrics_only=not args.full_results,
                   summary_only=not args.full_results)
    print(f"Methods: {', '.join(args.methods)}; "
          + ("full results" if args.full_results else "metrics only; compact writing skipped"), flush=True)
    results = {}
    for backend in args.backends:
        with _batch_gpu(args.device) if backend == "gpu" else nullcontext():
            hook = None
            if backend == "gpu":
                import cupy as cp

                class PeakHook(cp.cuda.MemoryHook):
                    """Largest GPU memory held by arrays at once."""
                    name = "peak"

                    def __init__(self):
                        self.pool, self.peak = cp.get_default_memory_pool(), 0

                    def malloc_postprocess(self, **kwargs):
                        self.peak = max(self.peak, self.pool.used_bytes())

                # Compile the kernels outside the timing (they are cached on disk afterwards).
                measure_cds(image[:200, :400], backend="gpu", **options)
                hook = PeakHook()
            start = time.perf_counter()
            if hook is not None:
                hook.__enter__()
            try:
                result = measure_cds(image, backend=backend, **options)
            finally:
                if hook is not None:
                    hook.__exit__(None, None, None)
            measured = time.perf_counter() - start
            writing = "compact writing skipped (metrics only)"
            if args.full_results:
                with tempfile.TemporaryDirectory() as tmp:
                    start = time.perf_counter()
                    save_cd_results(result, tmp, compact=True)
                    writing = f"write compact {time.perf_counter() - start:.1f} s"
            memory = f", peak GPU memory {hook.peak / 2**30:.2f} GiB" if hook is not None else ""
            print(f"{backend}: measure {measured:.1f} s, {writing} ({len(result.seeds_xy)} cells, "
                  f"deblend iterations {result.meta.get('deblend_iterations_used')}){memory}", flush=True)
            for method, m in result.measurements.items():
                values = m.metrics["cd_equivalent_px"][m.valid]
                median = f"{np.median(values):.4f}" if values.size else "n/a"
                print(f"  {method}: {m.valid.sum()}/{len(m.valid)} valid, median CD {median} px", flush=True)
            results[backend] = result
    if len(results) == 2:
        cpu, gpu = results["cpu"], results["gpu"]
        same_cells = len(cpu.seeds_xy) == len(gpu.seeds_xy) and np.array_equal(cpu.seeds_xy, gpu.seeds_xy)
        print(f"GPU vs CPU: identical cells {same_cells}", flush=True)
        if same_cells:
            for method in cpu.measurements:
                a, b = cpu.measurements[method], gpu.measurements[method]
                both = a.valid & b.valid
                delta = np.abs(a.metrics["cd_equivalent_px"][both] - b.metrics["cd_equivalent_px"][both])
                print(f"  {method}: {np.mean(a.status == b.status):.4%} same status, max |dCD| "
                      f"{delta.max() if delta.size else 0:.1e} px", flush=True)


if __name__ == "__main__":
    main()
