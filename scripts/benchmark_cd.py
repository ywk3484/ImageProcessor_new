"""Time CD measurement at full-image scale on a synthetic contact-hole lattice.

Defaults match ~2048x5408 px images with a 10 px pitch (~1.1e5 holes). Run it
on the processing machine to see its own numbers:

    python scripts/benchmark_cd.py                                   # cpu and gpu, pitch/FWHM 2
    python scripts/benchmark_cd.py --backends gpu --fwhm 7 --seeds peaks   # dense array
    python scripts/benchmark_cd.py --shape 512 1352 --backends cpu gpu     # quick, 1/16 size

Reports seconds for the measurement and for writing compact outputs, peak
GPU memory, and (with both backends) how far the GPU results are from the CPU's.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import measure_cds, save_cd_results


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
    parser.add_argument("--shape", type=int, nargs=2, default=(2048, 5408), metavar=("H", "W"))
    parser.add_argument("--pitch", type=float, default=10.0)
    parser.add_argument("--fwhm", type=float, default=5.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--methods", nargs="+", default=None, help="Default: all four")
    parser.add_argument("--backends", nargs="+", choices=("cpu", "gpu"), default=("cpu", "gpu"))
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)

    image, centers = lattice(tuple(args.shape), args.pitch, args.fwhm, args.snr)
    print(f"Synthetic {image.shape[0]}x{image.shape[1]}: {len(centers)} holes, pitch {args.pitch:g} px, "
          f"FWHM {args.fwhm:g} px (pitch/FWHM {args.pitch / args.fwhm:.2f}), SNR {args.snr:g}, seeds={args.seeds}", flush=True)
    results = {}
    for backend in args.backends:
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
            measure_cds(image[:200, :400], backend="gpu", seeds=args.seeds, area_max=400, device=args.device,
                        methods=None if args.methods is None else tuple(args.methods))
            hook = PeakHook()
        start = time.perf_counter()
        if hook is not None:
            hook.__enter__()
        try:
            result = measure_cds(image, backend=backend, seeds=args.seeds, area_max=400, device=args.device,
                                 methods=None if args.methods is None else tuple(args.methods))
        finally:
            if hook is not None:
                hook.__exit__(None, None, None)
        measured = time.perf_counter() - start
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            save_cd_results(result, tmp, compact=True)
            written = time.perf_counter() - start
        memory = f", peak GPU memory {hook.peak / 2**30:.2f} GiB" if hook is not None else ""
        print(f"{backend}: measure {measured:.1f} s, write compact {written:.1f} s ({len(result.seeds_xy)} cells, "
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
