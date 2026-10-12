"""Shared CUDA plumbing for the RawKernel GPU modules of the CD pipeline."""

from __future__ import annotations

try:
    import cupy as cp
except Exception:
    cp = None

_kernel_cache: dict[tuple, object] = {}


def require_cupy():
    if cp is None:
        raise RuntimeError("CuPy is required for the GPU CD backend.")
    return cp


def kernel(source: str, name: str, *, exact: bool = False, defines: tuple[str, ...] = ()):
    """Compile (once per device) and return ``name`` from ``source``.

    ``exact`` disables fused multiply-add contraction, for kernels that must
    reproduce NumPy's separately rounded arithmetic bit for bit.
    """
    options = ("--std=c++14",) + (("--fmad=false",) if exact else ()) + tuple(f"-D{d}" for d in defines)
    key = (cp.cuda.Device().id, name, source, options)
    if key not in _kernel_cache:
        _kernel_cache[key] = cp.RawKernel(source, name, options=options)
    return _kernel_cache[key]


def optin_shared_bytes() -> int:
    """Largest dynamic shared memory one block may request on the current device."""
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    return int(props.get("sharedMemPerBlockOptin", props["sharedMemPerBlock"]))


# Block reductions for one-block-per-cell kernels (blockDim.x a multiple of 32).
REDUCE_SRC = r"""
#define NV 28
__device__ __forceinline__ double warp_sum(double x) {
    for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffff, x, o);
    return x;
}
// Sums v[0..nv) over the block into tot; every thread may read tot afterwards.
__device__ void block_sum(const double* v, int nv, double* tot, double (*partial)[NV]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int i = 0; i < nv; ++i) {
        const double s = warp_sum(v[i]);
        if (lane == 0) partial[warp][i] = s;
    }
    __syncthreads();
    if (threadIdx.x < nv) {
        double s = 0;
        for (int k = 0; k < warps; ++k) s += partial[k][threadIdx.x];
        tot[threadIdx.x] = s;
    }
    __syncthreads();
}
__device__ double block_max(double x, double (*partial)[NV]) {
    for (int o = 16; o > 0; o >>= 1) x = fmax(x, __shfl_down_sync(0xffffffff, x, o));
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5][0] = x;
    __syncthreads();
    double m = partial[0][0];
    for (int k = 1; k < (blockDim.x >> 5); ++k) m = fmax(m, partial[k][0]);
    __syncthreads();
    return m;
}
"""
