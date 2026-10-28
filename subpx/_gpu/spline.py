"""Full 4x cubic interpolation without dense per-cell matrix products.

For a zero-extended sequence x, cubic B-spline coefficients are the convolution
of x with sqrt(3) * z**abs(i), z = sqrt(3) - 2. Two recurrences evaluate that
convolution without truncating its tails. Four coefficients then evaluate each
sample on the existing quarter-pixel grid. The old operator is made with 32
zeros plus SciPy's 12 prefilter zeros at each end; the boundary reflection is
below FP64 rounding at the retained samples. Arithmetic below remains FP64.
"""


# The four grid phases are t = 5/8, 7/8, 1/8, 3/8. Constants avoid evaluating
# the same cubic weight polynomials again for every sample of every cell.
CUBIC_SRC = r"""
__device__ __constant__ double cubic_weights[16] = {
    0.0087890625, 0.3981119791666667, 0.5524088541666666, 0.040690104166666664,
    0.0003255208333333333, 0.23600260416666666, 0.6520182291666666, 0.11165364583333333,
    0.11165364583333333, 0.6520182291666666, 0.23600260416666666, 0.0003255208333333333,
    0.040690104166666664, 0.5524088541666666, 0.3981119791666667, 0.0087890625
};

__device__ __forceinline__ double cubic_coefficient(
    const double* c, const int n, const int stride, const int i)
{
    const double z = -0.267949192431122706472553658494127633;
    return i < 0 ? c[0] * (i == -1 ? z : z * z) :
           i >= n ? c[(n - 1) * stride] * (i == n ? z : z * z) : c[i * stride];
}

__device__ __forceinline__ double cubic_sample4(
    const double* c, const int n, const int stride, const int i)
{
    // Original coordinate i/4 - 3/8; floor and fractional phase are exact.
    const int j = i / 4 - ((i & 3) < 2);
    const double* weight = cubic_weights + 4 * (i & 3);
    double sum = 0.0;
    sum += cubic_coefficient(c, n, stride, j - 1) * weight[0];
    sum += cubic_coefficient(c, n, stride, j) * weight[1];
    sum += cubic_coefficient(c, n, stride, j + 1) * weight[2];
    sum += cubic_coefficient(c, n, stride, j + 2) * weight[3];
    return sum;
}

// All threads in the block participate. `rows` has (4*h)*w doubles; scratch
// has at least that many and may reuse the output enlargement's allocation.
// The result holds x-axis spline coefficients for all 4*h enlarged rows.
__device__ void prepare_cubic_rows(
    const double* raw, const unsigned char* mask, const double baseline,
    const int h, const int w, double* rows, double* scratch)
{
    const int tid = threadIdx.x, bd = blockDim.x, H = 4 * h;
    const double z = -0.267949192431122706472553658494127633;
    const double gain = 1.732050807568877293527446341505872367;
    for (int x = tid; x < w; x += bd) {
        double left = 0.0, right = 0.0;
        for (int y = 0; y < h; ++y) {
            const int j = y * w + x;
            const double v = mask[j] ? raw[j] - baseline : 0.0;
            left = v + z * left;
            scratch[j] = left;
        }
        for (int y = h - 1; y >= 0; --y) {
            const int j = y * w + x;
            const double v = mask[j] ? raw[j] - baseline : 0.0;
            right = v + z * right;
            scratch[j] = gain * (scratch[j] + right - v);
        }
    }
    __syncthreads();
    for (int i = tid; i < H * w; i += bd)
        rows[i] = cubic_sample4(scratch + i % w, h, w, i / w);
    __syncthreads();
    for (int y = tid; y < H; y += bd) {
        double left = 0.0, right = 0.0;
        for (int x = 0; x < w; ++x) {
            const int j = y * w + x;
            left = rows[j] + z * left;
            scratch[j] = left;
        }
        for (int x = w - 1; x >= 0; --x) {
            const int j = y * w + x;
            const double v = rows[j];
            right = v + z * right;
            rows[j] = gain * (scratch[j] + right - v);
        }
    }
    __syncthreads();
}
"""
