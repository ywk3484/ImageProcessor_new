"""One-pass outline fit on the GPU (CuPy): the page's outline CD, x / y widths and blur in one pass over the image.

Library copy of the prototype (artifacts/outline_fit/fast/gpufit.py and pipeline.py, 2026-11-12), numerics unchanged;
its step D (a size per hole) is not here. Method, settings and measurements: docs/outline_fit_one_pass.md.

Per image, with the model, parameter vector, bounds and CD definition of ``subpx._outline``:

    A  seeds -> plain stack -> pair maps -> first page fit (grid of starts, round + superellipse)      theta_0
    B  hole pass with the size held (centre, brightness, level of every hole), two sweeps, template theta_0
    C  plain stack again with B's centres -> pair maps -> final page fit from theta_0, all free          theta_1
       gauss2 only: a fit that ends with its two Gaussians exchanged is run once more from the exchanged labels

- Plain stack: every stacked hole's window binned at its sub-pixel offset on a quarter-pixel grid; the plain mean
  (no hole is centred against the stack or divided by its fitted brightness: both bias the page's size).
- Pair maps (Gamma_q): for each of the 16 sub-pixel classes, where the neighbours and the hole itself stand (cell
  list + B-spline splat + FFT); this puts the neighbours' light into the stack's model.
- Page fit: ``subpx._outline.onepass.ShapeFitter``; the page fits of a batch of images run in lockstep.
- Hole pass: the library's hole stage (Gauss-Seidel over colours, complete page model with the far light, robust
  weights) for a fixed number of sweeps.

Lengths in pixels. Everything here needs CuPy; ``subpx.fit_outline_fast`` is the public entry.
"""
from __future__ import annotations

import math
import time

import cupy as cp
import numpy as np

from .._outline.model import Template
from .._outline.onepass import Band, ShapeFitter, Spec, exchanged, second_is_narrower

PAIR_REACH_MARGIN_PX = 13.0  # neighbours within the window's half width + this are in the stack's model
STACK_BINS = 4  # stack bins per pixel
PAIR_BINS = 8  # pair-map grid points per pixel
F_MAX = 0.6  # band of the page fit, cycles per pixel
N_POLY = 256  # outline points of the page fit
FIRST_STAGES, FIRST_ITERATIONS, FIRST_FTOL = ("round", "super"), (10, 12), 1e-6
FINAL_ITERATIONS = 30
CENTRING_SWEEPS = 2

SRC = r'''
typedef TTYPE TT;   // template grids (-DTTYPE)
typedef MTYPE MT;   // the hole pass's matrix sums of one window row

__device__ inline void keys(TT t, TT* w) {
    TT x0 = 1 + t, x1 = t, x2 = 1 - t, x3 = 2 - t;
    w[0] = (((TT)(-0.5) * x0 + (TT)2.5) * x0 - 4) * x0 + 2;
    w[1] = ((TT)1.5 * x1 - (TT)2.5) * x1 * x1 + 1;
    w[2] = ((TT)1.5 * x2 - (TT)2.5) * x2 * x2 + 1;
    w[3] = (((TT)(-0.5) * x3 + (TT)2.5) * x3 - 4) * x3 + 2;
}

// values of ng interleaved fine grids (N, N, ng) at fine coordinates (sx, sy); outside the grid reads 0
__device__ inline void sample(const TT* tmpl, int N, int ng, double sx, double sy, TT* v) {
    double fx = floor(sx), fy = floor(sy);
    int ix = (int) fx, iy = (int) fy;
    TT wx[4], wy[4];
    keys((TT)(sx - fx), wx);
    keys((TT)(sy - fy), wy);
    for (int g = 0; g < ng; ++g) v[g] = 0;
    for (int k = 0; k < 4; ++k) {
        int yy = iy + k - 1;
        if (yy < 0 || yy >= N) continue;
        TT row[5];
        for (int g = 0; g < ng; ++g) row[g] = 0;
        for (int l = 0; l < 4; ++l) {
            int xx = ix + l - 1;
            if (xx < 0 || xx >= N) continue;
            const TT* p = tmpl + ((long long) yy * N + xx) * ng;
            for (int g = 0; g < ng; ++g) row[g] += wx[l] * p[g];
        }
        for (int g = 0; g < ng; ++g) v[g] += wy[k] * row[g];
    }
}

// ---------------------------------------------------------------------------------------------- stack
// one thread per (hole, window row): the row's pixels into the stack's bins. sw = the hole's weight (0: not
// stacked), nrm = what its pixels are multiplied by (1 / brightness)
extern "C" __global__
void stack_acc(const float* image, const double* c, const double* sw, const double* nrm,
               int n, int R, int H, int W, int U, int Ms, double* sums, double* wsum)
{
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    int side = 2 * R + 1;
    if (t >= (long long) n * side) return;
    int i = (int)(t / side);
    int oy = (int)(t % side) - R;
    double w = sw[i];
    if (w == 0) return;
    double x = c[2 * i], y = c[2 * i + 1];
    long long rx = (long long) rint(x), ry = (long long) rint(y);
    long long py = ry + oy;
    if (py < 0 || py >= H) return;
    long long by = (long long) rint(((double) py - y) * U) + Ms / 2;
    if (by < 0 || by >= Ms) return;
    double m = w * nrm[i];
    for (int ox = -R; ox <= R; ++ox) {
        long long px = rx + ox;
        if (px < 0 || px >= W) continue;
        long long bx = (long long) rint(((double) px - x) * U) + Ms / 2;
        if (bx < 0 || bx >= Ms) continue;
        atomicAdd(sums + by * Ms + bx, m * (double) image[py * W + px]);
        atomicAdd(wsum + by * Ms + bx, w);
    }
}

// one thread per hole: brightness and level by least squares of its window against the stack (pixels within
// core2 of the hole), leftover rms over the whole window, one Gauss-Newton step of the centre (core pixels).
// stack bins without data hold NaN. out: n x 6 = a, level, rms, dx, dy, pixels used
extern "C" __global__
void hole_vs_stack(const float* image, const double* c, const double* S, const double* Sx, const double* Sy,
                   int n, int R, int H, int W, int U, int Ms, double core2, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = c[2 * i], y = c[2 * i + 1];
    long long rx = (long long) rint(x), ry = (long long) rint(y);
    double cn = 0, s1 = 0, s2 = 0, sy = 0, y1 = 0;
    for (int oy = -R; oy <= R; ++oy) {
        long long py = ry + oy;
        if (py < 0 || py >= H) continue;
        double uy = (double) py - y;
        long long by = (long long) rint(uy * U) + Ms / 2;
        if (by < 0 || by >= Ms) continue;
        for (int ox = -R; ox <= R; ++ox) {
            long long px = rx + ox;
            if (px < 0 || px >= W) continue;
            double ux = (double) px - x;
            if (ux * ux + uy * uy > core2) continue;
            long long bx = (long long) rint(ux * U) + Ms / 2;
            if (bx < 0 || bx >= Ms) continue;
            double s = S[by * Ms + bx];
            if (s != s) continue;
            double v = (double) image[py * W + px];
            cn += 1; s1 += s; s2 += s * s; sy += s * v; y1 += v;
        }
    }
    double det = cn * s2 - s1 * s1;
    double a = 1, lev = 0;
    if (det > 0) { a = (cn * sy - s1 * y1) / det; lev = (s2 * y1 - s1 * sy) / det; }
    double rss = 0, nf = 0, sxx = 0, sxy = 0, syy = 0, gx = 0, gy = 0;
    for (int oy = -R; oy <= R; ++oy) {
        long long py = ry + oy;
        if (py < 0 || py >= H) continue;
        double uy = (double) py - y;
        long long by = (long long) rint(uy * U) + Ms / 2;
        if (by < 0 || by >= Ms) continue;
        for (int ox = -R; ox <= R; ++ox) {
            long long px = rx + ox;
            if (px < 0 || px >= W) continue;
            double ux = (double) px - x;
            long long bx = (long long) rint(ux * U) + Ms / 2;
            if (bx < 0 || bx >= Ms) continue;
            long long b = by * Ms + bx;
            double s = S[b];
            if (s != s) continue;
            double res = (double) image[py * W + px] - a * s - lev;
            rss += res * res; nf += 1;
            if (ux * ux + uy * uy > core2) continue;
            double jx = -a * Sx[b], jy = -a * Sy[b];
            sxx += jx * jx; sxy += jx * jy; syy += jy * jy; gx += jx * res; gy += jy * res;
        }
    }
    double d2 = sxx * syy - sxy * sxy;
    double dx = 0, dy = 0;
    if (d2 > 0) { dx = (syy * gx - sxy * gy) / d2; dy = (sxx * gy - sxy * gx) / d2; }
    double* o = out + (long long) i * 6;
    o[0] = a; o[1] = lev; o[2] = sqrt(rss / (nf > 0 ? nf : 1)); o[3] = dx; o[4] = dy; o[5] = nf;
}

// ------------------------------------------------------------------------------------------- pair map
__device__ inline void bspline(double t, double* w) {
    double u = 1 - t;
    w[0] = u * u * u / 6;
    w[1] = (3 * t * t * t - 6 * t * t + 4) / 6;
    w[2] = (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6;
    w[3] = t * t * t / 6;
}

// one thread per stacked hole i: every hole j within the square of half side `reach` (j = i included) adds
// g_i a_j at c_j - c_i - s_i to the map of i's sub-pixel class, as a cubic B-spline on a grid of Ug per pixel
extern "C" __global__
void pair_splat(const double* c, const double* a, const double* g, const int* cls, const double* s,
                const int* order, const int* cell_start, int n, int ncx, int ncy, double cs, double reach,
                int Ug, int Ng, float* hist)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double gi = g[i];
    if (gi == 0) return;
    double x = c[2 * i], y = c[2 * i + 1];
    double sx = s[2 * i], sy = s[2 * i + 1];
    int cx = (int) floor(x / cs), cy = (int) floor(y / cs);
    int r = (int) ceil(reach / cs);
    float* h = hist + (long long) cls[i] * Ng * Ng;
    for (int yy = cy - r; yy <= cy + r; ++yy) {
        if (yy < 0 || yy >= ncy) continue;
        for (int xx = cx - r; xx <= cx + r; ++xx) {
            if (xx < 0 || xx >= ncx) continue;
            int cell = yy * ncx + xx;
            for (int k = cell_start[cell]; k < cell_start[cell + 1]; ++k) {
                int j = order[k];
                double dx = c[2 * j] - x, dy = c[2 * j + 1] - y;
                if (fabs(dx) > reach || fabs(dy) > reach) continue;
                double w = gi * a[j];
                double px = (dx - sx) * Ug + Ng / 2, py = (dy - sy) * Ug + Ng / 2;
                double fx = floor(px), fy = floor(py);
                double wx[4], wy[4];
                bspline(px - fx, wx);
                bspline(py - fy, wy);
                int x0 = (int) fx - 1, y0 = (int) fy - 1;
                for (int q = 0; q < 4; ++q) {
                    int ry = ((y0 + q) % Ng + Ng) % Ng;
                    for (int p = 0; p < 4; ++p) {
                        int rx = ((x0 + p) % Ng + Ng) % Ng;
                        atomicAdd(h + (long long) ry * Ng + rx, (float)(w * wy[q] * wx[p]));
                    }
                }
            }
        }
    }
}

// nearest other hole of every hole (cell list): distance
extern "C" __global__
void nearest(const double* c, const int* order, const int* cell_start, int n, int ncx, int ncy, double cs, double* out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = c[2 * i], y = c[2 * i + 1];
    int cx = (int) floor(x / cs), cy = (int) floor(y / cs);
    double best = 1e30;
    for (int yy = cy - 1; yy <= cy + 1; ++yy) {
        if (yy < 0 || yy >= ncy) continue;
        for (int xx = cx - 1; xx <= cx + 1; ++xx) {
            if (xx < 0 || xx >= ncx) continue;
            int cell = yy * ncx + xx;
            for (int k = cell_start[cell]; k < cell_start[cell + 1]; ++k) {
                int j = order[k];
                if (j == i) continue;
                double dx = c[2 * j] - x, dy = c[2 * j + 1] - y;
                double d = dx * dx + dy * dy;
                if (d < best) best = d;
            }
        }
    }
    out[i] = sqrt(best);
}

// ------------------------------------------------------------------------------------------ hole stage
// one thread per (hole of the colour, window row): 5 x 5 normal matrix (upper triangle, 15), right-hand side (5),
// leftover sum and pixel count of the hole's window against the complete page model. Library's fit_holes.
extern "C" __global__
void hole_normal(const float* image, const double* page, const TT* tmpl,
                 const int* idx, const double* c, const double* a, const double* level, const double* delta,
                 int n, int N, int U, int R, int H, int W, double nv0, double nv1, double rscale, int size,
                 double* out)
{
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    int side = 2 * R + 1;
    if (t >= (long long) n * side) return;
    int i = (int)(t / side);
    int oy = (int)(t % side) - R;
    int h = idx[i];
    double x = c[2 * h], y = c[2 * h + 1];
    double aa = a[h], lev = level[h], de = delta[h];
    long long rx = (long long) rint(x), ry = (long long) rint(y);
    long long py = ry + oy;
    if (py < 0 || py >= H) return;
    double sy = ((double) py - y) * U + N / 2;
    MT mat[15];
    double acc[7];
    for (int q = 0; q < 15; ++q) mat[q] = 0;
    for (int q = 0; q < 7; ++q) acc[q] = 0;
    MT faa = (MT) aa, fde = (MT) de;
    for (int ox = -R; ox <= R; ++ox) {
        long long px = rx + ox;
        if (px < 0 || px >= W) continue;
        double sx = ((double) px - x) * U + N / 2;
        TT v[5];
        sample(tmpl, N, 5, sx, sy, v);
        long long f = py * W + px;
        double model = page[f] + lev;
        double resid = (double) image[f] - model;
        double wgt = 1.0 / (nv0 + nv1 * fmax(model, 0.0));
        acc[5] += resid * resid;
        acc[6] += 1.0;
        if (rscale > 0) { double q = resid / rscale; wgt = wgt / (1.0 + q * q); }
        MT J[5];
        J[0] = -faa * (MT) v[2]; J[1] = -faa * (MT) v[3]; J[2] = (MT) v[0] + fde * ((MT) v[1] + fde * (MT) v[4]);
        J[3] = 1;
        J[4] = size ? faa * ((MT) v[1] + 2 * fde * (MT) v[4]) : 0;
        MT fw = (MT) wgt;
        double wr = wgt * resid;
        int q = 0;
        for (int r = 0; r < 5; ++r) {
            MT jw = J[r] * fw;
            for (int s = r; s < 5; ++s) mat[q++] += jw * J[s];
            acc[r] += (double) J[r] * wr;
        }
    }
    double* o = out + (long long) i * 22;
    for (int q = 0; q < 15; ++q) atomicAdd(o + q, (double) mat[q]);
    for (int q = 0; q < 7; ++q) atomicAdd(o + 15 + q, acc[q]);
}

// one thread per hole of the colour: solve its 5 x 5 system, bound and apply the step (library's rules), keep the
// centre it had (for taking its old light out). stat: n x 4 = largest centre move, edge-offset move, leftover rms,
// standard error of the edge offset
extern "C" __global__
void hole_update(const double* acc, const int* idx, double* c, double* a, double* level, double* delta,
                 double* c_old, int n, int size, double max_move, double max_delta, double a_med, double relax,
                 double* stat)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int h = idx[i];
    const double* o = acc + (long long) i * 22;
    int k = size ? 5 : 4;
    double A[5][6];
    int q = 0;
    for (int r = 0; r < 5; ++r) for (int s = r; s < 5; ++s) { A[r][s] = o[q]; A[s][r] = o[q]; ++q; }
    double tr = 0;
    for (int r = 0; r < k; ++r) tr += A[r][r];
    for (int r = 0; r < k; ++r) A[r][r] += 1e-9 * tr / k;
    double Ainv[5][5];
    double M[5][10];
    for (int r = 0; r < k; ++r) { for (int s = 0; s < k; ++s) { M[r][s] = A[r][s]; M[r][k + s] = (r == s) ? 1.0 : 0.0; } }
    bool ok = true;
    for (int col = 0; col < k; ++col) {   // Gauss-Jordan with partial pivoting
        int piv = col; double best = fabs(M[col][col]);
        for (int r = col + 1; r < k; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (best < 1e-300) { ok = false; break; }
        if (piv != col) for (int s = 0; s < 2 * k; ++s) { double tmp = M[col][s]; M[col][s] = M[piv][s]; M[piv][s] = tmp; }
        double d = 1.0 / M[col][col];
        for (int s = 0; s < 2 * k; ++s) M[col][s] *= d;
        for (int r = 0; r < k; ++r) if (r != col) { double f = M[r][col]; if (f != 0) for (int s = 0; s < 2 * k; ++s) M[r][s] -= f * M[col][s]; }
    }
    double step[5] = {0, 0, 0, 0, 0};
    if (ok) {
        for (int r = 0; r < k; ++r) { double v = 0; for (int s = 0; s < k; ++s) { Ainv[r][s] = M[r][k + s]; v += M[r][k + s] * o[15 + s]; } step[r] = v; }
        for (int r = 0; r < k; ++r) if (!(step[r] == step[r]) || fabs(step[r]) > 1e30) step[r] = 0;
    }
    step[2] *= relax; step[3] *= relax; step[4] *= relax;
    double* st = stat + (long long) i * 4;
    st[2] = sqrt(o[20] / (o[21] > 0 ? o[21] : 1));
    st[3] = (ok && size) ? sqrt(fmax(Ainv[4][4], 0.0)) : nan("");
    c_old[2 * i] = c[2 * h]; c_old[2 * i + 1] = c[2 * h + 1];
    double mx = fmin(fmax(step[0], -max_move), max_move), my = fmin(fmax(step[1], -max_move), max_move);
    c[2 * h] += mx; c[2 * h + 1] += my;
    st[0] = fmax(fabs(mx), fabs(my));
    double a0 = a[h];
    double an = fmin(fmax(a0 + step[2], 0.5 * a0), 2.0 * a0);
    a[h] = fmin(fmax(an, 1e-3), 5.0 * a_med);
    level[h] += step[3];
    st[1] = 0;
    if (size) {
        double dn = fmin(fmax(delta[h] + step[4], -max_delta), max_delta);
        st[1] = fabs(dn - delta[h]);
        delta[h] = dn;
    }
}

// one thread per (hole of the colour, render-window pixel): the hole's old light out of the page model, its new in
extern "C" __global__
void render_scatter(double* page, float* cache, const TT* tmpl3,
                    const int* idx, const double* c_old, const double* c, const double* a, const double* delta,
                    int n, int N, int U, int Rr, int H, int W, double rl2)
{
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    int side = 2 * Rr + 1;
    long long P = (long long) side * side;
    if (t >= n * P) return;
    int i = (int)(t / P);
    int p = (int)(t % P);
    int oy = p / side - Rr, ox = p % side - Rr;
    if ((double)(ox * ox + oy * oy) > rl2) return;   // the near template is zero there (and was when cached)
    int h = idx[i];
    float old = cache[t];
    if (old != 0.0f) {
        long long pxo = (long long) rint(c_old[2 * i]) + ox, pyo = (long long) rint(c_old[2 * i + 1]) + oy;
        atomicAdd(page + pyo * W + pxo, -(double) old);
    }
    double x = c[2 * h], y = c[2 * h + 1];
    long long px = (long long) rint(x) + ox, py = (long long) rint(y) + oy;
    float light = 0.0f;
    if (px >= 0 && px < W && py >= 0 && py < H) {
        TT v[3];
        sample(tmpl3, N, 3, ((double) px - x) * U + N / 2, ((double) py - y) * U + N / 2, v);
        double de = delta[h];
        light = (float)(a[h] * ((double) v[0] + de * ((double) v[1] + de * (double) v[2])));
        atomicAdd(page + py * W + px, (double) light);
    }
    cache[t] = light;
}

// cubic (Keys) splat of every hole's far-light amplitude on the padded pixel grid
extern "C" __global__
void far_splat(const double* c, const double* amp, int n, int mx, int my, int hp, int wp, float* grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double x = c[2 * i] + mx, y = c[2 * i + 1] + my;
    double fx = floor(x), fy = floor(y);
    int ix = (int) fx, iy = (int) fy;
    if (ix < 1 || ix + 2 >= wp || iy < 1 || iy + 2 >= hp) return;
    double wx[4], wy[4];
    {
        double t = x - fx; double x0 = 1 + t, x1 = t, x2 = 1 - t, x3 = 2 - t;
        wx[0] = ((-0.5 * x0 + 2.5) * x0 - 4) * x0 + 2; wx[1] = (1.5 * x1 - 2.5) * x1 * x1 + 1;
        wx[2] = (1.5 * x2 - 2.5) * x2 * x2 + 1; wx[3] = ((-0.5 * x3 + 2.5) * x3 - 4) * x3 + 2;
        t = y - fy; x0 = 1 + t; x1 = t; x2 = 1 - t; x3 = 2 - t;
        wy[0] = ((-0.5 * x0 + 2.5) * x0 - 4) * x0 + 2; wy[1] = (1.5 * x1 - 2.5) * x1 * x1 + 1;
        wy[2] = (1.5 * x2 - 2.5) * x2 * x2 + 1; wy[3] = ((-0.5 * x3 + 2.5) * x3 - 4) * x3 + 2;
    }
    double am = amp[i];
    for (int q = 0; q < 4; ++q) for (int p = 0; p < 4; ++p)
        atomicAdd(grid + (long long)(iy + q - 1) * wp + (ix + p - 1), (float)(am * wy[q] * wx[p]));
}
'''

_modules: dict[tuple, object] = {}
_bands: dict[tuple, tuple] = {}
_fitters: dict[tuple, ShapeFitter] = {}
_templates: dict[str, Template] = {}


def kernels(tt="float"):
    """The module of the hole kernels, compiled once per device and template precision."""
    key = (cp.cuda.Device().id, tt)
    if key not in _modules:
        _modules[key] = cp.RawModule(code=SRC, options=("--std=c++14", f"-DTTYPE={tt}", f"-DMTYPE={tt}"))
    return _modules[key]


def lib_template(blur):
    """The library ``Template`` used for the parameter bookkeeping (unpack, bounds, radius, CD), one per blur form."""
    if blur not in _templates:
        _templates[blur] = Template(upsample=8, size_px=24, blur=blur)
    return _templates[blur]


def fitter_for(page, blur):
    """The page fitter of a window size (its device tables are built once per device)."""
    key = (cp.cuda.Device().id, blur, page.R, page.U, page.f_max, page.L)
    if key not in _fitters:
        _fitters[key] = ShapeFitter(page.t, box_px=page.L, f_max=page.f_max, U=page.U, R=page.R, n_poly=N_POLY, xp=cp)
    return _fitters[key]


def _even_fast(n):
    from scipy.fft import next_fast_len

    m = next_fast_len(int(n), real=True)
    while m % 2:
        m = next_fast_len(m + 1, real=True)
    return m


# ---------------------------------------------------------------------------------------------- seeds
def find_seeds(img, smooth_sigma=1.0, nms=2, rel_threshold=0.3):
    """Bright blobs' centres (N, 2) [x, y] on the device: local maxima (within +-nms px) of the Gaussian-smoothed
    page above a level between the page's dark and bright ends, refined by a parabola through the log of the three
    samples round the maximum along x and along y. A blob wider than the search window has a flat top with several
    maxima: the blob's size is read from the area above the level per maximum, and the search is repeated with a
    window (and a smoothing) that covers the blob."""
    import cupyx.scipy.ndimage as ndi

    sub = img[::4, ::4].ravel()
    lo, hi = (float(v) for v in cp.percentile(sub, cp.asarray([5.0, 99.8])))
    thr = lo + rel_threshold * (hi - lo)
    above = float((img[::2, ::2] > thr).mean()) * img.size
    for _ in range(3):
        sm = ndi.gaussian_filter(img, smooth_sigma, mode="nearest", truncate=2.5)
        mx = ndi.maximum_filter(sm, size=2 * nms + 1, mode="nearest")
        peak = (sm == mx) & (sm > thr)
        peak[0, :] = peak[-1, :] = False
        peak[:, 0] = peak[:, -1] = False
        ys, xs = cp.nonzero(peak)
        n = int(xs.shape[0])
        if n == 0:
            break
        width = math.sqrt(4.0 * above / (math.pi * n))  # diameter of the mean blob above the level
        if width <= 2 * nms + 1:
            break
        nms = int(math.ceil(0.6 * width))
        smooth_sigma = max(smooth_sigma, 0.2 * width)
    if int(xs.shape[0]) == 0:
        return cp.zeros((0, 2))
    base = lo - 0.02 * (hi - lo)  # log of the height above the dark level: a Gaussian-like top becomes a parabola
    c0 = cp.log(cp.maximum(sm[ys, xs] - base, 1e-6))

    def offset(lft, rgt):
        den = lft - 2 * c0 + rgt
        return cp.clip(cp.where(den < 0, 0.5 * (lft - rgt) / den, 0.0), -0.5, 0.5)

    dx = offset(cp.log(cp.maximum(sm[ys, xs - 1] - base, 1e-6)), cp.log(cp.maximum(sm[ys, xs + 1] - base, 1e-6)))
    dy = offset(cp.log(cp.maximum(sm[ys - 1, xs] - base, 1e-6)), cp.log(cp.maximum(sm[ys + 1, xs] - base, 1e-6)))
    return cp.stack([xs + dx, ys + dy], axis=1).astype(cp.float64)


class CellList:
    def __init__(self, c, h, w, cs):
        self.cs = float(cs)
        self.ncx = int(math.ceil(w / cs)) + 1
        self.ncy = int(math.ceil(h / cs)) + 1
        cx = cp.clip(cp.floor(c[:, 0] / cs), 0, self.ncx - 1).astype(cp.int32)
        cy = cp.clip(cp.floor(c[:, 1] / cs), 0, self.ncy - 1).astype(cp.int32)
        self.cell = cy * self.ncx + cx
        self.order = cp.argsort(self.cell).astype(cp.int32)
        sorted_cell = self.cell[self.order]
        self.start = cp.searchsorted(sorted_cell, cp.arange(self.ncx * self.ncy + 1, dtype=cp.int32)).astype(cp.int32)
        self.rank = cp.empty(len(c), dtype=cp.int32)  # position of the hole inside its cell
        self.rank[self.order] = cp.arange(len(c), dtype=cp.int32) - self.start[sorted_cell]


# ------------------------------------------------------------------------------------------------ page
class GpuPage:
    """One image through the one-pass fit: seeds, plain stack, pair maps, templates, hole pass.

    ``trace``: a dict that receives what every step produced, as host copies (``fit_images`` describes it). The
    steps themselves do not depend on it: with ``trace=None`` nothing is recorded and nothing else differs."""

    def __init__(self, image, *, blur="gauss2", seeds=None, pair_reach_px=None, tt="float", trace=None):
        self.trace = trace
        self.img = cp.asarray(image).astype(cp.float32)
        self.h, self.w = self.img.shape
        self.U, self.pair_U, self.f_max = STACK_BINS, PAIR_BINS, F_MAX
        self.blur = blur
        self.tt = tt
        self.mod = kernels(tt)
        self.t = lib_template(blur)
        self.c = find_seeds(self.img) if seeds is None else cp.asarray(np.asarray(seeds, dtype=np.float64))
        self.n = int(self.c.shape[0])
        if self.n < 2:
            raise ValueError(f"{self.n} holes found: no page fit")
        self.cells = CellList(self.c, self.h, self.w, 16.0)
        nn = cp.empty(self.n)
        self.mod.get_function("nearest")(((self.n + 127) // 128,), (128,),
                                          (self.c, self.cells.order, self.cells.start, np.int32(self.n),
                                           np.int32(self.cells.ncx), np.int32(self.cells.ncy), np.float64(16.0), nn))
        self.spacing = float(cp.median(nn[nn < 1e20])) if bool((nn < 1e20).any()) else 16.0
        if self.spacing > 15.0:  # sparse holes: the 3 x 3 cells searched may miss the nearest hole; search wider cells
            self.cells = CellList(self.c, self.h, self.w, 2.2 * self.spacing)
            self.mod.get_function("nearest")(((self.n + 127) // 128,), (128,),
                                              (self.c, self.cells.order, self.cells.start, np.int32(self.n),
                                               np.int32(self.cells.ncx), np.int32(self.cells.ncy),
                                               np.float64(self.cells.cs), nn))
            self.spacing = float(cp.median(nn[nn < 1e20]))
        self.R = max(3, int(math.ceil(0.75 * self.spacing)))
        self.Ms = 2 * int(math.ceil((self.R + 0.5) * self.U)) + 1
        # neighbours whose light can reach the window: within the window's half width + 13 px (a hole's light is
        # below 1e-3 of its peak well inside that); the fit's box holds window + neighbours + their light without
        # folding back. Reach, box and template box follow the window: with fixed values a wide pattern lost its
        # neighbours from the model.
        self.reach = float(pair_reach_px) if pair_reach_px else float(self.R) + PAIR_REACH_MARGIN_PX
        self.L = int(8 * math.ceil((2 * self.reach + 8) / 8))
        self.a = cp.ones(self.n)
        self.level = cp.zeros(self.n)
        self.far = None
        self._chunks = None
        self._keep("window", seeds=self.c, nearest_px=nn, spacing_px=self.spacing, window_px=self.R,
                   bins_per_px=self.U, stack_bins=self.Ms, reach_px=self.reach, box_px=self.L)

    def _keep(self, name, **values):
        """Record what a step produced under ``name`` (device arrays as host copies) when the page is traced."""
        if self.trace is not None:
            self.trace[name] = {k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v for k, v in values.items()}

    # -------------------------------------------------------------------------------------------- stack
    def interior(self):
        m = self.R + 1
        c = self.c
        return (c[:, 0] >= m) & (c[:, 0] <= self.w - 1 - m) & (c[:, 1] >= m) & (c[:, 1] <= self.h - 1 - m)

    def _stack(self, use):
        """The plain mean of the windows of the holes ``use``: no estimated quantity touches a hole's own light."""
        ms = self.Ms
        sums, wsum = cp.zeros(ms * ms), cp.zeros(ms * ms)
        sw = use.astype(cp.float64)
        nrm = cp.ones(self.n)
        side = 2 * self.R + 1
        self.mod.get_function("stack_acc")(((self.n * side + 127) // 128,), (128,),
                                            (self.img, self.c, sw, nrm, np.int32(self.n), np.int32(self.R),
                                             np.int32(self.h), np.int32(self.w), np.int32(self.U), np.int32(ms), sums, wsum))
        stack = cp.where(wsum > 0, sums / cp.maximum(wsum, 1e-300), cp.nan).reshape(ms, ms)
        return stack, wsum.reshape(ms, ms), sw, nrm

    def _versus(self, stack, core_px):
        filled = cp.where(cp.isfinite(stack), stack, 0.0)
        gy, gx = cp.gradient(filled, 1.0 / self.U)
        out = cp.empty((self.n, 6))
        self.mod.get_function("hole_vs_stack")(((self.n + 63) // 64,), (64,),
                                                (self.img, self.c, stack, gx, gy, np.int32(self.n), np.int32(self.R),
                                                 np.int32(self.h), np.int32(self.w), np.int32(self.U), np.int32(self.Ms),
                                                 np.float64(core_px * core_px), out))
        return out

    def build_stack(self, core_frac=0.5):
        """Stack, then every hole's brightness / level / leftover against it, then the stack again without the
        unusual holes. core_frac x spacing = radius of the pixels that set the brightness."""
        inside = self.interior()
        if not bool(inside.any()):
            raise ValueError(f"no hole lies {self.R + 1} px or more inside the image (its window): nothing to stack")
        core = max(2.0, core_frac * self.spacing)
        stack, wsum, _, _ = self._stack(inside)
        o = self._versus(stack, core)
        a = o[:, 0]
        med = float(cp.median(a[inside]))
        a = cp.maximum(a / med, 1e-3)
        typical = float(cp.median(o[inside, 2]))
        use = inside & (a > 0.4) & (a < 2.5) & (o[:, 2] < 2.0 * typical) & (o[:, 5] > 0)
        if self.trace is not None:  # the average of every interior hole, each hole against it, and who is dropped
            self._keep("average_all", stack=stack, pixels=wsum, inside=inside, brightness=a, level=o[:, 1],
                       leftover=o[:, 2], compared=o[:, 5] > 0, kept=use, median_brightness=med,
                       typical_leftover=typical, core_px=core, brightness_limits=(0.4, 2.5), leftover_limit=2.0)
        if not bool(use.any()):
            raise ValueError("no hole of usual brightness and leftover to stack")
        self.a = a
        stack, wsum, sw, nrm = self._stack(use)
        o = self._versus(stack, core)  # brightness and level against the final stack
        self.a_rel = cp.maximum(o[:, 0], 1e-3)  # brightness in the final stack's unit (window = a_rel x stack + level)
        self.lev_rel = o[:, 1]
        self.rms = o[:, 2]
        self.noise = float(cp.median(self.rms[use]))
        self.stack, self.wsum, self.use, self.sw, self.nrm = stack, wsum, use, sw, nrm
        self._keep("average_1", stack=stack, pixels=wsum, kept=use, brightness=self.a_rel, level=self.lev_rel,
                   leftover=self.rms, noise=self.noise)

    def restack(self, good, scale):
        """The plain stack again, from the hole pass's centres (``good``: device mask of the holes to stack;
        brightness goes back to the stack's unit by the page fit's scale)."""
        if not bool(good.any()):
            raise ValueError("no hole of usual brightness and leftover to stack after the hole pass")
        self.model_a = self.a
        self.a = cp.maximum(self.a / scale, 1e-3)
        self.use = good
        self.stack, self.wsum, self.sw, self.nrm = self._stack(good)
        self.a_rel = self.a
        self.noise = float(cp.median(self.hole_rms[good]))
        self._keep("average_2", stack=self.stack, pixels=self.wsum, kept=good, noise=self.noise)

    # ------------------------------------------------------------------------------------------ pair map
    def pair_map(self, sf, keep=None):
        """Gamma_q (U, U, nky, nkx) on the page fitter's band. ``keep``: a dict that receives ``neighbour_map``
        (where the stacked holes' neighbours and they themselves stand, weighted by brightness: the sum of the 16
        classes' maps, ``neighbour_bins_per_px`` samples per pixel, the stacked hole in the middle) and
        ``holes_per_class`` (U, U), the stacked holes of each sub-pixel class."""
        U, Ug = self.U, self.pair_U
        Ng = self.L * Ug
        v = -self.c * U
        nb = cp.rint(v)
        q = cp.mod(nb, U).astype(cp.int32)
        s = cp.ascontiguousarray((v - nb) / U)
        cls = cp.ascontiguousarray(q[:, 1] * U + q[:, 0])
        wq = cp.bincount(cls, weights=self.sw, minlength=U * U)
        g = cp.where(self.sw > 0, self.sw * self.nrm / cp.maximum(wq[cls], 1e-300), 0.0)
        # stack = mean window: a neighbour counts by its brightness over the stacked mean
        nb = self.a / float(self.a[self.use].mean())
        # the cell list was built on the seeds; centres have moved by a fraction of a pixel since (the search square
        # is cut by distance, so a hole that changed cell is still found unless it sits within that move of the cut)
        hist = cp.zeros(U * U * Ng * Ng, dtype=cp.float32)
        self.mod.get_function("pair_splat")(((self.n + 63) // 64,), (64,),
                                             (self.c, cp.ascontiguousarray(nb), g, cls, s, self.cells.order, self.cells.start,
                                              np.int32(self.n), np.int32(self.cells.ncx), np.int32(self.cells.ncy),
                                              np.float64(self.cells.cs), np.float64(self.reach), np.int32(Ug),
                                              np.int32(Ng), hist))
        if keep is not None:
            keep.update(neighbour_map=cp.asnumpy(hist.reshape(U * U, Ng, Ng).sum(axis=0)), neighbour_bins_per_px=Ug,
                        holes_per_class=cp.asnumpy(wq).reshape(U, U))
        spec = cp.fft.rfft2(hist.reshape(U * U, Ng, Ng))
        ky = np.arange(-sf.kmax, sf.kmax + 1)
        rows = cp.asarray(np.mod(ky, Ng))
        spec = spec[:, rows, :sf.kmax + 1]
        fy, fx = np.meshgrid(ky / self.L, np.arange(sf.kmax + 1) / self.L, indexing="ij")
        centre = np.exp(-2j * np.pi * (fx + fy) * ((Ng // 2) / Ug))
        fix = np.sinc(fx / Ug) ** 4 * np.sinc(fy / Ug) ** 4
        gamma = spec / cp.asarray(centre * fix)[None]
        return gamma.reshape(U, U, sf.nky, sf.nkx)

    def stack_data(self, sf):
        """The stack and its weights by class for the page fitter. Weights in 1 / variance units (the variance of
        a bin's mean is noise^2 / count): the weak priors on the blur's second part need a chi-square cost."""
        y, mask = sf.to_classes(cp.where(cp.isfinite(self.stack), self.stack, 0.0))
        w, _ = sf.to_classes(self.wsum)
        w = cp.where(mask, w, 0.0) / (self.noise ** 2)
        return y, w

    def start_theta(self):
        """A round outline as the first guess: its half width from the stack's second moment about the centre (the
        filled bins within 0.6 spacings, light above the stack's low level), by the library's rule
        blob width^2 = blur width^2 + (outline / 1.1)^2 with a blur sigma of 1 px. The grid of starts scans 0.4 to
        1.8 times this size. (Filled bins only: a pitch of a whole number of pixels leaves most bins empty.)"""
        st = cp.asnumpy(self.stack)
        n = np.arange(self.Ms) - self.Ms // 2
        rr = np.hypot(n[None, :], n[:, None]) / self.U
        ok = np.isfinite(st)
        if not ok.any():
            return self.t.start(0.6, 1.0)
        base = np.percentile(st[ok], 10)
        core = ok & (rr <= 0.6 * self.spacing)
        light = np.clip(st - base, 0.0, None)
        total = float(light[core].sum())
        if total <= 0:
            return self.t.start(0.6, 1.0)
        sigma2 = float((light[core] * rr[core] ** 2).sum()) / total / 2.0  # per axis
        fw = 2.355 * math.sqrt(max(sigma2, 0.0))
        return self.t.start(max(0.5 * math.sqrt(max(fw ** 2 - 2.355 ** 2, 0.25)), 0.6), 1.0)

    # ---------------------------------------------------------------------------------------- hole pass
    def set_template(self, theta, scale, const, far_box=128, far_ramp=2.0, tU=8):
        """The fitted template on a fine grid (near part), its edge-offset and x / y derivative grids, the far
        kernel's transform, and every hole's starting brightness and level in the model's units."""
        self.theta = np.asarray(theta, float)
        tbox = max(32, int(8 * math.ceil(2 * (self.R + 6) / 8)))  # the fit window and the near light inside the box
        device = cp.cuda.Device().id
        key = (device, tbox, tU)
        if key not in _bands:
            _bands[key] = (Spec(self.t, n_poly=256, xp=cp), Band(box_px=tbox, f_max=0.75, upsample=tU, xp=cp))
        spec, band = _bands[key]
        spec.t = self.t
        h = 0.1
        hat = cp.stack([spec.template_hat(self.theta, band.fx, band.fy, d) for d in (0.0, h, -h)])
        t0g, tp, tm = band.to_grid(hat)
        td = (tp - tm) / (2 * h)
        t2 = (tp + tm - 2 * t0g) / (2 * h * h)
        N = band.N
        self.tN, self.tU = N, tU
        # reach of the template: where it falls below 1e-3 of its peak (library's rule), capped at the window + 3
        prof = cp.abs(t0g[N // 2])
        above = cp.nonzero(prof > 1e-3 * prof.max())[0]
        reach_px = float(above[-1] - above[0] + 1) / (2 * tU)
        Rr = min(int(math.ceil(reach_px)) + 1, N // (2 * tU) - 2, self.R + 3)
        xg = (cp.arange(N) - N // 2) / tU
        rr = cp.hypot(xg[None, :], xg[:, None])
        r0 = float(Rr)
        s = cp.clip((rr - r0) / far_ramp, 0.0, 1.0)
        wn = 0.5 * (1 + cp.cos(np.pi * s))
        t0n, tdn, t2n = t0g * wn, td * wn, t2 * wn
        Rr = min(int(math.ceil(r0 + far_ramp)) + 1, N // (2 * tU) - 2)
        # x / y derivatives of the near template (band-limited: by its spectrum, as the library)
        f1 = cp.fft.fftfreq(N, d=1.0 / tU)
        sp = cp.fft.fft2(cp.fft.ifftshift(t0n))
        tx = cp.fft.fftshift(cp.fft.ifft2(sp * (2j * np.pi * f1[None, :])).real)
        ty = cp.fft.fftshift(cp.fft.ifft2(sp * (2j * np.pi * f1[:, None])).real)
        dt = cp.float32 if self.tt == "float" else cp.float64
        self.g5 = cp.ascontiguousarray(cp.stack([t0n, tdn, tx, ty, t2n], axis=-1).astype(dt))
        self.g3 = cp.ascontiguousarray(cp.stack([t0n, tdn, t2n], axis=-1).astype(dt))
        self.Rr = Rr
        self.r_light = r0 + far_ramp  # no near light beyond this distance from a hole
        # the far kernel: the template's smooth tail at whole-pixel offsets
        fkey = (device, "far", far_box)
        if fkey not in _bands:
            _bands[fkey] = (Spec(self.t, n_poly=128, xp=cp), Band(box_px=far_box, f_max=0.49, upsample=1, xp=cp))
        far_spec, fb = _bands[fkey]
        far_spec.t = self.t
        b0 = fb.to_grid(far_spec.template_hat(self.theta, fb.fx, fb.fy, 0.0))
        B = far_box
        xb = cp.arange(B) - B // 2
        rb = cp.hypot(xb[None, :], xb[:, None])
        wf = 1.0 - 0.5 * (1 + cp.cos(np.pi * cp.clip((rb - r0) / far_ramp, 0.0, 1.0)))
        cmin, cmax = self.c.min(axis=0), self.c.max(axis=0)
        out_x = max(0.0, -float(cmin[0]), float(cmax[0]) - (self.w - 1))
        out_y = max(0.0, -float(cmin[1]), float(cmax[1]) - (self.h - 1))
        my, mx = B // 2 + int(math.ceil(out_y)) + 4, B // 2 + int(math.ceil(out_x)) + 4
        hp, wp = _even_fast(self.h + 2 * my), _even_fast(self.w + 2 * mx)
        pad = cp.zeros((hp, wp), dtype=cp.float32)
        pad[:B, :B] = (b0 * wf).astype(cp.float32)
        self.far = dict(B=B, my=my, mx=mx, hp=hp, wp=wp, kf=cp.fft.rfft2(pad), r0=r0)
        # model units: I_i = a_rel_i x (scale T*G + const) + lev_rel_i  ->  a = a_rel x scale, level = a_rel x const + lev_rel
        self.a = cp.maximum(self.a_rel * scale, 1e-3)
        self.level = self.a_rel * const + self.lev_rel
        self.delta = cp.zeros(self.n)
        self.cd0 = self.t.cd_px(self.theta)

    def far_field(self):
        """The holes' far light on the page (one FFT convolution of their brightness with the far kernel). A hole's
        far light follows its total light: brightness x area, so the edge offset enters through the area."""
        f = self.far
        grid = cp.zeros((f["hp"], f["wp"]), dtype=cp.float32)
        amp = self.a * (1.0 + 4.0 * self.delta / self.cd0)  # area ratio to first order: 1 + 2 delta / radius
        self.mod.get_function("far_splat")(((self.n + 127) // 128,), (128,),
                                            (self.c, amp, np.int32(self.n), np.int32(f["mx"]), np.int32(f["my"]),
                                             np.int32(f["hp"]), np.int32(f["wp"]), grid))
        out = cp.fft.irfft2(cp.fft.rfft2(grid) * f["kf"], s=(f["hp"], f["wp"]))
        oy, ox = f["my"] + f["B"] // 2, f["mx"] + f["B"] // 2
        return out[oy:oy + self.h, ox:ox + self.w].astype(cp.float64)

    def colours(self):
        """Colour classes for Gauss-Seidel: holes of one colour are at least 1.5 spacings apart. Cells of 1.5
        spacings; colour = (cell x parity, cell y parity, position of the hole inside its cell)."""
        if self._chunks is not None:
            return self._chunks
        cl = CellList(self.c, self.h, self.w, 1.5 * self.spacing)
        cx = cl.cell % cl.ncx
        cy = cl.cell // cl.ncx
        col = (cx % 2) + 2 * (cy % 2) + 4 * cl.rank
        ncol = int(col.max()) + 1
        order = cp.argsort(col).astype(cp.int32)
        counts = cp.asnumpy(cp.bincount(col, minlength=ncol))
        chunks, s = [], 0
        for k in counts:
            if k:
                chunks.append(order[s:s + int(k)])
            s += int(k)
        self._chunks = chunks
        return chunks

    def fit_holes(self, sweeps=2, size=False, max_move=0.25, robust_c=3.0, relax=1.0, keep=None):
        """The library's hole stage (Gauss-Seidel over colours, complete page model, robust weights) for a fixed
        number of sweeps: centre, brightness and level of every hole (``size``: its edge offset too). Returns
        per-sweep statistics. ``keep``: a dict that receives ``light``, the complete page model the last sweep
        ended with (device, (H, W): every hole's near and far light, without the holes' levels)."""
        kA, kU, kB = (self.mod.get_function(k) for k in ("hole_normal", "hole_update", "render_scatter"))
        chunks = self.colours()
        H, W, N, U, R, Rr = self.h, self.w, self.tN, self.tU, self.R, self.Rr
        side_r = 2 * Rr + 1
        P = side_r * side_r
        max_delta = 0.3 * 0.5 * self.cd0
        page = cp.zeros(H * W)
        rl2 = np.float64((self.r_light + 0.75) ** 2)  # beyond this offset from the nearest pixel a hole has no near light
        caches = [cp.zeros(int(idx.shape[0]) * P, dtype=cp.float32) for idx in chunks]
        c_olds = []
        for idx, cache in zip(chunks, caches):
            n = int(idx.shape[0])
            c_old = cp.ascontiguousarray(self.c[idx])
            kB(((n * P + 255) // 256,), (256,), (page, cache, self.g3, idx, c_old, self.c, self.a, self.delta,
                                               np.int32(n), np.int32(N), np.int32(U), np.int32(Rr), np.int32(H), np.int32(W), rl2))
            c_olds.append(c_old)
        far = self.far_field().ravel()
        page += far
        nv0, nv1 = self.noise ** 2, 0.0
        rms_typ = self.noise
        a_med = float(cp.median(self.a))
        nmax = max(int(i.shape[0]) for i in chunks)
        acc = cp.zeros((nmax, 22))
        stat = cp.zeros((self.n, 4))
        offs = np.cumsum([0] + [int(i.shape[0]) for i in chunks])
        log = []
        for it in range(sweeps):
            for j, idx in enumerate(chunks):
                n = int(idx.shape[0])
                acc.fill(0)
                kA(((n * (2 * R + 1) + 127) // 128,), (128,),
                   (self.img, page, self.g5, idx, self.c, self.a, self.level, self.delta, np.int32(n), np.int32(N),
                    np.int32(U), np.int32(R), np.int32(H), np.int32(W), np.float64(nv0), np.float64(nv1),
                    np.float64(robust_c * max(rms_typ, 1e-6)), np.int32(size), acc))
                st = stat[offs[j]:offs[j + 1]]
                kU(((n + 127) // 128,), (128,),
                   (acc, idx, self.c, self.a, self.level, self.delta, c_olds[j], np.int32(n), np.int32(size),
                    np.float64(max_move), np.float64(max_delta), np.float64(a_med), np.float64(relax), st))
                kB(((n * P + 255) // 256,), (256,),
                   (page, caches[j], self.g3, idx, c_olds[j], self.c, self.a, self.delta, np.int32(n), np.int32(N),
                    np.int32(U), np.int32(Rr), np.int32(H), np.int32(W), rl2))
            if it + 1 < sweeps:
                new_far = self.far_field().ravel()
                page += new_far - far
                far = new_far
            rms_typ = float(cp.median(stat[:, 2]))
            q = cp.asnumpy(cp.percentile(stat[:, :2], 99, axis=0))
            log.append(dict(sweep=it + 1, move99=float(q[0]), delta99=float(q[1]), rms=rms_typ))
        order = cp.concatenate(chunks)
        self.hole_rms = cp.empty(self.n)
        self.hole_rms[order] = stat[:, 2]
        if keep is not None:
            keep["light"] = page.reshape(H, W)
        return log

    def good_mask(self):
        """Device mask of the holes fit for the stack after a hole pass: window inside the page, usual brightness,
        usual leftover (the library's rule, without the edge-offset term: the size is held)."""
        inside = self.interior()
        med = cp.median(self.a[inside])
        typical = cp.median(self.hole_rms[inside])
        bright = (self.a > 0.4 * med) & (self.a < 2.5 * med)
        quiet = self.hole_rms < 2.0 * typical
        if self.trace is not None:
            self._keep("kept_2", inside=inside, usual_brightness=bright, usual_leftover=quiet,
                       median_brightness=float(med), typical_leftover=float(typical), brightness_limits=(0.4, 2.5),
                       leftover_limit=2.0)
        return inside & bright & quiet

    def picture(self, light):
        """The method's picture of the image (device, (H, W)): every hole's light (``light`` of ``fit_holes``)
        plus, at each pixel, the level of its nearest hole (a hole's level holds over its own window)."""
        from .voronoi import _labels_device

        return light + self.level[_labels_device(cp.asnumpy(self.c), (self.h, self.w))]


# ---------------------------------------------------------------------------------------------- driver
def _record_fits(trace, sf, members, pages, record, maps, starts, gam, ys, ws, theta, scale, const, cost, info):
    """What a page fit did, per page, into ``trace`` (``page_fits`` describes the record). The best match is the
    fit's own model at its end point (the same synthesis, in the same precision), and the hole alone is that
    model with nothing but the hole itself in the pair map."""
    data = sf.prepare(gam, ys, ws)
    hat = sf.template_hat(theta)
    with_neighbours = sf.synth(hat, data["gamma"])[:, 0].astype(cp.float64)
    alone = sf.synth(hat, cp.ones_like(data["gamma"]))[:, 0].astype(cp.float64)
    nb = 2 * sf.nmax + 1
    for k, i in enumerate(members):
        pg = pages[i]
        s, c = float(scale[k]), float(const[k])
        match = s * with_neighbours[k] + c
        distance = match - ys[k]
        lo = pg.Ms // 2 - sf.nmax
        steps = [dict(stage=r["stage"], iteration=int(r["iteration"]), theta=np.array(r["theta"][k], dtype=float),
                      cost=float(r["cost"][k]), tried=float(r["tried"][k]), accepted=bool(r["accepted"][k]),
                      damping=float(r["damping"][k]), done=bool(r["done"][k])) for r in record["iterations"]]
        trace[i] = dict(
            start=np.array(starts[k], dtype=float), grid=record["grid"][k] if "grid" in record else None,
            iterations=steps, stages=[(name, int(its), bool(stopped[k])) for name, its, stopped in info],
            theta=np.array(theta[k], dtype=float), scale=s, const=c, cost=float(cost[k]),
            data_cost=float((ws[k] * distance * distance).sum()), bins=int((ws[k] > 0).sum()), noise=float(pg.noise),
            bins_per_px=sf.U, average=cp.asnumpy(pg.stack)[lo:lo + nb, lo:lo + nb],
            pixels=cp.asnumpy(pg.wsum)[lo:lo + nb, lo:lo + nb], fitted=sf.from_classes(cp.asnumpy(ws[k] > 0)) > 0,
            match=sf.from_classes(cp.asnumpy(match)), alone=sf.from_classes(cp.asnumpy(s * alone[k])), **maps[k])


def page_fits(pages, blur, theta0=None, stages=("round", "super", "full"), grid=True, max_iter=(12, 20, 30), ftol=1e-7,
              trace=None):
    """The page fit of several pages at once (grouped by window size, each group in lockstep). Per page:
    (theta, scale, const, dict(chi2, bins, iterations, converged)).

    ``trace``: a list with one entry per page that receives a dict of what the fit did, without changing it:
    ``start`` (theta), ``grid`` (the scan of starts, or None), ``iterations`` (per stage start and iteration:
    stage, iteration, theta, cost, tried, accepted, damping, done), ``stages`` [(name, iterations, stopped by its
    tolerance)], ``theta``, ``scale``, ``const``, ``cost`` (with the priors), ``data_cost`` (the chi-square of the
    bins alone), ``bins``, ``noise``; on the fit's bins ((2 nmax + 1) square, ``bins_per_px`` per pixel, NaN
    outside): ``average`` (the stack), ``pixels`` (pixels per bin), ``fitted`` (the bins the fit used), ``match``
    (the best match: scale x hole with its neighbours + const) and ``alone`` (scale x the hole without its
    neighbours); and ``pair_map``'s ``neighbour_map`` and ``holes_per_class``."""
    out = [None] * len(pages)
    groups = {}
    for i, pg in enumerate(pages):
        groups.setdefault(fitter_for(pg, blur), []).append(i)
    for sf, members in groups.items():
        gam, ys, ws, th, maps = [], [], [], [], []
        for i in members:
            pg = pages[i]
            maps.append(None if trace is None else {})
            gam.append(pg.pair_map(sf, keep=maps[-1]))
            y, w = pg.stack_data(sf)
            ys.append(y)
            ws.append(w)
            th.append(pg.start_theta() if theta0 is None else theta0[i])
        ws, gam, ys, th = cp.stack(ws), cp.stack(gam), cp.stack(ys), np.stack(th)
        record = None if trace is None else {}
        theta, scale, const, cost, info = sf.fit(th, gam, ys, ws, stages=stages, grid=grid, max_iter=max_iter,
                                                 ftol=ftol, trace=record)
        bins = cp.asnumpy((ws > 0).reshape(len(members), -1).sum(axis=1))
        for k, i in enumerate(members):
            out[i] = (theta[k], float(scale[k]), float(const[k]),
                      dict(chi2=float(cost[k]), bins=int(bins[k]), iterations=int(info[-1][1]),
                           converged=bool(info[-1][2][k])))
        if trace is not None:
            _record_fits(trace, sf, members, pages, record, maps, th, gam, ys, ws, theta, scale, const, cost, info)
    return out


def in_order(pages, blur, fits, max_iter=FINAL_ITERATIONS, trace=None):
    """gauss2 only: pages whose final fit ended with the second Gaussian the narrower one (stopped by the bound on
    its weight, short of the minimum) are fitted again from the exchanged labels, together; per page the end with
    the lower cost is kept. Sets ``exchanged`` in every fit's record (False for any other ``blur``).

    ``trace``: a list with one entry per page that receives the record of the page's second fit (as
    ``page_fits``' ``trace``, with ``kept``: its end was taken), or None for a page that needed none."""
    for f in fits:
        f[3]["exchanged"] = False
    if blur != "gauss2":
        return fits
    idx = [i for i, f in enumerate(fits) if second_is_narrower(pages[i].t, f[0])]
    if not idx:
        return fits
    second = None if trace is None else [None] * len(idx)
    again = page_fits([pages[i] for i in idx], blur, theta0=[exchanged(pages[i].t, fits[i][0]) for i in idx],
                      stages=("full",), grid=False, max_iter=(max_iter,), trace=second)
    out = list(fits)
    for k, (i, f) in enumerate(zip(idx, again)):
        f[3]["exchanged"] = bool(f[3]["chi2"] < fits[i][3]["chi2"])
        if f[3]["exchanged"]:
            out[i] = f
        if trace is not None:
            trace[i] = dict(second[k], kept=f[3]["exchanged"])
    return out


def pixel_noise(leftover):
    """One pixel's noise (sd) from an image's leftover (device, (H, W)), by its pixel-to-pixel jitter alone
    (Immerkaer's estimate in its robust form): the leftover is filtered with [[1, -2, 1], [-2, 4, -2], [1, -2, 1]],
    which removes whatever is smooth over three pixels (what a blurred hole can leave behind) and multiplies
    uncorrelated noise by 6; sd = 1.4826 x the median absolute value / 6. Noise that neighbouring pixels share is
    not seen by it, and where the noise grows with the light this is the noise of the usual (dark) pixel."""
    r = leftover
    jitter = (r[:-2, :-2] + r[:-2, 2:] + r[2:, :-2] + r[2:, 2:] + 4.0 * r[1:-1, 1:-1]
              - 2.0 * (r[:-2, 1:-1] + r[2:, 1:-1] + r[1:-1, :-2] + r[1:-1, 2:]))
    return 1.4826 * float(cp.median(cp.abs(jitter))) / 6.0


def redraw(pg, theta, scale, const):
    """The method's picture of the image under its final answer, from a page that has been through steps A to C:
    the final outline and blur are set and every hole's centre, brightness and level are fitted under them (the
    hole pass, as step B); the picture is every hole's near and far light plus, at each pixel, the level of its
    nearest hole. The page is changed by this: call it after its results have been taken. Returns host arrays:
    ``picture`` (float32 (H, W), bright holes, no noise), ``pixel_noise`` (``pixel_noise`` of image minus picture),
    ``leftover`` (the median leftover of a stacked hole's window: noise and whatever the picture misses) and per
    hole ``centers_xy``, ``brightness``, ``level``, ``rms``. An image redrawn from the answer is the picture plus
    noise of ``pixel_noise``."""
    pg.trace = None
    pg.set_template(theta, scale, const)
    kept = {}
    pg.fit_holes(sweeps=CENTRING_SWEEPS, size=False, keep=kept)
    picture = pg.picture(kept["light"])
    return dict(picture=cp.asnumpy(picture.astype(cp.float32)), pixel_noise=pixel_noise(pg.img - picture),
                leftover=float(cp.median(pg.hole_rms[pg.use])), centers_xy=cp.asnumpy(pg.c),
                brightness=cp.asnumpy(pg.a), level=cp.asnumpy(pg.level), rms=cp.asnumpy(pg.hole_rms))


def fit_images(images, seeds=None, *, blur="gauss2", pair_reach_px=None, exchange=True, log=None, trace=None,
               redrawn=False):
    """Steps A to C for a list of images (bright holes; the page fits run in lockstep). ``seeds``: one (N, 2)
    array per image, or None for the method's own seed finder. ``pair_reach_px``: how far from a stacked hole its
    neighbours' light is in the stack's model (default: window half width + ``PAIR_REACH_MARGIN_PX``).
    ``exchange=False`` leaves out the second look at gauss2 fits that end exchanged (for measurements only).

    Returns one dict per image, host arrays: per hole ``centers_xy`` (after B), ``brightness`` and ``level`` (the
    model's units: clear-area light above the hole's level), ``rms`` (leftover of its window), ``good`` (stacked
    in C); per page ``theta``, ``scale``, ``const``, ``chi2``, ``bins``, ``iterations``, ``converged``,
    ``exchanged``, ``theta_first`` (step A), ``stack`` and ``stack_weight`` (bins of 1 / ``upsample`` px, NaN and 0
    where no pixel fell), ``spacing_px``, ``window_px``, ``noise_sd``, ``seconds`` (the call's time / images).

    ``trace``: a list that receives one dict per image with what every step produced (host arrays), for
    ``scripts/outline_fit_steps.py``; the returned values are the same with and without it. Keys, in the order of
    the calculation: ``window`` (seeds, nearest_px, spacing_px, window_px, bins_per_px, stack_bins, reach_px,
    box_px); ``average_all`` (the stack of every interior hole and its pixels per bin; per hole inside, brightness
    relative to the median, level, leftover, compared, kept; the limits used); ``average_1`` (the stack without
    the unusual holes: the first fit's data; per hole brightness, level, leftover against it; noise);
    ``fit_1`` (``page_fits``' record); ``holes`` (after the hole pass: centers_xy, brightness, level, leftover,
    passes, and ``picture``, the method's picture of the image, float32 (H, W)); ``kept_2`` (who is stacked
    again, by reason); ``average_2``; ``fit_2``; ``exchange`` (the second fit of a gauss2 fit that ended
    exchanged, with ``kept``, or None). With ``redrawn=True`` also ``redrawn``: ``redraw``'s picture of the image
    under the final answer, and the image's pixel noise."""
    started = time.perf_counter()

    def say(text):
        if log:
            cp.cuda.get_current_stream().synchronize()
            log(f"{text} ({time.perf_counter() - started:.2f} s)")

    tracing = trace is not None
    steps = [{} for _ in images] if tracing else [None] * len(images)
    if tracing:
        trace.extend(steps)  # filled as the steps run: a fit that fails leaves the steps it reached
    pages = []
    for k, image in enumerate(images):
        pg = GpuPage(image, blur=blur, seeds=None if seeds is None else seeds[k], pair_reach_px=pair_reach_px,
                     trace=steps[k])
        pg.build_stack()
        pages.append(pg)
    say("A: seeds and stack")
    recorded = [None] * len(pages) if tracing else None
    first = page_fits(pages, blur, stages=FIRST_STAGES, max_iter=FIRST_ITERATIONS, ftol=FIRST_FTOL, trace=recorded)
    if tracing:
        for s, r in zip(steps, recorded):
            s["fit_1"] = r
    say("A: first page fit")
    for pg, (theta, scale, const, _) in zip(pages, first):
        pg.set_template(theta, scale, const)
        kept = {} if tracing else None
        passes = pg.fit_holes(sweeps=CENTRING_SWEEPS, size=False, keep=kept)
        if tracing:
            pg._keep("holes", centers_xy=pg.c, brightness=pg.a, level=pg.level, leftover=pg.hole_rms, passes=passes,
                     picture=pg.picture(kept["light"]).astype(cp.float32))
        pg.restack(pg.good_mask(), scale)
    say("B: hole pass and stack")
    recorded = [None] * len(pages) if tracing else None
    fits = page_fits(pages, blur, theta0=[f[0] for f in first], stages=("full",), grid=False,
                     max_iter=(FINAL_ITERATIONS,), trace=recorded)
    second = [None] * len(pages) if tracing else None
    fits = in_order(pages, blur if exchange else None, fits, trace=second)
    if tracing:
        for s, r, x in zip(steps, recorded, second):
            s["fit_2"], s["exchange"] = r, x
    say("C: final page fit")
    out = []
    for pg, f0, (theta, scale, const, record) in zip(pages, first, fits):
        out.append(dict(
            centers_xy=cp.asnumpy(pg.c), brightness=cp.asnumpy(pg.model_a), level=cp.asnumpy(pg.level),
            rms=cp.asnumpy(pg.hole_rms), good=cp.asnumpy(pg.use), theta=np.asarray(theta, float), scale=scale,
            const=const, theta_first=np.asarray(f0[0], float), stack=cp.asnumpy(pg.stack),
            stack_weight=cp.asnumpy(pg.wsum), upsample=pg.U, spacing_px=pg.spacing, window_px=pg.R,
            pair_reach_px=pg.reach, noise_sd=pg.noise, **record))
    seconds = (time.perf_counter() - started) / max(len(images), 1)
    for o in out:
        o["seconds"] = seconds
    if tracing and redrawn:  # after the results are on the host: the picture changes the pages
        for s, pg, (theta, scale, const, _) in zip(steps, pages, fits):
            s["redrawn"] = redraw(pg, theta, scale, const)
    return out
