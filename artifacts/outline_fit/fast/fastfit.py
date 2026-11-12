"""Scratch: one-pass outline fit, accuracy prototype (NumPy, CPU; the speed prototype is gpufit.py).

The existing fit alternates 10-14 times between the holes and the page's outline + blur, because its average hole is
"cleaned" of the neighbours' light with the very template it is looking for. Here the average hole is NOT cleaned.

    stack      S(b) = sum a_i I(p) / sum a_i^2        over the samples (hole i, pixel p) whose offset p - c_i rounds to
                                                       bin b (U bins per pixel)
    model      I(p) = sum_j a_j T(p - c_j) + level     T = outline x blur x pixel, known by its spectrum T^ (spec.py)

Put the second into the first. All bins with the same remainder q = (bin mod U) are filled by the same holes H_q (those
with that sub-pixel position), each at the same sub-bin remainder s_i. So

    S(b) = const + sum_f T^(f) Gamma_q(f) exp(2 pi i f . u_b) / L^2
    Gamma_q(f) = sum_{i in H_q} a_i sum_j a_j exp(-2 pi i f . (c_j - c_i - s_i)) / sum_{i in H_q} a_i^2

Gamma_q ("who stands where round a hole of sub-pixel position q", j = i included) is computed once per page from the
centres and brightnesses alone. It is exact for any arrangement: no lattice, and no assumption that a hole's
sub-pixel position says nothing about its neighbours' places (on the test file it does: pitch exactly 8.5 px with a
+-0.3 px shift per hole). The neighbours' light is thus part of the model, with the SAME template, so a single
least-squares fit of the outline and blur on S is the answer: no alternation, no relaxation.
Parameters, stages, bounds, grid of starts and the CD definition are the library's.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]  # the repo root (this file sits in artifacts/outline_fit/fast)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx._outline.model import Template, fit_stages  # noqa: E402

from spec import Band, Spec  # noqa: E402


# ------------------------------------------------------------------------------------------------ windows
def windows(image, c, R):
    """Each hole's (2R+1)^2 window round its nearest pixel: values, validity, offsets (pixel - centre) in x and y."""
    oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
    ox, oy = ox.ravel(), oy.ravel()
    r = np.rint(c).astype(np.int64)
    px = r[:, 0, None] + ox[None, :]
    py = r[:, 1, None] + oy[None, :]
    h, w = image.shape
    valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    vals = image[np.clip(py, 0, h - 1), np.clip(px, 0, w - 1)]
    return vals, valid, px - c[:, 0, None], py - c[:, 1, None]


def bins_of(ux, uy, U, N):
    bx = np.rint(ux * U).astype(np.int64) + N // 2
    by = np.rint(uy * U).astype(np.int64) + N // 2
    inside = (bx >= 0) & (bx < N) & (by >= 0) & (by < N)
    return np.clip(by, 0, N - 1) * N + np.clip(bx, 0, N - 1), inside


def plain_stack(vals, valid, ux, uy, a, use, U, N):
    """a-weighted average of the windows on the fine grid (N, N): sum a I / sum a^2 per bin; (stack, weight)."""
    b, inside = bins_of(ux, uy, U, N)
    ok = valid & inside & use[:, None]
    aa = np.broadcast_to(a[:, None], vals.shape)
    sums = np.bincount(b[ok], weights=(vals * aa)[ok], minlength=N * N)
    wsum = np.bincount(b[ok], weights=(aa * aa)[ok], minlength=N * N)
    with np.errstate(invalid="ignore", divide="ignore"):
        stack = np.where(wsum > 0, sums / wsum, np.nan)
    return stack.reshape(N, N), wsum.reshape(N, N)


def amp_level(vals, valid, ux, uy, stack, U, N, core_px=None):
    """Each hole's brightness and level against the stack: least squares of window = a x stack(offset) + level
    (on the pixels within ``core_px`` of the hole, if given). Also the leftover rms per hole (whole window)."""
    b, inside = bins_of(ux, uy, U, N)
    s = stack.ravel()[b]
    full = valid & inside & np.isfinite(s)
    ok = full if core_px is None else full & (np.hypot(ux, uy) <= core_px)
    s = np.where(full, s, 0.0)
    y = np.where(full, vals, 0.0)
    so, yo = np.where(ok, s, 0.0), np.where(ok, y, 0.0)
    n = ok.sum(1)
    s1, s2, sy, y1 = so.sum(1), (so * so).sum(1), (so * yo).sum(1), yo.sum(1)
    det = n * s2 - s1 * s1
    det = np.where(det > 0, det, np.inf)
    a = (n * sy - s1 * y1) / det
    lev = (s2 * y1 - s1 * sy) / det
    res = np.where(full, y - a[:, None] * s - lev[:, None], 0.0)
    rms = np.sqrt((res * res).sum(1) / np.maximum(full.sum(1), 1))
    return a, lev, rms


def recentre(vals, valid, ux, uy, stack, U, N, a, lev, core_px):
    """One Gauss-Newton step of each hole's centre against the stack (its own gradient on the fine grid), on the
    pixels within ``core_px`` of the hole. Returns the move (dx, dy) in px."""
    gy, gx = np.gradient(np.where(np.isfinite(stack), stack, 0.0), 1.0 / U)
    b, inside = bins_of(ux, uy, U, N)
    s = stack.ravel()[b]
    ok = valid & inside & np.isfinite(s) & (np.hypot(ux, uy) <= core_px)
    res = np.where(ok, vals - a[:, None] * np.where(ok, s, 0.0) - lev[:, None], 0.0)
    jx = np.where(ok, -a[:, None] * gx.ravel()[b], 0.0)  # d model / d cx
    jy = np.where(ok, -a[:, None] * gy.ravel()[b], 0.0)
    sxx, sxy, syy = (jx * jx).sum(1), (jx * jy).sum(1), (jy * jy).sum(1)
    rx, ry = (jx * res).sum(1), (jy * res).sum(1)
    det = sxx * syy - sxy * sxy
    det = np.where(det > 0, det, np.inf)
    return np.stack([(syy * rx - sxy * ry) / det, (sxx * ry - sxy * rx) / det], axis=1)


# ------------------------------------------------------------------------------------- who stands where
def bspline_weights(t):
    """Cubic B-spline weights for taps at -1, 0, 1, 2; t in [0, 1)."""
    return np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6,
                     t ** 3 / 6], axis=-1)


def hole_classes(c, U):
    """Sub-pixel class (qx, qy in 0..U-1) of every hole and its sub-bin remainder s (px): for a pixel p the offset
    u = p - c lands in bin n = rint(u U), n mod U = q, and u = n / U + s."""
    v = -c * U
    n = np.rint(v)
    return np.mod(n, U).astype(np.int64), (v - n) / U


def pair_spectra(c, a, use, reach_px, U, band_g, resolved=True):
    """Gamma_q on the band's frequencies, shape (U * U or 1, ky, kx): for every stacked hole i of class q and every
    hole j within ``reach_px`` (j = i included), weight a_i a_j / sum_{i in q} a_i^2 at c_j - c_i - s_i, splatted with
    a cubic B-spline on band_g's grid, transformed, the spline's own transfer divided out.
    ``resolved`` False: one class for all holes and s = 0 (the stack's bin is then a box of 1 / U px in the model)."""
    from scipy.spatial import cKDTree

    xp = band_g.xp
    Ng, Ug = band_g.N, band_g.U
    q, s = hole_classes(c, U)
    if not resolved:
        q, s = np.zeros_like(q), np.zeros_like(s)
    nq = U * U if resolved else 1
    cls = q[:, 1] * U + q[:, 0] if resolved else np.zeros(len(c), dtype=np.int64)
    pairs = cKDTree(c).query_pairs(reach_px, output_type="ndarray")
    idx = np.arange(len(c))
    i = np.concatenate([pairs[:, 0], pairs[:, 1], idx])
    j = np.concatenate([pairs[:, 1], pairs[:, 0], idx])
    keep = use[i]
    i, j = i[keep], j[keep]
    wq = np.bincount(cls[use], weights=a[use] ** 2, minlength=nq)
    wgt = a[i] * a[j] / wq[cls[i]]
    d = (c[j] - c[i] - s[i]) * Ug + Ng // 2
    x0, y0 = np.floor(d[:, 0]).astype(np.int64), np.floor(d[:, 1]).astype(np.int64)
    wx, wy = bspline_weights(d[:, 0] - x0), bspline_weights(d[:, 1] - y0)
    hist = np.zeros(nq * Ng * Ng)
    base = cls[i] * Ng * Ng
    for ky in range(4):
        for kx in range(4):
            flat = base + np.mod(y0 + ky - 1, Ng) * Ng + np.mod(x0 + kx - 1, Ng)
            hist += np.bincount(flat, weights=wgt * wy[:, ky] * wx[:, kx], minlength=nq * Ng * Ng)
    spec = xp.fft.rfft2(xp.asarray(hist.reshape(nq, Ng, Ng)))[:, band_g.rows, :band_g.kmax + 1] / band_g.centre
    fix = xp.sinc(band_g.fx / Ug) ** 4 * xp.sinc(band_g.fy / Ug) ** 4
    return spec / fix, len(i)


# ------------------------------------------------------------------------------------------- the page fit
class FastFit:
    def __init__(self, image, seeds, *, blur="gauss2", U=4, pair_U=8, box_px=48, f_max=0.75, n_poly=256, pixel_nm=21.0,
                 pair_reach_px=20.0, window_px=None, resolved=True):
        self.image = np.asarray(image, dtype=np.float64)
        self.h, self.w = self.image.shape
        self.c = np.array(seeds, dtype=np.float64).reshape(-1, 2)
        self.n = len(self.c)
        self.U = int(U)
        self.pixel_nm = float(pixel_nm)
        self.resolved = bool(resolved)
        from scipy.spatial import cKDTree

        d, _ = cKDTree(self.c).query(self.c, k=2)
        self.spacing = float(np.median(d[:, 1]))
        self.R = int(window_px) if window_px else max(3, int(math.ceil(0.75 * self.spacing)))
        self.t = Template(upsample=8, size_px=24, blur=blur)  # parameter bookkeeping only
        self.spec = Spec(self.t, n_poly=n_poly)
        self.band = Band(box_px=box_px, f_max=f_max, upsample=self.U)
        self.band_g = Band(box_px=box_px, f_max=f_max, upsample=int(pair_U))
        self.N = self.band.N
        self.pair_reach = float(pair_reach_px)
        self.a = np.ones(self.n)
        self.level = np.zeros(self.n)
        self.log = []
        self.times = {}
        n = self.N
        yy, xx = np.mgrid[0:n, 0:n]
        self.bin_class = (np.mod(yy - n // 2, self.U) * self.U + np.mod(xx - n // 2, self.U)) if resolved else np.zeros((n, n), np.int64)

    def _tick(self, name, t0):
        self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - t0

    def interior(self, margin=None):
        m = self.R + 1 if margin is None else margin
        c = self.c
        return (c[:, 0] >= m) & (c[:, 0] <= self.w - 1 - m) & (c[:, 1] >= m) & (c[:, 1] <= self.h - 1 - m)

    def build_stack(self, passes=2, recentre_core=None, amp_core=None):
        """The plain stack with each hole's brightness from the stack itself; ``recentre_core`` (px) also moves
        every centre by one Gauss-Newton step against the stack before the last pass."""
        t0 = time.perf_counter()
        inside = self.interior()
        use = inside.copy()
        for k in range(passes):
            vals, valid, ux, uy = windows(self.image, self.c, self.R)
            stack, weight = plain_stack(vals, valid, ux, uy, self.a, use, self.U, self.N)
            a, lev, rms = amp_level(vals, valid, ux, uy, stack, self.U, self.N, core_px=amp_core)
            a = a / np.median(a[inside])
            typical = np.median(rms[inside])
            use = inside & (a > 0.4) & (a < 2.5) & (rms < 2.0 * typical)
            self.a, self.level, self.rms = np.maximum(a, 1e-3), lev, rms
            if recentre_core and k == passes - 2:
                move = recentre(vals, valid, ux, uy, stack, self.U, self.N, self.a, self.level, recentre_core)
                self.c = self.c + np.clip(move, -0.5, 0.5)
        vals, valid, ux, uy = windows(self.image, self.c, self.R)
        self.stack, self.weight = plain_stack(vals, valid, ux, uy, self.a, use, self.U, self.N)
        self.use = use
        self._tick("stack", t0)
        t0 = time.perf_counter()
        self.gamma, self.n_pairs = pair_spectra(self.c, self.a, use, self.pair_reach, self.U, self.band_g, self.resolved)
        self._tick("pairs", t0)
        return self.stack, self.weight

    def start_theta(self):
        """A round outline from the stack's half-height width (the library's own first guess)."""
        mid = self.N // 2
        prof = self.stack[mid]
        base = np.nanmin(prof)
        half = base + 0.5 * (prof[mid] - base)
        left = right = mid
        while left > 0 and np.isfinite(prof[left - 1]) and prof[left - 1] >= half:
            left -= 1
        while right + 1 < len(prof) and np.isfinite(prof[right + 1]) and prof[right + 1] >= half:
            right += 1
        fw = (right - left + 1) / self.U
        half_width = max(0.5 * math.sqrt(max(fw ** 2 - 2.355 ** 2, 0.25)), 0.6)
        return self.t.start(half_width, 1.0)

    def model(self, theta):
        """The stack's model without scale and constant, on the fine grid (N, N)."""
        box = 0.0 if self.resolved else 1.0 / self.U
        coeff = self.spec.template_hat(theta, self.band.fx, self.band.fy, 0.0, box_px=box)
        img = self.band.to_grid(coeff[None] * self.gamma)  # (classes, N, N)
        if not self.resolved:
            return img[0]
        return np.take_along_axis(img, self.bin_class[None], axis=0)[0]

    def fit_shape(self, theta0=None, stages=("round", "super", "full"), grid=True, radius_px=None):
        t0 = time.perf_counter()
        ok = np.isfinite(self.stack) & (self.weight > 0)
        if radius_px is not None:
            n = self.N
            yy, xx = np.mgrid[0:n, 0:n]
            ok &= np.hypot(xx - n // 2, yy - n // 2) <= radius_px * self.U
        sw = np.sqrt(self.weight[ok])
        y = self.stack[ok]
        self.evals = 0

        def render_for(theta):
            self.evals += 1
            return self.model(theta)[ok]

        theta0 = self.start_theta() if theta0 is None else theta0
        theta, scale, const, chi2 = fit_stages(self.t, render_for, y, sw, theta0, stages=stages, grid=grid,
                                               log=self.log.append, pixel_nm=self.pixel_nm, unit="nm")
        self.theta, self.scale, self.const = theta, scale, const
        res = scale * self.model(theta)[ok] + const - y
        self.stack_rms = float(np.sqrt(np.average(res * res, weights=sw * sw)))
        self.expected_rms = float(np.sqrt(np.average(1.0 / self.weight[ok], weights=sw * sw)))  # x noise sd
        self._tick("shape", t0)
        return theta

    def describe(self):
        d = self.t.describe(self.theta, self.pixel_nm)
        return dict(cd_nm=d["cd_nm"], blur=[float(v) for v in d["blur_fwhm_nm"]], p=d["superellipse"]["p"],
                    extent=(d["extent_x_nm"], d["extent_y_nm"]), harmonics=d["harmonics"], scale=self.scale,
                    const=self.const, evals=self.evals, stack_rms=self.stack_rms, good=int(self.use.sum()),
                    pairs=self.n_pairs)
