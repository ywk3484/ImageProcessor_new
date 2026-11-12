"""Scratch: the outline fit's hole template straight from its spectrum, at any frequency (no fine indicator grid).

The library's model (subpx._outline.model.Template):  T = pixel box * blur * indicator(outline moved by delta).
In frequency space every factor is known in closed form:

    T^(f) = O^(f; outline, delta) x OTF(f; blur) x sinc(fx) sinc(fy)

O^ is the Fourier transform of the sharp region. By Green's theorem it is a line integral round the outline; the
outline is taken as a polygon of ``n_poly`` vertices at equal angles, whose transform is exact:

    O^(f) = i / (2 pi |f|^2)  sum_edges (f x e) sinc(f . e) exp(-2 pi i f . m)      e = edge vector, m = edge midpoint
    O^(0) = area

Parameters, bounds, CD: the library Template's own (theta is the same vector), so the two can be compared directly.
Works with NumPy or CuPy arrays (``xp``); frequencies in cycles per pixel.
"""
from __future__ import annotations

import math

import numpy as np


class Spec:
    def __init__(self, lib_template, n_poly=512, xp=np):
        self.t = lib_template  # parameter bookkeeping only (unpack, _radius_at, bounds, cd_px)
        self.n_poly = int(n_poly)
        self.xp = xp
        self.phi = 2 * np.pi * np.arange(self.n_poly) / self.n_poly

    # ---------------------------------------------------------------------------------------------- outline
    def vertices(self, theta, delta=0.0):
        """Polygon vertices (x, y) of the outline moved ``delta`` px along its normal (the library's radial form:
        r -> r + delta hypot(r, r') / r)."""
        r, dr = self.t._radius_at(np.asarray(theta, float), self.phi)
        if delta:
            r = r + delta * np.hypot(r, dr) / r
        return r * np.cos(self.phi), r * np.sin(self.phi)

    def outline_hat(self, theta, fx, fy, delta=0.0):
        """O^ at frequencies (fx, fy) (same shape, any); returns complex array of that shape."""
        xp = self.xp
        x, y = self.vertices(theta, delta)
        x2, y2 = np.roll(x, -1), np.roll(y, -1)
        ex, ey = xp.asarray(x2 - x), xp.asarray(y2 - y)
        mx, my = xp.asarray(0.5 * (x + x2)), xp.asarray(0.5 * (y + y2))
        area = 0.5 * float(np.sum(x * y2 - x2 * y))
        shape = fx.shape
        fx = fx.reshape(-1, 1)
        fy = fy.reshape(-1, 1)
        f2 = fx * fx + fy * fy
        cross = fx * ey[None, :] - fy * ex[None, :]
        dot = fx * ex[None, :] + fy * ey[None, :]
        phase = -2 * np.pi * (fx * mx[None, :] + fy * my[None, :])
        s = (cross * xp.sinc(dot) * xp.exp(1j * phase)).sum(axis=1, keepdims=True)
        zero = f2 == 0
        out = xp.where(zero, area, 1j * s / (2 * np.pi * xp.where(zero, 1.0, f2)))
        return out.reshape(shape)

    # ------------------------------------------------------------------------------------------------- blur
    def otf(self, theta, fx, fy):
        xp = self.xp
        bp = self.t.unpack(np.asarray(theta, float))[5]
        blur = self.t.blur
        if blur == "gauss2":
            s1x, s1y, s2x, s2y = np.exp(bp[:4])
            w2 = 1.0 / (1.0 + math.exp(-bp[4]))
            g1 = xp.exp(-2 * np.pi ** 2 * (s1x ** 2 * fx ** 2 + s1y ** 2 * fy ** 2))
            g2 = xp.exp(-2 * np.pi ** 2 * (s2x ** 2 * fx ** 2 + s2y ** 2 * fy ** 2))
            return (1 - w2) * g1 + w2 * g2
        sx, sy = np.exp(bp[:2])
        g = xp.exp(-2 * np.pi ** 2 * (sx ** 2 * fx ** 2 + sy ** 2 * fy ** 2))
        if blur == "lens":
            s = xp.clip(xp.hypot(fx, fy) / math.exp(bp[2]), 0.0, 1.0)
            g = g * (2 / np.pi) * (xp.arccos(s) - s * xp.sqrt(1.0 - s * s))
        return g

    def template_hat(self, theta, fx, fy, delta=0.0, box_px=0.0):
        """T^(f): outline x blur x pixel box (x a further box of ``box_px``, the stack's bin)."""
        xp = self.xp
        out = self.outline_hat(theta, fx, fy, delta) * self.otf(theta, fx, fy) * xp.sinc(fx) * xp.sinc(fy)
        if box_px > 0:
            out = out * xp.sinc(fx * box_px) * xp.sinc(fy * box_px)
        return out


class Band:
    """The frequencies k / L inside |fx|, |fy| <= f_max (half plane kx >= 0) and the way back to a fine grid of
    ``upsample`` samples per pixel over the box L (irfft2 of the zero-padded coefficients; origin in the middle)."""

    def __init__(self, box_px=48, f_max=0.75, upsample=8, xp=np):
        self.L = int(box_px)
        self.U = int(upsample)
        self.N = self.L * self.U
        self.xp = xp
        self.kmax = int(math.floor(f_max * self.L))
        ky = np.concatenate([np.arange(0, self.kmax + 1), np.arange(-self.kmax, 0)])  # fft order
        kx = np.arange(0, self.kmax + 1)
        self.ky, self.kx = ky, kx
        fy, fx = np.meshgrid(ky / self.L, kx / self.L, indexing="ij")
        self.fx, self.fy = xp.asarray(fx), xp.asarray(fy)
        self.rows = xp.asarray(np.where(ky >= 0, ky, self.N + ky))
        # phase that moves the origin to the middle sample N // 2 of the grid
        c = (self.N // 2) / self.U
        self.centre = xp.exp(-2j * np.pi * (self.fx + self.fy) * c)

    def to_grid(self, coeff):
        """Real image (..., N, N) of sum_k coeff_k exp(2 pi i f_k . u) / L^2 at u = (n - N // 2) / U."""
        xp = self.xp
        full = xp.zeros(coeff.shape[:-2] + (self.N, self.N // 2 + 1), dtype=complex)
        full[..., self.rows, :self.kmax + 1] = coeff * self.centre
        return xp.fft.irfft2(full, s=(self.N, self.N)) * (self.U ** 2)

    def from_grid(self, image):
        """Band coefficients of a real image on the grid (the inverse of ``to_grid`` for band-limited images)."""
        xp = self.xp
        spec = xp.fft.rfft2(image) / (self.U ** 2)
        return spec[..., self.rows, :self.kmax + 1] / self.centre
