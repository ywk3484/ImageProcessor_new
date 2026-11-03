"""Prototype (not part of subpx): hole centers by fitting each hole with the image's own average hole.

Used by scripts/pitch_signature_test.py as the candidate "better method". It did not pass the gate set in the plan
(best on at least 6 of the 8 test images): it wins on large and elongated holes, ties on small round ones (the
Gaussian fit is already at the noise limit there) and is worse on the dense 60 nm array, where the average hole carries
the neighbours' light along (a moved row is seen 14 % too far). See docs/pitch_signature_test.md.

Model per hole i on the pixels of its own territory (closer to it than to any neighbour, 1 px soft edge):

    I(p) = a_i * T(p - c_i) + b_i

T is the average of all holes' pixels placed at their offsets from their centers on a grid 4 times finer than the
pixels, smoothed by 0.7 px (less leaves a once-per-pixel ripple when the holes sit at only a few heights inside their
pixel, e.g. 3 for rows 8.33 px apart), read with cubic interpolation. Gauss-Newton on (cx, cy, a, b) per hole, each
pixel weighted by 1 / (noise variance), the variance a straight line in the level fitted to the leftovers; then T is
rebuilt from the new centers, 3 rounds. The center is where T's light centroid inside its own territory is.

Any start within ~1 px works (the GPU weighted centroids here); one code path for NumPy and CuPy (``xp``).
"""

from __future__ import annotations

import math

import numpy as np


def _keys(t, xp):
    """Keys cubic (a = -0.5) weights and their derivatives for the taps at -1, 0, 1, 2; ``t`` in [0, 1)."""
    x = xp.stack((1 + t, t, 1 - t, 2 - t), axis=-1)
    sign = xp.asarray([1.0, 1.0, -1.0, -1.0])
    near = x <= 1
    w = xp.where(near, (1.5 * x - 2.5) * x * x + 1, ((-0.5 * x + 2.5) * x - 4) * x + 2)
    dw = xp.where(near, (4.5 * x - 5) * x, (-1.5 * x + 5) * x - 4)
    return w, dw * sign


def _sample(grid, sx, sy, xp):
    """Value and gradient (per grid step) of the cubic interpolant of ``grid`` at grid coordinates (sx, sy)."""
    size_y, size_x = grid.shape
    ix, iy = xp.floor(sx), xp.floor(sy)
    wx, dwx = _keys(sx - ix, xp)
    wy, dwy = _keys(sy - iy, xp)
    taps = xp.arange(-1, 3)
    cx = xp.clip(ix.astype(xp.int64)[..., None] + taps, 0, size_x - 1)
    cy = xp.clip(iy.astype(xp.int64)[..., None] + taps, 0, size_y - 1)
    values = grid[cy[..., :, None], cx[..., None, :]]  # (..., y tap, x tap)
    along_x = (values * wx[..., None, :]).sum(-1)
    slope_x = (values * dwx[..., None, :]).sum(-1)
    return (along_x * wy).sum(-1), (slope_x * wy).sum(-1), (along_x * dwy).sum(-1)


def _smooth(grid, sigma, xp):
    radius = max(1, int(math.ceil(4 * sigma)))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = xp.pad(grid, ((radius, radius), (0, 0)))
    out = sum(float(k) * padded[j:j + grid.shape[0]] for j, k in enumerate(kernel))
    padded = xp.pad(out, ((0, 0), (radius, radius)))
    return sum(float(k) * padded[:, j:j + grid.shape[1]] for j, k in enumerate(kernel))


class _Fitter:
    def __init__(self, image, centers, *, upsample, smooth, taper, neighbours, xp):
        from scipy.spatial import cKDTree

        self.xp = xp
        self.image = xp.asarray(image, dtype=xp.float64)
        self.h, self.w = self.image.shape
        distance, index = cKDTree(centers).query(centers, k=neighbours + 1)
        self.n = len(centers)
        self.neighbours = xp.asarray(index[:, 1:])
        # Window: three quarters of the 4th neighbour's distance covers the territory of square, rectangular and
        # staggered lattices.
        self.reach = int(math.ceil(0.75 * float(np.median(distance[:, min(4, neighbours)]))))
        self.upsample, self.smooth, self.taper = int(upsample), float(smooth), float(taper)
        r = self.reach
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
        self.ox, self.oy = xp.asarray(ox.ravel()), xp.asarray(oy.ravel())
        self.half = (r + 3) * self.upsample
        self.size = 2 * self.half + 1
        self.chunk = max(64, min(8000, int(3e6 // (2 * r + 1) ** 2)))

    def slices(self):
        return [slice(s, min(s + self.chunk, self.n)) for s in range(0, self.n, self.chunk)]

    def pixels(self, c, sl):
        """Window pixels of holes ``sl``: x, y, values, inside the image, territory weight."""
        xp = self.xp
        own = c[sl]
        px = xp.rint(own[:, 0]).astype(xp.int64)[:, None] + self.ox[None, :]
        py = xp.rint(own[:, 1]).astype(xp.int64)[:, None] + self.oy[None, :]
        inside = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        values = self.image[xp.clip(py, 0, self.h - 1), xp.clip(px, 0, self.w - 1)]
        mine = xp.hypot(px - own[:, 0:1], py - own[:, 1:2])
        others = c[self.neighbours[sl]]
        nearest = xp.hypot(px[:, :, None] - others[:, None, :, 0], py[:, :, None] - others[:, None, :, 1]).min(-1)
        t = xp.clip(0.5 + (nearest - mine) / (2 * self.taper), 0, 1)
        return px, py, values, inside, t * t * (3 - 2 * t) * inside

    def grid_coordinates(self, px, py, c, sl):
        """Template-grid coordinates of the pixels (px, py) of holes ``sl`` centered at ``c[sl]``."""
        return (px - c[sl, 0:1]) * self.upsample + self.half, (py - c[sl, 1:2]) * self.upsample + self.half

    def template(self, c, a, b, use):
        """Average hole of the holes ``use`` (scaled to their mean brightness and background) and its mean
        territory weight."""
        xp = self.xp
        size = self.size
        sums, counts, territory = (xp.zeros(size * size) for _ in range(3))
        mean_a, mean_b = float(a[use].mean()), float(b[use].mean())
        for sl in self.slices():
            px, py, values, inside, weight = self.pixels(c, sl)
            values = (values - b[sl, None]) / xp.where(use[sl], a[sl], 1.0)[:, None] * mean_a + mean_b
            sx, sy = self.grid_coordinates(px, py, c, sl)
            keep = inside & use[sl, None] & (sx >= 0) & (sx < size - 1) & (sy >= 0) & (sy < size - 1)
            sx, sy, values, weight = sx[keep], sy[keep], values[keep], weight[keep]
            x0, y0 = xp.floor(sx), xp.floor(sy)
            fx, fy = sx - x0, sy - y0
            x0, y0 = x0.astype(xp.int64), y0.astype(xp.int64)
            for dy, dx, f in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy),
                              (1, 1, fx * fy)):
                index = (y0 + dy) * size + x0 + dx
                sums += xp.bincount(index, weights=f * values, minlength=size * size)
                counts += xp.bincount(index, weights=f, minlength=size * size)
                territory += xp.bincount(index, weights=f * weight, minlength=size * size)
        sigma = self.smooth * self.upsample
        sums, counts, territory = (_smooth(g.reshape(size, size), sigma, xp) for g in (sums, counts, territory))
        filled = counts > 1e-6 * float(counts.max())
        grid = xp.where(filled, sums / xp.where(filled, counts, 1), xp.nan)
        grid = xp.where(filled, grid, float(xp.nanmedian(grid)))
        return grid, xp.where(filled, territory / xp.where(filled, counts, 1), 0)

    def centroid(self, grid, territory):
        """Light centroid (px) of the average hole inside its own territory, above the territory's lowest level."""
        xp = self.xp
        offsets = (xp.arange(self.size) - self.half) / self.upsample
        own = territory > 0.5
        light = xp.where(own, grid - float(grid[own].min()), 0)
        total = float(light.sum())
        return float((light.sum(0) * offsets).sum()) / total, float((light.sum(1) * offsets).sum()) / total

    def noise_model(self, c, a, b, grid, use):
        """Pixel noise variance = alpha + beta * level, fitted to the squared leftovers of the holes ``use``."""
        xp = self.xp
        sums = xp.zeros(5)
        for sl in self.slices():
            px, py, values, _, weight = self.pixels(c, sl)
            weight = weight * use[sl, None]
            sx, sy = self.grid_coordinates(px, py, c, sl)
            model = a[sl, None] * _sample(grid, sx, sy, xp)[0] + b[sl, None]
            r2 = (values - model) ** 2
            sums += xp.stack([weight.sum(), (weight * model).sum(), (weight * model ** 2).sum(),
                              (weight * r2).sum(), (weight * model * r2).sum()])
        s0, s1, s2, y0, y1 = (float(v) for v in sums)
        beta = (s0 * y1 - s1 * y0) / (s0 * s2 - s1 * s1)
        alpha = (y0 - beta * s1) / s0
        return max(alpha, 0.05 * y0 / s0), max(beta, 0.0)

    def fit(self, c, a, b, grid, *, iterations, noise, step_max=0.5):
        xp = self.xp
        c, a, b = c.copy(), a.copy(), b.copy()
        ok = xp.zeros(self.n, bool)
        for sl in self.slices():
            px, py, values, _, weight = self.pixels(c, sl)  # territory fixed while the hole moves
            cc, aa, bb = c[sl], a[sl], b[sl]
            full = weight.sum(1)
            for step_number in range(iterations):
                sx = (px - cc[:, 0:1]) * self.upsample + self.half
                sy = (py - cc[:, 1:2]) * self.upsample + self.half
                t, gx, gy = _sample(grid, sx, sy, xp)
                gx, gy = gx * self.upsample, gy * self.upsample
                if step_number == 0:  # brightness and background at the start: straight-line fit to T
                    s1, st, stt = weight.sum(1), (weight * t).sum(1), (weight * t * t).sum(1)
                    sv, stv = (weight * values).sum(1), (weight * t * values).sum(1)
                    det = s1 * stt - st * st
                    aa = xp.where(det > 0, (s1 * stv - st * sv) / xp.where(det > 0, det, 1), aa)
                    bb = xp.where(s1 > 0, (sv - aa * st) / xp.where(s1 > 0, s1, 1), bb)
                model = aa[:, None] * t + bb[:, None]
                w = weight if noise is None else weight / xp.maximum(noise[0] + noise[1] * model, 1e-6)
                jac = xp.stack((-aa[:, None] * gx, -aa[:, None] * gy, t, xp.ones_like(t)), axis=-1)
                jw = jac * w[..., None]
                normal = xp.einsum("kpi,kpj->kij", jw, jac) + 1e-9 * xp.eye(4)[None]
                step = xp.linalg.solve(normal, xp.einsum("kpi,kp->ki", jw, values - model)[..., None])[..., 0]
                move = xp.hypot(step[:, 0], step[:, 1])
                scale = xp.where(move > step_max, step_max / xp.maximum(move, 1e-12), 1.0)
                cc = cc + step[:, :2] * scale[:, None]
                aa, bb = aa + step[:, 2], bb + step[:, 3]
            good = (aa > 0) & xp.isfinite(cc).all(1) & xp.isfinite(aa) & xp.isfinite(bb)
            ok[sl] = good & (full > 0.5 * float(full.max()))
            # A failed hole keeps its start, so its neighbours' territories stay finite.
            c[sl] = xp.where(good[:, None], cc, c[sl])
            a[sl], b[sl] = xp.where(good, aa, a[sl]), xp.where(good, bb, b[sl])
        return c, a, b, ok


def template_fit_centers(image, centers_xy, *, rounds=3, iterations=6, upsample=4, smooth=0.7, taper=1.0,
                         neighbours=8, noise_weights=True, backend="cpu"):
    """Refined centers (N, 2) aligned with the start ``centers_xy`` (one per hole), NaN where the fit failed or the
    start was not finite. Holes whose fit failed in a round are left out of the next round's average hole."""
    if backend == "gpu":
        import cupy as xp
    else:
        xp = np
    start = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    result = np.full(start.shape, np.nan)
    finite = np.isfinite(start).all(axis=1)
    centers = start[finite]
    if len(centers) < neighbours + 2:
        return result
    fitter = _Fitter(np.asarray(image, dtype=np.float64), centers, upsample=upsample, smooth=smooth, taper=taper,
                     neighbours=neighbours, xp=xp)
    c = xp.asarray(centers)
    a, b = xp.ones(fitter.n), xp.zeros(fitter.n)
    ok = xp.ones(fitter.n, bool)
    for round_number in range(rounds):
        if int(ok.sum()) < neighbours + 2:
            return result
        grid, territory = fitter.template(c, a, b, ok)
        c = c + xp.asarray(fitter.centroid(grid, territory))  # the average hole's light centroid is the center
        grid, _ = fitter.template(c, a, b, ok)
        noise = fitter.noise_model(c, a, b, grid, ok) if noise_weights and round_number > 0 else None
        c, a, b, ok = fitter.fit(c, a, b, grid, iterations=iterations, noise=noise)
    out = c if xp is np else xp.asnumpy(c)
    good = ok if xp is np else xp.asnumpy(ok)
    result[finite] = np.where(good[:, None], out, np.nan)
    return result
