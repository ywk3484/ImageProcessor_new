"""Outline fit of a whole page: one sharp outline and one blur for the page, every hole at its own position,
brightness, local level and edge offset.

    page model  P(p) = sum_i a_i [ T + delta_i T_1 + delta_i^2 T_2 ](p - c_i)          (holes' light only)
    hole i      data(p) = P(p) + level_i + noise,   on its own window

T is the template of ``model.Template`` (outline x blur x pixel) on a fine grid, read by cubic interpolation;
T_1, T_2 = first and second edge-offset terms (model.Template.render_quad). Stages, alternated for ``rounds``:

1. holes: Gauss-Newton on (cx, cy, a, level[, delta]) of every hole against the full page model, one colour class
   of holes at a time (block Gauss-Seidel), pixels weighted by 1 / noise variance (a straight line in the level,
   fitted from the leftovers) times a robust weight against light the model does not contain;
2. shape: the neighbour-cleaned stacked average ((data - P + own) / a at each pixel's offset from its hole, on the
   fine grid) fitted by least squares with the template (box-blurred by one fine bin) over the outline and blur
   parameters, weights = the bins' summed weights.

One code path for NumPy and CuPy (``xp``); the shape stage runs on the host (SciPy least squares).

Library copy of the validated prototype (artifacts/outline_fit/fit.py, 2026-11-05). Changes, none of which moves
the fit: the stopping rule's 0.08 / 0.3 nm are given in pixels (their values at 21 nm pixels), so a page measured
in px stops where the prototype stopped; ``run`` reads the template after a box doubling (the prototype kept the
first box for its log and the returned ``template_box``); ``converged`` records whether the stopping rule was met;
the per-hole CD is the exact area of the radially offset outline for every page size.
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np

from .model import Template, fit_stages


def host(x):
    return x.get() if hasattr(x, "get") else np.asarray(x)


# ---------------------------------------------------------------------------------------- cubic interpolation
def _keys_weights(t, xp):
    """Keys cubic (a = -0.5) weights for taps at -1, 0, 1, 2; ``t`` in [0, 1)."""
    x = xp.stack((1 + t, t, 1 - t, 2 - t), axis=-1)
    near = x <= 1
    return xp.where(near, (1.5 * x - 2.5) * x * x + 1, ((-0.5 * x + 2.5) * x - 4) * x + 2)


def cubic_sample(grids, sx, sy, xp):
    """Values of each fine grid in ``grids`` (list of (N, N)) at fine-grid coordinates (sx, sy), shape of sx.
    Points outside the grid read 0."""
    n = grids[0].shape[0]
    ix, iy = xp.floor(sx), xp.floor(sy)
    wx = _keys_weights(sx - ix, xp)
    wy = _keys_weights(sy - iy, xp)
    taps = xp.arange(-1, 3)
    cx = ix.astype(xp.int64)[..., None] + taps
    cy = iy.astype(xp.int64)[..., None] + taps
    inside = (cx >= 0) & (cx < n)
    insidey = (cy >= 0) & (cy < n)
    cx = xp.clip(cx, 0, n - 1)
    cy = xp.clip(cy, 0, n - 1)
    wx = wx * inside
    wy = wy * insidey
    out = []
    for g in grids:
        # one tap pair at a time: the (..., 4, 4) gather of all taps at once was the hole stage's memory peak
        acc = xp.zeros(sx.shape)
        for k in range(4):
            row = xp.zeros(sx.shape)
            for l in range(4):
                row += wx[..., l] * g[cy[..., k], cx[..., l]]
            acc += wy[..., k] * row
        out.append(acc)
    return out


# ------------------------------------------------------------------------------------------------ page fit
class PageFit:
    def __init__(self, image, centers_xy, *, pixel_nm=21.0, unit="nm", template=None, window_px=None, xp=None, chunk=2000,
                 start_half_width_px=None, start_blur_sigma_px=1.0, gradient=False, reach_px=None,
                 far_field=True, far_r0_px=None, far_ramp_px=4.0, far_box_px=128, far_upsample=2,
                 background="level", grid_px=None):
        self.xp = xp if xp is not None else np
        xp = self.xp
        # reach_px: how far each hole's light is rendered into the page model (default: the fit window + 3 px).
        # The lens blur's tail is below 1e-3 of the peak beyond 9 px, but summed over the thirty holes within
        # 30 px it is 1 % of the peak: on a noise-free page with the exact template, the model at the truth leaves
        # 0.08 rms of structure when rendered to 9 px and 0.02 when rendered to 40 (runs/noisefree_check.log).
        # Rendering to 40 px moved the hole stage's edge offsets from 1.030 x true - 0.87 nm to 1.025 x - 0.76 nm
        # and left the light move (+0.11 nm per 10 %) as it was: a small part of those (runs/noisefree_check_reach40.log)
        self.reach_px = None if reach_px is None else int(reach_px)
        # far_field: each hole's light is split at the template's reach r0 (where it falls below 1e-3 of the peak,
        # at least the fit window): the near part, T w(r) with w = 1 inside r0 and a cosine ramp to 0 over
        # far_ramp_px, is rendered per hole as before; the far part, T (1 - w), by one FFT convolution per sweep of
        # the holes' brightness (and brightness x offset, x offset^2) splatted on the pixel grid with cubic weights,
        # with the far kernel sampled at whole-pixel offsets from a far_box_px template (upsample far_upsample: the
        # tail is smooth). Near + far = the whole template to the splat's interpolation error of the smooth far part
        # (0.001 grey levels rms against a direct render, runs/far_check.log). Without it the levels absorb the far
        # light (+0.8 grey levels) and the exact noise-free page reads edge offsets 1.017 x true - 0.3 nm
        # (runs/nf_K_own_R6_nofar.log) against 0.999 x - 0.01 nm with it: on by default.
        self.far = bool(far_field)
        self.far_r0_px = None if far_r0_px is None else float(far_r0_px)  # default: the template's 1e-3 reach
        self.far_ramp = float(far_ramp_px)
        self.far_box = int(far_box_px)
        self.far_upsample = int(far_upsample)
        self._far_template = None
        self._far_fft = None
        self._far_margin = None
        self._t0_full = None
        # gradient: each hole's local background is a plane (level + gx dx + gy dy over its window) instead of a
        # constant, to absorb smooth light the page model does not carry (far tails of the neighbours' blur, flare):
        # on noise-free test pages that light, 0.7 % of the peak, moved the edge offsets by 1 nm and gave the slope
        # and brightness effects (runs/tiny_mismatch*.log)
        self.gradient = bool(gradient)
        # background: "level" - a free constant per hole over its own window (not part of the page model);
        # "grid" - one smooth background for the page, bilinear on a grid of grid_px cells (default three hole
        # spacings), solved in alternation with the holes (a hole's level is then read off the grid). On the exact
        # noise-free test page (runs/nf_*_own_*.log, 2026-11-05) both reach the truth once the hole stage fits against
        # the complete page model (see fit_holes): level per hole 1.004 x true - 0.04 nm after two sweeps; grid
        # 0.999 x - 0.07 nm after eight (the common shift of size, brightness and background is solved in alternation
        # and settles slowly). The grid is kept as an option for pages whose background is not flat.
        if background not in ("level", "grid"):
            raise ValueError("background must be 'level' or 'grid'")
        if background == "grid" and self.gradient:
            raise ValueError("gradient=True needs background='level'")
        self.background = background
        self.grid_px = grid_px
        self.grid = None
        self.image = xp.asarray(np.asarray(image, dtype=np.float64))
        self.h, self.w = self.image.shape
        self.pixel_nm = float(pixel_nm)  # length per pixel, for the log and the readings only
        self.unit = str(unit)
        # a copy: the fit moves the centres in place (the prototype moved the caller's own seed array)
        c = np.array(centers_xy, dtype=np.float64, copy=True).reshape(-1, 2)
        self.n = len(c)
        self.template = template if template is not None else Template()
        self.chunk = int(chunk)
        # spacing and windows
        from scipy.spatial import cKDTree

        d, _ = cKDTree(c).query(c, k=2)
        self.spacing = float(np.median(d[:, 1]))
        self.R = int(window_px) if window_px else max(3, int(math.ceil(0.75 * self.spacing)))
        if self.background == "grid":
            cell = int(grid_px) if grid_px else max(12, int(round(3.0 * self.spacing)))
            self.grid_cell = cell
            self.grid_shape = (int(math.ceil((self.h - 1) / cell)) + 1, int(math.ceil((self.w - 1) / cell)) + 1)
            self.grid = xp.zeros(self.grid_shape)
        # colour classes for the hole stage (block Gauss-Seidel): chunks of holes of one colour
        self.colour = self._colour(c)
        self._colour_chunks = []
        for k in range(int(self.colour.max()) + 1):
            idx = np.flatnonzero(self.colour == k)
            for s in range(0, len(idx), self.chunk):
                self._colour_chunks.append(xp.asarray(idx[s:s + self.chunk]))
        self.c = xp.asarray(c)
        if self.far:
            # padded FFT grid: half the kernel box, plus room for seeds outside the frame (a hole's kernel must not
            # wrap round the padded grid), plus the cubic taps and a little drift
            B = self.far_box
            out_y = max(0.0, -c[:, 1].min(), c[:, 1].max() - (self.h - 1))
            out_x = max(0.0, -c[:, 0].min(), c[:, 0].max() - (self.w - 1))
            self._far_margin = (B // 2 + int(math.ceil(out_y)) + 4, B // 2 + int(math.ceil(out_x)) + 4)
        self.a = xp.zeros(self.n)
        self.level = xp.zeros(self.n)
        self.gx = xp.zeros(self.n)  # local background slope per px, x and y (zero unless gradient=True)
        self.gy = xp.zeros(self.n)
        self.delta = xp.zeros(self.n)
        self.frozen = xp.zeros(self.n, dtype=bool)
        self.delta_sd = xp.full(self.n, np.nan)  # per-hole standard error of the edge offset (px)
        self.size_on = True  # per-hole edge offsets fitted (False: the page's outline is every hole's size)
        self.clip_px = None
        self.var_line = (1.0, 0.0)  # noise variance = v0 + v1 * level
        self.rms_typ = None  # a typical hole's leftover rms with the current template (robust weights' scale)
        self.theta = None
        self.start_half = start_half_width_px
        self.start_sigma = float(start_blur_sigma_px)
        self.log = []
        self.timings = {}
        self.history = []
        self.converged = False
        self._templates = None

    # ------------------------------------------------------------------------------ windows and sampling
    def _colour(self, c):
        """Greedy colouring: holes within 1.5 spacings of each other get different colours (a square lattice takes
        four or five, a staggered one three or four), so that the hole stage can update one colour at a time with
        the neighbours fixed and refreshed."""
        from scipy.spatial import cKDTree

        nbrs = cKDTree(c).query_ball_point(c, 1.5 * self.spacing)
        colour = np.full(len(c), -1, dtype=np.int64)
        for i, nb in enumerate(nbrs):
            used = set(colour[nb].tolist())
            k = 0
            while k in used:
                k += 1
            colour[i] = k
        return colour

    def _window_offsets(self, R):
        xp = self.xp
        oy, ox = np.mgrid[-R:R + 1, -R:R + 1]
        return xp.asarray(ox.ravel()), xp.asarray(oy.ravel())

    def _pixels(self, sl, R):
        """Flat page indices (clamped), validity, and pixel coordinates of the windows of holes ``sl``."""
        xp = self.xp
        ox, oy = self._window_offsets(R)
        c = self.c[sl]
        r = xp.rint(c)
        px = r[:, 0, None].astype(xp.int64) + ox[None, :]
        py = r[:, 1, None].astype(xp.int64) + oy[None, :]
        valid = (px >= 0) & (px < self.w) & (py >= 0) & (py < self.h)
        flat = xp.clip(py, 0, self.h - 1) * self.w + xp.clip(px, 0, self.w - 1)
        return flat, valid, px, py

    def _background(self, sl, px, py):
        """The background under the pixels (px, py) of holes ``sl``: the page grid, or each hole's level plus the
        plane gx dx + gy dy when gradient=True (dx, dy from the window's centre pixel)."""
        if self.background == "grid":
            return self.background_at(px, py)
        back = self.level[sl, None]
        if self.gradient:
            c = self.xp.rint(self.c[sl])
            back = back + self.gx[sl, None] * (px - c[:, 0, None]) + self.gy[sl, None] * (py - c[:, 1, None])
        return back

    # ------------------------------------------------------------------------------ page background grid
    def _grid_basis(self, px, py):
        """Cell corner (j, i), and the four bilinear weights (ll, lr, ul, ur), of pixel coordinates (any shape)."""
        xp = self.xp
        ny, nx = self.grid_shape
        u = xp.clip(px, 0, self.w - 1) / self.grid_cell
        v = xp.clip(py, 0, self.h - 1) / self.grid_cell
        i = xp.clip(xp.floor(u), 0, nx - 2).astype(xp.int64)
        j = xp.clip(xp.floor(v), 0, ny - 2).astype(xp.int64)
        fu, fv = u - i, v - j
        return j, i, ((1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv)

    def background_at(self, px, py):
        j, i, (w00, w01, w10, w11) = self._grid_basis(px, py)
        g = self.grid
        return w00 * g[j, i] + w01 * g[j, i + 1] + w10 * g[j + 1, i] + w11 * g[j + 1, i + 1]

    def fit_background(self, light, robust_c=3.0):
        """Weighted least squares of the grid on image - light (the holes' light, near and far), pixels weighted
        by 1 / variance times a Cauchy weight against light the model does not contain."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve

        xp = self.xp
        ny, nx = self.grid_shape
        corners = ((0, 0), (0, 1), (1, 0), (1, 1))
        nk = ny * nx
        ncell = (ny - 1) * (nx - 1)
        cj, ci = np.divmod(np.arange(ncell), nx - 1)
        knot_of = [(cj + dj) * nx + (ci + di) for dj, di in corners]  # the four knots of every cell
        rhs = np.zeros(nk)
        sums = np.zeros((4, 4, ncell))  # per cell: sum over its pixels of w phi_a phi_b
        band = max(1, int(2e6 // self.w))  # rows per pass: the per-pixel arrays stay around 16 MB each
        for y0 in range(0, self.h, band):
            y1 = min(y0 + band, self.h)
            yy, xx = xp.meshgrid(xp.arange(y0, y1), xp.arange(self.w), indexing="ij")
            j, i, basis = self._grid_basis(xx.ravel(), yy.ravel())
            lt = light[y0:y1].ravel()
            r = (self.image[y0:y1] - light[y0:y1]).ravel()
            wgt = self._weights(lt)
            if robust_c and self.rms_typ:
                wgt = wgt / (1.0 + (r / (robust_c * self.rms_typ)) ** 2)
            cell = j * (nx - 1) + i
            for a, ba in enumerate(basis):
                ka = (j + corners[a][0]) * nx + (i + corners[a][1])
                rhs += host(xp.bincount(ka, weights=wgt * ba * r, minlength=nk))
                for b, bb in enumerate(basis):
                    if b < a:
                        continue
                    sums[a, b] += host(xp.bincount(cell, weights=wgt * ba * bb, minlength=ncell))
        rows, cols, vals = [], [], []
        for a in range(4):
            for b in range(4):
                s = sums[a, b] if b >= a else sums[b, a]
                rows.append(knot_of[a])
                cols.append(knot_of[b])
                vals.append(s)
        normal = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nk, nk)).tocsr()
        diag = normal.diagonal()
        ridge = 1e-8 * float(diag.mean()) + 1e-12
        normal = normal + coo_matrix((np.full(nk, ridge), (np.arange(nk), np.arange(nk))), shape=(nk, nk)).tocsr()
        g = spsolve(normal, rhs)
        self.grid = xp.asarray(np.asarray(g, dtype=np.float64).reshape(ny, nx))
        return self.grid

    def _fine(self, px, py, c):
        """Fine-grid coordinates of pixels (px, py) for hole centres c (n, 2)."""
        t = self.template
        sx = (px - c[:, 0, None]) * t.U + t.N // 2
        sy = (py - c[:, 1, None]) * t.U + t.N // 2
        return sx, sy

    def set_theta(self, theta):
        self.theta = np.asarray(theta, float)
        xp = self.xp
        t0, td, t2 = self.template.render_quad(self.theta)
        tx, ty = self.template.derivatives_xy(t0)
        self._templates = [xp.asarray(g) for g in (t0, td, tx, ty, t2)]
        self.rms_typ = None  # the leftovers change with the template
        # reach of the template: where it falls below 1e-3 of its peak
        prof = np.abs(t0[t0.shape[0] // 2])
        above = np.flatnonzero(prof > 1e-3 * prof.max())
        reach_px = (above[-1] - above[0] + 1) / (2 * self.template.U)
        cap = self.template.N // (2 * self.template.U) - 2
        if reach_px > cap and self.template.N // self.template.U < 96:
            # The hole's light reaches past the template's box: the stack and every hole's rendered light are cut
            # there (the 300 nm squares at 24 px read blur 89 nm for 65 and their centres 1.5 x the bound, with a
            # 0.22 nm pixel-position error; runs/centers_square300_square_t24.json). Double the box and render again.
            t = self.template
            self.template = Template(upsample=t.U, size_px=2 * (t.N // t.U), harmonics=t.ks, blur=t.blur, aa=t.aa)
            self.log.append(f"  template box {t.N // t.U} -> {self.template.N // self.template.U} px: the hole's light "
                            f"reaches {reach_px:.1f} px, past the {cap} px the box can render")
            return self.set_theta(theta)
        # the page model renders each hole out to its reach, capped at the fit window + 3 px: a long tail (the lens
        # family's rings) beyond that only adds a near-constant that the holes' levels absorb, at 2-3x the cost
        self.Rr = min(int(math.ceil(reach_px)) + 1, self.template.N // (2 * self.template.U) - 2, self.R + 3)
        if self.reach_px is not None:
            self.Rr = min(self.reach_px, self.template.N // (2 * self.template.U) - 2)
        self._t0_full = self._templates[0]
        if self.far:
            # the near zone is what was rendered per hole before (the fit window + 3 px, capped); the far field adds
            # the rest. On the small test page near + far agrees with a direct 46 px render to 0.001 grey levels rms
            # (2e-5 of the peak; far_check.py), the far field costing 0.35 s per call on a 1378 x 2048 page
            r0 = float(self.Rr) if self.far_r0_px is None else max(self.far_r0_px, float(self.R))
            self._set_far(t0, td, t2, r0)

    # -------------------------------------------------------------------------------------- far field
    def _ramp(self, r, r0):
        """Near-field weight: 1 inside r0, cosine ramp to 0 at r0 + far_ramp."""
        s = np.clip((r - r0) / self.far_ramp, 0.0, 1.0)
        return 0.5 * (1 + np.cos(np.pi * s))

    def _set_far(self, t0, td, t2, r0):
        """Split the templates at r0: near parts (per-hole rendering, with their x/y derivatives) and the far
        kernels' FFTs on the padded page grid."""
        xp = self.xp
        t = self.template
        self.far_r0 = float(r0)
        w = self._ramp(np.hypot(t.xx, t.yy), r0)
        t0n, tdn, t2n = t0 * w, td * w, t2 * w
        tx, ty = t.derivatives_xy(t0n)
        self._templates = [xp.asarray(g) for g in (t0n, tdn, tx, ty, t2n)]
        self.Rr = min(int(math.ceil(r0 + self.far_ramp)) + 1, t.N // (2 * t.U) - 2)
        # the far kernels at whole-pixel offsets, from a wide template of the same model
        if self._far_template is None:
            self._far_template = Template(upsample=self.far_upsample, size_px=self.far_box, harmonics=t.ks,
                                          blur=t.blur, aa=t.aa)
        ft = self._far_template
        b0, bd, b2 = ft.render_quad(self.theta)
        B, U = self.far_box, ft.U
        k = np.arange(B) * U  # fine index of whole-pixel offset (i - B/2)
        rr = np.hypot(*np.meshgrid(np.arange(B) - B // 2, np.arange(B) - B // 2))
        wf = 1.0 - self._ramp(rr, r0)
        my, mx = self._far_margin
        hp, wp = self.h + 2 * my, self.w + 2 * mx
        ffts = []
        for g in (b0, bd, b2):
            pad = np.zeros((hp, wp))
            pad[:B, :B] = g[np.ix_(k, k)] * wf
            ffts.append(xp.asarray(np.fft.rfft2(pad)))
        self._far_fft = ffts

    def far_field(self):
        """The holes' far light on the page: cubic splat of a, a delta, a delta^2 convolved with the far kernels."""
        xp = self.xp
        B = self.far_box
        my, mx = self._far_margin
        hp, wp = self.h + 2 * my, self.w + 2 * mx
        c = self.c + xp.asarray([mx, my], dtype=xp.float64)
        ix, iy = xp.floor(c[:, 0]), xp.floor(c[:, 1])
        wx = _keys_weights(c[:, 0] - ix, xp)
        wy = _keys_weights(c[:, 1] - iy, xp)
        taps = xp.arange(-1, 3)
        cx = ix.astype(xp.int64)[:, None] + taps
        cy = iy.astype(xp.int64)[:, None] + taps
        ok = (cx[:, 0] >= 0) & (cx[:, -1] < wp) & (cy[:, 0] >= 0) & (cy[:, -1] < hp)
        cx, cy = xp.clip(cx, 0, wp - 1), xp.clip(cy, 0, hp - 1)
        flat = (cy[:, :, None] * wp + cx[:, None, :]).reshape(len(c), -1)
        wgt = (wy[:, :, None] * wx[:, None, :]).reshape(len(c), -1) * ok[:, None]
        spec = None
        de = self.delta
        for g, val in zip(self._far_fft, (self.a, self.a * de, self.a * de * de)):
            s = xp.bincount(flat.ravel(), weights=(wgt * val[:, None]).ravel(), minlength=hp * wp).reshape(hp, wp)
            term = xp.fft.rfft2(s) * g
            spec = term if spec is None else spec + term
        far = xp.fft.irfft2(spec, s=(hp, wp))  # hole at padded n, kernel centre at B/2: page pixel p sits at p + m + B/2
        oy, ox = my + B // 2, mx + B // 2
        return far[oy:oy + self.h, ox:ox + self.w]

    # -------------------------------------------------------------------------------------- page model
    def render_page(self):
        xp = self.xp
        page = xp.zeros(self.h * self.w)
        t0, td, t2 = self._templates[0], self._templates[1], self._templates[4]
        for sl in self._slices():
            flat, valid, px, py = self._pixels(sl, self.Rr)
            sx, sy = self._fine(px, py, self.c[sl])
            v0, vd, v2 = cubic_sample([t0, td, t2], sx, sy, xp)
            de = self.delta[sl, None]
            light = self.a[sl, None] * (v0 + de * (vd + de * v2))
            light = xp.where(valid, light, 0.0)
            page += xp.bincount(flat.ravel(), weights=light.ravel(), minlength=self.h * self.w)
        page = page.reshape(self.h, self.w)
        if self.far:
            page = page + self.far_field()
        return page

    def _slices(self):
        return [slice(s, min(s + self.chunk, self.n)) for s in range(0, self.n, self.chunk)]

    # ---------------------------------------------------------------------------------- initial values
    def initialise(self):
        """Template from a round guess; a and level from a linear fit of each hole's window (two passes, the
        second against the page model of the others)."""
        xp = self.xp
        if self.theta is None:
            if self.start_half is None:
                # half width at half height of the plain stacked average along x
                stack = self._plain_stack()
                mid = stack.shape[0] // 2
                prof = stack[mid]
                # an empty middle row (few holes: no pixel lands on it) gives NaN and the floor below; the shape
                # stage's grid of starts takes it from there
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    base = np.nanmin(prof)
                half = base + 0.5 * (prof[mid] - base)
                # walk outward from the middle while the profile stays above half height
                left = right = mid
                while left > 0 and np.isfinite(prof[left - 1]) and prof[left - 1] >= half:
                    left -= 1
                while right + 1 < len(prof) and np.isfinite(prof[right + 1]) and prof[right + 1] >= half:
                    right += 1
                fw = (right - left + 1) / self.template.U
                # the hole's own width: blob fwhm^2 ~ blur fwhm^2 + (outline / 1.1)^2, rough
                half_width = max(0.5 * math.sqrt(max(fw ** 2 - (2.355 * self.start_sigma) ** 2, 0.25)), 0.6)
            else:
                half_width = float(self.start_half)
            self.set_theta(self.template.start(half_width, self.start_sigma))
        t0 = self._templates[0]
        page = xp.zeros((self.h, self.w))
        for _ in range(2):
            new_a = xp.zeros(self.n)
            new_l = xp.zeros(self.n)
            for sl in self._slices():
                flat, valid, px, py = self._pixels(sl, self.R)
                sx, sy = self._fine(px, py, self.c[sl])
                (v0,) = cubic_sample([t0], sx, sy, xp)
                data = self.image.ravel()[flat]
                others = page.ravel()[flat] - self.a[sl, None] * v0
                y = xp.where(valid, data - others, 0.0)
                wv = valid.astype(xp.float64)
                # 2 x 2 linear least squares per hole: y = a v0 + l
                s11 = (wv * v0 * v0).sum(1)
                s12 = (wv * v0).sum(1)
                s22 = wv.sum(1)
                r1 = (wv * v0 * y).sum(1)
                r2 = (wv * y).sum(1)
                det = s11 * s22 - s12 * s12
                new_a[sl] = (r1 * s22 - r2 * s12) / det
                new_l[sl] = (s11 * r2 - s12 * r1) / det
            self.a = xp.maximum(new_a, 1e-3)
            self.level = new_l
            page = self.render_page()
        if self.background == "grid":
            # the page background from the leftovers of that model, then every hole's brightness alone against it
            self.fit_background(page, robust_c=0.0)
            self.level = xp.zeros(self.n)
            new_a = xp.zeros(self.n)
            for sl in self._slices():
                flat, valid, px, py = self._pixels(sl, self.R)
                sx, sy = self._fine(px, py, self.c[sl])
                (v0,) = cubic_sample([t0], sx, sy, xp)
                others = page.ravel()[flat] - self.a[sl, None] * v0 + self._background(sl, px, py)
                y = xp.where(valid, self.image.ravel()[flat] - others, 0.0)
                wv = valid.astype(xp.float64)
                new_a[sl] = (wv * v0 * y).sum(1) / xp.maximum((wv * v0 * v0).sum(1), 1e-12)
            self.a = xp.maximum(new_a, 1e-3)

    def _plain_stack(self):
        xp = self.xp
        t = self.template
        sums = xp.zeros(t.N * t.N)
        cnt = xp.zeros(t.N * t.N)
        for sl in self._slices():
            flat, valid, px, py = self._pixels(sl, self.R)
            sx, sy = self._fine(px, py, self.c[sl])
            bx = xp.rint(sx).astype(xp.int64)
            by = xp.rint(sy).astype(xp.int64)
            ok = valid & (bx >= 0) & (bx < t.N) & (by >= 0) & (by < t.N)
            b = (by * t.N + bx)[ok]
            sums += xp.bincount(b, weights=self.image.ravel()[flat][ok], minlength=t.N * t.N)
            cnt += xp.bincount(b, minlength=t.N * t.N)
        with np.errstate(invalid="ignore", divide="ignore"):
            stack = host(sums) / host(cnt)
        return stack.reshape(t.N, t.N)

    # -------------------------------------------------------------------------------- noise variance
    def fit_noise(self, page):
        """Variance of the leftovers against the model level, as a straight line."""
        xp = self.xp
        res_all, lev_all = [], []
        for sl in self._slices()[:3]:
            flat, valid, px, py = self._pixels(sl, self.R)
            model = page.ravel()[flat] + self._background(sl, px, py)
            res = self.image.ravel()[flat] - model
            res_all.append(host(res[valid]))
            lev_all.append(host(model[valid]))
        res = np.concatenate(res_all)
        lev = np.concatenate(lev_all)
        edges = np.percentile(lev, np.linspace(0, 100, 13))
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (lev >= lo) & (lev < hi)
            if s.sum() > 200:
                r = res[s]
                mad = 1.4826 * np.median(np.abs(r - np.median(r)))
                xs.append(lev[s].mean())
                ys.append(mad ** 2)
        xs, ys = np.asarray(xs), np.asarray(ys)
        if len(xs) >= 3:
            v1, v0 = np.polyfit(xs, ys, 1)
            v1 = max(v1, 0.0)
            v0 = max(v0, 0.05 * ys.mean())
        else:
            v0, v1 = max(float(ys.mean()), 1e-3), 0.0
        self.var_line = (float(v0), float(v1))

    def _weights(self, model):
        v0, v1 = self.var_line
        return 1.0 / (v0 + v1 * self.xp.maximum(model, 0.0))

    # ---------------------------------------------------------------------------------- hole stage
    def _light(self, idx):
        """Flat page indices of the render windows of holes ``idx`` and their current light (0 off the page)."""
        xp = self.xp
        t0, td, t2 = self._templates[0], self._templates[1], self._templates[4]
        flat, valid, px, py = self._pixels(idx, self.Rr)
        sx, sy = self._fine(px, py, self.c[idx])
        v0, vd, v2 = cubic_sample([t0, td, t2], sx, sy, xp)
        de = self.delta[idx, None]
        light = self.a[idx, None] * (v0 + de * (vd + de * v2))
        return flat, xp.where(valid, light, 0.0)

    def delta_information(self):
        """Per-hole standard error of the edge offset from the normal matrix at the current state (no step), and
        the outline's radius: whether a hole's own size can be read at all. Median error above ``0.3`` radii means
        the quadratic edge-offset model cannot hold the noise (60 nm holes at 21 nm pixels: about 60 radii) and the
        per-hole size is switched off."""
        xp = self.xp
        t0, td, tx, ty, t2 = self._templates
        page = self.render_page()
        flat_page = page.ravel()
        for idx in self._colour_chunks:
            flat, valid, px, py = self._pixels(idx, self.R)
            sx, sy = self._fine(px, py, self.c[idx])
            v0, vd, vx, vy, v2 = cubic_sample([t0, td, tx, ty, t2], sx, sy, xp)
            a = self.a[idx, None]
            de = self.delta[idx, None]
            own = v0 + de * (vd + de * v2)
            model = flat_page[flat] + self._background(idx, px, py)
            wgt = xp.where(valid, self._weights(model), 0.0)
            cols = [-a * vx, -a * vy, own, xp.ones_like(v0), a * (vd + 2 * de * v2)]
            if self.gradient:
                r = xp.rint(self.c[idx])
                cols += [(px - r[:, 0, None]).astype(xp.float64), (py - r[:, 1, None]).astype(xp.float64)]
            J = xp.stack(cols, axis=-1)
            k = J.shape[-1]
            normal = xp.einsum("npi,npj->nij", J * wgt[..., None], J)
            normal += 1e-9 * xp.eye(k)[None] * xp.einsum("nii->n", normal)[:, None, None] / k
            try:
                cov = xp.linalg.inv(normal)
                self.delta_sd[idx] = xp.sqrt(xp.maximum(cov[:, 4, 4], 0.0))
            except Exception:
                self.delta_sd[idx] = np.nan
        radius = 0.5 * self.template.cd_px(self.theta)
        return float(np.nanmedian(host(self.delta_sd))), float(radius)

    def fit_holes(self, iterations=5, size=True, max_move=0.25, max_delta=None, stop_px=0.002, robust_c=3.0,
                  free_level=True, frame_size=True):
        """Gauss-Newton on (x, y, a, level[, delta]) of every hole, one colour class at a time with the page model
        refreshed in between (block Gauss-Seidel, which converges for any coupling of a least-squares problem).
        Updating all holes at once from one page model (Jacobi) ran away wherever a hole's size and brightness are
        soft against each other and the neighbours overlap: dense 60 nm page, exact template, interior CD error
        -7 nm after two passes and -10 nm after six, 1,600 holes at the offset clip; half steps only slowed it.

        Pixels get robust (Cauchy) weights on top of the noise weights: a leftover of ``robust_c`` typical hole rms
        counts half, larger ones less. Without them, light the model does not contain (a hole cut by the frame, a
        defect, a missing seed) made the hole beside it run off along its soft size-brightness direction (edge
        offset at the clip, brightness 1.5 x) and its wrong light did the same to the next hole: -23, -9, -3 nm in
        the first rings inside the frame of the small test page; with them -4, -2.4, -2.0 against -1.65 inside.

        ``free_level`` False keeps every hole's level as it is (its column is dropped from the solve): for tests
        with the level known, and the far field on so that the level has nothing left to absorb. ``frame_size``
        False keeps the page's size for the holes whose window reaches off the frame (their edge offset is then a
        frozen wrong boundary whose error the neighbours' windows carry inwards: on the exact noise-free page edge
        offsets 1.004 x true with 0.2 nm of per-hole scatter against 0.999 x and 0.003 nm with them fitted,
        runs/nf_H_own_R6.log and nf_H2_own_R6_frame.log; the clip bounds a hole cut by the frame).

        The edge offset is clipped at ``max_delta`` px, by default 0.3 of the outline's radius (the quadratic
        edge-offset model's range; a fixed 1 px was most of a 60 nm hole's radius). Stops early once 99 % of the
        holes move less than ``stop_px`` in position and in edge offset (``stop_px`` 0 runs all ``iterations``)."""
        xp = self.xp
        t0, td, tx, ty, t2 = self._templates
        if max_delta is None:
            max_delta = 0.3 * 0.5 * self.template.cd_px(self.theta)
        self.clip_px = float(max_delta)
        npx = self.h * self.w
        # the page model as the sum of every chunk's rendered light, kept per chunk so that a chunk's light is taken
        # out and put back exactly: one rendering per hole and sweep, as when all holes are updated at once
        cache = []
        flat_page = xp.zeros(npx)
        for idx in self._colour_chunks:
            flat_r, light = self._light(idx)
            cache.append(light)
            flat_page += xp.bincount(flat_r.ravel(), weights=light.ravel(), minlength=npx)
        far = self.far_field().ravel() if self.far else None  # the holes' far light, refreshed once per sweep
        if far is not None:
            flat_page += far
        if self.background == "grid":
            free_level = False  # the page background takes the level's place (refreshed once per sweep)
        if self.history == []:
            self.fit_noise(flat_page.reshape(self.h, self.w))
        # a hole whose window reaches off the frame has part of itself missing: with frame_size False it keeps the
        # page's size (positions, brightness and level are still fitted, so its light is in the model); brightness is
        # bounded (at most a factor 2 per sweep and 5 x the page's median: an unbounded fit of a hole cut by the frame
        # once reached 319 x)
        c_host = host(self.c)
        inside = ((c_host[:, 0] >= self.R + 1) & (c_host[:, 0] <= self.w - 2 - self.R)
                  & (c_host[:, 1] >= self.R + 1) & (c_host[:, 1] <= self.h - 2 - self.R))
        if frame_size:
            inside = np.ones(self.n, dtype=bool)
        self.inside = xp.asarray(inside)
        a_host = host(self.a)
        a_med = float(np.median(a_host[inside])) if inside.any() else float(np.median(a_host))
        for it in range(iterations):
            stat, dstat, rms_all = [], [], []
            for j, idx in enumerate(self._colour_chunks):
                flat_r_old = self._pixels(idx, self.Rr)[0]  # where the chunk's cached light sits (centres before the move)
                flat, valid, px, py = self._pixels(idx, self.R)
                c = self.c[idx]
                sx, sy = self._fine(px, py, c)
                v0, vd, vx, vy, v2 = cubic_sample([t0, td, tx, ty, t2], sx, sy, xp)
                a = self.a[idx, None]
                de = self.delta[idx, None]
                own = v0 + de * (vd + de * v2)
                # the page model already holds this hole's light at its current parameters (its cached render is the
                # same a * own), so the window's residual is the complete model's. Until 2026-11-05 the whole colour's
                # light was taken out here and only the hole's own put back: the same-coloured neighbours' light (1.5
                # spacings away: their tails, and with wide windows their cores) was missing from the model inside the
                # window. On the exact noise-free page that fixed point read edge offsets 1.03 x true - 0.97 nm and
                # +0.13 nm per 10 % brightness (runs/noisefree_far_patch512.log); against the complete model 1.004 x
                # - 0.04 nm and +0.03 (runs/nf_H_own_R6.log).
                model = flat_page[flat] + self._background(idx, px, py)
                resid = xp.where(valid, self.image.ravel()[flat] - model, 0.0)
                wgt = xp.where(valid, self._weights(model), 0.0)
                rms_sl = xp.sqrt((resid * resid).sum(1) / xp.maximum(valid.sum(1), 1))
                rms_all.append(host(rms_sl))
                if robust_c:
                    s = self.rms_typ if self.rms_typ is not None else float(np.median(rms_all[-1]))
                    wgt = wgt / (1.0 + (resid / (robust_c * max(s, 1e-6))) ** 2)
                # model of the hole: a (T0 + delta T1 + delta^2 T2) + level [+ gx dx + gy dy];
                # columns = d/dx, d/dy, d/da, d/dlevel [, d/ddelta] [, d/dgx, d/dgy]
                cols = [-a * vx, -a * vy, own, xp.ones_like(v0) if free_level else xp.zeros_like(v0)]
                if size:
                    cols.append(a * (vd + 2 * de * v2))
                if self.gradient:
                    r = xp.rint(c)
                    cols += [(px - r[:, 0, None]).astype(xp.float64), (py - r[:, 1, None]).astype(xp.float64)]
                J = xp.stack(cols, axis=-1)  # (n, pixels, params)
                JW = J * wgt[..., None]
                normal = xp.einsum("npi,npj->nij", JW, J)
                rhs = xp.einsum("npi,np->ni", JW, resid)
                k = normal.shape[-1]
                normal += 1e-9 * xp.eye(k)[None] * xp.einsum("nii->n", normal)[:, None, None] / k
                try:
                    step = xp.linalg.solve(normal, rhs[..., None])[..., 0]
                except Exception:  # singular holes: no move
                    step = xp.zeros_like(rhs)
                step = xp.where(xp.isfinite(step), step, 0.0)
                step = xp.where(self.frozen[idx, None], 0.0, step)
                if size:
                    try:  # standard error of the edge offset (the weights are 1 / variance)
                        cov = xp.linalg.inv(normal)
                        self.delta_sd[idx] = xp.sqrt(xp.maximum(cov[:, 4, 4], 0.0))
                    except Exception:
                        pass
                move = xp.clip(step[:, :2], -max_move, max_move)
                self.c[idx] = c + move
                a_old = self.a[idx]
                self.a[idx] = xp.clip(xp.clip(a_old + step[:, 2], 0.5 * a_old, 2.0 * a_old), 1e-3, 5.0 * a_med)
                self.level[idx] = self.level[idx] + step[:, 3]
                if size:
                    dstep = xp.where(self.inside[idx], step[:, 4], 0.0)
                    new_delta = xp.clip(self.delta[idx] + dstep, -max_delta, max_delta)
                    dstat.append(host(xp.abs(new_delta - self.delta[idx])))
                    self.delta[idx] = new_delta
                if self.gradient:
                    g0 = 5 if size else 4
                    self.gx[idx] = self.gx[idx] + step[:, g0]
                    self.gy[idx] = self.gy[idx] + step[:, g0 + 1]
                stat.append(host(xp.abs(move).max(axis=1)))
                flat_page -= xp.bincount(flat_r_old.ravel(), weights=cache[j].ravel(), minlength=npx)  # old light out
                flat_r, light = self._light(idx)  # the chunk's new light, seen by the colours that follow
                cache[j] = light
                flat_page += xp.bincount(flat_r.ravel(), weights=light.ravel(), minlength=npx)
            if far is not None:
                new_far = self.far_field().ravel()
                flat_page += new_far - far
                far = new_far
            if self.background == "grid":
                self.fit_background(flat_page.reshape(self.h, self.w), robust_c=robust_c)
            moves = np.concatenate(stat)
            dmoves = np.concatenate(dstat) if dstat else np.zeros(1)
            self.rms_typ = float(np.median(np.concatenate(rms_all)))
            move99, delta99 = float(np.percentile(moves, 99)), float(np.percentile(dmoves, 99))
            self.log.append(f"    holes it {it + 1}: median move {np.median(moves):.4f} px, 99% {move99:.4f}, "
                            f"delta 99% {delta99:.4f}, typical leftover rms {self.rms_typ:.2f}")
            if move99 < stop_px and delta99 < stop_px:
                break
        return flat_page.reshape(self.h, self.w)

    # --------------------------------------------------------------------------------- shape stage
    def cleaned_stack(self, page, good, radius_px=None):
        """Neighbour-cleaned stacked average on the fine grid: values (data - page + own - level) / a at each
        pixel's offset from its hole, weighted by a^2 / variance; returns (stack, weight) arrays (N, N).
        ``radius_px`` keeps only pixels within that distance of their hole (the core, while the template is still
        rough and the neighbours' cleaning with it is not to be trusted)."""
        xp = self.xp
        t = self.template
        t0, td, t2 = self._templates[0], self._templates[1], self._templates[4]
        sums = xp.zeros(t.N * t.N)
        wsum = xp.zeros(t.N * t.N)
        for sl in self._slices():
            g = good[sl]
            if not bool(g.any()):
                continue
            flat, valid, px, py = self._pixels(sl, self.R)
            sx, sy = self._fine(px, py, self.c[sl])
            v0, vd, v2 = cubic_sample([t0, td, t2], sx, sy, xp)
            a = self.a[sl, None]
            de = self.delta[sl, None]
            own = a * (v0 + de * (vd + de * v2))
            model = page.ravel()[flat] + self._background(sl, px, py)
            cleaned = (self.image.ravel()[flat] - model + own) / a
            wgt = self._weights(model) * a * a
            bx = xp.rint(sx).astype(xp.int64)
            by = xp.rint(sy).astype(xp.int64)
            ok = valid & g[:, None] & (bx >= 0) & (bx < t.N) & (by >= 0) & (by < t.N)
            if radius_px is not None:
                ok = ok & (xp.hypot(sx - t.N // 2, sy - t.N // 2) <= radius_px * t.U)
            b = (by * t.N + bx)[ok]
            sums += xp.bincount(b, weights=(cleaned * wgt)[ok], minlength=t.N * t.N)
            wsum += xp.bincount(b, weights=wgt[ok], minlength=t.N * t.N)
        sums, wsum = host(sums), host(wsum)
        with np.errstate(invalid="ignore", divide="ignore"):
            stack = np.where(wsum > 0, sums / wsum, np.nan)
        return stack.reshape(t.N, t.N), wsum.reshape(t.N, t.N)

    def fit_shape(self, stack, weight, stages=("round", "super", "full"), gate=None):
        """Least squares of the outline and blur parameters (with a free scale and constant) on the cleaned
        stack, in stages (model.fit_stages). The scale renormalises the holes' brightness."""
        t = self.template
        ok = np.isfinite(stack) & (weight > 0)
        sw = np.sqrt(weight[ok])
        y = stack[ok]
        box = 1.0 / t.U

        def render_for(theta):
            return t.render(theta, box_px=box)[ok]

        theta, scale, const, chi2 = fit_stages(t, render_for, y, sw, self.theta, stages=stages, gate=gate,
                                               log=self.log.append, pixel_nm=self.pixel_nm, unit=self.unit)
        return theta, scale, const, chi2

    def good_holes(self, page):
        """Holes fit for the shape stage: window inside the page, usual brightness, usual leftovers, small
        edge offset."""
        xp = self.xp
        c = host(self.c)
        a = host(self.a)
        inside = ((c[:, 0] >= self.R + 1) & (c[:, 0] <= self.w - 2 - self.R) & (c[:, 1] >= self.R + 1)
                  & (c[:, 1] <= self.h - 2 - self.R))
        med = np.median(a[inside]) if inside.any() else np.median(a)
        delta = host(self.delta)
        usual = (a > 0.4 * med) & (a < 2.5 * med) & (np.abs(delta - np.median(delta[inside] if inside.any() else delta)) < 0.6)
        rms = np.zeros(self.n)
        for sl in self._slices():
            flat, valid, px, py = self._pixels(sl, self.R)
            model = page.ravel()[flat] + self._background(sl, px, py)
            res = xp.where(valid, self.image.ravel()[flat] - model, 0.0)
            rms[sl] = host(xp.sqrt((res * res).sum(1) / xp.maximum(valid.sum(1), 1)))
        self.rms = rms
        typical = np.median(rms[inside]) if inside.any() else np.median(rms)
        return xp.asarray(inside & usual & (rms < 2.0 * typical)), rms

    # ------------------------------------------------------------------------------------------- run
    def run(self, rounds=14, iterations=5, log=None, relax=0.5, tol_px=0.08 / 21.0, release_px=0.3 / 21.0,
            per_hole_size="auto"):
        """Alternate the hole and shape stages. ``per_hole_size``: True / False, or "auto" (fitted when the median
        standard error of a hole's edge offset is under 0.3 outline radii after the first round, else every hole
        gets the page's outline size and only positions, brightness and level are fitted per hole).
        The shape stage's proposal is taken in full in the first round
        (crude start) and relaxed by ``relax`` afterwards: the two stages talk through the neighbours' cleaning
        and the holes' free levels, and the plain alternation over-shoots and swings. The outline stays round and
        the blur isotropic until that model has settled (proposal within ``release_px`` of the current template);
        then the superellipse and the blur's anisotropy are released, and once those have settled the harmonics
        (released early, they absorb the still-dirty stack and the fit settles in a worse place). Stops when the
        full model's proposal agrees with the current template to ``tol_px`` in CD and the stack chi2 has settled
        (< 5 % change). chi2 itself is not a usable gate: on a page of 50k holes the stack's noise is so small that a
        blur-family mismatch alone gives chi2 >> 1."""
        started = time.perf_counter()
        say = log if log is not None else (lambda s: None)
        self.initialise()
        t = self.template
        say(f"  start: {self.n:,} holes, spacing {self.spacing:.2f} px, fit window +-{self.R} px, render +-{self.Rr}, "
            f"{int(self.colour.max()) + 1} colours, template start {t.describe(self.theta, self.pixel_nm)['cd_nm']:.1f} {self.unit}")
        all_stages = ("round", "super", "full")
        level = 0
        chi2_prev = None
        self.size_on = per_hole_size is True
        for rnd in range(rounds):
            t_round = time.perf_counter()
            first = rnd == 0
            if rnd == 1 and per_hole_size == "auto":
                sd, radius = self.delta_information()
                self.size_on = sd < 0.3 * radius
                say(f"  per-hole edge offset: median standard error {sd:.3f} px against the outline radius "
                    f"{radius:.2f} px -> per-hole size {'on' if self.size_on else 'off (page size for every hole)'}")
            # the first rounds move the template by nanometres and need the long hole passes; later ones move it by
            # tenths and a few Gauss-Seidel sweeps are enough (the hole stage is where the time goes); every pass
            # stops early once the holes stand still
            page = self.fit_holes(iterations=iterations + 3 if rnd < 3 else max(3, iterations // 2 + 1),
                                  size=self.size_on and not first)
            self.fit_noise(page)
            good, rms = self.good_holes(page)
            # from the crude start, stack only each hole's core (the neighbours' cleaning is not yet to be trusted)
            stack, weight = self.cleaned_stack(page, good, radius_px=0.55 * self.spacing if first else None)
            stages = all_stages[:level + 1]
            proposal, scale, const, chi2 = self.fit_shape(stack, weight, stages=stages)
            t = self.template  # set_theta doubles the box when the hole's light reaches past it
            change_px = t.cd_px(proposal) - t.cd_px(self.theta)
            change = change_px * self.pixel_nm
            kappa = 1.0 if first else relax
            self.a = self.a * scale ** kappa
            self.set_theta(self.theta + kappa * (proposal - self.theta))
            t = self.template
            d = t.describe(self.theta, self.pixel_nm)
            n_good = int(host(good).sum())
            self.history.append(dict(round=rnd + 1, stages=list(stages), outline_cd=d["cd_nm"], blur_fwhm=d["blur_fwhm_nm"],
                                     chi2=chi2, proposal_change=change, const=const, good=n_good,
                                     var_line=self.var_line, mean_delta_px=float(host(self.delta).mean()),
                                     seconds=time.perf_counter() - t_round))
            say(f"  round {rnd + 1} ({stages[-1]}): outline CD {d['cd_nm']:.2f} {self.unit} (x {d['extent_x_nm']:.1f}, "
                f"y {d['extent_y_nm']:.1f}), p {d['superellipse']['p']:.2f}, blur fwhm {d['blur_fwhm_nm'][0]:.1f}/"
                f"{d['blur_fwhm_nm'][1]:.1f} {self.unit}, proposal {change:+.2f} {self.unit}, stack chi2 {chi2:.3f}, "
                f"const {const:+.4f}, "
                f"good {n_good:,}, noise var {self.var_line[0]:.3f} + {self.var_line[1]:.4f} level, mean delta "
                f"{float(host(self.delta).mean()):+.4f} px, {time.perf_counter() - t_round:.0f} s")
            for line in self.log:
                say(line)
            self.log = []
            settled = chi2_prev is not None and abs(chi2 - chi2_prev) < 0.05 * chi2_prev
            chi2_prev = chi2
            if not first and settled and abs(change_px) < release_px and level < len(all_stages) - 1:
                level += 1
                chi2_prev = None
                say(f"  {all_stages[level]} stage released")
                continue
            if level == len(all_stages) - 1 and settled and abs(change_px) < tol_px:
                self.converged = True
                say(f"  converged after round {rnd + 1}")
                break
        # final hole pass with the final template (to a standstill), then results
        page = self.fit_holes(iterations=2 * iterations, size=self.size_on)
        t = self.template
        if not self.size_on:
            self.delta_information()  # the standard errors, for the record
        good, rms = self.good_holes(page)
        self.page_model = page
        stack, weight = self.cleaned_stack(page, good)
        self.timings["total"] = time.perf_counter() - started
        out = self.results(good, rms)
        out["stack"] = stack
        out["stack_weight"] = weight
        out["template"] = host(self._t0_full)
        out["template_box"] = t.render(self.theta, box_px=1.0 / t.U)
        return out

    def results(self, good, rms):
        """Per-hole results. The edge offsets are reported as deviations from their mean over the good holes: the
        page's size is the outline's (fitted on the average hole with the blur free), and a hole's offset is how far
        it deviates. The per-hole fits, with the blur fixed, share one more bias (a slightly wrong blur reads every
        hole a little smaller or bigger, and the holes' common offset is a soft direction that the sweeps leave
        wherever they stopped: -0.1 to -0.9 nm on the small test page with the exact template, depending on
        the sweep order); the mean removes it."""
        t = self.template
        cx0, cy0 = t.centroid_px(self.theta)
        c = host(self.c) + np.array([cx0, cy0])
        g = host(good)
        delta_raw = host(self.delta)
        delta_mean = float(np.mean(delta_raw[g])) if g.any() else 0.0
        delta = delta_raw - delta_mean
        cd0 = t.cd_px(self.theta)
        # per-hole area-equivalent CD of the outline moved by delta, r -> r + delta g with g = hypot(r, r') / r (the
        # template's own offset): area = (1/2) int (r + delta g)^2 = A0 + delta int r g + delta^2 (1/2) int g^2, exact
        # (int r g = the perimeter). The prototype used the polar area of delta rounded to 1e-4 px below 2000 holes and
        # pi in place of (1/2) int g^2 above (Steiner's formula; the two differ by delta^2 (1/2) int (r'/r)^2)
        r, dr = t._radius_at(self.theta, t.angles)
        gr = np.hypot(r, dr) / r
        area0, perim, quad = math.pi * np.mean(r * r), 2 * math.pi * np.mean(r * gr), math.pi * np.mean(gr * gr)
        area = area0 + perim * delta + quad * delta ** 2
        cd_px = 2.0 * np.sqrt(np.maximum(area, 1e-6) / math.pi)
        delta_sd = host(self.delta_sd)
        # standard error of the CD from the edge offset's: dCD / ddelta = (perimeter + 2 Q delta) / sqrt(pi area) (2 for
        # a circle, as the prototype's 2 delta_sd)
        cd_sd_px = delta_sd * (perim + 2 * quad * delta) / np.sqrt(math.pi * np.maximum(area, 1e-6))
        # the level reported per hole: its own, or the page background under its centre
        level = self.background_at(self.c[:, 0], self.c[:, 1]) if self.background == "grid" else self.level
        return dict(
            centers_xy=c, brightness=host(self.a), level=host(level), edge_offset_px=delta,
            cd_px=cd_px, cd_outline_px=cd0, theta=self.theta,
            delta_sd_px=delta_sd, cd_sd_px=cd_sd_px, per_hole_size=self.size_on,
            edge_offset_mean_px=delta_mean, clip_px=self.clip_px,
            at_clip=np.abs(delta_raw) >= 0.99 * (self.clip_px or np.inf),
            describe=t.describe(self.theta, self.pixel_nm), good=host(good), rms=rms, var_line=self.var_line,
            history=self.history, seconds=self.timings.get("total"), spacing_px=self.spacing, window_px=self.R,
            converged=self.converged,
        )
