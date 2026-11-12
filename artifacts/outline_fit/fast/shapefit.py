"""Scratch: the page's outline + blur from its plain stack, as one least-squares fit; many pages in lockstep.

Model of the stack's bin at offset u_b, of sub-pixel class q (see fastfit.py):

    S(b) = scale x Re sum_f T^(f; theta) Gamma_q(f) exp(2 pi i f . u_b) / L^2 + const
    T^(f) = O^(f; outline) x OTF(f; blur) x sinc(fx) sinc(fy)

O^ is the outline's transform as a line integral round the outline, sampled at ``n_poly`` equal angles with the
outline's own tangent (trapezoid rule; O^(0) = area). Its derivative with respect to an outline parameter is another
integral with the same exponentials:  dO^/dp = int exp(-2 pi i f . x(phi)) r dr/dp dphi,  so one pass over the
outline gives the model and its whole Jacobian (no finite differences). Scale and constant are solved exactly for
every theta (variable projection, Kaufman's Jacobian).

Division of labour: the device evaluates, for a batch of pages, the cost, the gradient and the Gauss-Newton matrix
(one kernel for the outline integrals, single-precision synthesis, double-precision sums); the host runs
Levenberg-Marquardt on those small matrices (bounds as an active set, gain-ratio damping, weak priors on what a
page leaves undetermined). Stages, bounds and the grid of starts are the library's.
"""
from __future__ import annotations

import math

import numpy as np

KERNEL_SRC = r'''
extern "C" __global__
void outline_hat(const double* xs, const double* ys, const double* w0x, const double* w0y, const double* wk,
                 const double* fx, const double* fy, int B, int F, int n, int K, double dphi, const double* area,
                 double* out)
{
    // one thread per (page b, frequency f): sums over the n outline points
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long) B * F) return;
    int b = (int)(t / F), f = (int)(t % F);
    float fxx = (float) fx[f], fyy = (float) fy[f];
    double acc_re[1 + KMAX], acc_im[1 + KMAX];
    for (int k = 0; k <= K; ++k) { acc_re[k] = 0; acc_im[k] = 0; }
    const double* x = xs + (long long) b * n;
    const double* y = ys + (long long) b * n;
    const double* ax = w0x + (long long) b * n;
    const double* ay = w0y + (long long) b * n;
    const double* g = wk + (long long) b * n * K;
    for (int m = 0; m < n; ++m) {
        float ph = -6.2831853f * (fxx * (float) x[m] + fyy * (float) y[m]);
        float s, c;
        sincosf(ph, &s, &c);
        double w = (double) fxx * ay[m] - (double) fyy * ax[m];
        acc_re[0] += c * w; acc_im[0] += s * w;
        const double* gm = g + (long long) m * K;
        for (int k = 0; k < K; ++k) { acc_re[1 + k] += c * gm[k]; acc_im[1 + k] += s * gm[k]; }
    }
    double* o = out + (t * (1 + K)) * 2;
    double f2 = (double) fxx * fxx + (double) fyy * fyy;
    if (f2 == 0) { o[0] = area[b]; o[1] = 0; }
    else { double q = dphi / (6.283185307179586 * f2); o[0] = -acc_im[0] * q; o[1] = acc_re[0] * q; }  // i x acc
    for (int k = 0; k < K; ++k) { o[2 * (1 + k)] = acc_re[1 + k] * dphi; o[2 * (1 + k) + 1] = acc_im[1 + k] * dphi; }
}
'''


class ShapeFitter:
    def __init__(self, lib_template, *, box_px=48, f_max=0.6, U=4, R=7, n_poly=256, xp=np):
        self.t = lib_template
        self.xp = xp
        t = lib_template
        self.blur = t.blur
        self.ks = t.ks
        self.n_shape, self.n_blur, self.P = t.n_shape, t.n_blur, t.n_params
        self.L, self.U, self.R = int(box_px), int(U), int(R)
        self.n_poly = int(n_poly)
        self.phi = 2 * np.pi * np.arange(self.n_poly) / self.n_poly
        # frequencies k / L inside the circle |f| <= f_max, half plane kx >= 0 (kx = 0 keeps every ky: the row is
        # its own mirror image, so the synthesis counts it once and the others twice)
        kmax = int(math.floor(f_max * self.L))
        ky, kx = np.meshgrid(np.arange(-kmax, kmax + 1), np.arange(0, kmax + 1), indexing="ij")
        keep = (ky * ky + kx * kx) <= (f_max * self.L) ** 2
        self.kmax = kmax
        self.nky, self.nkx = 2 * kmax + 1, kmax + 1
        self.sel = np.flatnonzero(keep.ravel())
        self.F = len(self.sel)
        fx = (kx / self.L).ravel()[self.sel]
        fy = (ky / self.L).ravel()[self.sel]
        self.fx_h, self.fy_h = fx, fy
        self.fx, self.fy = xp.asarray(fx), xp.asarray(fy)
        self.pix = xp.sinc(self.fx) * xp.sinc(self.fy)
        self.dbl = xp.asarray(np.where(kx.ravel()[self.sel] == 0, 1.0, 2.0))
        # bins: n in [-nmax, nmax] per axis (offset n / U px); only bins every hole of the class fills
        self.nmax = int(math.floor((self.R - 0.5) * self.U))
        n = np.arange(-self.nmax, self.nmax + 1)
        cls = np.mod(n, self.U)
        self.npts = max(int(np.sum(cls == q)) for q in range(self.U))
        self.idx = np.full((self.U, self.npts), -1, dtype=np.int64)  # class -> positions in n (padded with -1)
        for q in range(self.U):
            where = np.flatnonzero(cls == q)
            self.idx[q, :len(where)] = where
        self.pad = self.idx >= 0
        u = np.where(self.pad, n[np.clip(self.idx, 0, None)], 0) / self.U  # offsets in px, (U, npts)
        kyv = np.arange(-kmax, kmax + 1) / self.L
        kxv = np.arange(0, kmax + 1) / self.L
        ey = np.exp(2j * np.pi * u[:, :, None] * kyv[None, None, :]) * self.pad[:, :, None]  # (U, npts, nky)
        ex = np.exp(2j * np.pi * u[:, :, None] * kxv[None, None, :]) * self.pad[:, :, None]  # (U, npts, nkx)
        ctype = np.complex64 if xp is not np else np.complex128
        self.ctype = ctype
        self.Ey = xp.asarray(ey.astype(ctype))
        self.ExT = xp.asarray(ex.transpose(0, 2, 1).astype(ctype) / (self.L ** 2))  # (U, nkx, npts), with the 1 / L^2
        self.sel_d = xp.asarray(self.sel)
        self.lo, self.hi = t.bounds()
        self._kernel = None
        self.evals = 0

    # ------------------------------------------------------------------------------------------ data layout
    def to_classes(self, grid):
        """A (..., Ms, Ms) bin grid (centre bin in the middle) -> (..., U, U, npts, npts) by class, 0 where padded."""
        xp = self.xp
        ms = grid.shape[-1]
        c = ms // 2
        pos = c - self.nmax + np.clip(self.idx, 0, None)  # (U, npts) index into the grid axis
        py = xp.asarray(pos)[:, None, :, None]
        px = xp.asarray(pos)[None, :, None, :]
        out = grid[..., py, px]
        mask = xp.asarray(self.pad[:, None, :, None] & self.pad[None, :, None, :])
        return xp.where(mask, out, 0.0), mask

    # ------------------------------------------------------------------------------------------- the model
    def _outline(self, th):
        """Host: outline points, tangents and the radius' derivatives for parameter vectors th (B, P)."""
        phi = self.phi[None, :]
        a, b = np.exp(th[:, 0:1]), np.exp(th[:, 1:2])
        p = 1.0 + np.exp(th[:, 2:3])
        t = phi - th[:, 3:4]
        ct, st = np.cos(t), np.sin(t)
        ua, ub = np.abs(ct / a), np.abs(st / b)
        A, Bq = ua ** p, ub ** p
        S = A + Bq
        base = S ** (-1.0 / p)
        dbase = -(1.0 / p) * base ** (1 + p) * (p * ua ** (p - 1) * np.sign(ct) * (-st / a)
                                                 + p * ub ** (p - 1) * np.sign(st) * (ct / b))
        mod = np.ones_like(t)
        dmod = np.zeros_like(t)
        for i, k in enumerate(self.ks):
            ck, sk = th[:, 4 + 2 * i:5 + 2 * i], th[:, 5 + 2 * i:6 + 2 * i]
            mod = mod + ck * np.cos(k * phi) + sk * np.sin(k * phi)
            dmod = dmod - ck * k * np.sin(k * phi) + sk * k * np.cos(k * phi)
        r = base * mod
        dr = dbase * mod + base * dmod
        with np.errstate(divide="ignore", invalid="ignore"):
            la = np.where(ua > 0, A * np.log(np.maximum(ua, 1e-300)), 0.0)
            lb = np.where(ub > 0, Bq * np.log(np.maximum(ub, 1e-300)), 0.0)
        g = [base * (A / S) * mod, base * (Bq / S) * mod,
             (p - 1.0) * base * (np.log(S) / (p * p) - (la + lb) / (p * S)) * mod, -dbase * mod]
        for k in self.ks:
            g += [base * np.cos(k * phi) + 0 * t, base * np.sin(k * phi) + 0 * t]
        g = np.stack(g, axis=-1)  # (B, n, n_shape)
        x, y = r * np.cos(phi), r * np.sin(phi)
        xd, yd = dr * np.cos(phi) - r * np.sin(phi), dr * np.sin(phi) + r * np.cos(phi)
        dphi = 2 * np.pi / self.n_poly
        area = 0.5 * (r * r).sum(axis=1) * dphi
        return x, y, xd, yd, g * r[..., None], area, dphi

    def outline_hat(self, th, grads=True):
        """O^ and its derivatives at the band's frequencies: (B, F, 1 + n_shape) complex. th: host (B, P)."""
        xp = self.xp
        th = np.asarray(th, dtype=np.float64)
        x, y, xd, yd, g, area, dphi = self._outline(th)
        B = th.shape[0]
        K = self.n_shape if grads else 0
        if xp is np:
            E = np.exp(-2j * np.pi * (self.fx[None, :, None] * x[:, None, :] + self.fy[None, :, None] * y[:, None, :]))
            f2 = self.fx ** 2 + self.fy ** 2
            s0 = self.fx[None, :] * (E @ yd[:, :, None])[..., 0] - self.fy[None, :] * (E @ xd[:, :, None])[..., 0]
            o0 = np.where(f2[None, :] == 0, area[:, None], 1j * s0 * dphi / (2 * np.pi * np.where(f2 == 0, 1.0, f2))[None, :])
            if not grads:
                return o0[..., None]
            return np.concatenate([o0[..., None], (E @ g) * dphi], axis=-1)
        if self._kernel is None:
            import cupy as cp

            self._kernel = cp.RawKernel(KERNEL_SRC.replace("KMAX", str(self.n_shape)), "outline_hat")
        out = xp.empty((B, self.F, 1 + K, 2))
        geo = xp.asarray(np.ascontiguousarray(np.stack([x, y, xd, yd])))  # one upload
        gk = xp.asarray(np.ascontiguousarray(g[..., :K])) if K else xp.zeros(1)
        self._kernel(((B * self.F + 127) // 128,), (128,),
                     (geo[0], geo[1], geo[2], geo[3], gk, self.fx, self.fy, np.int32(B), np.int32(self.F),
                      np.int32(self.n_poly), np.int32(K), np.float64(dphi), xp.asarray(area), out))
        return out[..., 0] + 1j * out[..., 1]

    def otf(self, th, grads=True):
        """Blur transfer function and its derivatives: (B, F, 1 + n_blur). th: host (B, P)."""
        xp = self.xp
        bp = xp.asarray(np.asarray(th, dtype=np.float64)[:, self.n_shape:])
        fx2, fy2 = (self.fx ** 2)[None, :], (self.fy ** 2)[None, :]
        c = 2 * np.pi ** 2
        if self.blur == "gauss2":
            s1x, s1y, s2x, s2y = (xp.exp(bp[:, i:i + 1]) for i in range(4))
            w2 = 1.0 / (1.0 + xp.exp(-bp[:, 4:5]))
            g1 = xp.exp(-c * (s1x ** 2 * fx2 + s1y ** 2 * fy2))
            g2 = xp.exp(-c * (s2x ** 2 * fx2 + s2y ** 2 * fy2))
            out = [(1 - w2) * g1 + w2 * g2]
            if grads:
                out += [(1 - w2) * g1 * (-2 * c * s1x ** 2 * fx2), (1 - w2) * g1 * (-2 * c * s1y ** 2 * fy2),
                        w2 * g2 * (-2 * c * s2x ** 2 * fx2), w2 * g2 * (-2 * c * s2y ** 2 * fy2),
                        (g2 - g1) * w2 * (1 - w2)]
            return xp.stack(out, axis=-1)
        sx, sy = xp.exp(bp[:, 0:1]), xp.exp(bp[:, 1:2])
        g = xp.exp(-c * (sx ** 2 * fx2 + sy ** 2 * fy2))
        if self.blur == "lens":
            s = xp.clip(xp.sqrt(fx2 + fy2) / xp.exp(bp[:, 2:3]), 0.0, 1.0)
            root = xp.sqrt(1.0 - s * s)
            lens = (2 / np.pi) * (xp.arccos(s) - s * root)
            out = [g * lens]
            if grads:
                out += [g * lens * (-2 * c * sx ** 2 * fx2), g * lens * (-2 * c * sy ** 2 * fy2),
                        g * (4 / np.pi) * s * root]
            return xp.stack(out, axis=-1)
        out = [g]
        if grads:
            out += [g * (-2 * c * sx ** 2 * fx2), g * (-2 * c * sy ** 2 * fy2)]
        return xp.stack(out, axis=-1)

    def template_hat(self, th, tm=None):
        """T^ at the band's frequencies, (B, F, 1); with ``tm`` (P, nf) also its derivatives with respect to the
        stage's free parameters, (B, F, 1 + nf)."""
        xp = self.xp
        grads = tm is not None
        o = self.outline_hat(th, grads)
        b = self.otf(th, grads)
        pix = self.pix[None, :, None]
        if not grads:
            return o * b * pix
        full = xp.concatenate([o[..., 1:] * b[..., :1], o[..., :1] * b[..., 1:]], axis=-1)  # d / d theta, (B, F, P)
        return xp.concatenate([o[..., :1] * b[..., :1], xp.matmul(full, xp.asarray(tm).astype(full.dtype))], axis=-1) * pix

    def synth(self, coeff, gamma, classes=None):
        """Stack model by class: coeff (B, F, C) of the band, gamma (B, U, U, nky, nkx) -> (B, C, U, U, npts, npts);
        ``classes`` = (qy, qx): that class alone, (B, C, npts, npts)."""
        xp = self.xp
        B, F, C = coeff.shape
        full = xp.zeros((B, C, self.nky * self.nkx), dtype=self.ctype)
        full[:, :, self.sel_d] = (coeff * self.dbl[None, :, None]).transpose(0, 2, 1).astype(self.ctype)
        if classes is not None:
            qy, qx = classes
            full = full.reshape(B, C, self.nky, self.nkx) * gamma[:, None, qy, qx]
            return xp.matmul(xp.matmul(self.Ey[qy][None, None], full), self.ExT[qx][None, None]).real
        full = full.reshape(B, C, 1, 1, self.nky, self.nkx) * gamma[:, None]
        left = xp.matmul(self.Ey[None, None, :, None], full)  # (B, C, U, U, npts, nkx)
        return xp.matmul(left, self.ExT[None, None, None, :]).real

    def prepare(self, gamma, y, w):
        """Device arrays in the precision the evaluation uses: gamma (B, U, U, nky, nkx), y, w (B, U, U, np, np)."""
        xp = self.xp
        B = y.shape[0]
        return dict(gamma=xp.asarray(gamma).astype(self.ctype), y=xp.asarray(y, dtype=xp.float64).reshape(B, -1),
                    w=xp.asarray(w, dtype=xp.float64).reshape(B, -1), B=B)

    def evaluate(self, th, data, tm=None):
        """Host results for parameter vectors th (B, P): cost (B,), scale, const and, with ``tm`` (P, nf), the
        gradient g (B, nf) = J^T r and the Gauss-Newton matrix A (B, nf, nf) = J^T J (scale and constant projected
        out), all for the stage's free parameters."""
        xp = self.xp
        self.evals += 1
        B = data["B"]
        M = self.synth(self.template_hat(th, tm), data["gamma"])  # (B, C, U, U, np, np) single precision
        C = M.shape[1]
        M = M.reshape(B, C, -1).astype(xp.float64)
        y, w = data["y"], data["w"]
        m = M[:, 0]
        ws = w.sum(axis=1)
        mw = (w * m).sum(axis=1) / ws
        yw = (w * y).sum(axis=1) / ws
        dm = m - mw[:, None]
        dy = y - yw[:, None]
        var = (w * dm * dm).sum(axis=1)
        scale = (w * dm * dy).sum(axis=1) / xp.where(var > 0, var, 1.0)
        const = yw - scale * mw
        res = scale[:, None] * dm - dy
        cost = (w * res * res).sum(axis=1)
        if tm is None:
            out = xp.stack([cost, scale, const], axis=1)
            out = out.get() if xp is not np else out
            return out[:, 0], out[:, 1], out[:, 2]
        nf = C - 1
        sw = xp.sqrt(w)
        r = sw * res
        D = M[:, 1:] * (scale[:, None] * sw)[:, None, :]  # (B, nf, n)
        u1 = sw / xp.sqrt(ws)[:, None]
        u2 = sw * dm
        u2 = u2 / xp.sqrt(xp.maximum(var, 1e-300))[:, None]
        D = D - u1[:, None, :] * xp.matmul(D, u1[:, :, None])
        D = D - u2[:, None, :] * xp.matmul(D, u2[:, :, None])
        A = xp.matmul(D, D.transpose(0, 2, 1))
        g = xp.matmul(D, r[:, :, None])[:, :, 0]
        out = xp.concatenate([cost[:, None], scale[:, None], const[:, None], g, A.reshape(B, nf * nf)], axis=1)
        out = out.get() if xp is not np else out  # one download
        return out[:, 0], out[:, 1], out[:, 2], out[:, 3:3 + nf], out[:, 3 + nf:].reshape(B, nf, nf)

    # --------------------------------------------------------------------------------------------- fitting
    def _stage(self, name):
        from subpx._outline.model import stage_map

        free, ties = stage_map(self.t, name)
        tm = np.zeros((self.P, len(free)))
        for i, k in enumerate(free):
            tm[k, i] = 1.0
        for k, src in ties.items():
            tm[k, free.index(src)] = 1.0
        return free, ties, tm

    def grid_start(self, base, data, chunk=8):
        """The library's grid of starts for the round stage (15 sizes x 12 blur widths): the cheapest per page.
        One outline per size and one transfer function per width; the 180 products are compared on the bins of
        the page's most populated sub-pixel class (a coarse scan: a fraction of the holes is plenty)."""
        xp = self.xp
        B = base.shape[0]
        j = self.n_shape
        a0 = np.exp(base[:, 0])
        sizes = a0[:, None] * np.linspace(0.4, 1.8, 15)[None, :]
        widths = np.linspace(0.45, 2.2, 12)
        tha = np.repeat(base[:, None, :], 15, axis=1)
        tha[:, :, 0] = tha[:, :, 1] = np.log(sizes)
        ths = np.repeat(base[:1, None, :], 12, axis=1)[0]
        ths[:, j] = ths[:, j + 1] = np.log(widths)
        if self.blur == "gauss2":
            ths[:, j + 2] = ths[:, j + 3] = np.log(2 * widths)
            ths[:, j + 4] = math.log(0.05 / 0.95)
        elif self.blur == "lens":
            ths[:, j + 2] = math.log(0.5)
        tha = np.clip(tha, self.lo + 1e-6, self.hi - 1e-6)
        ths = np.clip(ths, self.lo + 1e-6, self.hi - 1e-6)
        b = self.otf(ths, False)[..., 0].reshape(1, 1, 12, self.F)
        U, npts = self.U, self.npts
        yc = data["y"].reshape(B, U * U, npts * npts)
        wc = data["w"].reshape(B, U * U, npts * npts)
        # each page's most populated sub-pixel class (a pitch of a whole number of pixels leaves most classes empty)
        top = wc.sum(axis=2).argmax(axis=1)
        pick = xp.arange(B)
        y, w = yc[pick, top], wc[pick, top]
        qy, qx = top // U, top % U
        gam = data["gamma"].reshape(B, U * U, self.nky, self.nkx)[pick, top]
        out = base.copy()
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            o = self.outline_hat(tha[s:e].reshape((e - s) * 15, -1), False)[..., 0]
            coeff = (o.reshape(e - s, 15, 1, self.F) * b * self.pix).reshape(e - s, 180, self.F).transpose(0, 2, 1)
            self.evals += 1
            full = xp.zeros((e - s, 180, self.nky * self.nkx), dtype=self.ctype)
            full[:, :, self.sel_d] = (coeff * self.dbl[None, :, None]).transpose(0, 2, 1).astype(self.ctype)
            full = full.reshape(e - s, 180, self.nky, self.nkx) * gam[s:e, None]
            M = xp.matmul(xp.matmul(self.Ey[qy[s:e]][:, None], full), self.ExT[qx[s:e]][:, None]).real
            M = M.reshape(e - s, 180, -1).astype(xp.float64)
            yb, wb = y[s:e, None, :], w[s:e, None, :]
            ws = wb.sum(axis=2, keepdims=True)
            dm = M - (wb * M).sum(axis=2, keepdims=True) / ws
            dy = yb - (wb * yb).sum(axis=2, keepdims=True) / ws
            var = (wb * dm * dm).sum(axis=2, keepdims=True)
            scale = (wb * dm * dy).sum(axis=2, keepdims=True) / xp.where(var > 0, var, 1.0)
            res = scale * dm - dy
            best = (wb * res * res).sum(axis=2).argmin(axis=1)
            best = np.asarray(best.get() if hasattr(best, "get") else best)
            for bi in range(e - s):
                ia, isg = divmod(int(best[bi]), 12)
                out[s + bi, 0] = out[s + bi, 1] = tha[s + bi, ia, 0]
                out[s + bi, j:] = ths[isg, j:]
        return out

    def default_prior(self):
        """Weak Gaussian priors (in chi-square units) on the blur's secondary parameters, as linear forms of theta:
        rows (L, mean, sd). They decide only what the page leaves undetermined (a second Gaussian that the page's
        blur does not have: its width and weight are then arbitrary, and the fit would wander for ever)."""
        j = self.n_shape
        rows = []
        if self.blur == "gauss2":
            for k in (0, 1):
                L = np.zeros(self.P)
                L[j + 2 + k], L[j + k] = 1.0, -1.0
                rows.append((L, math.log(2.0), 0.7))
            L = np.zeros(self.P)
            L[j + 4] = 1.0
            rows.append((L, math.log(0.05 / 0.95), 1.5))
        return rows

    def fit(self, theta0, gamma, y, w, stages=("round", "super", "full"), grid=True, max_iter=(12, 20, 30), ftol=1e-7,
            xtol=1e-7, prior="default", log=None, trace=None):
        """Staged Levenberg-Marquardt for a batch of pages. theta0 (B, P) on the host; gamma (B, U, U, nky, nkx),
        y and w (B, U, U, npts, npts) on the device, w in 1 / variance units so that the cost is a chi-square.
        Returns theta (B, P), scale, const, cost (host arrays), [(stage, iterations)]."""
        data = self.prepare(gamma, y, w)
        theta = np.clip(np.asarray(theta0, float).copy(), self.lo + 1e-6, self.hi - 1e-6)
        B = theta.shape[0]
        rows = self.default_prior() if isinstance(prior, str) else (prior or [])
        PL = np.stack([r[0] for r in rows]) if rows else np.zeros((0, self.P))
        pm = np.array([r[1] for r in rows])
        pw = np.array([1.0 / r[2] ** 2 for r in rows])
        info = []
        if isinstance(max_iter, int):
            max_iter = (max_iter,) * len(stages)
        scale = const = cost = None
        for stage, cap in zip(stages, max_iter):
            free, ties, tm = self._stage(stage)
            base = theta.copy()
            if stage == "round":
                base[:, 1] = base[:, 0]
                base[:, 2] = 0.0
                base[:, 3] = 0.0
                base[:, 4:self.n_shape] = 0.0
                if grid:
                    base = self.grid_start(base, data)
            lo, hi = self.lo[free] + 1e-6, self.hi[free] - 1e-6
            fixed = base * (1.0 - tm.sum(axis=1))[None, :]  # entries no free parameter drives
            nf = len(free)
            pj = PL @ tm  # (n_prior, nf)
            pa = (pj.T * pw[None, :]) @ pj

            def expand(q):
                return fixed + q @ tm.T

            def evaluate(q):
                th = expand(q)
                c, s, k, g, A = self.evaluate(th, data, tm)
                dev = th @ PL.T - pm[None, :]
                return c + (pw[None, :] * dev * dev).sum(axis=1), s, k, g + (dev * pw[None, :]) @ pj, A + pa[None]

            q = np.clip(base[:, free], lo, hi)
            cost, scale, const, g, A = evaluate(q)
            lam = np.full(B, 1e-3)
            nu = np.full(B, 2.0)
            done = np.zeros(B, dtype=bool)
            its = 0
            for its in range(1, cap + 1):
                q_new = q.copy()
                for b in range(B):
                    if done[b]:
                        continue
                    # bounds as an active set: a parameter resting on a bound with the downhill direction pointing
                    # outside takes no step; one that the step would carry past a bound is put on the bound and the
                    # others are solved again with it there (clipping a solved step breaks it)
                    frozen = ((q[b] <= lo) & (g[b] > 0)) | ((q[b] >= hi) & (g[b] < 0))
                    held = np.zeros(nf)
                    step = held
                    for _ in range(nf):
                        k = ~frozen
                        if not k.any():
                            step = held
                            break
                        Ak = A[b][np.ix_(k, k)]
                        d = np.diag(Ak)
                        d = np.maximum(d, 1e-10 * d.max())
                        rhs = -(g[b][k] + A[b][np.ix_(k, ~k)] @ held[~k])
                        try:
                            sk = np.linalg.solve(Ak + np.diag(lam[b] * d), rhs)
                        except np.linalg.LinAlgError:
                            sk = np.zeros(int(k.sum()))
                        step = held.copy()
                        step[k] = np.where(np.isfinite(sk), sk, 0.0)
                        over = (q[b] + step > hi) & k
                        under = (q[b] + step < lo) & k
                        if not (over | under).any():
                            break
                        held = np.where(over, hi - q[b], np.where(under, lo - q[b], held))
                        frozen = frozen | over | under
                    q_new[b] = np.clip(q[b] + step, lo, hi)
                dq = q_new - q
                cost2, scale2, const2, g2, A2 = evaluate(q_new)
                # gain ratio against the quadratic model's prediction (Nielsen's damping update)
                pred = -(2.0 * (g * dq).sum(axis=1) + np.einsum("bi,bij,bj->b", dq, A, dq))
                rho = (cost - cost2) / np.maximum(pred, 1e-300)
                better = (cost2 < cost) & ~done
                small = (((cost - cost2) <= ftol * cost)
                         | (np.abs(dq).max(axis=1) <= xtol * (xtol + np.abs(q).max(axis=1))))
                q = np.where(better[:, None], q_new, q)
                A = np.where(better[:, None, None], A2, A)
                g = np.where(better[:, None], g2, g)
                scale, const = np.where(better, scale2, scale), np.where(better, const2, const)
                done = done | (better & small) | (~better & (lam > 1e7)) | (pred <= ftol * cost)
                cost = np.where(better, cost2, cost)
                grow = np.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                lam = np.where(better, lam * grow, lam * nu)
                nu = np.where(better, 2.0, nu * 2.0)
                if trace is not None:
                    trace.append((stage, its, float(cost[0]), float(lam[0]), bool(better[0])))
                if done.all():
                    break
            theta = expand(q)
            info.append((stage, its))
            if log:
                log(f"    shape stage {stage}: {its} iterations")
        return theta, scale, const, cost, info
