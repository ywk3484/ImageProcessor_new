"""Outline fit: forward model of a hole as a sharp outline blurred by the page's blur, on a fine grid.

Library copy of the validated prototype (artifacts/outline_fit/model.py, 2026-11-05), numerics unchanged.

Template T(x, y) (clear-area brightness 1 inside the hole far from its edge) on an N x N grid of spacing 1/U px:

    T = pixel_box * blur * indicator(outline + edge offset)

Outline (star-convex, radius r(angle) in px): a superellipse |x/a|^p + |y/b|^p = 1 turned by phi (p = 2 ellipse,
p -> large rectangle), times (1 + sum_k c_k cos k angle + s_k sin k angle) for the harmonics in ``harmonics``
(default 1 and 3: the asymmetries a superellipse lacks). The edge offset delta (px) moves the outline along its
normal. Blur: ``blur='gauss2'``, a mixture of two centred anisotropic Gaussians (s1x, s1y, s2x, s2y, w2), or
``blur='lens'``, one anisotropic Gaussian times the transfer function of a circular pupil with a free cut-off
(sx, sy, cut-off). Pixel box: sinc(fx) sinc(fy). Parameters are one flat vector theta; see ``names``.
The indicator has a short logistic edge (``edge_px``) so the fine grid does not alias.
"""
from __future__ import annotations

import math

import numpy as np


def lens_otf(nu, cutoff):
    s = np.clip(nu / cutoff, 0.0, 1.0)
    return (2 / np.pi) * (np.arccos(s) - s * np.sqrt(1.0 - s * s))


class Template:
    def __init__(self, upsample=8, size_px=24, harmonics=(1, 3), blur="gauss2", aa=4):
        self.U = int(upsample)
        self.N = int(size_px * upsample)
        self.ks = tuple(int(k) for k in harmonics)
        self.blur = str(blur)
        if self.blur not in ("gauss2", "lens", "gauss1"):
            raise ValueError("blur must be 'gauss2', 'lens' or 'gauss1'")
        self.aa = int(aa)  # sub-samples per fine pixel and axis for the outline's exact pixel coverage
        n, u = self.N, self.U
        x = (np.arange(n) - n // 2) / u
        self.xx, self.yy = np.meshgrid(x, x)
        # anti-aliasing grid: aa x aa samples inside each fine pixel
        m = n * self.aa
        xa = (np.arange(m) - m // 2 + 0.5) / (u * self.aa) - 0.5 / u
        xxa, yya = np.meshgrid(xa, xa)
        self.rho_aa = np.hypot(xxa, yya)
        # the radius function is tabulated on a fine angle table and looked up per aa sample
        self.n_table = 8192
        self.table_angles = -np.pi + 2 * np.pi * np.arange(self.n_table) / self.n_table
        idx = np.rint((np.arctan2(yya, xxa) + np.pi) / (2 * np.pi) * self.n_table).astype(np.int64)
        self.ang_idx_aa = np.mod(idx, self.n_table)
        f = np.fft.fftfreq(n, d=1.0 / u)  # cycles per pixel
        self.fx, self.fy = f[None, :], f[:, None]
        self.nu = np.hypot(self.fx, self.fy)
        self.pixel_otf = np.sinc(self.fx) * np.sinc(self.fy)
        self.angles = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
        # The coverage is the sharp indicator box-filtered by one fine pixel and by the one-sample ramp; dividing
        # by their transfer functions gives back the sharp outline's spectrum (exact up to the aa grid's aliasing;
        # the ramp acts along the edge normal, taken as radial here: a 0.1 % effect).
        self.edge_fix = 1.0 / (np.sinc(self.fx / u) * np.sinc(self.fy / u) * np.sinc(self.nu / (u * self.aa)))

    # ------------------------------------------------------------------ parameters
    @property
    def n_shape(self):
        return 4 + 2 * len(self.ks)

    @property
    def n_blur(self):
        return {"gauss2": 5, "lens": 3, "gauss1": 2}[self.blur]

    @property
    def n_params(self):
        return self.n_shape + self.n_blur

    @property
    def names(self):
        out = ["log a", "log b", "log(p-1)", "phi"]
        for k in self.ks:
            out += [f"c{k}", f"s{k}"]
        if self.blur == "gauss2":
            out += ["log s1x", "log s1y", "log s2x", "log s2y", "logit w2"]
        elif self.blur == "lens":
            out += ["log sx", "log sy", "log cutoff"]
        else:
            out += ["log sx", "log sy"]
        return out

    def pack(self, a, b, p, phi, coeffs, blur_params):
        head = [math.log(a), math.log(b), math.log(p - 1.0), phi]
        return np.concatenate([head, np.asarray(coeffs, float).ravel(), np.asarray(blur_params, float)])

    def blur_pack(self, **kw):
        if self.blur == "gauss2":
            w2 = kw.get("w2", 0.05)
            return [math.log(kw.get("s1x", 1.0)), math.log(kw.get("s1y", 1.0)), math.log(kw.get("s2x", 2.0)),
                    math.log(kw.get("s2y", 2.0)), math.log(w2 / (1 - w2))]
        if self.blur == "lens":
            return [math.log(kw.get("sx", 1.0)), math.log(kw.get("sy", 1.0)), math.log(kw.get("cutoff", 0.45))]
        return [math.log(kw.get("sx", 1.0)), math.log(kw.get("sy", 1.0))]

    def unpack(self, theta):
        theta = np.asarray(theta, float)
        a, b, p, phi = math.exp(theta[0]), math.exp(theta[1]), 1.0 + math.exp(theta[2]), theta[3]
        coeffs = theta[4:self.n_shape].reshape(-1, 2)
        return a, b, p, phi, coeffs, theta[self.n_shape:]

    def start(self, half_width_px, blur_sigma_px=1.0):
        """A round outline of the given half width and an isotropic blur as the first guess."""
        return self.pack(half_width_px, half_width_px, 2.0, 0.0, np.zeros((len(self.ks), 2)),
                         self.blur_pack(s1x=blur_sigma_px, s1y=blur_sigma_px, s2x=2 * blur_sigma_px,
                                        s2y=2 * blur_sigma_px, sx=blur_sigma_px, sy=blur_sigma_px))

    def bounds(self):
        lo = np.full(self.n_params, -np.inf)
        hi = np.full(self.n_params, np.inf)
        lo[0:2], hi[0:2] = math.log(0.2), math.log(8.0)
        lo[2], hi[2] = math.log(0.3), math.log(30.0)  # p in (1.3, 31)
        lo[3], hi[3] = -np.pi / 4, np.pi / 4
        lo[4:self.n_shape], hi[4:self.n_shape] = -0.3, 0.3
        j = self.n_shape
        if self.blur == "gauss2":
            lo[j:j + 4], hi[j:j + 4] = math.log(0.15), math.log(4.0)
            lo[j + 4], hi[j + 4] = math.log(0.005 / 0.995), math.log(0.6 / 0.4)
        elif self.blur == "lens":
            lo[j:j + 2], hi[j:j + 2] = math.log(0.1), math.log(4.0)
            lo[j + 2], hi[j + 2] = math.log(0.2), math.log(8.0)
        else:
            lo[j:j + 2], hi[j:j + 2] = math.log(0.15), math.log(4.0)
        return lo, hi

    # ------------------------------------------------------------------ outline
    def _radius_at(self, theta, angles):
        """r(angle) and dr/dangle for the given angles (px)."""
        a, b, p, phi, coeffs, _ = self.unpack(theta)
        t = angles - phi
        ct, st = np.cos(t), np.sin(t)
        # superellipse radius in its own frame
        base = (np.abs(ct / a) ** p + np.abs(st / b) ** p) ** (-1.0 / p)
        # d base / d t
        dbase = -(1.0 / p) * base ** (1 + p) * (
            p * np.abs(ct / a) ** (p - 1) * np.sign(ct) * (-st / a) + p * np.abs(st / b) ** (p - 1) * np.sign(st) * (ct / b))
        mod = np.ones_like(angles)
        dmod = np.zeros_like(angles)
        for i, k in enumerate(self.ks):
            mod = mod + coeffs[i, 0] * np.cos(k * angles) + coeffs[i, 1] * np.sin(k * angles)
            dmod = dmod - coeffs[i, 0] * k * np.sin(k * angles) + coeffs[i, 1] * k * np.cos(k * angles)
        return base * mod, dbase * mod + base * dmod

    def indicator(self, theta, delta=0.0):
        """Fraction of each fine pixel inside the outline (exact pixel coverage on the aa grid), outline moved
        ``delta`` px along its normal."""
        r, dr = self._radius_at(theta, self.table_angles)
        edge_r = r + delta * np.hypot(r, dr) / r if delta else r
        # coverage of each aa sample: a linear ramp one sample wide across the edge (the exact coverage of a
        # straight edge), so the template is continuous and piecewise linear in the outline parameters
        inside = np.clip(0.5 + (edge_r[self.ang_idx_aa] - self.rho_aa) * (self.U * self.aa), 0.0, 1.0)
        a = self.aa
        return inside.reshape(self.N, a, self.N, a).mean(axis=(1, 3))

    # ------------------------------------------------------------------ blur
    def otf(self, theta):
        bp = self.unpack(theta)[5]
        if self.blur == "gauss2":
            s1x, s1y, s2x, s2y = np.exp(bp[:4])
            w2 = 1.0 / (1.0 + math.exp(-bp[4]))
            g1 = np.exp(-2 * np.pi ** 2 * (s1x ** 2 * self.fx ** 2 + s1y ** 2 * self.fy ** 2))
            g2 = np.exp(-2 * np.pi ** 2 * (s2x ** 2 * self.fx ** 2 + s2y ** 2 * self.fy ** 2))
            return (1 - w2) * g1 + w2 * g2
        sx, sy = np.exp(bp[:2])
        g = np.exp(-2 * np.pi ** 2 * (sx ** 2 * self.fx ** 2 + sy ** 2 * self.fy ** 2))
        if self.blur == "lens":
            g = g * lens_otf(self.nu, math.exp(bp[2]))
        return g

    def blur_kernel(self, theta):
        """The blur alone on the fine grid (sums to 1)."""
        k = np.fft.fftshift(np.fft.ifft2(self.otf(theta)).real)
        return k / k.sum()

    def blur_fwhm_px(self, theta):
        """Half-height width of the blur along x and y (px), interpolated on the fine grid."""
        k = self.blur_kernel(theta)
        c = self.N // 2
        out = []
        for prof in (k[c], k[:, c]):
            half = prof[c] / 2
            # walk right from the centre to the half crossing, interpolate
            j = c
            while j + 1 < len(prof) and prof[j + 1] > half:
                j += 1
            frac = (prof[j] - half) / max(prof[j] - prof[j + 1], 1e-12)
            out.append(2 * (j - c + frac) / self.U)
        return tuple(out)

    def blur_sigma_px(self, theta):
        """Root-mean-square radius of the blur along x and y (px): a width that does not depend on its form."""
        k = self.blur_kernel(theta)
        return (math.sqrt((k * self.xx ** 2).sum()), math.sqrt((k * self.yy ** 2).sum()))

    # ------------------------------------------------------------------ template
    def render(self, theta, delta=0.0, box_px=0.0):
        ind = self.indicator(theta, delta)
        spec = np.fft.fft2(np.fft.ifftshift(ind)) * self.edge_fix * self.otf(theta) * self.pixel_otf
        if box_px > 0:
            spec = spec * np.sinc(self.fx * box_px) * np.sinc(self.fy * box_px)
        return np.fft.fftshift(np.fft.ifft2(spec).real)

    def render_pair(self, theta, h=0.05):
        """Template and its derivative with respect to the edge offset (per px)."""
        return self.render(theta), (self.render(theta, h) - self.render(theta, -h)) / (2 * h)

    def render_quad(self, theta, h=0.1):
        """Template T0 and the first and second edge-offset terms, T(delta) ~ T0 + delta T1 + delta^2 T2 (delta in px):
        the linear model alone reads a hole 6 nm off the outline about 1 nm too big (the response curves up both ways)."""
        t0 = self.render(theta)
        tp, tm = self.render(theta, h), self.render(theta, -h)
        return t0, (tp - tm) / (2 * h), (tp + tm - 2 * t0) / (2 * h * h)

    def derivatives_xy(self, t):
        """d/dx and d/dy (per px) of a fine-grid template, exact for the band-limited template."""
        spec = np.fft.fft2(np.fft.ifftshift(t))
        dx = np.fft.fftshift(np.fft.ifft2(spec * (2j * np.pi * self.fx)).real)
        dy = np.fft.fftshift(np.fft.ifft2(spec * (2j * np.pi * self.fy)).real)
        return dx, dy

    # ------------------------------------------------------------------ readings
    def outline_area(self, theta, delta=0.0):
        r, dr = self._radius_at(theta, self.angles)
        if delta:
            r = r + delta * np.hypot(r, dr) / r
        return float(np.pi * np.mean(r ** 2))

    def cd_px(self, theta, delta=0.0):
        """Area-equivalent diameter (px)."""
        return 2.0 * math.sqrt(self.outline_area(theta, delta) / math.pi)

    def extents_px(self, theta):
        """Width along x and y of the outline through its centre (px)."""
        r, _ = self._radius_at(theta, np.array([0.0, math.pi, math.pi / 2, -math.pi / 2]))
        return float(r[0] + r[1]), float(r[2] + r[3])

    def centroid_px(self, theta):
        ind = self.indicator(theta)
        s = ind.sum()
        return float((ind * self.xx).sum() / s), float((ind * self.yy).sum() / s)

    def outline_xy(self, theta, delta=0.0, n=360):
        angles = np.linspace(-np.pi, np.pi, n)
        r, dr = self._radius_at(theta, angles)
        if delta:
            r = r + delta * np.hypot(r, dr) / r
        return r * np.cos(angles), r * np.sin(angles)

    def describe(self, theta, pixel_nm=21.0):
        a, b, p, phi, coeffs, _ = self.unpack(theta)
        fw = self.blur_fwhm_px(theta)
        ex, ey = self.extents_px(theta)
        return dict(cd_nm=self.cd_px(theta) * pixel_nm, extent_x_nm=ex * pixel_nm, extent_y_nm=ey * pixel_nm,
                    superellipse=dict(a_nm=a * pixel_nm, b_nm=b * pixel_nm, p=p, phi_deg=math.degrees(phi)),
                    harmonics={f"k{k}": [float(coeffs[i, 0]), float(coeffs[i, 1])] for i, k in enumerate(self.ks)},
                    blur_fwhm_nm=[fw[0] * pixel_nm, fw[1] * pixel_nm],
                    blur_rms_nm=[v * pixel_nm for v in self.blur_sigma_px(theta)], blur_family=self.blur)


def stage_map(template, stage):
    """Indices of the full parameter vector that a stage fits, and ties (index -> index it copies)."""
    t = template
    j = t.n_shape
    if stage == "round":
        free = [0, j]
        ties = {1: 0}
        if t.blur == "gauss2":
            free += [j + 2, j + 4]
            ties.update({j + 1: j, j + 3: j + 2})
        elif t.blur == "lens":
            free += [j + 2]
            ties.update({j + 1: j})
        else:
            ties.update({j + 1: j})
        return free, ties
    if stage == "super":
        return [0, 1, 2, 3] + list(range(j, t.n_params)), {}
    return list(range(t.n_params)), {}


def fit_stages(template, render_for, y, sw, theta0, stages=("round", "super", "full"), gate=None, max_nfev=80,
               grid=True, log=None, pixel_nm=21.0, unit="nm"):
    """Least squares of the template parameters on data ``y`` (flat) with sqrt-weights ``sw``, model
    scale * render_for(theta) + const (scale and const solved linearly for every theta), in stages: a round outline
    with an isotropic blur (from a grid of starts over size and blur width), then the superellipse with the full
    blur, then the harmonics. ``gate`` = {stage: chi2 limit the previous stage must reach}. Returns
    (theta, scale, const, chi2)."""
    from scipy.optimize import least_squares

    t = template
    lo_full, hi_full = t.bounds()
    theta = np.clip(np.asarray(theta0, float), lo_full + 1e-6, hi_full - 1e-6)
    w = sw * sw
    wsum = float(w.sum())

    def linear(model):
        mw = float((w * model).sum()) / wsum
        yw = float((w * y).sum()) / wsum
        var = float((w * (model - mw) ** 2).sum())
        scale = float((w * (model - mw) * (y - yw)).sum()) / var if var > 0 else 1.0
        return scale, yw - scale * mw

    def resid_for(theta):
        model = render_for(theta)
        scale, const = linear(model)
        return sw * (scale * model + const - y), scale, const

    chi2 = float("nan")
    scale, const = 1.0, 0.0
    previous = None
    for stage in stages:
        if gate and stage in gate and previous is not None and previous > gate[stage]:
            if log:
                log(f"    shape stage {stage} skipped (chi2 {previous:.2f} > {gate[stage]})")
            break
        free, ties = stage_map(t, stage)
        base = theta.copy()
        if stage == "round":
            base[1] = base[0]
            base[2] = 0.0
            base[3] = 0.0
            base[4:t.n_shape] = 0.0
            if grid:
                best = None
                a0 = math.exp(base[0])
                for a in a0 * np.linspace(0.4, 1.8, 15):
                    for s in np.linspace(0.45, 2.2, 12):
                        q = base.copy()
                        q[0] = q[1] = math.log(a)
                        q[t.n_shape] = q[t.n_shape + 1] = math.log(s)
                        if t.blur == "gauss2":
                            q[t.n_shape + 2] = q[t.n_shape + 3] = math.log(2 * s)
                            q[t.n_shape + 4] = math.log(0.05 / 0.95)
                        elif t.blur == "lens":
                            q[t.n_shape + 2] = math.log(0.5)
                        qc = np.clip(q, lo_full + 1e-6, hi_full - 1e-6)
                        r, _, _ = resid_for(qc)
                        chi = float(np.mean(r * r))
                        if best is None or chi < best[0]:
                            best = (chi, qc)
                base = best[1]

        def expand(q, base=base, free=free, ties=ties):
            th = base.copy()
            th[free] = q
            for k, src in ties.items():
                th[k] = th[src]
            return th

        def fun(q):
            return resid_for(expand(q))[0]

        lo, hi = lo_full[free], hi_full[free]
        q0 = np.clip(base[free], lo + 1e-6, hi - 1e-6)
        r = least_squares(fun, q0, bounds=(lo, hi), method="trf", x_scale="jac", xtol=1e-9, ftol=1e-10,
                          max_nfev=max_nfev)
        theta = expand(r.x)
        _, scale, const = resid_for(theta)
        chi2 = float(np.mean(r.fun ** 2))
        previous = chi2
        if log:
            log(f"    shape stage {stage}: chi2 {chi2:.3f}, CD {t.cd_px(theta) * pixel_nm:.2f} {unit}, scale {scale:.4f}, "
                f"evaluations {r.nfev}")
    return theta, scale, const, chi2
