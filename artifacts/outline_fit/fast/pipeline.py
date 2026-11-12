"""Scratch: the one-pass outline fit, end to end on the GPU (gpufit.py + shapefit.py), one page or a batch.

    A. seeds -> plain stack -> pair map -> page fit (grid of starts, round and superellipse stages)   -> theta_0
    B. hole pass, size off: centre, brightness, level of every hole against the complete page model with theta_0
       (the seeds' centres are pulled towards near neighbours; stacking with them biases the page's size)
    C. plain stack again with B's centres -> pair map -> page fit from theta_0, every parameter free -> theta_1
    D. (per-hole sizes wanted) hole pass, size on, with theta_1: every hole's edge offset -> its CD

``level=1`` stops after C (the page's outline CD, x / y widths, blur); ``level=2`` adds D.
The page fits of a batch run in lockstep (one device evaluation serves all its pages); everything else is per page.
"""
from __future__ import annotations

import time

import cupy as cp
import numpy as np

from gpufit import GpuPage, make_fitter, sync

FITTERS = {}
TEMPLATES = {}


def lib_template(family):
    from subpx._outline.model import Template

    if family not in TEMPLATES:
        TEMPLATES[family] = Template(upsample=8, size_px=24, blur=family)  # parameter bookkeeping only
    return TEMPLATES[family]


def fitter_for(pg, family):
    key = (family, pg.R, pg.U, pg.f_max, pg.L)
    if key not in FITTERS:
        FITTERS[key] = make_fitter(family, pg.R, U=pg.U, box_px=pg.L, f_max=pg.f_max, lib_template=pg.t)
    return FITTERS[key]


def page_fits(pages, family, theta0=None, stages=("round", "super", "full"), grid=True, max_iter=(12, 20, 30),
              ftol=1e-7):
    """The page fit of several pages at once (grouped by window size). Returns lists (theta, scale, const, info)."""
    n = len(pages)
    out = [None] * n
    groups = {}
    for i, pg in enumerate(pages):
        groups.setdefault(fitter_for(pg, family), []).append(i)
    for sf, members in groups.items():
        t0 = time.perf_counter()
        gam, ys, ws, th = [], [], [], []
        for i in members:
            pg = pages[i]
            gam.append(pg.pair_map(sf))
            y, w = pg.stack_data(sf)
            ys.append(y)
            ws.append(w)
            th.append(pg.start_theta() if theta0 is None else theta0[i])
        t1 = time.perf_counter()
        theta, scale, const, cost, info = sf.fit(np.stack(th), cp.stack(gam), cp.stack(ys), cp.stack(ws), stages=stages,
                                                 grid=grid, max_iter=max_iter, ftol=ftol)
        sync()
        dt = (time.perf_counter() - t1) / len(members)
        bins = int((ws[0] > 0).sum())
        for k, i in enumerate(members):
            pages[i].times["shape"] = pages[i].times.get("shape", 0.0) + dt
            out[i] = (theta[k], float(scale[k]), float(const[k]), dict(chi2=float(cost[k]), bins=bins, stages=info))
    return out


def page_fit(pg, family, **kw):
    if "theta0" in kw and kw["theta0"] is not None:
        kw["theta0"] = [kw["theta0"]]
    return page_fits([pg], family, **kw)[0]


def describe(pg, theta, pixel_nm):
    d = pg.t.describe(theta, pixel_nm)
    return dict(cd=d["cd_nm"], blur=[float(v) for v in d["blur_fwhm_nm"]], p=d["superellipse"]["p"],
                extent=(d["extent_x_nm"], d["extent_y_nm"]))


def fit_pages(images, *, family="gauss2", level=2, sweeps_b=2, sweeps_d=3, pixel_nm=21.0, seeds=None, weighting="raw",
              tt="float", f_max=0.6, U=4, first_stages=("round", "super"), first_iter=(10, 12), relax=1.3, log=None):
    """A batch of pages. Returns a list of (GpuPage, summary dict, per-hole results or None)."""
    t_all = time.perf_counter()
    phase = {}

    def mark(name, t0):
        sync()
        phase[name] = phase.get(name, 0.0) + time.perf_counter() - t0
        return time.perf_counter()

    t = lib_template(family)
    pages = []
    tp = time.perf_counter()
    for k, image in enumerate(images):
        pg = GpuPage(image, blur=family, U=U, f_max=f_max, seeds=None if seeds is None else seeds[k], weighting=weighting,
                     tt=tt, pixel_nm=pixel_nm, lib_template=t)
        pg.build_stack(recentre=False)
        pages.append(pg)
    tp = mark("A stack", tp)
    fits = page_fits(pages, family, stages=first_stages, max_iter=first_iter, ftol=1e-6)  # A
    tp = mark("A page fit", tp)
    outs = []
    for pg, (theta, scale, const, fa) in zip(pages, fits):
        out = dict(holes=pg.n, spacing=pg.spacing, R=pg.R, first=dict(describe(pg, theta, pixel_nm), **fa))
        pg.set_template(theta, scale, const)  # B
        out["sweeps_b"] = pg.fit_holes(sweeps=sweeps_b, size=False)
        if level >= 2:
            pg.fit_noise()  # the per-hole pass D weights its pixels by 1 / (v0 + v1 level)
        pg.restack(pg.good_mask(), scale)
        outs.append(out)
    tp = mark("B hole pass + stack", tp)
    fits = page_fits(pages, family, theta0=[f[0] for f in fits], stages=("full",), grid=False, max_iter=(30,))  # C
    tp = mark("C page fit", tp)
    results = []
    for pg, out, (theta, scale, const, fc) in zip(pages, outs, fits):
        out.update(describe(pg, theta, pixel_nm), **fc, stacked=int(pg.use.sum()), theta=theta, scale=scale, const=const)
        res = None
        if level >= 2:  # D
            pg.set_template(theta, scale, const, keep_holes=True)
            out["sweeps_d"] = pg.fit_holes(sweeps=sweeps_d, size=True, relax=relax)
            res = pg.results()
            g = res["good"] & ~res["at_clip"]
            out.update(cd_mean=float(res["cd_px"][g].mean() * pixel_nm), cd_sd=float(res["cd_px"][g].std() * pixel_nm),
                       delta_mean=res["delta_mean"] * pixel_nm, good=int(g.sum()), at_clip=int(res["at_clip"].sum()),
                       Rr=pg.Rr)
        out["times"] = dict(pg.times)
        results.append((pg, out, res))
    tp = mark("D hole pass", tp)
    dt = time.perf_counter() - t_all
    for _, out, _ in results:
        out["seconds"] = dt / len(images)
        out["phase"] = {k: v / len(images) for k, v in phase.items()}
    return results


def fit_page(image, **kw):
    """One page: (GpuPage, summary dict, per-hole results or None)."""
    if kw.get("seeds") is not None:
        kw["seeds"] = [kw["seeds"]]
    return fit_pages([image], **kw)[0]
