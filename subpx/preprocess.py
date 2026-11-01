"""Preprocessing for images of repeated features (contact-hole arrays): denoising and the super-resolved average
feature.

From the ACTIS preprocessing proposal (``docs/actis_preprocessing_proposal.html``, sections "Noise reduction"
and "Resolution improvement"):

- :func:`smooth_gaussian`: a fixed light Gaussian smoothing, the same on every image.
- :func:`denoise_wiener`: a linear filter that keeps the frequencies where the image holds more power than the
  noise and removes the rest; the noise level is picked from the image itself by Noise2Self.
- :func:`denoise_mean_hole`: every feature is its own brightness and background times the mean feature plus a few
  main differences (principal components); what those do not describe is removed as noise.
- :func:`average_hole`: every feature stacked at its own subpixel position into a finer, almost noise-free
  average feature.

Every function takes any finite 2-D image and needs NumPy only; processed images are float32 of the input's
shape. Centres are (N, 2) ``[x, y]`` pixel positions, pixel centres at integers.

Noise2Self (Batson & Royer 2019) picks a setting without a clean image: each pixel is predicted from the other
pixels only, and the setting with the smallest prediction error wins. It assumes each pixel's noise is independent
of its neighbours'; noise shared along rows or columns makes it smooth too little.
"""

from __future__ import annotations

import numpy as np

from .types import AverageHoleResult, PreprocessResult

__all__ = ["smooth_gaussian", "denoise_wiener", "denoise_mean_hole", "average_hole", "median_spacing",
           "gaussian_kernel"]


# --------------------------------------------------------------------------------------------- helpers
def _image(image):
    data = np.asarray(image)
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("Expected a 2-D image of at least 2 x 2 pixels.")
    if data.dtype.kind not in "buif":
        raise ValueError("The image must hold real numbers.")
    data = data.astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError("The image must be finite (no NaN or infinity).")
    return data


def _centers(centers_xy, shape):
    """Finite centres inside the image and their indices in the input."""
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    h, w = shape
    keep = (np.isfinite(centers).all(axis=1) & (centers[:, 0] > -0.5) & (centers[:, 0] < w - 0.5)
            & (centers[:, 1] > -0.5) & (centers[:, 1] < h - 0.5))
    return centers[keep], np.flatnonzero(keep)


def gaussian_kernel(sigma, truncate=4.0):
    """Normalised 1-D Gaussian of standard deviation ``sigma`` px, cut at ``truncate`` sigma (as SciPy)."""
    radius = int(truncate * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return kernel / kernel.sum()


def _convolve(data, kernel, axis, mode):
    """Filter along one axis with a symmetric kernel; borders extended by ``np.pad`` mode ``mode``."""
    radius = len(kernel) // 2
    if radius == 0:
        return data * kernel[0]
    width = [(0, 0)] * data.ndim
    width[axis] = (radius, radius)
    padded = np.pad(data, width, mode=mode)
    out = np.zeros(data.shape, dtype=np.float64)
    index = [slice(None)] * data.ndim
    for i, weight in enumerate(kernel):
        index[axis] = slice(i, i + data.shape[axis])
        out += weight * padded[tuple(index)]
    return out


def median_spacing(centers_xy, *, sample=2000, chunk=256):
    """Median distance from a centre to its nearest neighbour (NumPy only; up to ``sample`` centres probed)."""
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    centers = centers[np.isfinite(centers).all(axis=1)]
    if len(centers) < 2:
        return float("nan")
    probes = centers[np.linspace(0, len(centers) - 1, min(sample, len(centers))).astype(int)]
    nearest = []
    for start in range(0, len(probes), chunk):
        part = probes[start:start + chunk]
        distance = np.hypot(part[:, None, 0] - centers[None, :, 0], part[:, None, 1] - centers[None, :, 1])
        distance[distance == 0] = np.inf  # the probe itself (and exact duplicates)
        nearest.append(distance.min(axis=1))
    nearest = np.concatenate(nearest)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.median(nearest)) if len(nearest) else float("nan")


def _fit_level(values, template, weight):
    """Per row of ``values``: offset c and gain b of the weighted least-squares fit values ~ c + b * template."""
    total = weight.sum()
    t_sum = (weight * template).sum()
    tt_sum = (weight * template * template).sum()
    v_sum = values @ weight
    tv_sum = values @ (weight * template)
    determinant = total * tt_sum - t_sum ** 2
    if determinant <= 0:
        return np.zeros(len(values)), np.ones(len(values))
    gain = (total * tv_sum - t_sum * v_sum) / determinant
    offset = (v_sum - gain * t_sum) / total
    return offset, gain


# ----------------------------------------------------------------------------------------- smoothing
def smooth_gaussian(image, sigma=0.6, *, truncate=4.0):
    """Fixed light smoothing: a Gaussian of ``sigma`` px (the proposal: 0.5-0.7 px on every page).

    Linear and the same on every image, so a brighter or dimmer image is smoothed alike. Borders are mirrored
    (SciPy's ``mode='reflect'``); ``sigma=0`` returns the image unchanged.
    """
    data = _image(image)
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be a finite number >= 0.")
    radius = 0
    out = data
    if sigma > 0:
        kernel = gaussian_kernel(sigma, truncate)
        radius = len(kernel) // 2
        out = _convolve(_convolve(data, kernel, 1, "symmetric"), kernel, 0, "symmetric")
    return PreprocessResult(out.astype(np.float32), "gaussian", dict(sigma=sigma, truncate=float(truncate)),
                            dict(kernel_radius=radius, removed_std=float(np.std(data - out))))


# ------------------------------------------------------------------------------------------- Wiener
def denoise_wiener(image, noise="auto", *, smoothing=2.0, levels=25):
    """Noise-matched linear (Wiener) filter, its noise level picked by Noise2Self.

    The image (mirrored at its borders) is taken to the Fourier domain. Its power spectrum, smoothed over
    ``smoothing`` frequency bins, is P. Each frequency is kept by the factor max(P - N, 0) / P, where N is the
    noise power (the noise variance, for noise independent from pixel to pixel): frequencies where the image holds
    much more power than the noise (the lattice's repeats, the features' shape) pass, frequencies that hold
    noise only are removed. The filter is linear on the image; its shape comes from the image's own spectrum.

    ``noise`` is the noise standard deviation in grey levels, or ``'auto'``: the level at which the filter, with
    each pixel's own weight moved to the mean of its 8 neighbours, predicts the image best (Noise2Self; computed
    exactly in the Fourier domain). ``meta['loss']`` holds the curve as (noise, root-mean-square prediction
    error); a minimum at the end of the searched range is flagged in ``meta['at_edge']``.
    """
    data = _image(image)
    h, w = data.shape
    mean = float(data.mean())
    extended = np.pad(data - mean, ((0, h), (0, w)), mode="symmetric")
    n = extended.size
    spectrum = np.fft.rfft2(extended)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / n  # mean over the full spectrum = pixel variance
    multiplicity = np.full(power.shape[1], 2.0)  # each rfft column stands for itself and its mirror
    multiplicity[0] = multiplicity[-1] = 1.0  # zero and Nyquist frequency (the extended width is even)
    smooth = power
    if smoothing and smoothing > 0:
        kernel = gaussian_kernel(float(smoothing))
        smooth = _convolve(_convolve(power, kernel, 0, "wrap"), kernel, 1, "reflect")
    inverse = 1.0 / np.maximum(smooth, np.finfo(float).tiny)
    del smooth

    def response(variance):
        gain = 1.0 - variance * inverse
        np.clip(gain, 0.0, 1.0, out=gain)
        gain[0, 0] = 1.0
        return gain

    # Noise2Self's J-invariant version of the filter: each pixel's own weight goes to the mean of its 8 neighbours
    # (the "donut"), so a pixel is predicted from the others only. Its mean squared error against the image is
    # computed exactly with Parseval's theorem.
    fy = np.fft.fftfreq(extended.shape[0])[:, None]
    fx = np.fft.rfftfreq(extended.shape[1])[None, :]
    cy, cx = np.cos(2 * np.pi * fy), np.cos(2 * np.pi * fx)
    donut_gap = (2 * cx + 2 * cy + 4 * cx * cy) / 8.0 - 1.0
    weighted_power = power * multiplicity

    def loss(variance):
        gain = response(variance)
        center = float(gain.sum(axis=0) @ multiplicity) / n  # the filter's own-pixel weight
        error = gain - 1.0
        error += center * donut_gap
        error *= error
        return float(np.vdot(error, weighted_power)) / n, center

    pixel_variance = float(data.var())
    curve, at_edge = [], False
    if isinstance(noise, str):
        if noise != "auto":
            raise ValueError("noise must be 'auto' or a standard deviation in grey levels.")
        if pixel_variance <= 0:
            variance = 0.0
        else:
            grid = pixel_variance * np.geomspace(1e-4, 1.0, int(levels))
            values = [loss(v)[0] for v in grid]
            best = int(np.argmin(values))
            at_edge = best in (0, len(grid) - 1)
            low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
            fine = np.geomspace(low, high, 7) if high > low else np.array([grid[best]])
            fine_values = [loss(v)[0] for v in fine]
            variance = float(fine[int(np.argmin(fine_values))])
            curve = sorted(zip(np.sqrt(np.concatenate((grid, fine))).tolist(),
                               np.sqrt(np.asarray(values + fine_values)).tolist()))
    else:
        sigma = float(noise)
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("noise must be 'auto' or a finite standard deviation >= 0.")
        variance = sigma ** 2
    gain = response(variance)
    score, center = loss(variance)
    out = np.fft.irfft2(spectrum * gain, s=extended.shape)[:h, :w] + mean
    kept = float(((gain > 0.5) * multiplicity).sum() / (multiplicity.sum() * gain.shape[0]))  # share passed
    meta = dict(noise=float(np.sqrt(variance)), loss=[(float(s), float(v)) for s, v in curve], at_edge=at_edge,
                center_weight=center, kept_share=kept, removed_std=float(np.std(data - out)),
                self_supervised_error=float(np.sqrt(score)) if np.isfinite(score) else float("nan"))
    parameters = dict(noise=float(np.sqrt(variance)), noise_setting=noise if isinstance(noise, str) else float(noise),
                      smoothing=float(smoothing))
    return PreprocessResult(out.astype(np.float32), "wiener", parameters, meta)


# ---------------------------------------------------------------------------------------- mean hole
def _noise2self_components(data, vectors, limit):
    """Number of components (0..limit) whose leave-one-pixel-out reconstruction error is smallest, and the curve.

    For a projection onto components V, the prediction of pixel i from the box's other pixels has the error
    r_i / (1 - P_ii), with r the reconstruction residual and P_ii the projection's own-pixel weight.
    """
    residual = data.copy()
    own = np.zeros(data.shape[1])
    losses = [float(np.mean(residual ** 2))]
    for k in range(limit):
        vector = vectors[:, k]
        residual -= np.outer(residual @ vector, vector)
        own += vector ** 2
        scale = np.where(own < 1 - 1e-9, 1.0 / (1.0 - np.minimum(own, 1 - 1e-9)), 0.0)
        losses.append(float(np.mean((residual * scale) ** 2)))
    return int(np.argmin(losses)), np.sqrt(losses).tolist()


def denoise_mean_hole(image, centers_xy, *, components="auto", radius="auto", normalize=True, max_components=60):
    """The proposal's "mean hole plus main differences" denoiser.

    Each feature gets a square box of ``radius`` px around its rounded centre (``'auto'``: 0.75 of the median
    nearest-neighbour spacing, so the box holds the feature and its own surroundings). With ``normalize`` each
    box's brightness and background are fitted to the mean box and divided out, so brightness flicker does not use
    up components and a brighter feature passes through scaled exactly. The boxes, weighted by a smooth taper
    (1 at the centre, 0 at radius + 1 px), give the mean box and its principal components: the few ways the
    features really differ (sub-pixel shift, size, shape, neighbours). Each box is replaced by the mean plus its
    first ``components`` components (``'auto'``: picked by Noise2Self, ``meta['loss']`` the root-mean-square
    leave-one-pixel-out error for 0, 1, 2, ... components), its own brightness and background put back, and the
    boxes are blended with the taper as weight. Pixels in no box are left unchanged.

    No feature shape is assumed; the components are learned from the image. A feature unlike the others (a
    defect, a different shape in one region) is pulled towards the common ones: check what was removed
    (input minus output) before trusting per-feature maps.
    """
    data = _image(image)
    h, w = data.shape
    centers, index = _centers(centers_xy, data.shape)
    if len(centers) < 3:
        raise ValueError("The mean-hole method needs at least 3 feature centres inside the image.")
    spacing = median_spacing(centers)
    if isinstance(radius, str):
        if radius != "auto":
            raise ValueError("radius must be 'auto' or a number of pixels.")
        if not np.isfinite(spacing):
            raise ValueError("Cannot find the feature spacing: give radius in pixels.")
        box_radius = int(np.ceil(0.75 * spacing))
    else:
        box_radius = int(np.ceil(float(radius)))
    box_radius = max(box_radius, 1)
    side = 2 * box_radius + 1
    oy, ox = np.mgrid[-box_radius:box_radius + 1, -box_radius:box_radius + 1]
    taper = (np.cos(0.5 * np.pi * np.minimum(np.hypot(ox, oy) / (box_radius + 1), 1.0)) ** 2).ravel()
    taper[taper < 1e-3] = 0.0  # exactly out: a pixel of almost no weight would divide rounding noise by ~0
    rounded = np.rint(centers).astype(np.int64)
    rows = rounded[:, 1, None] + box_radius + oy.ravel()  # in the mirrored, padded image
    cols = rounded[:, 0, None] + box_radius + ox.ravel()
    padded = np.pad(data, box_radius, mode="symmetric")
    boxes = padded[rows, cols]
    whole = ((rounded[:, 0] >= box_radius) & (rounded[:, 0] < w - box_radius)
             & (rounded[:, 1] >= box_radius) & (rounded[:, 1] < h - box_radius))
    fit = whole if whole.sum() >= 3 else np.ones(len(boxes), bool)
    offset, gain = np.zeros(len(boxes)), np.ones(len(boxes))
    usable = np.ones(len(boxes), bool)
    if normalize:
        template = boxes[fit].mean(axis=0)
        for _ in range(3):
            offset, gain = _fit_level(boxes, template, taper ** 2)
            typical = float(np.median(gain[fit]))
            usable = np.isfinite(gain) & (gain > 0.1 * typical) if typical > 0 else np.zeros(len(gain), bool)
            if (usable & fit).sum() < 3:
                raise ValueError("The features' brightness could not be fitted: try normalize=False.")
            chosen = usable & fit
            template = ((boxes[chosen] - offset[chosen, None]) / gain[chosen, None]).mean(axis=0)
        safe = np.where(usable, gain, 1.0)
        boxes = (boxes - np.where(usable, offset, 0.0)[:, None]) / safe[:, None]
        offset, gain = np.where(usable, offset, 0.0), safe
    chosen = usable & fit
    mean = boxes[chosen].mean(axis=0)
    weighted = (boxes - mean) * taper
    del boxes
    fitted = weighted[chosen]
    covariance = fitted.T @ fitted / len(fitted)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    limit = int(min(max_components, side * side - 1, max(len(fitted) // 10, 1)))
    curve = []
    if isinstance(components, str):
        if components != "auto":
            raise ValueError("components must be 'auto' or a whole number.")
        count, curve = _noise2self_components(fitted, vectors, limit)
    else:
        count = int(components)
        if count < 0:
            raise ValueError("components must be >= 0.")
        count = min(count, limit)
    basis = vectors[:, :count]
    rebuilt = (weighted @ basis) @ basis.T * (taper > 0)
    del weighted
    # Blend: each box contributes taper * (its own reconstruction); pixels covered by no box keep their value.
    contribution = taper * (offset[:, None] + gain[:, None] * mean) + gain[:, None] * rebuilt
    del rebuilt
    width = w + 2 * box_radius
    flat = (rows * width + cols)[usable].ravel()
    size = (h + 2 * box_radius) * width
    numerator = np.bincount(flat, weights=contribution[usable].ravel(), minlength=size)
    denominator = np.bincount(flat, weights=np.broadcast_to(taper, (usable.sum(), taper.size)).ravel(), minlength=size)
    crop = (slice(box_radius, box_radius + h), slice(box_radius, box_radius + w))
    numerator = numerator.reshape(-1, width)[crop]
    denominator = denominator.reshape(-1, width)[crop]
    covered = denominator > 0
    out = data.copy()
    out[covered] = numerator[covered] / denominator[covered]
    brightness = np.full(len(np.asarray(centers_xy).reshape(-1, 2)), np.nan)
    background = brightness.copy()
    if normalize:
        brightness[index[usable]] = gain[usable] / np.median(gain[usable])
        background[index[usable]] = offset[usable]
    parameters = dict(components=count, components_setting=components, radius=box_radius,
                      radius_setting=radius, normalize=bool(normalize))
    meta = dict(components=count, loss=curve, component_variance=eigenvalues[:max(limit, count)].tolist(),
                spacing=spacing, box_radius=box_radius, boxes=int(usable.sum()), fitted=int(chosen.sum()),
                skipped=int(len(brightness) - usable.sum()),
                covered_share=float(covered.mean()), removed_std=float(np.std((data - out)[covered]))
                if covered.any() else 0.0, brightness=brightness, background=background, limit=limit)
    return PreprocessResult(out.astype(np.float32), "mean_hole", parameters, meta)


# ------------------------------------------------------------------------------------- average hole
def average_hole(image, centers_xy, *, upsample=4, radius="auto", normalize=True, iterations=4):
    """The super-resolved average feature: every feature's pixels placed at their offset from its own subpixel
    centre on a grid ``upsample`` times finer, and averaged per bin.

    ``radius`` (px, ``'auto'``: the median nearest-neighbour spacing) sets the half-width of the square that is
    stacked. With ``normalize`` each feature's brightness and background are fitted to the average (``iterations``
    rounds) and matched to the population's mean before stacking, so brightness flicker does not blur the result.
    Bins no pixel fell into are NaN (``counts`` 0): with a pitch close to a whole number of pixels all features
    share one subpixel position and many bins stay empty. Centre errors blur the average.
    """
    data = _image(image)
    h, w = data.shape
    centers, _ = _centers(centers_xy, data.shape)
    upsample = int(upsample)
    if upsample < 1:
        raise ValueError("upsample must be a whole number >= 1.")
    if not len(centers):
        raise ValueError("No feature centres inside the image.")
    spacing = median_spacing(centers)
    if isinstance(radius, str):
        if radius != "auto":
            raise ValueError("radius must be 'auto' or a number of pixels.")
        if not np.isfinite(spacing):
            raise ValueError("Cannot find the feature spacing: give radius in pixels.")
        radius = spacing
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive number of pixels.")
    half = int(np.ceil(radius * upsample))
    size = 2 * half + 1
    reach = int(np.ceil(radius)) + 1
    oy, ox = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    rounded = np.rint(centers).astype(np.int64)
    px = rounded[:, 0, None] + ox.ravel()
    py = rounded[:, 1, None] + oy.ravel()
    bx = np.rint((px - centers[:, 0, None]) * upsample).astype(np.int64) + half
    by = np.rint((py - centers[:, 1, None]) * upsample).astype(np.int64) + half
    keep = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (bx >= 0) & (bx < size) & (by >= 0) & (by < size)
    holes = np.broadcast_to(np.arange(len(centers))[:, None], px.shape)[keep]
    values = data[py[keep], px[keep]]
    bins = (by * size + bx)[keep]

    def stack(samples, members, smooth=False):
        sums = np.bincount(bins[members], weights=samples[members], minlength=size * size)
        counts = np.bincount(bins[members], minlength=size * size).astype(float)
        if smooth:  # mixes neighbouring bins, i.e. features of different subpixel phases
            kernel = gaussian_kernel(1.0)
            sums = _convolve(_convolve(sums.reshape(size, size), kernel, 0, "symmetric"), kernel, 1, "symmetric")
            counts = _convolve(_convolve(counts.reshape(size, size), kernel, 0, "symmetric"), kernel, 1, "symmetric")
            sums, counts = sums.ravel(), counts.ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 1e-9, sums / np.maximum(counts, 1e-9), np.nan), counts

    members = np.ones(len(values), bool)
    samples = values
    average, counts = stack(samples, members)
    good = np.ones(len(centers), bool)
    gain, offset = np.ones(len(centers)), np.zeros(len(centers))
    if normalize:
        # Each feature's pixels fall only into the bins of its own subpixel phase, so a fit against the plain
        # average would hand every phase group its own mean brightness back. The fit uses the average smoothed
        # over neighbouring bins (1 bin = 1/upsample px) instead, which mixes the phase groups.
        template_all, _ = stack(samples, members, smooth=True)
        for _ in range(int(iterations)):
            template = template_all[bins]
            valid = np.isfinite(template)
            n = np.bincount(holes[valid], minlength=len(centers)).astype(float)
            t1 = np.bincount(holes[valid], weights=template[valid], minlength=len(centers))
            t2 = np.bincount(holes[valid], weights=template[valid] ** 2, minlength=len(centers))
            v1 = np.bincount(holes[valid], weights=values[valid], minlength=len(centers))
            tv = np.bincount(holes[valid], weights=template[valid] * values[valid], minlength=len(centers))
            determinant = n * t2 - t1 ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                gain = (n * tv - t1 * v1) / determinant
                offset = (v1 - gain * t1) / n
            typical = float(np.nanmedian(gain))
            good = (n >= 3) & np.isfinite(gain) & (gain > 0.1 * typical) & (determinant > 0)
            if not good.any():
                raise ValueError("The features' brightness could not be fitted: try normalize=False.")
            mean_gain, mean_offset = float(gain[good].mean()), float(offset[good].mean())
            safe_gain = np.where(good, gain, 1.0)
            samples = (values - np.where(good, offset, 0.0)[holes]) / safe_gain[holes] * mean_gain + mean_offset
            members = good[holes] & valid
            template_all, _ = stack(samples, members, smooth=True)
        average, counts = stack(samples, members)
    offsets = (np.arange(size) - half) / upsample
    within = (np.abs(offsets)[:, None] <= radius) & (np.abs(offsets)[None, :] <= radius)
    counts = counts.astype(np.int64)
    filled = counts.reshape(size, size)[within]
    meta = dict(holes=int(good.sum()), spacing=spacing, radius=radius, normalize=bool(normalize),
                empty_bins=int((filled == 0).sum()), fewest_samples=int(filled.min()) if filled.size else 0,
                brightness_spread=float(np.std(gain[good]) / np.mean(gain[good])) if normalize and good.any() else
                float("nan"),
                background_spread=float(np.std(offset[good])) if normalize and good.any() else float("nan"))
    return AverageHoleResult(average.reshape(size, size), counts.reshape(size, size), offsets, upsample, meta)
