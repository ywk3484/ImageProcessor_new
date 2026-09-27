"""Gaussian and radial-profile CD measurements on isolated cells."""

from __future__ import annotations

import numpy as np

from ..types import CDMethodResult


GEOMETRY_METRICS = (
    "area_px2", "cd_equivalent_px", "cd_x_px", "cd_y_px", "perimeter_px",
)
FIT_METRICS = (
    "sigma_major_px", "sigma_minor_px", "fwhm_major_px", "fwhm_minor_px",
    "angle_deg", "log_fit_rmse",
)


def _inside(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    return map_coordinates(
        mask.astype(float), points.T[::-1], order=1, mode="constant", cval=0,
        prefilter=False,
    ) >= 1 - 1e-8


def _background(roi: np.ndarray, mask: np.ndarray) -> float:
    from scipy.ndimage import binary_erosion

    border = mask & ~binary_erosion(mask, border_value=0)
    return float(np.percentile(roi[border], 10))


def _geometry(contour: np.ndarray) -> dict[str, float]:
    # Translate first to avoid cancellation for cells far from image origin.
    vertices = contour - contour[0]
    area = abs(np.sum(
        vertices[:-1, 0] * vertices[1:, 1] - vertices[1:, 0] * vertices[:-1, 1]
    )) / 2
    return {
        "area_px2": float(area),
        "cd_equivalent_px": float(2 * np.sqrt(area / np.pi)),
        "cd_x_px": float(np.ptp(contour[:, 0])),
        "cd_y_px": float(np.ptp(contour[:, 1])),
        "perimeter_px": float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum()),
    }


def _logquad(
    roi: np.ndarray, mask: np.ndarray, seed: np.ndarray,
    background: float, angles: np.ndarray, fit_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str]:
    """Recover covariance from the negative Hessian of log(I - background)."""
    signal = roi - background
    peak = float(signal[mask].max())
    missing = np.full((len(angles) + 1, 2), np.nan)
    extra = {"background": background, "peak_signal": peak}
    if peak <= 0:
        return seed, missing, extra, "no_signal"
    fit_mask = mask & (signal > fit_fraction * peak)
    yy, xx = np.nonzero(fit_mask)
    if len(xx) < 6:
        return seed, missing, extra, "insufficient_fit_pixels"
    x, y = xx - seed[0], yy - seed[1]
    design = np.column_stack((x * x, y * y, x * y, x, y, np.ones(len(x))))
    z = np.log(signal[fit_mask])
    # Intensity weights reduce the influence of noisy log-transformed tails.
    weight = signal[fit_mask] / peak
    coef, _, rank, _ = np.linalg.lstsq(design * weight[:, None], z * weight, rcond=None)
    if rank < 6:
        return seed, missing, extra, "singular_fit"
    a, b, c, d, e, _ = coef
    precision = -np.array([[2 * a, c], [c, 2 * b]])
    eigenvalues, axes = np.linalg.eigh(precision)
    if eigenvalues[0] <= 1e-10 or eigenvalues[-1] / eigenvalues[0] > 1e8:
        return seed, missing, extra, "non_gaussian_fit"
    offset = np.linalg.solve(precision, [d, e])
    center = seed + offset
    sigma = 1 / np.sqrt(eigenvalues)  # major, minor
    k = np.sqrt(2 * np.log(2))
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    contour = center + (unit_circle * (k * sigma)) @ axes.T
    contour = np.vstack((contour, contour[0]))
    covariance = (axes * (sigma * sigma)) @ axes.T
    extra.update(
        sigma_major_px=float(sigma[0]), sigma_minor_px=float(sigma[1]),
        fwhm_major_px=float(2 * k * sigma[0]), fwhm_minor_px=float(2 * k * sigma[1]),
        angle_deg=float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0])) % 180),
        log_fit_rmse=float(np.sqrt(np.average((design @ coef - z) ** 2, weights=weight**2))),
    )
    if not _inside(np.vstack((center, contour)), mask).all():
        return center, contour, extra, "contour_outside_cell"
    extra.update(_geometry(contour))
    # Analytic ellipse area and projected widths avoid polygon sampling error.
    area = float(np.pi * k**2 * np.prod(sigma))
    extra.update(
        area_px2=area, cd_equivalent_px=float(2 * np.sqrt(area / np.pi)),
        cd_x_px=float(2 * k * np.sqrt(covariance[0, 0])),
        cd_y_px=float(2 * k * np.sqrt(covariance[1, 1])),
    )
    return center, contour, extra, "ok"


def _radial_profiles(
    roi: np.ndarray, mask: np.ndarray, background: float,
    angles: np.ndarray, radial_step: float, smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates

    signal = np.where(mask, roi - background, 0)
    # Retain the central signal while excluding low-amplitude background noise.
    weights = np.where(mask & (signal > 0.1 * signal.max()), signal, 0)
    yy, xx = np.indices(roi.shape)
    center = np.array([(weights * xx).sum(), (weights * yy).sum()]) / weights.sum()
    if smooth_sigma > 0:
        support = gaussian_filter(mask.astype(float), smooth_sigma, mode="constant")
        signal = gaussian_filter(signal, smooth_sigma, mode="constant") / np.maximum(support, 1e-12)
    radii = np.arange(0, np.hypot(*roi.shape) + radial_step, radial_step)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    points = center + directions[:, None, :] * radii[None, :, None]
    profiles = map_coordinates(signal, points.transpose(2, 0, 1)[::-1], order=3, mode="nearest")
    # Keep interpolation/gradient stencils away from the background-filled edge.
    safe = binary_erosion(mask, border_value=0)
    valid = _inside(points.reshape(-1, 2), safe).reshape(profiles.shape)
    valid = np.logical_and.accumulate(valid, axis=1)
    return center, radii, profiles, valid


def _radial_contour(
    method: str, center: np.ndarray, radii: np.ndarray, profiles: np.ndarray,
    valid: np.ndarray, angles: np.ndarray, background: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str]:
    n = len(angles)
    edge = np.full(n, np.nan)
    peak = float(profiles[0, 0])
    extra = {"background": background, "peak_signal": peak, "ray_coverage": 0.0}
    if peak > 0:
        if method == "halfmax":
            crossings = (profiles[:, :-1] >= peak / 2) & (profiles[:, 1:] < peak / 2)
            crossings &= valid[:, :-1] & valid[:, 1:]
            found = crossings.any(axis=1)
            ray = np.flatnonzero(found)
            j = np.argmax(crossings[found], axis=1)
            fraction = (profiles[ray, j] - peak / 2) / (profiles[ray, j] - profiles[ray, j + 1])
            edge[found] = radii[j] + fraction * (radii[1] - radii[0])
        else:
            step = radii[1] - radii[0]
            gradient = -np.gradient(profiles, step, axis=1)
            candidate = np.zeros_like(valid)
            candidate[:, 2:-2] = valid[:, :-4] & valid[:, 4:]
            candidate &= radii[None, :] >= max(0.35, 2 * step)
            score = np.where(candidate, gradient, -np.inf)
            j = np.argmax(score, axis=1)
            ray = np.arange(n)
            jm = np.maximum(j - 1, 0)
            jp = np.minimum(j + 1, len(radii) - 1)
            left, mid, right = gradient[ray, jm], gradient[ray, j], gradient[ray, jp]
            # A maximum at the search boundary is not a measured edge.
            found = candidate[ray, j] & candidate[ray, jm] & candidate[ray, jp]
            found &= (mid >= left) & (mid >= right) & (mid > 0.01 * peak)
            curvature = left - 2 * mid + right
            found &= curvature < -1e-12
            delta = np.zeros(n)
            delta[found] = 0.5 * (left[found] - right[found]) / curvature[found]
            edge[found] = radii[j[found]] + np.clip(delta[found], -0.5, 0.5) * step
    points = center + edge[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    contour = np.vstack((points, points[0]))
    extra["ray_coverage"] = float(np.isfinite(edge).mean())
    if not np.isfinite(edge).all():
        return center, contour, extra, "incomplete_rays"
    extra.update(_geometry(contour))
    return center, contour, extra, "ok"


def measure_cells(
    rois: np.ndarray, masks: np.ndarray, origins_xy: np.ndarray, seeds_xy: np.ndarray,
    *, methods: tuple[str, ...], n_angles: int, radial_step: float,
    smooth_sigma: float, fit_fraction: float,
) -> dict[str, CDMethodResult]:
    """Measure all methods on the same ordered cell stack."""
    n = len(rois)
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    results = {}
    for method in methods:
        names = GEOMETRY_METRICS + ("background", "peak_signal")
        names += FIT_METRICS if method == "logquad" else ("ray_coverage",)
        results[method] = CDMethodResult(
            method=method, centers_xy=np.full((n, 2), np.nan),
            contours_xy=np.full((n, n_angles + 1, 2), np.nan),
            valid=np.zeros(n, dtype=bool), status=np.full(n, "no_signal", dtype="U32"),
            metrics={name: np.full(n, np.nan) for name in names},
        )
    for i, (roi, mask, origin, seed) in enumerate(zip(rois, masks, origins_xy, seeds_xy)):
        if mask.sum() < 6:
            for result in results.values():
                result.status[i] = "insufficient_cell_pixels"
            continue
        background = _background(roi, mask)
        if roi[mask].max() <= background:
            continue
        radial = None
        for method, result in results.items():
            if method == "logquad":
                center, contour, metrics, status = _logquad(
                    roi, mask, seed - origin, background, angles, fit_fraction,
                )
            else:
                if radial is None:
                    radial = _radial_profiles(roi, mask, background, angles, radial_step, smooth_sigma)
                center, contour, metrics, status = _radial_contour(method, *radial, angles, background)
            result.centers_xy[i] = center + origin
            result.contours_xy[i] = contour + origin
            result.status[i] = status
            result.valid[i] = status == "ok"
            for name, value in metrics.items():
                result.metrics[name][i] = value
    return results
