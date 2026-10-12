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
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str, tuple | None]:
    """Recover covariance from the negative Hessian of log(I - background).

    The last value is ``(eigenvalues, axes)`` of the precision matrix when a
    contour was drawn, else None.
    """
    signal = roi - background
    peak = float(signal[mask].max())
    missing = np.full((len(angles) + 1, 2), np.nan)
    extra = {"background": background, "peak_signal": peak}
    if peak <= 0:
        return seed, missing, extra, "no_signal", None
    fit_mask = mask & (signal > fit_fraction * peak)
    yy, xx = np.nonzero(fit_mask)
    if len(xx) < 6:
        return seed, missing, extra, "insufficient_fit_pixels", None
    x, y = xx - seed[0], yy - seed[1]
    design = np.column_stack((x * x, y * y, x * y, x, y, np.ones(len(x))))
    z = np.log(signal[fit_mask])
    # Intensity weights reduce the influence of noisy log-transformed tails.
    weight = signal[fit_mask] / peak
    coef, _, rank, _ = np.linalg.lstsq(design * weight[:, None], z * weight, rcond=None)
    if rank < 6:
        return seed, missing, extra, "singular_fit", None
    a, b, c, d, e, _ = coef
    precision = -np.array([[2 * a, c], [c, 2 * b]])
    eigenvalues, axes = np.linalg.eigh(precision)
    if eigenvalues[0] <= 1e-10 or eigenvalues[-1] / eigenvalues[0] > 1e8:
        return seed, missing, extra, "non_gaussian_fit", None
    offset = np.linalg.solve(precision, [d, e])
    center = seed + offset
    contour, sigma, covariance, fit = _half_max_ellipse(center, eigenvalues, axes, angles)
    extra.update(fit, log_fit_rmse=float(np.sqrt(np.average((design @ coef - z) ** 2, weights=weight**2))))
    if not _inside(np.vstack((center, contour)), mask).all():
        return center, contour, extra, "contour_outside_cell", (eigenvalues, axes)
    extra.update(_ellipse_geometry(contour, sigma, covariance))
    return center, contour, extra, "ok", (eigenvalues, axes)


def _half_max_ellipse(center, eigenvalues, axes, angles):
    """Closed half-maximum contour and widths of a Gaussian with this precision."""
    sigma = 1 / np.sqrt(eigenvalues)  # major, minor
    k = np.sqrt(2 * np.log(2))
    unit_circle = np.column_stack((np.cos(angles), np.sin(angles)))
    contour = center + (unit_circle * (k * sigma)) @ axes.T
    contour = np.vstack((contour, contour[0]))
    covariance = (axes * (sigma * sigma)) @ axes.T
    fit = dict(
        sigma_major_px=float(sigma[0]), sigma_minor_px=float(sigma[1]),
        fwhm_major_px=float(2 * k * sigma[0]), fwhm_minor_px=float(2 * k * sigma[1]),
        angle_deg=float(np.degrees(np.arctan2(axes[1, 0], axes[0, 0])) % 180),
    )
    return contour, sigma, covariance, fit


def radial_contours(center_local: np.ndarray, edge_px: np.ndarray, origins_xy: np.ndarray, n_angles: int) -> np.ndarray:
    """Closed global contours from per-ray edge radii, exactly as ``measure_cells`` builds them."""
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    points = center_local[:, None, :] + edge_px[:, :, None] * directions[None]
    return np.concatenate((points, points[:, :1]), axis=1) + origins_xy[:, None, :]


def ellipse_contours(
    centers: np.ndarray, eigenvalues: np.ndarray, axes: np.ndarray, n_angles: int, offsets=None,
) -> np.ndarray:
    """Closed half-maximum ellipses (NaN where no ellipse was fitted), as the fits draw them."""
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    out = np.full((len(centers), n_angles + 1, 2), np.nan)
    # Per cell, like the measurement: the same small matmul keeps the vertices bit-identical.
    for i in np.flatnonzero(np.isfinite(eigenvalues).all(axis=1)):
        contour = _half_max_ellipse(centers[i], eigenvalues[i], axes[i], angles)[0]
        out[i] = contour if offsets is None else contour + offsets[i]
    return out


def _ellipse_geometry(contour, sigma, covariance) -> dict[str, float]:
    k = np.sqrt(2 * np.log(2))
    geometry = _geometry(contour)
    # Analytic ellipse area and projected widths avoid polygon sampling error.
    area = float(np.pi * k**2 * np.prod(sigma))
    geometry.update(
        area_px2=area, cd_equivalent_px=float(2 * np.sqrt(area / np.pi)),
        cd_x_px=float(2 * k * np.sqrt(covariance[0, 0])),
        cd_y_px=float(2 * k * np.sqrt(covariance[1, 1])),
    )
    return geometry


def gaussian_method(
    params: np.ndarray, status: np.ndarray, cleaned: np.ndarray, masks: np.ndarray,
    origins_xy: np.ndarray, n_angles: int,
) -> CDMethodResult:
    """Half-maximum ellipses of the deblended Gaussian fits (ROI-local params)."""
    from .deblend import model_on_rois, precision_matrices

    n = len(params)
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    names = GEOMETRY_METRICS + ("background", "peak_signal") + FIT_METRICS[:-1] + ("fit_rmse",)
    result = CDMethodResult(
        method="gaussian", centers_xy=np.full((n, 2), np.nan),
        contours_xy=np.full((n, n_angles + 1, 2), np.nan),
        valid=np.zeros(n, dtype=bool), status=np.asarray(status, dtype="U32").copy(),
        metrics={name: np.full(n, np.nan) for name in names},
        geometry={"eigenvalues": np.full((n, 2), np.nan), "axes": np.full((n, 2, 2), np.nan)},
    )
    if n == 0:
        return result
    residual = np.where(masks, cleaned - model_on_rois(params, cleaned.shape[1:]), 0.0)
    rmse = np.sqrt((residual**2).sum(axis=(1, 2)) / np.maximum(masks.sum(axis=(1, 2)), 1))
    precision = precision_matrices(params)
    for i in range(n):
        result.metrics["peak_signal"][i] = params[i, 0]
        result.metrics["fit_rmse"][i] = rmse[i]
        if not np.isfinite(params[i]).all():
            continue
        center = params[i, 1:3] + origins_xy[i]
        result.centers_xy[i] = center
        eigenvalues, axes = np.linalg.eigh(precision[i])
        if eigenvalues[0] <= 1e-10:
            result.status[i] = "non_gaussian_fit"
            continue
        contour, sigma, covariance, fit = _half_max_ellipse(center, eigenvalues, axes, angles)
        result.contours_xy[i] = contour
        result.geometry["eigenvalues"][i] = eigenvalues
        result.geometry["axes"][i] = axes
        for name, value in fit.items():
            result.metrics[name][i] = value
        if result.status[i] == "ok":
            for name, value in _ellipse_geometry(contour, sigma, covariance).items():
                result.metrics[name][i] = value
    result.valid[:] = result.status == "ok"
    return result


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
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str, np.ndarray]:
    """Edge per ray (NaN where none is found); the last value is that edge radius array."""
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
        return center, contour, extra, "incomplete_rays", edge
    extra.update(_geometry(contour))
    return center, contour, extra, "ok", edge


def measure_cells(
    rois: np.ndarray, masks: np.ndarray, origins_xy: np.ndarray, seeds_xy: np.ndarray,
    *, methods: tuple[str, ...], n_angles: int, radial_step: float,
    smooth_sigma: float, fit_fraction: float, backgrounds: np.ndarray | None = None,
) -> dict[str, CDMethodResult]:
    """Measure all methods on the same ordered cell stack.

    ``backgrounds`` overrides the per-cell border estimate, e.g. zeros for
    cells whose background was already removed by deblending.
    """
    n = len(rois)
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    results = {}
    for method in methods:
        names = GEOMETRY_METRICS + ("background", "peak_signal")
        names += FIT_METRICS if method == "logquad" else ("ray_coverage",)
        geometry = {"center_local": np.full((n, 2), np.nan)}
        if method == "logquad":
            geometry.update(eigenvalues=np.full((n, 2), np.nan), axes=np.full((n, 2, 2), np.nan))
        else:
            geometry["edge_px"] = np.full((n, n_angles), np.nan)
        results[method] = CDMethodResult(
            method=method, centers_xy=np.full((n, 2), np.nan),
            contours_xy=np.full((n, n_angles + 1, 2), np.nan),
            valid=np.zeros(n, dtype=bool), status=np.full(n, "no_signal", dtype="U32"),
            metrics={name: np.full(n, np.nan) for name in names}, geometry=geometry,
        )
    for i, (roi, mask, origin, seed) in enumerate(zip(rois, masks, origins_xy, seeds_xy)):
        if mask.sum() < 6:
            for result in results.values():
                result.status[i] = "insufficient_cell_pixels"
            continue
        background = _background(roi, mask) if backgrounds is None else float(backgrounds[i])
        if roi[mask].max() <= background:
            continue
        radial = None
        for method, result in results.items():
            if method == "logquad":
                center, contour, metrics, status, ellipse = _logquad(
                    roi, mask, seed - origin, background, angles, fit_fraction,
                )
                if ellipse is not None:
                    result.geometry["eigenvalues"][i], result.geometry["axes"][i] = ellipse
            else:
                if radial is None:
                    radial = _radial_profiles(roi, mask, background, angles, radial_step, smooth_sigma)
                center, contour, metrics, status, edge = _radial_contour(method, *radial, angles, background)
                result.geometry["edge_px"][i] = edge
            result.geometry["center_local"][i] = center
            result.centers_xy[i] = center + origin
            result.contours_xy[i] = contour + origin
            result.status[i] = status
            result.valid[i] = status == "ok"
            for name, value in metrics.items():
                result.metrics[name][i] = value
    return results
