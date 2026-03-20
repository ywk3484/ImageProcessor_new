"""Spectral analysis for pitch and residual errors."""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import lombscargle as scipy_lombscargle  # type: ignore
except Exception:  # pragma: no cover
    scipy_lombscargle = None

from .types import SpectrumResult


def _resample_uniform(x: np.ndarray, y: np.ndarray, dx: float | None = None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 2:
        return x, y, np.nan
    if dx is None:
        diffs = np.diff(x)
        dx = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    xu = np.arange(x[0], x[-1] + 0.5 * dx, dx, dtype=np.float64)
    yu = np.interp(xu, x, y)
    return xu, yu, float(dx)


def _detrend(y: np.ndarray, x: np.ndarray, mode: str):
    if mode == "none":
        return y
    if mode == "mean":
        return y - np.mean(y)
    if mode == "linear":
        A = np.column_stack([x, np.ones_like(x)])
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - (a * x + b)
    raise ValueError("detrend must be one of: 'none', 'mean', 'linear'.")


def fft_pitch_error(
    x,
    y,
    *,
    dx: float | None = None,
    detrend: str = "linear",
    window: str = "hann",
    one_sided: bool = True,
    amplitude_units: str = "same_as_y",
) -> SpectrumResult:
    """FFT of pitch/residual error versus position.

    Returns spatial frequency in 1/(x units) and spatial period in x units.
    The amplitude remains in the same units as y.
    """
    xu, yu, dxu = _resample_uniform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), dx=dx)
    if xu.size < 2 or not np.isfinite(dxu):
        return SpectrumResult(
            frequency=np.zeros((0,), dtype=np.float64),
            amplitude=np.zeros((0,), dtype=np.float64),
            period=np.zeros((0,), dtype=np.float64),
            method="fft",
            meta={"dx": dxu, "n_samples": int(xu.size)},
        )

    sig = _detrend(yu, xu, detrend)
    if window == "hann":
        w = np.hanning(sig.size)
    elif window == "none":
        w = np.ones(sig.size, dtype=np.float64)
    else:
        raise ValueError("window must be 'hann' or 'none'.")

    sw = sig * w
    n = sw.size
    freq = np.fft.fftfreq(n, d=dxu)
    fftv = np.fft.fft(sw)
    amp = 2.0 * np.abs(fftv) / np.sum(w)
    if not one_sided:
        period = np.where(np.abs(freq) > 0, 1.0 / np.abs(freq), np.inf)
        return SpectrumResult(freq, amp, period, meta={"dx": dxu, "n_samples": int(n), "detrend": detrend, "window": window, "amplitude_units": amplitude_units})

    keep = freq > 0
    freq = freq[keep]
    amp = amp[keep]
    period = 1.0 / freq
    return SpectrumResult(
        frequency=freq,
        amplitude=amp,
        period=period,
        method="fft",
        meta={
            "dx": dxu,
            "n_samples": int(n),
            "detrend": detrend,
            "window": window,
            "one_sided": bool(one_sided),
            "amplitude_units": amplitude_units,
        },
    )


def top_periodic_errors(spec: SpectrumResult, *, n: int = 5, min_frequency: float | None = None):
    freq = np.asarray(spec.frequency, dtype=np.float64)
    amp = np.asarray(spec.amplitude, dtype=np.float64)
    period = np.asarray(spec.period, dtype=np.float64)
    keep = np.isfinite(freq) & np.isfinite(amp) & np.isfinite(period)
    if min_frequency is not None:
        keep &= freq >= float(min_frequency)
    freq = freq[keep]
    amp = amp[keep]
    period = period[keep]
    order = np.argsort(amp)[::-1][:int(max(0, n))]
    return {
        "frequency": freq[order],
        "amplitude": amp[order],
        "period": period[order],
    }


def fft_periodicity_uniform(x_um, pitch, detrend_order: int | None = 1, window: str = "hann"):
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(pitch, dtype=np.float64)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    if detrend_order is not None and detrend_order >= 0 and x.size >= detrend_order + 1:
        X = np.vander(x - x.mean(), detrend_order + 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        trend = X @ coef
        e = y - trend
    else:
        e = y - np.nanmedian(y)
    dx = float(np.median(np.diff(x))) if x.size >= 2 else np.nan
    n = e.size
    if window == "hann":
        w = np.hanning(n)
    elif window is None or window == "none":
        w = np.ones(n)
    else:
        raise ValueError("window must be 'hann' or None")
    ew = (e - np.mean(e)) * w
    E = np.fft.rfft(ew)
    freq = np.fft.rfftfreq(n, d=dx) if np.isfinite(dx) else np.zeros((0,), dtype=np.float64)
    amp = (2.0 / np.sum(w)) * np.abs(E) if n else np.zeros((0,), dtype=np.float64)
    if amp.size:
        amp[0] = 0.0
    period = np.full_like(freq, np.inf)
    m = freq > 0
    period[m] = 1.0 / freq[m]
    return {"freq_cyc_per_um": freq, "period_um": period, "amplitude": amp}


def fft_periodicity_resample(x_um, pitch, dx_um: float | None = None, detrend_order: int | None = 1):
    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(pitch, dtype=np.float64)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    if dx_um is None and x.size >= 2:
        dx_um = float(np.median(np.diff(x)))
    if dx_um is None or not np.isfinite(dx_um) or dx_um <= 0:
        dx_um = 1.0
    xg = np.arange(x[0], x[-1] + 0.5 * dx_um, dx_um) if x.size else np.zeros((0,), dtype=np.float64)
    yg = np.interp(xg, x, y) if x.size else np.zeros((0,), dtype=np.float64)
    return fft_periodicity_uniform(xg, yg, detrend_order=detrend_order, window="hann")


def periodicity_lombscargle(
    x_um,
    pitch,
    *,
    freq_min: float | None = None,
    freq_max: float | None = None,
    n_freq: int = 4096,
    detrend_order: int | None = 1,
    normalize: bool = True,
    floating_mean: bool = True,
    center_data: bool = True,
):
    """Spectral estimation for irregularly sampled data using Lomb-Scargle.

    Frequencies are returned in cycles / um and periods in um. The ordinate is
    Lomb-Scargle power, not direct amplitude in y-units.
    """
    if scipy_lombscargle is None:
        raise RuntimeError("scipy.signal.lombscargle is required for irregular-sampling spectral analysis.")

    x = np.asarray(x_um, dtype=np.float64)
    y = np.asarray(pitch, dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 3:
        return SpectrumResult(
            frequency=np.zeros((0,), dtype=np.float64),
            amplitude=np.zeros((0,), dtype=np.float64),
            period=np.zeros((0,), dtype=np.float64),
            method="lomb_scargle",
            meta={"n_samples": int(x.size)},
        )

    if detrend_order is not None and detrend_order >= 0 and x.size >= detrend_order + 1:
        X = np.vander(x - x.mean(), detrend_order + 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y = y - (X @ coef)
    elif center_data:
        y = y - np.nanmean(y)

    dx_med = float(np.median(np.diff(x))) if x.size >= 2 else 1.0
    span = float(x[-1] - x[0]) if x.size >= 2 else 1.0
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    if freq_min is None:
        freq_min = max(1.0 / span, 1e-12)
    if freq_max is None:
        freq_max = 0.5 / dx_med if np.isfinite(dx_med) and dx_med > 0 else float(freq_min) * 10.0
    if freq_max <= freq_min:
        freq_max = float(freq_min) * 2.0

    freq = np.linspace(float(freq_min), float(freq_max), int(max(16, n_freq)), dtype=np.float64)
    omega = 2.0 * np.pi * freq
    power = scipy_lombscargle(
        x,
        y,
        omega,
        normalize=bool(normalize),
        floating_mean=bool(floating_mean),
    )
    period = 1.0 / freq
    return SpectrumResult(
        frequency=freq,
        amplitude=np.asarray(power, dtype=np.float64),
        period=period,
        method="lomb_scargle",
        meta={
            "n_samples": int(x.size),
            "freq_min": float(freq_min),
            "freq_max": float(freq_max),
            "normalize": bool(normalize),
            "floating_mean": bool(floating_mean),
            "center_data": bool(center_data),
            "detrend_order": detrend_order,
            "x_units": "um",
            "ordinate": "power",
        },
    )


def periodicity_irregular(
    x_um,
    pitch,
    *,
    method: str = "lomb_scargle",
    **kwargs,
):
    if method != "lomb_scargle":
        raise ValueError("method must be 'lomb_scargle'.")
    return periodicity_lombscargle(x_um, pitch, **kwargs)
