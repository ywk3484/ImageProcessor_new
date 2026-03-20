"""Compatibility wrappers for old notebook function names."""

from __future__ import annotations

import warnings

from .centers import detect_centers, detect_centers_tiled, refine_weighted_centroid, refine_logquadratic, filter_centers, dedupe_centers
from .pitch import estimate_pitch, estimate_pitch_lines
from .calibration import find_y_cluster_split_indices
from .spectra import fft_pitch_error, fft_periodicity_uniform, fft_periodicity_resample


def _warn(old: str, new: str):
    warnings.warn(f"{old} is deprecated; use {new} instead.", DeprecationWarning, stacklevel=2)


def find_subpixel_centers_otsu_opencv(*args, **kwargs):
    _warn("find_subpixel_centers_otsu_opencv", "detect_centers(...)")
    method = kwargs.pop("method", "weighted")
    return detect_centers(*args, refine=method, **kwargs).centers_xy


def find_centers_hybrid_gpu_cpu_logquad(*args, **kwargs):
    _warn("find_centers_hybrid_gpu_cpu_logquad", "detect_centers(..., backend='gpu', refine='logquad')")
    kwargs.pop("method", None)
    return detect_centers(*args, backend="gpu", refine="logquad", **kwargs).centers_xy


def find_subpixel_centers_tiled_logquad_gpu(*args, **kwargs):
    _warn("find_subpixel_centers_tiled_logquad_gpu", "detect_centers_tiled(..., backend='gpu', refine='logquad')")
    if "device_id" in kwargs:
        kwargs["device"] = kwargs.pop("device_id")
    return detect_centers_tiled(*args, backend="gpu", refine="logquad", **kwargs).centers_xy


def refine_center_weighted_centroid(*args, **kwargs):
    _warn("refine_center_weighted_centroid", "refine_centers(..., method='weighted')")
    return refine_weighted_centroid(*args, **kwargs)


def refine_center_log_quadratic(*args, **kwargs):
    _warn("refine_center_log_quadratic", "refine_centers(..., method='logquad')")
    return refine_logquadratic(*args, **kwargs)


def filter_centers_by_margin(*args, **kwargs):
    _warn("filter_centers_by_margin", "filter_centers(...)")
    return filter_centers(*args, **kwargs)


def cluster_centers_radius(*args, **kwargs):
    _warn("cluster_centers_radius", "dedupe_centers(...)")
    return dedupe_centers(*args, **kwargs)


def estimate_pitch_knn(*args, **kwargs):
    _warn("estimate_pitch_knn", "estimate_pitch(..., method='knn')['initial']")
    return estimate_pitch(*args, method="knn", **kwargs)["initial"]


def pitch_per_line(*args, **kwargs):
    _warn("pitch_per_line", "estimate_pitch_lines(...)")
    res = estimate_pitch_lines(*args, **kwargs)
    return {
        "line_centers_perp": res.meta["line_centers_perp"],
        "line_counts": res.meta["line_counts"],
        "line_pitch": res.values,
        "line_ids_per_point": res.meta["line_ids_per_point"],
    }


def fourier_pitch_error(*args, **kwargs):
    _warn("fourier_pitch_error", "fft_pitch_error(...)")
    return fft_pitch_error(*args, **kwargs)


def refine_centers_edges_centeranchored_gpu(*args, **kwargs):
    _warn("refine_centers_edges_centeranchored_gpu", "refine_centers_edge_moment(..., backend='gpu')")
    from .centers import refine_centers_edge_moment
    return refine_centers_edge_moment(*args, backend='gpu', **kwargs)


def fft_periodicity_uniform_legacy(*args, **kwargs):
    _warn("fft_periodicity_uniform", "fft_periodicity_uniform(...)")
    return fft_periodicity_uniform(*args, **kwargs)


def fft_periodicity_resample_legacy(*args, **kwargs):
    _warn("fft_periodicity_resample", "fft_periodicity_resample(...)")
    return fft_periodicity_resample(*args, **kwargs)
