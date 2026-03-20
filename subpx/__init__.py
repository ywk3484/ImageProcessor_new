"""subpx: notebook-friendly subpixel image-processing utilities."""

try:
    from .viewer import imshow_huge, show_image, GLTiledImshow, GLOrtho2D
except Exception:
    imshow_huge = None
    show_image = None
    GLTiledImshow = None
    GLOrtho2D = None

from .centers import (
    recommend_refine_method,
    detect_centers,
    detect_centers_tiled,
    detect_centers_tiled_global_otsu,
    refine_centers,
    refine_weighted_centroid,
    refine_logquadratic,
    refine_centers_edge_moment,
    filter_centers,
    dedupe_centers,
)
from .components import connected_components_stats
from .pitch import estimate_pitch, estimate_pitch_lines, pitch_residuals
from .registration import estimate_shift, apply_shift, crop_overlap
from .spectra import (
    fft_pitch_error,
    top_periodic_errors,
    fft_periodicity_uniform,
    fft_periodicity_resample,
)
from .calibration import (
    find_y_cluster_split_indices,
    split_clusters_1d,
    assign_clusters_1d,
    residuals_vs_x_by_line,
    line_offset_summary,
    build_row_residual_map,
    smooth_row_residual_map,
    interpolate_row_residual_field,
    fit_rowwise_distortion_field,
)
from .batch import process_frames, process_stack, detect_centers_batch, detect_centers_tiled_batch, estimate_shift_batch
from .mosaic import (
    TiffKey,
    parse_tiff_key,
    build_tiff_index,
    iter_pages_across_vertical,
    iter_pages_canonical,
    map_centers_to_global,
    true_runs,
    cell_runs_to_global_rects,
    process_stripe_board,
)

__all__ = [
    "imshow_huge",
    "show_image",
    "GLTiledImshow",
    "GLOrtho2D",
    "detect_centers",
    "detect_centers_tiled",
    "detect_centers_tiled_global_otsu",
    "refine_centers",
    "refine_weighted_centroid",
    "refine_logquadratic",
    "refine_centers_edge_moment",
    "filter_centers",
    "dedupe_centers",
    "connected_components_stats",
    "estimate_pitch",
    "estimate_pitch_lines",
    "pitch_residuals",
    "estimate_shift",
    "apply_shift",
    "crop_overlap",
    "fft_pitch_error",
    "top_periodic_errors",
    "fft_periodicity_uniform",
    "fft_periodicity_resample",
    "find_y_cluster_split_indices",
    "split_clusters_1d",
    "assign_clusters_1d",
    "residuals_vs_x_by_line",
    "line_offset_summary",
    "build_row_residual_map",
    "smooth_row_residual_map",
    "interpolate_row_residual_field",
    "fit_rowwise_distortion_field",
    "process_frames",
    "process_stack",
    "detect_centers_batch",
    "detect_centers_tiled_batch",
    "estimate_shift_batch",
    "TiffKey",
    "parse_tiff_key",
    "build_tiff_index",
    "iter_pages_across_vertical",
    "iter_pages_canonical",
    "map_centers_to_global",
    "true_runs",
    "cell_runs_to_global_rects",
    "process_stripe_board",
]
