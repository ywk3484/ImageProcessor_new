# Calibration helpers

## Public functions
- `find_y_cluster_split_indices(y, threshold=10.0)`
- `split_clusters_1d(values, threshold)`
- `assign_clusters_1d(values, threshold)`
- `residuals_vs_x_by_line(centers_xy, ...)`
- `line_offset_summary(residual_cloud)`

## Purpose
These helpers support the row/column grouping and residual-cloud workflow used during blob-center and pitch calibration.

### Notes
- `find_y_cluster_split_indices(...)` is preserved as a compatibility function because it was already used directly in notebooks.
- `residuals_vs_x_by_line(...)` is the cleaned API for flattening line-wise residuals into a common residual-vs-x cloud while still retaining line identities for later rowwise modeling.
- When distortion varies slowly across rows, prefer keeping `line_id` and `line_coord` rather than forcing one global common distortion model.


## 2D row-coupled distortion field

Added notebook-facing helpers:
- `build_row_residual_map(...)`
- `smooth_row_residual_map(...)`
- `interpolate_row_residual_field(...)`
- `fit_rowwise_distortion_field(...)`

These implement the row-coupled model discussed in the chat: build a row-by-x residual map, smooth along x and then across neighboring rows, and evaluate corrections per point.
