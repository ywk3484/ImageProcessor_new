
# Center Detection and Refinement

## Detection flow
Current stable implementation:
1. Otsu thresholding on the grayscale image
2. connected components extraction
3. area filtering
4. ROI-based subpixel center refinement

## Refinement methods
### Weighted centroid
A background level is estimated from border pixels around the ROI. Positive signal
above background is used as weight. The center is then:

- `cx = sum(x * w) / sum(w)`
- `cy = sum(y * w) / sum(w)`

This is simple and robust when the blob is compact but not strongly PSF-like.

### Log-quadratic fit
A quadratic surface is fit to `log(I - bg)` over masked pixels. The stationary point
of the fitted quadratic gives the subpixel location. This approximates a 2D Gaussian
peak fit and is often better for peaked spots.

## Public surface
- `detect_centers(...)`
- `refine_centers(...)`
- `filter_centers(...)`
- `dedupe_centers(...)`


## Center-anchored edge-moment refinement

Added `refine_centers_edge_moment(...)` and GPU support for `detect_centers(..., refine="edge_gradmoment")`.

This method predicts each edge from coarse center ± half-width, then localizes each edge independently from a small gradient-moment window. It is designed to reduce pixel locking without imposing a global lattice calibration.
