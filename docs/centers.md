
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

## GPU edge-erf: holes far from the page's top-left corner (fixed 2026-11-03)

`detect_centers(..., backend="gpu", refine="edge_erf")` fits each of a hole's four edges with a blurred step plus
a straight-line background. The background's slope used page coordinates, so the fit got worse-conditioned the
farther the edge sat from the page's left or top edge, and every edge beyond about 140 px was refused: on a page of
300 nm squares it found 0.6 % of the holes. The slope now uses coordinates relative to the edge's own start (the
same model, so edges found before move only by rounding): 100 % of those squares found, 0.28 nm per hole in y.
Test: `tests/test_gpu_edge_erf.py`. Edge-erf still finds few holes smaller than about 250 nm (7–20 % of them on the
test pages of `docs/pitch_signature_test.md`); use it for large holes.
