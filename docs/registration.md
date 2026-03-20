
# Registration

## Phase cross correlation
Phase cross correlation estimates translational shift using FFT phase information.
It is appropriate when images differ primarily by translation. Subpixel shift can be
estimated through upsampled correlation around the peak.

## Public surface
- `estimate_shift(ref, mov, method="phase_xcorr")`
- `apply_shift(image, shift_yx)`
- `crop_overlap(ref, mov, shift_yx)`

The current implementation uses scikit-image when available.
