
# Pitch Estimation

## Global pitch
The current stable global estimator uses nearest-neighbor vectors.
For each point, the nearest neighbor vectors are examined and classified as:
- x-like when their angle is close to the x-axis
- y-like when their angle is close to the y-axis

Pitch is estimated as the median absolute projected displacement along each axis.

## Refinement by index regression
Given initial pitch values, integer lattice indices are assigned by rounding.
Then a least-squares line is fit:
- `x ~ a * ix + b`
- `y ~ c * iy + d`

Residual-based trimming is used to reject outliers and refine the pitch values.

## Per-line pitch
Points are grouped into rows or columns by clustering the perpendicular coordinate.
Within each line, sorted coordinate differences are computed and robustly trimmed
using a MAD-based rule.
