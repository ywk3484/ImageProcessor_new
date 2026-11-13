# Outline fit (`outline_fit`)

A mask hole has a sharp edge; the tool blurs it into a soft hill. Every other CD
method reads the hill (its width, or the area above some level), so it reads the
hole's size and the blur together: a hole 4 nm bigger reads about 2 nm bigger, and
a focus drift between two scans shows up as a CD change of several nanometres.

The outline fit separates the two. It fits the whole image with **one sharp
outline** (any star-convex shape: a superellipse plus harmonics) convolved with
**one blur**, both shared by every hole of the image. Each hole has its own
position, brightness, local level and **edge offset** (how far its edge sits
outside or inside the shared outline). A hole's CD is that of the outline moved
by its edge offset. The blur is fitted per image, so a focus change moves the
fitted blur, not the CD, and the brightness is free per hole, so flicker does not
enter the size either.

On the test pages (80 to 150 nm holes, 21 nm pixels, about 65 nm blur):

- a 1 nm bigger hole reads 1.01 to 1.05 nm bigger (the other methods 0.35 to 1.10);
- a blur change from 58 to 72 nm moves the mean CD of 100 nm holes by 0.2 nm
  (the other methods 3.9 to 11 nm);
- the mean CD is within 0.2 to 0.7 nm of the truth.

The method, its derivation and the full test results are on the method page,
`artifacts/outline_fit/outline_fit.html`.

## Folder report: the 3σ of every image

```bat
python scripts\cd_report.py D:\data\png_images --output D:\data\report --pixel-size 21 --unit nm --outline-fit
python scripts\cd_fast_report.py D:\data\png_images --output D:\data\report_fast --pixel-size 21 --unit nm --outline-fit
```

`--outline-fit` adds `outline_fit` beside the methods the report already runs;
nothing else changes. Its numbers go where every method's go:

| Output | What it holds for `outline_fit` |
|---|---|
| `summary.csv` | per image: valid holes, mean CD, σ and **within-image 3σ %** (= 300 σ / mean), for the CD and the x / y widths |
| `between_images.csv` | over the images: mean of the image means, their σ, **between-image 3σ** (nm and %) |
| `index.html` | the same tables, the CD map, and on each image tab a line *Outline fit: outline CD …, blur FWHM … × … nm, …; converged after N rounds; a size per hole; …* |
| console | one line per image with the same page readings |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--outline-blur gauss2` | `gauss2` | the blur's family: two Gaussians (`gauss2`) or a Gaussian times a circular pupil (`lens`) |
| `--outline-rounds 14` | 14 | the most rounds of the fit; it stops earlier once the outline settles |

**Which holes count.** The same interior holes as the other methods. A hole is
left out only where its fit failed: its edge offset reached the fit's bound
(status `edge_offset_at_clip`) or a value is not finite (`not_finite`).

**Time.** 1.5 to 4.5 minutes per 2048 × 1378 image on the GPU (a page of
28,500 holes of 100 nm: 89 s, converged in 11 rounds; the prototype's pages of
14,000 to 49,000 holes: 143 to 270 s), 15 to 20 times longer on the CPU (25 to
70 minutes). Run it on the GPU; check that nothing else is using it first. A
folder of 400 images is 10 to 30 hours on the GPU.

## When to trust the numbers

Read the page line of each image (console, image tab, workbench log):

- **`converged after N rounds`**: the outline stopped moving. **`NOT converged
  in N rounds`**: the CDs are not final; run again with more `--outline-rounds`.
  The report adds a note for every such image.
- **`a size per hole`**: every hole got its own edge offset. **`per-hole size
  off`**: the holes are too small for a size each at this noise (the median
  standard error of the edge offset reached 0.3 of the outline radius). Every
  hole then reads the image's CD (status `page_size`): the image's mean is still
  measured and enters the between-image 3σ, but its within-image σ is not: the
  report shows "–" there (`nan` in `summary.csv`) and adds a note.
- **The blur.** A blur far from the other images' blurs is the one sign of a
  wrong fit (below).

Limits:

- **Small holes.** Below about 70 nm at 21 nm pixels and 65 nm blur, no method
  can tell size and blur apart: the edge's mark lies beyond what the optics pass.
  The fit can then settle on a wrong pair (a 50 nm dense page read 66.7 nm under
  a 39 nm blur; true 50 and 65) and nothing in the fit flags it except the odd
  blur. Use it from about 80 nm up.
- **The blur family is the model's one assumption.** The fit finds the best blur
  within the family it is given. With the true form the size is exact; with a
  different form it is off by about −0.1 nm at 100 nm and −1.6 nm at 80 nm. Run
  both families (`--outline-blur gauss2`, then `lens`): where they agree the CD
  is safe; where they differ, the difference is the uncertainty.
- **Light the model does not contain** (a particle, a merged hole) is held off by
  robust weights and shows in that hole's `leftover_rms`. A hole whose edge
  offset reaches the fit's bound is left out (`edge_offset_at_clip`).
- **GPU runs repeat to about 0.01 nm**, not bit for bit (the GPU's sums run in
  a varying order). CPU runs repeat exactly.

## The same fit in one pass (`outline_fit_fast`)

`outline_fit_fast` fits the same model (one sharp outline, one blur, the same
parameters and the same CD) without the 10 to 14 rounds: it averages the image's
holes as they are, puts the neighbours' light into the model of that average,
and fits outline and blur to it once, centres the holes, and fits once more. It
gives the **image's** outline CD, x / y widths and blur in a fraction of a
second. It is there to be compared with `outline_fit` on real images; a size per
hole and the TIFF batch come after that comparison.

```bat
python scripts\cd_fast_report.py D:\data\png_images --output D:\data\report_fast_cmp --pixel-size 21 --unit nm --outline-fit --outline-fit-fast
```

`--outline-fit-fast` works in `cd_report.py` and `cd_fast_report.py`, alone or
beside `--outline-fit`; `--outline-blur` applies to both methods. What the
report shows for it:

- per image, on the console and the image tab: *Outline fit (fast): outline CD
  100.71 nm, blur FWHM 63.1 × 62.8 nm (gauss2; second Gaussian 3.4 x 3.4 px,
  23 % of the light), exponent p 2.00; leftover 0.99 (chi-square per bin: 1 = at
  the noise); first fit 100.57 nm; 38,240 of 39,042 holes stacked; page size for
  every hole; 0.21 s*
- in the tables: the image's mean CD and x / y widths. **Every hole reads the
  image's CD** (status `page_size`), so the within-image σ is shown as "–"
  (`nan` in `summary.csv`); the between-image 3σ is measured.
- with both flags, per image: *blur second part: outline_fit second Gaussian
  2.0 x 2.0 px, 19 % of the light; outline_fit_fast second Gaussian 3.5 x
  3.6 px, 25 % of the light*
- with both flags, as the last line of the console and among the report's
  notes: `outline_fit_fast - outline_fit: mean +0.08, largest 0.14 nm over 12
  images; 0.30 s against 95 s per image` (the mean and the largest absolute
  difference of the two outline CDs per image; each method's seconds per image).

Reading the line:

- **leftover** is the chi-square per bin of the final fit to the average hole:
  near 1 the model fits the average hole to its noise (0.9 to 1.7 on generated
  images); far above 1 the model does not fit the image's average hole.
- **first fit** is the CD before the holes were centred; the final one is the
  reading.
- **the blur's second part** (the second Gaussian, or the pupil's cut-off for
  `lens`) is what tells two fits apart that sit at different end points of the
  same model. On some images the fit has more than one end (below); if the two
  methods' CDs differ and their second parts clearly differ too, that is a
  question about the model, not about the one-pass procedure.
- *final fit NOT converged in 30 iterations*: the fit stopped at its iteration
  cap; the report adds a note.

Limits (all numbers are from generated images; nothing has been run on a real
one):

- **GPU only.** On the CPU the method is reported as not fitted (status
  `needs_gpu`) and the other methods run as usual. The first run on a machine
  compiles the GPU routines (about half a minute, once).
- **Page values only.** No size per hole, no within-image σ.
- **Few holes.** Error of the page CD on generated images, gauss2 / lens: 135
  stacked holes +0.63 / +0.31 nm, 999 holes +0.34 / +0.16, 4,941 holes +0.09 /
  −0.06, full pages of 30,000 holes −0.14 / +0.01 (100 nm circles). The cause of
  the small images' offset was not found. With fewer than 10 holes nothing is
  fitted (`too_few_holes`).
- **More than one end point.** On some images the final fit can end in more
  than one place, and its start decides: with `lens` on 7 of 10 generated full
  pages (the ends 0.02 to 0.65 nm apart in page CD), with `gauss2` on 3 of 10.
  It belongs to the model (one outline, one free blur shape), and whether
  `outline_fit` ends at the same end on a given image is not known. Compare with
  `gauss2` (the default) first. The `lens` errors on generated pages are
  flattered: the generator's pupil is where the fit starts.
- **Small holes** as for `outline_fit`: below about 70 nm size and blur cannot
  be told apart (60 nm dense holes read +1.7 nm, 50 nm holes +9.6 nm).
- **It does not know an image without the pattern**: it fits whatever bright
  blobs it is given and reports a CD with a good leftover.
- **Repeats** to 0.0003 nm on well determined images, to 0.007 nm where the fit
  has several ends, to 0.02 to 0.07 nm on small dense patterns (the GPU's sums
  run in a varying order).
- **Time** on an RTX 4060 Ti, 2048 × 1378 px, 39,000 holes: about 0.2 s per
  image with `gauss2`, 0.15 s with `lens` (0.02 to 0.2 s on small images),
  beside the report's other methods.

```python
fit = subpx.fit_outline_fast(image, pixel_size=21, unit="nm")          # own seed finder; or pass seeds_xy
fit.outline_cd_px, fit.cd_x_px, fit.cd_y_px, fit.blur_fwhm_px, fit.summary
result = subpx.measure_cds(image, methods=("half_area", "outline_fit_fast"), pixel_size=21, unit="nm")
result.meta["outline_fit_fast"]                                         # the page fit, or {"fitted": False, "reason": ...}
```

Per-hole metrics of the method in `measure_cds`: `cd_equivalent_px`, `cd_x_px`,
`cd_y_px`, `area_px2`, `perimeter_px` (the image's values, repeated),
`brightness`, `background` (the hole's level) and `leftover_rms`. Statuses:
`page_size`; `not_finite`; `too_few_holes`, `needs_gpu`, `not_fitted` (no fit).

Method, settings and every measurement: `docs/outline_fit_one_pass.md` (page:
`docs/outline_fit_one_pass.html`). Code: `fit_outline_fast` and the adapter in
`subpx/outline.py`, the page fit in `subpx/_outline/onepass.py`, the GPU steps
in `subpx/_gpu/outline_onepass.py`. Tests: `tests/test_outline_fit_fast.py`.

## Workbench

Tick **Outline fit** (group *Page fit*) in **CD methods**; its settings, **Blur**
and **Most rounds**, are under *Parameters of the ticked methods*. Try it on a
region first (draw one on the image): a 120 × 170 px region of 220 holes takes
about 30 s on the CPU. The log gets the same page line as the report, and a
warning when the fit did not converge or switched the per-hole size off.

The hole inspector shows the hole's raw pixels with the fitted outline over them
(the fit models the whole image, not a cleaned cell) and, under *Metrics*:
the CD and x / y widths, the CD's standard error, the edge offset, the hole's
brightness and level, and the leftover RMS.

## Python

```python
import subpx

# beside the other methods; the page readings are in result.meta["outline_fit"]
result = subpx.measure_cds(image, methods=("half_area", "outline_fit"), pixel_size=21, unit="nm",
                           outline_blur="gauss2", outline_rounds=14)
m = result.measurements["outline_fit"]
cd_nm = m.metrics["cd_equivalent_px"][m.valid] * result.pixel_size

# the fit alone, from your own seeds: outline, blur, every hole's numbers and outline rays
fit = subpx.fit_outline(image, seeds_xy, pixel_size=21, unit="nm", backend="gpu")
fit.cd_px, fit.cd_se_px, fit.edge_offset_px, fit.blur_fwhm_px, fit.converged, fit.summary
```

`measure_cds` needs `background="model"` (the default) for `outline_fit`. It
fits every hole it modelled, frame-cut ones included (their light belongs to the
page), and reports the measured ones, as the other methods do. With fewer than
10 holes nothing is fitted (status `too_few_holes`).

Per-hole metrics: `cd_equivalent_px` (2 √(area / π)), `cd_x_px` / `cd_y_px`
(the x / y diameters of the ellipse with the outline's area and x / y spread, as
for the area methods), `area_px2`, `perimeter_px`, `cd_se_px` (the CD's standard
error from the fit), `edge_offset_px`, `brightness`, `background` (the hole's
level) and `leftover_rms`. CDs are NaN where the hole is not valid.

Statuses: `ok`; `page_size` (valid, per-hole size off: the image's CD);
`edge_offset_at_clip` and `not_finite` (failed fits); `too_few_holes`.

Code: `subpx/outline.py` (public functions, the CD-method adapter), the fit in
`subpx/_outline/` (`model.py`: outline and blur on a fine grid; `page.py`: the
page fit), NumPy or CuPy in float64. Tests: `tests/test_outline_fit.py`.
