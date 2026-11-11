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
