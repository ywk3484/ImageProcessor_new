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
- *final fit NOT converged in 30 iterations*: the final fit was cut at its cap
  of 30 iterations; the report adds a note. The message alone does not say how
  far the CD is from the fit's end: on generated images it is usually within
  0.02 nm, but one fit stood 2.9 nm from its end at iteration 30. The check
  below measures it on a folder.

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
  than one place, and its start decides (and the start is not a fixed point:
  see the check below): with `lens` on 7 of 10 generated full
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

### Does the image's CD depend on where the final fit starts or stops?

First run on real images (2026-11-13, 8 images of one area): every hole of an
image has one colour on the map, the colours differ from image to image (page
CDs 97.13 to 98.89 nm), and the final fit stopped at its cap on all 8.

- One colour per image is what the method gives: it has one CD per image, and
  the map paints it on every hole. It measures no size per hole.
- The page CD of repeated images moving by 1.8 nm, and the cap being hit, is
  not what a settled fit does. On generated images of the same holes two causes
  were found (`docs/outline_fit_one_pass.md`, 6.18): the final fit **starts**
  where the first fit left the blur's second part, which is anywhere, and from
  some starts it ends at another, worse-fitting outline (0.9 nm away in one
  case); and the **cap** of 30 iterations can cut a slow fit nanometres from
  its end. Neither is what separates the real images: the result is below.

The check, on the folder the report was run on (GPU; the report's
`--outline-blur`, `--seeds`, `--invert` apply):

```bat
python scripts\outline_fast_check.py D:\data\png_images --pixel-size 21 --unit nm
```

Per image it runs the one-pass fit as the reports do, then repeats the final
fit (a) from the same start without the cap and (b) from 9 other starts of the
blur's second part (4 with `lens`), each to its end. It prints one line per
image, saves every start's result as JSON under `artifacts/`, and ends with the
line to type back:

```text
check N=8 | rep 97.13-98.89 3s 1.82 nc 8 | end 3s 1.80 it<=212 | best 98.02-98.11 3s 0.09 moved 5 | ends 1-3 gap 344 | left 1.3-2.0 | rblur -0.93
```

| Field | Meaning |
|---|---|
| `rep A-B 3s X nc K` | the page CD the report shows: lowest and highest image, 3 × the standard deviation over the images, and how many images stopped at the cap |
| `end 3s X it<=M` | the same start run to its end; M = the most iterations an image needed |
| `best A-B 3s X moved K` | the end with the lowest leftover of all starts; K images whose best end is not the reported CD |
| `ends a-b gap G` | fewest and most different ends per image; G = how much worse the next end fits than the best (chi-square units; a few units = they fit alike) |
| `left a-b` | leftover of the best end (1 = the average hole is fitted to its noise) |
| `rblur R` | correlation over the images between the best page CD and its fitted blur width |

Reading it (each case planted on generated images, 8 images of the same holes
with the blur changing by ±2 nm):

| The line shows | It means | Planted case |
|---|---|---|
| `best 3s` far below `rep 3s`, `moved` > 0 | the reported CD depends on the final fit's start; the fit should start from several points and keep the best end | blur 73–77 nm: `rep 99.24-100.13 3s 1.13 nc 2`, `best 99.98-100.18 3s 0.20 moved 2`, `ends 1-2 gap 710` |
| `end 3s` far below `rep 3s` | the cap cuts the fit short | (seen on single images, hidden in the folders by the fit's exchange step) |
| `nc` > 0 but `rep` = `end` = `best`, one end | the cap is hit harmlessly | blur 73–77 nm with `lens`: `nc 6`, `moved 0`, `ends 1-1` |
| several ends, large `gap`, `left` well above 1 | the model cannot follow the blur's shape; the best end is still the repeatable one | 60 nm lens in a 70 nm blur: `rep 3s 0.33`, `best 3s 0.13 moved 3`, `ends 2-3 gap 325`, `left 3.1-7.5` |
| several ends with a `gap` of a few units | two outlines fit the images alike: a question about the model | (not met on full images) |
| `rep` = `end` = `best`, one end, and still a large `3s` | the fit is at its end: the spread is in the images | (not planted) |

The remedy the check measures as `best` (several starts, each run to its end,
the lowest leftover kept) is not in the library yet.

**Result on the 8 real images (2026-11-13):**

```text
check N=8 | rep 97.13-98.88 3s 2.01 nc 5 | end 3s 2.14 it<=56 | best 97.13-98.88 3s 2.14 moved 3 | ends 1-1 gap - | left 430.8-526.0 | rblur -0.21
```

- Not the start: from 10 starts every image reaches one end (`ends 1-1`).
- Not the cap: run to their ends (56 iterations at most) the images are as far
  apart as before, 3 s 2.14 nm against 2.01 nm; three images changed when
  allowed to finish.
- **The best match is 21 to 23 times further from the real average hole than
  noise explains** (leftover 431 to 526; generated images: 1.3 to 2.0; with a
  blur shape the model cannot follow 3 to 8, and the page CD was then 1.2 nm
  off). Each image has one settled answer, and it differs by up to 1.75 nm
  between images of one area. The CD does not follow the fitted blur width
  (`rblur -0.21`).
- The leftover does not say how large that gap is, nor what it is made of: see
  the next section. Not known either: where the best match misses (the blur's
  shape, the neighbours' light, holes that are not alike, holes missing from
  the seed list), and whether `outline_fit`, which has the same model, is
  affected in the same way (not measured).

### What the leftover is, and what the images hold

The leftover has no unit and is not a brightness: it is (the distance between
the fit's best match and the average hole) / (the distance noise alone would
leave), squared, averaged over the average hole's bins. 484 = 22 times the
noise, in an 8 bit image as in any other. The noise of the average hole is one
pixel's noise / sqrt(pixels averaged per bin), about holes / 16, so the same
distance in grey levels gives a larger leftover on a quieter image and on one
with more holes. Three things raise it, and the number alone does not tell
them apart:

- the best match misses the average hole (what the model assumes does not
  hold);
- a quiet image or many holes: a small miss counts;
- few grey levels for the image's noise: with noise well below one grey step,
  rounding to whole grey steps does not average out over the holes.

```
python scripts\image_values.py D:\data\png_images
```

prints, per image and as a last line for the folder, how the files are stored
and the fit's numbers in grey steps of the file (about 2 s per image; GPU for
the fit's half, `--no-fit` without; `.tif` pages with `--page`; the library is
not changed):

```text
values N=8 | uint16 3251-23544 | levels 17175-18217 step 1 | top 0.00% zero 0.00% | noise 299 hole 14767 | gap 6.10-9.81 (5.47) = 0.04-0.06%
```

| Field | Meaning |
|---|---|
| `uint16 A-B` | file type as read; lowest and highest pixel value of all images |
| `levels a-b` | different grey values in one image (fewest, most): 256 or fewer in a 16 bit file = 8 bit content |
| `step s` | the values lie a whole number of steps apart: 1 normally, 257 or 256 = 8 bit content in a 16 bit file, 16 = 12 bit |
| `top p% zero q%` | most pixels of one image at the file type's highest value, and at 0: above about 0.01 % the values are cut off |
| `noise n` | one pixel's noise as the fit takes it (a single hole against the average hole, rms, median hole), in grey steps; differences between holes are in it |
| `hole h` | brightness of the average hole: its top above its darkest point |
| `gap a-b (n) = p%` | distance between the best match and the average hole (rms, each bin by its pixels), lowest and highest image; in brackets what noise alone leaves; then the gap in percent of the hole's brightness. (gap / bracket)^2 = leftover |

Generated images (26,000 to 28,000 holes of 100 nm, 55 to 59 grey levels
bright in 8 bit; the noise is the one printed):

| Case | noise | gap (noise alone) | % of the hole | leftover | CD error |
|---|---|---|---|---|---|
| the model fits, usual noise | 1.2 | 0.022-0.038 (0.019-0.030) | 0.04-0.06 | 1.2-1.8 | < 0.1 nm |
| a blur the model cannot describe (60 nm lens in 70 nm) | 1.2 | 0.034-0.054 (0.019) | 0.06-0.10 | 3-8 | 1.2 nm |
| 8 bit, quiet image (true noise 0.1 grey level) | 0.39 | 0.058-0.069 (0.006-0.010) | 0.10-0.12 | 49-95 | 0.3-0.4 nm low |
| the same quiet image stored in 16 bit | 71 | 3.1-3.9 (1.2-1.8) | 0.02-0.03 | 4.6-6.9 | < 0.1 nm |

- **Bit depth as such changes nothing.** One image stored in 8 bit, the same
  values in a 16 bit file (x 257), with 257 times finer steps, and as 12 bit
  values in a 16 bit file: page CD 100.07 nm four times, leftover 1.30 to
  1.36. The files are read at their full depth (`PIL`, then `float32` on the
  GPU, sums in `float64`), and the fit has no constant in grey levels.
- **Grey steps too coarse for the noise do.** The same holes at falling noise,
  each stored in 8 and in 16 bit (noise between the holes in 8 bit grey
  levels; true CD 99.98 nm):

  | True noise | 8 bit: CD, leftover | 16 bit: CD, leftover |
  |---|---|---|
  | 1.0 | 100.10, 1.2 | 100.11, 1.3 |
  | 0.5 | 100.07, 2.0 | 100.07, 2.3 |
  | 0.3 | 100.01, 4.5 | 100.07, 3.7 |
  | 0.2 | 99.82, 23 | 100.05, 5.0 |
  | 0.1 | 99.61, 86 | 100.03, 6.9 |

  At a pitch that is not a whole number of pixels (10.3 px) the same: 99.66 nm
  and 49 against 99.94 nm and 4.6 at a noise of 0.1.
- **A gap far too small to see already moves the CD.** In the second row of
  the first table the best match is 0.05 grey levels from an average hole 54
  grey levels bright (0.1 %), and the page CD is 1.2 nm off.
- The leftover of a generated image is not the leftover of a real one even for
  the same miss: besides noise and hole count, the generated pages have a
  pitch of exactly 10 px, so 1 bin in 16 of the average hole holds nearly all
  pixels and the others a few dozen (the leftover is a mean over bins).

Not yet run on the real images.

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
