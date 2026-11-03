# Pitch signature test: which center method shows a 1 nm row shift?

**Goal.** Real pages carry a known y-pitch signature: rows about 1 µm apart sit about 1 nm off. It only shows
after many columns are averaged. On synthetic pages where every center is known, this test asks how few columns
each center method needs to show it.

**Picture.** Holes on a grid; every 5th row (about 1 µm) moved 1 nm down. The y pitch between rows r and r + 1,
averaged over N columns, then shows +1 nm into the moved row and −1 nm out of it. Two things blur that curve:

- the mask's own random placement (0.5 nm per hole here): no method removes it, only averaging more columns;
- the method's own center error: random error averages away with more columns, but an error that depends on where
  the hole sits inside its pixel is shared by the whole row (rows run along the pixel rows) and never does.

Three numbers per method answer the question: **columns needed** (fewest columns whose curve noise is at most a
third of the 1 nm step), **center error** (nm per hole) and **pixel-position error** (nm, the part that never
averages away).

## Run

    python scripts/pitch_signature_test.py                       # 8 pages, all methods, ~4 min on an RTX 4060 Ti
    python scripts/pitch_signature_test.py --images square300_square --placement-sigma 0 --output artifacts/ps_quick
    python scripts/pitch_signature_test.py --help                 # every setting below is an option

Writes into `artifacts/pitch_signature/` (git-ignored): `report.html` (open in a browser, no network needed),
`results.json`, `pitch_signature.csv` (one row per page × method), `settings.json`. `--backend cpu` runs the CD
methods and the prototype on the CPU (the GPU finders need a GPU; `--finders` with no names skips them).

## Test pages

Rendered by `subpx.synthetic_arrays.render_hole_array`: each hole's light is computed exactly from its shape's
Fourier transform × blur × pixel area at its exact center, so the rendering adds no center error (a noiseless
hole's pixel centroid equals its true center to < 1e-6 px; `tests/test_synthetic_arrays.py`).

| page | holes (nm) | layout | x pitch / row pitch (nm) | moved rows |
|---|---|---|---|---|
| circle100_square | circles 100 | square | 200 / 200 | every 5th |
| circle100_stagger | circles 100 | staggered | 200 / 175 | every 6th |
| circle60_dense | circles 60 | staggered | 140 / 121 | every 8th |
| circle250_square | circles 250 | square | 500 / 500 | every 2nd |
| square300_square | squares 300 | square | 600 / 600 | every 2nd |
| square150_stagger | squares 150 | staggered | 300 / 300 | every 3rd |
| ellipse160x80_square | ellipses 160 × 80 (wide) | square | 320 / 200 | every 5th |
| ellipse100x200_stagger | ellipses 100 × 200 (tall) | staggered | 300 / 300 | every 3rd |

Every page: 2048 × 1378 px, 21 nm pixels, 8-bit; bright holes (background 15, open area 103 grey levels); blur
65 nm wide at half height (a 45 nm lens with faint rings + scan smear); 30 photons per grey level + read noise 0.65;
per hole CD noise 1 nm, placement noise 0.5 nm (x and y), brightness 5 % (sd); moved rows +1.0 nm in y, the whole
number of rows closest to 1 µm apart. Rows parallel to the pixel rows (`--rotation` tilts them). These are test
assumptions, not a calibrated ACTIS model.

## What the report shows

- **Found**: share of holes (one pitch or more from the page edge) with a center within half a spacing of the true
  one.
- **Center error**: sd of measured − true y.
- **Pixel-position error**: half the peak-to-peak of the mean y error against where the true center sits inside its
  pixel (16 bins).
- **y pitch**: as `analyze_pitch`, the mean of the (row r, column k) → (row r + 1, column k) pairs over a band of N
  columns, for every band of the page. Curve noise = root-mean-square error over all bands and row intervals after
  removing each band's mean, against design + planted step (**with mask noise**: what one would see) or against the
  true centers (**method alone**).
- **Columns needed**: fewest N in 1, 2, 4, … 128, all, whose curve noise is ≤ 1/3 nm. "Never" = not even with all
  columns.
- **Step seen**: least-squares size of the planted pattern in the all-columns curve (true centers: about 1.00 nm).
- **Best possible per hole**: the Cramér–Rao bound for one hole alone (brightness and background fitted too; photon,
  read and rounding noise), averaged over 16 places inside the pixel; neighbours and CD noise left out.

Methods: the 19 CD methods from one `measure_cds(..., backend="gpu", summary_only=True)` call (the blurred-hole fit
and flux ratio are given the true blur, which real pages lack, so they are left out of "best"); the 6 GPU finders,
`detect_centers(backend="gpu", refine=...)` with Otsu threshold and area limits from the design hole, as the
workbench's Pitch panel offers them; the template-fit prototype below; the true centers as a check. Best CD method /
best GPU finder per page = fewest columns needed with ≥ 99 % found (ties: smaller method-alone noise at 1 column).

## Results (2026-11-03, seed 1, defaults above, 250 s on the RTX 4060 Ti)

Columns needed to show the 1 nm step (with the mask's 0.5 nm placement noise), and in brackets the center error
per hole (y, nm):

| page | perfect centers | best CD method | best GPU finder | template prototype | best possible per hole |
|---|---|---|---|---|---|
| circle100_square | 8 | Gaussian fit 8 (0.39) | log-quadratic 16 (0.47) | 8 (0.38) | 0.36 |
| circle100_stagger | 8 | Gaussian fit 8 (0.39) | radial symmetry 16 (0.51) | 8 (0.39) | 0.36 |
| circle60_dense | 8 | Gaussian fit 16 (0.72) | radial symmetry 32 (0.94) | 32 (0.75) | 0.62 |
| circle250_square | 8 | Gaussian fit 8 (0.28) | edge-erf 8 (0.28) | 8 (0.20) | 0.20 |
| square300_square | 8 | edge area 8 (0.36) | edge-erf 8 (0.28) | 8 (0.17) | 0.17 |
| square150_stagger | 8 | Gaussian fit 8 (0.29) | radial symmetry 16 (0.54) | 8 (0.26) | 0.25 |
| ellipse160x80_square | 8 | Gaussian fit 8 (0.30) | log-quadratic 8 (0.36) | 8 (0.30) | 0.28 |
| ellipse100x200_stagger | 8 | Gaussian fit 8 (0.41) | radial symmetry 32 (0.79) | 8 (0.36) | 0.35 |

What it says:

1. **The mask's own placement noise sets the floor.** With 0.5 nm per hole even perfect centers need 8 columns on
   every page. No center method can go below that; only a mask with less placement noise can
   (`--placement-sigma 0.25` to try). Without it ("method alone" in the report) the best methods need 1–4 columns.
2. **The best CD methods are already close to the best possible.** The best CD method reaches the floor on 7 of 8
   pages; the Gaussian fit alone on 6, its center error 6–17 % above the best possible per hole on small and
   elongated holes. It misses on the dense 60 nm array (16 columns: neighbours' light overlaps) and on 300 nm
   squares, where the Gaussian, log, gradient and half-height fits never show the step (their pixel-position error,
   0.37–0.53 nm, is shared by the whole row); edge area does there.
3. **The GPU finders reach the floor on 3 of 8 pages** (edge-erf on 250 nm circles and 300 nm squares,
   log-quadratic on 160 × 80 ellipses); elsewhere 16–32 columns. Weighted and isophote never show the step except
   on the dense array (64 and 128 columns; pixel-position error 0.5–4.6 nm elsewhere); edge-moment and edge-erf find
   only 7–71 % of the holes below 250 nm; log-quadratic finds none of the 250–300 nm holes.
4. **Never shows the step on any page:** the projection methods (center error 2–6 nm).

## Template-fit prototype (not in subpx)

`scripts/template_fit_prototype.py` (NumPy or CuPy), started from the GPU weighted centroids. Each hole is fitted as
brightness × the page's own average hole, shifted, + background, on the pixels closer to it than to any neighbour;
the average hole is rebuilt from the new centers, 3 rounds. It assumes no shape and ignores per-hole brightness.

- It removes its start's pixel-position error (weighted 1.19 → 0.02 nm on 100 nm circles) and reaches the floor on
  7 of 8 pages, at 5–11 s per page.
- Set before the test: added to subpx and the workbench only if best on at least 6 of 8 pages (method-alone noise
  at 1 and at 8 columns below every existing method's, step within 5 %, ≥ 99 % found). It is best on 4: 250 nm
  circles, 300 and 150 nm squares, 100 × 200 ellipses (9–41 % less noise than the best method usable on real pages;
  on 300 nm squares 0.17 nm per hole, about the best possible, against 0.28 nm for edge-erf). It ties on 100 nm circles in
  a square array and on 160 × 80 ellipses, and loses on staggered 100 nm circles (2–6 % more noise) and the dense
  60 nm array (6–10 % more). So: within about 10 % of the best existing method on every page, which no other
  method manages, but not best on most.
- Why it loses: its average hole carries the neighbours' light, so a moved row also moves its neighbour rows' fits:
  the step reads 4–5 % too large on 100 nm circles and 160 × 80 ellipses and 14 % too large on the dense array
  (Gaussian fit: at most 2 % and 5 %).
- Its average hole is built from holes at different heights inside their pixel; with few such heights (rows
  8.33 px apart give 3) it must be smoothed by 0.7 px, or it keeps a once-per-pixel ripple.

## Limits of this test

- Rows run along the pixel rows (as on the user's pages): the worst case for pixel-position error. A tilt
  (`--rotation`) spreads that error along a row and helps every method.
- Not on these pages: brightness or background changing row by row, stage or scan wobble, distortion, the real ACTIS
  blur shape, line/space areas beside the holes, larger or correlated placement noise.
- The blurred-hole fit and flux ratio are given the true blur here; real pages lack it, so they are not counted as
  "best".

## Fixed on the way

The GPU edge-erf finder refused every edge more than ~140 px from the page's left or top edge; see `docs/centers.md`.
