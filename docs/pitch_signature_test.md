# Pitch signature test: which center method shows a 1 nm row shift?

**Goal.** Real pages carry a known y-pitch signature: rows about 1 µm apart sit about 1 nm off. It only shows
after many columns are averaged. On synthetic pages where every center is known, this test asks how few columns
each center method needs to show it.

**Picture.** Holes on a grid; every 5th row (1 µm at 200 nm pitch) moved 1 nm down. The y pitch between rows r and
r + 1, averaged over N columns, then shows +1 nm into the moved row and −1 nm out of it, and is flat in between. Two
things blur that curve:

- the mask's own random placement (0.2 nm per hole here): no method removes it, only averaging more columns;
- the method's own center error: random error averages away with more columns, but an error that depends on where
  the hole sits inside its pixel is shared by the whole row (rows run along the pixel rows) and never does.

Three numbers per method answer the question: **columns needed** (fewest columns whose curve noise is at most a
third of the 1 nm step), **center error** (nm per hole) and **pixel-position error** (nm, the part that never
averages away).

## Run

    python scripts/pitch_signature_test.py                       # 13 pages, all methods, ~11 min on an RTX 4060 Ti
    python scripts/pitch_signature_test.py --images square300_square --placement-sigma 0 --output artifacts/ps_quick
    python scripts/pitch_signature_test.py --help                 # every setting below is an option

Writes into `artifacts/pitch_signature/` (git-ignored): `report.html` (open in a browser, no network needed),
`results.json`, `pitch_signature.csv` (one row per page × method), `settings.json`. `--backend cpu` runs the CD
methods and the prototype on the CPU (the GPU finders need a GPU; `--finders` with no names skips them).
`python scripts/pitch_signature_view.py artifacts/pitch_signature/results.json` rewrites the report alone.

## Test pages

Rendered by `subpx.synthetic_arrays.render_hole_array`: each hole's light is computed exactly from its shape's
Fourier transform × blur × pixel area at its exact center, so the rendering adds no center error (a noiseless
hole's pixel centroid equals its true center to < 1e-6 px; `tests/test_synthetic_arrays.py`).

| page | holes (nm) | layout | x pitch / row pitch (nm) | moved rows |
|---|---|---|---|---|
| circle250_square | circles 250 | square | 500 / 500 | every 5th (2.5 µm) |
| circle100_square | circles 100 | square | 200 / 200 | every 5th (1.0 µm) |
| circle100_stagger | circles 100 | staggered | 200 / 175 | every 6th (1.05 µm) |
| circle80_square | circles 80 | square | 160 / 160 | every 6th (0.96 µm) |
| circle60_dense | circles 60 | staggered | 140 / 121 | every 8th (0.97 µm) |
| circle50_dense | circles 50 | staggered | 120 / 104 | every 10th (1.04 µm) |
| square300_square | squares 300 | square | 600 / 600 | every 5th (3.0 µm) |
| square150_stagger | squares 150 | staggered | 300 / 300 | every 5th (1.5 µm) |
| ellipse160x80_square | ellipses 160 × 80 (wide) | square | 320 / 200 | every 5th (1.0 µm) |
| ellipse100x200_stagger | ellipses 100 × 200 (tall) | staggered | 300 / 300 | every 5th (1.5 µm) |
| ellipse90x60_stagger | ellipses 90 × 60 (wide) | staggered | 180 / 120 | every 8th (0.96 µm) |
| ellipse60x100_square | ellipses 60 × 100 (tall) | square | 140 / 180 | every 6th (1.08 µm) |
| ellipse70x45_dense | ellipses 70 × 45 (wide) | staggered | 140 / 100 | every 10th (1.0 µm) |

Every page: 2048 × 1378 px, 21 nm pixels, 8-bit; bright holes (background 15, open area 103 grey levels); blur
65 nm wide at half height (a 45 nm lens with faint rings + scan smear); 30 photons per grey level + read noise 0.65;
per hole CD noise 0.5 nm, placement noise 0.2 nm (x and y), brightness 5 % (sd); neighbouring holes at least 70 nm
apart (edge to edge). Moved rows: +1.0 nm in y, the whole number of rows closest to 1 µm apart but at least 5
(`--row-period`, `--min-rows-apart`), so each moved row shows as its own up-down pair; at 500–600 nm pitch 1 µm
would be every 2nd row and the curve a zig-zag even for perfect centers. Rows parallel to the pixel rows
(`--rotation` tilts them). These are test assumptions, not a calibrated ACTIS model.

## What the report shows

**Graphs.** Every y-pitch graph is a stack of panels on one x axis (µm down the page), one line per panel, all on the
same scale (±2 nm): the true centers in grey on top, measured methods in blue below. The line is the y pitch minus
the design pitch, averaged over N columns in the middle of the page. Truth and measurement are never drawn over each
other, and nothing marks where the moved rows are: the grey panel shows it from the data alone.

**One image at a time.** Image chips above; on the left a list of every method with its columns needed (colored
chip) and center error. Click a method, or use ↑ / ↓ once the list has focus, and the graph changes in place; the
list stays beside the graphs while the page scrolls. "Compare with" picks a second method for the bottom panel (the
image's best GPU finder by default). Below: curve noise against columns, error against place inside the pixel, every
method at once, and the image's table.

**Numbers.**

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

## Results (2026-11-03, seed 1, defaults above, 664 s on the RTX 4060 Ti)

The run was done twice with identical numbers; another program shared the GPU for part of each run, so every time
in the report is the faster of the two. The dense pages are the slow ones: 50 nm circles 150 s (one CD call 90 s),
70 × 45 nm ellipses 100 s, 60 nm circles 66 s; the others 25–45 s.

Columns needed to show the 1 nm step (with the mask's 0.2 nm placement noise), and in brackets the center error
per hole (y, nm):

| page | perfect centers | best CD method | best GPU finder | template prototype | best possible per hole |
|---|---|---|---|---|---|
| circle250_square | 1 | Gaussian fit 4 (0.28) | edge-erf 4 (0.28) | 2 (0.20) | 0.20 |
| circle100_square | 1 | Gaussian fit 4 (0.39) | log-quadratic 8 (0.47) | 4 (0.38) | 0.36 |
| circle100_stagger | 1 | Gaussian fit 4 (0.38) | radial symmetry 8 (0.51) | 4 (0.39) | 0.36 |
| circle80_square | 1 | Gaussian fit 8 (0.48) | log-quadratic 16 (0.60) | 8 (0.49) | 0.44 |
| circle60_dense | 1 | Gaussian fit 16 (0.72) | radial symmetry 32 (0.93) | 16 (0.75) | 0.62 |
| circle50_dense | 1 | Gaussian fit 32 (1.06) | radial symmetry 64 (1.40) | 32 (1.11) | 0.80 |
| square300_square | 1 | edge area 4 (0.38) | edge-erf 4 (0.28) | 2 (0.18) | 0.17 |
| square150_stagger | 1 | Gaussian fit 4 (0.29) | radial symmetry 8 (0.54) | 2 (0.26) | 0.25 |
| ellipse160x80_square | 1 | Gaussian fit 4 (0.30) | log-quadratic 4 (0.36) | 4 (0.30) | 0.28 |
| ellipse100x200_stagger | 1 | Gaussian fit 4 (0.41) | radial symmetry 16 (0.78) | 4 (0.36) | 0.35 |
| ellipse90x60_stagger | 1 | Gaussian fit 8 (0.52) | radial symmetry 16 (0.72) | 8 (0.55) | 0.46 |
| ellipse60x100_square | 1 | Gaussian fit 8 (0.56) | log-quadratic 16 (0.70) | 8 (0.56) | 0.52 |
| ellipse70x45_dense | 1 | Gaussian fit 16 (0.85) | radial symmetry 64 (1.16) | 32 (0.91) | 0.66 |

What it says:

1. **The true centers show the step with 1 column on every page** (0.28 nm curve noise from the 0.2 nm placement
   noise), so the columns needed come from the methods.
2. **The best CD method needs 4 columns on 7 pages**, 8 on the 80 nm circles and the 90 × 60 and 60 × 100 nm
   ellipses, 16 on the dense 60 nm circles and 70 × 45 nm ellipses, 32 on the 50 nm circles. It is the Gaussian fit
   on 12 of the 13 pages. On 300 nm squares the Gaussian, log, gradient-edge and half-height fits never show the
   step (their error repeats with the place in the pixel, shared by each row); edge area needs 4 there.
3. **The best GPU finder never needs fewer columns than the best CD method**: as many on 3 pages (edge-erf on
   250 nm circles and 300 nm squares, log-quadratic on 160 × 80 nm ellipses), more on the other 10 (8–64 columns).
   Weighted never shows the step on 11 of the 13 pages and isophote on 10 (pixel-position error 0.5–1.2 nm for
   weighted, up to 4.6 nm for isophote); edge-moment finds 43–70 % and edge-erf 4–32 % of the holes below 250 nm;
   log-quadratic finds almost none of the 250–300 nm holes.
4. **Small, dense holes are where methods fall short of the best possible.** The Gaussian fit's center error is
   5–17 % above it on the 80–200 nm holes, 15 % on the 60 nm circles and 30–32 % on the 50 nm circles and 70 × 45 nm
   ellipses, where neighbours' light overlaps. There every method also reads the step too large: Gaussian fit
   9–19 % on the three dense pages, template fit 18–30 %. (Large holes leave room too: 41 % above on the 250 nm
   circles, where the template fit reaches the best possible.)
5. **Never shows the step:** the projection methods (center error 2–6 nm per hole): projection on no page, raw-cell
   projection only on the 50 nm circles, and only with all 356 columns.

## Template-fit prototype (not in subpx)

`scripts/template_fit_prototype.py` (NumPy or CuPy), started from the GPU weighted centroids. Each hole is fitted as
brightness × the page's own average hole, shifted, + background, on the pixels closer to it than to any neighbour;
the average hole is rebuilt from the new centers, 3 rounds. It assumes no shape and ignores per-hole brightness.

- It removes its start's pixel-position error (weighted 1.18 → 0.02 nm on 100 nm circles).
- It needs fewer columns than the best existing method on 3 of 13 pages (250 nm circles, 300 nm and 150 nm squares:
  2 instead of 4; center error 0.20, 0.18 and 0.26 nm, at the best possible), as many on 9 and more on 1 (70 × 45 nm
  dense ellipses: 32 instead of 16).
- Why it loses on dense pages: its average hole carries the neighbours' light, so a moved row also moves its
  neighbour rows' fits; the step reads 18–30 % too large on the three dense pages (Gaussian fit 9–19 %).
- Its average hole is built from holes at different heights inside their pixel; with few such heights (rows
  8.33 px apart give 3) it must be smoothed by 0.7 px, or it keeps a once-per-pixel ripple.
- Not added to the library: the bar set before the first test was "best on at least 6 of 8 pages"; it was best on 4
  of 8 then, and now needs fewer columns than every existing method on 3 of 13 pages, all with large holes.

## Limits of this test

- Rows run along the pixel rows (as on the user's pages): the worst case for pixel-position error. A tilt
  (`--rotation`) spreads that error along a row and helps every method.
- Not on these pages: brightness or background changing row by row, stage or scan wobble, distortion, the real ACTIS
  blur shape, line/space areas beside the holes, larger or correlated placement noise.
- The blurred-hole fit and flux ratio are given the true blur here; real pages lack it, so they are not counted as
  "best".

## First version (2026-11-03, commit 9ec9b0a)

The first version used 0.5 nm placement noise and 1 nm CD noise, and moved the rows closest to 1 µm apart on every
page, which was every 2nd row at 500–600 nm pitch. Even perfect centers needed 8 columns on every page, so the good
methods all looked the same, and on large pitches the pitch curve was a zig-zag. Its graphs drew the planted step,
the true centers and the method as three lines over each other. `--placement-sigma 0.5 --cd-sigma 1
--min-rows-apart 2` goes back to those settings (the page seeds follow the page order, which changed).

## Fixed on the way

The GPU edge-erf finder refused every edge more than ~140 px from the page's left or top edge; see `docs/centers.md`.
