# Outline fit in one pass: builder's notes

**Status, 2026-11-12.** A measured scratch prototype; nothing in `subpx/` has been changed. The user approved
building it as a new method beside `outline_fit`, first in the folder reports and with page values only; **what to
build and how to check it is in `docs/outline_fit_fast_handoff.md`**. This file is the reference behind that
handoff: the exact method, the settings, every measurement, what was tried and dropped. The page the user read is
`docs/outline_fit_one_pass.html`. It replaces `docs/outline_fit_speed.md` as the plan for the batch; that file's
measurements of the existing fit still stand.

Added later the same day, after running the prototype the way the reports will (their seeds, small images) and
from many starts: findings 6.13 to 6.17 (the fit's end points, small images, the gauss2 exchange step now in the
prototype, the pair reach, repeats). Section 5's tables were made before the exchange step; with it they are
reproduced within 0.0004 nm (`runs/*_v5.*`).

The prototype is in `artifacts/outline_fit/fast/` (its method and measurement scripts are tracked in git although
`artifacts/` is ignored; the run results are on the development PC only); its `README.txt` lists every script and
command.

## 1. The task, in the user's words (2026-11-12)

- One TIFF is one image of up to 1,500 pages, 2 to 4 GB. There are thousands of them.
- Not every page holds the pattern. The user selects what to measure by phase cross-correlation against a reference
  page; the reference is cut into tiles so that a page holding the pattern on part of its area still contributes
  that part.
- Reading one TIFF takes about one minute. The fit of one TIFF must take less than reading it. One page at a time is
  only acceptable at hundreds of milliseconds per page. Some accuracy may be given up if it cannot be avoided.
- The earlier proposal "has some limits and misunderstood the actual application at first".

One minute per 1,500 pages is 40 ms per page for the whole machine; with the remote's four GPUs, 160 ms per page and
GPU.

## 2. Why the existing fit cannot get there

`PageFit.run` alternates: hole stage (every hole against the page model, template fixed) and shape stage (outline
and blur fitted to the average hole, from which the neighbours' light was removed with the current template and the
current holes). The average depends on the template through that cleaning, so the two stages are solved in turn,
with half steps and three released stages: 10 to 14 rounds, 43 hole sweeps and about 4,000 template renders per
page. `docs/outline_fit_speed.md` made the passes cheaper (104 s to 10 s per page measured) and kept the rounds.

## 3. The method

Same model as `subpx._outline` (one sharp outline, one blur, per hole a centre, a brightness, a level and an edge
offset), same parameter vector `theta`, bounds, stages, grid of starts and CD definition.

### 3.1 The average hole with the neighbours left in

Page model: `I(p) = sum_j a_j T(p - c_j) + level`, `T = pixel box * blur * indicator(outline)`.

Plain stack, `U` bins per pixel (prototype: 4): every pixel `p` of the window of a stacked hole `i` goes to bin
`n = rint((p - c_i) U)` per axis; `S(b) = mean of I(p)` over the samples in the bin. No weights and no division by
a fitted brightness (section 6.2).

All bins with the same remainder `q = n mod U` are filled by the same holes `H_q` (those whose centre sits at that
sub-pixel position), and each such hole puts its samples at the same sub-bin remainder
`s_i = (-c_i U - rint(-c_i U)) / U`. Putting the page model into the stack:

    S(b) = const + (1 / L^2) sum_f T^(f) Gamma_q(f) exp(2 pi i f . u_b)            u_b = n / U
    Gamma_q(f) = (1 / |H_q|) sum_{i in H_q} sum_j (a_j / a_mean) exp(-2 pi i f . (c_j - c_i - s_i))

`j` runs over every hole within `reach` of `i`, `j = i` included (the hole itself; with `s_i` this also replaces
the library's box correction for the bin). `Gamma_q` needs the centres and rough brightnesses only. It is exact for
any arrangement: no lattice, and no assumption that a hole's sub-pixel position says nothing about where its
neighbours stand. On the test file it does say so (pitch exactly 8.5 px, shift within +-0.3 px): one common map
left stripes of 2 grey levels at the neighbours, 6 times the noise.

How `Gamma_q` is computed: cell list of the holes; one GPU thread per stacked hole splats `(c_j - c_i - s_i)` of its
neighbours with a cubic B-spline onto a grid of 8 per pixel for its class (`U^2` grids of `(8 L)^2` float32); one
batched `rfft2`; the band's rows and columns are kept and divided by the spline's transfer `sinc^4`.

Sizes: window half width `R = max(3, ceil(0.75 x spacing))` (library's rule); `reach = R + 13` px; box
`L = 8 ceil((2 reach + 8) / 8)` px (48 px for the test file); frequencies `k / L` inside `|f| <= 0.6` cycles per
pixel, half plane (1,329 at `L = 48`); bins used in the fit: `|n| <= floor((R - 0.5) U)` (the bins every hole of the
class fills; the outermost bins are filled by part of a class and are left out).

### 3.2 The page fit

`T^(f) = O^(f) x OTF(f) x sinc(fx) sinc(fy)`, every factor in closed form:

- `O^` (the outline's transform) by Green's theorem as a line integral round the outline, trapezoid rule on 256
  equal angles with the outline's own tangent:
  `O^(f) = i / (2 pi |f|^2) sum_m exp(-2 pi i f . x_m) (fx y'_m - fy x'_m) dphi`, `O^(0) = area`.
- Its derivative with respect to an outline parameter `p` uses the same exponentials:
  `dO^/dp = sum_m exp(-2 pi i f . x_m) r_m (dr/dp)_m dphi`. `dr/dp` is analytic for the superellipse and the
  harmonics. The blur's derivatives are analytic.
- So one pass over the outline gives the model and its whole Jacobian. One GPU kernel (thread per page and
  frequency, single-precision `sincos`, double sums); the synthesis at the bins is two small batched matrix
  products in single precision; the sums that enter the normal equations are in double.
- Scale and constant are solved exactly for every `theta` (variable projection, Kaufman's Jacobian).
- Levenberg-Marquardt on the host for a batch of pages in lockstep (the device returns cost, gradient and
  Gauss-Newton matrix per page in one download): gain-ratio damping; bounds as an active set (a parameter that the
  step would carry past a bound is put on it and the others are solved again; clipping a solved step broke the
  fit); weak Gaussian priors on what a page may leave undetermined (section 6.3).
- Stages as in the library. First fit: a first guess of the size from the stack's second moment, the grid of
  starts round it (15 sizes x 12 blur widths, compared on the bins of the page's most populated class), `round`
  (at most 10 iterations), `super` (12), `ftol 1e-6`. Final fit: `full` from the first fit's result, at most 30
  iterations, `ftol 1e-7`.

### 3.3 The procedure

    A  seeds -> plain stack -> Gamma -> first page fit                                  theta_0
    B  hole pass, size held: centre, brightness, level of every hole, two sweeps        (centres)
    C  plain stack with B's centres -> Gamma -> final page fit from theta_0             theta_1 = the page
       (gauss2: a fit that ends with the two Gaussians exchanged is run again; 6.15)
    D  (optional) hole pass, size free, three sweeps with theta_1                       edge offset per hole

- Seeds: Gaussian smoothing (sigma 1 px), local maxima within +-2 px above `lo + 0.3 (hi - lo)` (5 % and 99.8 %
  levels of the page), parabola through the log of three samples per axis. If the mean blob above the level is
  wider than the search window, the search repeats with a window of 0.6 blob widths (flat-topped big holes).
- Brightness for `Gamma` in A: least squares of each window against the stack inside 0.5 spacings; in C: B's.
- B and D are the library's hole stage (`fit_holes`: Gauss-Seidel over colours, complete page model, robust
  weights, the same bounds on move, brightness and offset) as fused kernels: `hole_normal` (per hole and window
  row), `hole_update` (per hole: 5 x 5 solve and the bounded step), `render_scatter` (per hole and render pixel).
  Differences from the library: colours from a cell hash (cells of 1.5 spacings; parity of the cell x position in
  the cell: 16 colours on the test file instead of 6); templates in single precision; near / far split with a 2 px
  ramp; the far field is one FFT convolution of brightness x (1 + 4 offset / CD) (the library convolves three
  terms); the far kernel is built once per page; the noise line (variance = v0 + v1 level) from a histogram of
  the leftovers after B; in D every step in brightness, level and offset is taken 1.3 times as far.
- Outputs per page: outline CD, x / y extents, blur, chi-square per bin of the final fit, the first fit's CD;
  with D per hole: centre, brightness, level, edge offset, CD (offsets reported as deviations from their mean over
  the good holes, as the library does), leftover rms, standard error of the offset.

## 4. Prototype settings

| Setting | Value | Note |
|---|---|---|
| Stack bins per pixel `U` | 4 | 16 sub-pixel classes; 8 was not needed once the sub-bin remainder is in the model |
| Pair grid | 8 per pixel, cubic B-spline | aliasing of the splat below 1e-4 at 0.6 c/px |
| Band | `abs(f) <= 0.6` c/px | the blur's transfer is below 1e-5 there for blur sigma >= 1.2 px |
| Outline points | 256 (page fit), 256 (templates), 128 (far kernel) | |
| Template grid for the hole passes | 8 per pixel, box `max(32, 2 (R + 6))` px | library: 24 or 48 px |
| Sweeps | B 2, D 3 with step factor 1.3 | section 6.5, 6.6 |
| Pair reach | `R + 13` px (20 or 21 px on the pages used) | 30 px changes which lens end fits better: 6.16 |
| gauss2 exchange step | on, after C | 6.15 |
| Pages per call | 20 | 10 to 40 make no difference |

## 5. Measurements

All on the development PC (RTX 4060 Ti 16 GB, Ryzen 5 7500F), prototype defaults, pages in memory.

<!-- measurements: begin (make_notes.py) -->
### 5.1 Time

The 40 hole pages of the test file (39,042 or 37,842 holes each), 20 pages per call, three repeats.

| | ms per page | 1,500-page image, this GPU | four such GPUs (not measured) |
|---|---|---|---|
| Page values (A to C) | 80 (78 to 89) | 2.0 min | 30 s |
| With a size per hole (A to D) | 137 (129 to 151) | 3.4 min | 51 s |
| Existing fit, library | 104,000 to 150,000 | 2 to 2.5 days | |
| Existing fit, same-numbers prototype of the earlier proposal | 11,043 (9,400 to 14,700) | 4.6 h | |

By phase, ms per page: A stack 11.2, A page fit 11.6, B hole pass + stack 42.6, C page fit 12.3, D hole pass 53.1.
Inside them: upload 0.5, seeds 3.9, spacing 1.0, stacks 7.4, pair maps 6.2, both page fits 15.4 at 20 pages per call, templates 12.3, hole-pass set-up 11.9, five sweeps 60.3, noise line 1.3, results 2.4.
One sweep (39,042 holes, window 15 x 15 px, render 23 x 23 px): `hole_normal` 6.3 ms, `render_scatter` 4.5 ms, `hole_update` 0.5 ms.
Pages per call 4 / 10 / 20 / 40: 111 / 89 / 80 / 87 ms (page values), 164 / 144 / 136 / 134 ms (with sizes).
GPU busy 67 % (page values) and 72 % (with sizes); one CPU core busy all the time; GPU memory pool 2.8 to 3.0 GB at 20 pages per call, 5.5 GB at 40.
Remote estimate on the page (19 to 30 s and 30 to 51 s per image): lower end = GPU share of the time x 1 / 2.3 (the remote's ratio for the other CD methods, 3.5 s against 7.9 s per page), CPU share unchanged, four GPUs; upper end = this PC's time / 4.

### 5.2 Generated pages with known sizes

`subpx.synthetic_arrays.render_hole_array`, 2048 x 1378 px, 21 nm pixels, blur 65 nm with the lens term 45 nm, sizes spread 2.5 nm, placement 1 nm, brightness 5 %, photon noise 30 per level, unless said. Error = page CD - mean true CD of the stacked holes (squares: against the area-equivalent diameter). Response = slope of the fitted against the true hole CD; scatter in nm of true CD; light = move for a 10 % brighter hole. Existing = the existing fit on the same page (`*` = not settled in 14 rounds).

| Page | Blur form | One pass: error | response | scatter | light | Existing: error | response | scatter | light |
|---|---|---|---|---|---|---|---|---|---|
| 100 nm circles, square array | gauss2 | -0.137 | 1.037 | 1.22 | +0.142 | -0.476 | 1.055 | 1.21 | +0.192 |
| 100 nm circles, square array | lens | +0.014 | 1.002 | 1.24 | -0.018 | -0.198 | 1.008 | 1.24 | +0.001 |
| 80 nm circles, square array | gauss2 | -0.820 | 1.029 | 2.00 | -0.029 | -1.028 * | 1.083 | 1.92 | +0.198 |
| 80 nm circles, square array | lens | -0.231 | 1.023 | 1.98 | +0.018 | -0.228 | 1.024 | 1.97 | +0.068 |
| 100 nm circles, staggered | lens | +0.087 | 1.003 | 1.30 | -0.015 | +0.789 * | 0.979 | 1.26 | +0.050 |
| 150 nm squares, staggered | gauss2 | +0.112 | 1.015 | 0.60 | +0.106 | +0.207 * | 1.008 | 0.61 | +0.191 |
| 150 nm squares, staggered | lens | +0.120 | 1.004 | 0.61 | -0.010 | +0.194 * | 1.003 | 0.61 | +0.150 |
| 250 nm circles, square array | lens | +0.139 | 0.995 | 0.40 | -0.049 | +0.137 | 0.995 | 0.40 | -0.042 |
| 160 x 80 nm ellipses | lens | -0.130 | 0.996 | 1.17 | +0.005 | -2.010 * | 0.994 | 1.14 | +0.107 |
| 100 x 200 nm ellipses | lens | -0.003 | 0.994 | 0.83 | +0.007 | -0.111 * | 1.011 | 0.82 | +0.229 |
| 100 nm circles, square array (blur 58 nm) | lens | +0.033 | 1.003 | 1.02 | -0.001 | -0.075 | 1.001 | 1.01 | -0.005 |
| 100 nm circles, square array (blur 72 nm) | lens | -0.115 | 1.025 | 1.49 | +0.080 | -0.327 | 1.029 | 1.48 | +0.087 |
| 100 nm circles, square array (blur 58 nm) | gauss2 | +0.003 | 1.010 | 1.01 | +0.036 | | | | |
| 100 nm circles, square array (blur 72 nm) | gauss2 | -0.164 | 1.052 | 1.46 | +0.212 | | | | |
| 80 nm circles, square array (blur 58 nm) | lens | -0.063 | 1.002 | 1.55 | +0.023 | | | | |
| 80 nm circles, square array (blur 72 nm) | lens | -0.087 | 1.022 | 2.55 | +0.056 | | | | |
| 100 nm circles, square array (grid turned 3 deg) | lens | +0.014 | 1.006 | 1.24 | -0.015 | -0.217 | 1.011 | 1.24 | +0.001 |
| 100 nm circles, square array (holes on 40 % of the page) | lens | +0.006 | 1.003 | 1.24 | -0.022 | -0.231 | 1.011 | 1.23 | +0.004 |
| 100 nm circles, square array (brightness wave +-10 %) | lens | +0.008 | 1.005 | 1.37 | -0.014 | -0.190 | 1.011 | 1.36 | +0.008 |
| 100 nm circles, square array (8.5 px pitch, placement 3.6 nm) | lens | +0.060 | 1.016 | 1.29 | +0.036 | -0.128 | 1.045 | 1.25 | +0.174 |
| 100 nm circles, square array (sizes spread 0.5 nm) | lens | -0.004 | 0.981 | 1.26 | +0.015 | | | | |
| 100 nm circles, square array (sizes spread 5.0 nm) | lens | +0.125 | 1.005 | 1.26 | +0.016 | -0.298 | 1.006 | 1.25 | +0.010 |
| 100 nm circles, square array (brightness spread 15 %) | lens | +0.046 | 0.999 | 1.27 | -0.009 | | | | |
| 60 nm circles, dense | lens | +1.641 | 1.116 | 3.95 | +0.703 | +10.757 * | 1.062 | 2.68 | +1.238 |
| 50 nm circles, dense | lens | +9.584 | 0.756 | 7.48 | +0.911 | +16.783 * | 1.108 | 2.91 | +1.935 |
| 90 x 60 nm ellipses, dense | lens | -0.546 | 1.069 | 2.66 | +0.209 | +9.645 * | 0.718 | 2.27 | +2.148 |

Copy of the test file (100 nm circles, pitch 8.5 px, placement 3.6 nm, Gaussian blur 63 nm, sizes spread 2.6 nm), gauss2: one pass +0.022 nm (first page fit +0.165, page fit with the true centres +0.044), response 1.002, scatter 1.06 nm; existing -0.036 nm, response 0.994, scatter 1.04 nm (14 rounds, not settled).

### 5.3 The test file (40 hole pages, gauss2)

- Page CD: one pass 100.799 nm (sd over the pages 0.029), existing 100.693 (sd 0.033); difference +0.106 +- 0.018, largest 0.141 nm; correlation 0.84.
- Per hole against the existing fit: CD difference 0.049 nm rms about the page's offset, response 0.9994, centres 0.009 nm apart.
- Lens form with one pass: page CD -0.056 +- 0.006 nm from gauss2's. The existing fit's lens results on pages 0, 12, 24, 36 (earlier session): 100.488, 100.575, 100.691, 100.664; one pass: 100.661, 100.731, 100.774, 100.723.
- Repeats (pages 0, 5, 24, 30, both forms): page CD equal to 0.0002 nm, hole CDs to 0.0006 nm rms (one hole 0.06 nm).

### 5.4 Other checks

- Centring passes against the settled page CD (pages 0 and 24): one pass 0.017 / 0.022 nm short, two 0.003 / 0.004, three 0.0003 / 0.002; a further round of B and C after two passes adds 0.002 to 0.003 nm.
- Seeds on the 13 generated patterns: all holes on 10; missed 0.3 % (50 nm dense), 1.2 % (300 nm squares), 0.01 % (70 x 45 nm dense); no strays; 2 to 6 ms.
- Pages of the test file without the hole pattern: blank pages 3, 43, 47: about 27,700 seeds, page CD 50 nm, blur 14 nm, chi-square per bin 1.2; page 46 (another pattern): 5,845 seeds, chi-square per bin 227.
<!-- measurements: end -->

## 6. Found on the way

1. **The test TIFF is generated.** `artifacts/cd_test/data/stripe1_board1.tif`: holes on a lattice of exactly
   8.500 px, no rotation, each shifted uniformly within +-0.3 px; `artifacts/cd_acceleration_v3/make_long_fixture.py`
   calls its pages synthetic. No real ACTIS page is on this PC. Earlier notes (todo, the speed proposal, memory)
   called it real.
2. **Normalising a window by its hole's fitted brightness biases the page.** A bigger hole gets a bigger divisor,
   so the average leans towards small holes: -0.09 nm on the copy of the test file. Weighting by brightness, as the
   existing stack does, leans the other way (the brightness-weighted mean of the true sizes is 0.04 nm above their
   plain mean). The plain mean has neither and reaches the noise floor.
3. **gauss2 on a page whose blur is one Gaussian is degenerate**: the second Gaussian's width and weight are free
   (and a very wide one trades against the constant). SciPy's solver took 1,289 evaluations to stop; the existing
   fit's blur parameters differ from page to page for this reason (s2 1.4 to 3.4 px, weight 0.007 to 0.10 on the
   test file). On the test file the page CD does not depend on where they end (+-0.002 nm with the weight held
   anywhere from 0.02 to 0.45); on pages with another blur shape it can (6.13). The prototype holds them with weak
   priors, in chi-square units: `log(s2 / s1) ~ N(log 2, 0.7)` per axis, `logit w2 ~ N(logit 0.05, 1.5)`. They
   need the stack's weights in 1 / variance units.
4. **Centring against the plain stack is wrong.** One Gauss-Newton step of every centre against the stack itself
   (which holds the neighbours' mean light) pulled the centres with their neighbours and read the page +0.38 nm
   too big. Centring needs the page model (step B).
5. **Centring passes.** One sweep in B leaves the page CD 0.017 to 0.022 nm below its settled value (and dependent
   on the seeds at that level); two sweeps 0.003 to 0.004 nm; three 0.0003 to 0.002 nm. A second round of B and C
   after one sweep gains the same as the second sweep. Two sweeps are the default.
6. **Size sweeps.** From a zero offset the response (fitted against true size) is 0.969 after two sweeps, 0.997
   after four, 1.000 after eight; with the 1.3 step factor 0.990 after two and 1.002 after three; 1.6 overshoots
   (0.89). The slow part is the common mode of size, brightness and level between neighbours, not the single hole.
7. **Holes cut in half inside the page** (an artificial test: the right 60 % of a page blanked through the holes)
   read the page -0.25 nm low: their light is modelled as whole holes. With the pattern ending between two columns
   the error is +0.006 nm. An area boundary must not cut holes, or the cut holes must be left out of the model.
8. **Pages without the pattern.** The seed finder returns 27,000 noise maxima on a blank page and the fit reports
   CD 50 nm, blur 14 nm, chi-square per bin 1.2: nothing but the blur looks wrong. On the file's page with another
   pattern the chi-square per bin is 230.
9. **Small holes.** The size and blur valley below about 70 nm is the model's, not the procedure's: 60 nm dense
   +1.7 nm (existing +10.8), 50 nm dense +8.9 (existing +16.8). On the 50 nm page the median standard error of an
   offset is 1.1 outline radii: the library's rule (above 0.3: per-hole size off) has to be carried over.
10. **The existing fit did not settle in 14 rounds on 9 of the 20 generated pages** it was run on, among them both
    ellipse pages and the staggered array; those are the pages where it is furthest from the truth.
11. **Speed details.** Single-precision templates: hole kernels 2.5 times faster, CDs equal to 4 decimals. Matrix
    sums of a window row in single precision: no gain. `cupy` small-call overhead dominated the page fit until the
    solver moved to the host and the pages of a call were fitted together (213 ms per fit to about 8 ms per page
    and fit at 20 pages per call). The GPU is busy 67 to 72 % of the time; one CPU core is busy all the time.
12. **A pitch of a whole number of pixels** (the library's test page: exactly 10 px) puts nearly all holes into two
    to four of the 16 sub-pixel classes. Class (0, 0), on which the grid of starts was first compared, was empty,
    and the walk along the stack's middle row that gave the first size guess found no data: the first fit landed on
    a 300 nm outline (chi-square per bin 20) and stayed there. Fixed in the prototype: the grid is compared on
    each page's most populated class, and the first guess is the stack's second moment over its filled bins.
    Full pages did not change (lens: 0.000 nm on every page; gauss2: 0.013 nm at most on the test file).
13. **On some pages the final fit has more than one end point; its start decides** (`basins.py`,
    `runs/check_basins_a.log`, `_b.log`). On the pipeline's final stack the final stage was started from 81 points
    round the pipeline's end (sizes x 0.92 / 1 / 1.08, exponent 1.7 / 2 / 2.5, blur widths x 0.8 / 1 / 1.25, three
    settings of the blur's second part) and run to its end (400 iterations at most, `ftol 1e-12`). Error of the
    page CD in nm (test file: the page CD); in brackets the starts that end there; * = the pipeline's own end:

    | Page (full size) | lens | gauss2 (with 15) |
    |---|---|---|
    | 100 nm circles, square array | +0.01 * (43), -0.17 (36) | -0.14 * (81) |
    | 100 nm circles, blur 72 nm | -0.11 * (47), +0.09 (30) | -0.16 * (40), +0.09 (39) |
    | 100 nm circles, staggered | +0.09 * (41), +0.06 (28) | +0.06 * (81) |
    | 80 nm circles | -0.23 * (40), -0.80 (31) | -0.82 * (43), -0.80 (7), -0.74 (30) |
    | 80 nm circles, blur 72 nm | -0.08 * (52), -0.70 (19), -0.72 (6) | -0.68 * (38), -0.73 (21), -0.57 (3), -0.52 (14) |
    | 150 nm squares, staggered | +0.12 * (80) | +0.11 * (67; 14 starts fell off) |
    | 160 x 80 nm ellipses | -0.13 * (81) | -0.26 * (81) |
    | 250 nm circles | +0.14 * (81) | +0.06 * (81) |
    | test file, page 0 | 100.66 * (57), 100.70 (23) | 100.71 * (81) |
    | test file, page 22 | 100.75 * (56), 100.80 (24) | 100.82 * (81) |

    - lens: the two ends are a pupil end (cut-off 0.46 to 0.52 cycles/px) and an end with the pupil outside the
      band (cut-off 0.75 up to the bound of 8) and a wider smear. On all seven pages with two ends the second
      has the lower cost (by 16 to 533 chi-square units), although the generated pages were drawn with a pupil;
      see 16 for why.
    - The pipeline starts where the library's fit starts: cut-off 0.5; second Gaussian twice as wide with weight
      0.05. The generator's pupil cut-off is 0.480 (`LENS_FWHM_FACTOR / lens_px`), so the lens form starts next
      to the generator's own end: **the lens errors of 5.2 are flattered by the generated pages.** gauss2 is not
      (the generator's blur is not of its form).
    - It is a property of the model (one outline, one free blur shape): size trades against the blur's shape.
      Whether the library's alternating fit ends at the same end on a given page was not tested.
    - An earlier version of this entry said "gauss2 depends on the start at the 0.1 nm level, every lens result
      stayed within 0.002 nm". That came from two changes of the start only and was wrong as a general statement.
14. **Small images** (`small_pages.py`, `runs/small_pages_v5.log`; pattern of the library's test page). Error of
    the page CD with `measure_cds`' seeds, gauss2 / lens: 135 stacked holes +0.63 / +0.31 nm; 999 holes +0.34 /
    +0.16; 4,941 holes (pitch 200 nm) +0.09 / -0.06; 21 holes +0.95 / +1.17. With the method's own seeds the
    same within 0.005 nm (gauss2) and 0.02 nm (lens). The library's `outline_fit` on the same images: +0.08, -0.16
    (not settled), -0.52, +0.35 nm; its gauss2 second Gaussian ends at 1.98 px with 19 % of the light, 2.26 px
    with 14 %, 1.56 px with 42 %, where the one-pass minimum has 3.3 to 3.5 px with 21 to 25 %.
    - The fit's own standard error of the page CD (inverse of its Gauss-Newton matrix with the priors): 0.5 nm at
      21 holes, 0.23 to 0.25 at 135, 0.08 at 999, 0.04 at 4,941. The errors above are 1 to 4 of those.
    - Holes near the image border against inner holes (`ring.py`; same centres and model, only the stacked set
      changes): lens, inner / ring: +0.78 (55 holes) / -0.04 (80 holes); +0.15 / +0.22; +0.03 / -0.21; -0.05 /
      +0.01 (120 x 170, 300 x 400, 600 x 800, 900 x 1200 px). No trend: light from holes beyond the border is not
      what makes small images read high. Not pinned down beyond that.
    - Centres after B: 0.53 to 0.58 nm rms from the truth (both axes together), from seeds that are 2.5 to 3.1 nm
      off (`measure_cds`) or 0.66 to 0.76 nm off (own).
    - One image per call takes 0.23 s (gauss2) or 0.12 s (lens) for a full page.
15. **gauss2: the exchanged half** (found with 14: the page CD depended on which seeds were handed in, by 0.62,
    0.62 and 0.40 nm on the three larger small images). The two Gaussians can be exchanged without changing the
    blur, but the bounds (`w2 <= 0.6`) and the priors are written for a wide, weak second one. A final fit that
    drifts to a narrower second Gaussian stops with `w2` on the bound: cost 23 to 96 above the minimum, page CD
    0.4 to 0.6 nm away. Whether it drifts depends on the first fit (the seeds' centres blur the first stack).
    Started from the exchanged labels the fit walks to the minimum in about 20 iterations; nothing else holds
    the two apart (restarting with only the blur's parameters from the other end reaches the other end; with only
    the outline's it does not: a scratch test, not kept).
    - The step (`pipeline.in_order`): after C, pages with `log s2x + log s2y < log s1x + log s1y` are fitted again
      (`full`, 30 iterations) from `s1 <-> s2`, `logit w2 -> -logit w2`, clipped to the bounds; per page the end
      with the lower cost is kept.
    - After it the two seed sets agree to 0.005, 0.005, 0.004 nm. With 21 stacked holes the exchanged end has the
      lower cost by 1.5 and is kept (0.01 nm from the other).
    - Full pages: 2 of 58 ended exchanged (`pipeline` before the step): 100 nm circles staggered, +0.043 ->
      +0.061 nm with the cost 101 lower; test file page 13, where the second Gaussian has 5 % of the light: no
      change. With the step the test file's page CDs differ from before by 0.0004 nm at most
      (`runs/file_l2_v5.json` against `_v4`), the generated pages of 5.2 by 0.0003 nm (`truth_*_v5.json`), and the
      time is the same (79 ms per page for the page values).
    - The library's `outline_fit` does not end exchanged on the small images (14).
16. **The pair reach decides which lens end fits better** (`reach.py`, `runs/check_reach_ends.log`,
    `check_reach_time.log`). The final stage from three starts (pupil cut-off 0.5; 0.9; 3) on four generated pages
    whose blur is in the lens family (smear 1.0072 px, or 1.1819 px at blur 72 nm; cut-off 0.4802). Cost of the
    pupil end minus cost of the other end:

    | Page | reach `R + 13` (21 px; 19 for 80 nm) | 30 px | 40 px |
    |---|---|---|---|
    | 100 nm circles, square array | +22 | -39 | -25 |
    | 100 nm circles, staggered | +533 | -36 | -103 |
    | 80 nm circles | +62 | -25 | -36 |
    | 100 nm circles, blur 72 nm | +78 (from 13; these three starts all end at the other end) | only the other end found | +29 |

    - The fitted blur at the pupil end, staggered page: smear 1.026 / 1.030 px, cut-off 0.496 at 21 px; 1.009 /
      1.013, 0.481 at 30 px; 1.007 / 1.011, 0.480 at 40 px.
    - Inside one end the page CD moves by 0.02 nm at most with the reach (100 nm circles +0.014 / +0.005 /
      +0.003). gauss2 does not depend on the reach (0.002 nm on the test file, 0.005 nm on the generated pages).
    - Time for the page values, 20 pages of the test file per call, at 20 / 30 / 40 px: gauss2 80.5 / 92.5 / 105.0
      ms per page; lens 73.4 / 78.6 / 85.5.
    - Reading (not measured further): the lens blur has faint far rings; a neighbour's rings beyond the reach are
      missing from the stack's model, and the fit makes up for them with the blur's shape. The generator draws
      each hole's light on a periodic patch of 64 px, so its pages hold nothing beyond 32 px from a hole: what a
      real page needs is not known.
    - With the blur held at the generator's values and only the outline fitted, the fit is worse than at the
      pupil end (cost 5,864 against 5,339 on the 100 nm circles; CD +0.29 nm; the same on the other pages): the
      free fit takes 2 % more smear (1.027 px for 1.007) and a 2 % higher cut-off than one hole was drawn with.
      An oracle test must leave the smear free.
    - Not adopted: the default stays `R + 13`; the numbers of section 5 were made with it.
17. **Poorly determined pages repeat worse.** The same page fitted in two runs: 0.0003 nm where the page has one
    end; 0.005 to 0.007 nm on the pages with several ends at 72 nm blur; 0.02 to 0.07 nm on the small dense
    patterns (60 nm: +1.64 and +1.71; 50 nm: +9.58 and +9.56; 90 x 60 nm ellipses: -0.55 and -0.53). The hole
    kernels add in an order that differs from run to run.
18. **The final fit's start and its cap of 30 iterations can decide the page CD** (2026-11-13, after the user's
    first run on real images: the final fit stopped at its cap on all 8 images of one area and their page CDs span
    97.13 to 98.89 nm). No real image is on this PC; what follows is from generated images, measured with
    `scripts/outline_fast_check.py` and scratch runs. 8 images of the same 28,000 holes (100 nm circles, pitch
    210 nm), new noise each, the blur changing by +-2 nm from image to image, the report's seeds:

    | Planted | Page CD as built, 3 s over the 8 images | Lowest-leftover end of 10 starts | Ends per image |
    |---|---|---|---|
    | blur 63 to 67 nm (as section 5.2), gauss2 | 100.04 to 100.09, 0.05 | the same | 1 |
    | blur 73 to 77 nm, gauss2 | 99.24 to 100.13, 1.13 (2 images 0.9 nm low, both "NOT converged") | 99.98 to 100.18, 0.20 | 1 to 2, the next end 710 chi-square units worse |
    | blur 68 to 72 nm with a 60 nm lens (a shape gauss2 cannot follow, leftover 3 to 8), gauss2 | 98.59 to 98.88, 0.33 | 98.72 to 98.88, 0.13 | 2 to 3, the next 325 worse |
    | blur 73 to 77 nm, lens form | 99.74 to 99.92, 0.17 (6 of 8 "NOT converged") | the same within 0.02 | 1 |
    | blur 73 to 77 nm at pitch 178.5 nm; 90 nm holes at blur 68 to 72 nm; Gaussian blur 68 to 72 nm | 0.11; 0.27; 0.09 | the same | 1 (to 2, alike) |

    - **The start.** The first fit leaves the blur's second part anywhere: on the 8 images of the second row the
      second Gaussian ends at 1.9 x 2.0 px with 13 %, 2.2 px with 2 %, 1.7 px with 60 % (the bound), 1.5 px with
      3 %, 4.0 x 1.0 px with 2 %, 1.7 px with 50 %, 2.4 px with 2 %, 4.0 x 1.1 px with 3 %. The final fit starts
      there. From the two elongated ones it ends at another outline (CD 99.24 nm, leftover 1.8 to 2.0, exponent
      2.11) than from anywhere else (100.12 to 100.18 nm, leftover 1.37 to 1.44): of 10 starts per image (the
      built one and 9 with the second part put to 1.3 / 2 / 3 times the first width at 5 / 20 / 45 % of the light)
      only the built one reaches the worse end. On the third row no single start reaches the best end on every
      image (the built one on 5 of 8).
    - **The cap.** On generated images the final fit needs 3 to 40 iterations in most cases and up to 99 in some;
      the cap is 30. Where it is hit the CD is usually at its end (0.02 nm at most on the fourth row). Not always:
      on image 4 of the second row the fit stands at 96.57 nm with a leftover of 8.0 after 30 iterations; run on,
      it ends after 84 iterations at 99.46 nm with the two Gaussians exchanged, and from there the exchange step
      (15) reaches 100.02 nm. The report shows 100.02 nm only because that step happened to apply at iteration 30.
    - Where the exchange step is left out, the 8 images of the second row read 96.61 to 100.13 nm after 30
      iterations (3 s 3.6 nm) and 99.21 to 100.13 nm at their ends.
    - The same full-size image fitted five times gives the same reading (0.001 nm; 0.01 nm where the cap is
      hit): the jumps are between images. On the library's test page (135 stacked holes) the fit repeats to
      0.04 nm only with the seed list the reports pass (100.996 or 100.955 nm from identical seeds, 221 of them),
      and with the list that `tests/test_outline_fit_fast.py` and 6.14 used (`measure_cds(...).seeds_xy`: the 187
      reported holes, without those cut by the frame) it ends 0.4 nm away, at 100.58 nm. The small-image numbers
      of 6.14 are therefore not what the reports give; not looked into further.
    - **On the 8 real images the check gives** `check N=8 | rep 97.13-98.88 3s 2.01 nc 5 | end 3s 2.14 it<=56 | best 97.13-98.88 3s 2.14 moved 3 | ends 1-1 gap - | left 430.8-526.0 | rblur -0.21`.
      Neither cause above is what separates them: one end per image from 10 starts, and the ends are as far apart
      as the capped fits (3 s 2.14 against 2.01 nm; three images changed when run on). The model does not fit the
      real average hole: leftover 431 to 526 chi-square per bin, a gap of 21 to 23 bin noises (generated: 1.3 to
      2.0; the third row above 3 to 8). With N / 16 pixels per bin that is roughly 0.25 to 0.5 of one pixel's
      noise for 109,000 to 30,000 holes (the images' hole count and noise are not known here). Where it misses is
      not known. `outline_fit` has the same model and was not measured.
    - Remedy measured, not built: the final fit from several starts of the blur's second part, each run to its end,
      the end with the lowest leftover kept (the table's third column). Estimated cost: one more page fit per
      start (about 12 ms per page and start at 20 pages per call).
19. **The leftover is a ratio to the average hole's noise, not a distance, and it is not comparable between a
    generated and a real image** (2026-11-13; the user read 430 to 526 as grey levels and asked whether the 16 bit
    PNGs or 8 bit TIFFs are the cause). Measured with `scripts/image_values.py` (how the files are stored; the
    fit's noise, the average hole's brightness and the gap between best match and average hole in grey steps;
    table and fields in `docs/outline_fit.md`, "What the leftover is"):
    - Bit depth as such: no effect. One image as 8 bit, as the same values x 257 in 16 bit, with 257 times finer
      steps and as 12 bit values: 100.07 nm four times, leftover 1.30 to 1.36.
    - Rounding to whole grey steps on a quiet image: an effect. True noise 0.1 grey level between the holes: 8 bit
      99.61 nm with a leftover of 86, the same image in 16 bit 100.03 nm with 6.9 (true 99.98). No difference down
      to a true noise of 0.5; 0.06 nm at 0.3; 0.23 nm at 0.2. The fit's own noise figure is 0.39 there (it holds the
      rounding and the differences between holes), 0.69 at a true noise of 0.5.
    - The leftover rises on a quiet image with the fit still right: 16 bit copies 1.25, 2.3, 3.7, 5.0, 6.9 at a
      true noise of 1.0, 0.5, 0.3, 0.2, 0.1 (CD 100.03 to 100.11 nm).
    - In grey levels the gaps are tiny: 0.022 to 0.038 of a hole 55 to 59 bright where the model fits (noise alone
      0.019 to 0.030), 0.034 to 0.054 (0.06 to 0.10 % of the hole) on the third row of 18's table, where the page
      CD is 1.2 nm off. The CD reacts to misses far below what can be seen in a picture of the average hole.
    - The generated pages have a pitch of exactly 10 px: nearly every hole has the same sub-pixel position, 306
      bins of the stack hold about 24,000 pixels each (92 % of all) and the other 1,800 to 2,300 filled bins a
      median of 18 to 66. The leftover (a mean over bins) is then mostly sparse bins at their noise. At 10.3 px (all 4,761 bins
      about 1,600 pixels) the leftover is 1.6 to 1.75 at the usual noise and the CD 100.02 nm: the conclusions
      above hold there, but every earlier leftover in these notes is from whole-pixel pitches.
    - For the real images' 431 to 526 this leaves three readings that the number cannot separate: a real miss of
      the model, a quiet image with many holes, or coarse grey steps (only if the PNGs hold 8 bit content). The
      line of `image_values.py` separates them; not yet run there.

## 7. Not done

- Nothing was run on real pages by me, on the remote, or on more than one GPU. The user's first run on real images
  (2026-11-13) is in 6.18; `scripts/outline_fast_check.py` is the check to run there, `scripts/image_values.py`
  the one that puts its leftover into grey levels (6.19).
- Generated pages at pitches that are not a whole number of pixels as the standard test (6.19): the measurements of
  section 5 and 6.1 to 6.18 are at whole-pixel pitches, where the average hole is sampled at one sub-pixel position.
- The final fit from several starts, each run to its end, the lowest leftover kept (6.18): measured by the check,
  not in the library.
- Area per page (the user's tiles): seeds on the whole page, stack and report inside the area, one window from its
  border. Tested only as a pattern that ends inside the page.
- `invert` (dark holes), the per-hole-size switch-off rule, x / y widths per hole (`ellipse_widths`), outline rays
  for the workbench, matching to the holes that `measure_cds` reports.
- A CPU path. The prototype's page fit runs under NumPy (`ShapeFitter(xp=np)`, about 1.7 s per fit); the hole
  passes are CUDA kernels only.
- Two processes per GPU, to use the idle third of the GPU.
- Choosing between the fit's ends (6.13): a second start with the pupil outside the band (lens) and keeping the
  lower cost. Not built; with the default reach it would pick the end the generator did not use (6.16).
- The existing `outline_fit` from several starts (does it have the same ends, and which does it reach?).
- What makes small images read high (6.14).

## 8. Build order and checks

`docs/outline_fit_fast_handoff.md`. In short, as the user decided on 2026-11-12: phase 1 is the library method
`outline_fit_fast` and both folder reports, GPU only, page values only, so that it can be compared with
`outline_fit` on real images; the TIFF batch (areas from the hole check, one process per GPU) follows the user's
verdict on that comparison; a size per hole only if the method proves itself.

The user's answers: page values for now; the minute of reading is measured on the remote, which has 64 CPU threads;
the remote selects pattern areas by phase correlation of page tiles against the tiles of a reference page (the
handoff's section 8 has the details and two options for the batch). Still open: whether a kept tile always holds
only the reference's pattern, and whether the real test images are PNGs in a folder.
