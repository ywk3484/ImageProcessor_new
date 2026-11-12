# Outline fit in one pass: builder's notes

**Status, 2026-11-12.** Proposal with a measured scratch prototype. Nothing in `subpx/` has been changed and the user
has not decided. The page for the user is `docs/outline_fit_one_pass.html` (what it is, the numbers, the limits, the
four open questions). This file holds what a session needs to build it: the exact method, the settings, what was
tried and dropped, the build order and the checks. It replaces `docs/outline_fit_speed.md` as the plan for the batch;
that file's measurements of the existing fit still stand.

The prototype is in `artifacts/outline_fit/fast/` (git-ignored, on the development PC only); its `README.txt` lists
every script and command.

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
- Stages as in the library. First fit: grid of starts (15 sizes x 12 blur widths, compared on the bins of one
  class), `round` (at most 10 iterations), `super` (12), `ftol 1e-6`. Final fit: `full` from the first fit's
  result, at most 30 iterations, `ftol 1e-7`.

### 3.3 The procedure

    A  seeds -> plain stack -> Gamma -> first page fit                                  theta_0
    B  hole pass, size held: centre, brightness, level of every hole, two sweeps        (centres)
    C  plain stack with B's centres -> Gamma -> final page fit from theta_0             theta_1 = the page
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

By phase, ms per page: A stack 11.7, A page fit 11.2, B hole pass + stack 42.0, C page fit 11.8, D hole pass 54.2.
Inside them: upload 0.5, seeds 3.9, spacing 1.0, stacks 7.4, pair maps 6.2, both page fits 15.4 at 20 pages per call, templates 12.3, hole-pass set-up 11.9, five sweeps 60.3, noise line 1.3, results 2.4.
One sweep (39,042 holes, window 15 x 15 px, render 23 x 23 px): `hole_normal` 6.3 ms, `render_scatter` 4.5 ms, `hole_update` 0.5 ms.
Pages per call 4 / 10 / 20 / 40: 111 / 89 / 80 / 87 ms (page values), 164 / 144 / 136 / 134 ms (with sizes).
GPU busy 67 % (page values) and 72 % (with sizes); one CPU core busy all the time; GPU memory pool 2.8 to 3.0 GB at 20 pages per call, 5.5 GB at 40.
Remote estimate on the page (19 to 30 s and 30 to 51 s per image): lower end = GPU share of the time x 1 / 2.3 (the remote's ratio for the other CD methods, 3.5 s against 7.9 s per page), CPU share unchanged, four GPUs; upper end = this PC's time / 4.

### 5.2 Generated pages with known sizes

`subpx.synthetic_arrays.render_hole_array`, 2048 x 1378 px, 21 nm pixels, blur 65 nm with the lens term 45 nm, sizes spread 2.5 nm, placement 1 nm, brightness 5 %, photon noise 30 per level, unless said. Error = page CD - mean true CD of the stacked holes (squares: against the area-equivalent diameter). Response = slope of the fitted against the true hole CD; scatter in nm of true CD; light = move for a 10 % brighter hole. Existing = the existing fit on the same page (`*` = not settled in 14 rounds).

| Page | Blur form | One pass: error | response | scatter | light | Existing: error | response | scatter | light |
|---|---|---|---|---|---|---|---|---|---|
| 100 nm circles, square array | gauss2 | -0.135 | 1.036 | 1.22 | +0.141 | -0.476 | 1.055 | 1.21 | +0.192 |
| 100 nm circles, square array | lens | +0.014 | 1.002 | 1.24 | -0.018 | -0.198 | 1.008 | 1.24 | +0.001 |
| 80 nm circles, square array | gauss2 | -0.737 | 1.124 | 1.86 | +0.357 | -1.028 * | 1.083 | 1.92 | +0.198 |
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
| 100 nm circles, square array (blur 72 nm) | gauss2 | -0.294 | 1.062 | 1.45 | +0.238 | | | | |
| 80 nm circles, square array (blur 58 nm) | lens | -0.064 | 1.002 | 1.55 | +0.023 | | | | |
| 80 nm circles, square array (blur 72 nm) | lens | -0.085 | 1.026 | 2.54 | +0.067 | | | | |
| 100 nm circles, square array (grid turned 3 deg) | lens | +0.014 | 1.006 | 1.24 | -0.015 | -0.217 | 1.011 | 1.24 | +0.001 |
| 100 nm circles, square array (holes on 40 % of the page) | lens | +0.006 | 1.003 | 1.24 | -0.022 | -0.231 | 1.011 | 1.23 | +0.004 |
| 100 nm circles, square array (brightness wave +-10 %) | lens | +0.008 | 1.005 | 1.37 | -0.014 | -0.190 | 1.011 | 1.36 | +0.008 |
| 100 nm circles, square array (8.5 px pitch, placement 3.6 nm) | lens | +0.060 | 1.016 | 1.29 | +0.036 | -0.128 | 1.045 | 1.25 | +0.174 |
| 100 nm circles, square array (sizes spread 0.5 nm) | lens | -0.004 | 0.981 | 1.26 | +0.015 | | | | |
| 100 nm circles, square array (sizes spread 5.0 nm) | lens | +0.125 | 1.005 | 1.26 | +0.016 | -0.298 | 1.006 | 1.25 | +0.010 |
| 100 nm circles, square array (brightness spread 15 %) | lens | +0.046 | 0.999 | 1.27 | -0.009 | | | | |
| 60 nm circles, dense | lens | +1.696 | 1.084 | 4.13 | +0.687 | +10.757 * | 1.062 | 2.68 | +1.238 |
| 50 nm circles, dense | lens | +8.936 | 0.664 | 9.09 | +0.764 | +16.783 * | 1.108 | 2.91 | +1.935 |
| 90 x 60 nm ellipses, dense | lens | -0.531 | 1.070 | 2.66 | +0.212 | +9.645 * | 0.718 | 2.27 | +2.148 |

Copy of the test file (100 nm circles, pitch 8.5 px, placement 3.6 nm, Gaussian blur 63 nm, sizes spread 2.6 nm), gauss2: one pass +0.022 nm (first page fit +0.165, page fit with the true centres +0.044), response 1.002, scatter 1.06 nm; existing -0.036 nm, response 0.994, scatter 1.04 nm (14 rounds, not settled).

### 5.3 The test file (40 hole pages, gauss2)

- Page CD: one pass 100.798 nm (sd over the pages 0.029), existing 100.693 (sd 0.033); difference +0.105 +- 0.018, largest 0.141 nm; correlation 0.84.
- Per hole against the existing fit: CD difference 0.050 nm rms about the page's offset, response 0.9991, centres 0.009 nm apart.
- Lens form with one pass: page CD -0.055 +- 0.005 nm from gauss2's. The existing fit's lens results on pages 0, 12, 24, 36 (earlier session): 100.488, 100.575, 100.691, 100.664; one pass: 100.661, 100.731, 100.774, 100.723.
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
   test file). The page CD does not depend on where they end (+-0.002 nm with the weight held anywhere from 0.02 to
   0.45). The prototype holds them with weak priors, in chi-square units: `log(s2 / s1) ~ N(log 2, 0.7)` per axis,
   `logit w2 ~ N(logit 0.05, 1.5)`. They need the stack's weights in 1 / variance units.
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

## 7. Not done

- Nothing was run on real pages, on the remote, or on more than one GPU.
- Area per page (the user's tiles): seeds on the whole page, stack and report inside the area, one window from its
  border. Tested only as a pattern that ends inside the page.
- `invert` (dark holes), the per-hole-size switch-off rule, x / y widths per hole (`ellipse_widths`), outline rays
  for the workbench, matching to the holes that `measure_cds` reports.
- A CPU path. The prototype's page fit runs under NumPy (`ShapeFitter(xp=np)`, about 1.7 s per fit); the hole
  passes are CUDA kernels only.
- Two processes per GPU, to use the idle third of the GPU.

## 8. Build order and checks

1. `subpx/_outline/fast.py` (page fit: spectrum, stack, pair maps, solver) and the hole kernels beside
   `subpx/_outline/page.py`; public `subpx.fit_outline_fast` and a `measure_cds` method name `outline_fit_fast`
   beside `outline_fit` (nothing replaced). GPU only at first; a clear error without CuPy.
2. Tests (`tests/test_outline_fit_fast.py`): the analytic template against `Template.render` (1.2e-4 of the peak:
   the library's own discretisation); the Jacobian against central differences; a small generated page on the GPU
   with its truth (page CD within 0.1 nm, response 0.98 to 1.03); the page fit under NumPy against CuPy on the same
   stack (0.001 nm); repeat runs equal to 0.001 nm.
3. Regression numbers: section 5's truth table (the script is `artifacts/outline_fit/fast/truth_matrix.py`).
4. `scripts/cd_batch.py`: method `outline_fit_fast`, one process per GPU fitting 20 pages per call, the area per
   page from the hole check, the per-page columns; `timings.csv` gets the phase times.
5. Remote check, one command, about 150 characters back: seconds per page for both methods on the first hole pages
   of one image, the two page CDs and their difference, chi-square per bin.
6. Texts that state times or limits: `docs/outline_fit.md`, the docstrings of `fit_outline` and `measure_cds`.

Questions to the user that decide parts of this: the four in the page's section 9 (size per hole or page values
only; what the hole check hands over and whether a kept tile always holds the reference's pattern; where the minute
of reading is measured and the remote's CPU threads; build it or not).
