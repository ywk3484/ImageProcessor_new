# Outline fit: speed-up proposal (not built)

**Status, 2026-11-12.** Nothing in `subpx/` has been changed. This is a proposal with measurements; the user
has not decided. Everything was measured with scratch prototypes patched into the unchanged library at run
time. The prototypes live in `artifacts/outline_fit/speed/` (git-ignored, on the development PC only; its
`README.txt` lists every script and number). The code that matters is copied into the appendices here, so this
file is enough to continue.

Read sections 7 to 9 before proposing anything new: they say what the evidence does not cover, what must not
be assumed, and which questions to the user are still open.

## 1. The task in the user's units

- One TIFF is one **image** of up to 1,500 **pages** (2048 x 1378 px, 8-bit). **Every page holds a different
  pattern.** A **stripe** is at least 4 images; a dataset is about 900 stripes (fewer for now).
- The remote has 4 x RTX 6000 Ada 48 GB (Windows Server 2022). Its CPU core count is not known.
- Development PC: RTX 4060 Ti 16 GB, Ryzen 5 7500F (6 cores). All measurements below are from this PC.
- Today the outline fit takes **104 to 150 s per page** on that GPU: 2 to 2.5 days per 1,500-page image.
  The user's words: "extremely slow"; 10 s per page "may not be enough".

Test data: `artifacts/cd_test/data/stripe1_board1.tif`, the only real file available. 48 pages, 40 with
holes (39,042 or 37,842 holes per page), **all of them round holes of about 100 nm**. See section 7.

## 2. Where the time goes today

Unchanged library, page 0 (39,042 holes), blur family `gauss2`, GPU, float64: 103 to 106 s; 10 rounds, 43
hole sweeps, 11 calls of `fit_holes`. With a synchronising timer round every stage (107.0 s):

| Part | Seconds | What it is |
|---|---|---|
| Hole stage, `PageFit.fit_holes` | 68.2 | `cubic_sample` is 58.3 s of it (19.6 s in the fit windows, 38.6 s re-rendering each chunk's light in `_light`); `einsum` 3.1; far field 1.0; solve and inverse 0.3 |
| Shape stage, `PageFit.fit_shape` | 27.4 | 4,031 template renders on one CPU core at 6.6 ms each; 5.1 ms of each is `Template.indicator` |
| Stack, start, size check | 7.3 | `cleaned_stack` 3.7, `initialise` 2.3, `delta_information` 1.3: `cubic_sample` again (5.2 s) |
| Template update, `set_theta` | 2.8 | 2.4 s is three NumPy `rfft2` of 1514 x 2184 per round in `_set_far` |
| Noise line, `fit_noise` | 1.1 | percentiles and medians on the host |
| Rest | 0.3 | |

Two causes carry 90 % of the time:

1. **`cubic_sample` (58 %).** It reads the templates at every window pixel with one gather, one multiply and
   one add per tap, per template: about 17,000 new arrays and 82 GB of allocations **per sweep** (a call's
   set-up: 6,000 arrays, 47 GB). The page is 22 MB as float64. The GPU is 94 to 96 % busy (median) moving
   memory; the arithmetic of a sweep takes 0.06 s when done in one kernel.
2. **`Template.indicator` (20 %).** Every render recomputes the outline's coverage on all 768 x 768
   sub-samples (589,824). Only the fine pixels the edge crosses can differ from exactly 0 or 1: 152 to 160 of
   36,864 (0.4 %) on this page.

Other cases, unchanged library: page 24: 132.6 s (13 rounds, converged) and 153.8 s (14 rounds, not
converged) in two runs; page 0 with the `lens` family: 255.0 s; four pages stacked to 5408 x 2048 (117,126
holes): 330.0 s; a 120 x 170 px region of 285 holes on the CPU backend: 29.6 s. The report's other methods on
page 0 (`measure_cds`, GPU): 10.6 s for the fast report's 14 methods, 7.9 s for the report's 10 (2.2 s of it
the deblend).

## 3. Proposed changes that keep the numbers

Same model, same procedure, same stop rule. Each step was prototyped and run end to end.

### Step 1. Hole stage as fused GPU kernels

- **Where:** `subpx/_outline/page.py`: `cubic_sample`, `fit_holes`, `_light`. GPU (CuPy) path only; the NumPy
  path stays as it is and remains the reference.
- **What:**
  - 1a. `cubic_sample` as one `ElementwiseKernel` when the arrays are CuPy arrays (Appendix A.1). Every
    caller benefits (hole stage, stack, start, size check). This alone: page 0 in 46.9 s.
  - 1b. Two `RawKernel`s for `fit_holes` (Appendix A.2): `hole_normal` (one thread per hole and window row:
    samples the five templates in the library's tap order and sums the 5 x 5 normal matrix, the right-hand
    side, the leftover sum and the pixel count) and `render_scatter` (one thread per hole and render-window
    pixel: takes the hole's old light out of the page model and puts its new light in). The sweep logic round
    them (chunks of 2000 holes per colour, clips, robust weights, far field, stop rule) is unchanged.
- **Why the numbers stay:** same arithmetic in float64, same tap order, same chunk order. From one pushed-off
  state, one and three sweeps by the library and by the kernels agree to 2e-13 px in the centres, 3e-14 px in
  the edge offsets and 1e-11 grey levels in the page model.
- **Measured:** one sweep 1.294 s -> 0.062 s (21 x); a call's set-up 0.72 s -> 0.042 s; page 0 in 41.8 s with
  1a and 1b; GPU memory peak 1.75 GB (2.0 GB today).
- **Watch out:**
  - The prototype covers the default options only (`background="level"`, `gradient=False`,
    `free_level=True`, `frame_size=True`, default `max_delta`, `robust_c=3`). Other options must keep the
    existing code path.
  - While `rms_typ` is `None` (first sweep after a template change) the robust scale is the chunk's own median
    leftover: the prototype runs `hole_normal` once without robust weights to get it, then again.
  - Keep the chunks. One launch per colour is faster (0.048 s per sweep) but is another sweep order: 0.026 nm
    away after three sweeps.
  - float32 template arithmetic gains nothing once fused at one page per launch (0.053 s) and moves a CD by
    2e-5 nm. Keep float64.
  - The kernels use `atomicAdd` on doubles: sums run in a varying order, as `bincount` does today.

### Step 2. Template render with the same values, bit for bit

- **Where:** `subpx/_outline/model.py`: `Template.__init__`, `indicator`, `render`.
- **What** (Appendix B):
  - Coverage only where the edge can be. Fine pixels are kept sorted by the radius of their centre; pixels
    whose centre is closer to the middle than `min(edge radius) - margin` are exactly 1, those beyond
    `max(edge radius) + margin` exactly 0, and only the ring between is sub-sampled. `margin` is the largest
    distance of a sub-sample from its fine pixel's centre plus half the ramp. The ring's sub-samples are summed
    in the library's order (sub-row, then sub-column).
  - The outline's spectrum (`fft2(ifftshift(indicator)) * edge_fix`) is remembered for the last 16 outlines
    and the blur's transfer function for the last 16 blurs. A finite-difference step changes one of the two.
    The box sincs are remembered per box width. The order of the multiplications is unchanged.
- **Why the numbers stay:** every rendered value is computed by the same operations in the same order.
  Largest difference from the library's render: 0.0 on 60 random outlines and blurs for each of three
  templates (`gauss2` 24 px, `lens` 48 px, `gauss2` 48 px). The ten shape-stage calls of page 0, replayed:
  identical fitted parameters (difference 0.0).
- **Measured:** render 6.6-7.0 ms -> 1.64 ms (1.12 ms when only the blur changed); 48 px box 31-34 ms ->
  8.5-9.2 ms. Shape stage of page 0: 27.8 s -> 5.4 s. Page 0 with steps 1 and 2: 15.2 s. On the CPU backend
  the 285-hole region goes from 29.6 s to 9.3 s with the same rounds, sweeps and result, bit for bit.
- **Watch out:** the memory holds 16 complex N x N arrays per template: 9 MB at 24 px, 38 MB at 48 px,
  151 MB at 96 px. Fewer entries for the large boxes would do.

### Step 3. Two leftovers on the device

- **Where:** `page.py`: `_set_far`, `fit_noise` (Appendix C).
- **What:** the far kernels' FFT with `xp.fft.rfft2` on the device instead of `np.fft.rfft2` on the host
  (67 ms per transform on the CPU, 6 ms with the upload on the GPU; three per round); the noise line's
  percentiles and medians with `xp` instead of after a download.
- **Measured:** page 0 with steps 1 to 3: 10.3 s. That run also grew the far field's padded grid to an
  FFT-friendly size (1514 = 2 x 757 -> 1536), worth about 0.5 s.
- **Watch out:** do not change the padded grid on the CPU path. It is only a rounding-level change, but on the
  285-hole CPU region it moved the outline CD by 0.26 nm (section 6). On NumPy the other two changes are
  the same functions and leave CPU results bit-identical.

### Step 4 (optional). Half-spectrum FFT in the render

`rfft2` / `irfft2` in `Template.render` with the parameter-free factors multiplied once
(`artifacts/outline_fit/speed/rfft_render.py`). Render 1.15 ms (0.30 ms blur only) at 24 px, 3.8 / 2.0 ms at
48 px, 15 / 8 ms at 96 px (library 130 ms). Values equal to 7e-16 of the peak, not bit for bit. Page 0:
10.3 -> 7.9 s; lens: 36.3 -> 16.9 s. On page 0 (`gauss2` and `lens`) the outline CD stayed within about
0.001 nm. On page 24
(`gauss2`) the fit landed on blur 63.3 / 62.3 nm instead of 62.6 / 63.2: outline CD -0.015 nm, per-hole CD
0.086 nm rms. Use it for the lens family only, or not at all.

### Results, end to end

Seconds per page; CD differences are from an unchanged run of the same page.

| Page 0, `gauss2` | s | Outline CD | Per-hole CD, rms |
|---|---|---|---|
| Unchanged, two runs | 104.3 / 102.8 | 0.0003 nm apart | 0.009 nm (4 holes over 0.05 nm) |
| Step 1a | 46.9 | -0.001 nm | 0.010 nm |
| Step 1 | 41.8 | 0.0000 nm | 0.0000 nm |
| Steps 1, 2 | 15.2 | -0.001 nm | 0.007 nm |
| Steps 1a, 2, 3 (no hole kernels) | 14.7 | +0.007 nm | not compared |
| **Steps 1, 2, 3** | **10.3** | +0.0015 nm | 0.004 nm |
| Steps 1 to 4 | 7.9 | -0.001 nm | 0.010 nm |

| Other cases | Unchanged | Steps 1 to 3 | CD difference |
|---|---|---|---|
| Page 24, `gauss2` | 132.6 s (153.8 s not converged) | 14.2 / 13.9 s | outline -0.005 / +0.005 nm, per hole 0.006-0.008 nm rms |
| Page 0, `lens` | 255.0 s | 36.3 s (16.9 s with step 4) | 0.0000 nm (0.0001 nm) |
| 5408 x 2048, 117,126 holes | 330.0 s | 26.7 s | outline +0.0001 nm, per hole 0.005 nm rms |
| 285-hole region, CPU backend | 29.6 s | 9.3 s (step 2 only) | bit-identical |

What is left in the page's 8 to 10 s: hole stage 3.1 s (43 sweeps x 23 chunks of Python and small CuPy calls),
shape stage 5.4 s (3.4 s with step 4; about 4,000 renders driven by SciPy's least squares), the rest 1.5 s.

## 4. Batch: whole images on several GPUs

Every page is fitted on its own, from scratch. Nothing is shared between pages (section 8).

**Does not work: several worker processes per GPU**, which is how `scripts/cd_batch.py` runs today
(`--workers-per-gpu`). With steps 1 to 3 on one GPU: 1 / 2 / 3 / 4 / 6 workers -> 10.8 / 7.3 / 6.1 / 5.4 /
5.1 s per page; a page takes 10.8 -> 28.8 s inside its worker. Windows gives the processes turns on the GPU.

### Step 5. One job per GPU that fits several pages at once

- One process per GPU (one CUDA context). Each page is fitted on its own thread with its own CuPy stream
  (`cp.cuda.Stream(non_blocking=True)`) by the unchanged single-page fit; the shape fits (CPU, SciPy) of all
  pages go to a pool of CPU processes (`thread_batch.py` in the prototype folder).
- Measured on this PC, from scratch, 5 CPU processes for the shape fits: 11.6 / 6.3 / 5.8 / 5.4 s per page
  with 1 / 3 / 6 / 10 pages at once; 10.5 GB of GPU memory pool for 10 pages. The limit is the Python side of
  the hole stage (one thread at a time).
- Needs, in the library and the batch tool: `outline_fit` as a `cd_batch` method; seeds from the threshold
  pass instead of the deblend that `measure_cds` runs for it today (2.2 s per page); a per-page row in
  `pages.csv`.

### Step 6. If more is needed: the pages inside the GPU routines

One launch handles a chunk of every page of the batch (a page index per hole; templates, noise line and
robust scale per page), so the per-chunk Python work is paid once per batch, not once per page. Not
prototyped end to end. Measured parts on this GPU, K pages as one problem with one launch per colour:

| | ms per page and sweep |
|---|---|
| One page alone, chunks of 2000 (step 1) | 60 (kernels 24.5, solve and updates 26-30, far field 11) |
| K = 8, float64 | 36 (kernels 23, updates 2, far field 11) |
| K = 8, float32 template arithmetic | 22 (kernels 10) |

Set-up of a `fit_holes` call: 23 ms per page. Template renders as a GPU batch (FFT pair and multiply): 0.03 ms
each at 192 x 192, 0.13 ms at 384 x 384 (1.6 ms on the CPU). From these, a page from scratch costs about
2.3 s of GPU work here and 5.1 core-seconds of shape fits.

### Estimates for the remote

Not measured there. Assumed: its GPUs are 3 x the RTX 4060 Ti for this work and its CPU cores as fast as here.

| | Per page | 1,500-page image | Stripe of 4 images |
|---|---|---|---|
| Today's code, this PC, one GPU (measured) | 104-150 s | 2-2.5 days | 7-10 days |
| Step 5, one job per GPU, 4 GPUs | 1-1.5 s | 25-40 min | 1.7-2.7 h |
| Step 6, 4 GPUs | 0.2-0.3 s | 5-8 min | 20-30 min |

Step 6 needs about 25 free CPU cores for the shape fits, or the shape fit on the GPU as well (new numerics:
SciPy's least squares would have to be replaced by a batched one). At 0.2 to 0.3 s per page, 900 stripes take
12 to 19 days.

## 5. Tried and rejected, or withdrawn

| Idea | Result | Verdict |
|---|---|---|
| **Reference start**: start a page from another page's fitted outline and blur | 3.7 s per page instead of 11.1 on the test file | **Withdrawn. It assumed that the pages of an image share a pattern. The user: "I never said that my 1500 pages took the same area. They all have different patterns for each page." Do not propose it again.** The numbers hold only for pages of one pattern |
| Outline fit continued from the current template without its grid of starts | renders 3,959 -> 2,296, page 14.8 s against 15.2 s | Rejected: no time gained, outline CD -0.051 nm, 8,314 holes over 0.05 nm |
| Threads for the renders of one fit (`least_squares(workers=...)`) | shape stage 5.4 -> 4.0 s at 4 threads, worse beyond | Rejected |
| float32 in the hole kernels, one page per launch | 0.062 -> 0.053 s per sweep | Rejected (only pays inside step 6) |
| One chunk per colour | 0.048 s per sweep | Rejected: another sweep order |
| Several worker processes per GPU | 2 x at 6 workers | Rejected |
| CPU backend's hole stage | a full page stays tens of minutes | Not in the proposal: needs a compiled dependency (numba is not installed) |

## 6. Found about the fit itself

Not speed matters and not caused by the proposal. Causes not isolated unless said.

- **The fit keeps drifting after the stop rule has ended it.** Left running for 16 rounds from another
  page's fit: `lens`, page 24: +0.004 nm per round through round 16 with a flat chi2 (100.542 -> 100.639; its
  from-scratch value is 100.691). `gauss2`, page 0: 100.64 to 100.67 nm from round 2 to 16; from scratch it
  stopped at 100.591 with proposals still at +0.03 to +0.06 nm. The stop rule is in `PageFit.run`
  (`tol_px = 0.08 / 21` and a settled chi2).
- **Runs of the unchanged code differ on the GPU with `gauss2`.** Page 0: outline CD 0.0003 nm apart, but
  page 24: 100.72 nm (converged in 13 rounds) and 100.83 nm (round limit, flagged NOT converged). Eight
  pages fitted twice with steps 1 to 3: page CD sd 0.016 nm, largest 0.046 nm.
- **`lens` repeats exactly** from scratch (0.000 nm on four pages, and unchanged against steps 1 to 3). It is
  the strict regression check on the GPU.
- **A small region is very sensitive.** On the 285-hole CPU region a change in the 13th decimal (the padded
  FFT size) gave 10 rounds instead of 8 and outline CD 100.66 instead of 100.40 nm, both runs "converged".
- `docs/outline_fit.md` says GPU runs repeat to about 0.01 nm. That holds on page 0, not on page 24.

## 7. Limits of the evidence

- One real file. Every hole page carries about 39,000 round holes of about 100 nm. Other hole sizes, shapes
  and counts will take different times (the template box doubles for large holes, the lens family uses a
  48 px box), and no CD comparison here covers them.
- One PC. Nothing was measured on the remote; its rows are estimates.
- Seeds came from `detect_centers(refine="logquad")`, not from the deblend seed list `measure_cds` passes.
- The prototypes drive `PageFit` directly, as `subpx.outline._fit` does. They were not run through
  `measure_cds`, the reports or the workbench, and the test suite was not run with them.
- Step 6 and the remote's numbers are assembled from measured parts, not run end to end.

## 8. Do not assume

- That the pages of an image share anything: pattern, hole size, shape, pitch, an outline fitted elsewhere.
- That the tool's blur is the same on all pages of an image. It may be; it has to be checked on pages that
  really differ, with the user's knowledge of the tool.
- That a test on `stripe1_board1.tif` says anything about pages with other patterns.
- That the remote behaves like this PC.

## 9. Open questions to the user (asked, not answered)

1. Build steps 1 to 3?
2. What differs from page to page: hole size, shape, pitch? Does one page always hold a single pattern?
3. A TIFF, or a handful of pages, with that real variety to time and test on.
4. The remote's CPU thread count: `$env:NUMBER_OF_PROCESSORS` (one number; everything on the remote is typed
   by hand).
5. Step 4 for the lens family: yes or no.

## 10. Build order and checks

1. Step 1a (`cubic_sample` kernel). Check: `pytest tests/test_outline_fit.py` (it holds
   `test_gpu_fit_matches_the_cpu_fit`: outline CD 1e-4 px, CD 1e-3 px, centres 2e-4 px); page 0 about 47 s.
2. Step 1b (hole kernels). Add a test: one and three sweeps from a pushed-off state, kernels against the
   existing path, equal to 1e-9. Page 0 about 42 s.
3. Step 2 (render). Add a test: new render against the old one on random outlines, blurs, offsets and box
   widths, difference exactly 0. All CPU tests must pass unchanged. Page 0 about 15 s.
4. Step 3. Page 0 about 10 to 11 s.
5. Numbers. GPU: `lens`, page 0, against the unchanged code: within 0.001 nm in outline CD and per hole.
   `gauss2`, page 0: outline CD within 0.005 nm, per-hole CD within 0.012 nm rms. CPU: the 285-hole region
   (`crop=600,720,900,1070`) bit-identical. Two synthetic stress pages rerun for slope and mean error.
6. Texts that state times: `docs/outline_fit.md` ("Time"), `subpx/outline.py` (`fit_outline` docstring),
   `subpx/cd.py` (`measure_cds` docstring), the hint printed by `scripts/cd_fast_report.py` and
   `scripts/cd_report.py`.
7. Steps 5 and 6 only after the answers to section 9.

The measuring scripts (Appendix D) give the unchanged numbers and the differences in nm.

## Appendix A. GPU kernels (validated prototype)

### A.1 `cubic_sample` as one kernel

From `artifacts/outline_fit/speed/tier1.py`. `grids` are the fine grids (N, N); they are stacked to (N, N, ng)
so that a tap's values lie together.

```python
PRE = r'''
__device__ inline void keys_w(double t, double* w) {
    double x0 = 1 + t, x1 = t, x2 = 1 - t, x3 = 2 - t;
    w[0] = ((-0.5 * x0 + 2.5) * x0 - 4) * x0 + 2;
    w[1] = (1.5 * x1 - 2.5) * x1 * x1 + 1;
    w[2] = (1.5 * x2 - 2.5) * x2 * x2 + 1;
    w[3] = ((-0.5 * x3 + 2.5) * x3 - 4) * x3 + 2;
}'''

kernels = {}

def cubic_sample(grids, sx, sy, xp_):
    ng = len(grids)
    if ng not in kernels:
        body = f'''
        double fx = floor(sx), fy = floor(sy);
        int ix = (int) fx, iy = (int) fy;
        double wx[4], wy[4];
        keys_w(sx - fx, wx);
        keys_w(sy - fy, wy);
        double acc[{ng}];
        for (int g = 0; g < {ng}; ++g) acc[g] = 0;
        for (int k = 0; k < 4; ++k) {{
            int yy = iy + k - 1;
            if (yy < 0 || yy >= n) continue;
            double row[{ng}];
            for (int g = 0; g < {ng}; ++g) row[g] = 0;
            for (int l = 0; l < 4; ++l) {{
                int xx = ix + l - 1;
                if (xx < 0 || xx >= n) continue;
                long long base = ((long long) yy * n + xx) * {ng};
                for (int g = 0; g < {ng}; ++g) row[g] += wx[l] * tmpl[base + g];
            }}
            for (int g = 0; g < {ng}; ++g) acc[g] += wy[k] * row[g];
        }}
        ''' + " ".join(f"v{g} = acc[{g}];" for g in range(ng))
        kernels[ng] = cp.ElementwiseKernel("float64 sx, float64 sy, raw float64 tmpl, int32 n",
                                           ", ".join(f"float64 v{g}" for g in range(ng)), body,
                                           f"cubic_sample_{ng}", preamble=PRE)
    tmpl = cp.ascontiguousarray(cp.stack(grids, axis=-1))
    out = kernels[ng](sx, sy, tmpl, np.int32(grids[0].shape[0]))
    return list(out) if isinstance(out, tuple) else [out]

P.cubic_sample = cubic_sample
```

### A.2 The hole stage's two kernels and the sweep round them

From `artifacts/outline_fit/speed/fused_proto.py`. `SRC` is formatted with `SRC % dict(TT="double")`, so `%%`
in it is a literal `%`. `tt="float"` is the float32 variant (not proposed). The `timing` branches only
measure. `Fused.fit_holes` is `PageFit.fit_holes` with the two heavy steps replaced; `f` is the `PageFit`.

```python
SRC = r'''
typedef %(TT)s TT;
#define NGMAX 5

__device__ inline void keys(TT t, TT* w) {
    TT x0 = 1 + t, x1 = t, x2 = 1 - t, x3 = 2 - t;
    w[0] = (((TT)(-0.5) * x0 + (TT)2.5) * x0 - 4) * x0 + 2;
    w[1] = ((TT)1.5 * x1 - (TT)2.5) * x1 * x1 + 1;
    w[2] = ((TT)1.5 * x2 - (TT)2.5) * x2 * x2 + 1;
    w[3] = (((TT)(-0.5) * x3 + (TT)2.5) * x3 - 4) * x3 + 2;
}

// values of ng interleaved fine grids (N, N, ng) at fine coordinates (sx, sy); outside the grid reads 0
__device__ inline void sample(const TT* tmpl, int N, int ng, double sx, double sy, TT* v) {
    double fx = floor(sx), fy = floor(sy);
    int ix = (int) fx, iy = (int) fy;
    TT wx[4], wy[4];
    keys((TT)(sx - fx), wx);
    keys((TT)(sy - fy), wy);
    for (int g = 0; g < ng; ++g) v[g] = 0;
    for (int k = 0; k < 4; ++k) {
        int yy = iy + k - 1;
        if (yy < 0 || yy >= N) continue;
        TT row[NGMAX];
        for (int g = 0; g < ng; ++g) row[g] = 0;
        for (int l = 0; l < 4; ++l) {
            int xx = ix + l - 1;
            if (xx < 0 || xx >= N) continue;
            const TT* p = tmpl + ((long long) yy * N + xx) * ng;
            for (int g = 0; g < ng; ++g) row[g] += wx[l] * p[g];
        }
        for (int g = 0; g < ng; ++g) v[g] += wy[k] * row[g];
    }
}

extern "C" __global__
void hole_normal(const double* image, const double* page, const TT* tmpl,
                 const long long* idx, const double* c, const double* a, const double* level, const double* delta,
                 int n, int N, int U, int R, int H, int W, double nv0, double nv1, double rscale, int size,
                 int free_level, double* out)
{
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    int side = 2 * R + 1;
    if (t >= (long long) n * side) return;
    int i = (int)(t / side);
    int oy = (int)(t %% side) - R;
    long long h = idx[i];
    double x = c[2 * h], y = c[2 * h + 1];
    double aa = a[h], lev = level[h], de = delta[h];
    long long rx = (long long) rint(x), ry = (long long) rint(y);
    long long py = ry + oy;
    if (py < 0 || py >= H) return;
    double sy = ((double) py - y) * U + N / 2;
    double acc[22];
    for (int q = 0; q < 22; ++q) acc[q] = 0;
    for (int ox = -R; ox <= R; ++ox) {
        long long px = rx + ox;
        if (px < 0 || px >= W) continue;
        double sx = ((double) px - x) * U + N / 2;
        TT v[5];
        sample(tmpl, N, 5, sx, sy, v);
        double v0 = v[0], vd = v[1], vx = v[2], vy = v[3], v2 = v[4];
        long long f = py * W + px;
        double model = page[f] + lev;
        double resid = image[f] - model;
        double wgt = 1.0 / (nv0 + nv1 * fmax(model, 0.0));
        acc[20] += resid * resid;
        acc[21] += 1.0;
        if (rscale > 0) { double q = resid / rscale; wgt = wgt / (1.0 + q * q); }
        double J[5];
        J[0] = -aa * vx; J[1] = -aa * vy; J[2] = v0 + de * (vd + de * v2);
        J[3] = free_level ? 1.0 : 0.0;
        J[4] = size ? aa * (vd + 2 * de * v2) : 0.0;
        int q = 0;
        for (int r = 0; r < 5; ++r) {
            double jw = J[r] * wgt;
            for (int s = r; s < 5; ++s) acc[q++] += jw * J[s];
            acc[15 + r] += jw * resid;
        }
    }
    double* o = out + (long long) i * 22;
    for (int q = 0; q < 22; ++q) atomicAdd(o + q, acc[q]);
}

extern "C" __global__
void render_scatter(double* page, double* cache, const TT* tmpl3,
                    const long long* idx, const double* c_old, const double* c, const double* a, const double* delta,
                    int n, int N, int U, int Rr, int H, int W)
{
    long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    int side = 2 * Rr + 1;
    long long P = (long long) side * side;
    if (t >= n * P) return;
    int i = (int)(t / P);
    int p = (int)(t %% P);
    int oy = p / side - Rr, ox = p %% side - Rr;
    long long h = idx[i];
    double old = cache[t];
    if (old != 0.0) {
        long long pxo = (long long) rint(c_old[2 * i]) + ox, pyo = (long long) rint(c_old[2 * i + 1]) + oy;
        atomicAdd(page + pyo * W + pxo, -old);
    }
    double x = c[2 * h], y = c[2 * h + 1];
    long long px = (long long) rint(x) + ox, py = (long long) rint(y) + oy;
    double light = 0.0;
    if (px >= 0 && px < W && py >= 0 && py < H) {
        TT v[3];
        sample(tmpl3, N, 3, ((double) px - x) * U + N / 2, ((double) py - y) * U + N / 2, v);
        double de = delta[h];
        light = a[h] * ((double) v[0] + de * ((double) v[1] + de * (double) v[2]));
        atomicAdd(page + py * W + px, light);
    }
    cache[t] = light;
}
'''

TRI = [(r, s) for r in range(5) for s in range(r, 5)]


class Fused:
    """The library's fit_holes with the two heavy steps replaced by the kernels above."""

    def __init__(self, fitter, tt="double", chunks=None):
        import cupy as cp

        self.cp = cp
        self.f = fitter
        self.tt = tt
        mod = cp.RawModule(code=SRC % dict(TT=tt), options=("-std=c++11",))
        self.kA = mod.get_function("hole_normal")
        self.kB = mod.get_function("render_scatter")
        self.dtype = cp.float64 if tt == "double" else cp.float32
        self.chunks = chunks if chunks is not None else fitter._colour_chunks
        self.times = dict(A=0.0, lin=0.0, B=0.0, far=0.0, init=0.0)
        self.sync = lambda: cp.cuda.get_current_stream().synchronize()

    def _grids(self):
        cp = self.cp
        t0, td, tx, ty, t2 = self.f._templates
        g5 = cp.ascontiguousarray(cp.stack([t0, td, tx, ty, t2], axis=-1).astype(self.dtype))
        g3 = cp.ascontiguousarray(cp.stack([t0, td, t2], axis=-1).astype(self.dtype))
        return g5, g3

    def fit_holes(self, iterations=5, size=True, max_move=0.25, stop_px=0.002, robust_c=3.0, timing=False):
        cp = self.cp
        f = self.f
        xp = cp
        g5, g3 = self._grids()
        N, U, R, Rr, H, W = f.template.N, f.template.U, f.R, f.Rr, f.h, f.w
        max_delta = 0.3 * 0.5 * f.template.cd_px(f.theta)
        f.clip_px = float(max_delta)
        npx = H * W
        side_r = 2 * Rr + 1
        P = side_r * side_r
        tick = time.perf_counter
        if timing:
            self.sync()
        t0 = tick()
        flat_page = cp.zeros(npx)
        cache = []
        for idx in self.chunks:
            n = len(idx)
            cj = cp.zeros(n * P)
            c_old = cp.ascontiguousarray(f.c[idx])
            self.kB(((n * P + 255) // 256,), (256,),
                    (flat_page, cj, g3, idx, c_old, f.c, f.a, f.delta, np.int32(n), np.int32(N), np.int32(U),
                     np.int32(Rr), np.int32(H), np.int32(W)))
            cache.append(cj)
        far = f.far_field().ravel() if f.far else None
        if far is not None:
            flat_page += far
        if timing:
            self.sync()
            self.times["init"] += tick() - t0
        if f.history == []:
            f.fit_noise(flat_page.reshape(H, W))
        a_med = float(cp.median(f.a))
        nv0, nv1 = f.var_line
        eye = cp.eye(5)
        iu = ([r for r, s in TRI], [s for r, s in TRI])
        for it in range(iterations):
            stat, dstat, rms_all = [], [], []
            for j, idx in enumerate(self.chunks):
                n = len(idx)
                if timing:
                    self.sync()
                t0 = tick()
                out = cp.zeros((n, 22))
                s = f.rms_typ if f.rms_typ is not None else None
                if robust_c and s is None:
                    # first sweep after a template change: the scale is this chunk's own median leftover (one cheap
                    # pass without the weights' effect on the sums that are read: rss and count)
                    self.kA(((n * (2 * R + 1) + 127) // 128,), (128,),
                            (f.image, flat_page, g5, idx, f.c, f.a, f.level, f.delta, np.int32(n), np.int32(N),
                             np.int32(U), np.int32(R), np.int32(H), np.int32(W), np.float64(nv0), np.float64(nv1),
                             np.float64(0.0), np.int32(size), np.int32(1), out))
                    s = float(cp.median(cp.sqrt(out[:, 20] / cp.maximum(out[:, 21], 1))))
                    out[:] = 0
                rscale = robust_c * max(s, 1e-6) if robust_c else 0.0
                self.kA(((n * (2 * R + 1) + 127) // 128,), (128,),
                        (f.image, flat_page, g5, idx, f.c, f.a, f.level, f.delta, np.int32(n), np.int32(N),
                         np.int32(U), np.int32(R), np.int32(H), np.int32(W), np.float64(nv0), np.float64(nv1),
                         np.float64(rscale), np.int32(size), np.int32(1), out))
                if timing:
                    self.sync()
                    self.times["A"] += tick() - t0
                    t0 = tick()
                k = 5 if size else 4
                normal = cp.zeros((n, 5, 5))
                normal[:, iu[0], iu[1]] = out[:, :15]
                normal[:, iu[1], iu[0]] = out[:, :15]
                normal = normal[:, :k, :k]
                rhs = out[:, 15:15 + k]
                rms_sl = cp.sqrt(out[:, 20] / cp.maximum(out[:, 21], 1))
                rms_all.append(rms_sl.get())
                normal = normal + 1e-9 * eye[None, :k, :k] * cp.einsum("nii->n", normal)[:, None, None] / k
                step = cp.linalg.solve(normal, rhs[..., None])[..., 0]
                step = cp.where(cp.isfinite(step), step, 0.0)
                step = cp.where(f.frozen[idx, None], 0.0, step)
                if size:
                    cov = cp.linalg.inv(normal)
                    f.delta_sd[idx] = cp.sqrt(cp.maximum(cov[:, 4, 4], 0.0))
                c = f.c[idx]
                c_old = cp.ascontiguousarray(c)
                move = cp.clip(step[:, :2], -max_move, max_move)
                f.c[idx] = c + move
                a_old = f.a[idx]
                f.a[idx] = cp.clip(cp.clip(a_old + step[:, 2], 0.5 * a_old, 2.0 * a_old), 1e-3, 5.0 * a_med)
                f.level[idx] = f.level[idx] + step[:, 3]
                if size:
                    new_delta = cp.clip(f.delta[idx] + step[:, 4], -max_delta, max_delta)
                    dstat.append(cp.abs(new_delta - f.delta[idx]).get())
                    f.delta[idx] = new_delta
                stat.append(cp.abs(move).max(axis=1).get())
                if timing:
                    self.sync()
                    self.times["lin"] += tick() - t0
                    t0 = tick()
                self.kB(((n * P + 255) // 256,), (256,),
                        (flat_page, cache[j], g3, idx, c_old, f.c, f.a, f.delta, np.int32(n), np.int32(N),
                         np.int32(U), np.int32(Rr), np.int32(H), np.int32(W)))
                if timing:
                    self.sync()
                    self.times["B"] += tick() - t0
            t0 = tick()
            if far is not None:
                new_far = f.far_field().ravel()
                flat_page += new_far - far
                far = new_far
            if timing:
                self.sync()
                self.times["far"] += tick() - t0
            moves = np.concatenate(stat)
            dmoves = np.concatenate(dstat) if dstat else np.zeros(1)
            f.rms_typ = float(np.median(np.concatenate(rms_all)))
            move99, delta99 = float(np.percentile(moves, 99)), float(np.percentile(dmoves, 99))
            f.log.append(f"    holes it {it + 1}: median move {np.median(moves):.4f} px, 99% {move99:.4f}, "
                         f"delta 99% {delta99:.4f}, typical leftover rms {f.rms_typ:.2f}")
            if move99 < stop_px and delta99 < stop_px:
                break
        return flat_page.reshape(H, W)
```

## Appendix B. Template render with the same values

From `artifacts/outline_fit/speed/fast_render.py`. A subclass here; in the library the three methods replace
`Template.indicator` and `Template.render` and extend `Template.__init__`.

```python
class ExactTemplate(M.Template):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        n, aa = self.N, self.aa
        rho_c = np.hypot(self.xx, self.yy).ravel()
        order = np.argsort(rho_c, kind="stable")
        self._order = order
        self._rho_sorted = rho_c[order]

        def rows(arr):
            return arr.reshape(n, aa, n, aa).transpose(0, 2, 1, 3).reshape(n * n, aa * aa)[order]

        # sub-samples of every fine pixel, pixels sorted by radius; columns in the library's summation order
        self._rho_cols = np.ascontiguousarray(rows(self.rho_aa).T)
        self._ang_cols = np.ascontiguousarray(rows(self.ang_idx_aa).T)
        self._margin = float(np.abs(self._rho_cols - self._rho_sorted[None, :]).max()) + 0.5 / (self.U * self.aa) + 1e-9
        self._outlines = OrderedDict()
        self._blurs = OrderedDict()
        self._boxes = OrderedDict()

    def indicator(self, theta, delta=0.0):
        r, dr = self._radius_at(theta, self.table_angles)
        edge_r = r + delta * np.hypot(r, dr) / r if delta else r
        i0 = int(np.searchsorted(self._rho_sorted, edge_r.min() - self._margin, side="right"))
        i1 = int(np.searchsorted(self._rho_sorted, edge_r.max() + self._margin, side="left"))
        out = np.zeros(self.N * self.N)
        out[self._order[:i0]] = 1.0
        scale = self.U * self.aa
        total = np.zeros(i1 - i0)
        for k in range(self.aa * self.aa):  # the library's sum runs over the sub-samples in this order
            total += np.clip(0.5 + (edge_r[self._ang_cols[k, i0:i1]] - self._rho_cols[k, i0:i1]) * scale, 0.0, 1.0)
        out[self._order[i0:i1]] = total / (self.aa * self.aa)
        return out.reshape(self.N, self.N)

    @staticmethod
    def _remember(memo, key, make, keep=16):
        value = memo.get(key)
        if value is None:
            value = make()
            memo[key] = value
            if len(memo) > keep:
                memo.popitem(last=False)
        else:
            memo.move_to_end(key)
        return value

    def render(self, theta, delta=0.0, box_px=0.0):
        theta = np.asarray(theta, float)
        spec = self._remember(self._outlines, (theta[:self.n_shape].tobytes(), float(delta)),
                              lambda: np.fft.fft2(np.fft.ifftshift(self.indicator(theta, delta))) * self.edge_fix)
        otf = self._remember(self._blurs, theta[self.n_shape:].tobytes(), lambda: self.otf(theta))
        spec = spec * otf * self.pixel_otf
        if box_px > 0:
            sx, sy = self._remember(self._boxes, float(box_px),
                                    lambda: (np.sinc(self.fx * box_px), np.sinc(self.fy * box_px)), keep=8)
            spec = spec * sx * sy
        return np.fft.fftshift(np.fft.ifft2(spec).real)
```

## Appendix C. Leftovers on the device

From `artifacts/outline_fit/speed/extra_patches.py`: copies of `PageFit._set_far`, `far_field` and `fit_noise`.
Against the library they differ only in: the FFT of the far kernels on the device (`xp.fft.rfft2(pad)` with
`pad` an `xp` array), the padded grid from `far_shape` (grown to an FFT-friendly size; GPU only, see step 3),
and `fit_noise` using `xp.percentile` / `xp.median` on the device.

```python
def _even_fast(n):
    from scipy.fft import next_fast_len

    m = next_fast_len(int(n), real=True)
    while m % 2:
        m = next_fast_len(m + 1, real=True)
    return m


def install(P):
    def far_shape(self):
        my, mx = self._far_margin
        return _even_fast(self.h + 2 * my), _even_fast(self.w + 2 * mx)

    def _set_far(self, t0, td, t2, r0):
        xp = self.xp
        t = self.template
        self.far_r0 = float(r0)
        w = self._ramp(np.hypot(t.xx, t.yy), r0)
        t0n, tdn, t2n = t0 * w, td * w, t2 * w
        tx, ty = t.derivatives_xy(t0n)
        self._templates = [xp.asarray(g) for g in (t0n, tdn, tx, ty, t2n)]
        self.Rr = min(int(math.ceil(r0 + self.far_ramp)) + 1, t.N // (2 * t.U) - 2)
        if self._far_template is None:
            self._far_template = P.Template(upsample=self.far_upsample, size_px=self.far_box, harmonics=t.ks,
                                            blur=t.blur, aa=t.aa)
        ft = self._far_template
        b0, bd, b2 = ft.render_quad(self.theta)
        B, U = self.far_box, ft.U
        k = np.arange(B) * U
        rr = np.hypot(*np.meshgrid(np.arange(B) - B // 2, np.arange(B) - B // 2))
        wf = 1.0 - self._ramp(rr, r0)
        hp, wp = far_shape(self)
        ffts = []
        for g in (b0, bd, b2):
            pad = xp.zeros((hp, wp))
            pad[:B, :B] = xp.asarray(g[np.ix_(k, k)] * wf)
            ffts.append(xp.fft.rfft2(pad))
        self._far_fft = ffts

    def far_field(self):
        xp = self.xp
        B = self.far_box
        my, mx = self._far_margin
        hp, wp = far_shape(self)
        c = self.c + xp.asarray([mx, my], dtype=xp.float64)
        ix, iy = xp.floor(c[:, 0]), xp.floor(c[:, 1])
        wx = P._keys_weights(c[:, 0] - ix, xp)
        wy = P._keys_weights(c[:, 1] - iy, xp)
        taps = xp.arange(-1, 3)
        cx = ix.astype(xp.int64)[:, None] + taps
        cy = iy.astype(xp.int64)[:, None] + taps
        ok = (cx[:, 0] >= 0) & (cx[:, -1] < wp) & (cy[:, 0] >= 0) & (cy[:, -1] < hp)
        cx, cy = xp.clip(cx, 0, wp - 1), xp.clip(cy, 0, hp - 1)
        flat = (cy[:, :, None] * wp + cx[:, None, :]).reshape(len(c), -1)
        wgt = (wy[:, :, None] * wx[:, None, :]).reshape(len(c), -1) * ok[:, None]
        spec = None
        de = self.delta
        for g, val in zip(self._far_fft, (self.a, self.a * de, self.a * de * de)):
            s = xp.bincount(flat.ravel(), weights=(wgt * val[:, None]).ravel(), minlength=hp * wp).reshape(hp, wp)
            term = xp.fft.rfft2(s) * g
            spec = term if spec is None else spec + term
        far = xp.fft.irfft2(spec, s=(hp, wp))
        oy, ox = my + B // 2, mx + B // 2
        return far[oy:oy + self.h, ox:ox + self.w]

    def fit_noise(self, page):
        xp = self.xp
        res_all, lev_all = [], []
        for sl in self._slices()[:3]:
            flat, valid, px, py = self._pixels(sl, self.R)
            model = page.ravel()[flat] + self._background(sl, px, py)
            res = self.image.ravel()[flat] - model
            res_all.append(res[valid])
            lev_all.append(model[valid])
        res = xp.concatenate(res_all)
        lev = xp.concatenate(lev_all)
        edges = xp.percentile(lev, xp.linspace(0, 100, 13))
        xs, ys = [], []
        for i in range(12):
            s = (lev >= edges[i]) & (lev < edges[i + 1])
            if int(s.sum()) > 200:
                r = res[s]
                mad = 1.4826 * float(xp.median(xp.abs(r - xp.median(r))))
                xs.append(float(lev[s].mean()))
                ys.append(mad ** 2)
        xs, ys = np.asarray(xs), np.asarray(ys)
        if len(xs) >= 3:
            v1, v0 = np.polyfit(xs, ys, 1)
            v1 = max(v1, 0.0)
            v0 = max(v0, 0.05 * ys.mean())
        else:
            v0, v1 = max(float(ys.mean()), 1e-3), 0.0
        self.var_line = (float(v0), float(v1))

    P.PageFit._set_far = _set_far
    P.PageFit.far_field = far_field
    P.PageFit.fit_noise = fit_noise
```

## Appendix D. Measuring

From the repo root, `.venv/Scripts/python.exe -X utf8 artifacts/outline_fit/speed/<script> ...`; results go to
`artifacts/outline_fit/speed/runs/`. Check `nvidia-smi` first: the GPU is shared with the user's own jobs.

```text
prof_outline.py mode=plain out=base                      the unchanged fit of page 0: seconds, rounds, sweeps
prof_outline.py mode=timed                               the same with a timer round every stage
prof_outline.py mode=plain fused=2 shape=exact extra=1 out=new     steps 1 to 3 patched in
        page=24 | blur=lens | tile=4 rows=5408 | backend=cpu crop=600,720,900,1070 | shape=real (step 4)
compare_runs.py base new                                 differences in nm (outline CD, per hole, centres)
fused_proto.py                                           one and three sweeps, library against kernels
fast_render.py runs/dump_p0_shape.npz                    render new against old; shape stage replayed
batch_bench.py seeds pages=0:48                          seeds of every page, once
batch_bench.py launch workers=4 per=3                    worker processes on one GPU
thread_batch.py threads=10 pool=5 n=20                   one GPU job, pages on threads (step 5)
batch_holes.py k=1,2,4,8                                 K pages as one GPU problem (step 6's parts)
```

`batch_bench.py worker warm=...` and `batch_compare.py` belong to the withdrawn reference start.
