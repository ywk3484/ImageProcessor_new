One-pass outline fit: scratch prototype and measurements (2026-11-12). Nothing in subpx/ was changed.
----------------------------------------------------------------------------------------------------
What to build from this:     docs/outline_fit_fast_handoff.md   (the user approved the build on 2026-11-12)
Method, settings, numbers:   docs/outline_fit_one_pass.md       (its measurements section is written by make_notes.py)
Page the user read:          docs/outline_fit_one_pass.html     (built by make_page.py from runs/)

The .py files, page_template.html, rerun_all.sh and this file are tracked in git (force-added: artifacts/ is
ignored); runs/ (results, logs, the existing fit's reference runs) is on the development PC only.
Run from this folder:  ..\..\..\.venv\Scripts\python.exe -X utf8 <script> key=value ...
The GPU is shared with the user's own jobs: look at nvidia-smi before a run.

The method (the reference implementation for the port)
  spec.py          the hole template from its spectrum (polygon form; the hole passes' templates, the far kernel)
  shapefit.py      the page fit: outline integrals (one kernel), synthesis at the bins, scale and constant
                   projected out, Levenberg-Marquardt on the host for many pages in lockstep (NumPy or CuPy)
  gpufit.py        GpuPage: seeds, plain stack, pair maps (Gamma), templates, hole passes (CUDA kernels), noise
                   line, results
  pipeline.py      fit_pages / fit_page: steps A to D for a batch of pages; level=1 page values, level=2 with a
                   size per hole
  fastfit.py       the first, CPU-only version of the stack and the pair maps (NumPy); used by truth.py

Measurements
  truth_gpu.py     one generated page with known sizes: one pass, the page fit with the true centres, the
                   existing fit.   truth_gpu.py circle100_square family=lens variants=fast,truth,existing
  truth_matrix.py  sets of generated pages: set=main | blur | more [existing=0] out=TAG -> runs/TAG.json
  small_pages.py   small images (the library's test page and larger) with the seeds measure_cds hands over and
                   with the method's own; existing=1 adds the library's outline_fit
  port_checks.py   the numbers quoted for the port's tests: Jacobian against differences and recovery of a known
                   outline under NumPy; seconds for one full page per call
  run_file.py      one pass over the test file's 40 hole pages against runs/ref: page CD, per-hole CD, ms per
                   page by phase.   run_file.py batch=20 level=2|1 [family=lens] lines=0 repeat=3 out=TAG
  ref_runs.py      the existing fit (the same-numbers prototype in ../speed, this PC only) on every page of the
                   test file -> runs/ref/ref.json, ref_pN.npz                              about 8 minutes
  truth_extra.py   the existing fit on the blur series and four further pages -> runs/truth_extra_existing.json
  truth.py         a generated page through the CPU prototype (fastfit.py + shapefit.py under NumPy)
  checks.py        other (pages without the hole pattern) | stable (further centring rounds) | load (CPU, GPU
                   memory, pages per call) | repeat (the same pages twice)
  stable2.py       page CD against the number of centring sweeps
  util.py          GPU utilisation while fitting
  seed_test.py     the seed finder on the 13 generated patterns
  bench1.py        page fit against the number of pages fitted together; hole passes single / double precision
  bench2.py        one hole sweep, kernel by kernel
  check_spec.py    the analytic template against the library's Template.render
  table.py         prints runs/truth_*.json as a table
  rerun_all.sh     every one-pass measurement behind the page, in order (about 10 minutes)

The page and the notes
  make_page_data.py   runs/*.json -> runs/page_data.json and the picture runs/fig_average_hole.png
  make_page.py        page_template.html + page_data.json -> docs/outline_fit_one_pass.html
  make_notes.py       the measurements section of docs/outline_fit_one_pass.md

Result files in runs/ (the ones the page and the notes use; this PC only)
  ref/                          the existing fit on the test file (gauss2)
  truth_main.json, truth_more.json, truth_extra_existing.json, truth_part.json     the existing fit on generated pages
  truth_main_v4.json, truth_blur_v4.json, truth_more_v4.json                       one pass, final prototype
  file_l2_v4.json, file_l2_lens_v4.json, file_l1_v3.log                            one pass on the test file
  check_load.log, check_repeat.log, check_util.log, truth_copy.log
  v3 = before the two start fixes for whole-pixel pitches (lens results identical, gauss2 within 0.013 nm on the
  test file); v2 and older = one centring sweep instead of two.

Settings the numbers were made with: two centring sweeps, three size sweeps with step factor 1.3, plain stack,
4 bins per pixel, single-precision templates, 20 pages per call.

What the test file is: artifacts/cd_test/data/stripe1_board1.tif is GENERATED (exact 8.5 px lattice, +-0.3 px
uniform shift per hole); there is no real ACTIS page on this PC.
