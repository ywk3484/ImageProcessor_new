Finished tasks up to 2026-10-26 (both parts of the diagnostic) are in `tasks/archive/todo-2026-10.md`;
from November 2026 in `tasks/archive/todo-2026-11.md` (including the dropped learned-template plan).

# Workbench UI/UX rework (review 2026-10-31; decisions made, implementation in a new session)

Review, target layout and phases: `docs/superpowers/specs/2026-10-31-workbench-ux-review-design.md`. Baseline:
`scripts/workbench_timing.py` on the 39,042-hole page (click 0.9-2.0 s, full-view repaint 0.55 s; default preset:
gradient 94 % and halfmax 74 % failed). No library change; CD numbers unchanged. Goal now: an inspection UI for
testing interactively on sample images.

- [x] Decisions, baseline, P0 fixes, P1 speed, P2 new layout: done, with their reviews, in
      `tasks/archive/todo-2026-11.md`
- [ ] P3 later: open a cd_batch run (run.json settings, pages.csv page list, batch vs workbench mean)

# Outline fit: CD and centers from the hole's sharp outline and the page's own blur (2026-11-04)

User: the report's CD / registration methods are "not so satisfactory in terms of accuracy"; wants a completely new
method that breaks through the limit, not polishing of the test pages; interpolation / upscaling allowed.

Diagnosis (from the report and the equipment notes, not a new test): for pitch, the per-hole noise floor set by the
page's light is already reached by the dropped learned method (Gaussian fit 7-34 % above it), so no center method can
be much better at this light; what is left there is the dense-page bias (step read 1.08-1.19). For CD every existing
method reads ~0.5 nm per true nm (slope 0.35-0.58 on 80 nm holes) because it measures blur and hole size together:
image width^2 = blur^2 + size^2/16, and the blur is the larger part. The same measure moves ~3 nm per 1 nm of blur
change, so focus drift between scans would dominate repeatability. Brightness leaks in through the background.

Idea: a mask hole is a sharp (binary) outline; the tool's blur is smooth and the same for every hole on the page.
Fit the page with one shared outline (any star-convex shape) convolved with one shared blur, each hole at its own
position, brightness, local level and edge offset (size). The sharp edge puts a zero ring in the hole's spectrum at a
radius set by its size alone; the blur dims the ring but cannot move it, so size and blur separate on the page
average (30k holes, noise /170). Per-hole CD = outline CD + 2 x edge offset; slope 1 by construction, brightness
free, blur fitted per page (so focus moves the blur, not the CD). Centers come from the same fit (neighbours
modelled, so no dense-page bias). Premise checks: binary edges (mask absorber, LER << pixel), shared compact blur
(same optics; TDI smear makes it anisotropic -> anisotropic blur model), real average hole is flatter-topped and
steeper-sided than a Gaussian (measured on page 0: at r = 3 px 24 vs 31 grey levels for a Gaussian of the same
half width), real noise 7.5 grey levels flat across levels (measured).

- [x] Finished items of this task (identifiability, prototype, slope check, hole stage, the slope and offset's
      cause and fix) are in `tasks/archive/todo-2026-11.md` (moved 2026-11-05)
- [ ] CD scoring on the stress pages exactly like the report (slope, scatter in true nm, +10 % light) + page-mean
      error + a focus test (same mask rendered at blur 58 / 65 / 72 nm; existing methods for comparison); the same
      pages with the generator's own blur form (what the family costs). Queue: `runs/driver.py` (CPU only, two at a
      time: the machine's commit limit is nearly filled by other work; a CUDA context alone commits 2-3 GB). First
      results (before the hole-stage rework, copies in runs/*_v1.json): circle80 slope 1.080, scatter 1.86 nm,
      +0.24 nm per 10 % light, mean -0.85 (Gaussian fit 0.477 / 1.81 / +0.22 / +4.51); circle100 1.054 / 1.19 /
      +0.21 / -0.48; square150 1.004 / 0.53 / +0.21 / +0.27 (edge area 1.021 / 0.86 / -0.61 / +36.2).
      Generator's own blur form (lens family, reworked hole stage): circle80 1.065 / 1.89 / +0.25 / -0.36 (outline
      79.64, p 1.90, blur 64.2 for a true 65); circle100 1.033 / 1.22 / +0.13 / -0.24: the right family takes about
      half the slope excess and the offset away, the light move stays. RERUN with the fixed hole stage (driver5, Nov
      5; slope / scatter / +10 % light / mean error): circle80 lens 1.013 / 1.96 / +0.08 / -0.16 (was 1.065 / 1.89 /
      +0.25 / -0.36), circle100 lens 1.008 / 1.24 / +0.03 / -0.19 (was 1.033 / 1.22 / +0.135 / -0.24); centers step
      read 1.011 and 1.000 (was 1.030, 1.012); 1.7-1.9 x the time per page (far field, 14 px render, frame holes).
      Focus series, 80 nm circles, blur 58 / 65 /
      72 nm (two Gaussians): page CD 79.14 / 79.15 / 78.95 (moves 0.2 nm; fitted blur 57.7 / 64.9 / 71.9), slope
      1.078 / 1.080 / 1.085, scatter 1.49 / 1.86 / 2.44; Gaussian fit page mean +0.03 / +4.51 / +8.67 (moves 8.6 nm),
      edge area +22.1 / +27.3 / +32.3 (10.1 nm), half area -2.1 / +0.3 / +2.6 (4.7 nm). 100 nm circles: page CD
      99.67 / 99.51 / 99.42 (moves 0.25 nm; fitted blur 57.0 / 64.5 / 72.0), slope 1.040 / 1.054 / 1.067; Gaussian
      fit -10.0 / -5.8 / +0.4 (10.4 nm), edge area +15.5 / +20.1 / +26.5 (11.0 nm), half area -5.9 / -4.2 / -2.0 (3.9 nm)
- [ ] What is left on the exact noise-free page after the hole-stage fix (Nov 5, runs/nf_M/N/O/P*.log): the whole
      fit (template free, new defaults) reads CD slope 1.022, mean error +0.13 nm, scatter 0.06 nm, +0.10 nm per
      10 % light, blur 64.0 for 65, p 2.03 (before the fix 1.071, +0.25). Each stage alone is exact: the hole stage
      with the exact template 0.999 x - 0.01 nm (nf_H2), the shape stage on the stack of the holes held at the
      truth CD +0.03 nm, blur +0.05 nm, p +0.005 (nf_N, 96 and 48 px templates). ORACLE (nf_P, every level held at
      the true background through the alternation): slope 0.999, +0.01 nm per 10 %, blur 64.9, p 1.99, scatter
      0.02 nm, converged in 11 rounds - the holes' free levels are the coupling that pulls the joint point off (the
      mean error +0.12 nm stays: outline 80.15 for 80.0). The page-background grid in the levels' place is no remedy
      as built (nf_O: 1.065, -0.11 nm, scatter 0.42 nm, +0.21 nm per 10 %, blur 64.3 after 13 rounds: the common
      shift of size, brightness and background settles too slowly in alternation). Costs 2 % of the CD spread and
      0.1 nm per 10 % light. Next step if wanted: the background solved jointly with the holes' common size and
      brightness shift (two dense columns in fit_background), not built; production default stays level per hole
- [ ] Centers scored with pitch_signature_test.evaluate on the six report pages (columns, step, noise at 8, error).
      50 nm dense page: the fit FAILS as the bound predicts - outline 66.7 nm under a 36.9 nm blur (true 50 / 65), the
      size-blur degeneracy below the cut-off; the per-hole switch did not catch it (sd 0.11 px against a sharp
      template looks readable); centers 1.10 nm vs Gaussian 1.06, step 1.32 vs 1.19 (worse). The only sign is the
      fitted blur, half of the other pages'. Page and todo say so; a blur prior from a big-hole page of the same tool
      is the obvious (unbuilt) remedy. QUEUE: paused twice on Nov 5 when the machine's committed memory filled (other
      work; 90.7 of 91.2 GB both times), resumed as driver3/driver4 with the per-hole-level code (12 of 17 done),
      then stopped by me at about 09:05 when the hole-stage defect (slope item, (6)) was found: every result so far
      came from the defective code (kept in runs/before_fix/). Relaunched in full at about 09:20 as driver5 (two
      workers, 17 jobs, the new defaults): `cd artifacts/outline_fit; python -X utf8 runs/driver.py workers=2
      status=driver5`, then `python -X utf8 make_page.py` (to show the before_fix numbers beside the new ones). The
      grid background can run beside it later with `extra=bg=grid suffix=_grid`. MOVED TO THE GPU at 13:00 after the
      user asked why the CPU (the morning's memory squeeze was over, the GPU idle): the CPU driver stopped after 9 of
      17 (its two in-flight centers pages finish on their own), the 80 nm stress page run on the GPU as the test
      (`backend=gpu`, far field included: fit 190 s for 48,689 holes and 14 rounds, 7-10 s a round once settled,
      against about 3,000 s on two CPU threads; the existing methods for comparison take 279 s on the CPU), then
      the remaining six jobs as `runs/driver.py workers=2 gpu=1 status=driver6 only=... skip=stress_circle100_square_lens`.
      Lesson applied: recheck the machine's state before relaunching a queue under a rule made for an earlier state.
      QUEUE COMPLETE 14:10 (17 of 17, all exit 0; GPU jobs 143-269 s of fit per full page, the 96k-hole 50 nm page
      238 s); page rebuilt (`outline_fit.html`, before-fix rows beside the new ones). Old -> new (slope / light per
      10 % / mean error): circle80 gauss2 1.080 / +0.24 / -0.85 -> 1.043 / +0.12 / -0.72; circle80 lens 1.065 /
      +0.25 / -0.36 -> 1.013 / +0.08 / -0.16; circle100 gauss2 1.054 / +0.21 / -0.48 -> 1.053 / +0.21 / -0.45;
      circle100 lens 1.033 / +0.13 / -0.24 -> 1.008 / +0.03 / -0.19; square150 1.004 / +0.21 / +0.27 -> 1.008 /
      +0.20 / +0.20; focus 80 nm slope 1.078 / 1.080 / 1.085 -> 1.066 / 1.043 / 1.069; centers step read 1.021 ->
      1.012 (circle80), 1.030 -> 1.011 (circle80 lens). Unchanged where the degeneracy dominates (dense 50 / 60 /
      70x45 pages) and on the two sparser ELLIPSE pages, which land on blur 52 for 65 and p 2.9-3.5 for 2.0 with
      the CD +5 to +8 nm (90x60 stagger +7.8, 60x100 square +5.4): the outline exponent trades against the blur
      for elongated holes - unresolved, said on the page; a blur prior from a round-hole page is the remedy to
      build. Real page 0: outline CD 100.6 (gauss2) / 100.5 (lens), blur 63 / 63, p 2.00 / 2.05, per-hole robust
      sd 2.6 nm, leftover 7.32 against noise sd 7.7, stack chi2 0.96; the two families agree to 0.1 nm; Gaussian
      fit 96.1, edge area 122.0, half area 92.6 on the same holes (corr with outline 0.83 / 0.81 / 0.41)
- [x] Outline fit in the learned-hole report's format (user, Nov 5 afternoon: "add the results for the new method
      in the report.html format so that I can visually diagnose what is going on"). Plan: (1) run the outline fit
      on the report's seven remaining pages and the three two-exposure pages on the GPU (runs/driver.py gpu=1
      only=centers_,two_ resume=1 status=driver7; run_synth.py gained read_noise=); (2) make_report3.py in
      artifacts/learned_hole_prototype on top of make_report2: "Outline fit" as a method (group new) on every page it
      ran on, scored by the report's evaluate() from the saved centers, in the explorer, matrix, best table (own
      column) and floor chart (own marker); a section on top with, per page, the page piece, the fit's model rebuilt
      from the saved fit, the leftover at 4 sigma, the fitted outline over the true one and the fitted hole profile
      over the true hole's, plus the pitch table (existing -> learned -> outline) and a CD table on the four stress
      pages; (3) report.html rewritten (v2 kept as report_v2.html), layout checked in the browser.
      DONE 2026-11-05 ~17:40. Queue driver7: 10 pages on the GPU in 1559 s (60-312 s each). Report: outline fit on
      18 of 18 pages (13 report pages, 2 stress, 3 two-exposure), 18 picture rows, pitch and CD tables, floor chart
      mark, best-table column, explorer default. Found by the pictures: (a) 300 nm squares at the 24 px template
      box read blur 89 nm for 65, centres 1.5 x the bound with a 0.22 nm pixel-position error (the hole's light
      reaches 11 px, the box rendered 10): fit.py now doubles the template box when the hole's light reaches past
      it; rerun reads blur 65.1, centres 0.18 nm against the 0.17 bound, CD -0.2 nm, 2 columns (was 4), at 2.7 x
      the time (runs/centers_square300_square_t24.* kept); the 250 nm circles grow too and are unchanged. (b) The
      60 nm dense page was never a reading: the fit walks the size-blur valley round after round (CD 72 -> 62 nm,
      blur 46 -> 61 over 14 rounds, still moving 5 nm a round when the rounds ran out); with two exposures the walk
      goes further and the last shape proposal jumps to CD 31 nm with 74 % of the holes at the edge-offset clip.
      Both said on the page (status per page: read right 9, size-blur 3, walking or ran away 2, boxy 4). Columns
      needed against the best existing method: fewer on 0 pages, as many on 14, more on 4 (the dense ones); step
      0.99-1.03 on the 9 pages read right. Method page (outline_fit.html): centers table extended to the 13 pages,
      template-growth finding in the open item. Still open: the dense and elongated degeneracies (blur prior from a
      round-hole page), the alternation's stopping rule on a degenerate page (it should stop and say so, not walk)
- [ ] Real page 0 (and several pages): residual vs noise, fitted blur, CD distribution; page-mean CD across the hole
      pages against the Gaussian fit's
- [ ] HTML page (problem, idea, flow, each step with why, payoff, is it general, checks on own images); review here;
      proposal for the library (new methods beside the existing ones)
      2026-11-08: the idea section expanded after review (the 1.22/D looked like the Airy formula; arrays vs
      isolated holes): shapes and their spectra with how they are computed (disk, blurred disk, edge-less blob,
      square; signed cuts), where 1.22 comes from (2 J1(pi D nu)/(pi D nu), the Airy duality, zero-ring table
      against the cut-off), isolated vs array (comb under the envelope, harmonic tables for 160/200 nm pitch,
      neighbour share 7.5 % in the window, the fit's de-neighboured average) and an assumption box: the model adds
      light (incoherent imaging); partial coherence on the tool is unchecked. make_page_data.py -> page_detail.npz
      2026-11-09: rewritten as lecture notes after review (compact prose rejected): 8 numbered parts, waves ->
      slit -> disk (chord integral to 2 J1) -> Airy duality -> convolution theorem and the transfer functions ->
      how the pictures are made and why the 0.24 c/um padding step carries no information (un-padded samples
      shown on the padded curve; a hard-cut 12 px window adds +-2 thousandths of ripple, as large as the mark, so
      the fit compares pixels, not spectra) -> gallery -> array (comb derivation, line tables, de-neighbouring) ->
      how light adds (1-D coherent vs incoherent example, 5.7 % difference) -> summary
      2026-11-09 (later): review: still not understandable; "cycles per um" had no illustration and the text never
      said which image F belonged to (whole page? Voronoi cell?). Added part 0 "Which picture, exactly" from a real
      fit on the smoke page (make_which_picture.py: page piece with the window outlined, the 13 px window as pixels /
      model of everything else / cleaned, raw vs cleaned average hole, fitted template, leftover at the noise:
      2.4 thousandths measured vs 2.2 expected) and rewrote part 1 (make_primer_data.py): wave strips on a 1 um
      scale beside the hole, period table, 2-D waves with the frequency plane, the hole rebuilt from waves up to
      3/6/9/12.5 c/um (energy 38/84/98/99.98 %), F(0,0) and the sign stated on picture (c), a 1-D worked case
      (sharp disk's column sums = sum of cosines with weights 2J1 at k/W, sign change from wave 5). Found on the
      way: the 320 nm picture field holds 95 % of the blurred hole's light and the 12 px window 93 % (lens tail);
      the 5.3 text now says so
      2026-11-09 (night): third review: panel letters reused across figures ("picture (c)" was ambiguous), maths
      unreadable as HTML, 1.5 not followable, repeats, the o / h sentence without a picture, static where moving
      would help. Done: every figure numbered in page order (figure()/fig() placeholders, number_figures), letters
      only inside one figure, text refers to "Fig. N(b)"; TeX via a local KaTeX 0.16.11 (katex/, works from
      file://); 1.5 rewritten in four steps (column sums as a sweep animation, which waves fit the field, the weight
      integral and its link to F(k/W, 0), the sum built one cosine per frame, why waves 5-8 are subtracted: they
      take height off the middle and put it at the shoulders); 4.1 has the mask / point image / picture figure, the
      picture built from 80 point images row by row (1 % from the exact convolution) and the 1-D sliding product;
      1.3 is an animation over every whole c/um (energy share per frame); repeats cut (part 0's F(0,0) sentence,
      1.3's and 6's "this is why the average hole", 5.3's window description, 7.3's re-description of the cleaning,
      the post-lecture "never computes a spectrum", part 8 down to four lines). Animations: play / pause, step,
      scrub, autoplay in view, off under prefers-reduced-motion. Page 1.7 MB

# Outline fit: speed (2026-11-12; proposal only, nothing built)

User: the outline fit "seems to work pretty well and I would like to test this on my real data. However, its
extremely slow. Diagnose why this method takes so long ... Do not change any code at the moment and just make a
proposal."

Measured with the unchanged library on real pages of stripe1_board1.tif (2048 x 1378, 39,042 holes), RTX 4060 Ti
idle, float64, gauss2 (scripts, logs and every number: `artifacts/outline_fit/speed/README.txt`): 104 s for page 0
(10 rounds, 43 hole sweeps), 133-154 s for page 24 (13-14 rounds), 255 s with the lens family, 330 s for a
raw-image-sized picture (5408 x 2048, 117k holes), 30 s for a 285-hole region on the CPU. Where the 104 s go:
- hole stage 68 s: 58 s of it is `cubic_sample` (reading the templates at every window pixel), run as 17,000
  separate GPU operations per sweep that create 82 GB of temporary arrays per sweep for a 22 MB page: the GPU is
  96 % busy moving memory. The 5 x 5 solves take 0.3 s, the far-field FFT 1.0 s
- shape stage 27 s on one CPU core: 4,000 template renders at 6.6 ms, 5.1 ms of each recomputing the outline's
  coverage on all 768 x 768 sub-samples although only the ring on the edge (under 1 % of the pixels) is not 0 or 1
- the same sampler in the stack, start and size check 5 s; far kernels' FFT on the CPU 2.4 s; noise line 1.1 s

Proposal (same model, procedure and stop rule; each step prototyped and run end to end on page 0):
- [ ] 1. Hole stage on the GPU as two fused kernels (per hole: sample the templates and sum the normal equations;
      per pixel: old light out, new light in), the sampler as one kernel for its other callers; the NumPy path stays
      as it is (the reference). One sweep 1.29 -> 0.062 s, every parameter equal to 1e-12; page 104 -> 42 s
- [ ] 2. Template render with the same values bit for bit: coverage sub-sampled only in the ring the edge can
      reach, outline spectrum / blur transfer function remembered between renders. Render 6.6 -> 1.6 ms, shape
      stage 27.8 -> 5.4 s with identical fitted parameters; page -> 15 s; the CPU region 30 -> 9 s, same result
- [ ] 3. Far kernels' FFT and the noise line on the device (GPU only). Page -> 10 s
- [ ] Tests (one sweep kernel against NumPy, render new against old, GPU against CPU fit), the times in
      docs/outline_fit.md, cd.py and the report scripts' hint, pages 0 / 24 and two stress pages rerun for the numbers
Result of 1-3 in the prototype: page 0 104 -> 10.3 s, page 24 133 -> 14 s, lens 255 -> 36 s, the 117k-hole picture
330 -> 27 s. Outline CD within 0.005 nm, per-hole CD 0.004-0.008 nm rms: what two runs of today's code differ by on
page 0 (0.009 nm rms). 400 pages: 12-17 h -> 1.1-1.6 h on this GPU, plus the other methods' 8-11 s per image.
- [ ] 4. (optional) half-spectrum FFT in the render: page 0 10.3 -> 7.9 s, lens 36 -> 17 s. Values equal to 1e-15,
      but on page 24 the fit then lands on blur 63.3 / 62.3 instead of 62.6 / 63.2 nm (page CD -0.015 nm, per-hole
      0.09 nm rms; the lens family did not move): for the lens family only, or not at all
Rejected after measuring: the outline fit continued from the current template without its grid of starts (no time
gained, page CD -0.05 nm); threads for the renders (5.4 -> 4.0 s); float32 (no gain once fused); one chunk per
colour (another sweep order). Not in the proposal: the CPU backend's hole stage (a full page stays tens of minutes;
a kernel there needs a compiled dependency); fewer rounds by starting from the previous page's outline and blur
(ties pages together, not measured).
Found on the way, not a speed matter and not caused by the proposal (cause not isolated): two runs of today's code
on page 24 give page CD 100.72 (converged in 13 rounds) and 100.83 nm (round limit reached, flagged NOT converged);
on the 285-hole CPU region a change in the 13th decimal moves the outline CD by 0.26 nm (both runs converged).

## Batch (2026-11-12, the user's reply; proposal only)

User: 10 s per page "may not be enough": an image is one TIFF of up to 1,500 pages, a stripe at least 4 images,
~900 stripes per dataset (not that many for now). "Is there a way to batch process this method?" Remote: 4 x RTX
6000 Ada 48 GB. Measured on this PC (RTX 4060 Ti, 6 cores) on real pages with the prototype of steps 1-3 (gauss2
unless said; numbers and logs in `artifacts/outline_fit/speed/README.txt`, part "Batch"):
- several worker processes per GPU (what cd_batch does today): 1 / 2 / 3 / 4 / 6 workers -> 10.8 / 7.3 / 6.1 / 5.4 /
  5.1 s per page. Twice the throughput at best: not the way
- reference start (a page started from another page's fitted outline and blur, every stage released, same stop
  rule): 2.2 rounds and 14 hole sweeps instead of 10.6 and 42, 3.7 s per page instead of 11.1 (8 pages). Page CD
  against the from-scratch fit: sd 0.024, largest 0.037 nm (two from-scratch runs: sd 0.016, largest 0.046 nm).
  Taking page 24 instead of page 0 as the reference moves all 8 pages together by +0.029 nm (sd 0.012), and a stop
  tolerance of 0.02 nm does not change that. Lens family: from-scratch fits repeat exactly, the two references are
  0.130 nm apart (sd 0.002 over the pages); pages 0 and 24 are 0.20 nm apart from scratch, 0.07 with either reference
- why: left running, a fit keeps drifting after the stop rule has ended it. Lens, page 24 from page 0's fit:
  +0.004 nm per round through round 16 with a flat chi2. gauss2, page 0 from page 24's fit: 100.64-100.67 from
  round 2 to 16; from scratch it stopped at 100.59 with proposals still at +0.03 to +0.06 nm
- one GPU job with the pages on threads (one CUDA context) and the shape fits in 5 CPU processes: from scratch
  5.4 s per page with 10 pages at once, reference start 1.9 s per page (10 GB of GPU memory for 10 pages); the same
  pages in two runs differ by up to 0.016 nm (from scratch 0.04-0.06)
- K pages inside one GPU problem: hole stage 36 ms per page and sweep (60 for a page alone in chunks of 2000);
  float32 template arithmetic 22 ms; template renders as a GPU batch 0.03 ms each (1.6 ms on the CPU)
Proposal, after steps 1-3:
- [ ] 5. `outline_fit` as a cd_batch method: one job per GPU fitting several pages at once, shape fits on a pool of
      CPU processes, seeds from the threshold pass (no deblend), reference start per image (reference = the image's
      first hole page fitted from scratch, or a saved fit given on the command line so that datasets share it).
      This PC: 1.9 s per page (104 today). Remote, 4 GPUs: 0.3-0.5 s per page estimated, not measured there: a
      1,500-page image in 8-12 min, a stripe of 4 in 30-50 min
- [ ] 6. if more is needed: the pages inside the GPU routines (one launch for a chunk of every page of the batch,
      templates and noise lines per page). From the measured parts about 1 s per page here and 0.1 s on the remote
      if it has ~20 free CPU cores: an image in 2.5 min, a stripe in 10 min, 900 stripes in about 6 days
Unknown: the remote's CPU cores, its GPUs' speed against the 4060 Ti (3 x assumed), the share of hole pages.
For the user to decide: the reference start (it ties every page of an image to the reference's level).
