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

- [x] Identifiability check (`check_identify.py`): noise-free hole, two blur families. With the generator's own
      family (lens) the size is exact on a 48 px template grid (the 24 px grid wrapped the lens tail: -1.7 nm);
      with a family unlike the generator's (two Gaussians) the size is recovered to -0.1 nm at 100 nm, -1.5 at 80,
      -3 at 70, -6 at 60, -12 at 50: below ~70 nm the sharp edge's zero ring lies beyond the optics' cut-off and
      size and blur cannot be told apart by any method. Focus drift 58 -> 72 nm moves the fitted CD 0.4 nm (100 nm
      circle), 2 nm (80 nm), against ~0.67 nm of image width per nm of blur for width-type CDs
- [x] Prototype `artifacts/outline_fit/` (model.py, fit.py; NumPy/CuPy): detect -> alternate per-hole fits (x, y,
      brightness, level, edge offset against the full page model) and the shape fit on the neighbour-cleaned stack
      (superellipse + harmonics, two anisotropic Gaussians or Gaussian x pupil), staged (round -> superellipse ->
      harmonics) with a relaxed update (plain alternation swung +-2 nm between rounds: the two stages talk through
      the neighbours' cleaning and the holes' free levels). Small page: CD 79.7 (truth 80.0), slope 1.10, +0.07 nm
      per 10 % light, centers 0.47 nm (floor ~0.5)
- [x] Slope check (small page, template fixed): with the exact template the hole stage reads slope 1.008; with the
      two-Gaussian template fitted to the same hole 1.080 and -1.57 nm, so the slope excess and the offset on the test
      pages are the blur family, not the estimator. The first-order edge-offset model read a hole 6 nm off the outline
      1 nm too big (response curved up both ways); the second-order term (T0 + d T1 + d^2 T2) makes it exact to 0.01 nm
- [x] Hole stage made to hold on small test pages (2026-11-05): (1) all holes at once (Jacobi) ran away wherever a
      hole's size and brightness are soft against each other and neighbours overlap (dense 60 nm page, exact
      template: -7 then -10 nm pass after pass; half steps only slowed it) -> one colour class at a time with the
      page model refreshed in between (block Gauss-Seidel; holes within 1.5 spacings never share a colour), cached
      per-chunk light so a sweep costs the same; (2) light the model does not contain (the generator's holes cut by
      the frame are not seeded) made the ring beside it run off (edge offset at the clip, brightness 1.5 x) and the
      damage walked 3-4 pitches in (-23, -9, -3 nm) -> robust (Cauchy) pixel weights at 3 typical hole rms (now
      -4.5, -2.8, -2.4 against -1.9 inside; with the frame-cut holes seeded, all rings within 0.1 nm of the inside);
      (3) a 60 nm hole's own size has a standard error of ~60 radii -> per-hole size only when the median standard
      error (from the normal matrix) is under 0.3 radii, else every hole gets the page's outline size; edge-offset
      clip 0.3 radii (was a fixed 1 px); (4) brightness bounded (factor 2 per sweep, 5 x median: a frame-cut hole
      reached 319 x and turned the superellipse square), frame-cut holes keep the page size; (5) edge offsets
      reported as deviations from their mean over good holes (the common offset is a soft direction the sweeps leave
      wherever they stopped, -0.1 to -0.9 nm with the exact template). Small 80 nm page, 30 photons: slope 1.106,
      scatter 1.85 nm, +0.16 nm per 10 % light, centers 0.472 nm, converged in 11 rounds (174 s)
- [ ] CD scoring on the stress pages exactly like the report (slope, scatter in true nm, +10 % light) + page-mean
      error + a focus test (same mask rendered at blur 58 / 65 / 72 nm; existing methods for comparison); the same
      pages with the generator's own blur form (what the family costs). Queue: `runs/driver.py` (CPU only, two at a
      time: the machine's commit limit is nearly filled by other work; a CUDA context alone commits 2-3 GB). First
      results (before the hole-stage rework, copies in runs/*_v1.json): circle80 slope 1.080, scatter 1.86 nm,
      +0.24 nm per 10 % light, mean -0.85 (Gaussian fit 0.477 / 1.81 / +0.22 / +4.51); circle100 1.054 / 1.19 /
      +0.21 / -0.48; square150 1.004 / 0.53 / +0.21 / +0.27 (edge area 1.021 / 0.86 / -0.61 / +36.2).
      Generator's own blur form (lens family, reworked hole stage): circle80 1.065 / 1.89 / +0.25 / -0.36 (outline
      79.64, p 1.90, blur 64.2 for a true 65); circle100 1.033 / 1.22 / +0.13 / -0.24: the right family takes about
      half the slope excess and the offset away, the light move stays. Focus 58 nm (two Gaussians): circle80 1.078 /
      1.49 / +0.21 / -0.86, fitted blur 57.7 (Gaussian fit slope 0.508, mean +0.03)
- [ ] Where the rest of the slope and the light move come from. Measured so far (`runs/slope_template.log`, one
      noise-free hole, the page fit's window and weights): with the very templates the page fits landed on, an
      isolated hole reads slope 0.99-1.02 and a light move under 0.02 nm per 10 % (the exact template 1.00 and 0), so
      neither is the template nor the estimator; both appear only on the page (neighbours, their cleaning, or the
      noise: the slope part shrinks with light, the light move does not). The same isolated hole read with the 80 nm
      page's lens template comes out 1.6 nm bigger than the page reads its holes (81.6 vs 79.6): the page's average
      hole is not the isolated hole. Next: the page test with the exact template fixed, robust / variance weights on
      and off, three light levels (scratchpad `slope_diag.py`, log `runs/slope_diag.log`); it was stopped by the
      machine's memory guard (other work holds the memory) and is to be run when memory is free; also a tiny
      noise-free page through the whole fit (`tiny_page.py`): does the outline come back exact without noise
- [ ] Centers scored with pitch_signature_test.evaluate on the six report pages (columns, step, noise at 8, error)
- [ ] Real page 0 (and several pages): residual vs noise, fitted blur, CD distribution; page-mean CD across the hole
      pages against the Gaussian fit's
- [ ] HTML page (problem, idea, flow, each step with why, payoff, is it general, checks on own images); review here;
      proposal for the library (new methods beside the existing ones)

# Context cleanup (2026-11-04, done)

- [x] lessons.md cut from 268 lines of incident stories to ~70 lines of general rules; CLAUDE.md now limits new
      lessons to short, general rules
- [x] Dropped learned-template plan moved to the archive; Claude's memory trimmed, and stale or wrong notes fixed
