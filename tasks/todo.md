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
      grid background can run beside it later with `extra=bg=grid suffix=_grid`
- [ ] Real page 0 (and several pages): residual vs noise, fitted blur, CD distribution; page-mean CD across the hole
      pages against the Gaussian fit's
- [ ] HTML page (problem, idea, flow, each step with why, payoff, is it general, checks on own images); review here;
      proposal for the library (new methods beside the existing ones)

# Context cleanup (2026-11-04, done)

- [x] lessons.md cut from 268 lines of incident stories to ~70 lines of general rules; CLAUDE.md now limits new
      lessons to short, general rules
- [x] Dropped learned-template plan moved to the archive; Claude's memory trimmed, and stale or wrong notes fixed
