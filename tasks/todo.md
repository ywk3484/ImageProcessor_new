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
- [x] Outline fit in the learned-hole report's format (18 of 18 pages; the findings and numbers are in
      `tasks/archive/todo-2026-11.md`, moved 2026-11-12)
- [ ] Page 0 of the test file (and several pages; `stripe1_board1.tif` is a generated file, not a real page:
      found 2026-11-12): residual vs noise, fitted blur, CD distribution; page-mean CD across the hole
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

# Outline fit: speed and batch, first proposal (2026-11-12; superseded for the batch by the section below)

`docs/outline_fit_speed.md`: the existing fit's time split and steps that make its passes cheaper while keeping its
10-14 rounds (104 -> 10 s per page measured; prototype in `artifacts/outline_fit/speed/`). Not built. Steps 1-3 are
still what would make the existing `outline_fit` itself faster (workbench, small regions, CPU); the batch needs the
one-pass method below. Its "reference start" stays withdrawn. Its "real pages" are pages of the generated test file.

# Outline fit in one pass: the batch at reading speed (2026-11-12; phase 1 built 2026-11-13, waiting for real images)

User (2026-11-12): one TIFF = up to 1,500 pages, 2-4 GB, thousands of them; the pattern areas are picked by PCC against
a reference page cut into tiles (partial pages count); reading a TIFF takes about 1 minute and the fit must take less;
one page at a time only at hundreds of ms per page; some accuracy may be given up if it cannot be avoided.

**Handoff for the build: `docs/outline_fit_fast_handoff.md`.** Page for the user: `docs/outline_fit_one_pass.html`.
Notes (method, settings, measurements, findings): `docs/outline_fit_one_pass.md`. Prototype, tracked:
`artifacts/outline_fit/fast/` (`README.txt`; run results on this PC only). The proposal's finished items (method,
time, accuracy, checks, first review) are in `tasks/archive/todo-2026-11.md`.

Decision (user, 2026-11-12): build it beside `outline_fit`, in the folder reports first, page mean only, in a fresh
session. Answers: page mean for now (a size per hole only once the method is proven); the minute of reading is
measured on the remote; the remote has 64 CPU threads; the remote picks pattern areas by PCC of page tiles against
reference tiles, (N, m) valid indices.

- [x] Phase 1 (handoff section 5): library `fit_outline_fast` + `measure_cds` method `outline_fit_fast` (GPU, page
      values), `--outline-fit-fast` in `cd_report.py` and `cd_fast_report.py` with one comparison line and the
      blur's second part of both methods per image, tests, docs (2026-11-13; review below)
- [x] Give the user the one remote command (handoff 5.6)
- [x] First result on real images (user, 2026-11-13): odd; diagnosed in the section "outline_fit_fast on real
      images" below
- [ ] The user types back the line of `scripts\outline_fast_check.py` (and the comparison line of the report)
- [ ] With the result on real images: the pair reach (30 px?) and whether the fit should try both of its ends
      (handoff 6.3, 6.13)
- [ ] Phase 2 after the user's verdict: TIFF batch, areas (tiles or per-hole selection: handoff section 8)
- [ ] Ask: are the real test images PNGs in a folder; does a kept tile always hold only the reference's pattern

The reviews of 2026-11-12 (the prototype run the way the reports use it) and of the phase 1 build (2026-11-13)
are in `tasks/archive/todo-2026-11.md`.

# outline_fit_fast on real images: one colour per image, never converged (2026-11-13)

User, after the first run on real images (8 images of one area): the between-image 3 sigma is smaller than
`outline_fit`'s, but every image is one colour on the map (colour bar 97.13 to 98.89 nm), the colours differ from
image to image, and the final fit did not converge in 30 iterations on any image. "Diagnose what's happening and
explain why this happened, and what can be done about it."

- [x] One colour per image: by construction. Phase 1 gives one CD per image (the user's choice of 2026-11-12) and
      `measure_cds` hands it to every hole; the report's map and its shared colour range (2 to 98 % of the pooled
      values = lowest and highest image) then show exactly this. Not a fit of every hole to one value: no size per
      hole is measured.
- [x] The page CD moving 1.76 nm between images of one area, the cap hit on all 8: reproduced in kind on generated
      images (8 noisy images of the same holes, the blur changing by +-2 nm). Two causes, both in the final fit
      (`docs/outline_fit_one_pass.md` 6.18):
      - its start is wherever the first fit left the blur's second part, which is anywhere; from some starts it
        ends at another, worse-fitting outline (wide blur, gauss2: 2 of 8 images 0.9 nm low; 3 s over the images
        1.13 nm, 0.20 nm with the best of 10 starts);
      - its cap of 30 iterations: generated images need 3 to 40 iterations, some up to 99; one fit stood at
        96.57 nm when the cap cut it (its end: 99.46, and 100.02 after the exchange step).
      Which acts on the real images is not known (no real image here).
- [x] `scripts/outline_fast_check.py` (+ `tests/test_outline_fast_check.py`): per image the report's reading, the
      same start run to its end, the best of 10 starts (5 with lens); one line for the folder to type back. Its
      reading guide was tested on planted folders (start, harmless cap, a blur the model cannot follow, nothing).
      The library is unchanged.
- [x] The user ran the check on the 8 real images (2026-11-13):
      `check N=8 | rep 97.13-98.88 3s 2.01 nc 5 | end 3s 2.14 it<=56 | best 97.13-98.88 3s 2.14 moved 3 | ends 1-1 gap - | left 430.8-526.0 | rblur -0.21`
      Reading: not the start (one end per image from 10 starts), not the cap (the ends are as far apart, 3 s
      2.14 against 2.01 nm; three images changed when run on). The model does not fit the real average hole:
      leftover 431 to 526, a gap of 21 to 23 times the average's noise. My two causes from generated images do
      not act on the real images. User: "Before fixing anything, explain what exactly is going on based on
      this." Explained in chat; nothing built. The first explanation (the line field by field, gaps in noise
      units, width arithmetic) was not understood; explained again from what the method does.
- [x] User (2026-11-13): "Sounds like a datatype issue not accuracy. My remote images are 16 bit images for png
      and 8 bit for tiff. Is this the case? If it is, then you should have known since 430~526 is an impossible
      number to obtain with a 8 bit image." Answer, measured on generated images (`docs/outline_fit.md`, "What the
      leftover is"; notes 6.19):
      - the leftover is a ratio to the average hole's noise, without a unit (I had not said so): 484 = 22 times
        the noise, in 8 bit as in 16 bit;
      - bit depth as such: no effect (the same image in 8 bit, x 257 in 16 bit, finer steps, 12 bit values:
        100.07 nm four times, leftover 1.30 to 1.36);
      - rounding to whole grey steps on a quiet image: an effect (true noise 0.1 grey level: 8 bit 99.61 nm,
        leftover 86; 16 bit 100.03 nm, 6.9; none down to a noise of 0.5). For the 16 bit PNGs only if they hold
        8 bit content; open for the 8 bit TIFFs;
      - what I had overstated: the leftover says the gap is far above the noise, not that it is large. On
        generated images a gap of 0.05 grey levels on a hole 54 bright (0.1 %) goes with 1.2 nm of CD; the same
        fit on a quieter image has a higher leftover with the CD still right.
- [x] `scripts/image_values.py` (+ `tests/test_image_values.py`): how the files are stored (type, range, different
      grey values, grey step, values cut off) and the fit's noise, hole brightness and gap in grey steps; one line
      to type back. The library is unchanged.
- [ ] The user types back the line of `python scripts\image_values.py D:\data\png_images` (offered, not yet run)
- [ ] Not known, the next thing to measure if the user wants it: where the model misses the real average hole
      (centre, edge, towards the neighbours), whether holes are missing from the seed list, and the same number
      for `outline_fit` (same model, not measured)
- [ ] Generated test pages at a pitch that is not a whole number of pixels: at 210 nm = 10 px one bin in 16 of
      the average hole holds 92 % of the pixels, so the fit was never tested on an evenly sampled average hole
      except in the three runs of 6.19 (CD right there)
- [ ] On hold: the final fit from several starts, each run to its end (the check's `best`). It would not change
      the real images' spread; the cap still moves single images (three of eight) and the generated cases stand
- [ ] The user's decision: a size per hole now (the prototype's step D), or only after the page value holds
- [ ] Small images: the reports pass every seed (with the holes cut by the frame), the tests and notes 6.14 used
      the reported holes only; on the library's test page the two lists end 0.4 nm apart (100.99 against 100.58)

Review (2026-11-13). No library change, by choice: the cause on real images is not measured yet, and the remedy
changes the method's numbers and its time. Scratch runs (session scratchpad, not kept): the final fit recorded
iteration by iteration, 8-image series for 9 planted conditions, 27 and 10 starts per image, the same image five
times, the stop rule against stricter ones.
- What I got wrong before: phase 1 was checked on single pages and on the ends of the final fit from 81 starts
  round the pipeline's own end, never as repeated images of the same holes with the blur changing a little, and
  never with a blur wider than 72 nm. The cap of 30 was taken from the prototype without testing what the result
  is when it is hit.
- Not explained: why all 8 real images hit the cap (2 to 6 of 8 on generated ones); what the real leftover is.
