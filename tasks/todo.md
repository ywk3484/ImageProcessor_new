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

# Outline fit (`outline_fit`): what is still open (2026-11-04 task; record moved to the archive 2026-11-13)

The method is built and in the reports. The task's record with every number, and the first speed proposal
(`docs/outline_fit_speed.md`, not built, superseded for the batch), are in `tasks/archive/todo-2026-11.md`
("moved 2026-11-13"). Open items, each with its full record there:

- [ ] CD scoring on the stress pages exactly like the report (slope, scatter, +10 % light, focus series): the
      pages of the record are done; nothing is queued
- [ ] The whole fit on the exact noise-free page (slope 1.022, +0.10 nm per 10 % light): the background solved
      jointly with the holes' common size and brightness shift is the next step, not built
- [ ] Centers on the six report pages: dense 50 nm pages and elongated holes fail by the size-blur trade; a blur
      prior from a round-hole page is the remedy, not built
- [ ] Several real pages: leftover against noise, fitted blur, CD distribution (the local test file is generated)
- [ ] The method page (problem, idea, flow, steps, payoff) and the proposal for the library: reviewed three
      times, last rewrite 2026-11-09

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
- [x] The user typed back the line of `scripts\outline_fast_check.py` (2026-11-13, section below)
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
- [x] `scripts\image_values.py` was offered and rejected, with every further typed-back check (user,
      2026-11-13): "you're just suggesting me to become a test bed and type down all the things you just come up
      with every single time ... implement a visual debugging UI that would visualize every single step of the
      calculation, and allow me to instantly spot what step is looking odd." Do not offer it again; the section
      "Step-by-step visual debugger" below is the way forward
- [ ] Not known until the user has looked at the steps: where the model misses the real average hole (centre,
      edge, towards the neighbours), whether holes are missing from the seed list, and the same for `outline_fit`
      (same model)
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

# Step-by-step visual debugger for outline_fit_fast (2026-11-13; plan approved by the user)

The cause of the odd results on the real images is not known and I cannot see them. The user asked for a visual
debugging UI of every step and chose its form: a page in the browser written by one command on the image folder
(`python scripts\outline_fit_steps.py FOLDER`). Every step is shown for the image and, beside it, for a copy of
the image redrawn from the fit's own answer and run through the same steps; a second view shows all images of the
folder. Nothing in the method is fixed or changed. Plan: `C:\Users\USER\.claude\plans\effervescent-gathering-ember.md`.
Not in it: the slow `outline_fit`, a workbench window, any fix.

- [x] 1 Recording in the library: `fit_images(trace=)`, `GpuPage` trace points, `ShapeFitter.fit(trace=)` per
      iteration and per page, `grid_start`'s table, best match and hole alone on the bin grid,
      `fit_outline_fast(trace=, redraw_seed=)`; `tests/test_outline_fit_trace.py` (8). Found on the way: the fit is
      not the same from run to run even without recording (the device adds in another order: CD 3e-4 nm
      apart on a 4,400-hole image, 22 to 30 iterations of the final fit for the same image), so "unchanged"
      is tested within that width; under NumPy the page fit is bit-identical with and without it
- [ ] 2 `scripts/outline_fit_steps.py` + `scripts/outline_fit_steps.html`: the 10 steps for one image (your-image
      column), data reduced to a few MB per image, offline from file://
- [ ] 3 The redrawn copy (final outline and blur, every hole refitted, near and far light + level of the nearest
      hole + noise, rounded like the file), the second column, "your image minus copy"
- [ ] 4 All-images view (`data/all.js`): the steps' numbers in one table, one row of small pictures per step
- [ ] 5 Planted-fault folders (nothing; a blur form the model cannot describe; tops cut off; two hole sizes; blur
      changing across the image; 5 % of the holes not found; coarse grey steps on a quiet image; 8 repeats with the
      blur changing): each must show at its step; screenshots looked at; `docs/outline_fit_steps.md` with the
      gallery; pointer in `docs/outline_fit.md`
- [ ] 6 Tests (`tests/test_outline_fit_trace.py`, `tests/test_outline_fit_steps.py`), whole suite, time per image
      at 2048 x 1378 and 2048 x 5408, commit and push after each stage

State 2026-11-13 (session cut off by the usage limit; continue from here):
- Done and pushed: items 1 to 4 (recording, script, page, redrawn copy, all-images view) and the tests of item 6
  (tests/test_outline_fit_trace.py 9, tests/test_outline_fit_steps.py 12; 54 outline tests pass). Added beyond the plan:
  the copy takes its noise from the pixel-to-pixel jitter of image minus picture; picture 8e (average hole of each
  ninth of the image against the whole image); graph 9e (difference averaged round the centre).
- Looked at in a browser on generated 600 x 800 folders: nothing planted (image and copy alike), blur form (9e wave;
  CD 1.5 nm off at 1.13 x the noise), tops cut off (step 1, 4a, 9d), two sizes and blur changing across the image
  (8e, 7e), holes not found (2a, 7a, 7e), coarse grey steps (image and copy both 4.6 to 4.9 x), 8 repeats (A4).
- Time: 3.3 s per 1378 x 2048 image, 8 s per 2048 x 5408 image (104,000 holes) with its copy, RTX 4060 Ti; 4 to 5 MB of
  page data per image.
- LEFT: the full-size page not yet looked at in a browser; docs/outline_fit_steps.md with the gallery and the pointer
  in docs/outline_fit.md (item 5); the whole test suite; notes 6.20 in docs/outline_fit_one_pass.md (the fit is not
  the same from run to run: 22 to 30 iterations and up to 0.04 nm for one image; a perfect copy can hit the limit of
  30; the lens form reads its own copies 0.1 to 0.3 nm low; holes cut by the frame are placed poorly); the memory
  note; giving the user the command.
