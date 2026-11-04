Finished tasks up to 2026-10-26 (both parts of the diagnostic) are in `tasks/archive/todo-2026-10.md`;
from November 2026 in `tasks/archive/todo-2026-11.md`.

# Dense arrays: one learned hole picture, all holes fitted together (2026-11-04, plan awaiting approval)

User: on the small dense pages of the pitch signature report no method shows the 1 nm row step within 8 columns.
Develop a method stable enough for accurate pitch variation that is also used for CD; judge pitch first, then CD.

Prototype (scratch code, copy in the git-ignored `artifacts/learned_hole_prototype/`, README there). Every hole is
the same small picture times its own brightness, at its own position, on its own local level, and all holes are
fitted together, so a neighbour's light is not read as a shift. The picture is learned from the page, from how the
holes differ in brightness (no blur, CD or pitch given; lattice and noise come from the image). A simulated copy of
the page (fitted holes + the page's own leftover noise) shows what the noise does to the learned picture; that is
subtracted.

Six small dense test pages (test noise 30 photons per grey level, read noise 0.65; columns = columns averaged
until the pitch noise is <= 1/3 nm; step = measured / planted 1 nm row step, true centers read 1.00):

| page | best existing (columns, step) | new: columns | new: step | hole noise nm, new / best | s |
|---|---|---|---|---|---|
| 50 nm circles, dense | 32, 1.19 (Gaussian) | 16 | 0.995 | 0.90 / 1.06 | 220 |
| 60 nm circles, dense | 16, 1.09 (Gaussian) | **8** | 1.010 | 0.65 / 0.72 | 212 |
| 70 x 45 ellipses, dense | 16, 1.15 (Gaussian) | 16 (0.335 nm at 8, needs 0.333) | 0.978 | 0.72 / 0.85 | 291 |
| 80 nm circles, square | 8, 1.01 (Gaussian) | 8 | 1.006 | 0.47 / 0.48 | 132 |
| 90 x 60 ellipses, stagger | 8, 1.03 (Gaussian) | 8 | 1.003 | 0.48 / 0.52 | 158 |
| 60 x 100 ellipses, square | 8, 1.00 (Gaussian) | 8 | 1.024 | 0.55 / 0.56 | 182 |

- "Best existing" is ranked like the report's best_of (columns, then method-alone noise at 1 column, then center
  error); an earlier version of this table listed half area min (1.13, 1.28 nm) for the 50 nm page by mistake.
- Report: `artifacts/learned_hole_prototype/report.html` (the pitch signature report with the prototype added, a
  summary on top, and two stress pages at 2 nm size spread; made by `make_report.py` in the same folder).
- Limit: given the exact hole picture, the 50 nm circles and 70 x 45 ellipses still need 16 columns at this noise
  (50 nm: 0.40 nm at 8 columns; about 1.5x more light would reach 8).
- Weak point: holes 4x more different in size than on the test pages (2 nm sd instead of 0.5) make most of the
  brightness differences come from size; the learned picture widens and the step reads 0.87 (50 nm) / 0.95
  (80 nm) in the report (learn7; learn8 gave 0.86 / 0.93), while the columns still drop (32 -> 16, 8 -> 4 against
  the Gaussian fit's 1.19 / 1.02). The true picture is fine there (1.011). A per-hole size term did not fix it (compact: wrong shape;
  full: absorbs the light outside the picture, step 0.93). The user's failed light-sum CD suggests flicker > size
  effect on real images, the good case.
- Noise correction: the page's own leftover noise includes model error when the picture is wrong (80 nm, 2 nm
  spread: 2.1x the true noise), so the library needs a noise model, not the leftovers.
- Assumes: flicker independent from hole to hole; each hole looks the same after a half turn.
- CD: no gain. Size on neighbour-cleaned holes (area above half of the own peak), true-CD slope / scatter / move per
  10 % pulse: 50 nm 0.13 / 17 nm / -3.1 nm (existing half area 0.24 / 15 / +1.5); 80 nm 0.46 / 3.5 / -1.1 vs
  Gaussian 0.48 / 1.8 / +0.2 and edge area 0.58 / 2.4 / +0.2. On 50 nm holes every method that looks better
  (Gaussian 0.70 / 3.0 / +1.6) reads brightness.
- Speed: single precision is 3.4x faster on this GPU (pool 2.3 -> 1.0 GB, centers equal to 1e-7 px).

Plan after approval:
- [ ] Library: `subpx/learned_hole.py` (public) + CPU / GPU backends (GPU single precision): seeds -> lattice ->
      rounds -> simulated noise correction (noise model) -> centers, brightness, learned picture, lattice, timings
- [ ] New center method next to the existing ones (own name): pitch signature test + report rerun, workbench Pitch
- [ ] CD: no CD method from it unless the user wants the cleaned-hole size anyway (opt-in, not in batch defaults)
- [ ] Tests on a small CPU page (step, lattice, noise correction sign), docs, lessons, commit + push

Round 2 (user: "barely see differences against the Gaussian fit; propose a better method; enlarge the page before
Otsu, just do it"). Scratch code and data in the same artifacts folder (README there):
- Floor (Cramer-Rao bound of the report's pitch noise; exact hole pictures given; x, y, brightness unknown; the
  page's own noise) at 8 columns: 50 nm 0.400, 60 nm 0.319, 70 x 45 0.338, 80 nm 0.240, 90 x 60 0.250, 60 x 100
  0.271 nm. Learned method 0.405 / 0.322 / 0.335 / 0.244 / 0.250 / 0.283: on the floor. The Gaussian fit is 7-34 %
  above it, all that any method could win. Brightness known exactly lowers the floor 2 %. So 8 columns on the
  50 nm and 70 x 45 dense pages is impossible at this light for every method.
- Enlarged first (bicubic, 8-bit, before Otsu; pixel settings scaled; x1 reproduces the report), x2 on all six
  pages: Gaussian and template fits unchanged; simple GPU finders much better on dense pages (GPU log-quadratic
  60 nm 304 -> 32 columns); several area methods worse on larger holes; nothing passes the Gaussian fit.
  x4 not run: stopped by Claude Code for low system memory (free RAM 3.8 of 31.6 GB afterwards).
- Two exposures averaged (every noise variance halved): floor 0.291 / 0.237 / 0.249 at 8 columns; Gaussian fit
  16 / 8 / 8 columns (0.389 / 0.270 / 0.320). Learned method not run (same memory stop).
- Report v2: `artifacts/learned_hole_prototype/report.html` (floor section on top, floor line in every noise
  chart, "Page enlarged x2 first" group, three two-exposure pages; `make_report2.py`); v1 kept as report_v1.html.
- Proposed: build the learned method as planned above; 8 columns on 50 nm needs more light; enlarging only for the
  simple GPU finders. Waiting on the go-ahead, and on whether to rerun x4 and the learned two-exposure pages.
- Explainer (user: "I still don't understand how the learned method works"):
  `artifacts/learned_hole_prototype/how_it_works.html` (`make_explainer.py`). Swap test on the 50 nm page, same
  fit and seeds, only the picture changed: bell (the Gaussian fit's median width) step 1.24, noise at 8 columns
  0.519, 32 columns; learned 1.00 / 0.405 / 16; exact picture 1.00 / 0.405 / 16. The gain is the shape alone.

# Workbench UI/UX rework (review 2026-10-31; decisions made, implementation in a new session)

Review, target layout and phases: `docs/superpowers/specs/2026-10-31-workbench-ux-review-design.md`. Baseline:
`scripts/workbench_timing.py` on the 39,042-hole page (click 0.9-2.0 s, full-view repaint 0.55 s; default preset:
gradient 94 % and halfmax 74 % failed). No library change; CD numbers unchanged. Goal now: an inspection UI for
testing interactively on sample images.

- [x] Decisions, baseline, P0 fixes, P1 speed, P2 new layout: done, with their reviews, in
      `tasks/archive/todo-2026-11.md`
- [ ] P3 later: open a cd_batch run (run.json settings, pages.csv page list, batch vs workbench mean)
