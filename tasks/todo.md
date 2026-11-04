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

# Workbench UI/UX rework (review 2026-10-31; decisions made, implementation in a new session)

Review, target layout and phases: `docs/superpowers/specs/2026-10-31-workbench-ux-review-design.md`. Baseline:
`scripts/workbench_timing.py` on the 39,042-hole page (click 0.9-2.0 s, full-view repaint 0.55 s; default preset:
gradient 94 % and halfmax 74 % failed). No library change; CD numbers unchanged. Goal now: an inspection UI for
testing interactively on sample images.

- [x] Decisions (user, 2026-10-31): pages counted from 0 like pages.csv; page navigation AND "Stack vertically" both
      kept (the stack made to handle many pages); opening a batch run later
- [x] Baseline: the user's pending workbench files committed unchanged (e99feb5, user's choice 2026-10-31)
- [x] P0 fixes in the current window (spec section 5)
  - [x] D1/F13 disabled styles + reason under Run; spin-box arrows
  - [x] F3/F4/H4/F12 one task name, no SI prefixes, arrow text, "Placement error" vs "Align"
  - [x] E1/E2/E10/B7 PageUp/Down scoped, arrow keys select, pan to off-screen hole, hole box tracking off
  - [x] C1 table clicks keep the table; A2/D4 failed counts per method, notes with a level
  - [x] D3/D8/D5 live checks + Run reasons, plain errors without the Activity jump, exception hook, hover field
  - [x] H1/H5 batch hole-check error in Align; moving image kept across pages, "use this file"
  - [x] G4 zoom, region, bottom tab, selected position (and locked contrast) kept across pages
  - [x] A1/A3/A6 Batch default preset pinned to cd_batch, calibrated preset dropped, compute Auto, area 3-200
  - [x] A4 QSettings (controls, window layout) + Reset to defaults; F1 display names for all 19 methods
  - [x] E4 no pre-selected hole; A5 pages from 0; E9 drop a file to open
  - [x] tests, timing on the 39k page, screenshots, docs/workbench.md, commit + push
  P0 review: the batch preset gives the batch's failures on the 39,042-hole page (projection 2,048, foot area
  471, foot area ratio 430, tail area norm 23, tail area 9; gaussian and the half areas 0); Measure CD 10.1 s
  in the window (10 methods, GPU); click 0.8-1.3 s, method switch 0.8 s, CD colors 1.9 s, repaint 0.53 s:
  unchanged, that is P1. Found on the way: tests must never use the real QSettings (they wrote area max 1
  into the registry; cleared, the fixture now uses a temp file); the summary above the image is two fixed
  rows so the zoom does not shift when its text changes. Not done in P0 (P2): C3 sizes, C11 pinned Run.
- [x] P1 speed at 40k holes: click < 0.1 s, method switch / layer toggle < 0.2 s, full-view repaint < 0.05 s; the
      stack decoded on demand with a memory cap
  - [x] B2 selection marker of its own; maps cached per result and method (`HoleLayers`)
  - [x] B1/B3 + F5 CD map and failed cells as rasters at every zoom, centers/contours/pairs for the holes in view
        only (at most 2,000); CD scale 1-99 %, draggable, Lock
  - [x] B4 + F2 NumPy table model: fixed column order, names in headers, columns as wide as their names, sort kept
  - [x] B6 results kept for the open page and the most recent others: at most 3 pages and 3 GB
  - [x] G1 stack: page sizes read once, file kept open, pages decoded in view (previews zoomed out, full pages
        zoomed in, 128 MB cap); F fits the page, Shift+F the stack
  - [x] tests, timing on the 39k page and the 48-page stack, screenshots, docs/workbench.md, commit + push
  P1 review: on the 39,042-hole page (ten methods, GPU) click 0.03-0.06 s (was 0.8-1.3), method switch 0.05-0.10 s
  (0.8), layer toggles 0.01-0.05 s (CD colors 1.9), repaint 0.004 s at full view (0.53) and 0.03 s zoomed in with
  1,551 holes drawn. Memory 1.9 -> 3.0 -> 4.2 -> 4.4 -> 4.3 GB over five pages (one page with ten methods holds
  1.25 GB, not the 0.47 GB measured with four, so the limit counts bytes as well as pages). Stack of 48 pages:
  sizes read in 0.1 s, view filled in 0.3-1.2 s between events, repaint 0.004 s, hole click 0.04 s, another page
  0.12 s, 5 MB of page images. Found on the way: Qt elides a two-line header badly (it cut the unit line), so
  columns are sized to their names; previews must be averaged like the active page or pages look different.
- [x] P2 new layout: session object, Setup steps, hole inspector, bottom tabs, page list next to the stack, Align view
  - [x] Package `subpx/workbench/` (names, settings, jobs, session, canvas, window, panels/*); entry points unchanged;
        `subpx/_workbench_data.py` stays where it is (Qt-free, used by examples and tests) instead of `data.py`
  - [x] Session: open file with the TIFF kept open, page, results per page {holes preview, CD, pitch} (shown = the
        latest of preview and CD; pitch never replaces it), selection, method, moving image, alignment; signals
  - [x] Docks: Setup (steps Image / Holes / CD methods / Pitch with marks, pinned Run, live checks; methods grouped
        with all / none and a Parameters fold; pitch Advanced fold) + Pages tab; Inspector; bottom tabs Holes /
        CD distribution / Pitch / Log; dialogs gone; Ctrl+1/2/3; layout remembered
  - [x] Stale banner (what changed, Re-run, Revert); job progress with elapsed time and Cancel (Esc, result
        discarded); errors as one line with Details
  - [x] Page list (from 0, thumbnails on demand, measured / failed marks) linked with Stack vertically
  - [x] Align view (flicker with Space, differences with a colorbar, checkerboard, blend; shift, RMSE, hole check,
        correlation peak with the runner-up); moving page from a list, kept across reference pages
  - [x] Pitch tab: method choice, plain summary, larger plots, a click selects the holes of that point; pitch uses
        the result on screen (no hidden re-measure; "Measure CD first")
  - [x] Menus, recent files, keys of spec 4.10 with keys in tooltips, table right-click / Ctrl+C copy, export
        dialog, ask before discarding results
  - [x] Tests moved to the new widgets; the 7 walk-throughs with screenshots; timing script; docs; commit + push
  P2 review: the 7 walk-throughs pass on the 48-page file (GPU, 1440 x 900, screenshots in
  artifacts/workbench_p2): page 5 from the page list, batch methods in 7.3 s, failures of every method in the status
  bar, N x10 through Projection's failures 0.04 s each with the table following; area max banner, Revert, Re-run;
  ten methods compared on one hole; a corner kept over five pages (0.02-0.07 s each); pitch from the shown CD in
  0.6 s (same arrays), a placement point picks its 162 holes; three moving pages aligned to page 5 with one click
  each (1.0 s), the moving page kept across reference pages; stack Shift+F / F, page 20 clicked, list follows.
  Timing as P1 (click 0.03-0.05 s, method switch 0.04-0.06 s, repaint 0.003 s, memory stops at 4.4 GB).
  Found on the way: an empty page counted as a kept result page and dropped another page's result; Shift+N
  extended the table selection (Ctrl+C then copied a range); the table did not tell J / N its order after a
  header sort; panels appearing beside the image rescaled it (the image now keeps center and zoom); page-list rows
  without a thumbnail did not line up; wide spin-box ranges widened Setup. Not done: stage progress inside a CD
  run (the library reports none), a typed region. 9 GPU / library tests (tiled GPU pipeline, pitch GPU, radial
  symmetry, isophote) fail, the same at 6aca236 without any pending edits: not from the workbench.
- [ ] P3 later: open a cd_batch run (run.json settings, pages.csv page list, batch vs workbench mean)
