Finished tasks up to 2026-10-26 (both parts of the diagnostic) are in `tasks/archive/todo-2026-10.md`;
from November 2026 in `tasks/archive/todo-2026-11.md`.

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

# Preprocessing from the ACTIS proposal: noise reduction and resolution "Do"s (2026-11-01)

Goal: try the proposal's recommended preprocessing on real pages in the workbench and measure CD / pitch with and
without it. Source: docs/actis_preprocessing_proposal.html, sections "Noise reduction" and "Resolution improvement".

Scope. Recommended there: fixed light smoothing (Gaussian 0.5-0.7 px) and the mean-hole method ("my order"), the
super-resolved average hole ("Do"). Borderline: Wiener filter tuned by Noise2Self (all "Yes" in the table, not in the
order): built (user, 2026-11-01); blur matching ("Do, if Step 0 shows blur differences matter"): not now. Not built (proposal:
skip / avoid / later): deconvolution (the only sharpening filter, "Skip for CD"), BM3D / NLM, AI denoisers, AI
super-resolution, merging datasets.

Probe, page 0 of artifacts/cd_test/data/stripe1_board1.tif (39,042 holes, spacing 8.2 px), CPU: Gaussian 0.09 s;
mean-hole boxes + PCA 0.3 s, number of components by Noise2Self 3.4 s (best 15 of 60); hole finding 2.8 s. Noise
left after 15 components ~7.3 grey levels; neighbouring pixels' noise nearly independent (lag-1 correlation -0.05),
which Noise2Self assumes. No GPU needed for one page.

- [x] Library `subpx/preprocess.py` (NumPy only, any 2-D image, dataclass results with meta): `smooth_gaussian`;
      `denoise_wiener` (Noise2Self on the filter with each pixel's own weight moved to its 8 neighbours, exact in
      Fourier space); `denoise_mean_hole` (K by leave-one-pixel-out); `average_hole` (each hole's brightness fitted
      against the average smoothed by one bin). `PreprocessResult`, `AverageHoleResult` in types.py; exported from
      `subpx`. 9 tests in tests/test_preprocess.py
- [x] Workbench
  - [x] Setup step "2 Preprocess": Gaussian / Wiener / Mean hole with their settings (0 = auto), a help line with the
        synthetic result, a line on what the run found. No "None" entry: the Preprocessed switch is the on/off
  - [x] "Preprocessed" switch (P, View menu). PageResults holds raw and preprocessed results and the preprocessed
        page; runs measure the version on screen (a missing or out-of-date preprocessed page is made first, same
        job); results store the Preprocess settings (None for raw), so only the preprocessed version goes stale.
        Turning it on, or showing another page while it is on, preprocesses that page; flipping keeps the contrast
  - [x] Show > "Removed by preprocessing": raw - preprocessed, blue-black-red, levels symmetric about 0
  - [x] Other version: summary line, CD distribution outline with mean and sd, inspector column "CD raw" / "CD pre"
  - [x] Bottom tab "Average hole": the current method's centres, raw and preprocessed, x / y profiles, width at half
        height
  - [x] Hover shows both values; export writes preprocessed_image.npy, preprocess_meta.npz and the settings in
        source.json; the stack's other pages and Align stay raw (Help text, image hint)
  - [x] Tests (2 workbench, 2 data), timing on page 0, screenshots, docs/preprocess.md, docs/workbench.md,
        scripts/preprocess_check.py

## Review (2026-11-01)

- Tests: 56 pass (workbench 33, data 14, preprocess 9). Full suite: 517 pass; the same 9 GPU tests fail as before.
- Page 0, CPU, whole page: Gaussian 0.1-0.2 s; Wiener 2.1-2.3 s (noise 7.3, auto); mean hole auto 5.0-5.4 s
  (K = 2, includes finding the holes); K = 32 2.2 s; average hole of 1,610 holes 0.4 s.
- Synthetic truth (scripts/preprocess_check.py, half area, 3 layouts), a +5 % step measured: raw +4.7 to +4.9 %,
  Gaussian 0.6 +3.8 to +4.4 %, Wiener auto +4.2 to +4.9 % (but hole-to-hole slope 0.50-0.54 vs raw 0.85-0.88),
  mean hole auto K +1.0 to +2.2 %, K = 32 +4.3 to +4.6 %.
- Real region (1,610 holes): the automatic settings cut the CD sd 1.5-7 times, but each hole's CD follows its raw CD
  less (r 0.85 down to 0.08); K = 32 keeps r 0.94 (half area) with an sd close to raw. Without truth, only repeated
  exposures of the same holes can tell noise from real differences.
- Found while testing on the real page: the average hole's width at half height took the profile ends (the
  neighbours' peaks) as its base, giving 0.42 px; it now uses the profile minimum (4.28 px), with a test. The mean
  hole on a non-hole page (page 3) still runs and picks K at its limit; the summary now says so.
