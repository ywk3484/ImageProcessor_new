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

# Workbench: measure all pages of a multipage TIFF (2026-11-02)

User: the workbench cannot process all pages of a multipage TIFF at once; add it, with GPU acceleration and batch
processing. User's choices: per page the means plus one CSV of hole CDs (no full results; Measure CD on a page to
inspect it); skip pages without holes by the batch's hole check against the page on screen; CD and pitch.
Plan: ~/.claude/plans/zany-enchanting-cerf.md. Measured first (RTX 4060 Ti, 39,042-hole page): the batch's fast mode
(summary_only) 3.0 s/page vs 7.3 s in the window, same CDs bit for bit; two pages in flight on one GPU 2.79 vs 3.31 s.

- [x] Engine `subpx/_workbench_pages.py`: reader thread, one worker process per GPU with two pages in flight, hole
      check, preprocess / region, summary-mode CD, pitch + GPU finders, pages.csv (cd_batch columns) + holes CSVs +
      run.json, failed / hung workers, stop
- [x] Session: run / stop, progress, page marks, runs and open/close refused while it runs
- [x] Window: Run menu + Pages tab button, dialog, status-bar progress with Stop, All pages tab (table + plot)
- [x] Tests (engine + window), timing on the 48-page file, screenshot, docs/workbench.md, commit + push

## Review (2026-11-02)

- stripe1_board1.tif, 48 pages (40 with 39,042 holes, 8 without: 3, 11, 19, 27, 35, 43, 46, 47), ten batch methods,
  RTX 4060 Ti: 132 s without GPU finders (2.6 s/page running), 209 s with the 3 default finders (4.1 s/page); Measure CD
  page by page 8.1-8.8 s/page (~5.5 min for the 40). Pages 5 and 20: Measure CD's mean CDs equal the run's (0 diff).
- One page alone: hole check 0.17 s, CD 3.25 s, pitch 0.5 s, 3 finders +2.2 s, hole CSV 0.5 s (was 1.0 s with
  np.char.mod; Python lists halve it).
- Found on the way: starting a worker froze the window 1.6 s (Windows writes process arguments, here the 2.8 MB
  reference page, into the child's start-up pipe and waits until the child has imported its modules); the settings now
  go through the worker's queue (start 0.10 s). Waiting for the workers to exit at the end (0.4 s) moved off the
  window's thread. Longest window pause during a run now 0.05 s.
- Not done: reopening an earlier run's folder (P3), resume of a stopped run, more than one GPU measured (one here).
- Tests: tests/test_workbench_pages.py (4: rows equal cd_batch page_values and the window's Measure CD / pitch, blank
  page skipped, region + preprocessing, failed page, stop, GPU with a finder); tests/test_workbench.py (2: the run
  beside the window, Esc stops). Workbench suite 57 passed.
