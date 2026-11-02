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

# Pitch / placement error from the old GPU center finders, beside the CD methods (2026-11-02)

Goal: the user doubts the placement error from the CD methods' centers; the GPU center finders (March 2026) gave
registration they trusted. Measure pitch also finds centers with the ticked GPU finders on the same holes, so the
Pitch method box lists them next to the CD methods. Plan: ~/.claude/plans/swirling-stirring-sketch.md.

- [x] `_workbench_data`: `GPU_CENTER_FINDERS`, `gpu_center_measurements` (detect on the GPU, nearest hole within half
      the spacing), `analyze_patterns(..., extra)`
- [x] Settings key `pitch_finders` (stale banner); Pitch step checkboxes (log-quadratic, edge moment, edge erf on)
- [x] Session pitch run adds the finders and logs one line each; names; Pitch summary "centers for N of M holes"
- [x] Tests (data + window), full suite, real page timing + screenshot, docs/workbench.md, commit + push

## Review (2026-11-02)

- Page 0 of stripe1_board1.tif (39,042 holes, 8.5 px apart, GPU), Measure pitch: 0.58 s without finders, 2.83 s
  with the 3 defaults, 8.38 s with all 6. Holes with a center / x and y placement range (px): log-quadratic 39,042,
  -0.041..+0.071 / -0.047..+0.035; weighted 39,042; radial symmetry 39,042, -0.046..+0.061 / -0.054..+0.020;
  isophote curvature 39,042, -0.134..+0.115 / -0.209..+0.082; edge moment 19,769 and edge erf 19 (made for large
  features; their placement, up to 3.5 px, means nothing here). CD methods for comparison: Gaussian fit
  -0.018..+0.051 / -0.054..+0.034, Half area -0.024..+0.057 / -0.037..+0.023, Foot area -0.292..0 in x.
- Synthetic example (truth known): GPU log-quadratic centers 0.024 px rms, radial symmetry 0.023, CD Gaussian 0.020.
- Screenshot: artifacts/pitch_gpu_finders/pitch_gpu_logquad_placement.png. Tests: 2 data + 1 window added.

# Pitch plots: plain lines, no markers or error bars (2026-11-02)

User: the dots and vertical lines in the Pitch plots hide the data. The vertical lines were ±1σ error bars (single
pair spread ~0.24 px on the 39k page) that stretched the y-axis ~15x over the per-interval means (range 8.46-8.53
px); the dots were the measured means themselves; placement had a dashed vertical line at the reference node.

- [x] Both views: a plain line through the measured values; no markers, error bars or vertical reference line;
      a value with no measured neighbour gets a dot (a line cannot show it); spread moved to the summary text
- [x] Tests (plot items, lone-value dot), real-page render, docs/workbench.md, commit + push

## Review (2026-11-02)

- Page 0 of stripe1_board1.tif, Gaussian fit: the x pitch plot now spans 8.46-8.53 px instead of 8.2-8.8. The
  interval-to-interval wiggle (sd 0.017 px) is about what noise alone gives (0.24 px / sqrt(162 pairs) = 0.019 px).
