Finished tasks up to 2026-10-26 (part 1 of the diagnostic) are in `tasks/archive/todo-2026-10.md`.

# Part 2 of the diagnostic: find the cure on the remote's pages in one run (2026-10-26, approved plan)

Plan file: `C:\Users\USER\.claude\plans\streamed-exploring-cray.md`. The user's part-1 lines (remote, 4x RTX 6000 Ada,
Windows Server 2022, commit e27d5):

```
v8ea8/e27d5/fc874 n0 pnpty wa-n0 u4w/4/0 c80/77/64 a378/541
x1.8/7/= f4.6 t4.1/3.9/4.2/3.5 m2.3/0.6/0.7/0.5 h20/52 d30/2/63/17/6 j4/12x11 y0/0/0/9
r11.1/46/3/17/39x19/0 e0/0/0 k0.17/0.64
```

Read: no CD bug (e0/0/0, x=); the PC waits (GPU busy 20 %, 378 us per GPU call vs 30 here, one core 1.6x slower,
Balanced plan); deblend never settles (30 rounds, 2 passes, last step 1.5e-3 raw, no tile switch, no cycle: a slow
monotone drift); the CDs depend on where it stops (1-3 more rounds: page mean up to 0.17 nm, hole p99 0.64 nm); the
slowest page refits from scratch after two seed re-checks (46 rounds, 3 passes, 11 s). Open: workers per GPU (a 2nd
worker did not slow the first here, the user's trials saw no gain) and which cure settles the drift.

Build (code by Opus 5.5 subagents, reviewed here; library untouched, today's numbers unchanged):

- [x] `scripts/deblend_variants.py`: the GPU deblend loop with switches (inner LM 100 steps, background frozen,
      free pixels frozen, Anderson acceleration with safeguards, 2 % wider start, continue after the seed re-check,
      and a 6th found on the way: the tile rule at 16 free pixels), a per-round trace, and `Patch` that puts it in
      place of `subpx._gpu.deblend.deblend`; all switches off = the library bit for bit (`tests/test_deblend_variants.py`)
- [x] `scripts/cd_diagnose.py --part 2` -> `scripts/cd_diagnose_part2.py`, sections 1-2: GPUs and Windows (round
      trip on every GPU, processes per GPU, Remote Desktop GPU policy, PCIe power saving, the High-performance test
      with the plan restored afterwards); workers per GPU (the earlier trial folders; the real cd_batch with 1-4
      workers per GPU on 160 pages, rate from the 25-75 % row arrivals, GPU / shared / committed memory and CPU
      sampled by one WMI loop, batch output to a file, leftover workers killed by process tree, a stall rule)
- [x] Section 3, deblend drift on the median and the slowest page: the reference = today's loop run on to a tight
      stop (step 1e-5, at most 200 rounds; also at most 100: still moving?), what moves in today's last rounds,
      today vs the reference, five cures at a practical stop (1e-4), the slowest page with continuing after the
      re-check alone and with Anderson, the page-to-page spread of pages.csv
- [x] A test page here that drifts like the remote's: `scripts/drift_page.py` (dense 8.5-px pitch, blurred disks,
      cut by the edges, noisy, 2-px margin). Cause there: no background tile ever has enough free pixels, so the map
      is "inferred" from every residual pixel and walks away from the true level (37 -> 33 of 40 grey levels) while
      the widths grow; the step shrinks 4 % per round and never reaches the stop rule. Background known (or any tile
      measured): settles in 5-7 rounds
- [x] Local test dataset again (the old 48-page file and speed_w1 run are gone): artifacts/cd_test, 48 pages of the
      user's size, 24 drifting + 16 settling + 8 without holes; one-worker trial 48 pages in 115 s
- [x] Tests: variants bit for bit (16), part-2 helpers (21 with part 1's and cd_batch's: 37); full suite = the same
      9 old failures
- [x] `--part 2` end to end here in PowerShell with the venv (one GPU, 1-4 workers on it), twice, 18 and 22 min;
      power plan back to Balanced, no worker left, temp folders gone
- [x] docs/remote_server.md: the part-2 command and what comes back; commit + push my files only

## Review (2026-10-26)

Built as planned, with three changes found on the way: (1) a reference that stops at today's rule (step 1e-3) is no
reference: on the drifting test page the plain loop, Anderson and a 2 % wider start all end at ONE point when run to a
tight stop (1e-5; the plain loop needs 242 rounds), while the 1e-3 run stops at round 36 still 1 nm from it; so the
reference uses the tight stop and the cures a practical one (1e-4: within 0.05 nm of the end), both switches of
`Variant`; (2) `deblend_iterations_used` counts the last pass only, so rounds are summed over the passes;
(3) six lines instead of five (the drift line was 155 characters) and the holes moved as a percentage.

What the local drifting page (39,042 holes, no background tile measured) says: today's 30 rounds stop 1.0 nm (cell
methods) to 1.3 nm (wide methods) from the end point, 71 % of the holes move more than 1 nm, 946 valid flags flip;
widths move 4x more than centres, the background falls 3.8 grey levels over the last pass. Cures at the 1e-4 stop:
Anderson 32 rounds and 0.01 nm from the end; the plain loop 110 rounds; LM 100 steps and tiles-from-16-px identical
to it (no tile reaches 16 free pixels here); the wider start 96 rounds to the same end; the frozen background 15
rounds but 3.25 nm away (ruled out). Slowest page: continuing + Anderson 33 rounds, 4.5 s, 0.00 nm from the end
(the re-check changed no seed here, so "continuing" alone is the plain loop: only the remote's 3-pass page tests it).

This PC's six lines (one RTX 4060 Ti 16 GB; 4 workers per GPU overflow its memory: 28 s per page, 2 pages failed,
in an earlier run a stall stopped by the 10-minute rule; section 3 came out the same in both runs):

```
v2e59/75323+4/75323 a26 p26 u-c e1 h28/27/52/51/3.5/3.6 wa
t1/0.35/3.4/96/1.5 n0
k0.34/3.5/5/0.3/41/23/12.5+0.39/6.4/9.3/0.6/46/32/11.1+0.37/10.4/12.1/0.9/50/49/9.4+0.03/28/15.5/3.6/54/45/5.2
y1/0 d200/2/1.3/0.07/0/-3.8 m31/125/143/127/123/0.17/96/100
c0.99/1.27/1.26/71/946 x110/0.05/0.6+15/3.25/20.58+32/0.01/0.4+96/0.04/0.59+110/0.05/0.6/0
r200/2/1.24/105/2/9.1/0.06/33/2/4.5/0 f2.3/3.05/40
```

Open until the remote's lines: whether their pages have measured tiles (then the drift may be another one), how far
their stop is from the end, which cure wins there; the `f` spread here mixes two kinds of synthetic page and means
nothing. Step 2 (the cure into the library, CPU + GPU, as default; a deblend version key in run.json; workers per
GPU and power-plan advice) waits for those lines and its own approval.

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
- [ ] P1 speed at 40k holes: click < 0.1 s, method switch / layer toggle < 0.2 s, full-view repaint < 0.05 s; the
      stack decoded on demand with a memory cap
- [ ] P2 new layout: session object, Setup steps, hole inspector, bottom tabs, page list next to the stack, Align view
- [ ] P3 later: open a cd_batch run (run.json settings, pages.csv page list, batch vs workbench mean)
