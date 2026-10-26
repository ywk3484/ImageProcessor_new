Finished tasks up to 2026-10-24 are in `tasks/archive/todo-2026-10.md`.

# Read speed on the remote; running a dataset with subfolders (2026-10-25)

The user asked whether tifffile.imread is not optimal, how to test the reader on the remote, and how to run a
dataset whose TIFFs sit in subfolders (cd_batch already searches folders recursively).

- [x] Re-measure the cause: tifffile reads uncompressed pages one at a time (maxworkers changes nothing); imread first
      walks every page header through the file (13 s cold for 1500 pages), then reads the pixels. "Parallel reads"
      (my earlier reason) was wrong, and the earlier 57 s did not reproduce
- [x] User-size file on the HDD (740 pages of 2048 x 1378, 2.09 GB, cold): plain read 13.4 / 14.4 s, batch reader
      16.7 s, tifffile.imread 20.5 s (1500 pages of 1196 x 1196: 15.1 / 15.3, 17.7, 28.6 s)
- [x] `scripts/read_speed.py`: the three reads of one TIFF, each after dropping the file from the OS cache
      (Windows: unbuffered open; Linux: posix_fadvise, checked in a container with mincore: 100 % -> 0 % resident);
      `tests/test_read_speed.py`
- [x] First run with an empty CuPy kernel cache: 32.8 s before the first page (121 kernels), then 1.7 s per page;
      the user's compose keeps the cache in the `cupy-cache` volume
- [x] docs/remote_server.md: read check, warm-up, trials on one whole file each into its own --output (a file given
      alone is keyed by its name, so the old "trial pages are kept" was wrong for subfolders), the whole dataset in
      one command, correct reading explanation

# The reference is one page of a TIFF (2026-10-26)

The user: the reference image is a single page in a TIFF, not a whole TIFF file. `--reference-page` already picks
it, but the commands in the docs never showed it, and the reference was first read by the workers, after run.json
was written: a wrong page number stopped the run, and the corrected command was then refused as "other settings".

- [x] Main process reads the reference page once, before anything is written; a page that cannot be read stops with
      a clear message; the workers get that image (no re-reading); the first line names the page used
- [x] Test: the reference is a page of one of the dataset's own TIFFs; a page past the end is refused and leaves no
      output; the corrected command then runs
- [x] docs/remote_server.md: commands with `--reference-page`, the one-page rule, a reference inside the dataset
- [x] Measure the read on the HDD: first read of the 740-page file's headers 4.9-5.2 s cold, then 0.1-0.3 s (the
      disk's own cache), so the per-worker reads cost seconds once per run, not more
- Result: a page past the end now stops with "--reference-page 740 of ...scan_user.tif: IndexError: page 740 is out
  of range (740 pages)" and leaves no output folder; with page 5 of the same HDD file as the reference, GPU workers:
  that page's error 0.0, the other pages measured as usual. Tests: 11 passed (test_cd_batch, test_read_speed).

# The remote is Windows Server 2022, not Docker (2026-10-26)

The user: the step-3 trials command gave a parse error; the remote runs Windows Server 2022 without Docker. The docs
were written for Linux and bash (I had taken the repo's Dockerfile / compose.yaml for the remote).

- [x] Cause: PowerShell cannot parse the bash loop (its parser: "Missing opening '(' after keyword 'for'"); the `\`
      line continuations and the lines starting with `--` fail too
- [x] docs/remote_server.md for PowerShell: one line per command, Windows paths, the three trials as three lines;
      Remote Desktop Disconnect instead of tmux / nohup; a click into the console pauses the run; no MPS or Docker
- [x] cd_batch: a results file held open by another program when the run ends (Excel) raised WinError 5 and left a
      .tmp file (reproduced); now it stays in the order measured, with a note
- [x] Checked: the docs' PowerShell blocks parse (0 errors); the three trial lines ran one after another in
      PowerShell on this PC (GPU workers, 1 / 2 / 3 per GPU); tests 12 passed (test_cd_batch, test_read_speed)

# Why the remote is slow (2026-10-26, investigation only: no code changes, as the user asked)

Remote trial: 209 pages (133 with holes) in 146 s on 4 GPUs, 3.5 s per hole page (deblend 2.04, tail_area 0.17),
cells 11 px wide; workers per GPU make no difference.

- [x] Here, both synthetic pages (mine, and one like the user's example: pitch 8, 43,095 holes) settle deblend in one
      pass of 6-7 rounds: 0.55 s. Forced to never settle (tolerance 0): 10 + 20 rounds, 1.6 s = the remote's 2.0 s
- [x] One page of my own 48-page test file never settles either (page 7: 1.67 s deblend, 2.8 s page)
- [x] GPU busy 59 % of a page here (nvidia-smi); a bigger GPU only speeds up the big kernels (tail_area 0.29 -> 0.17)
- [x] Blob page: 3.3 s vs 1.7 s; deblend passes 0.65 vs 0.50 s, the rest from full-size (51 x 49) cell stacks
- [x] User ran the rounds check: page 94 (29,638 holes, cells 12x11) 10 + 20 rounds, never settled, deblend 2.43 s
      of 4.45 s; page 46 (38,386 holes, cells 39x19) hole list changed twice, so 10 + 10 + 30 rounds from scratch,
      never settled, deblend 8.34 s of 11.09 s (each round 2.5x slower: bigger cells)
- [x] Cause (page 7 here, wrappers only): the background map's rule "a tile is measured if >= 10 % of its pixels
      are free, else interpolated from measured tiles". Only 13 % of tiles pass (median tile 7 % free). Tile
      (13,25) sits at 135/136 free px of the 136 needed: it switches every 4 rounds; each switch moves the
      background of 1,400 of 2,128 tiles (up to 1.6 grey levels), the fits follow, the tile switches back.
      p99 change per round repeats 0.0079 0.0049 0.0022 0.0011 forever (stop needs < 0.0006)
- [x] So the CDs depend on where the loop stops in the cycle: 10 and 30 rounds agree (20 rounds apart = 5
      cycles), 20 rounds differs (page means up to 0.36 nm, single holes p99 up to 9.3 nm). A repeatability bug,
      not only speed. Settling pages also switch tiles, but early (round 2), then settle in 7-8 rounds
- [ ] User runs the tile-switch check (PowerShell paste) on the same 5 trial pages, to confirm this on real data
- [ ] Then a plan for approval: a background map without the on/off switch (numbers change once, all methods),
      no restart from scratch after the hole re-check, then the remaining ~1.5 s per page from the remote's stages

# One diagnostic for the remote (2026-10-26, approved plan)

The user types everything from the remote by hand: one command, three short lines back, no more pastes.

- [x] scripts/cd_diagnose.py RUN_FOLDER TIFF [--pages A,B]: machine (code fingerprint, CUDA/CuPy settings, GPUs and
      driver model, other load, latency, FP64, CPU), real pages (alone vs every GPU busy vs the trial, GPU busy,
      stage groups, deblend rounds/passes/split, size groups, tile switches, slowest page), numbers (same page twice,
      stop 1-3 rounds later, GPU vs CPU on a crop); diagnosis.txt with everything
- [x] Run here under PowerShell as the user will; check against earlier separate measurements; robustness cases
- [x] tests/test_cd_diagnose.py: page picking, fingerprint ignores CRLF, token line with missing values
- [x] Commit + push; my own line recorded below as the reference

## Review

- Reference lines (this PC, RTX 4060 Ti, speed_w1 run of stripe1_board1.tif, pages 20 and 7, 56 s, `.venv` Python
  3.11; the commit field reads `8526d+1` before the commit, the new commit after it):
  `v8ea8/8526d+1/fc874 n0 p= wa-n0 u1w/1/1 c51/51/12 a30/401`
  `x1.6/7/= f1.9 t1.6/2.7/1.6/1.6 m0.5/0.3/0.6/0.2 h61/87 d7/1/56/35/5 j1/12x13 y2/0/0/5`
  `r2.9/30/2/2/13x13/1 e0/0/0 k0/0`
- Agrees with the earlier separate checks: page 20 settles in 7 rounds (1.6 s); page 7 runs 10 + 20 rounds (2 passes),
  2.9 s; GPU = CPU code on a crop (0); same page twice 0; a second worker on one GPU 1.6 -> 2.7 s (turns on WDDM).
- `--pages 7,12`: tile switches 6, cycle length 4, last steps 79/49/22/11 x1e-4, k 0.38/9.58 nm (the stop-round
  dependence found before). `--pages 2,12` (a page without holes forced in): 50 rounds in 3 passes, 36 size groups,
  35x35 cells, 4.2 s; its fresh worker's first page 8.1 s. Kernels are cached per source and options, never per cell
  size, so that is not per-size compiling; the f token and r's last field measure the first-page cost.
- Robustness: missing run.json / timings.csv rows / TIFF, bad --pages, page out of range: one sentence, no traceback.
- Letters inside values were changed so none looks like a digit (power plan b/s -> a/p, opencv o -> v, scipy s -> p,
  synthetic i/r -> d/n, "holes" -> d); the n token now names the slow settings that are on, the commit field counts
  the CD files that differ (+N). tests/test_cd_diagnose.py: 9 tests.
- The interpreter matters here: PowerShell's `python` on this PC is C:\Python313 without tifffile; the runs above use
  `.venv\Scripts\python.exe`. On the remote the user's own environment is the one the batch ran with.
- Waiting for the user's 3 lines. No fix before they are back.
