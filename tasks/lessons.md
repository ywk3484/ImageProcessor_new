# Lessons

- **Committing:** commit directly on `main` when asked; no feature branch. Stage only the files
  you changed, by explicit path: the working tree usually holds the user's own uncommitted work,
  which must stay out of your commit.
- **Keep your edits out of the user's pending files.** Appending to a file the user has not
  committed yet (e.g. an untracked doc) entangles your work with theirs at commit time. Put new
  content in files that carry no pending changes of theirs.
- **A nuisance the user names is a hard constraint.** When asked for a method that is insensitive
  to something (e.g. per-hole source-intensity fluctuation the reference cannot remove), every
  candidate must be invariant to it; do not propose one that needs the nuisance removed by a
  reference. Ask what has already been tried before proposing a method.
- **Explain in the user's terms.** Build the argument step by step in plain language, with the
  physical picture first. Define every term, avoid statistics/optics jargon (Cramér–Rao bound, SNR,
  slope, d/PSF), and give only the few numbers that matter, in nm with an example, not tables of
  percentages.
- **Every number shown must be explained before it is used.** Before showing a table or example,
  say what each row, column and value is (e.g. "brightness counts of 9 pixels in a line through the
  middle of one spot"), where it comes from (simulated, which assumptions, or measured), and only
  then what to look at. Build examples on the user's real sizes, not on made-up ones.
- **Ask which population a spread statistic is over.** "3 sigma" for CD can mean hole-to-hole
  inside one image or image-to-image of the image means (repeatability). The user meant between
  images; I built within-image. When a statistic could be over more than one population, state the
  equation in the output and confirm which one is wanted.
- **Never calibrate each image to the same target when images are compared.** Matching every
  image's median to a nominal CD forces their means equal and fakes perfect repeatability. Calibrate
  once and reuse the calibration for every image in a comparison.
- **A smaller spread is not better precision unless the method follows the true CD at slope 1.**
  A method that moves 0.5 nm per 1 nm of real change also halves the real image-to-image
  differences. Say so whenever methods are ranked by their spread.
- **Test a diagnostic on planted causes before shipping its reading guide.** My first band-profile guide
  ("CD follows brightness → light leak") was wrong: real CD changes also change brightness. Plant each
  candidate cause in a synthetic image and check the guide names the right one.
- **Keep display data at the precision the method used where the method is sensitive.** 8-bit cells were
  fine to look at but moved steepest-slope edges by up to 1.5 px; store what the method computed on exactly.
- **Console output must survive non-UTF-8 consoles** (the user's Windows uses cp949): print ASCII only.
- **Never assume the pattern shape.** The user's patterns can be any shape, unknown in advance. I proposed a
  round-hole profile model, then tested on holes and squares only; both were assumptions the user never made.
  Ask what shapes occur before using a shape model; otherwise use shape-free measures and test them on
  mixed, irregular, rotated shapes the method knows nothing about.
- **Do not build new methods on a method the user found failing on real data.** The disk fit (and anything
  that needs `--psf-fwhm` / `--nominal-cd`) did not give good range or repeatability on the user's images;
  re-proposing it in a new form was rejected twice.
- **Check that the suspected cause is the real error source before replacing it.** "No interpolation" was
  the user's guess; the spline was there. A better interpolation did not help; the width definition (2
  points per edge) and noise did. Test the replacement before proposing it.
- **Report speed matters on the user's machine.** Time every stage on a full-size image and drop or move to
  the GPU what dominates before adding more.
- **Never state a cause in a result summary that was not measured.** "Neighbours' light gets in" at a low
  threshold was a guess; the measured cause was the feature's own tail being wider than its cell (Voronoi
  cell clipped to box + pad) and the edge-based local level holding that tail. Diagnose (e.g. render each
  feature alone, noiseless) before explaining a failure rate.
- **Test on harder synthetic patterns before shipping a method.** The wide tail_area passed on 100 nm round-ish
  features but failed on the user's patterns; bigger or square features (Gaussian neighbour models misfit) made
  almost all fail. Run the stress set (bigger, squares, more blur) and check the share measured, not just error.
- **Test array layouts, not only square grids, and use the user's own example image.** tail_area took its level
  from "own territory beyond 0.6 pitch"; on a staggered array the territory ends at ~0.56 pitch, so the level was
  empty and almost every hole got no value. examples/data/synthetic_ch (staggered) reproduces it in seconds.
  Define per-hole regions relative to the hole's own territory, never by a fixed fraction of the pitch.
- **Measure the user's suspected noise source before building around it.** "The peak makes it noisy" was
  plausible but the oracle test (true peak instead of the estimate) showed the peak causes ~5 % of the noise.
  Run an oracle test first and report it with the new method.
- **Build what was asked, at the place asked.** The user asked for a peak-free TAIL boundary; I shipped a
  steepest-slope level (~0.4 of the peak, what projection reads) and called it done. Check the delivered level
  against the request (here: cut/peak must be near the background) before reporting.
- **Never replace or rename a method the user already has.** Add new methods next to existing ones; the user
  compares them on real data.
- **A measurement tool measures; do not add statistics the user did not ask for.** For the batch run I proposed a
  between-page 3 sigma per TIFF; the user computes statistics afterwards, and repeatability ("between-image") is
  between 3 datasets of the same area, while hundreds of TIFFs are ONE dataset. Ask what one file / one dataset is
  before grouping anything.
- **Ask questions in the user's terms.** "Which pages should the between-image 3 sigma compare?" was not understood.
  Say what the choice changes in their output, with their own objects (dataset, TIFF, page), not my internal plan.
- **When raising a defect, answer what the user will ask next.** "Do you want me to fix it?" about the cell-size
  coupling made the user ask whether it is a bug and how exactly I would fix it. Say: bug or not, the measured cause
  (isolated by a test that changes only that one thing), how often it triggers on their data, the exact fix with a
  prototype result, and how the fix changes their numbers.
- **For a batch tool, give the whole-dataset time up front.** Explaining the per-page design did not answer "is
  this realistic on my dataset?". Give pages x seconds per page / GPUs, the assumptions behind it, and what the
  trial run measures.
- **Measure I/O under the user's conditions.** "Opening a 1500-page TIFF takes 0.03 s" came from a 64x64-px file
  already in RAM; the user's TIFFs are ~2 GB on an HDD (~1 min to load). Cold on this PC's HDD (evict the file:
  open it unbuffered), counting the pages alone took 14 s and the batch's 8-worker reads 56 s vs 19 s sequential.
  Use the real file size, the real disk and a cold cache, or say the number is unknown.
- **Ask for the direct quantity; do not derive it from two rough totals.** From "~2 GB, up to 1500 pages" I derived
  1.4 MB per page and guessed the page size; the real page is 2048x1378 8-bit (2.8 MB), so a 2 GB file holds ~700
  pages. Before estimating a run, ask page size (px), bit depth, compression, pages per file, files per dataset and how
  many pages hold holes, then measure one page of that size.
- **"The GPU is busy" is not "the GPU is used well".** Two workers on one GPU giving no speed-up only showed
  contention (without MPS, processes take turns and evict each other's cache). Before claiming a limit, compare
  the work with the hardware: bytes moved vs data size (one page created 157 GB of temporaries for 2.7 MB of
  pixels), double-precision use (1/64 rate on Ada), GPU idle time (29 %), number of launches (~16,000 per page).
- **A test threshold tuned on a buggy implementation encodes the bug.** After the grid fix, unit tests failed whose
  limits had been met by the old code only at the one cell size the test used (edge_area's bright-top shift: 1.8 % in
  a 31-px cell, 4.2 % in a 41-px cell, old code). Before changing a test, recompute its quantity with the old code at
  another array size or mask shape; report the change, do not just relax the limit.
- **Profile before trusting a "the GPU is the limit" picture, and measure the worker counts the target will run.**
  After the kernel work, half of each page was CPU-side (many small steps), and a second worker gave no gain on this
  Windows GPU. Give the remote estimate as a range with the deciding unknown named (workers per GPU on Linux, MPS),
  and the trial run that measures it.
- **The library call is not the disk, and read the library before naming why it is slow.** tifffile.imread read a
  2 GB HDD file in 29-33 s against 16 s for a plain read. I told the user "it reads pages in parallel, so the head
  jumps", unmeasured; tifffile reads uncompressed pages one at a time (so maxworkers changed nothing, as the user had
  seen). Measured: it first walks every page header through the whole file (13 s cold, one seek per page), then reads
  the pixels. Time a plain read, then each step of the library call, before giving a cause.

- **Show every option the user's own case needs in the example commands, and check inputs before saving state.**
  The docs' commands had `--reference /data/reference.tif` without `--reference-page` (it was only in the table), so
  the user read the reference as a whole TIFF; it is one page of one. And the reference page was first read by the
  workers, after run.json was saved: a wrong page number stopped the run and the corrected command was then refused
  as "other settings". Read and check every input before writing anything a later run is compared with.
- **Never infer the target machine from files in the repo; write commands for the shell it really has.** I took the
  repo's Dockerfile / compose.yaml for the remote and wrote bash loops, `\` continuations, tmux / nohup, MPS and
  Docker notes; the remote is Windows Server 2022 without Docker, and the first loop the user pasted failed to parse
  in PowerShell. Ask for (or state) the target OS and shell before writing commands for another machine, and give
  copy-paste instructions as one plain command per line, without loops or line continuations.
- **Estimate speed from pages that behave like real data, and explain every outlier before reporting.** I told the
  user 4-16 pages/s for 4 RTX 6000 Ada from GPU specs and clean synthetic pages; the remote measured 3.5 s per hole
  page, 2.0 s of it deblend. Clean synthetic pages meet deblend's stop rule (99 % of holes move < 0.001 px) in 6-7
  rounds; real pages likely never do and run all 30 (forced here: 1.6 s). My own test file had such a page (2.7 s)
  and I reported "slowest page 2.7 s" without asking why. Also the GPU is busy only ~60 % of a page, mostly with small
  kernels a bigger GPU does not speed up. Before any estimate: measure the GPU's busy share, time a page that never
  settles, and find the cause of each slow page.
- **Never call a partial fix "fixed".** The blob page went from 15 s to 3.3 s (normal page 1.7 s): the numbers were
  fixed, the time only partly (full-size cell stacks remain). Say what is left, with its cost.
- **The user types every line from the remote by hand: give one command and a short result, never a paste.** I
  explained the slowness one hypothesis at a time and each time asked the user to paste a long PowerShell script and
  type back its output; they called it using them as a testbed and stopped trusting the diagnosis. For another
  machine, build one script that measures every candidate at once (code version, environment, machine, GPU sharing,
  the real pages' stage split and deblend behaviour, the numbers), verify it here first, and make it print a few
  short lines of labelled tokens in which no letter looks like a digit. Ask for those lines only.
- **Count the characters before asking for typed output, and say what the goal is before any method.** Part 2
  printed six lines of about 400 characters of letter tokens; the user, who types them by hand, asked whether I
  really thought that was typable and what I was even trying to do. Hand-typed output must be the few numbers that
  decide the next step (about 150 characters at most), each with its meaning in plain words next to it, with the
  full record left in a file for later questions. And every report starts with the goal in one plain sentence.
- **A reference that uses the stop rule under test is no reference.** "Today's loop allowed 200 rounds" stopped at
  round 36 by the same 1e-3 rule, 1 nm short of where the loop ends; cures compared with it looked like different
  fixed points. Run the reference to a stop the drift cannot meet by chance (here 1e-5; the plain loop, Anderson and
  a wider start then agree within 0.01 nm) and judge each cure by its distance from that end point.
