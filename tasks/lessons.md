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
