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
