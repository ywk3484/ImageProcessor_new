# Lessons

- **Committing:** commit directly on `main` when asked; no feature branch. Stage only the files
  you changed, by explicit path: the working tree usually holds the user's own uncommitted work,
  which must stay out of your commit.
- **Keep your edits out of the user's pending files.** Appending to a file the user has not
  committed yet (e.g. an untracked doc) entangles your work with theirs at commit time. Put new
  content in files that carry no pending changes of theirs.
