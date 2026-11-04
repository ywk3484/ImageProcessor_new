# Lessons

General rules from past corrections, one or two lines each. Incident details are in git history and tasks/archive/.

## Before building
- Ask for the quantity you need directly (image size, bit depth, what one file or dataset is, which population a
  statistic is over, the target OS and shell). Never derive it from rough totals or infer it from files in the repo.
- When a word names two things in this repo, ask which one, naming both in the user's terms.
- A nuisance the user names is a hard constraint: every candidate must be insensitive to it. Ask what has already
  been tried; never re-propose, in a new form, a method the user found failing on real data.
- Before building a method, list what its core premise needs from the data (what varies together, over what
  distance, how the instrument forms the image) and check each against what is known about the equipment; where
  that is silent, research or ask. A synthetic test with the premise built in cannot show it false.
- Do not assume shapes, layouts, noise levels or other properties the user has not stated. Label each assumed value,
  and pick it so it does not hide the differences the user asked about.
- Before replacing what the user suspects is the error source, measure that it is (oracle test: give the method the
  true value of the suspect and see how much error goes away).
- When the user proposes a test that is cheap to run, run it and report the numbers; keep counter-theory to a line.

## Testing and comparing
- Test beyond the happy path: the user's example image (examples/data/synthetic_ch), staggered as well as square
  arrays, larger, irregular and rotated shapes, more blur; check the share of features measured, not only the error.
- Define per-feature regions relative to the feature's own territory, never as a fixed fraction of the pitch.
- Test a diagnostic's reading guide on planted causes: plant each candidate and check it names the right one.
- Comparisons: calibrate once and reuse it for every image compared (per-image calibration fakes repeatability); a
  smaller spread is better precision only if the method follows the true change at slope 1; a reference must not
  use the stop rule under test; show the physical floor in every comparison.
- Fill comparison tables with the report's own ranking code, never by eye, and quote ranges from that output.
- A claimed gain from one step needs a control run that changes only that step.
- Whether a step is needed follows from the effect it removes and the data property behind that effect, not from
  synthetic scores (test pages obey the method's assumptions by construction). "I found no job for it" is not "it
  has no job": list what it absorbs, or say "not pinned down".
- A test limit met only by buggy code encodes the bug: after a fix, recompute the quantity with the old code at
  another size before changing the limit, and report the change rather than just relaxing it.
- Keep stored or displayed data at the precision the method used wherever the method is sensitive to it.

## Speed and I/O
- Measure under the user's conditions: real file size, real disk, cold cache, pages that behave like real data
  (real pages may never converge the way synthetic ones do), and the worker count the target runs. If not
  possible, say the number is unknown.
- Profile before naming a bottleneck or its cause: time each stage; for a library call, time a plain read, then
  each step of the call, and read its source. "The GPU is busy" is not "the GPU is used well": compare bytes moved
  with the data size, float64 use, idle share and number of launches.
- Give the whole-dataset time first (pages x seconds per page / GPUs), as a range that names the deciding unknown,
  and explain every outlier before reporting.

## Reporting
- Never state a cause that was not measured. Never call a partial fix "fixed": say what is left and what it costs.
- Before reporting, check the deliverable against the literal request (the level, the place, the view asked for).
- When something failed, say so plainly; never explain a miss by what the user did not say.
- When raising a defect, say: bug or not, the measured cause (isolated by a test that changes one thing), how often
  it triggers on the user's data, the exact fix with a prototype result, and how it changes the user's numbers.
- A measurement tool measures: add no statistics the user did not ask for.
- Look at every plot as the user will, at its default view: truth and measured in separate panels, planted events
  rare enough to read as events, controls next to the graph they change.

## Commands, tools and UI
- Commands for another machine: written for the shell it really has, one plain command per line, with every option
  the user's case needs shown in the example itself.
- Check every input before writing any state that a later run is compared with.
- Console output: ASCII only (the user's Windows console is cp949).
- For a "looks wrong / differs from before" report, ask which view and which image first. Compare what the user
  looked at (every stage, layer and overlay of the old UI), not the widget with the same name. A view for judging
  a result shows it on the image.

## Repo hygiene
- Stage only your own files, by explicit path; the working tree holds the user's uncommitted work. Do not edit
  files that hold the user's pending changes.

## Writing lessons
- Add a lesson only for a rule that would change behaviour on a different task: one or two lines, no story, quotes,
  dates or incident numbers. Sharpen an existing rule rather than append a new one. Task notes go in tasks/todo.md.
