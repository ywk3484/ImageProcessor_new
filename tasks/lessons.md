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
  that is silent, research or ask. A test with the premise built in (a synthetic page, or the one real file at
  hand) cannot show it false.
- Do not assume shapes, layouts, noise levels, what the pages or images of a dataset have in common, or other
  properties the user has not stated. Label each assumed value, and pick it so it does not hide the differences
  the user asked about.
- Before replacing what the user suspects is the error source, measure that it is (oracle test: give the method the
  true value of the suspect and see how much error goes away).
- When the user proposes a test that is cheap to run, run it and report the numbers; keep counter-theory to a line.

## Testing and comparing
- Test beyond the happy path: the user's example image (examples/data/synthetic_ch), staggered as well as square
  arrays, larger, irregular and rotated shapes, more blur; check the share of features measured, not only the error.
  Before handing a method on, also run it the way its caller will: the caller's own inputs (its seeds), its small
  images (hundreds of holes, not only tens of thousands), a pitch of a whole number of pixels, and repeated images
  of the same content (new noise, the nuisance changed a little): the spread between repeats is the user's first test.
- An iteration cap is a safety net, not a stop: set it several times above the most iterations seen, and measure
  what the result is when it is hit before shipping it.
- Before quoting a fit's accuracy or stability, start it from many points on the same data and list where it ends
  (two changed starts are not a map), and check that no start value equals the value the test data were made with.
- Define per-feature regions relative to the feature's own territory, never as a fixed fraction of the pitch.
- Test a diagnostic's reading guide on planted causes: plant each candidate and check it names the right one.
- Comparisons: calibrate once and reuse it for every image compared (per-image calibration fakes repeatability); a
  smaller spread is better precision only if the method follows the true change at slope 1; a reference must not
  use the stop rule under test; show the physical floor in every comparison.
- Fill comparison tables with the report's own ranking code, never by eye, and quote ranges from that output.
- A claimed gain from one step needs a control run that changes only that step.
- A stage that converges on an exact noise-free page to a leftover above the model's precision has a defect in what
  it shows each block (its bookkeeping), whatever the model looks like: check that before chasing model errors.
- Whether a step is needed follows from the effect it removes and the data property behind that effect, not from
  synthetic scores (test pages obey the method's assumptions by construction). "I found no job for it" is not "it
  has no job": list what it absorbs, or say "not pinned down".
- A test limit met only by buggy code encodes the bug: after a fix, recompute the quantity with the old code at
  another size before changing the limit, and report the change rather than just relaxing it.
- Keep stored or displayed data at the precision the method used wherever the method is sensitive to it.
- Check where a test file comes from (the script that made it, an exact lattice, its metadata) before calling a
  result real-data; name generated data as generated wherever its numbers are quoted.
- Generate test patterns at a pitch that is not a whole number of pixels: on an exact pixel lattice every feature
  has the same sub-pixel position, so anything that averages or samples by sub-pixel position goes untested.

## Speed and I/O
- Measure under the user's conditions: real file size, real disk, cold cache, pages that behave like real data
  (real pages may never converge the way synthetic ones do), and the worker count the target runs. If not
  possible, say the number is unknown.
- Profile before naming a bottleneck or its cause: time each stage; for a library call, time a plain read, then
  each step of the call, and read its source. "The GPU is busy" is not "the GPU is used well": compare bytes moved
  with the data size, float64 use, idle share and number of launches.
- Give the time for the user's own units of work first (their image, stripe and dataset, not a page: pages x
  seconds per page / GPUs), as a range that names the deciding unknown, and explain every outlier before reporting.
- Turn the user's target into a budget per unit of work first and hold the approach's best case against it. When an
  iteration cannot meet it, find what makes it iterate and put that into the model; cheaper passes are not enough.

## Reporting
- A method page's physics section is written as lecture notes: say first which data object every quantity refers
  to (which image, which crop, which average), show each unit as a drawing on the object's own scale before using
  it, define each quantity, derive each equation, justify each numerical choice (grid, frequency step, window), walk
  through each figure. Compact prose there is not an explanation, and a derivation without its object is not one either.
- Explain a result from what the method does, in everyday words, then one number with what it should be and what
  it is. Never by walking through the fields of a tool's output line, and never in units the user does not use.
- Give every number its unit the first time it appears, or say that it has none and what 1 means. A fit score that
  is a ratio to the noise is not compared between data of different noise or size: convert it to the data's own
  units (grey levels, percent of the signal) first, and never read "far above the noise" as "large".
- Never state a cause that was not measured: a model error seen at the truth is the cause only once removing it
  moves the estimate. Never call a partial fix "fixed": say what is left and what it costs.
- Before reporting, check the deliverable against the literal request (the level, the place, the view asked for).
- When something failed, say so plainly; never explain a miss by what the user did not say.
- When raising a defect, say: bug or not, the measured cause (isolated by a test that changes one thing), how often
  it triggers on the user's data, the exact fix with a prototype result, and how it changes the user's numbers.
- A measurement tool measures: add no statistics the user did not ask for. Show a value at the level it was
  measured: one number per image is not painted on every hole of a map.
- Look at every plot as the user will, at its default view: truth and measured in separate panels, planted events
  rare enough to read as events, controls next to the graph they change.
- In a page with many figures, number every figure in page order and refer to it by that number; letters name
  panels inside one figure only. Typeset mathematics (TeX), never as HTML entities. A process (a sum built term by
  term, a kernel sliding) is shown as a stepped animation the reader can scrub, not as a row of stills. Say each
  thing once; a later mention is a pointer to the figure or part that said it.

## Commands, tools and UI
- Commands for another machine: written for the shell it really has, one plain command per line, with every option
  the user's case needs shown in the example itself.
- Check every input before writing any state that a later run is compared with.
- Console output: ASCII only (the user's Windows console is cp949).
- For a "looks wrong / differs from before" report, ask which view and which image first, and for what the tool
  itself printed about its fit on that data (leftover, counts), before reproducing anything on generated data. Compare what the user
  looked at (every stage, layer and overlay of the old UI), not the widget with the same name. A view for judging
  a result shows it on the image.

- When the cause of a fault on data I cannot see is not known, do not hand the user checks to run and type back
  one after another: build a view that shows every step of the calculation on their data (the input, each
  intermediate result, the output) so they can see which step is odd. A typed-back number can confirm a cause
  already suspected; it cannot find one.

## Repo hygiene
- Stage only your own files, by explicit path; the working tree holds the user's uncommitted work. Do not edit
  files that hold the user's pending changes.

## Writing lessons
- Add a lesson only for a rule that would change behaviour on a different task: one or two lines, no story, quotes,
  dates or incident numbers. Sharpen an existing rule rather than append a new one. Task notes go in tasks/todo.md.
