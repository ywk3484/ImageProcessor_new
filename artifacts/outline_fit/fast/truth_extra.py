"""Scratch: the existing fit on the blur-series pages and a few of the further cases (like-for-like with the one-pass runs)."""
import json, time, traceback
from pathlib import Path
from truth_gpu import run_case
cases = [("circle100_square", "lens", dict(blur=58.0)), ("circle100_square", "lens", dict(blur=72.0)),
         ("circle100_square", "lens", dict(flicker=0.10)), ("circle100_square", "lens", dict(rot=3.0)),
         ("circle100_square", "lens", dict(pitch=(178.5, 178.5), place=3.6)), ("circle100_square", "lens", dict(cd=5.0))]
rows = []
for name, family, over in cases:
    try:
        rows.append(run_case(name, family=family, variants=("existing",), say=lambda s: print(s, flush=True), **over))
    except Exception:
        print(traceback.format_exc(), flush=True)
    Path("runs/truth_extra_existing.json").write_text(json.dumps(rows, indent=1, default=float))
