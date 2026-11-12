"""Scratch: the one-pass fit and the existing fit against the truth on a set of generated pages.

    python truth_matrix.py [set=main|more|blur] [out=TAG]

Each case is one generated page (subpx.synthetic_arrays.render_hole_array through truth_gpu.run_case). Results go to
runs/TAG.json and are printed as they come.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from truth_gpu import run_case

SETS = {
    # name, family, variants, overrides
    "main": [
        ("circle100_square", "gauss2", ("fast", "existing"), {}),
        ("circle100_square", "lens", ("fast", "existing"), {}),
        ("circle80_square", "gauss2", ("fast", "existing"), {}),
        ("circle80_square", "lens", ("fast", "existing"), {}),
        ("circle100_stagger", "lens", ("fast", "existing"), {}),
        ("square150_stagger", "gauss2", ("fast", "existing"), {}),
        ("square150_stagger", "lens", ("fast", "existing"), {}),
        ("circle250_square", "lens", ("fast", "existing"), {}),
        ("ellipse160x80_square", "lens", ("fast", "existing"), {}),
        ("ellipse100x200_stagger", "lens", ("fast", "existing"), {}),
    ],
    "more": [
        ("circle100_square", "lens", ("fast",), dict(rot=3.0)),
        ("circle100_square", "lens", ("fast",), dict(part=0.4)),
        ("circle100_square", "lens", ("fast",), dict(flicker=0.10)),
        ("circle100_square", "lens", ("fast",), dict(pitch=(178.5, 178.5), place=3.6)),
        ("circle100_square", "lens", ("fast",), dict(cd=0.5)),
        ("circle100_square", "lens", ("fast",), dict(cd=5.0)),
        ("circle100_square", "lens", ("fast",), dict(bright=0.15)),
        ("circle60_dense", "lens", ("fast", "existing"), {}),
        ("circle50_dense", "lens", ("fast", "existing"), {}),
        ("ellipse90x60_stagger", "lens", ("fast", "existing"), {}),
    ],
    "blur": [
        ("circle100_square", "lens", ("fast",), dict(blur=58.0)),
        ("circle100_square", "lens", ("fast",), dict(blur=72.0)),
        ("circle100_square", "gauss2", ("fast",), dict(blur=58.0)),
        ("circle100_square", "gauss2", ("fast",), dict(blur=72.0)),
        ("circle80_square", "lens", ("fast",), dict(blur=58.0)),
        ("circle80_square", "lens", ("fast",), dict(blur=72.0)),
    ],
}


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    which = args.get("set", "main")
    tag = args.get("out", f"truth_{which}")
    out_file = Path(__file__).parent / "runs" / f"{tag}.json"
    rows = []
    for name, family, variants, over in SETS[which]:
        t0 = time.perf_counter()
        if args.get("existing", "1") == "0":
            variants = tuple(v for v in variants if v != "existing")
        try:
            rows.append(run_case(name, family=family, variants=variants, say=lambda s: print(s, flush=True),
                                 sweeps_d=int(args.get("sweeps_d", 3)), relax=float(args.get("relax", 1.3)), **over))
        except Exception:  # a case that fails is part of the result
            print(f"{name} [{family}] {over}: FAILED\n{traceback.format_exc()}", flush=True)
            rows.append(dict(name=name, family=family, settings=over, failed=traceback.format_exc()))
        print(f"  ({time.perf_counter() - t0:.0f} s)\n", flush=True)
        out_file.write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
