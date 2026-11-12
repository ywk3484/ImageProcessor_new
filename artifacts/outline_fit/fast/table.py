"""Scratch: table of a truth run (runs/TAG.json): one pass against the existing fit."""
import json, sys
from pathlib import Path
for tag in sys.argv[1:]:
    rows = json.loads((Path("runs") / f"{tag}.json").read_text())
    print(f"== {tag}")
    print(f"{'page':<26s} {'family':<7s} {'what':<10s} {'mean err':>9s} {'slope':>7s} {'scatter':>8s} {'+10% light':>10s} {'centres':>8s} {'blur':>12s} {'s/page':>8s}")
    for r in rows:
        extra = ",".join(f"{k}={v}" for k, v in r.get("settings", {}).items() if k in ("rot", "part", "flicker", "pitch", "place", "cd", "bright", "blur") and v not in (None, 0.0, 2.5, 1.0, 0.05, 65.0))
        name = r["name"] + (f" [{extra}]" if extra else "")
        if "failed" in r:
            print(f"{name:<26s} {r['family']:<7s} FAILED: {r['failed'].strip().splitlines()[-1][:120]}")
            continue
        for what in ("fast", "existing"):
            if what not in r:
                continue
            d = r[what]; ph = d["per_hole"]
            print(f"{name:<26s} {r['family']:<7s} {'one pass' if what == 'fast' else 'existing':<10s} {d['error']:+9.3f} {ph['slope']:7.3f} {ph['scatter']:8.2f} {ph['light10']:+10.3f} "
                  f"{ph['centre_sd']:8.2f} {d['blur'][0]:5.1f}/{d['blur'][1]:5.1f} {d['seconds']:8.3f}" + ("" if what == "fast" else f"  ({d['rounds']} rounds{'' if d['converged'] else ', NOT converged'})"))
