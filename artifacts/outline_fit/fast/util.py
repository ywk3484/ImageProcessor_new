"""Scratch: GPU utilisation while the one-pass fit runs over the test file's hole pages (nvidia-smi samples)."""
import json, subprocess, sys, threading, time
from pathlib import Path
import cupy as cp, numpy as np, tifffile
from gpufit import ROOT, sync
from pipeline import fit_pages
REF = Path("runs/ref")
rows = [r["page"] for r in json.loads((REF / "ref.json").read_text()) if not r.get("skipped") and r["holes"] > 30000]
with tifffile.TiffFile(ROOT / "artifacts/cd_test/data/stripe1_board1.tif") as t:
    images = [t.pages[p].asarray() for p in rows]
fit_pages(images[:4])
def sampler(stop, out):
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "100"], stdout=subprocess.PIPE, text=True)
    try:
        while not stop.is_set():
            line = proc.stdout.readline()
            try:
                u, m = (float(v) for v in line.split(","))
                out.append((u, m))
            except ValueError:
                pass
    finally:
        proc.kill(); proc.wait()
for level in (1, 2):
    stop, out = threading.Event(), []
    th = threading.Thread(target=sampler, args=(stop, out), daemon=True); th.start()
    time.sleep(0.5); out.clear()
    t0 = time.perf_counter(); n = 0
    while time.perf_counter() - t0 < 15:
        for s in range(0, len(images), 20):
            fit_pages(images[s:s + 20], level=level, sweeps_d=3, relax=1.3); n += len(images[s:s + 20])
    sync(); dt = time.perf_counter() - t0
    stop.set(); th.join(timeout=3)
    u = np.array([o[0] for o in out]); m = np.array([o[1] for o in out])
    print(f"level {level}: {n} pages in {dt:.1f} s = {dt / n * 1e3:.1f} ms per page; GPU utilisation mean {u.mean():.0f} %, median {np.median(u):.0f} %, memory peak {m.max():.0f} MiB ({len(u)} samples)")
