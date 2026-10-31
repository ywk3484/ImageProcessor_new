"""Time workbench interactions on a user-size page and save screenshots.

Baseline for docs/superpowers/specs/2026-10-31-workbench-ux-review-design.md. It drives the current
InspectionWindow through its attributes, so adapt it when the layout changes.

    .venv\\Scripts\\python.exe scripts\\workbench_timing.py --out artifacts\\workbench_timing

Runs offscreen. On Windows the offscreen Qt platform has no fonts unless QT_QPA_FONTDIR points at
C:/Windows/Fonts; this script sets both variables when they are missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIFF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiff", nargs="?", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--page", type=int, default=0, help="0 = first page")
    parser.add_argument("--backend", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--out", type=Path, help="Folder for screenshots (none when omitted)")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    sys.path.insert(0, str(ROOT))
    from PyQt6 import QtCore, QtWidgets
    from subpx.workbench import InspectionWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InspectionWindow()
    window.show()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    def wait():
        while window.busy:
            app.processEvents()
            time.sleep(0.005)
        app.processEvents()

    def timed(label, action, *, job=False):
        start = time.perf_counter()
        action()
        if job:
            wait()
        app.processEvents()
        print(f"{label:<40} {time.perf_counter() - start:7.2f} s", flush=True)

    def repaint(label, repeats=3):
        start = time.perf_counter()
        for _ in range(repeats):
            window.canvas.plot.grab()
        print(f"{label:<40} {(time.perf_counter() - start) / repeats:7.2f} s", flush=True)

    def shot(name):
        if args.out:
            window.grab().save(str(args.out / f"{name}.png"))

    timed("open page", lambda: window.open_image(path=str(args.tiff), page=args.page), job=True)
    window.backend.setCurrentIndex(window.backend.findData(args.backend))
    window.task.setCurrentIndex(window.task.findData("cd"))
    timed(f"Measure CD, default preset ({args.backend})", window.run_button.click, job=True)
    review = window.review
    print(f"holes: {len(review.seeds_xy):,}")
    for name, measured in review.measurements.items():
        failed = int((~measured.valid).sum())
        print(f"  {name}: {failed:,} failed ({100 * failed / max(len(measured.valid), 1):.0f} %)")
    window.canvas.fit_image()
    app.processEvents()
    shot("full_view")
    repaint("repaint at full view")
    middle = len(review.seeds_xy) // 2
    timed("select a hole", lambda: window.select_feature(middle))
    timed("select the next hole", lambda: window.select_feature(middle + 1))
    window.zoom_selected()
    app.processEvents()
    shot("selected_hole")
    timed("switch method", lambda: window.method.setCurrentIndex(1))
    timed("toggle Voronoi edges", lambda: window.layer_checks["Voronoi edges"].setChecked(True))
    timed("toggle CD colors", lambda: window.layer_checks["CD colors"].setChecked(True))
    timed("show failed only", lambda: window.failed_only.setChecked(True))
    timed("sort table by CD", lambda: window.table.sortByColumn(4, QtCore.Qt.SortOrder.DescendingOrder))
    window.close()


if __name__ == "__main__":
    raise SystemExit(main())
