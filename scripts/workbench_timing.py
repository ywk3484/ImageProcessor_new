"""Time workbench interactions on a user-size page and save screenshots.

Baseline and phase checks for docs/superpowers/specs/2026-10-31-workbench-ux-review-design.md. It drives the
InspectionWindow through its attributes, so adapt it when the layout changes.

    .venv\\Scripts\\python.exe scripts\\workbench_timing.py --out artifacts\\workbench_timing

Runs offscreen. On Windows the offscreen Qt platform has no fonts unless QT_QPA_FONTDIR points at
C:/Windows/Fonts; this script sets both variables when they are missing.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIFF = ROOT / "artifacts" / "cd_test" / "data" / "stripe1_board1.tif"


def memory_gb():
    try:
        import psutil
    except ImportError:
        return float("nan")
    return psutil.Process().memory_info().rss / 1e9


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tiff", nargs="?", type=Path, default=DEFAULT_TIFF)
    parser.add_argument("--page", type=int, default=0, help="0 = first page")
    parser.add_argument("--backend", choices=("gpu", "cpu", "auto"), default="gpu")
    parser.add_argument("--more-pages", type=int, default=4,
                        help="Measure this many following pages too (memory once the page limit is reached)")
    parser.add_argument("--no-stack", action="store_true", help="Skip the Stack vertically timings")
    parser.add_argument("--out", type=Path, help="Folder for screenshots (none when omitted)")
    args = parser.parse_args(argv)

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if sys.platform == "win32":
        os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")
    sys.path.insert(0, str(ROOT))
    from PyQt6 import QtCore, QtWidgets
    from subpx.workbench import InspectionWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # A throw-away settings file: the defaults are timed and the user's saved settings stay untouched.
    settings_dir = tempfile.TemporaryDirectory()
    settings = QtCore.QSettings(str(Path(settings_dir.name) / "workbench.ini"), QtCore.QSettings.Format.IniFormat)
    window = InspectionWindow(settings=settings)
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
        print(f"{label:<44} {time.perf_counter() - start:7.3f} s", flush=True)

    def repaint(label, repeats=3):
        window.canvas.plot.grab()  # the first paint after a change builds caches
        start = time.perf_counter()
        for _ in range(repeats):
            window.canvas.plot.grab()
        print(f"{label:<44} {(time.perf_counter() - start) / repeats:7.3f} s", flush=True)

    def shot(name):
        if args.out:
            window.grab().save(str(args.out / f"{name}.png"))

    def drain_stack(timeout=60):
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            app.processEvents()
            stack = window.stack
            if stack is None or not (stack._queue or stack._decode_timer.isActive() or stack._update_timer.isActive()):
                break
            time.sleep(0.002)
        return time.perf_counter() - start

    timed("open page", lambda: window.open_image(path=str(args.tiff), page=args.page), job=True)
    window.backend.setCurrentIndex(window.backend.findData(args.backend))
    window.task.setCurrentIndex(window.task.findData("cd"))
    timed(f"Measure CD, default preset ({args.backend})", window.run_button.click, job=True)
    review = window.review
    print(f"holes: {len(review.seeds_xy):,}   memory {memory_gb():.2f} GB")
    for name, measured in review.measurements.items():
        failed = int((~measured.valid).sum())
        print(f"  {name}: {failed:,} failed ({100 * failed / max(len(measured.valid), 1):.0f} %)")
    window.canvas.fit_image()
    app.processEvents()
    shot("full_view")
    repaint("repaint at full view (default layers)")
    timed("toggle CD colors on", lambda: window.layer_checks["CD colors"].setChecked(True))
    shot("full_view_cd")
    repaint("repaint at full view, CD colors")
    middle = len(review.seeds_xy) // 2
    timed("select a hole", lambda: window.select_feature(middle))
    timed("select the next hole", lambda: window.select_feature(middle + 1))
    timed("click a hole on the image", lambda: window._select_at(*(review.seeds_xy[middle + 2] + review.origin_xy)))
    timed("switch method", lambda: window.method.setCurrentIndex(1))
    timed("switch method back", lambda: window.method.setCurrentIndex(0))
    for name in ("Voronoi edges", "CD colors", "Failed cells", "Centers"):
        check = window.layer_checks[name]
        timed(f"toggle {name}", lambda: check.setChecked(not check.isChecked()))
        timed(f"toggle {name} back", lambda: check.setChecked(not check.isChecked()))
    timed("zoom to the selected hole", window.zoom_selected)
    view = window.canvas.plot.getViewBox()
    center = review.seeds_xy[middle] + review.origin_xy
    view.setRange(xRange=(center[0] - 200, center[0] + 200), yRange=(center[1] - 130, center[1] + 130), padding=0)
    app.processEvents()
    print(f"holes in view at 400 x 260 px: {window.layers.in_view:,} (vectors drawn: {window.layers.near})")
    shot("zoomed_400px")
    repaint("repaint zoomed in, vectors drawn")
    window.zoom_selected()
    app.processEvents()
    shot("selected_hole")
    timed("show failed only", lambda: window.failed_only.setChecked(True))
    timed("show all", lambda: window.failed_only.setChecked(False))
    column = window.table_model.METHOD_COLUMNS
    timed("sort table by the first method's CD", lambda: window.table.sortByColumn(column, QtCore.Qt.SortOrder.DescendingOrder))
    timed("switch method (sort kept)", lambda: window.method.setCurrentIndex(2))
    values = [window.table_model.value(row, column) for row in range(50)]
    print(f"  still sorted after the switch: {all(a >= b for a, b in zip(values, values[1:]))}")
    window.table.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)

    pages = window.loaded.pages
    for page in range(args.page + 1, min(pages, args.page + 1 + args.more_pages)):
        window.page_spin.setValue(page)
        wait()
        start = time.perf_counter()
        window.run_button.click()
        wait()
        kept = sorted({key[0] for key in window._task_reviews})
        print(f"page {page}: measured in {time.perf_counter() - start:5.1f} s, results kept for pages {kept}, "
              f"memory {memory_gb():.2f} GB", flush=True)

    if not args.no_stack and pages > 1:
        before = memory_gb()
        timed("Stack vertically on (read page sizes)", lambda: window.stack_pages.setChecked(True), job=True)
        print(f"  pages decoded while filling the view: {drain_stack():.2f} s")
        shot("stack_full")
        repaint("repaint, whole stack in view")
        timed("click a hole on the active page", lambda: window._select_at(
            *(window.review.seeds_xy[100] + window.review.origin_xy + window._display_offset())))
        other = (window.loaded.page + 1) % pages
        timed("click another page (make it active)", lambda: window._select_at(10, window.stack.offsets[other] + 10),
              job=True)
        timed("Shift+F, then F", lambda: (window.fit_stack(), window.canvas.fit_image()))
        drain_stack()
        repaint("repaint, active page fitted")
        window.fit_stack()
        for _ in range(3):  # pan through the stack
            view = window.canvas.plot.getViewBox()
            (x0, x1), (y0, y1) = view.viewRange()
            view.setRange(yRange=(y0 + (y1 - y0) / 3, y1 + (y1 - y0) / 3), padding=0)
            drain_stack()
        stack = window.stack
        print(f"stack: {len(stack.previews)} previews, {len(stack.full)} full pages, {stack.memory() / 1e6:.0f} MB in "
              f"the stack; memory {before:.2f} -> {memory_gb():.2f} GB")
    window.close()
    settings_dir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
