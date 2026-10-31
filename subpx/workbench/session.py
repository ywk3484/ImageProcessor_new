"""The workbench's state: the open file and page, the results of each page, the selected hole and method, the
moving image and the alignment.

Panels read the session and listen to its signals; they never call each other. Results are kept per page:
the previewed holes, the CD measurement and the pitch, side by side. The method box decides which of the first
two is on screen ("Previewed holes" or a CD method); running a step never hides another step's result.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np
from PyQt6 import QtCore

from .._workbench_data import (
    HOLE_CHECK_MAX_ERROR, AnalysisReview, LoadedImage, PageReader, analyze_image, analyze_patterns, array_bytes,
    demo_image, export_review, grayscale, page_preview, register_images,
)
from .jobs import JobRunner
from .names import method_name, plain_error
from .settings import RESULT_KEYS, STEP_KEYS, changes, run_problems

# Results are kept for the open page and the most recent other pages: at most this many pages and bytes.
RESULT_PAGES = 3
RESULT_BYTES = 3e9


class PageSource:
    """An open file: its page table read once, the file kept open, decoded pages and small previews cached.

    Full pages are kept up to FULL_BYTES, the least recently used dropped first; previews (block means at most
    256 pixels wide) are kept for every page decoded. ``read`` may run on the worker and in the window at once.
    """

    FULL_BYTES = 128e6

    def __init__(self, reader):
        self.reader = reader
        self.path = str(Path(reader.path).resolve())
        self.name = Path(reader.path).name
        self.shapes = [tuple(int(size) for size in shape) for shape in reader.shapes]
        self.pages = len(self.shapes)
        self.full = OrderedDict()
        self.previews = {}
        self._lock = threading.Lock()

    def cached(self, page):
        with self._lock:
            loaded = self.full.get(page)
            if loaded is not None:
                self.full.move_to_end(page)
            return loaded

    def read(self, page):
        loaded = self.cached(page)
        if loaded is None:
            if not 0 <= page < self.pages:
                raise ValueError(f"Page must be between 0 and {self.pages - 1}.")
            loaded = self.reader.read(page)
            self.remember(loaded)
        return loaded

    def remember(self, loaded):
        preview = None if loaded.page in self.previews else page_preview(loaded.image)
        with self._lock:
            if preview is not None:
                self.previews[loaded.page] = preview
            self.full[loaded.page] = loaded
            self.full.move_to_end(loaded.page)
            total = sum(item.image.nbytes for item in self.full.values())
            while len(self.full) > 1 and total > self.FULL_BYTES:
                _, dropped = self.full.popitem(last=False)
                total -= dropped.image.nbytes

    def preview(self, page):
        return self.previews.get(page)

    def memory(self):
        with self._lock:
            return (sum(data.nbytes for data, _ in self.previews.values())
                    + sum(loaded.image.nbytes for loaded in self.full.values()))

    def close(self):
        self.reader.close()


@dataclass
class PageResults:
    """One page's results: previewed holes, CD and pitch live together."""

    detect: AnalysisReview | None = None
    cd: AnalysisReview | None = None
    pitch: AnalysisReview | None = None
    pitch_from: AnalysisReview | None = None  # the result the pitch was measured on

    def reviews(self):
        return [review for review in (self.detect, self.cd, self.pitch) if review is not None]

    def nbytes(self):
        return array_bytes(*self.reviews())


def hole_near(review, xy):
    """The hole whose cell holds the loaded-page point xy, else the nearest within 3/4 of a pitch; -1: none."""
    if review is None or xy is None or not len(review.seeds_xy):
        return -1
    local = np.asarray(xy, float) - review.origin_xy
    h, w = review.image.shape
    if review.cd is not None and -0.5 <= local[0] < w - 0.5 and -0.5 <= local[1] < h - 0.5:
        index = int(review.cd.cell_labels[int(np.floor(local[1] + .5)), int(np.floor(local[0] + .5))])
        if index >= 0:
            return index
    distance = np.hypot(*(review.seeds_xy - local).T)
    nearest = int(np.argmin(distance))
    spacing = np.sqrt(review.image.size / len(review.seeds_xy))
    return nearest if distance[nearest] <= 0.75 * spacing else -1


def failure_counts(review):
    """[(failed holes, method)] of every method with failures, largest first."""
    return sorted(((int((~measured.valid).sum()), name) for name, measured in review.measurements.items()
                   if not measured.valid.all()), reverse=True)


class Session(QtCore.QObject):
    image_changed = QtCore.pyqtSignal(bool)  # True: another file or image; False: another page of the open file
    results_changed = QtCore.pyqtSignal()  # the active page's results or the result on screen changed
    method_changed = QtCore.pyqtSignal()
    selection_changed = QtCore.pyqtSignal(str)  # why: image, table, key, failed, inspector, pitch, result
    highlight_changed = QtCore.pyqtSignal()
    settings_changed = QtCore.pyqtSignal()
    stacked_changed = QtCore.pyqtSignal(bool)
    moving_changed = QtCore.pyqtSignal()
    registration_changed = QtCore.pyqtSignal()
    busy_changed = QtCore.pyqtSignal(bool)
    log = QtCore.pyqtSignal(str)
    status = QtCore.pyqtSignal(str, str)  # text, level: info, warning or error
    error = QtCore.pyqtSignal(str)  # one plain line for the banner; details are in the log

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = JobRunner(self)
        self.jobs.started.connect(lambda label: (self.log.emit(label), self.busy_changed.emit(True)))
        self.jobs.stopped.connect(lambda: self.busy_changed.emit(False))
        self.jobs.failed.connect(self.fail)
        self.jobs.log.connect(self.log)
        self.loaded = None
        self.source = None  # PageSource of the open file (None for arrays and the demo)
        self.results = OrderedDict()  # page -> PageResults, the most recent last
        self.marks = {}  # page -> (short line, detail) on what was measured there (kept when results are dropped)
        self.method = None
        self._wanted_cd = None  # the CD method last chosen, shown again when a CD result comes back
        self.selected = None  # hole index in the result on screen
        self.selected_xy = None  # loaded-page [x, y] of the selected hole, kept across pages and runs
        self.order = None  # hole order of the holes table (previous / next hole)
        self.highlight, self.highlight_label = None, ""  # holes picked in CD distribution or Pitch
        self.stacked = False
        self.moving = None
        self.moving_source = None  # PageSource of a moving file other than the open one
        self.registration = None
        self.parameters = {}  # the Setup settings as analysis parameters
        self.bounds = None  # analysis region, page-local [x0, y0, x1, y1], or None

    # ------------------------------------------------------------------------------------------- state
    @property
    def busy(self):
        return self.jobs.busy

    @property
    def page_results(self):
        return self.results.get(self.loaded.page) if self.loaded is not None else None

    @property
    def review(self):
        """The result on screen: the previewed holes or the CD measurement, as the method box says."""
        results = self.page_results
        if results is None:
            return None
        if self.method == "detection" or results.cd is None:
            return results.detect if results.detect is not None else results.cd
        return results.cd

    @property
    def pitch_review(self):
        results = self.page_results
        return results.pitch if results is not None else None

    @property
    def measured(self):
        """The current method's measurement in the result on screen, or None."""
        review = self.review
        return review.measurements.get(self.method) if review is not None else None

    def methods(self):
        """The methods to choose from: the CD methods, then 'detection' when holes were previewed."""
        results = self.page_results
        if results is None:
            return []
        names = list(results.cd.measurements) if results.cd is not None else []
        return names + (["detection"] if results.detect is not None else [])

    def has_results(self):
        return any(results.reviews() for results in self.results.values()) or self.registration is not None

    def current_parameters(self):
        return dict(self.parameters, bounds=self.bounds)

    def problems(self):
        return run_problems(self.parameters, loaded=self.loaded, results=self.page_results, moving=self.moving)

    # ------------------------------------------------------------------------------------------ settings
    def set_parameters(self, parameters):
        self.parameters = dict(parameters)
        self.settings_changed.emit()

    def set_bounds(self, bounds):
        bounds = list(bounds) if bounds is not None else None
        if bounds != self.bounds:
            self.bounds = bounds
            self.settings_changed.emit()

    def stale(self):
        """[(key, text)] of the settings changed since the results on screen were made."""
        results, current = self.page_results, self.current_parameters()
        review = self.review
        found = []
        if review is not None:
            found += changes(review.parameters, current, RESULT_KEYS["detect" if review.cd is None else "cd"])
        pitch = results.pitch if results is not None else None
        if pitch is not None:
            found += [item for item in changes(pitch.parameters, current, RESULT_KEYS["pitch"]) if item not in found]
            if results.pitch_from is not self._pitch_source(results, pitch.parameters.get("pitch_source")):
                found.append(("pitch_from", "pitch measured on an earlier result"))
        return found

    def step_states(self):
        """Each Setup step: 'none' (no result made with it), 'ok' or 'changed'."""
        results, current = self.page_results, self.current_parameters()
        review = self.review
        states = {}
        for step in ("image", "holes", "cd", "pitch"):
            target = (review if step in ("image", "holes") else
                      results.cd if step == "cd" and results is not None else
                      results.pitch if step == "pitch" and results is not None else None)
            if target is None:
                states[step] = "none"
                continue
            kind = "pitch" if step == "pitch" else "detect" if target.cd is None else "cd"
            keys = [key for key in STEP_KEYS[step] if key in RESULT_KEYS[kind]]
            changed = bool(changes(target.parameters, current, keys))
            if step == "pitch":
                changed |= results.pitch_from is not self._pitch_source(results, target.parameters.get("pitch_source"))
            states[step] = "changed" if changed else "ok"
        return states

    def revert_parameters(self):
        """The settings of the results on screen (Revert in the banner)."""
        review, pitch = self.review, self.pitch_review
        parameters = dict(review.parameters) if review is not None else {}
        if pitch is not None:
            parameters.update({key: pitch.parameters.get(key) for key in RESULT_KEYS["pitch"]})
        return parameters

    def rerun(self):
        """Re-run the step whose result on screen is out of date (the banner's Re-run)."""
        review, results, current = self.review, self.page_results, self.current_parameters()
        if review is not None and changes(review.parameters, current,
                                          RESULT_KEYS["detect" if review.cd is None else "cd"]):
            self.run("detect" if review.cd is None else "cd")
        elif results is not None and results.pitch is not None:
            self.run("pitch")

    @staticmethod
    def _pitch_source(results, source):
        return results.cd if source == "cd" else results.detect

    # --------------------------------------------------------------------------------------- files, pages
    def open_file(self, path, page=0):
        path, page = str(path), int(page)

        def read():
            reader = PageReader(path)
            try:
                if not 0 <= page < len(reader.shapes):
                    raise ValueError(f"Page must be between 0 and {len(reader.shapes) - 1}.")
                return reader, reader.read(page)
            except Exception:
                reader.close()
                raise

        def accept(value):
            reader, loaded = value
            self._close_sources()
            self.source = PageSource(reader)
            self.source.remember(loaded)
            self._set_loaded(loaded, new_file=True)

        self._run(f"Opening {Path(path).name}", read, accept)

    def set_array(self, image, name="Array image"):
        if self.busy:
            raise RuntimeError("Wait for the current job before replacing the image.")
        data, conversion = grayscale(image)
        self._close_sources()
        self._set_loaded(LoadedImage(data, name, conversion=conversion), new_file=True)

    def load_demo(self):
        def accept(loaded):
            self._close_sources()
            self._set_loaded(loaded, new_file=True)

        self._run("Making the demonstration image",
                  lambda: LoadedImage(demo_image(), "Synthetic staggered contact holes (seed 42)"), accept)

    def set_page(self, page):
        """Show another page of the open file; zoom, region, selected position and moving image stay."""
        source = self.source
        if self.loaded is None or source is None or self.busy:
            return
        page = int(page)
        if page == self.loaded.page or not 0 <= page < source.pages:
            return
        cached = source.cached(page)
        if cached is not None:
            self._set_loaded(cached, new_file=False)
        else:
            self._run(f"Reading page {page}", lambda: source.read(page),
                      lambda loaded: self._set_loaded(loaded, new_file=False))

    def set_stacked(self, stacked):
        stacked = bool(stacked) and self.source is not None and self.source.pages > 1
        if stacked != self.stacked:
            self.stacked = stacked
            self.stacked_changed.emit(stacked)

    def _set_loaded(self, loaded, *, new_file):
        if new_file:
            self.results.clear()
            self.marks.clear()
            self.moving = None
            self._close_moving_source()
            self.selected_xy = None
            self.bounds = None
            if self.stacked:
                self.stacked = False
                self.stacked_changed.emit(False)
        self.registration = None  # the reference page changed; the moving image stays
        self.loaded = loaded
        self.order = None
        self.log.emit(f"Loaded {loaded.name}, page {loaded.page}: {loaded.image.shape[1]} x {loaded.image.shape[0]} px, "
                      f"{loaded.image.dtype}. {loaded.conversion}".rstrip())
        self.image_changed.emit(new_file)
        if new_file:
            self.moving_changed.emit()
        self.registration_changed.emit()
        self._results_updated()
        self._keep_results()

    def _close_sources(self):
        self._close_moving_source()
        if self.source is not None:
            self.source.close()
            self.source = None

    def _close_moving_source(self):
        if self.moving_source is not None:
            self.moving_source.close()
            self.moving_source = None

    def close(self):
        self._close_sources()

    # ------------------------------------------------------------------------------------------ measuring
    def run(self, step):
        """Preview holes ('detect'), measure CD ('cd') or pitch ('pitch') on the active page."""
        if self.loaded is None or self.busy:
            return
        problem = self.problems()[0][step]
        if problem:
            self.error.emit(problem)
            return
        parameters, bounds = self.current_parameters(), self.bounds
        image, page = self.loaded.image, self.loaded.page
        source = None
        if step == "pitch":
            results = self.page_results
            source = self._pitch_source(results, parameters["pitch_source"])
            options = dict(parameters["pitch_options"])
            shown = self.review
            if parameters.get("reference_selected") and shown is not None and self.selected is not None:
                options["reference_xy"] = (shown.seeds_xy[self.selected] + shown.origin_xy - source.origin_xy).tolist()
            pitch_keys = {key: parameters.get(key) for key in RESULT_KEYS["pitch"]}

            def calculate():
                started = perf_counter()
                patterns = analyze_patterns(source, options)
                return replace(source, patterns=patterns, seconds=perf_counter() - started,
                               parameters=dict(source.parameters, operation="pitch", **pitch_keys))
        else:
            def calculate():
                return analyze_image(image, parameters, operation=step, bounds=bounds)

        labels = {"detect": "Previewing holes", "cd": "Measuring CD", "pitch": "Measuring pitch"}
        self._run(labels[step], calculate, lambda review: self._accept(page, step, review, source))

    def _accept(self, page, step, review, source):
        results = self.results.setdefault(page, PageResults())
        self.results.move_to_end(page)
        if step == "detect":
            results.detect = review
            self.method = "detection"
        elif step == "cd":
            results.cd = review
            self.method = self._wanted_cd if self._wanted_cd in review.measurements else next(iter(review.measurements), None)
        else:
            results.pitch, results.pitch_from = review, source
        self.log.emit(f"{step} done in {review.seconds:.2f} s; region origin [x, y] = {review.origin_xy.tolist()}.")
        if step == "cd":
            self.log.emit(f"Background: {review.cd.meta.get('background_definition')}")
            seconds = review.cd.meta.get("seconds", {})
            self.log.emit("Stage timings: " + ", ".join(f"{stage} {value:.2f} s" for stage, value in seconds.items()
                                                      if isinstance(value, (int, float))))
        if step != "pitch":
            for text in review.warnings:
                self.log.emit("Warning: " + text)
            for text in review.notes:
                self.log.emit(text)
        self._mark(page)
        holes = len(review.seeds_xy)
        if step == "pitch":
            self.status.emit(f"Pitch measured for {len(review.patterns)} method{'s' * (len(review.patterns) != 1)}",
                             "info")
        else:
            failed = failure_counts(review) if review.cd is not None else []
            self.status.emit(f"{holes:,} holes | {review.seconds:.1f} s" + (
                f" | most failures: {method_name(failed[0][1])} {failed[0][0]:,}" if failed else ""),
                "warning" if failed or not holes else "info")
        self._results_updated()
        self._keep_results()

    def _mark(self, page):
        results = self.results.get(page)
        if results is None:
            return
        short, detail = [], []
        if results.cd is not None:
            failed = failure_counts(results.cd)
            short.append(f"CD: most failed {failed[0][0]:,}" if failed else "CD: none failed")
            detail.append(f"CD of {len(results.cd.seeds_xy):,} holes; " + (
                "failed: " + ", ".join(f"{method_name(name)} {count:,}" for count, name in failed) if failed
                else "no method failed"))
        elif results.detect is not None:
            short.append(f"{len(results.detect.seeds_xy):,} holes previewed")
            detail.append(short[-1])
        if results.pitch is not None:
            short.append("pitch")
            detail.append("pitch measured")
        self.marks[page] = (", ".join(short), "\n".join(detail))

    def _results_updated(self):
        """The result on screen may differ: settle the method and the selection, then tell the panels."""
        names = self.methods()
        if self.method not in names:
            cd = [name for name in names if name != "detection"]
            self.method = (self._wanted_cd if self._wanted_cd in cd else cd[0] if cd else
                           "detection" if names else None)
        self._settle_selection()
        self.highlight, self.highlight_label = None, ""
        self.results_changed.emit()
        self.method_changed.emit()
        self.highlight_changed.emit()
        self.selection_changed.emit("result")

    def _settle_selection(self):
        """No hole is selected until a click; a hole selected before stays selected at its position."""
        index = hole_near(self.review, self.selected_xy)
        self.selected = index if index >= 0 else None

    def _keep_results(self):
        """Keep the open page's results and the most recent other pages': at most RESULT_PAGES pages and
        RESULT_BYTES; older pages' results are dropped (measure the page again to see them)."""
        current = self.loaded.page if self.loaded is not None else None
        kept, total, dropped = set(), 0, []
        for page in [current] + [page for page in reversed(self.results) if page != current]:
            results = self.results.get(page)
            if results is None:  # a page without results takes no place
                continue
            size = results.nbytes()
            if page == current or (len(kept) < RESULT_PAGES and total + size <= RESULT_BYTES):
                kept.add(page)
                total += size
            else:
                dropped.append(page)
        for page in dropped:
            del self.results[page]
        if dropped:
            self.log.emit(
                f"Results of page {', '.join(map(str, sorted(dropped)))} dropped to limit memory (kept: "
                f"{len(kept)} page{'s' * (len(kept) > 1)}, {total / 1e9:.1f} GB; at most {RESULT_PAGES} pages and "
                f"{RESULT_BYTES / 1e9:.0f} GB).")

    # --------------------------------------------------------------------------------- method, holes
    def set_method(self, name):
        if name == self.method or name not in self.methods():
            return
        before = self.review
        self.method = name
        if name != "detection":
            self._wanted_cd = name
        if self.review is not before:  # previewed holes <-> CD: another result on screen
            self._results_updated()
        else:
            self.method_changed.emit()

    def select(self, index, reason="image"):
        review = self.review
        if review is None or index is None or not len(review.seeds_xy):
            return
        self.selected = int(np.clip(int(index), 0, len(review.seeds_xy) - 1))
        self.selected_xy = review.seeds_xy[self.selected] + review.origin_xy
        self.selection_changed.emit(reason)

    def select_at(self, x, y):
        """Select the hole at the loaded-page position (x, y), if any."""
        index = hole_near(self.review, (x, y))
        if index >= 0:
            self.select(index, "image")

    def set_order(self, order):
        self.order = None if order is None else np.asarray(order)

    def _order(self):
        review = self.review
        if review is None:
            return np.empty(0, int)
        return self.order if self.order is not None and len(self.order) else np.arange(len(review.seeds_xy))

    def step_hole(self, direction):
        """The previous (-1) or next (+1) hole in the table's order."""
        order = self._order()
        if not len(order):
            return
        where = np.flatnonzero(order == self.selected) if self.selected is not None else []
        if not len(where):
            self.select(order[0 if direction > 0 else -1], "key")
        else:
            self.select(order[int(np.clip(where[0] + direction, 0, len(order) - 1))], "key")

    def step_failed(self, direction):
        """The next (+1) or previous (-1) hole the current method failed on, in the table's order (wraps)."""
        measured = self.measured
        if measured is None:
            return
        order = self._order()
        failed = order[~measured.valid[order]] if len(order) else order
        if not len(failed):
            self.status.emit(f"No failed holes for {method_name(self.method)}.", "info")
            return
        position = np.flatnonzero(order == self.selected)
        rank = np.flatnonzero(~measured.valid[order])  # table rows of the failed holes
        if not len(position):
            target = failed[0 if direction > 0 else -1]
        elif direction > 0:
            later = rank[rank > position[0]]
            target = order[later[0]] if len(later) else failed[0]
        else:
            earlier = rank[rank < position[0]]
            target = order[earlier[-1]] if len(earlier) else failed[-1]
        self.select(target, "failed")

    def set_highlight(self, holes, label=""):
        self.highlight = None if holes is None else np.asarray(holes, int)
        self.highlight_label = label if holes is not None else ""
        self.highlight_changed.emit()

    def holes_at(self, points_xy):
        """Holes of the result on screen at loaded-page points (pitch points of another result)."""
        review = self.review
        found = [hole_near(review, point) for point in np.asarray(points_xy, float).reshape(-1, 2)]
        return np.unique([index for index in found if index >= 0]).astype(int)

    # -------------------------------------------------------------------------------------- alignment
    def moving_file(self):
        """The file the moving page comes from: a separate moving file, else the open file."""
        if self.moving_source is not None:
            return self.moving_source
        return self.source

    def open_moving(self, path=None, page=0, *, align=False):
        """Use page ``page`` of ``path`` (None: the open file) as the moving image; it stays across reference pages."""
        if self.loaded is None or self.busy:
            return
        page = int(page)
        own = path is None or (self.source is not None and Path(path).resolve() == Path(self.source.path))
        if own and self.source is None:
            self.error.emit("This image has no other pages: open a moving file.")
            return
        known = self.moving_source
        if not own and known is not None and Path(path).resolve() != Path(known.path):
            known = None

        def read():
            source = self.source if own else known or PageSource(PageReader(path))
            return source, source.read(page)

        def accept(value):
            source, loaded = value
            if source is self.source:
                self._close_moving_source()
            elif source is not self.moving_source:
                self._close_moving_source()
                self.moving_source = source
            if loaded.image.shape != self.loaded.image.shape:
                self.moving_changed.emit()  # the file's pages are listed; this page does not fit
                (mh, mw), (h, w) = loaded.image.shape, self.loaded.image.shape
                raise ValueError(f"The moving page is {mw} x {mh} px; the reference page is {w} x {h} px.")
            self.moving, self.registration = loaded, None
            self.moving_changed.emit()
            self.registration_changed.emit()
            self.status.emit(f"Moving: {loaded.name}, page {loaded.page}", "info")
            if align:
                self.run_align()

        self._run(f"Reading moving page {page}", read, accept)

    def run_align(self):
        if self.loaded is None or self.busy:
            return
        problem = self.problems()[0]["register"]
        if problem:
            self.error.emit(problem)
            return
        reference, moving = self.loaded.image, self.moving.image
        factor = int(self.parameters.get("upsample", 20))
        mode = self.parameters.get("hole_check_mode", "centered")

        def accept(result):
            self.registration = result
            dy, dx = result.result.shift_yx
            limit = HOLE_CHECK_MAX_ERROR[result.hole_check_mode]
            passes = result.hole_check_error < limit
            self.log.emit(f"Aligned page {self.moving.page} of {self.moving.name} to page {self.loaded.page}: "
                          f"[dy, dx] = [{dy:.4f}, {dx:.4f}] px, RMSE {result.rmse_before:.4g} -> {result.rmse_after:.4g}, "
                          f"hole check ({result.hole_check_mode}) {result.hole_check_error:.3f}")
            if passes:
                self.status.emit(f"Aligned | dy {dy:.3f}, dx {dx:.3f} px", "info")
            else:
                self.status.emit(f"Aligned, but the hole check failed ({result.hole_check_error:.2f})", "warning")
            self.registration_changed.emit()

        self._run("Aligning", lambda: register_images(reference, moving, upsample_factor=factor,
                                                       hole_check_mode=mode), accept)

    # ----------------------------------------------------------------------------------------- export
    def export(self, folder, *, arrays=True, table=None, alignment=True, on_done=None):
        """Write source.json, the result arrays, the holes table as shown ((header, rows)) and the alignment."""
        if self.loaded is None or self.busy:
            return
        folder = Path(folder)
        results, review, registration = self.page_results, self.review, self.registration if alignment else None
        pitch = results.pitch if results is not None else None
        loaded, moving = self.loaded, self.moving
        source = dict(path=loaded.path, name=loaded.name, page=loaded.page, pages=loaded.pages,
                      display_mode="stack" if self.stacked else "page", shape=list(loaded.image.shape),
                      dtype=str(loaded.image.dtype), conversion=loaded.conversion, method=self.method,
                      moving_path=moving.path if moving else None, moving_page=moving.page if moving else None,
                      moving_conversion=moving.conversion if moving else None)

        def save():
            import csv

            folder.mkdir(parents=True, exist_ok=True)
            if arrays and review is not None:
                if pitch is not None and results.pitch_from is review:
                    export_review(pitch, folder)  # the same arrays plus the pitch profiles
                else:
                    export_review(review, folder)
                    if pitch is not None:
                        export_review(pitch, folder / "pitch")
            if table is not None:
                header, rows = table
                with (folder / "holes_table.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(header)
                    writer.writerows(rows)
            if registration is not None:
                np.savez_compressed(folder / "registration.npz", shift_yx=registration.result.shift_yx,
                                    aligned=registration.aligned, overlap=registration.overlap,
                                    correlation=registration.correlation,
                                    difference_before=registration.difference_before,
                                    difference_after=registration.difference_after)
                source["registration"] = dict(shift_yx=registration.result.shift_yx.tolist(),
                                              rmse_before=registration.rmse_before, rmse_after=registration.rmse_after,
                                              hole_check_error=registration.hole_check_error,
                                              hole_check_mode=registration.hole_check_mode,
                                              **registration.result.meta)
            (folder / "source.json").write_text(json.dumps(source, indent=2, default=str), encoding="utf-8")
            return folder

        def accept(path):
            self.log.emit(f"Exported to {path}")
            self.status.emit(f"Saved: {path}", "info")
            if on_done is not None:
                on_done(path)

        self._run("Exporting", save, accept)

    # ----------------------------------------------------------------------------------------- jobs
    def _run(self, label, function, on_result):
        return self.jobs.run(label, function, on_result)

    def cancel(self):
        self.jobs.cancel()

    def fail(self, context, detail):
        """A failed job or handler: the full text in the log, one plain line in the banner."""
        self.log.emit(detail)
        self.error.emit((f"{context}: " if context else "") + plain_error(detail))
        self.status.emit("Failed", "error")
