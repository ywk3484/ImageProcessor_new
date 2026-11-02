"""Measure CD and pitch on every page of a multipage file: the workbench's Measure all pages (Qt-free).

One reader thread reads the chosen pages front to back, one pass through the file as ``scripts/cd_batch.py`` reads,
and hands them to worker processes: one per GPU with two pages in flight (one page's CPU steps overlap the other's
GPU work: 2.8 instead of 3.3 s per 39,000-hole page on an RTX 4060 Ti, the same numbers), or one worker on the CPU.
A worker checks the page for holes against the reference page (the batch's hole check), measures CD in the batch's
fast mode (``measure_cds(summary_only=True)``: all holes of the page at once on the GPU, the same CDs as Measure CD
without contours or pixel stacks), the pitch of every method and of the ticked GPU center finders, writes the page's
hole CSV and sends back one row. The window calls :meth:`PagesRun.step` from a timer and never waits.

Output folder:
    pages.csv   one row per page, in page order: scripts/cd_batch.py's columns with the same values (file, page,
                status, pcc_error, holes, image_intensity, cell_intensity, deblend_converged, deblend_iterations,
                deblend_passes, per method <m>_valid, <m>_cd, <m>_cd_x, <m>_cd_y in the unit), then per pitch method
                (the CD methods and the GPU center finders, gpu_<finder>) <m>_pitch_x, <m>_pitch_y,
                <m>_placement_x_min, <m>_placement_x_max, <m>_placement_y_min, <m>_placement_y_max (unit), then
                pitch_error, error and seconds. status: ok, no_holes (the hole check failed) or failed (error says why).
    holes/page_0007.csv  every hole of that page: hole, x, y (page pixels), per method <m>_cd, <m>_cd_x, <m>_cd_y
                (unit; empty where the method failed) and <m>_status.
    run.json    the file, pages, settings, hole check, compute devices, times and counts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from math import isfinite
import os
from pathlib import Path
import queue
import signal
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np

from ._workbench_data import (HOLE_CHECK_MAX_ERROR, cd_options, crop_image, gpu_center_measurements,
                              hole_check_error, pitch_numbers, preprocess_page)

PAGES_PER_GPU = 2       # pages measured at once by a GPU's worker (two threads)
PREFETCH = 1            # pages waiting at a worker beyond those it measures
PAGE_TIMEOUT = 600.0    # seconds before a page counts as hung (its worker is restarted)
MAX_FAILED_IN_A_ROW = 20
CD_AXES = (("", "cd_equivalent_px"), ("_x", "cd_x_px"), ("_y", "cd_y_px"))
PITCH_KEYS = ("pitch_x", "pitch_y", "placement_x_min", "placement_x_max", "placement_y_min", "placement_y_max")
NAN = float("nan")
_GPUS = None


# ------------------------------------------------------------------------------------------------- devices
def gpu_list():
    """[(id, name)] of the GPUs workers can use, without starting CUDA here: CUDA_VISIBLE_DEVICES, else nvidia-smi,
    else CuPy's device count. Read once."""
    global _GPUS
    if _GPUS is not None:
        return _GPUS
    found = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        found = [(item.strip(), "") for item in visible.split(",") if item.strip() and item.strip() != "-1"]
    else:
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window flashing up on Windows
            out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30, creationflags=flags)
            for line in out.stdout.splitlines():
                index, _, name = line.partition(",")
                if index.strip().isdigit():
                    found.append((index.strip(), name.strip()))
        except Exception:
            pass
        if not found:
            try:
                import cupy

                found = [(str(i), "") for i in range(cupy.cuda.runtime.getDeviceCount())]
            except Exception:
                found = []
    _GPUS = found
    return found


def devices(parameters):
    """('gpu', [(id, name)]) or ('cpu', []) for the Compute setting (auto: the GPUs when CuPy works)."""
    backend = parameters.get("backend", "auto")
    if backend == "cpu":
        return "cpu", []
    from .backends import is_gpu_available

    gpus = gpu_list() if is_gpu_available() else []
    if gpus:
        return "gpu", gpus
    if backend == "gpu":
        raise ValueError("No GPU found (CuPy, nvidia-smi): set Compute to CPU or Auto.")
    return "cpu", []


# ---------------------------------------------------------------------------------------------- one page
def page_fields(methods, pitch_methods):
    fields = ["file", "page", "status", "pcc_error", "holes", "image_intensity", "cell_intensity",
              "deblend_converged", "deblend_iterations", "deblend_passes"]
    for name in methods:
        fields += [f"{name}_valid", f"{name}_cd", f"{name}_cd_x", f"{name}_cd_y"]
    for name in pitch_methods:
        fields += [f"{name}_{key}" for key in PITCH_KEYS]
    return fields + ["pitch_error", "error", "seconds"]


def _mean(values):
    values = values[np.isfinite(values)]
    return (float(values.mean()) if values.size else NAN), int(values.size)


def page_values(summary, image, methods):
    """The page's cd_batch values (scripts/cd_batch.py ``page_values``; a test pins them equal): holes, mean pixel,
    cell intensity, deblend state and per method the holes measured and the mean CD, x and y width in the unit."""
    values = {"holes": len(summary.seeds_xy), "image_intensity": float(np.mean(image, dtype=np.float64)),
              "cell_intensity": NAN, "deblend_converged": summary.meta.get("deblend_converged", ""),
              "deblend_iterations": summary.meta.get("deblend_iterations_used", ""),
              "deblend_passes": summary.meta.get("deblend_passes", "")}
    tail = summary.measurements.get("tail_area")
    if tail is not None:  # the batch's cell intensity: tail_area's mean light above its local level
        values["cell_intensity"] = _mean(np.where(tail.valid, tail.metrics["mean_signal"], np.nan))[0]
    for name in methods:
        measured = summary.measurements[name]
        for suffix, metric in CD_AXES:
            mean, count = _mean(measured.metrics[metric][measured.valid] * summary.pixel_size)
            values[f"{name}_cd{suffix}"] = mean
            if not suffix:
                values[f"{name}_valid"] = count
    return values


def pitch_values(summary, image, parameters, finders=()):
    """({<method>_pitch_x, ..., <method>_placement_y_max}, notes): the pitch of every method of the page and of the
    GPU center finders on the same holes, with the Pitch settings and the default reference hole."""
    from .pattern import analyze_pitch
    from .types import CDSummary

    extra, notes = {}, []
    if finders:
        holes = SimpleNamespace(seeds_xy=summary.seeds_xy, image=image, parameters=parameters)
        extra, notes = gpu_center_measurements(holes, finders)
    source = CDSummary(summary.seeds_xy, {**summary.measurements, **extra}, summary.pixel_size, summary.unit)
    patterns = analyze_pitch(source, **dict(parameters["pitch_options"], reference_xy=None))
    values = {}
    for name, pattern in patterns.items():
        for axis in ("x", "y"):
            mean, _, low, high = pitch_numbers(pattern.profiles[axis])
            values.update({f"{name}_pitch_{axis}": mean, f"{name}_placement_{axis}_min": low,
                           f"{name}_placement_{axis}_max": high})
    return values, notes


def write_holes(path, summary, origin_xy, methods):
    """One page's hole CSV: hole, x, y in page pixels, per method its CD, x and y width (unit) and status."""
    seeds = np.asarray(summary.seeds_xy, float).reshape(-1, 2) + np.asarray(origin_xy, float)

    def text(values):  # 7 significant digits, empty where not finite (Python lists: twice as fast as np.char)
        return ["" if not isfinite(value) else f"{value:.7g}" for value in np.asarray(values, float).tolist()]

    header, columns = ["hole", "x", "y"], [range(len(seeds)), text(seeds[:, 0]), text(seeds[:, 1])]
    for name in methods:
        measured = summary.measurements[name]
        for suffix, metric in CD_AXES:
            header.append(f"{name}_cd{suffix}")
            columns.append(text(np.where(measured.valid, measured.metrics[metric] * summary.pixel_size, np.nan)))
        header.append(f"{name}_status")
        columns.append(np.asarray(measured.status).astype(str).tolist())
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(zip(*columns))


def measure_page(image, page, settings):
    """Check and measure one page: its pages.csv row (holes CSV written)."""
    from .cd import measure_cds

    started = time.perf_counter()
    row = {"file": settings["file"], "page": page, "status": "no_holes", "pcc_error": NAN,
           "image_intensity": float(np.mean(image, dtype=np.float64))}
    reference = settings["reference"]
    if reference is not None:
        row["pcc_error"] = hole_check_error(reference, image, settings["check_mode"])
        if not row["pcc_error"] < settings["max_error"]:  # NaN: no holes either
            row["seconds"] = time.perf_counter() - started
            return row
    parameters = settings["parameters"]
    shown = preprocess_page(image, parameters).image if parameters.get("preprocess") else image
    shown, origin = crop_image(shown, parameters.get("bounds"))
    options = dict(cd_options(parameters), backend=settings["backend"])
    summary = measure_cds(shown, **options, outlines=False, summary_only=True)
    row.update(page_values(summary, image, settings["methods"]), status="ok")
    try:
        values, notes = pitch_values(summary, shown, parameters, settings["finders"])
        row.update(values)
        skipped = [note for note in notes if " skipped: " in note]
        if skipped:
            row["pitch_error"] = "; ".join(skipped)
    except ValueError as error:  # lattice indexing failed: the CD stays
        row["pitch_error"] = str(error)
    write_holes(Path(settings["holes"]) / f"page_{page:04d}.csv", summary, origin, settings["methods"])
    row["seconds"] = time.perf_counter() - started
    return row


def describe(error):
    return f"{type(error).__name__}: {error}".replace("\n", " ")[:2000]


def _is_cuda_failure(error):
    """A CUDA error after which the worker's GPU context cannot be trusted (out of memory is not one)."""
    module = type(error).__module__ or ""
    return module.startswith(("cupy_backends.cuda", "cupy.cuda")) and type(error).__name__ != "OutOfMemoryError"


def _worker(conn, tasks):
    """Worker process: take the settings, then measure the pages sent on ``tasks``, ``settings['threads']`` at once,
    and send back ('ready',), ('start', page), ('row', row), ('failed', page, error) or ('fatal', error).

    The settings (with the reference page) come through the queue, not as process arguments: on Windows the
    arguments are written into the new process's start-up pipe, which waits until the child has imported its modules
    (1.6 s with the window frozen); the queue's feeder thread writes them without making the window wait."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    lock = threading.Lock()
    settings = tasks.get()

    def send(*message):
        with lock:
            conn.send(message)

    gpu = settings["backend"] == "gpu"
    try:
        if gpu:
            import cupy
            from ._gpu import _cuda

            cupy.cuda.Device(0).use()  # CUDA_VISIBLE_DEVICES shows this worker one GPU
            int(cupy.arange(4).sum())
            _cuda.KEEP_MEMORY = True  # the next page reuses this page's GPU memory
    except BaseException as error:
        send("fatal", f"the worker could not start the GPU: {describe(error)}")
        return
    send("ready")

    def loop():
        if gpu:
            cupy.cuda.Device(0).use()
        while True:
            task = tasks.get()
            if task is None:
                return
            page, image = task
            send("start", page)
            try:
                row = measure_page(image, page, settings)
            except Exception as error:
                send("failed", page, describe(error))
                if _is_cuda_failure(error):
                    os._exit(3)  # the window starts a fresh worker
                continue
            send("row", row)

    threads = [threading.Thread(target=loop, daemon=True) for _ in range(settings["threads"])]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            thread.join()
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return


# ------------------------------------------------------------------------------------------------- the run
class _Reader(threading.Thread):
    """Reads the pages in file order into ``pages`` (bounded): (page, image or exception), None at the end."""

    def __init__(self, path, pages, size):
        super().__init__(daemon=True, name="all-pages-reader")
        self.path, self.wanted = path, sorted(pages)
        self.pages = queue.Queue(maxsize=max(1, size))
        self.stop = threading.Event()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.pages.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        from ._workbench_data import PageReader

        try:
            reader = PageReader(self.path)
        except Exception as error:
            for page in self.wanted:
                if not self._put((page, error)):
                    return
            self._put(None)
            return
        try:
            for page in self.wanted:
                try:
                    item = reader.read(page).image
                except Exception as error:
                    item = error
                if not self._put((page, item)):
                    return
        finally:
            reader.close()
        self._put(None)


@dataclass
class _Worker:
    wid: int
    gpu: str
    process: object
    conn: object
    tasks: object
    inflight: dict = field(default_factory=dict)  # page -> image: sent, not finished
    started: dict = field(default_factory=dict)  # page -> start time, of the pages being measured
    ready: bool = False


class PagesRun:
    """Measure CD and pitch on ``pages`` of the file at ``path`` into the folder ``output``.

    ``parameters`` are the workbench's analysis parameters (CD, Holes, Pitch settings; ``bounds`` the analysis region;
    ``preprocess`` set to preprocess every page first). ``reference`` (an image) turns on the hole check: a page whose
    error is not below ``HOLE_CHECK_MAX_ERROR[check_mode]`` is listed as no_holes. Call :meth:`start`, then
    :meth:`step` until :attr:`finished`; :meth:`stop` ends the run early (pages being measured are dropped).
    """

    def __init__(self, path, pages, parameters, output, *, reference=None, reference_page=None,
                 check_mode="centered", page_timeout=PAGE_TIMEOUT):
        self.path = str(Path(path).resolve())
        self.name = Path(path).name
        self.pages = sorted({int(page) for page in pages})
        if not self.pages:
            raise ValueError("No pages to measure.")
        self.parameters = dict(parameters)
        if not self.parameters.get("methods"):
            raise ValueError("Tick at least one CD method.")
        self.output = Path(output)
        self.backend, self.gpus = devices(self.parameters)
        self.methods = list(self.parameters["methods"])
        self.finders = tuple(self.parameters.get("pitch_finders") or ()) if self.backend == "gpu" else ()
        self.pitch_methods = self.methods + [f"gpu_{finder}" for finder in self.finders]
        self.fields = page_fields(self.methods, self.pitch_methods)
        self.reference = None if reference is None else np.ascontiguousarray(reference)
        self.reference_page = reference_page
        self.check_mode = check_mode
        self.max_error = HOLE_CHECK_MAX_ERROR[check_mode]
        self.page_timeout = float(page_timeout)
        self.threads = PAGES_PER_GPU if self.backend == "gpu" else 1
        self.rows = {}  # page -> row, as they come
        self.counts = {"ok": 0, "no_holes": 0, "failed": 0}
        self.state = "new"  # running, done, stopped or failed
        self.fatal = ""
        self.workers = []
        self._retry = []
        self._reader_done = False
        self._in_a_row = 0
        self._wid = 0
        self.started = self.finished_at = self.first_row = None

    # ----------------------------------------------------------------------------------------- state
    @property
    def running(self):
        return self.state == "running"

    @property
    def finished(self):
        return self.state in ("done", "stopped", "failed")

    def elapsed(self):
        if self.started is None:
            return 0.0
        return (self.finished_at or time.perf_counter()) - self.started

    def progress(self):
        """(pages done, pages, pages per second, seconds left): the rate from the first finished page on, so the
        workers' start-up is not counted; NaN until two pages are done."""
        done, total = len(self.rows), len(self.pages)
        now = self.finished_at or time.perf_counter()
        rate = (done - 1) / (now - self.first_row) if done >= 2 and now > self.first_row else NAN
        left = (total - done) / rate if rate > 0 else NAN
        return done, total, rate, left

    def describe_devices(self):
        if self.backend == "cpu":
            return "CPU, one page at a time"
        names = ", ".join(f"{gpu} {name}".strip() for gpu, name in self.gpus)
        return (f"{len(self.gpus)} GPU{'s' * (len(self.gpus) != 1)} ({names}), {self.threads} pages at a time on "
                "each")

    # ---------------------------------------------------------------------------------------- running
    def start(self):
        """Write run.json and the CSV header, start the reader and the workers (returns at once)."""
        if self.state != "new":
            return
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "pages.csv").exists():
            raise ValueError(f"{self.output} already holds a pages.csv: pick another folder.")
        self.holes = self.output / "holes"
        self.holes.mkdir(exist_ok=True)
        self._csv = (self.output / "pages.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=self.fields, restval="", extrasaction="ignore")
        self._writer.writeheader()
        self._csv.flush()
        self.started = time.perf_counter()
        self._clock = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_manifest()
        slots = len(self.gpus) if self.backend == "gpu" else 1
        self.reader = _Reader(self.path, self.pages, size=2 * slots)
        self.reader.start()
        self.state = "running"
        self.workers = [self._start_worker(gpu) for gpu, _ in (self.gpus or [("cpu", "")])]

    def _settings(self):
        parameters = dict(self.parameters, backend=self.backend)
        return {"file": self.name, "parameters": parameters, "backend": self.backend, "methods": self.methods,
                "finders": self.finders, "reference": self.reference, "check_mode": self.check_mode,
                "max_error": self.max_error, "holes": str(self.holes), "threads": self.threads}

    def _start_worker(self, gpu):
        import multiprocessing as mp

        context = mp.get_context("spawn")  # never fork a process that may hold a CUDA context
        parent, child = context.Pipe()
        tasks = context.Queue()
        env = {}
        if self.backend == "gpu":  # the worker sees one GPU, numbered as nvidia-smi numbers them
            env = {"CUDA_VISIBLE_DEVICES": gpu,
                   "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")}
        saved = {name: os.environ.get(name) for name in env}
        os.environ.update(env)  # a spawned process starts with the parent's environment at that moment
        try:
            process = context.Process(target=_worker, args=(child, tasks), daemon=True, name=f"all-pages-{self._wid}")
            process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child.close()
        tasks.put(self._settings())  # the first item: see _worker
        self._wid += 1
        return _Worker(self._wid - 1, gpu, process, parent, tasks)

    def step(self):
        """Collect finished pages, hand out new ones, watch the workers; never waits. Returns events:
        ('row', row) per finished page, ('log', text), ('done', state)."""
        events = []
        if not self.running:
            return events
        now = time.perf_counter()
        for worker in list(self.workers):
            try:
                while worker.conn.poll():
                    self._handle(worker, worker.conn.recv(), events)
            except (EOFError, OSError):
                worker.process.join(2)  # the pipe closes a moment before the process has exited
                self._lost(worker, f"worker stopped (exit code {worker.process.exitcode})", events)
                continue
            if not worker.process.is_alive():
                self._lost(worker, f"worker stopped (exit code {worker.process.exitcode})", events)
            elif any(now - begun > self.page_timeout for begun in worker.started.values()):
                self._lost(worker, f"timeout: no result after {self.page_timeout:g} s", events)
            if not self.running:
                break
        if self.fatal and self.running:
            self._finish("failed", events)
            return events
        if self.running:
            self._dispatch(events)
            if self._reader_done and not self._retry and all(not w.inflight for w in self.workers):
                self._finish("done", events)
        return events

    def stop(self):
        """End the run now: the pages being measured are dropped, the rows so far stay."""
        if self.running:
            self._finish("stopped", [])

    def _next(self, events):
        if self._retry:
            return self._retry.pop(0)
        while not self._reader_done:
            try:
                item = self.reader.pages.get_nowait()
            except queue.Empty:
                return None
            if item is None:
                self._reader_done = True
            elif isinstance(item[1], Exception):
                self._failed(item[0], f"cannot read the page: {describe(item[1])}", events)
            else:
                return item
        return None

    def _dispatch(self, events):
        for worker in self.workers:
            while worker.ready and len(worker.inflight) < self.threads + PREFETCH:
                item = self._next(events)
                if item is None:
                    return
                worker.inflight[item[0]] = item[1]
                worker.tasks.put(item)

    def _handle(self, worker, message, events):
        kind = message[0]
        if kind == "ready":
            worker.ready = True
        elif kind == "fatal":
            self.fatal = message[1]
        elif kind == "start":
            worker.started[message[1]] = time.perf_counter()
        elif kind == "row":
            row = message[1]
            worker.inflight.pop(row["page"], None)
            worker.started.pop(row["page"], None)
            self._record(row, events)
            self._in_a_row = 0
        elif kind == "failed":
            worker.inflight.pop(message[1], None)
            worker.started.pop(message[1], None)
            self._failed(message[1], message[2], events)

    def _record(self, row, events):
        if self.first_row is None:
            self.first_row = time.perf_counter()
        self.rows[row["page"]] = row
        self.counts[row["status"]] += 1
        self._writer.writerow(row)
        self._csv.flush()
        events.append(("row", row))

    def _failed(self, page, error, events):
        self._record({"file": self.name, "page": page, "status": "failed", "error": error}, events)
        events.append(("log", f"Page {page} failed: {error}"))
        self._in_a_row += 1
        if self._in_a_row >= MAX_FAILED_IN_A_ROW and not self.fatal:
            self.fatal = f"{self._in_a_row} pages failed in a row (see pages.csv); the run stopped."

    def _lost(self, worker, reason, events):
        """A worker died or hung: the pages it was measuring failed, the others go to the next worker."""
        if not worker.ready and not worker.inflight and not self.fatal:
            self.fatal = f"the worker could not start: {reason}"
        for page in list(worker.started):
            worker.inflight.pop(page, None)
            self._failed(page, reason, events)
        self._retry = sorted(worker.inflight.items()) + self._retry
        self._close_worker(worker, kill=True)
        index = self.workers.index(worker)
        if self.running and not self.fatal:
            self.workers[index] = self._start_worker(worker.gpu)
            events.append(("log", f"Worker on {self.backend.upper()} {worker.gpu} restarted ({reason})."))
        else:
            self.workers.pop(index)

    @staticmethod
    def _close_worker(worker, *, kill):
        """Kill the worker, or tell it to stop after its pages; the wait for it to exit (0.4 s) happens on a
        background thread, never in the window."""
        try:
            if kill:
                worker.process.kill()
            else:
                for _ in range(8):
                    worker.tasks.put(None)
        except Exception:
            pass

        def reap():
            worker.process.join(0.5 if kill else 10)
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(2)
            worker.conn.close()
            worker.tasks.cancel_join_thread()  # pages it never read are dropped, not waited for
            worker.tasks.close()

        threading.Thread(target=reap, daemon=True, name="all-pages-reap").start()

    def _finish(self, state, events):
        self.state = state
        self.finished_at = time.perf_counter()
        self.reader.stop.set()
        for worker in self.workers:
            self._close_worker(worker, kill=state != "done")
        self.workers = []
        self._csv.close()
        self._sort_rows()
        self._write_manifest()
        events.append(("done", state))

    def _sort_rows(self):
        """pages.csv in page order (a file another program holds open stays in the order measured)."""
        path = self.output / "pages.csv"
        temporary = path.with_name("pages.csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields, restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows[page] for page in sorted(self.rows))
        try:
            os.replace(temporary, path)
        except PermissionError:
            temporary.unlink()

    def _write_manifest(self):
        manifest = {
            "file": self.path, "pages": [self.pages[0], self.pages[-1]], "page_count": len(self.pages),
            "settings": self.parameters,
            "hole_check": None if self.reference is None else {
                "reference_page": self.reference_page, "mode": self.check_mode, "max_error": self.max_error},
            "compute": {"backend": self.backend, "gpus": [{"id": gpu, "name": name} for gpu, name in self.gpus],
                        "pages_at_a_time_per_worker": self.threads},
            "measurement": "measure_cds(summary_only=True, outlines=False): the same CDs as the workbench's Measure CD",
            "state": self.state, "started": self._clock, "elapsed_s": round(self.elapsed(), 2), "counts": self.counts,
            "error": self.fatal or None, "git_commit": _git_commit(),
        }
        (self.output / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")


def _git_commit():
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, timeout=10, creationflags=flags)
        return out.stdout.strip() or None
    except Exception:
        return None


def run_pages(path, pages, parameters, output, *, poll=0.05, **options):
    """Run :class:`PagesRun` to the end in this thread (scripts and tests); returns the run."""
    run = PagesRun(path, pages, parameters, output, **options)
    run.start()
    while not run.finished:
        run.step()
        time.sleep(poll)
    if run.state == "failed":
        raise RuntimeError(run.fatal)
    return run
