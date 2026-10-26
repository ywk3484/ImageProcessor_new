"""Measure every page of every TIFF of one dataset with the report's CD methods, on all GPUs: the average CD of each
page and method.

    python scripts/cd_batch.py /data/dataset1 --reference /data/reference.tif --output /results/dataset1 \
        --pixel-size 21 --unit nm

Inputs are TIFF files, folders (searched recursively for .tif / .tiff) or wildcards; every page of every file is
one image. A page is measured only when it holds holes: the error of its phase cross-correlation with the reference
(one page with holes: page --reference-page of the --reference TIFF, 0 = first) must be below --max-error (default
0.3). The error is scikit-image's ``phase_cross_correlation(reference, page, normalization=None)`` error,
whole-pixel: 0 when the page is the reference pattern at some shift, about 0.5 and more for other patterns
(scikit-image's default "phase" normalization gives ~1.0 for every pair, so it cannot tell them apart). Pages of
another size are compared on the centre part both share. --error-function FILE.py:NAME uses your own function
NAME(reference, image) -> error instead; --no-hole-check measures every page. The reference page is read once, at the
start; it may be a page of one of the dataset's own TIFFs.

Each page with holes is measured exactly as by scripts/cd_report.py (same methods, options and defaults; only the
display outlines, which the numbers do not depend on, are skipped).

Outputs in --output:
    pages.csv    one row per page: file (path below the common input folder), page (0 = first), status (ok or
                 no_holes), pcc_error, holes (cells measured), image_intensity (mean of all pixels, counts),
                 cell_intensity (mean over the holes of tail_area's mean light above the local level, counts), then
                 per method <method>_valid (holes it measured) and <method>_cd, <method>_cd_x, <method>_cd_y: the
                 mean CD, x width and y width over those holes, in --unit. No other statistics.
    timings.csv  one row per page: seconds of each stage, cell stack size, worker peak memory.
    failed.csv   pages that failed, with the error; the next run tries them again.
    run.json     the settings. Running again into the same --output resumes (pages already in pages.csv are
                 skipped); other settings are refused.

One reader thread reads the files one after another, each front to back, the way a disk reads fastest: no page is
read twice and the pages are not counted beforehand (on an HDD that alone walks the whole file). Pages already done,
or outside --pages, are skipped without reading their pixels. The reader hands each page to a free worker; every GPU
runs --workers-per-gpu workers (default 2), each measuring one page at a time with all of that page's holes batched
on its GPU, so one worker's CPU work overlaps another's GPU work. A page over --page-timeout, or a worker that dies,
is listed in failed.csv and the worker is restarted. Ctrl+C stops after the pages being measured; run the same command
again to continue.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass, field
import glob
import importlib.util
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cd_report import ALL_METHODS, _spread, cd_statistics, cell_intensity  # the report's numbers

SCHEMA = 1
PREFETCH = 2            # pages sent to a worker before its earlier ones are done
PROGRESS_SECONDS = 30
MAX_FAILURES_IN_A_ROW = 20
STAGES = ("segmentation", "deblend", "gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg",
          "tail_window", "tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio", "result")
TIMING_FIELDS = (["file", "page", "status", "gpu", "worker", "height", "width", "holes", "stack_h", "stack_w",
                  "window_px", "read_s", "check_s", "measure_s"] + [f"{s}_s" for s in STAGES] + ["peak_rss_mb"])
FAILED_FIELDS = ["file", "page", "error", "time"]


def page_fields(methods) -> list[str]:
    fields = ["file", "page", "status", "pcc_error", "holes", "image_intensity", "cell_intensity"]
    for name in methods:
        fields += [f"{name}_valid", f"{name}_cd", f"{name}_cd_x", f"{name}_cd_y"]
    return fields


# ---------------------------------------------------------------------------------------------------------- inputs
def find_tiffs(inputs) -> list[Path]:
    """TIFF files from files, folders (recursive) and wildcards, each once, sorted."""
    found = set()
    for item in inputs:
        item = str(item)
        matches = glob.glob(item, recursive=True) if glob.has_magic(item) else [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                found.update(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
            elif path.is_file() and path.suffix.lower() in (".tif", ".tiff"):
                found.add(path)
            elif not path.exists():
                raise SystemExit(f"Input not found: {item}")
    return sorted(p.resolve() for p in found)


def file_keys(files: list[Path]) -> list[str]:
    """Each file's path below the folder all inputs share (the ``file`` column; forward slashes)."""
    if len(files) == 1:
        return [files[0].name]
    root = Path(os.path.commonpath([str(p.parent) for p in files]))
    return [p.relative_to(root).as_posix() for p in files]


def parse_pages(text: str | None) -> slice:
    """'A:B' (0-based, B excluded, either may be empty) or 'A' (that page alone)."""
    if not text:
        return slice(None)
    try:
        if ":" not in text:
            page = int(text)
            return slice(page, page + 1)
        start, stop = text.split(":", 1)
        pages = slice(int(start) if start.strip() else None, int(stop) if stop.strip() else None)
    except ValueError:
        raise SystemExit(f"--pages must look like 0:20, 100: or 7, not {text!r}") from None
    if (pages.start or 0) < 0 or (pages.stop or 0) < 0:
        raise SystemExit(f"--pages counts from the first page (0): no negative numbers, not {text!r}")
    return pages


class FrameReader:
    """The 2-D frames of one TIFF: its pages, or the frames of an ImageJ stack over 4 GB that hides them behind a
    single page (read through a memory map)."""

    def __init__(self, path):
        import tifffile

        self.path = str(path)
        self._tif = tifffile.TiffFile(self.path)
        self._stack = None
        self.count = len(self._tif.pages)
        series = self._tif.series[0] if self._tif.series else None
        if (series is not None and self._tif.is_imagej and series.axes.endswith("YX") and len(series.shape) > 2
                and int(np.prod(series.shape[:-2])) > self.count):
            self._stack = tifffile.memmap(self.path, mode="r").reshape(-1, *series.shape[-2:])
            self.count = len(self._stack)

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < self.count:
            raise IndexError(f"page {index} is out of range ({self.count} pages)")
        image = np.array(self._stack[index]) if self._stack is not None else self._tif.pages[index].asarray()
        if image.ndim != 2:
            raise ValueError(f"page {index} is not a 2-D grayscale image (shape {image.shape})")
        return image

    def close(self) -> None:
        self._stack = None
        self._tif.close()


def iter_frames(path, pages: slice = slice(None), skip=()):
    """The 2-D frames of one TIFF in file order, as (index, image), without counting them first: the pages of a
    multi-page file read one after another, or the frames of an ImageJ stack (one contiguous block of data, as
    FrameReader reads it). Frames outside ``pages`` (non-negative start / stop) or in ``skip`` are passed over
    without reading their pixels. A frame that is not a 2-D grayscale image yields (index, ValueError)."""
    import tifffile

    start, stop = pages.start or 0, pages.stop
    with tifffile.TiffFile(str(path)) as tif:
        series = tif.series[0] if tif.is_imagej and tif.series else None
        stack = None
        if series is not None and series.axes.endswith("YX") and len(series.shape) > 2:
            try:
                stack = tifffile.memmap(str(path), mode="r").reshape(-1, *series.shape[-2:])
            except ValueError:  # not one contiguous block: read its pages
                stack = None
        if stack is not None:
            for index in range(len(stack))[pages]:
                if index not in skip:
                    yield index, np.array(stack[index])
            return
        for index, page in enumerate(tif.pages):
            if stop is not None and index >= stop:
                return
            if index < start or index in skip:
                continue
            image = page.asarray()
            yield index, (image if image.ndim == 2 else
                          ValueError(f"page {index} is not a 2-D grayscale image (shape {image.shape})"))


def read_image(path: Path, page: int = 0) -> np.ndarray:
    """One 2-D image: a TIFF page, or any image Pillow reads (page 0)."""
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        reader = FrameReader(path)
        try:
            return reader.read(page)
        finally:
            reader.close()
    from scripts.analyze_contact_holes import load_image
    return load_image(Path(path), page=page)


# ------------------------------------------------------------------------------------------------------ hole check
def _common_centre(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])

    def cut(x):
        y0, x0 = (x.shape[0] - h) // 2, (x.shape[1] - w) // 2
        return x[y0:y0 + h, x0:x0 + w]
    return cut(a), cut(b)


def pcc_error(reference, image, *, xp=np) -> float:
    """scikit-image's ``phase_cross_correlation(reference, image, normalization=None)`` error, whole-pixel:
    sqrt(|1 - max|cc|^2 / (sum reference^2 x sum image^2)|), cc the circular cross-correlation of the two images
    (the centre part they share when their sizes differ). 0: the image is the reference at some shift."""
    return PccCheck(reference, xp=xp)(image)


class PccCheck:
    """``pcc_error`` against one reference, its spectrum kept per image size (computed once per worker)."""

    def __init__(self, reference, *, xp=np):
        self.reference = np.asarray(reference, dtype=np.float64)
        self.xp = xp
        self._spectra = {}

    def __call__(self, image) -> float:
        xp = self.xp
        reference, image = _common_centre(self.reference, np.asarray(image))
        shape = image.shape
        if shape not in self._spectra:
            ref = xp.asarray(reference, dtype=xp.float64)
            self._spectra[shape] = (xp.fft.rfft2(ref), float((ref * ref).sum()))
        spectrum, reference_amp = self._spectra[shape]
        moving = xp.asarray(image, dtype=xp.float64)
        cc = xp.fft.irfft2(spectrum * xp.conj(xp.fft.rfft2(moving)), s=shape)
        peak = float(xp.abs(cc).max())
        amp = reference_amp * float((moving * moving).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sqrt(abs(1.0 - peak * peak / amp))) if amp > 0 else float("nan")


def load_error_function(spec: str):
    """``FILE.py:NAME`` -> the function NAME(reference, image) -> error defined in FILE.py."""
    path, _, name = spec.rpartition(":")
    if not path or not name or not Path(path).is_file():
        raise SystemExit(f"--error-function must be FILE.py:NAME with an existing file, not {spec!r}")
    module_spec = importlib.util.spec_from_file_location("cd_batch_error_function", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not callable(getattr(module, name, None)):
        raise SystemExit(f"{path} defines no function {name!r}")
    return getattr(module, name)


def make_check(settings: dict, xp):
    """The hole check of these settings: image -> error, or None (every page measured)."""
    check = settings["hole_check"]
    if check is None:
        return None
    reference = settings["reference_image"]  # read once by the main process
    if check["error_function"]:
        function = load_error_function(check["error_function"])
        return lambda image: float(function(reference, image))
    return PccCheck(reference, xp=xp)


# --------------------------------------------------------------------------------------------------------- measuring
def page_values(result, image, methods) -> dict:
    """The page's pages.csv values from its CDResult: the report's statistics, means only."""
    stats = cd_statistics(result)
    intensity = cell_intensity(result)
    values = {"holes": len(result.seeds_xy), "image_intensity": float(np.mean(image, dtype=np.float64)),
              "cell_intensity": _spread(intensity)["mean"] if intensity is not None else float("nan")}
    for name in methods:
        s = stats[name]
        values.update({f"{name}_valid": s["valid"], f"{name}_cd": s["mean"], f"{name}_cd_x": s["mean_x"],
                       f"{name}_cd_y": s["mean_y"]})
    return values


def measure_page(image: np.ndarray, file_key: str, page: int, settings: dict, check, device: int = 0,
                 read_s: float = float("nan")) -> tuple[dict, dict]:
    """Check and measure one page: (pages.csv row, timings.csv row)."""
    from subpx import measure_cds

    start = time.perf_counter()
    error = check(image) if check is not None else float("nan")
    check_s = time.perf_counter() - start
    row = {"file": file_key, "page": page, "status": "no_holes", "pcc_error": error,
           "image_intensity": float(np.mean(image, dtype=np.float64))}
    timing = {"file": file_key, "page": page, "status": "no_holes", "height": image.shape[0],
              "width": image.shape[1], "read_s": read_s, "check_s": check_s}
    if check is not None and not error < settings["hole_check"]["max_error"]:  # NaN: no holes either
        return row, timing
    result = measure_cds(image, backend=settings["backend"], device=device, **settings["measure"])
    row.update(page_values(result, image, settings["measure"]["methods"]), status="ok")
    seconds = result.meta.get("seconds", {})
    timing.update(status="ok", holes=len(result.seeds_xy), stack_h=result.masks.shape[1],
                  stack_w=result.masks.shape[2], window_px=result.meta.get("tail_window_px", ""),
                  measure_s=seconds.get("total", ""), **{f"{s}_s": seconds.get(s, "") for s in STAGES})
    return row, timing


def _peak_rss_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except ImportError:
        try:
            import psutil
            info = psutil.Process().memory_info()
            return getattr(info, "peak_wset", info.rss) / 2**20
        except Exception:
            return float("nan")


def _is_cuda_failure(error: BaseException) -> bool:
    """A CUDA error after which the worker's GPU context cannot be trusted (out of memory is not one)."""
    module = type(error).__module__ or ""
    return module.startswith(("cupy_backends.cuda", "cupy.cuda")) and type(error).__name__ != "OutOfMemoryError"


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".replace("\n", " ")[:2000]


def _worker(conn, pages, stop, settings: dict, wid: int, gpu: str) -> None:
    """Worker process: measure the pages it is sent on ``pages`` (a queue: the sender never waits for this process to
    read), one at a time, and send back their rows on ``conn``."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is the main process's: it lets running pages finish
    try:
        import cv2
        cv2.setNumThreads(settings["threads"])
    except Exception:
        pass
    try:
        xp = np
        if settings["backend"] == "gpu":
            import cupy
            from subpx._gpu import _cuda
            cupy.cuda.Device(0).use()  # CUDA_VISIBLE_DEVICES shows this worker one GPU
            int(cupy.arange(4).sum())
            _cuda.KEEP_MEMORY = True  # the next page reuses this page's GPU memory
            xp = cupy
        check = make_check(settings, xp)
    except BaseException as error:
        conn.send(("fatal", f"worker on GPU {gpu} could not start: {_describe(error)}"))
        return
    conn.send(("ready",))
    try:
        while True:
            task = pages.get()
            if task is None:
                return
            file_key, page, image, read_s = task
            if stop.is_set():
                continue  # stopping: pages not started go back unmeasured
            conn.send(("start", file_key, page))
            try:
                row, timing = measure_page(image, file_key, page, settings, check, read_s=read_s)
            except Exception as error:
                conn.send(("failed", file_key, page, _describe(error)))
                if _is_cuda_failure(error):
                    return  # the main process starts a fresh worker
                if settings["backend"] == "gpu":
                    import cupy
                    cupy.get_default_memory_pool().free_all_blocks()
                continue
            timing.update(gpu=gpu, worker=wid, peak_rss_mb=round(_peak_rss_mb(), 1))
            conn.send(("row", row, timing))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        return


# ------------------------------------------------------------------------------------------------------- bookkeeping
class CsvRows:
    """An append-only CSV, flushed after every row. Opening an existing one checks its columns and drops a last line
    cut off by a crash."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        if path.exists() and path.stat().st_size:
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            if header != fields:
                raise SystemExit(f"{path} has other columns than this run writes; use another --output")
            _drop_partial_line(path)
            self.handle = path.open("a", encoding="utf-8", newline="")
        else:
            self.handle = path.open("w", encoding="utf-8", newline="")
            csv.writer(self.handle).writerow(fields)
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, restval="", extrasaction="ignore")

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def rows(self) -> list[dict]:
        self.handle.flush()
        with self.path.open(encoding="utf-8", newline="") as handle:
            return [row for row in csv.DictReader(handle) if None not in row.values()]

    def keys(self) -> set[tuple[str, int]]:
        """(file, page) of every complete row (resume): the first two columns only."""
        self.handle.flush()
        with self.path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            return {(row[0], int(row[1])) for row in reader if len(row) == len(self.fields)}

    def close(self) -> None:
        self.handle.close()


def _drop_partial_line(path: Path) -> None:
    with path.open("rb+") as handle:
        size = handle.seek(0, os.SEEK_END)
        if not size:
            return
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return
        position = size
        while position > 0:  # back to the last complete line
            step = min(65536, position)
            handle.seek(position - step)
            block = handle.read(step)
            cut = block.rfind(b"\n")
            if cut >= 0:
                handle.truncate(position - step + cut + 1)
                return
            position -= step
        handle.truncate(0)


def _sorted_rewrite(path: Path, fields: list[str], order) -> None:
    """Rewrite a finished CSV sorted by (file, page) (temporary file, then replace); rows held as lists. A file another
    program holds open (Windows refuses to replace it; Excel keeps the files it opens) stays in the order measured."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = [row for row in reader if len(row) == len(fields)]
    rows.sort(key=order)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or fields)
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    except PermissionError:
        temporary.unlink()
        print(f"{path.name} is open in another program: left in the order measured (all rows are there)", flush=True)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def check_settings(output: Path, settings: dict) -> None:
    """Write run.json, or make sure a resumed run uses the settings of the earlier one."""
    path = output / "run.json"
    if path.exists():
        earlier = json.loads(path.read_text(encoding="utf-8"))
        before, now = earlier.get("settings", {}), json.loads(json.dumps(settings))
        if before != now:
            keys = sorted({k for k in set(before) | set(now) if before.get(k) != now.get(k)})
            raise SystemExit(f"{output} holds results of other settings (differ: {', '.join(keys)}); "
                             "use another --output or the earlier settings")
        return
    path.write_text(json.dumps({"schema": SCHEMA, "settings": settings, "git_commit": _git_commit(),
                                "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n", encoding="utf-8")


def visible_gpus() -> list[str]:
    """GPU ids for the workers, without touching CUDA here: CUDA_VISIBLE_DEVICES, else nvidia-smi."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [g.strip() for g in env.split(",") if g.strip() and g.strip() != "-1"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


# ------------------------------------------------------------------------------------------------------------ running
@dataclass
class Worker:
    wid: int
    gpu: str
    process: object
    conn: object
    pages: object                                   # multiprocessing queue of the pages sent to it
    inflight: deque = field(default_factory=deque)  # pages sent, not finished: (file_key, page, image, read_s)
    current: tuple | None = None                    # (file_key, page, start time) of the page being measured
    ready: bool = False


class Reader(threading.Thread):
    """Reads the dataset's pages in file order, each file front to back, into ``pages`` (bounded: the reader runs a
    few pages ahead of the workers): (file_key, page, image, read seconds), ("error", file_key, message) for a file
    that cannot be read, and None at the end."""

    def __init__(self, files, keys, selection: slice, done: dict, size: int):
        super().__init__(daemon=True, name="cd_batch-reader")
        self.files, self.keys, self.selection, self.done = files, keys, selection, done  # done: file -> pages
        self.pages = queue.Queue(maxsize=max(1, size))
        self.stop = threading.Event()
        self.total_bytes = sum(p.stat().st_size for p in files)
        self.files_started = 0
        self.bytes_behind = 0     # of the files already finished
        self.page_bytes = 0       # pixels of the current file read so far
        self.pages_read = 0

    def _put(self, item) -> bool:
        while not self.stop.is_set():
            try:
                self.pages.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def run(self) -> None:
        for key, path in zip(self.keys, self.files):
            self.files_started += 1
            self.page_bytes = 0
            try:
                started = time.perf_counter()
                for page, image in iter_frames(path, self.selection, self.done.get(key, ())):
                    seconds = time.perf_counter() - started
                    self.pages_read += 1
                    if isinstance(image, np.ndarray):
                        self.page_bytes += image.nbytes
                    if not self._put((key, page, image, seconds)):
                        return
                    started = time.perf_counter()
            except Exception as error:
                if not self._put(("error", key, f"cannot read the file: {_describe(error)}")):
                    return
            self.bytes_behind += path.stat().st_size
        self._put(None)

    def read_bytes(self) -> int:
        """Bytes of the dataset behind the reader (finished files, plus the current file's pixels read)."""
        return self.bytes_behind + self.page_bytes


class Run:
    """The main process: hands out pages, writes rows, watches the workers."""

    def __init__(self, args, settings: dict, files: list[Path], keys: list[str]):
        self.args, self.settings = args, settings
        self.output = args.output
        self.files, self.keys = files, keys
        self.methods = settings["measure"]["methods"]
        self.fields = page_fields(self.methods)
        self.pages = CsvRows(self.output / "pages.csv", self.fields)
        self.timings = CsvRows(self.output / "timings.csv", TIMING_FIELDS)
        self.failed = CsvRows(self.output / "failed.csv", FAILED_FIELDS)
        self.done = self.pages.keys()
        self.retry: deque = deque()   # pages taken back from a worker that was lost, measured again first
        self.reader_done = False
        self.counts = {"ok": 0, "no_holes": 0, "failed": 0}
        self.in_a_row = 0
        self.started = time.perf_counter()
        self.stopping = 0             # 1: finish running pages, 2: stop now
        self.fatal = None
        self.workers: list[Worker] = []

    # pages ------------------------------------------------------------------------------------------------------
    def write(self, row: dict, timing: dict) -> None:
        self.pages.write(row)
        self.timings.write(timing)
        self.done.add((row["file"], int(row["page"])))
        self.counts[row["status"]] += 1
        self.in_a_row = 0

    def fail(self, key: str, page: int, error: str) -> None:
        self.failed.write({"file": key, "page": page, "error": error, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.counts["failed"] += 1
        self.in_a_row += 1
        print(f"  failed: {key} page {page}: {error[:300]}", flush=True)
        if self.in_a_row >= MAX_FAILURES_IN_A_ROW and not self.stopping:
            print(f"Stopping: {self.in_a_row} pages failed in a row (see failed.csv)", flush=True)
            self.stopping = 1

    def done_by_file(self) -> dict[str, set[int]]:
        done = {}
        for key, page in self.done:
            done.setdefault(key, set()).add(page)
        return done

    def next_page(self):
        """The next page to measure, or None when none is ready (or all are read)."""
        if self.retry:
            return self.retry.popleft()
        while not self.reader_done:
            try:
                item = self.reader.pages.get_nowait()
            except queue.Empty:
                return None
            if item is None:
                self.reader_done = True
            elif item[0] == "error":
                self.fail(item[1], -1, item[2])
            elif isinstance(item[2], Exception):
                self.fail(item[0], item[1], _describe(item[2]))
            else:
                return item
        return None

    def progress(self) -> None:
        done = sum(self.counts.values())
        elapsed = time.perf_counter() - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        reader = self.reader
        read, total = reader.read_bytes(), reader.total_bytes
        speed = read / elapsed if elapsed > 0 else 0.0
        eta = time.strftime("%H:%M:%S", time.gmtime((total - read) / speed)) if speed > 0 and total > read else "-"
        print(f"[{time.strftime('%H:%M:%S')}] this run: {done} pages ({self.counts['ok']} ok, "
              f"{self.counts['no_holes']} no holes, {self.counts['failed']} failed) | {rate:.2f} pages/s | "
              f"file {reader.files_started} of {len(self.files)}, {read / 1e9:.1f} of {total / 1e9:.1f} GB read "
              f"({speed / 1e6:.0f} MB/s) | ETA {eta}", flush=True)

    # workers ----------------------------------------------------------------------------------------------------
    def start_worker(self, wid: int, gpu: str) -> Worker:
        import multiprocessing as mp

        context = mp.get_context("spawn")  # never fork a process that may hold a CUDA context
        parent, child = context.Pipe()
        pages = context.Queue()
        threads = str(self.settings["threads"])
        env = {"OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads, "MKL_NUM_THREADS": threads}
        if self.settings["backend"] == "gpu":  # the worker sees one GPU; ids as nvidia-smi numbers them
            env.update(CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER=os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID"))
        saved = {name: os.environ.get(name) for name in env}
        os.environ.update(env)  # a spawned process starts with the parent's environment at that moment
        try:
            process = context.Process(target=_worker, args=(child, pages, self.stop_event, self.settings, wid, gpu),
                                      daemon=True, name=f"cd_batch-{wid}")
            process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child.close()
        return Worker(wid, gpu, process, parent, pages)

    def lost(self, worker: Worker, reason: str) -> None:
        """A worker died or hung: its page failed, its other pages are measured again, a new worker starts."""
        if worker.current is not None:
            key, page, _ = worker.current
            self.fail(key, page, reason)
        for item in reversed(worker.inflight):
            if worker.current is None or item[:2] != worker.current[:2]:
                self.retry.appendleft(item)
        worker.inflight.clear()
        worker.current = None
        try:
            worker.process.kill()
        except Exception:
            pass
        worker.process.join(10)
        worker.conn.close()
        worker.pages.cancel_join_thread()  # pages it never read are dropped, not waited for
        worker.pages.close()
        index = self.workers.index(worker)
        if self.stopping or self.fatal:
            self.workers.pop(index)
        else:
            self.workers[index] = self.start_worker(worker.wid, worker.gpu)

    def handle(self, worker: Worker, message) -> None:
        kind = message[0]
        if kind == "ready":
            worker.ready = True
        elif kind == "fatal":
            self.fatal = message[1]
        elif kind == "start":
            _, key, page = message
            worker.current = (key, page, time.perf_counter())
        elif kind in ("row", "failed"):
            key, page = (message[1]["file"], message[1]["page"]) if kind == "row" else (message[1], message[2])
            worker.current = None
            for i, item in enumerate(worker.inflight):
                if item[0] == key and item[1] == page:
                    del worker.inflight[i]
                    break
            if kind == "row":
                self.write(message[1], message[2])
            else:
                self.fail(key, page, message[3])

    def run(self) -> int:
        import multiprocessing as mp
        from multiprocessing.connection import wait

        args = self.args
        backend = self.settings["backend"]
        if backend == "gpu":
            gpus = [str(g) for g in args.gpus] if args.gpus else visible_gpus()
            if not gpus:
                raise SystemExit("No GPU found (nvidia-smi / CUDA_VISIBLE_DEVICES); pass --gpus or --backend cpu")
            slots = [g for g in gpus for _ in range(args.workers_per_gpu)]
        else:
            gpus, slots = [], ["cpu"] * args.workers
        self.settings["threads"] = max(1, (os.cpu_count() or 1) // len(slots))  # CPU threads per worker
        print(f"Backend: {backend}; {len(slots)} workers" + (f" on GPUs {', '.join(gpus)}" if gpus else "")
              + f", {self.settings['threads']} CPU threads each", flush=True)
        self.stop_event = mp.get_context("spawn").Event()
        self.reader = Reader(self.files, self.keys, self.settings["pages"], self.done_by_file(), PREFETCH * len(slots))
        self.reader.start()
        self.workers = [self.start_worker(wid, gpu) for wid, gpu in enumerate(slots)]

        def interrupt(signum, frame):
            self.stopping += 1
            if self.stopping == 1:
                print("Stopping after the pages being measured (Ctrl+C again: stop now)...", flush=True)
            self.stop_event.set()
            self.reader.stop.set()

        previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        last_progress = time.perf_counter()
        try:
            while self.workers:
                if self.fatal:
                    break
                if not self.stopping:
                    for worker in self.workers:
                        while worker.ready and len(worker.inflight) < PREFETCH:
                            item = self.next_page()
                            if item is None:
                                break
                            worker.inflight.append(item)
                            worker.pages.put(item)
                    if self.reader_done and not self.retry and all(not w.inflight for w in self.workers):
                        break
                elif all(w.current is None for w in self.workers) or self.stopping > 1:
                    break
                try:
                    ready = wait([w.conn for w in self.workers], timeout=0.2)
                except InterruptedError:  # Ctrl+C during the wait (Windows does not retry it)
                    ready = []
                for worker in list(self.workers):
                    if worker.conn not in ready:
                        continue
                    try:
                        while worker.conn.poll():
                            self.handle(worker, worker.conn.recv())
                    except (EOFError, OSError):
                        worker.process.join(10)  # the pipe closes a moment before the process has exited
                        self.lost(worker, f"worker stopped (exit code {worker.process.exitcode})")
                now = time.perf_counter()
                for worker in list(self.workers):
                    if worker.current is not None and now - worker.current[2] > args.page_timeout:
                        self.lost(worker, f"timeout: no result after {args.page_timeout:g} s")
                    elif not worker.process.is_alive() and worker in self.workers:
                        try:  # collect what it sent before stopping
                            while worker.conn.poll():
                                self.handle(worker, worker.conn.recv())
                        except (EOFError, OSError):
                            pass
                        if worker.inflight or worker.current is not None or not self.stopping:
                            self.lost(worker, f"worker stopped (exit code {worker.process.exitcode})")
                if now - last_progress >= PROGRESS_SECONDS:
                    self.progress()
                    last_progress = now
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.stop_event.set()
            self.reader.stop.set()
            for worker in self.workers:
                try:
                    worker.pages.put(None)
                    worker.pages.cancel_join_thread()
                except Exception:
                    pass
            for worker in self.workers:
                worker.process.join(5)
                if worker.process.is_alive():
                    worker.process.kill()
            self.pages.close()
            self.timings.close()
            self.failed.close()
        if self.fatal:
            raise SystemExit(self.fatal)
        complete = not self.stopping and self.reader_done and not self.retry
        return self.finish(complete=complete)

    def run_here(self) -> int:
        """Everything in this process, one page after another (debugging and tests)."""
        backend = self.settings["backend"]
        xp = np
        if backend == "gpu":
            import cupy
            device = int(self.args.gpus[0]) if self.args.gpus else 0
            cupy.cuda.Device(device).use()
            xp = cupy
        else:
            device = 0
        check = make_check(self.settings, xp)
        done = self.done_by_file()
        try:
            for key, path in zip(self.keys, self.files):
                try:
                    started = time.perf_counter()
                    for page, image in iter_frames(path, self.settings["pages"], done.get(key, ())):
                        read_s = time.perf_counter() - started
                        try:
                            if isinstance(image, Exception):
                                raise image
                            row, timing = measure_page(image, key, page, self.settings, check, device, read_s)
                        except Exception as error:
                            self.fail(key, page, _describe(error))
                        else:
                            timing.update(gpu="", worker=0, peak_rss_mb=round(_peak_rss_mb(), 1))
                            self.write(row, timing)
                        started = time.perf_counter()
                except Exception as error:
                    self.fail(key, -1, f"cannot read the file: {_describe(error)}")
        finally:
            self.pages.close()
            self.timings.close()
            self.failed.close()
        return self.finish(complete=True)

    def finish(self, complete: bool) -> int:
        elapsed = time.perf_counter() - self.started
        done = sum(self.counts.values())
        print(f"This run: {done} pages in {elapsed:.0f} s ({self.counts['ok']} ok, {self.counts['no_holes']} no "
              f"holes, {self.counts['failed']} failed); results: {self.output / 'pages.csv'}", flush=True)
        if complete:
            order = lambda row: (row[0], int(row[1]))  # noqa: E731
            _sorted_rewrite(self.output / "pages.csv", self.fields, order)
            _sorted_rewrite(self.output / "timings.csv", TIMING_FIELDS, order)
        else:
            print("Not finished: run the same command again to continue.", flush=True)
        return 1 if self.counts["failed"] else 0


def build_settings(args) -> dict:
    """What decides the numbers (``measure``, ``hole_check``: checked on resume) and how to run (the rest)."""
    methods = [m for m in ALL_METHODS if m in args.methods]
    hole_check = reference_image = None
    if not args.no_hole_check:
        if args.reference is None:
            raise SystemExit("Give --reference (an image with holes) or --no-hole-check")
        reference = args.reference.resolve()
        if not reference.is_file():
            raise SystemExit(f"--reference not found: {reference}")
        try:  # before run.json is written, so that the corrected command is not refused as other settings
            reference_image = read_image(reference, args.reference_page)
        except Exception as error:
            raise SystemExit(f"--reference-page {args.reference_page} of {reference}: {_describe(error)}") from None
        stat = reference.stat()
        hole_check = {"reference": str(reference), "reference_page": args.reference_page,
                      "reference_bytes": stat.st_size, "reference_mtime": int(stat.st_mtime),
                      "max_error": args.max_error,
                      "error_function": str(Path(args.error_function.rpartition(":")[0]).resolve()) + ":"
                      + args.error_function.rpartition(":")[2] if args.error_function else None}
    pages = parse_pages(args.pages)
    return {
        "measure": {"methods": methods, "seeds": args.seeds, "invert": args.invert, "area_min": args.area_min,
                    "area_max": args.area_max, "pixel_size": args.pixel_size, "unit": args.unit,
                    "half_area_level": args.half_level, "half_area_peak_pixels": args.peak_pixels,
                    "tail_area_level": args.tail_level, "tail_window": args.tail_window,
                    "n_angles": 16, "outlines": False},  # outlines only draw maps; the numbers do not use them
        "hole_check": hole_check,
        "reference_image": reference_image,  # the reference page itself, sent to the workers (not in run.json)
        "pages": pages,
        "backend": args.backend,
        "threads": 1,
    }


def argument_parser() -> argparse.ArgumentParser:
    """main's command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="TIFF files, folders (searched recursively) or wildcards")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="The TIFF (or other image) holding the reference: one page "
                        "with holes, the pattern pages must match")
    parser.add_argument("--reference-page", type=int, default=0, help="The reference's page in that TIFF (0 = first)")
    parser.add_argument("--max-error", type=float, default=0.3,
                        help="Pages whose cross-correlation error with the reference is at least this have no holes")
    parser.add_argument("--error-function", help="FILE.py:NAME, your own NAME(reference, image) -> error")
    parser.add_argument("--no-hole-check", action="store_true", help="Measure every page")
    parser.add_argument("--pages", help="Pages of every file, e.g. 0:20 (first 20) or 100: (0 = first page)")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--half-level", type=float, default=0.5)
    parser.add_argument("--peak-pixels", type=float, default=4.0)
    parser.add_argument("--tail-level", type=float, default=0.2)
    parser.add_argument("--tail-window", type=float, default=1.2)
    parser.add_argument("--pixel-size", type=float, default=1.0)
    parser.add_argument("--unit", default="px")
    parser.add_argument("--seeds", choices=("threshold", "peaks"), default="threshold")
    parser.add_argument("--area-min", type=int, default=3)
    parser.add_argument("--area-max", type=int, default=200)
    parser.add_argument("--invert", action="store_true", help="Dark holes on a bright background")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpus", nargs="+", help="GPU ids (default: all, as nvidia-smi numbers them)")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Workers with --backend cpu")
    parser.add_argument("--page-timeout", type=float, default=600, help="Seconds before a page counts as hung")
    parser.add_argument("--in-process", action="store_true", help="No worker processes (debugging)")
    return parser


def main(argv=None) -> int:
    args = argument_parser().parse_args(argv)
    if args.workers_per_gpu < 1 or args.workers < 1:
        raise SystemExit("--workers-per-gpu and --workers must be at least 1")
    if args.pixel_size != 1 and args.unit == "px":
        raise SystemExit("give --unit (e.g. nm) with --pixel-size")
    settings = build_settings(args)
    files = find_tiffs(args.inputs)
    if not files:
        raise SystemExit("No .tif / .tiff files in the inputs")
    args.output.mkdir(parents=True, exist_ok=True)
    check_settings(args.output, {"measure": settings["measure"], "hole_check": settings["hole_check"]})
    run = Run(args, settings, files, file_keys(files))
    if settings["hole_check"] is not None:
        height, width = settings["reference_image"].shape
        reference = settings["hole_check"]["reference"]
        print(f"Reference: page {args.reference_page} of {reference} ({width} x {height} px)", flush=True)
    print(f"{len(files)} TIFF files; {len(run.done)} pages already done in {args.output}", flush=True)
    return run.run_here() if args.in_process else run.run()


if __name__ == "__main__":
    raise SystemExit(main())
