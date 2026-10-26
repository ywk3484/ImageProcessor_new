"""Why CD batch pages take as long as they do on this machine, and whether its numbers are right: one command, a few
minutes, then three short lines to type back. With --part 2 it runs the follow-up instead (scripts/cd_diagnose_part2.py:
the GPUs and Windows, workers per GPU, deblend drift): about 20 minutes, then six lines.

    python scripts/cd_diagnose.py D:\\results\\trial_w1 D:\\data\\dataset1\\stripe_001\\scan_001.tif

RUN_FOLDER is the --output folder of a scripts/cd_batch.py run that measured TIFF: its run.json gives the measuring
settings, its timings.csv the run's median and slowest page with holes (--pages A,B names two pages instead).
Nothing is changed; everything measured is also written to RUN_FOLDER/diagnosis.txt.

Every GPU measurement runs in worker processes started exactly as cd_batch starts its workers. In order: this
machine (code, settings, library versions, Windows, GPUs, CPU); a worker alone on the first GPU (GPU round trip,
the median page, a fixed synthetic page whose numbers must equal those of the PC this script was written on); a
second worker on the same GPU (what two workers per GPU do); a worker on every GPU at once (as the batch runs); a
worker that times every deblend step of the median and the slowest page and checks the numbers.

The three lines (labels and letters in values do not look like digits; "-" = could not be measured):
  v code fingerprint / git commit (+N: N CD files differ from it) / the run's commit   n debug settings in the
    environment: how many, then the slow ones that are on (! CUDA_LAUNCH_BLOCKING, j no CUDA kernel cache or forced
    JIT, d CuPy debug kernels, a CuPy accelerators, p Python debug)   p libraries of another version than on the
    reference PC (n numpy, p scipy, c cupy, v opencv, t tifffile, y python, r CUDA runtime; =: none)
  w Windows: power plan (a balanced, h high performance, u ultimate, p power saver), GPU scheduling (1 on, 0 off),
    virtual machine (v) or not (n), memory integrity (1 on, 0 off)
  u GPUs and driver model (w WDDM, t TCC; v: virtual GPU) / different GPUs the workers saw / busiest GPU before, %
  c CPU benchmark ms alone / with every worker busy / logical CPUs   a us per GPU call and wait: alone / with a
    second worker on the GPU
  x synthetic page s / deblend rounds / = when its image and numbers equal the reference PC's (d: other image,
    n: other numbers)   f a new worker's first page, s
  t median page s: one worker / with a second worker on its GPU / with a worker on every GPU / the run's own
  m deblend / cell methods / wide methods / rest, s   h GPU busy % / SM clock % of its maximum (separate run)
  d deblend rounds / passes / ms per round / of which LM fits / background maps   j cell size groups / largest cell
  y in the last 12 rounds: background tiles switching measured/interpolated / fit statuses changing / cycle length
    in rounds (0: none) / last 99th-percentile parameter step x 1e4
  r slowest page s / rounds / passes / size groups / largest cell / how many of the run's 4 slowest pages were a
    worker's first page
  e largest change: same page again after another page (a hole) / against the run's pages.csv (a page mean) / GPU
    against CPU code on a crop, 3 deblend rounds (a hole); d: different hole lists   k 1-3 more deblend rounds:
    largest page-mean / 99th-percentile hole change (e and k in the run's unit, e.g. nm)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cd_batch import FrameReader, visible_gpus  # noqa: E402  the batch's own page reader and GPU list

CODE = ("subpx/cd.py", "subpx/_cpu", "subpx/_gpu", "scripts/cd_batch.py", "scripts/cd_report.py",
        "scripts/cd_diagnose.py", "scripts/cd_diagnose_part2.py",
        "scripts/deblend_variants.py")  # what a CD measurement and this diagnosis run (folders: their .py files)
CELL_STAGES = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg")
WIDE_STAGES = ("tail_window", "tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
HARMLESS = ("CUDA_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUPY_CACHE_DIR")
PYTHON_DEBUG = ("PYTHONDEVMODE", "PYTHONTRACEMALLOC", "PYTHONMALLOC", "PYTHONPROFILEIMPORTTIME", "PYTHONDEBUG")
SLOW_SETTINGS = {"CUDA_LAUNCH_BLOCKING": "!", "CUDA_CACHE_DISABLE": "j", "CUDA_FORCE_PTX_JIT": "j",
                 "CUPY_CUDA_COMPILE_WITH_DEBUG": "d", "CUPY_ACCELERATORS": "a"}  # the n token's letters
TIMED_RUNS = 3
CROP = (256, 384)  # rows, columns of the central crop measured with both the GPU and the CPU code (~1,000 holes)
SYNTHETIC = ((1378, 2048), 9.5, 0.44 * 9.5, 20.0)  # scripts/benchmark_cd.lattice: 30,175 holes, alike everywhere
SYNTHETIC_SETTINGS = {  # cd_batch's defaults, whatever the run used, so that every machine measures the same thing
    "methods": ["gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
                "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio"],
    "seeds": "threshold", "invert": False, "area_min": 3, "area_max": 200, "pixel_size": 21.0, "unit": "nm",
    "half_area_level": 0.5, "half_area_peak_pixels": 4.0, "tail_area_level": 0.2, "tail_window": 1.2,
    "n_angles": 16, "outlines": False}
REFERENCE = {  # the PC this script was written on (RTX 4060 Ti, Ryzen 5 7500F, Windows 11), with this commit's code
    "versions": {"numpy": "2.4", "scipy": "1.17", "cupy": "14.2", "cv2": "5.0", "tifffile": "2026.3", "python": "3.11",
                 "cuda_runtime": "12"},
    "synthetic_image": "a5dd1c71", "synthetic_numbers": "65ab1be6"}
VERSION_LETTERS = {"numpy": "n", "scipy": "p", "cupy": "c", "cv2": "v", "tifffile": "t", "python": "y",
                   "cuda_runtime": "r"}
PLANS = {"381b4222-f694-41f0-9685-ff5bb260df2e": "a", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "h",
         "e9a42b02-d5df-448d-aa00-03f14749eb61": "u", "a1841308-3541-4fab-bc81-f71556f20b4a": "p"}
WORKER_TIMEOUT = 900  # s; a first page that compiles every GPU kernel takes about a minute
LINES = (("v", "n", "p", "w", "u", "c", "a"),
         ("x", "f", "t", "m", "h", "d", "j", "y"),
         ("r", "e", "k"))


# ---------------------------------------------------------------------------------------------------------- inputs
def load_run(run: Path) -> dict:
    """run.json of a cd_batch run."""
    path = run / "run.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found: RUN_FOLDER must be the --output folder of a cd_batch run")
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path, tiff: Path) -> list[dict]:
    """The rows of a cd_batch CSV that belong to TIFF (its file column ends with TIFF's name)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if Path(row.get("file") or "").name.lower() == tiff.name.lower()]


def measured(rows: list[dict]) -> list[dict]:
    """timings.csv rows of measured pages, fastest first."""
    return sorted((row for row in rows if row.get("status") == "ok" and row.get("measure_s")),
                  key=lambda row: float(row["measure_s"]))


def pick_pages(rows: list[dict], pages: str | None) -> list[tuple[int, float]]:
    """(page, the run's seconds for it) of the median and the slowest measured page, or of the pages named."""
    seconds = {int(row["page"]): float(row["measure_s"]) for row in rows}
    if pages:
        try:
            named = [int(page) for page in pages.split(",") if page.strip()]
        except ValueError:
            named = []
        if not named or min(named) < 0:
            raise SystemExit(f"--pages must look like 94,46 (0 = first page), not {pages!r}")
        named = (named + named)[:2]  # one page: the median and the slowest page both
        return [(page, seconds.get(page, float("nan"))) for page in named]
    if not rows:
        raise SystemExit("timings.csv of RUN_FOLDER has no measured page of TIFF: name two pages with --pages A,B")
    return [(int(row["page"]), float(row["measure_s"])) for row in (rows[len(rows) // 2], rows[-1])]


def first_pages_among_slowest(rows: list[dict], count: int = 4) -> int | None:
    """How many of the ``count`` slowest pages were a worker's first page (each worker's lowest page: the reader hands
    pages out in order), i.e. paid for loading or compiling the GPU code."""
    if not rows or "worker" not in rows[0]:
        return None
    first = {}
    for row in rows:
        key = (row.get("gpu"), row.get("worker"))
        first[key] = min(first.get(key, int(row["page"])), int(row["page"]))
    firsts = set(first.values())
    return sum(int(row["page"]) in firsts for row in rows[-count:])


# --------------------------------------------------------------------------------------------------------- machine
def code_files(root: Path = ROOT) -> list[Path]:
    files = []
    for item in CODE:
        path = root / item
        files += sorted(path.glob("*.py")) if path.is_dir() else [path] if path.is_file() else []
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def fingerprint(files: list[Path], root: Path = ROOT) -> tuple[str, dict[str, str]]:
    """4 hex digits for the files' names and contents, line endings ignored (git may check files out with CRLF), and
    every file's own 8."""
    total, each = hashlib.sha1(), {}
    for path in files:
        name = path.relative_to(root).as_posix()
        data = path.read_bytes().replace(b"\r\n", b"\n")
        each[name] = hashlib.sha1(data).hexdigest()[:8]
        total.update(name.encode() + b"\0" + data + b"\0")
    return total.hexdigest()[:4], each


def git_commit(files: list[Path]) -> str:
    """The checked-out commit (5 hex digits), +N when N of ``files`` differ from it (changed or new); - without git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=5", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=30).stdout.strip()[:5]
        if not commit:
            return "-"
        changed = subprocess.run(["git", "status", "--porcelain", "--"] + [str(f.relative_to(ROOT)) for f in files],
                                 cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        return commit + (f"+{len(changed.splitlines())}" if changed else "")
    except Exception:
        return "-"


def debug_settings(env=None) -> tuple[list[str], str]:
    """Environment variables that can slow CUDA, CuPy or Python down (names), and the letters of the known slow ones
    that are on (SLOW_SETTINGS; p: a Python debug setting)."""
    env = os.environ if env is None else env
    names = sorted(name for name in env if name.upper() in PYTHON_DEBUG or (
        name.upper().startswith(("CUDA_", "CUPY_")) and name.upper() not in HARMLESS
        and not name.upper().startswith("CUDA_PATH")))
    on = [name.upper() for name in names if str(env[name]).strip() not in ("", "0")]
    letters = {SLOW_SETTINGS.get(name, "p" if name in PYTHON_DEBUG else "") for name in on}
    return names, "".join(sorted(letters))


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "cupy", "cv2", "tifffile"):
        try:
            versions[name] = __import__(name).__version__
        except Exception as error:
            versions[name] = f"missing ({type(error).__name__})"
    return versions


def other_versions(versions: dict) -> str:
    """Letters of the libraries whose major.minor version (CUDA runtime: major) differs from the reference PC's."""
    letters = ""
    for name, letter in VERSION_LETTERS.items():
        mine = str(versions.get(name, ""))
        parts = mine.split(".")
        short = parts[0] if name == "cuda_runtime" else ".".join(parts[:2])
        if short != REFERENCE["versions"][name]:
            letters += letter
    return letters or "="


def windows_facts() -> dict:
    """Power plan, hardware GPU scheduling, virtual machine, memory integrity, sockets, CPU (Windows only)."""
    if sys.platform != "win32":
        return {}
    script = ("$o = @{}; "
              "try { $o.plan = (powercfg /getactivescheme) -join ' ' } catch {}; "
              "try { $c = Get-CimInstance Win32_ComputerSystem; $o.maker = [string]$c.Manufacturer; "
              "$o.model = [string]$c.Model; $o.sockets = $c.NumberOfProcessors; "
              "$o.logical = $c.NumberOfLogicalProcessors } catch {}; "
              "try { $o.cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name } catch {}; "
              "try { $g = Get-CimInstance -ClassName Win32_DeviceGuard -Namespace root\\Microsoft\\Windows\\DeviceGuard;"
              " $o.integrity = [bool]($g.SecurityServicesRunning -contains 2) } catch {}; "
              "try { $o.scheduling = (Get-ItemProperty 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers' "
              "-ErrorAction Stop).HwSchMode } catch {}; "
              "$o | ConvertTo-Json -Compress")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, timeout=120)
        return json.loads(out.stdout.decode("utf-8", "replace").strip() or "{}")
    except Exception:
        return {}


def windows_token(facts: dict) -> str:
    if not facts:
        return "-"
    plan = next((letter for guid, letter in PLANS.items() if guid in str(facts.get("plan", "")).lower()), "-")
    scheduling = {2: "1", 1: "0"}.get(facts.get("scheduling"), "-")
    machine = f"{facts.get('maker', '')} {facts.get('model', '')}".lower()
    virtual = "-" if not machine.strip() else "v" if any(
        word in machine for word in ("virtual", "vmware", "qemu", "kvm", "xen", "hyper-v", "bochs")) else "n"
    integrity = {True: "1", False: "0"}.get(facts.get("integrity"), "-")
    return plan + scheduling + virtual + integrity


def nvidia_smi(*arguments: str) -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", *arguments], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


def gpu_table() -> list[dict] | None:
    """nvidia-smi's GPUs: index, name, bus, use, memory, and where the driver offers them, its model and clocks."""
    def query(fields: str, keys: tuple[str, ...]) -> list[dict] | None:
        out = nvidia_smi(f"--query-gpu={fields}", "--format=csv,noheader,nounits")
        if out is None:
            return None
        return [dict(zip(keys, (cell.strip() for cell in line.split(",")))) for line in out.splitlines() if line.strip()]

    gpus = query("index,name,pci.bus_id,utilization.gpu,memory.used,memory.total",
                 ("index", "name", "bus", "busy_pct", "memory_used_mb", "memory_total_mb"))
    if gpus is None:
        return None
    extra = query("driver_model.current,driver_version,pstate,clocks.sm,clocks.max.sm,power.draw,power.limit",
                  ("driver_model", "driver", "pstate", "clock_mhz", "clock_max_mhz", "power_w", "power_limit_w"))
    if extra and len(extra) == len(gpus):
        for gpu, more in zip(gpus, extra):
            gpu.update(more)
    detail = nvidia_smi("-q") or ""
    modes = [line.split(":", 1)[1].strip() for line in detail.splitlines()
             if line.strip().startswith("Virtualization Mode")]
    for gpu, mode in zip(gpus, modes):
        gpu["virtualization"] = mode
    return gpus


class Sampler:
    """nvidia-smi every 100 ms, with its timestamps: utilization (share of time a kernel runs), SM clock and its
    maximum, memory used (MiB)."""

    def __init__(self, gpu: str | None = None):
        command = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.max.sm,memory.used",
                   "--format=csv,noheader,nounits", "-lms", "100"] + (["-i", str(gpu)] if gpu is not None else [])
        self.samples: list[tuple[float, str, float, float, float, float]] = []
        self.thread = None
        try:
            self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            parts = [part.strip() for part in line.split(",")]
            try:
                stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
                self.samples.append((stamp, parts[1], float(parts[2]), float(parts[3]), float(parts[4]),
                                     float(parts[5])))
            except (ValueError, IndexError):
                continue

    def stop(self) -> list[tuple[float, str, float, float, float, float]]:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except Exception:
                self.process.kill()
            self.thread.join(5)
        return self.samples


def cpu_benchmark() -> float:
    """Milliseconds for a fixed single-thread mix like the pipeline's host steps: small NumPy calls and a Python loop
    (no library algorithm, so that it measures the CPU, not a library version); best of 3."""
    values = np.random.default_rng(0).random(30000)
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        for i in range(3000):
            part = values[i:i + 100]
            np.median(part)
            np.where(part > 0.5, part, 0.0).sum()
        total = 0
        for i in range(600000):
            total += i % 7
        best = min(best, time.perf_counter() - start)
    return 1000 * best


# ------------------------------------------------------------------------------------------------- worker processes
class _Session:
    """What a worker process does when asked (each method answers one request from the main process)."""

    def __init__(self, cp, settings: dict):
        self.cp, self.settings = cp, settings
        self.images, self.results = {}, {}

    def measure_now(self, image, backend: str = "gpu", **extra):
        from subpx import measure_cds
        return measure_cds(image, backend=backend, **{**self.settings, **extra})

    def page(self, conn, name, image):
        self.images[name] = image

    def crop(self, conn, name, source):
        image = self.images[source]
        h, w = CROP
        y0, x0 = max(0, (image.shape[0] - h) // 2), max(0, (image.shape[1] - w) // 2)
        self.images[name] = image[y0:y0 + h, x0:x0 + w]

    def versions(self, conn):
        runtime = self.cp.cuda.runtime
        return {"cuda_runtime": str(runtime.runtimeGetVersion() // 1000), "cuda_runtime_full": runtime.runtimeGetVersion(),
                "cuda_driver": runtime.driverGetVersion()}

    def latency(self, conn):
        """us per small GPU call that waits for its result (at most 500 calls or 5 s)."""
        x = self.cp.zeros(1)
        for _ in range(50):
            x += 1
            float(x[0])
        start, calls = time.perf_counter(), 0
        while calls < 500 and time.perf_counter() - start < 5:
            x += 1
            float(x[0])
            calls += 1
        return 1e6 * (time.perf_counter() - start) / calls

    def first(self, conn, name):
        start = time.perf_counter()
        self.measure_now(self.images[name])
        return time.perf_counter() - start

    def timed(self, conn, name, runs):
        start, seconds = time.time(), []
        for _ in range(runs):
            seconds.append(self.measure_now(self.images[name]).meta["seconds"])
        return start, time.time(), seconds

    def loop(self, conn, name):
        """Measure the page again and again until the main process sends anything (it is then read here)."""
        count = 0
        while not conn.poll():
            self.measure_now(self.images[name])
            count += 1
        conn.recv()
        return count

    def cpu(self, conn):
        return cpu_benchmark()

    def synthetic(self, conn):
        from scripts.benchmark_cd import lattice
        from scripts.cd_report import cd_statistics
        from subpx import measure_cds

        image = lattice(*SYNTHETIC)[0]
        measure_cds(image, backend="gpu", **SYNTHETIC_SETTINGS)  # the first use of its cell sizes
        start = time.perf_counter()
        result = measure_cds(image, backend="gpu", **SYNTHETIC_SETTINGS)
        seconds = time.perf_counter() - start
        stats = cd_statistics(result)
        numbers = ";".join(f"{name}={stats[name]['mean']:.4f}/{stats[name]['valid']}" for name in sorted(stats))
        return {"seconds": seconds, "rounds": int(result.meta["deblend_iterations_used"]),
                "settled": bool(result.meta["deblend_converged"]), "holes": len(result.seeds_xy),
                "image": hashlib.sha1(np.ascontiguousarray(image).tobytes()).hexdigest()[:8],
                "numbers": hashlib.sha1(numbers.encode()).hexdigest()[:8], "means": numbers}

    def measure(self, conn, key, name, extra=None, backend="gpu"):
        from scripts.cd_report import cd_statistics

        start = time.perf_counter()
        result = self.results[key] = self.measure_now(self.images[name], backend, **(extra or {}))
        return {"seconds": time.perf_counter() - start, "holes": len(result.seeds_xy),
                "rounds": int(result.meta["deblend_iterations_used"]), "settled": bool(result.meta["deblend_converged"]),
                "means": {name: s["mean"] for name, s in cd_statistics(result).items()}}

    def probe(self, conn, key, name):
        with Probe() as probe:
            start = time.perf_counter()
            result = self.results[key] = self.measure_now(self.images[name])
            seconds = time.perf_counter() - start
        return dict(probe.summary(), seconds=seconds, holes=len(result.seeds_xy),
                    largest_cell=tuple(int(v) for v in result.meta.get("cell_stack_shape", result.masks.shape[1:])),
                    stage_seconds=result.meta["seconds"], added=result.meta.get("seeds_added_from_residual"),
                    dropped=result.meta.get("seeds_dropped_as_insignificant"))

    def variant(self, conn, key, name, variant, full_iterations, extra=None):
        """measure() with scripts/deblend_variants' loop in place of the library's: the switches of ``variant`` (a
        dict of Variant's fields), every pass and round recorded. Rounds are the total over the passes (measure_cds'
        deblend_iterations_used counts the last pass only)."""
        from scripts.cd_report import cd_statistics
        from scripts.deblend_variants import Patch, Variant

        start = time.perf_counter()
        with Patch(Variant(**variant), full_iterations=full_iterations) as patch:
            result = self.results[key] = self.measure_now(self.images[name], **(extra or {}))
        meta = result.meta
        return {"seconds": time.perf_counter() - start, "holes": len(result.seeds_xy),
                "rounds": sum(p["used"] for p in patch.passes), "settled": bool(meta["deblend_converged"]),
                "means": {name: s["mean"] for name, s in cd_statistics(result).items()},
                "measured_tiles": meta.get("background_measured_tiles"), "tiles": meta.get("background_tiles"),
                "source": meta.get("background_source"), "passes": patch.passes}

    def compare(self, conn, a, b):
        return compare(self.results[a], self.results[b])


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}".replace("\n", " ")[:500]


def _worker(conn, settings: dict, threads: int) -> None:
    """A worker as cd_batch runs one, on the one GPU it sees, answering the main process's requests."""
    import warnings
    warnings.filterwarnings("ignore")
    try:
        try:
            import cv2
            cv2.setNumThreads(threads)
        except Exception:
            pass
        import cupy as cp
        from subpx._gpu import _cuda

        cp.cuda.Device(0).use()  # CUDA_VISIBLE_DEVICES shows this process one GPU, as in the batch
        int(cp.arange(4).sum())
        _cuda.KEEP_MEMORY = True
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
        session = _Session(cp, settings)
        conn.send(("ready", f"{props.get('pciDomainID', 0):04x}:{props['pciBusID']:02x} {name}"))
    except BaseException as error:
        conn.send(("error", _describe(error)))
        return
    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            return
        if request[0] == "stop":
            return
        try:
            conn.send(("done", getattr(session, request[0])(conn, *request[1:])))
        except BaseException as error:
            conn.send(("error", _describe(error)))


class Worker:
    """A worker process started as cd_batch starts one: spawned, seeing only ``gpu``, with the batch's thread
    limits (one worker per GPU: logical CPUs / GPUs)."""

    def __init__(self, gpu: str, settings: dict, threads: int):
        import multiprocessing as mp

        context = mp.get_context("spawn")
        self.conn, child = context.Pipe()
        env = {"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
               "CUDA_VISIBLE_DEVICES": gpu, "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")}
        saved = {name: os.environ.get(name) for name in env}
        os.environ.update(env)  # a spawned process starts with the parent's environment at that moment
        try:
            self.process = context.Process(target=_worker, args=(child, settings, threads), daemon=True)
            self.process.start()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        child.close()
        self.gpu = self.receive("ready")

    def send(self, *request) -> None:
        self.conn.send(request)

    def receive(self, kind: str = "done"):
        if not self.conn.poll(WORKER_TIMEOUT):
            raise RuntimeError(f"no answer from a worker after {WORKER_TIMEOUT} s")
        try:
            answer, value = self.conn.recv()
        except EOFError:
            raise RuntimeError(f"a worker stopped (exit code {self.process.exitcode})") from None
        if answer == "error":
            raise RuntimeError(value)
        if answer != kind:
            raise RuntimeError(f"a worker answered {answer} instead of {kind}")
        return value

    def ask(self, *request):
        self.send(*request)
        return self.receive()

    def close(self) -> None:
        try:
            self.send("stop")
        except Exception:
            pass
        self.process.join(30)
        if self.process.is_alive():
            self.process.kill()


def run_stages(rows: list[dict]) -> list[dict]:
    """timings.csv rows as measure_cds' meta["seconds"] (measure_s is the total)."""
    return [{("total" if key == "measure_s" else key[:-2]): float(row[key]) for key in row
             if key.endswith("_s") and key not in ("read_s", "check_s") and row[key]} for row in rows]


def stage_groups(seconds: list[dict]) -> dict:
    """Median seconds per stage over runs, and the groups: deblend, cell methods, wide methods, the rest."""
    stages = {key: float(np.median([s.get(key, 0.0) for s in seconds])) for key in seconds[0]}
    deblend = stages.get("deblend", 0.0)
    cell = sum(stages.get(key, 0.0) for key in CELL_STAGES)
    wide = sum(stages.get(key, 0.0) for key in WIDE_STAGES)
    return {"total": stages["total"], "deblend": deblend, "cell": cell, "wide": wide,
            "rest": stages["total"] - deblend - cell - wide, "stages": stages}


# --------------------------------------------------------------------------------------------------------- probing
class Probe:
    """Wrappers around the deblend steps during measure_cds calls (no code changed): every pass (rounds asked for,
    used, settled), every hole re-check, the time of each LM fit, background map and render (waiting for the GPU
    before and after each, so a probed page runs a little slower than a plain one), the fit statuses, the cell size
    groups, each background map's free pixels per tile, and the parameters after every round."""

    def __enter__(self):
        import cupy as cp
        import subpx._cpu.deblend as cpu
        import subpx._gpu.deblend as gpu

        self.passes, self.rechecks, self.events, self.parts = [], [], [], []
        self.tiles, self.params, self.statuses = [], [], []
        self.current, self.inside = -1, False  # the pass under way; only steps inside a deblend pass are recorded
        names = [(gpu, "deblend"), (gpu, "_fit_device"), (gpu, "_background_device"), (gpu, "_render_device"),
                 (gpu, "_cell_parts"), (cpu, "revise_seeds"), (cpu, "background_tiles"), (cpu, "fit_status")]
        self.saved = [(module, name, getattr(module, name)) for module, name in names]
        deblend, fit, background, render, cell_parts, revise, tiles, status = (f for _, _, f in self.saved)
        sync = cp.cuda.Device().synchronize

        def timed(name, function):
            def wrapper(*args, **kwargs):
                if not self.inside:
                    return function(*args, **kwargs)
                sync()
                start = time.perf_counter()
                out = function(*args, **kwargs)
                sync()
                self.events.append((self.current, name, start, time.perf_counter()))
                return out
            return wrapper

        def probed_deblend(*args, **kwargs):
            self.current, self.inside = len(self.passes), True
            start = time.perf_counter()
            try:
                state = deblend(*args, **kwargs)
            finally:
                self.inside = False
            self.passes.append({"asked": kwargs.get("iterations"), "used": int(state.iterations),
                                "settled": bool(state.converged), "continued": kwargs.get("init") is not None,
                                "holes": len(args[1]) if len(args) > 1 else None,
                                "seconds": time.perf_counter() - start})
            return state

        timed_render = timed("render", render)

        def probed_render(params_global, *args, **kwargs):
            out = timed_render(params_global, *args, **kwargs)
            if self.inside:
                self.params.append((self.current, np.array(params_global)))
            return out

        def probed_parts(masks):
            out = cell_parts(masks)
            if self.inside:
                self.parts.append((len(out), tuple(int(v) for v in masks.shape[1:])))
            return out

        def probed_revise(*args, **kwargs):
            drop, new = revise(*args, **kwargs)
            self.rechecks.append((int(np.sum(drop)), len(new)))
            return drop, new

        def probed_tiles(tile_medians, free, usable, tile, shape):
            if not self.inside:
                return tiles(tile_medians, free, usable, tile, shape)
            counts = []

            def recording(mask):  # its first call is over the free pixels (see background_tiles)
                out = tile_medians(mask)
                counts.append(np.asarray(out[1]))
                return out
            medians, info = tiles(recording, free, usable, tile, shape)
            # background_tiles' rule: a tile is measured with at least max(16, 10 % of its pixels) free pixels.
            self.tiles.append((self.current, counts[0], max(16, int(0.1 * tile * tile))))
            return medians, info

        def probed_status(params, masks, caps, status_array, *args, **kwargs):
            out = status(params, masks, caps, status_array, *args, **kwargs)
            if self.inside:
                self.statuses.append((self.current, np.array(status_array)))
            return out

        for module, name, wrapper in (
                (gpu, "deblend", probed_deblend), (gpu, "_fit_device", timed("fit", fit)),
                (gpu, "_background_device", timed("background", background)), (gpu, "_render_device", probed_render),
                (gpu, "_cell_parts", probed_parts), (cpu, "revise_seeds", probed_revise),
                (cpu, "background_tiles", probed_tiles), (cpu, "fit_status", probed_status)):
            setattr(module, name, wrapper)
        return self

    def __exit__(self, *exc) -> None:
        for module, name, original in self.saved:
            setattr(module, name, original)

    def summary(self) -> dict:
        rounds = []  # (seconds, LM fit seconds, background seconds) of every round: one render to the next in a pass
        for k in range(len(self.passes)):
            renders = [e for e in self.events if e[0] == k and e[1] == "render"]
            for a, b in zip(renders, renders[1:]):
                inside = [e for e in self.events if e[0] == k and a[3] <= e[2] and e[3] <= b[3]]
                rounds.append((b[3] - a[3], sum(e[3] - e[2] for e in inside if e[1] == "fit"),
                               sum(e[3] - e[2] for e in inside if e[1] == "background")))
        last = len(self.passes) - 1
        maps = [(counts, rule) for k, counts, rule in self.tiles if k == last][-13:]
        switches = [int(((a >= ra) != (b >= rb)).sum()) for (a, ra), (b, rb) in zip(maps, maps[1:])
                    if a.shape == b.shape]
        statuses = [s for k, s in self.statuses if k == last][-13:]
        flips = [int((a != b).sum()) for a, b in zip(statuses, statuses[1:]) if a.shape == b.shape]
        params = [p for k, p in self.params if k == last]
        steps = [float(np.percentile(np.abs(b[:, 1:] - a[:, 1:]).max(axis=1), 99))
                 for a, b in zip(params, params[1:]) if len(a) == len(b) and len(a)]
        ms = list(np.median(np.array(rounds), axis=0) * 1000) if rounds else [None] * 3
        return {"passes": self.passes, "rechecks": self.rechecks, "rounds": sum(p["used"] for p in self.passes),
                "round_ms": ms[0], "fit_ms": ms[1], "background_ms": ms[2],
                "size_groups": self.parts[-1][0] if self.parts else None,
                "stack": self.parts[-1][1] if self.parts else None,
                "switches": sum(switches[-12:]) if maps else None, "switches_per_round": switches,
                "status_flips": sum(flips[-12:]) if statuses else None, "flips_per_round": flips,
                "measured_tiles": float(np.mean(maps[-1][0] >= maps[-1][1])) if maps else None,
                "cycle": cycle_length(params), "steps": steps}


def cycle_length(params: list[np.ndarray], longest: int = 8) -> int:
    """The smallest k for which the last 3 rounds' parameters repeat those k rounds earlier far more closely (< 5 %)
    than consecutive rounds differ: the loop is going round a cycle of k rounds. 0: no such cycle."""
    params = [p[:, 1:] for p in params if len(p) == len(params[-1])] if params else []
    if len(params) < 6:
        return 0
    moving = max(float(np.abs(params[-1 - j] - params[-2 - j]).max()) for j in range(3))
    if moving == 0:
        return 0
    for k in range(2, longest + 1):
        if len(params) < k + 3:
            break
        if max(float(np.abs(params[-1 - j] - params[-1 - j - k]).max()) for j in range(3)) < 0.05 * moving:
            return k
    return 0


def compare(a, b) -> dict:
    """How far result b is from result a: the largest change of a page mean (over all methods, the cell methods, the
    wide methods), and where both hole lists are the same, of a single hole's CD: the largest, the 99th percentile,
    the median, and the holes that moved more than 1 (each the largest over the methods), in the results' unit;
    holes whose valid flag changed."""
    from scripts.cd_report import cd_statistics

    same = len(a.seeds_xy) == len(b.seeds_xy) and bool(np.array_equal(a.seeds_xy, b.seeds_xy))
    stats_a, stats_b = cd_statistics(a), cd_statistics(b)
    means = {n: abs(stats_a[n]["mean"] - stats_b[n]["mean"]) for n in a.measurements if n in b.measurements}
    means = {n: m for n, m in means.items() if np.isfinite(m)}
    hole_max = hole_p99 = hole_median = 0.0
    flipped = above = 0
    for name in a.measurements if same else ():
        ma, mb = a.measurements[name], b.measurements[name]
        both = ma.valid & mb.valid
        flipped += int((ma.valid != mb.valid).sum())
        d = np.abs(ma.metrics["cd_equivalent_px"][both] - mb.metrics["cd_equivalent_px"][both]) * a.pixel_size
        d = d[np.isfinite(d)]
        if d.size:
            hole_max, hole_p99 = max(hole_max, float(d.max())), max(hole_p99, float(np.percentile(d, 99)))
            hole_median, above = max(hole_median, float(np.median(d))), max(above, int((d > 1.0).sum()))

    def largest(names):
        return max((means[n] for n in names if n in means), default=None)
    return {"same_holes": same, "holes": (len(a.seeds_xy), len(b.seeds_xy)),
            "page_mean": largest(means), "page_mean_cell": largest(CELL_STAGES),
            "page_mean_wide": largest(WIDE_STAGES), "hole_max": hole_max if same else None,
            "hole_p99": hole_p99 if same else None, "hole_median": hole_median if same else None,
            "holes_above": above if same else None, "valid_changed": flipped}


# ---------------------------------------------------------------------------------------------------------- output
def num(value, digits: int = 1) -> str:
    """A short number: trailing zeros dropped (2.0 -> 2, 0.30 -> 0.3); - when missing."""
    if value is None or not np.isfinite(value):
        return "-"
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def token_lines(tokens: dict, lines=LINES) -> list[str]:
    """The lines to type (part 1's three unless ``lines`` says otherwise): every token, "-" for one that is missing."""
    return [" ".join(f"{key}{tokens.get(key) or '-'}" for key in line) for line in lines]


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# ------------------------------------------------------------------------------------------------------------ main
@dataclass
class Context:
    """Where both parts start: the run, its TIFF and measuring settings, its measured pages (timings.csv rows), the
    median and the slowest page (numbers, the run's seconds, images), the GPUs, CPU threads per worker, the start."""
    run: Path
    tiff: Path
    run_info: dict
    settings: dict
    timings: list
    median_page: int
    median_run_s: float
    slowest_page: int
    slowest_run_s: float
    median_image: np.ndarray
    slowest_image: np.ndarray
    gpus: list
    threads: int
    started: float


def setup(args) -> Context:
    """The start both parts share: every input is read and checked before anything is measured."""
    started = time.perf_counter()
    run, tiff = args.run_folder, args.tiff
    if not tiff.is_file():
        raise SystemExit(f"TIFF not found: {tiff}")
    run_info = load_run(run)
    timings = measured(csv_rows(run / "timings.csv", tiff))
    (median_page, median_run_s), (slowest_page, slowest_run_s) = pick_pages(timings, args.pages)
    gpus = visible_gpus()
    if not gpus:
        raise SystemExit("No GPU found (nvidia-smi / CUDA_VISIBLE_DEVICES)")
    reader = FrameReader(tiff)
    try:
        median_image, slowest_image = reader.read(median_page), reader.read(slowest_page)
    except (IndexError, ValueError) as error:
        raise SystemExit(f"{tiff.name}: {error}") from None
    finally:
        reader.close()
    threads = max(1, (os.cpu_count() or 1) // len(gpus))  # cd_batch's threads with one worker per GPU
    return Context(run, tiff, run_info, run_info["settings"]["measure"], timings, median_page, median_run_s,
                   slowest_page, slowest_run_s, median_image, slowest_image, gpus, threads, started)


def progress(started: float, problems: list):
    """step(title) prints a progress line (seconds since ``started``); attempt(title, function) runs one part, and a
    failure prints one sentence and is kept in ``problems`` (that part's tokens stay "-")."""

    def step(title: str) -> None:
        print(f"[{time.perf_counter() - started:4.0f} s] {title}", flush=True)

    def attempt(title: str, function) -> None:
        try:
            function()
        except Exception as error:
            problems.append(f"{title}: {_describe(error)}")
            print(f"       {title} failed: {_describe(error)}", flush=True)
    return step, attempt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_folder", type=Path, help="The --output folder of a cd_batch run that measured TIFF")
    parser.add_argument("tiff", type=Path, help="The TIFF that run measured")
    parser.add_argument("--pages", help="Two pages to use instead of the run's median and slowest, e.g. 94,46")
    parser.add_argument("--part", type=int, choices=(1, 2), default=1,
                        help="2: the follow-up run (scripts/cd_diagnose_part2.py)")
    args = parser.parse_args(argv)
    context = setup(args)
    if args.part == 2:
        from scripts import cd_diagnose_part2
        return cd_diagnose_part2.main(context)
    return part_one(context)


def part_one(context: Context) -> int:
    """This machine; the median page in a worker alone, beside a second one, with one on every GPU; every deblend step
    of both pages; the numbers. Three lines."""
    run, tiff, run_info, settings, timings = (context.run, context.tiff, context.run_info, context.settings,
                                              context.timings)
    median_page, median_run_s, slowest_page = context.median_page, context.median_run_s, context.slowest_page
    median_image, slowest_image = context.median_image, context.slowest_image
    gpus, threads, started = context.gpus, context.threads, context.started
    tokens, problems = {}, []
    details = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "tiff": str(tiff), "run": str(run),
               "pages": {"median": median_page, "slowest": slowest_page}, "gpus_used": gpus,
               "threads_per_worker": threads}
    step, attempt = progress(started, problems)

    step("this machine")
    files = code_files()
    code, each = fingerprint(files)
    trial_commit = (run_info.get("git_commit") or "-")[:5]
    tokens["v"] = f"{code}/{git_commit(files)}/{trial_commit}"
    names, slow = debug_settings()
    tokens["n"] = f"{len(names)}{slow}"
    versions = library_versions()
    facts = windows_facts()
    tokens["w"] = windows_token(facts)
    table = gpu_table()
    sampler, start = Sampler(), time.perf_counter()
    cpu_alone = cpu_benchmark()  # meanwhile nvidia-smi samples the GPUs before this script uses them
    time.sleep(max(0.0, 2.5 - (time.perf_counter() - start)))
    idle = sampler.stop()
    models = {gpu.get("driver_model", "") for gpu in table or []}
    model = {"WDDM": "w", "TCC": "t", "MCDM": "m"}.get(models.pop(), "-") if len(models) == 1 else "-"
    virtual = "v" if any(gpu.get("virtualization", "None") not in ("None", "N/A") for gpu in table or []) else ""
    busy_before = {}
    for _, index, busy, *_ in idle:
        busy_before.setdefault(index, []).append(busy)
    gpus_token = f"{len(table)}{model}{virtual}" if table else "-"
    before_token = num(max(np.mean(v) for v in busy_before.values()), 0) if busy_before else "-"
    details.update(code=code, files=each, debug_settings={name: os.environ[name] for name in names}, versions=versions,
                   windows=facts, nvidia_smi=table, busy_before_pct={k: float(np.mean(v)) for k, v in busy_before.items()},
                   cpu_alone_ms=cpu_alone, cpu=platform.processor(), platform=platform.platform(),
                   run_commit=run_info.get("git_commit"), measure=settings,
                   run_stage_medians=stage_groups(run_stages(timings)) if timings else None)
    seen = {}

    step(f"a worker alone on GPU {gpus[0]}: GPU round trip, page {median_page} {TIMED_RUNS + 1} times, synthetic page")
    alone = {}

    def run_alone():
        worker = Worker(gpus[0], settings, threads)
        try:
            seen["alone"] = worker.gpu
            versions.update(worker.ask("versions"))
            worker.ask("page", "median", median_image)
            alone["latency_us"] = worker.ask("latency")
            alone["first_s"] = worker.ask("first", "median")
            alone["timed"] = worker.ask("timed", "median", TIMED_RUNS)[2]
            sampler = Sampler(gpus[0])
            time.sleep(1.5)  # nvidia-smi's first samples
            start, end, _ = worker.ask("timed", "median", 1)
            samples = [s for s in sampler.stop() if start <= s[0] <= end]
            alone["busy_pct"] = float(np.mean([s[2] for s in samples])) if samples else None
            alone["clock_pct"] = float(np.mean([100 * s[3] / s[4] for s in samples if s[4] > 0])) if samples else None
            alone["synthetic"] = worker.ask("synthetic")
            step(f"a second worker on GPU {gpus[0]} beside the first")
            second = Worker(gpus[0], settings, threads)
            try:
                second.ask("page", "median", median_image)
                second.ask("first", "median")
                second.send("loop", "median")
                time.sleep(1.0)  # its loop is under way
                alone["latency_shared_us"] = worker.ask("latency")
                alone["shared_s"] = worker.ask("timed", "median", 1)[2][0]["total"]
                second.send("halt")
                alone["second_pages"] = second.receive()
            finally:
                second.close()
        finally:
            worker.close()

    attempt("a worker alone", run_alone)
    groups = stage_groups(alone["timed"]) if alone.get("timed") else None
    synthetic = alone.get("synthetic") or {}
    tokens["p"] = other_versions(versions)
    if synthetic:
        same_image = synthetic["image"] == REFERENCE["synthetic_image"]
        same_numbers = synthetic["numbers"] == REFERENCE["synthetic_numbers"]
        tokens["x"] = (f"{num(synthetic['seconds'])}/{synthetic['rounds']}/"
                       f"{'=' if same_image and same_numbers else 'd' if not same_image else 'n'}")
    tokens["f"] = num(alone.get("first_s"))
    tokens["h"] = f"{num(alone.get('busy_pct'), 0)}/{num(alone.get('clock_pct'), 0)}"
    if groups:
        tokens["m"] = "/".join(num(groups[key]) for key in ("deblend", "cell", "wide", "rest"))
    details["alone"] = dict(alone, groups=groups)

    step(f"a worker on each of {len(gpus)} GPUs at once: CPU benchmark, then page {median_page} {TIMED_RUNS} times")
    everywhere = {}

    def run_everywhere():
        workers = []
        try:
            for gpu in gpus:
                workers.append(Worker(gpu, settings, threads))
            for worker in workers:
                worker.ask("page", "median", median_image)
            for worker in workers:
                worker.send("first", "median")
            for worker in workers:
                worker.receive()
            for worker in workers:
                worker.send("cpu")
            everywhere["cpu_ms"] = [worker.receive() for worker in workers]
            for worker in workers:
                worker.send("timed", "median", TIMED_RUNS)
            timed = [worker.receive() for worker in workers]
            everywhere["gpus"] = [worker.gpu for worker in workers]
            everywhere["pages_s"] = [s["total"] for _, _, seconds in timed for s in seconds]
        finally:
            for worker in workers:
                worker.close()

    attempt("a worker on every GPU", run_everywhere)
    distinct = len(set(everywhere["gpus"])) if everywhere.get("gpus") else None
    tokens["u"] = f"{gpus_token}/{distinct if distinct is not None else '-'}/{before_token}"
    cpu_load = float(np.median(everywhere["cpu_ms"])) if everywhere.get("cpu_ms") else None
    tokens["c"] = f"{num(cpu_alone, 0)}/{num(cpu_load, 0)}/{os.cpu_count()}"
    tokens["a"] = f"{num(alone.get('latency_us'), 0)}/{num(alone.get('latency_shared_us'), 0)}"
    alone_s = groups["total"] if groups else None
    everywhere_s = float(np.median(everywhere["pages_s"])) if everywhere.get("pages_s") else None
    tokens["t"] = "/".join(num(v) for v in (alone_s, alone.get("shared_s"), everywhere_s, median_run_s))
    details["everywhere"] = everywhere

    step(f"a probing worker: every deblend step of pages {median_page} and {slowest_page}; the numbers")
    probed = {}

    def run_probes():
        worker = Worker(gpus[0], settings, threads)
        try:
            worker.ask("page", "median", median_image)
            worker.ask("page", "slowest", slowest_image)
            probed["median"] = worker.ask("probe", "probed", "median")
            probed["slowest"] = worker.ask("probe", "slowest", "slowest")
            probed["again"] = worker.ask("measure", "again", "median")
            probed["repeat"] = worker.ask("compare", "probed", "again")
            rounds = int(settings.get("deblend_iterations", 30))
            probed["more_rounds"] = []
            for extra in (1, 2, 3):
                worker.ask("measure", f"more{extra}", "median", {"deblend_iterations": rounds + extra})
                probed["more_rounds"].append(worker.ask("compare", "again", f"more{extra}"))
            worker.ask("crop", "crop", "median")
            probed["crop_gpu"] = worker.ask("measure", "crop_gpu", "crop", {"deblend_iterations": 3}, "gpu")
            probed["crop_cpu"] = worker.ask("measure", "crop_cpu", "crop", {"deblend_iterations": 3}, "cpu")
            probed["crop"] = worker.ask("compare", "crop_cpu", "crop_gpu")
        finally:
            worker.close()

    attempt("the probing worker", run_probes)
    if probed.get("median"):
        m = probed["median"]
        tokens["d"] = (f"{m['rounds']}/{len(m['passes'])}/{num(m['round_ms'], 0)}/{num(m['fit_ms'], 0)}/"
                       f"{num(m['background_ms'], 0)}")
        tokens["j"] = f"{m['size_groups']}/{m['largest_cell'][0]}x{m['largest_cell'][1]}"
        tokens["y"] = "/".join("-" if v is None else str(v) for v in (m["switches"], m["status_flips"], m["cycle"])) \
            + f"/{num(1e4 * m['steps'][-1], 0) if m['steps'] else '-'}"
    if probed.get("slowest"):
        s = probed["slowest"]
        firsts = first_pages_among_slowest(timings)
        tokens["r"] = (f"{num(s['seconds'])}/{s['rounds']}/{len(s['passes'])}/{s['size_groups']}/"
                       f"{s['largest_cell'][0]}x{s['largest_cell'][1]}/{'-' if firsts is None else firsts}")
    pages_csv = [row for row in csv_rows(run / "pages.csv", tiff) if str(row.get("page")) == str(median_page)]
    against_run = None
    if pages_csv and probed.get("again"):
        changes = [abs(float(pages_csv[0][f"{name}_cd"]) - mean) for name, mean in probed["again"]["means"].items()
                   if pages_csv[0].get(f"{name}_cd") not in (None, "") and np.isfinite(mean)]
        changes = [c for c in changes if np.isfinite(c)]
        against_run = max(changes) if changes else None
    if probed.get("repeat"):
        repeat, crop = probed["repeat"], probed.get("crop") or {}
        tokens["e"] = "/".join((num(repeat["hole_max"], 2) if repeat["same_holes"] else "d",
                                num(against_run, 2),
                                num(crop.get("hole_max"), 2) if crop.get("same_holes") else "d" if crop else "-"))
    if probed.get("more_rounds"):
        more = probed["more_rounds"]
        tokens["k"] = (f"{num(max((r['page_mean'] or 0) for r in more), 2)}/"
                       f"{num(max((r['hole_p99'] or 0) for r in more), 2)}")
    details.update(probed=probed, against_pages_csv=against_run, seen_gpus=seen)

    lines = token_lines(tokens)
    text = summary(tokens, details, problems, median_page, slowest_page)
    try:
        (run / "diagnosis.txt").write_text(text + "\n\n" + "\n".join(lines) + "\n\nDetails\n"
                                           + json.dumps(details, indent=1, default=_plain) + "\n", encoding="utf-8")
        saved = f"All details: {run / 'diagnosis.txt'}"
    except OSError as error:
        saved = f"diagnosis.txt could not be written: {error}"
    print(f"\n{text}\n{saved}\nDone in {time.perf_counter() - started:.0f} s.\n\nType these 3 lines to Claude:\n"
          + "\n".join(lines), flush=True)
    return 0


def summary(tokens: dict, details: dict, problems: list[str], median_page: int, slowest_page: int) -> str:
    """The same values in words, for whoever runs the script."""
    t = lambda key: tokens.get(key) or "-"  # noqa: E731
    gpus = details.get("nvidia_smi") or []
    lines = [
        f"Code         {t('v')}  (files / git commit now / the run's commit)",
        f"Machine      {len(gpus)} x {gpus[0]['name'] if gpus else '?'}; GPUs, workers' GPUs, busy before: {t('u')};"
        f" Windows {t('w')}; debug settings {t('n')}; other library versions {t('p')}",
        f"Speed        CPU {t('c')} (ms alone / ms with all workers busy / CPUs); GPU round trip {t('a')} us "
        f"(alone / beside a second worker)",
        f"Synthetic    {t('x')} (s / rounds / = same as the reference PC); a new worker's first page {t('f')} s",
        f"Page {median_page:<7} {t('t')} s (one worker / two on its GPU / one on every GPU / the run's own); "
        f"GPU busy / clock {t('h')} %",
        f"             deblend / cell methods / wide methods / rest {t('m')} s; deblend {t('d')} (rounds / passes / "
        f"ms per round / LM fit ms / background ms)",
        f"             cell size groups / largest cell {t('j')}; last 12 rounds {t('y')} (tile switches / status "
        f"changes / cycle length / last step x1e4)",
        f"Page {slowest_page:<7} slowest: {t('r')} (s / rounds / passes / size groups / largest cell / slowest pages "
        f"that were first pages)",
        f"Numbers      {t('e')} (same page again / against pages.csv / GPU vs CPU); 1-3 more rounds {t('k')}"]
    return "\n".join(lines + [f"PROBLEM      {p}" for p in problems])


if __name__ == "__main__":
    sys.exit(main())
