"""Part 2 of the CD diagnosis, after part 1's three lines: what part 1 left open, in one run of about 20 minutes, then
six short lines to type back.

    python scripts\\cd_diagnose.py D:\\results\\trial_w1 D:\\data\\dataset1\\stripe_001\\scan_001.tif --part 2

RUN_FOLDER and TIFF are those of part 1; everything measured is also written to RUN_FOLDER/diagnosis2.txt. In order:

1. The GPUs and Windows: other processes on each GPU, Remote Desktop and the GPUs, PCIe power saving, the GPU round
   trip on every GPU, and the power plan test: the round trip, the CPU and the median page on the current plan, then
   on High performance. That switch is the one change this script makes: for less than a minute, then the plan it
   found is set again (also after Ctrl+C or an error).
2. Workers per GPU: the earlier trial folders beside RUN_FOLDER, from their files; then the real cd_batch with 1, 2,
   3 and 4 workers per GPU on the 160 pages of TIFF with the most hole pages (read once beforehand, so that the disk
   does not decide), with the run's settings, into a temporary folder removed afterwards. Its output goes to a file
   there, never to this window (a click into the window would stop it). A worker count is stopped after 40 minutes,
   or after 10 minutes without a new hole page.
3. Deblend drift, in one worker on the first GPU, every round of every deblend pass recorded. The median page with
   today's loop, stop and rounds (what the batch computes), and the reference: the same loop run on to the tight
   stop (scripts/deblend_variants.TIGHT_TOLERANCE instead of today's 1e-3), where the loop ends; at most 200 rounds
   (and once at most 100: is it still moving?). Then five candidate cures, each to the practical stop
   (PRACTICAL_TOLERANCE, a stop a batch could afford), at most 200 rounds: how many rounds each needs, and does it
   end where the reference does? The slowest page with today's loop, its reference, and two cures to the practical
   stop: continuing after the seed re-check instead of fitting again, alone and with Anderson acceleration. And the
   page-to-page spread of the run's page means (pages.csv). A few minutes.

The six lines (labels and letters in values do not look like digits; "-" = could not be measured, or nothing to
measure):
  v code fingerprint / git commit (+N: N CD files differ from it) / the run's commit   a us per GPU call on each GPU,
    one GPU at a time   p processes on each GPU before this script started any (nvidia-smi's process table)   u the
    policy "use hardware graphics adapters for all Remote Desktop Services sessions" (1 on, 0 off, - not set: Remote
    Desktop sessions render in software), then where this script runs (r Remote Desktop, c console)   e PCIe link
    state power management on AC power (0 off, 1 moderate, 2 maximum power savings)   h us per GPU call / CPU
    benchmark ms / median page s, each on the plan found and on High performance (- on High performance: not
    switched)   w power plan after the test (a balanced, h high performance, u ultimate, p power saver)
  t earlier trial folders (+ between them, fewest workers first): workers per GPU / hole pages per s / median page s
    / workers busy % / peak memory of all workers GB   n settings that differ between the cd_batch runs of k and the
    run's run.json (0: none)
  k cd_batch with 1, 2, 3 and 4 workers per GPU (+ between them): hole pages per s (the middle half of them; pages
    never done count as done when the batch stopped) / median page s / most GPU memory on one GPU, GB / most GPU
    memory moved to system RAM on one GPU, GB / most committed memory, % of the limit / mean CPU % / least available
    RAM, GB
  y the median / the slowest page used   d the reference on the median page (today's loop run on to the tight stop,
    at most 200 rounds: where the loop ends): rounds / passes / last raw step, 99th percentile over the holes, in
    units of the tight stop (below 1: it got there) / page-mean change between at most 100 and at most 200 rounds /
    background tiles measured at the end, % (0: the map is inferred) / background level change over its last pass,
    grey levels   m what moves in today's last 12 rounds on the median page: centre p99 / width p99 / p99 of the
    holes < 1, 1-2, >= 2 pitches from the image edge (these five: the last round, in units of the tight stop) /
    background change per round, grey levels (each round's largest) / each round's step against the one before, % /
    cosine between consecutive steps, % (these three: medians)   c today's loop against the reference (how far
    today's stop is from the end): page-mean change, cell methods / wide methods / median hole change / holes that
    moved more than 1 unit, % / valid flags that flip   x candidates on the median page, each to the practical stop, at
    most 200 rounds (+ between them: LM fits of up to 100 steps, background map frozen, Anderson acceleration with
    the free pixels frozen, a start 2 % wider, tiles measured from 16 free pixels): rounds / page-mean change from the
    reference / hole p99 (near 0: the same end point); the last one also measured tiles %
  r the slowest page: the reference's rounds / passes / today's loop against it, page mean / then for continuing
    after the re-check, and for that with Anderson (free pixels frozen), each to the practical stop: rounds / passes /
    s / page mean against the reference   f page-to-page spread (standard deviation) of the run's page means
    (pages.csv, hole pages of TIFF): largest of the cell methods / of the wide methods / pages
  Stops: the loop stops once the 99th-percentile raw step of centres and widths is below a limit: today's 1e-3, the
  tight stop scripts/deblend_variants.TIGHT_TOLERANCE, the practical stop PRACTICAL_TOLERANCE. Rounds count every
  pass. c, x, r and f are in the run's unit (e.g. nm), each the largest over the methods; d in c or x: the hole lists
  differ from the reference's.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import locale
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from scripts.cd_batch import FrameReader
from scripts.cd_diagnose import (CELL_STAGES, PLANS, ROOT, WIDE_STAGES, Sampler, Worker, _plain, code_files, csv_rows,
                                 fingerprint, git_commit, num, nvidia_smi, progress, token_lines)
from scripts.deblend_variants import PRACTICAL_TOLERANCE, TIGHT_TOLERANCE

LINES = (("v", "a", "p", "u", "e", "h", "w"),
         ("t", "n"),
         ("k",),
         ("y", "d", "m"),
         ("c", "x"),
         ("r", "f"))
HIGH_PERFORMANCE = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
ASPM = ("501a4d13-42af-4429-9fd1-a8218c268e20", "ee12f906-d277-404b-b6da-e5fa1a576df5")  # PCI Express: link state
REMOTE_DESKTOP = r"SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services"
WINDOW = 160                    # pages of the scaling test
WORKERS_PER_GPU = (1, 2, 3, 4)
BATCH_TIMEOUT = 40 * 60         # s for one worker count
STALL = 10 * 60                 # s without a new hole page after which a worker count is stopped
FLAGS = {"seeds": "--seeds", "area_min": "--area-min", "area_max": "--area-max", "pixel_size": "--pixel-size",
         "unit": "--unit", "half_area_level": "--half-level", "half_area_peak_pixels": "--peak-pixels",
         "tail_area_level": "--tail-level", "tail_window": "--tail-window"}  # run.json's measure keys: cd_batch's flags
GUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
# A row of nvidia-smi's process table: GPU, [GI CI,] PID, type, name, memory.
PROCESS = re.compile(r"\|\s*(\d+)\s+(?:(?:N/A|\d+)\s+(?:N/A|\d+)\s+)?(\d+)\s+([CGM+]+)\s+(.*?)\s+(N/A|\d+\s*MiB)\s*\|")
WMI_LOOP = (  # one JSON line every ~2 s; single quotes only, so that it passes through the command line unchanged
    "while ($true) { $o = @{}; "
    "try { $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; $o.c = [double]$m.CommittedBytes; "
    "$o.l = [double]$m.CommitLimit; $o.a = [double]$m.AvailableMBytes; $o.p = [double]$m.PagesPersec } catch {}; "
    "try { $o.u = [double](Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | "
    "Where-Object { $_.Name -eq '_Total' }).PercentProcessorTime } catch {}; "
    "try { $o.g = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory | "
    "ForEach-Object { @{n = $_.Name; d = [double]$_.DedicatedUsage; s = [double]$_.SharedUsage} }) } catch {}; "
    "[Console]::Out.WriteLine(($o | ConvertTo-Json -Compress -Depth 3)); [Console]::Out.Flush(); "
    "Start-Sleep -Seconds 2 }")
REFERENCE_ROUNDS = 200          # the rounds allowed to the references and the candidates
CANDIDATES = (                  # x's order: key, scripts/deblend_variants.Variant's switches, what it is
    ("lm100", {"lm_steps": 100}, "LM fits of up to 100 steps"),
    ("frozen", {"freeze_background": True}, "background map frozen"),
    ("anderson", {"anderson": 5, "freeze_free": True}, "Anderson acceleration, free pixels frozen"),
    ("wider", {"widen": 0.02}, "a start 2 % wider"),
    ("tiles16", {"tile_free_px": 16}, "tiles measured from 16 free pixels"))


# ------------------------------------------------------------------------------------------ section 1: GPUs, Windows
def run_text(command: list[str]) -> str | None:
    """What an external program printed, or None when it could not run or failed."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def gpu_processes(text: str, gpus: list[str]) -> dict[str, list[str]]:
    """The process names nvidia-smi's process table lists on each of ``gpus`` (an empty list: none)."""
    found = {gpu: [] for gpu in gpus}
    for line in text.splitlines():
        match = PROCESS.match(line.strip())
        if match and match[1] in found:
            found[match[1]].append(match[4])
    return found


def session_letter(name: str | None) -> str:
    """Where this script runs, from SESSIONNAME: r a Remote Desktop session, c the console, - unknown."""
    name = (name or "").lower()
    return "r" if name.startswith("rdp-") else "c" if name == "console" else "-"


def remote_desktop_policy() -> int | None:
    """bEnumerateHWBeforeSW, the policy "use hardware graphics adapters for all Remote Desktop Services sessions";
    None when it is not set (then Remote Desktop sessions render in software)."""
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REMOTE_DESKTOP) as key:
            return int(winreg.QueryValueEx(key, "bEnumerateHWBeforeSW")[0])
    except FileNotFoundError:
        return None


def aspm_index(text: str | None) -> int | None:
    """PCIe link state power management on AC power from ``powercfg /query``: the first 0x number of its output (the
    second is DC power). Its words are in Windows' language, so only the number is read."""
    match = re.search(r"\b0x([0-9a-f]+)", text or "", re.IGNORECASE)
    return int(match[1], 16) if match else None


def plan_guid(text: str | None) -> str | None:
    """The power plan GUID in ``powercfg /getactivescheme``'s output, lower case."""
    match = GUID.search(text or "")
    return match[0].lower() if match else None


def active_plan() -> str | None:
    return plan_guid(run_text(["powercfg", "/getactivescheme"]))


def plan_letter(guid: str | None) -> str:
    """Part 1's letter for a power plan (a balanced, h high performance, u ultimate, p power saver); - for another."""
    return PLANS.get(guid or "", "-")


def processes(context, tokens: dict, details: dict) -> None:
    """p: the processes on each GPU before this script starts any."""
    text = nvidia_smi()
    if text is None:
        raise RuntimeError("nvidia-smi gave no process table")
    found = details["processes"] = gpu_processes(text, context.gpus)
    tokens["p"] = "/".join(str(len(found[gpu])) for gpu in context.gpus)


def remote_desktop(tokens: dict, details: dict) -> None:
    """u: Remote Desktop's GPU policy and this session."""
    value, session = remote_desktop_policy(), os.environ.get("SESSIONNAME")
    details["remote_desktop"] = {"bEnumerateHWBeforeSW": value, "SESSIONNAME": session}
    tokens["u"] = ("-" if value is None else str(value)) + session_letter(session)


def pcie_power_saving(tokens: dict, details: dict) -> None:
    """e: PCIe link state power management of the current plan, on AC power."""
    index = details["pcie_aspm_ac"] = aspm_index(run_text(["powercfg", "/query", "SCHEME_CURRENT", *ASPM]))
    if index is None:
        raise RuntimeError("powercfg shows no PCIe link state power management")
    tokens["e"] = str(index)


def plan_numbers(worker) -> list[float]:
    """us per GPU call, CPU benchmark ms and median page s, in ``worker``."""
    return [worker.ask("latency"), worker.ask("cpu"), worker.ask("timed", "median", 1)[2][0]["total"]]


def power_plan_test(context, worker, tokens: dict, details: dict, problems: list) -> None:
    """h: plan_numbers on the current power plan, then on High performance; the plan found is set again afterwards,
    whatever happens."""
    worker.ask("page", "median", context.median_image)
    worker.ask("first", "median")  # GPU code loaded and memory kept before anything is timed
    before, original = plan_numbers(worker), active_plan()
    plan = details["power_plan"] = {"found": original, "numbers_found": before}
    high = [None] * 3
    if original == HIGH_PERFORMANCE:
        plan["already_high"] = True
    elif original is None:
        problems.append("power plan test: powercfg /getactivescheme named no plan, so it was not switched")
    else:
        try:
            if run_text(["powercfg", "/setactive", HIGH_PERFORMANCE]) is None:
                problems.append("power plan test: powercfg could not switch to High performance")
            else:
                time.sleep(3)
                plan["during"] = active_plan()
                high = plan["numbers_high"] = plan_numbers(worker)
        finally:
            plan["restored"] = run_text(["powercfg", "/setactive", original]) is not None and active_plan() == original
            if not plan["restored"]:
                problems.append(f"the power plan was NOT set back: run  powercfg /setactive {original}")
    tokens["h"] = "/".join((num(before[0], 0), num(high[0], 0), num(before[1], 0), num(high[1], 0), num(before[2]),
                            num(high[2])))


def gpus_section(context, tokens: dict, details: dict, problems: list) -> None:
    """a: a worker on every GPU (started as cd_batch starts its workers), the round trip of each in turn while the
    others wait; then h, the power plan test, in the first of them."""
    step, attempt = progress(context.started, problems)
    workers = []
    try:
        for gpu in context.gpus:
            workers.append(Worker(gpu, context.settings, context.threads))
        latency = [worker.ask("latency") for worker in workers]
        details.update(round_trip_us=latency, seen_gpus=[worker.gpu for worker in workers])
        tokens["a"] = "/".join(num(us, 0) for us in latency)
        step(f"the power plan test on GPU {context.gpus[0]}: page {context.median_page} now, then on High performance "
             "(the plan found is set back afterwards)")
        attempt("the power plan test", lambda: power_plan_test(context, workers[0], tokens, details, problems))
    finally:
        for worker in workers:
            worker.close()


# ---------------------------------------------------------------------------------------- section 2: workers per GPU
def read_rows(path: Path) -> list[dict]:
    """Every row of a CSV (none when it is missing)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def column(rows: list[dict], key: str) -> list[float]:
    """The finite numbers of one column (empty cells left out)."""
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return [value for value in values if np.isfinite(value)]


def trial(folder: Path) -> dict | None:
    """One earlier cd_batch run, from its files alone; None when the folder holds none of 20 hole pages or more. Its
    wall time runs from run.json's "created" (the start) to the last change of pages.csv (the end)."""
    if not all(os.path.isfile(folder / name) for name in ("run.json", "timings.csv", "pages.csv")):
        return None
    pages = read_rows(folder / "pages.csv")
    holes = sum(row.get("status") == "ok" for row in pages)
    if holes < 20:
        return None
    created = json.loads((folder / "run.json").read_text(encoding="utf-8"))["created"]
    wall = (folder / "pages.csv").stat().st_mtime - datetime.strptime(created, "%Y-%m-%d %H:%M:%S").timestamp()
    wall = wall if wall > 0 else float("nan")
    timings = read_rows(folder / "timings.csv")
    workers = {}  # GPU -> its worker ids
    for row in timings:
        workers.setdefault(row.get("gpu", ""), set()).add(row.get("worker", ""))
    total = sum(len(ids) for ids in workers.values()) or float("nan")
    seconds = column([row for row in timings if row.get("status") == "ok"], "measure_s")
    memory = column(timings, "peak_rss_mb")
    return {"name": folder.name, "workers_per_gpu": max(map(len, workers.values()), default=0), "workers": total,
            "pages": len(pages), "hole_pages": holes, "wall_s": wall, "hole_pages_per_s": holes / wall,
            "median_page_s": float(np.median(seconds)) if seconds else None,
            "busy_pct": 100 * sum(seconds) / (total * wall), "read_s": sum(column(timings, "read_s")),
            "peak_memory_gb": max(memory) * total / 1024 if memory else None}


def earlier_trials(run: Path) -> list[dict]:
    """trial() of every folder beside RUN_FOLDER (itself included) that holds one, fewest workers per GPU first."""
    folders = sorted(path for path in run.resolve().parent.iterdir() if os.path.isdir(path))
    return sorted((found for found in map(trial, folders) if found), key=lambda t: (t["workers_per_gpu"], t["name"]))


def trial_token(found: dict) -> str:
    return "/".join((str(found["workers_per_gpu"]), num(found["hole_pages_per_s"], 2), num(found["median_page_s"]),
                     num(found["busy_pct"], 0), num(found["peak_memory_gb"])))


def trials_section(context, tokens: dict, details: dict) -> None:
    """t: the earlier trial folders."""
    found = details["trials"] = earlier_trials(context.run)
    if found:
        tokens["t"] = "+".join(trial_token(t) for t in found)


def page_window(rows: list[dict], size: int = WINDOW) -> tuple[int, int]:
    """The ``size`` consecutive pages A:B of a TIFF that hold the most pages of status ok (the first such window; all
    pages when the file has fewer). ``rows``: its pages.csv rows."""
    if not rows:
        raise RuntimeError("pages.csv of RUN_FOLDER has no row of TIFF")
    count = max(int(row["page"]) for row in rows) + 1
    if count <= size:
        return 0, count
    holes = np.zeros(count, int)
    holes[[int(row["page"]) for row in rows if row.get("status") == "ok"]] = 1
    start = int(np.argmax(np.convolve(holes, np.ones(size, int), mode="valid")))  # [a]: hole pages in a..a+size-1
    return start, start + size


def batch_command(tiff: Path, output: Path, pages: tuple[int, int], workers_per_gpu: int, settings: dict) -> list[str]:
    """The cd_batch command that measures pages A:B of TIFF into ``output`` with ``workers_per_gpu`` and the settings
    of a run.json (its measure and hole_check written back as the flags they came from)."""
    measure, check = settings["measure"], settings["hole_check"]
    command = [sys.executable, str(ROOT / "scripts" / "cd_batch.py"), str(tiff), "--output", str(output),
               "--pages", f"{pages[0]}:{pages[1]}", "--workers-per-gpu", str(workers_per_gpu),
               "--methods", *measure["methods"]]
    for key, flag in FLAGS.items():
        command += [flag, str(measure[key])]
    command += ["--invert"] if measure["invert"] else []
    if check is None:
        return command + ["--no-hole-check"]
    command += ["--reference", check["reference"], "--reference-page", str(check["reference_page"]),
                "--max-error", str(check["max_error"])]
    return command + (["--error-function", check["error_function"]] if check["error_function"] else [])


def settings_differences(a: dict, b: dict) -> int:
    """How many settings differ between two run.json "settings": the keys of measure and of hole_check (a part that is
    null in one of them and not in the other counts once)."""
    count = 0
    for part in set(a) | set(b):
        x, y = a.get(part), b.get(part)
        if isinstance(x, dict) and isinstance(y, dict):
            count += sum(x.get(key) != y.get(key) for key in set(x) | set(y))
        else:
            count += x != y
    return count


def ok_rows(path: Path) -> int:
    """Complete rows of status ok in a pages.csv being written (a row counts once its line has ended)."""
    try:
        lines = path.read_bytes().decode("utf-8", "replace").split("\n")[1:-1]
    except OSError:  # not created yet, or being replaced at this moment
        return 0
    return sum(row[2:3] == ["ok"] for row in csv.reader(lines))


def middle_half(run: dict, expected: int) -> tuple[float, float, int] | None:
    """When cd_batch measured the middle half of the hole pages (workers starting and the last pages left out): the
    arrival of the 25 % row and of the 75 % row, and the rows between them. ``expected``: the hole pages it was given;
    rows that never arrived (it stopped first) count as arriving when it stopped, so that a run that stalled does not
    look like a fast one. None below 8 rows."""
    arrivals = run["arrivals"]
    n = max(len(arrivals), expected)
    if n < 8:
        return None
    times = arrivals + [run["end"]] * (n - len(arrivals))
    return times[n // 4], times[3 * n // 4], 3 * n // 4 - n // 4


def hole_rate(run: dict, expected: int) -> float | None:
    """Hole pages per second over the middle half of the hole pages."""
    middle = middle_half(run, expected)
    return middle[2] / (middle[1] - middle[0]) if middle and middle[1] > middle[0] else None


def usage(run: dict, expected: int, gpu_samples: list, windows_samples: list) -> dict:
    """What one worker count used while cd_batch ran: the most GPU memory on one GPU and the most GPU memory moved to
    system RAM on one adapter (GB), the most committed memory (% of the limit), the least available RAM (MB), the most
    pages per second, and the mean CPU % over the middle half of the hole pages."""
    start, end = run["start"], run["end"]
    during = [values for stamp, values in windows_samples if start <= stamp <= end]
    first, last, _ = middle_half(run, expected) or (np.inf, -np.inf, 0)
    middle = [values["u"] for stamp, values in windows_samples if first <= stamp <= last and "u" in values]
    memory = [sample[5] for sample in gpu_samples if start <= sample[0] <= end]
    shared = [adapter["s"] for values in during for adapter in values.get("g") or []]
    commit = [100 * values["c"] / values["l"] for values in during if values.get("l")]
    available = [values["a"] for values in during if "a" in values]
    paging = [values["p"] for values in during if "p" in values]
    return {"gpu_gb": max(memory) / 1024 if memory else None, "shared_gb": max(shared) / 2**30 if shared else None,
            "commit_pct": max(commit) if commit else None, "cpu_pct": float(np.mean(middle)) if middle else None,
            "available_mb_least": min(available) if available else None,
            "pages_per_s_most": max(paging) if paging else None}


def setting_token(result: dict | None) -> str:
    """One worker count in k ("-": it did not run)."""
    if result is None:
        return "-"
    least = result["available_mb_least"]
    return "/".join((num(result["hole_pages_per_s"], 2), num(result["median_page_s"]), num(result["gpu_gb"]),
                     num(result["shared_gb"]), num(result["commit_pct"], 0), num(result["cpu_pct"], 0),
                     num(None if least is None else least / 1024)))


class WindowsSampler(Sampler):
    """Windows' counters every ~2 s from one PowerShell loop (its first query takes seconds, so it runs throughout):
    committed memory (c) and its limit (l), available MB (a), pages per s (p), CPU % (u), and per GPU adapter (g) its
    name (n), dedicated (d) and shared (s) memory in bytes; shared is GPU memory moved to system RAM. ``samples``: (time
    received, values)."""

    def __init__(self):
        self.samples: list[tuple[float, dict]] = []
        self.thread = None
        try:
            self.process = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", WMI_LOOP],
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True, errors="replace")
        except OSError:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            try:
                self.samples.append((time.time(), json.loads(line)))
            except ValueError:
                continue


def gpu_memory() -> dict[str, float]:
    """MiB in use on each GPU, by nvidia-smi index (none when nvidia-smi fails)."""
    out = nvidia_smi("--query-gpu=index,memory.used", "--format=csv,noheader,nounits") or ""
    pairs = (line.split(",") for line in out.splitlines() if line.strip())
    return {index.strip(): float(used) for index, used in pairs}


def wait_until_free(before: dict[str, float], seconds: float = 30, margin: float = 300) -> dict[str, float]:
    """Wait until no GPU holds more than ``margin`` MiB above ``before`` (what a worker count left is gone), at most
    ``seconds``; the MiB still above it per GPU (empty: none)."""
    deadline = time.time() + seconds
    while True:
        excess = {gpu: used - before.get(gpu, 0.0) for gpu, used in gpu_memory().items()
                  if used - before.get(gpu, 0.0) > margin}
        if not excess or time.time() > deadline:
            return excess
        time.sleep(1)


def kill_tree(process: subprocess.Popen) -> None:
    """End a process and every process it started (Popen.kill would leave cd_batch's workers running)."""
    try:
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        process.kill()
    process.wait(60)


def run_batch(command: list[str], folder: Path) -> dict:
    """Run cd_batch with its output in folder/batch.log, noting when each hole page's row arrives in folder/pages.csv
    (looked at every 0.5 s). It is stopped after BATCH_TIMEOUT, or after STALL without a new hole page (on a GPU too
    small for its workers it can stall for good), and whatever is left of it is ended in any case."""
    arrivals = []
    with (folder / "batch.log").open("wb") as log:
        start = time.time()
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            while process.poll() is None:
                time.sleep(0.5)
                now = time.time()
                arrivals += [now] * max(0, ok_rows(folder / "pages.csv") - len(arrivals))
                if now - start > BATCH_TIMEOUT or now - (arrivals[-1] if arrivals else start) > STALL:
                    break
        finally:
            stopped = process.poll() is None
            kill_tree(process)
    end = time.time()
    arrivals += [end] * max(0, ok_rows(folder / "pages.csv") - len(arrivals))  # rows of its last half second
    return {"start": start, "end": end, "arrivals": arrivals, "exit_code": process.returncode, "stopped": stopped}


def batch_setting(context, workers: int, window: tuple[int, int], expected: int,
                  windows_sampler: WindowsSampler) -> dict:
    """cd_batch with ``workers`` per GPU on the window's pages (``expected`` of them hole pages), in a temporary folder
    removed afterwards: its speed, the memory and CPU it used, how many settings differ from the run's."""
    folder = Path(tempfile.mkdtemp(prefix=f"cd_diagnose_w{workers}_"))
    try:
        before = gpu_memory()
        sampler = Sampler()
        try:
            run = run_batch(batch_command(context.tiff, folder, window, workers, context.run_info["settings"]), folder)
        finally:
            gpu_samples = sampler.stop()
        saved = folder / "run.json"
        settings = json.loads(saved.read_text(encoding="utf-8"))["settings"] if saved.is_file() else None
        seconds = column([row for row in read_rows(folder / "timings.csv") if row.get("status") == "ok"], "measure_s")
        failed = len(read_rows(folder / "failed.csv"))
        log = (folder / "batch.log").read_text(encoding=locale.getpreferredencoding(False), errors="replace")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return {"workers_per_gpu": workers, "exit_code": run["exit_code"], "stopped": run["stopped"],
            "seconds": run["end"] - run["start"], "hole_pages": len(run["arrivals"]), "expected": expected,
            "failed_pages": failed,
            "hole_pages_per_s": hole_rate(run, expected),
            "median_page_s": float(np.median(seconds)) if seconds else None,
            "differ": None if settings is None else settings_differences(settings, context.run_info["settings"]),
            "left_on_gpus_mib": wait_until_free(before), "log": log.strip().splitlines()[-5:],
            **usage(run, expected, gpu_samples, windows_sampler.samples)}


def setting_problems(result: dict) -> list[str]:
    """What went wrong in one worker count, in words (ASCII: the batch's own words may be in Windows' language)."""
    title, found = f"cd_batch with {result['workers_per_gpu']} per GPU", []
    if result["stopped"]:
        found.append(f"{title}: stopped after {result['seconds'] / 60:.0f} min with {result['hole_pages']} of "
                     f"{result['expected']} hole pages done (at most {BATCH_TIMEOUT // 60} min, or {STALL // 60} min "
                     "without a new one)")
    elif result["exit_code"]:
        last = result["log"][-1] if result["log"] else ""
        found.append(f"{title}: exit code {result['exit_code']}, {result['failed_pages']} pages failed: "
                     + last.encode("ascii", "replace").decode())
    if result["left_on_gpus_mib"]:
        left = ", ".join(f"GPU {gpu} {mib:.0f}" for gpu, mib in result["left_on_gpus_mib"].items())
        found.append(f"after {title}: GPU memory still above its level before it, 30 s after it ended (MiB: {left})")
    return found


def scaling_section(context, tokens: dict, details: dict, problems: list) -> None:
    """k and n: the real cd_batch with 1-4 workers per GPU on the WINDOW pages of TIFF with the most hole pages."""
    step, attempt = progress(context.started, problems)
    rows = csv_rows(context.run / "pages.csv", context.tiff)
    window = page_window(rows)
    expected = sum(row.get("status") == "ok" and window[0] <= int(row["page"]) < window[1] for row in rows)
    results = []
    details["scaling"] = {"pages": list(window), "hole_pages": expected, "settings": results}
    windows_sampler = WindowsSampler()  # started first: its first query takes seconds
    try:
        step(f"reading pages {window[0]}:{window[1]} once, so that cd_batch reads them from memory, not the disk")
        reader = FrameReader(context.tiff)
        try:
            for page in range(window[0], min(window[1], reader.count)):
                reader.read(page)
        finally:
            reader.close()
        for workers in WORKERS_PER_GPU:
            step(f"cd_batch with {workers} worker{'s' if workers > 1 else ''} per GPU on pages {window[0]}:{window[1]}")
            attempt(f"cd_batch with {workers} per GPU",
                    lambda: results.append(batch_setting(context, workers, window, expected, windows_sampler)))
    finally:
        windows_sampler.stop()
    for result in results:
        problems.extend(setting_problems(result))
    done = {result["workers_per_gpu"]: result for result in results}
    tokens["k"] = "+".join(setting_token(done.get(workers)) for workers in WORKERS_PER_GPU)
    differ = [result["differ"] for result in results if result["differ"] is not None]
    if differ:
        tokens["n"] = str(max(differ))


# ---------------------------------------------------------------------------------------- section 3: deblend drift
def finite_median(values, scale: float = 1.0) -> float | None:
    """``scale`` times the median of the finite values; None without one."""
    values = [value for value in values if value is not None and np.isfinite(value)]
    return scale * float(np.median(values)) if values else None


def last_pass(found: dict) -> list[dict]:
    """The trace rows of a measurement's last deblend pass (one variant() answer)."""
    return found["passes"][-1]["rounds"] if found.get("passes") else []


def tiles_pct(found: dict) -> float | None:
    """Background tiles measured at the end, % of all tiles (0 when the map was inferred; None for a fixed level)."""
    return 100 * found["measured_tiles"] / found["tiles"] if found.get("tiles") else None


def hole_number(comparison: dict | None, key: str) -> str:
    """A per-hole number of compare(): d when the hole lists differ, - when there is no comparison."""
    if not comparison:
        return "-"
    return num(comparison[key], 2) if comparison["same_holes"] else "d"


def reference_token(found: dict, against_100: dict | None) -> str:
    """d: rounds / passes / the last raw step in units of the tight stop's limit (below 1: it stopped there) /
    page-mean change between 100 rounds and the reference / measured tiles % / background level change over the last
    pass."""
    rows = last_pass(found)
    step = rows[-1]["step_p99"] / TIGHT_TOLERANCE if rows else None
    background = rows[-1]["bg_mean"] - rows[0]["bg_mean"] if rows else None
    return "/".join((str(found["rounds"]), str(len(found["passes"])), num(step, 1),
                     num((against_100 or {}).get("page_mean"), 2), num(tiles_pct(found), 1), num(background, 1)))


def movement_token(rows: list[dict]) -> str | None:
    """m, from the trace rows of one pass: its last round's centre and width p99 and its p99 in the three edge bands,
    in units of the tight stop's limit; over its last 12 rounds, the medians of: the background map's largest change
    per round, each round's step against the round before (%), the cosine between consecutive steps (%)."""
    if not rows:
        return None
    last, window, recent = rows[-1], rows[-12:], rows[-13:]
    ratios = [b["step_p99"] / a["step_p99"] for a, b in zip(recent, recent[1:]) if a["step_p99"] > 0]
    moving = [last["centre_p99"], last["width_p99"], *last["edge_p99"]]
    return "/".join([num(value / TIGHT_TOLERANCE, 0) for value in moving] + [
        num(finite_median(row["bg_max"] for row in window), 2), num(finite_median(ratios, 100), 0),
        num(finite_median((row["cos"] for row in window), 100), 0)])


def holes_pct(comparison: dict) -> str:
    """Holes that moved more than 1 unit, % of the holes compared (a count of tens of thousands is long to type); d
    when the hole lists differ."""
    if not comparison["same_holes"]:
        return "d"
    holes = comparison["holes"][0]
    return num(100 * comparison["holes_above"] / holes, 0) if holes else "-"


def change_token(comparison: dict) -> str:
    """c: page-mean change of the cell methods / of the wide methods / median hole change / holes that moved more
    than 1 unit, % / valid flags that flip."""
    return "/".join((num(comparison["page_mean_cell"], 2), num(comparison["page_mean_wide"], 2),
                     hole_number(comparison, "hole_median"), holes_pct(comparison), str(comparison["valid_changed"])))


def candidate_token(found: dict | None, comparison: dict | None, tiles: bool = False) -> str:
    """One candidate of x: rounds / page-mean change from the reference / hole p99 (/ measured tiles %)."""
    if not found:
        return "-"
    numbers = [str(found["rounds"]), num((comparison or {}).get("page_mean"), 2), hole_number(comparison, "hole_p99")]
    return "/".join(numbers + ([num(tiles_pct(found), 1)] if tiles else []))


def slowest_token(found: dict, compared: dict) -> str:
    """r: the reference's rounds / passes / today against it, then for continuing after the re-check and for that
    with Anderson: rounds / passes / s / against the reference."""
    def rounds(measured):
        return (str(measured["rounds"]), str(len(measured["passes"]))) if measured else ("-", "-")

    def page_mean(key):
        return num((compared.get(key) or {}).get("page_mean"), 2)

    def cure(key):
        return (*rounds(found.get(key)), num((found.get(key) or {}).get("seconds"), 1), page_mean(key))
    return "/".join((*rounds(found.get("reference_s")), page_mean("today_s"), *cure("continued_s"), *cure("cured_s")))


def drift_tokens(found: dict, compared: dict) -> dict:
    """d m c x r from section 3's measurements (variant() answers) and comparisons with their page's reference
    (compare() answers), by key; a token none of whose measurements was made is left out."""
    tokens = {}
    if found.get("reference"):
        tokens["d"] = reference_token(found["reference"], compared.get("ref100"))
    if found.get("today"):
        tokens["m"] = movement_token(last_pass(found["today"]))
    if compared.get("today"):
        tokens["c"] = change_token(compared["today"])
    if any(found.get(key) for key, _, _ in CANDIDATES):
        tokens["x"] = "+".join(candidate_token(found.get(key), compared.get(key), key == "tiles16")
                               for key, _, _ in CANDIDATES)
    if any(found.get(key) for key in ("reference_s", "continued_s", "cured_s")):
        tokens["r"] = slowest_token(found, compared)
    return tokens


def page_spread(rows: list[dict]) -> dict[str, float | None]:
    """Per method (a <method>_cd column of pages.csv): the standard deviation of its page means over the hole pages
    (rows of status ok); None below two pages."""
    holes = [row for row in rows if row.get("status") == "ok"]
    spread = {}
    for key in (rows[0] if rows else {}):
        if key.endswith("_cd"):
            values = column(holes, key)
            spread[key[:-3]] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return spread


def spread_token(spread: dict, pages: int) -> str:
    """f: the largest spread of the cell methods / of the wide methods / hole pages."""
    def largest(names):
        return max((spread[name] for name in names if spread.get(name) is not None), default=None)
    return f"{num(largest(CELL_STAGES), 2)}/{num(largest(WIDE_STAGES), 2)}/{pages}"


def drift_plan(cap: int) -> list[tuple]:
    """Section 3's measurements in order: key, page, Variant switches, rounds allowed, the key it is compared with,
    what it is. today and today_s are what the batch computes (today's loop, stop and rounds). The references stop at
    TIGHT_TOLERANCE, where the loop ends when it runs on; the candidates and the slowest page's cures at
    PRACTICAL_TOLERANCE, a stop a batch could afford; all of them after REFERENCE_ROUNDS rounds at the latest (100:
    ref100). Each page's reference comes first, so that every other measurement is compared with it at once."""
    tight, practical = {"tolerance": TIGHT_TOLERANCE}, {"tolerance": PRACTICAL_TOLERANCE}
    end, stop = f"stop at {TIGHT_TOLERANCE:g}", f"stop at {PRACTICAL_TOLERANCE:g}"
    plan = [("reference", "median", tight, REFERENCE_ROUNDS, None,
             f"the reference: {end}, at most {REFERENCE_ROUNDS} rounds"),
            ("today", "median", {}, cap, "reference", f"today's {cap} rounds"),
            ("ref100", "median", tight, 100, "reference", f"{end}, at most 100 rounds")]
    plan += [(key, "median", {**switches, **practical}, REFERENCE_ROUNDS, "reference", f"{title}, {stop}")
             for key, switches, title in CANDIDATES]
    return plan + [
        ("reference_s", "slowest", tight, REFERENCE_ROUNDS, None,
         f"the reference: {end}, at most {REFERENCE_ROUNDS} rounds"),
        ("today_s", "slowest", {}, cap, "reference_s", f"today's {cap} rounds"),
        ("continued_s", "slowest", {"continue_after_recheck": True, **practical}, REFERENCE_ROUNDS, "reference_s",
         f"continuing after the seed re-check, {stop}"),
        ("cured_s", "slowest", {"continue_after_recheck": True, "anderson": 5, "freeze_free": True, **practical},
         REFERENCE_ROUNDS, "reference_s", f"continuing after the re-check and Anderson, {stop}")]


def drift_section(context, tokens: dict, details: dict, problems: list) -> None:
    """y d m c x r f, deblend drift: the page-to-page spread of the run's page means (pages.csv); then in one worker
    on the first GPU, the median and the slowest page with today's loop, run on to the tight stop (the reference; on
    the median page also at most 100 rounds) and with each cure to the practical stop (drift_plan), every round of
    every pass recorded, each compared with its page's reference."""
    step, attempt = progress(context.started, problems)
    cap = int(context.settings.get("deblend_iterations", 30))
    found, compared = {}, {}
    drift = details["drift"] = {"today_rounds": cap, "reference_rounds": REFERENCE_ROUNDS,
                                "tight_tolerance": TIGHT_TOLERANCE, "practical_tolerance": PRACTICAL_TOLERANCE,
                                "measured": found, "compared": compared}
    tokens["y"] = f"{context.median_page}/{context.slowest_page}"

    def spread():
        rows = csv_rows(context.run / "pages.csv", context.tiff)
        drift["spread"] = page_spread(rows)
        drift["hole_pages"] = sum(row.get("status") == "ok" for row in rows)
        tokens["f"] = spread_token(drift["spread"], drift["hole_pages"])
    attempt("the page-to-page spread", spread)
    pages = {"median": context.median_page, "slowest": context.slowest_page}
    worker = Worker(context.gpus[0], context.settings, context.threads)
    try:
        worker.ask("page", "median", context.median_image)
        worker.ask("page", "slowest", context.slowest_image)

        def measure(key, page, switches, rounds, reference):
            found[key] = worker.ask("variant", key, page, switches, rounds, {"deblend_iterations": rounds})
            if reference in found:
                compared[key] = worker.ask("compare", reference, key)
        for key, page, switches, rounds, reference, title in drift_plan(cap):
            step(f"page {pages[page]}: {title}")
            attempt(f"{title} on page {pages[page]}", lambda: measure(key, page, switches, rounds, reference))
    finally:
        worker.close()
    tokens.update(drift_tokens(found, compared))


# ------------------------------------------------------------------------------------------------------------ main
def main(context) -> int:
    """Sections 1-3 (see the top of this file), the summary, diagnosis2.txt and the six lines."""
    tokens, problems = {}, []
    step, attempt = progress(context.started, problems)
    details = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "tiff": str(context.tiff), "run": str(context.run),
               "pages": {"median": context.median_page, "slowest": context.slowest_page}, "gpus_used": context.gpus,
               "threads_per_worker": context.threads, "run_commit": context.run_info.get("git_commit")}

    step("code; processes on the GPUs; Remote Desktop; PCIe power saving")
    files = code_files()
    code, each = fingerprint(files)
    tokens["v"] = f"{code}/{git_commit(files)}/{(context.run_info.get('git_commit') or '-')[:5]}"
    details.update(code=code, files=each)
    attempt("processes on the GPUs", lambda: processes(context, tokens, details))
    attempt("Remote Desktop", lambda: remote_desktop(tokens, details))
    attempt("PCIe power saving", lambda: pcie_power_saving(tokens, details))

    step(f"a worker on each of {len(context.gpus)} GPUs: the GPU round trip of each, one at a time")
    attempt("a worker on every GPU", lambda: gpus_section(context, tokens, details, problems))
    details["power_plan_after"] = active_plan()
    tokens["w"] = plan_letter(details["power_plan_after"])

    step(f"workers per GPU: the earlier trial folders beside {context.run.resolve().name}")
    attempt("the earlier trials", lambda: trials_section(context, tokens, details))
    attempt("workers per GPU", lambda: scaling_section(context, tokens, details, problems))

    step(f"deblend drift on pages {context.median_page} and {context.slowest_page}")
    attempt("deblend drift", lambda: drift_section(context, tokens, details, problems))

    lines = token_lines(tokens, LINES)
    text = summary(tokens, details, problems, context)
    path = context.run / "diagnosis2.txt"
    try:
        path.write_text(text + "\n\n" + "\n".join(lines) + "\n\nDetails\n"
                        + json.dumps(details, indent=1, default=_plain) + "\n", encoding="utf-8")
        saved = f"All details: {path}"
    except OSError as error:
        saved = f"diagnosis2.txt could not be written: {error}"
    print(f"\n{text}\n{saved}\nDone in {time.perf_counter() - context.started:.0f} s.\n\n"
          f"Type these {len(lines)} lines to Claude:\n" + "\n".join(lines), flush=True)
    return 0


def summary(tokens: dict, details: dict, problems: list[str], context) -> str:
    """The same values in words, for whoever runs the script."""
    t = lambda key: tokens.get(key) or "-"  # noqa: E731
    plan = details.get("power_plan") or {}
    if plan.get("already_high"):
        switched = "already High performance, so not switched"
    elif "restored" in plan:
        switched = "the plan found was set back" if plan["restored"] else "the plan found was NOT set back"
    else:
        switched = "not switched"
    trials = ", ".join(found["name"] for found in details.get("trials") or []) or "none found"
    window = (details.get("scaling") or {}).get("pages")
    pages = f"pages {window[0]}:{window[1]}" if window else "160 pages"
    end, stop = f"{TIGHT_TOLERANCE:g}", f"{PRACTICAL_TOLERANCE:g}"
    cap = (details.get("drift") or {}).get("today_rounds", 30)
    lines = [
        f"Code         {t('v')}  (files / git commit now / the run's commit)",
        f"GPUs         round trip {t('a')} us (each GPU, one at a time); processes on each GPU before: {t('p')}",
        f"Windows      Remote Desktop GPU policy and this session {t('u')} (1 on, 0 off, - not set: software "
        f"rendering; r Remote Desktop, c console); PCIe power saving {t('e')} (0 off, 1 moderate, 2 maximum)",
        f"Power plan   {t('h')} (us per GPU call, CPU ms, page {context.median_page} s: each on the plan found / on "
        f"High performance); {switched}; plan now {t('w')}",
        f"Trials       {t('t')} (workers per GPU / hole pages per s / median page s / busy % / memory GB): {trials}",
        f"Workers      {pages}, 1 + 2 + 3 + 4 per GPU: {t('k')} (hole pages per s / median page s / GPU GB / moved to "
        f"RAM GB / committed % / CPU % / least free RAM GB); settings that differ from the run's: {t('n')}",
        f"Deblend      pages {t('y')} (median / slowest). The loop stops at a step of 1e-3 after at most {cap} rounds "
        f"today; the references at {end} (where it ends), the cures at {stop}, each after {REFERENCE_ROUNDS} at most",
        f"             reference {t('d')} (rounds / passes / last step, 1 = {end} / page-mean change from at most 100 "
        "rounds / measured tiles % / background change)",
        f"             today's last 12 rounds {t('m')} (centre / width p99 / p99 < 1, 1-2, >= 2 pitches from the "
        f"edge, 1 = {end} / background per round / step against the round before % / cosine %)",
        f"             today against the reference {t('c')} (page mean cell / wide methods / median hole / holes over 1 "
        "unit % / valid flips)",
        f"             cures {t('x')} (LM 100 steps + background frozen + Anderson + 2 % wider + tiles from 16 px: "
        "rounds / page mean / hole p99 against the reference / tiles %)",
        f"Page {context.slowest_page:<7} slowest: {t('r')} (reference rounds / passes / today against it / continuing "
        "after the re-check, then that with Anderson: rounds / passes / s / against the reference)",
        f"Spread       page means of the run {t('f')} (largest standard deviation of the cell methods / wide methods / "
        "hole pages)",
        "             d: the hole lists differ from the reference's; -: not measured (a PROBLEM line says why) or "
        "nothing to measure (an empty edge band, a pass of one round)"]
    return "\n".join(lines + [f"PROBLEM      {p}" for p in problems])
