"""scripts/cd_diagnose_part2.py: the pieces that decide what the five typed lines say (no GPU needed)."""
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import cd_batch, cd_diagnose_part2 as part2
from scripts.cd_diagnose import compare
from scripts.cd_report import ALL_METHODS

TIGHT = part2.TIGHT_TOLERANCE  # the unit of d's and m's steps


def page_rows(count, holes):
    """pages.csv rows of one TIFF: pages 0..count-1, status ok on the pages in ``holes``."""
    return [{"file": "scan.tif", "page": str(p), "status": "ok" if p in holes else "no_holes"} for p in range(count)]


def test_the_scaling_window_holds_the_most_hole_pages():
    holes = set(range(100, 260)) - {120, 130} | {5, 7, 390}
    assert part2.page_window(page_rows(400, holes)) == (100, 260)
    assert part2.page_window(page_rows(48, {3, 4})) == (0, 48)  # fewer pages than the window: all of them
    assert part2.page_window(page_rows(160, set())) == (0, 160)
    assert part2.page_window(page_rows(10, {0, 1, 6, 7}), size=2) == (0, 2)  # a tie: the first window
    assert part2.page_window(page_rows(10, {1, 6, 7}), size=2) == (6, 8)
    with pytest.raises(RuntimeError, match="no row"):
        part2.page_window([])


def test_the_rate_leaves_out_the_start_and_the_last_pages_but_not_a_stall():
    arrivals = [10.0 + 2 * k for k in range(16)]  # a row every 2 s after 10 s of start-up
    assert part2.hole_rate({"arrivals": arrivals, "end": 41.0}, 16) == pytest.approx(0.5)
    slow_end = arrivals[:13] + [100.0, 200.0, 300.0]
    assert part2.hole_rate({"arrivals": slow_end, "end": 301.0}, 16) == pytest.approx(0.5)  # changes nothing
    assert part2.hole_rate({"arrivals": arrivals[:7], "end": 30.0}, 7) is None  # too few rows
    assert part2.hole_rate({"arrivals": [5.0] * 8, "end": 6.0}, 8) is None
    # Stopped after 9 of 16 hole pages: rows 9-15 count as done when it stopped (at 1000 s), not left out.
    assert part2.hole_rate({"arrivals": arrivals[:9], "end": 1000.0}, 16) == pytest.approx(8 / (1000 - 18))
    assert part2.hole_rate({"arrivals": arrivals[:3], "end": 1000.0}, 16) is None  # not even a quarter done
    assert part2.middle_half({"arrivals": arrivals, "end": 41.0}, 16) == (18.0, 34.0, 8)


def test_hole_rows_count_once_their_line_has_ended(tmp_path):
    path = tmp_path / "pages.csv"
    assert part2.ok_rows(path) == 0  # not written yet
    path.write_bytes(b"file,page,status,pcc_error\r\nscan.tif,0,ok,0.1\r\nscan.tif,1,no_holes,0.6\r\n"
                     b"\"a,b.tif\",2,ok,0.1\r\nscan.tif,3,ok,0.")
    assert part2.ok_rows(path) == 2


def write_trial(folder, created, wall, workers, seconds, memory, holes=20, others=4):
    """A cd_batch output folder: run.json, pages.csv (``holes`` ok rows, then ``others`` no_holes rows, its last change
    ``wall`` s after ``created``) and timings.csv, its rows handed out in turn to ``workers`` [(gpu, worker), ...]."""
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"schema": 1, "settings": {}, "created": created}), encoding="utf-8")
    statuses = ["ok"] * holes + ["no_holes"] * others
    pages = "".join(f"scan.tif,{p},{status}\n" for p, status in enumerate(statuses))
    (folder / "pages.csv").write_text("file,page,status\n" + pages, encoding="utf-8")
    timings = "".join(f"scan.tif,{p},{status},{workers[p % len(workers)][0]},{workers[p % len(workers)][1]},0.5,"
                      f"{seconds if status == 'ok' else ''},{memory[p % len(memory)]}\n"
                      for p, status in enumerate(statuses))
    (folder / "timings.csv").write_text("file,page,status,gpu,worker,read_s,measure_s,peak_rss_mb\n" + timings,
                                        encoding="utf-8")
    (folder / "invocations.jsonl").write_text(json.dumps({"elapsed_s": wall,
        "counts": {"ok": holes, "no_holes": others, "failed": 0}}) + "\n", encoding="utf-8")
    start = datetime.strptime(created, "%Y-%m-%d %H:%M:%S").timestamp()
    os.utime(folder / "pages.csv", (start + wall, start + wall))


def test_earlier_trials_from_their_files(tmp_path):
    results = tmp_path / "results"
    write_trial(results / "trial_w2", "2026-10-26 11:00:00", 50, [("0", 0), ("1", 1), ("0", 2), ("1", 3)], 6.0,
                [4096.0])
    write_trial(results / "trial_w1", "2026-10-26 10:00:00", 100, [("0", 0), ("1", 1)], 4.0, [3800.0, 4000.0])
    write_trial(results / "warmup", "2026-10-26 09:00:00", 30, [("0", 0)], 4.0, [3800.0], holes=10)
    (results / "notes").mkdir()
    found = part2.earlier_trials(results / "trial_w1")
    assert [t["name"] for t in found] == ["trial_w1", "trial_w2"]  # fewest workers per GPU first; warmup too short
    w1, w2 = found
    assert (w1["workers_per_gpu"], w1["workers"], w2["workers_per_gpu"], w2["workers"]) == (1, 2, 2, 4)
    assert w1["wall_s"] == pytest.approx(100) and w1["hole_pages_per_s"] == pytest.approx(0.2)
    assert w1["busy_pct"] == pytest.approx(100 * 20 * 4.0 / (2 * 100))  # all measuring seconds / (workers x wall)
    assert w2["busy_pct"] == pytest.approx(100 * 20 * 6.0 / (4 * 50))
    assert w1["read_s"] == pytest.approx(24 * 0.5) and w1["pages"] == 24 and w1["hole_pages"] == 20
    assert "+".join(map(part2.trial_token, found)) == "1/0.2/4/40/7.8+2/0.4/6/60/16"
    assert part2.trial(results / "notes") is None and part2.trial(results / "warmup") is None


def test_trial_rate_ignores_csv_timestamp_and_noop_resumes(tmp_path):
    folder = tmp_path / "run"
    write_trial(folder, "2026-10-26 10:00:00", 100, [("0", 0)], 4.0, [1000.0])
    original = part2.trial(folder)
    os.utime(folder / "pages.csv", None)  # copying or sorting is not measurement time
    with (folder / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"elapsed_s": 10000, "counts": {"ok": 0, "no_holes": 0, "failed": 0}}) + "\n")
    assert part2.trial(folder)["hole_pages_per_s"] == original["hole_pages_per_s"]
    (folder / "invocations.jsonl").unlink()
    assert np.isnan(part2.trial(folder)["wall_s"])  # old timestamps cannot establish runtime


def run_settings(reference, **changes):
    """A run.json "settings" as cd_batch writes it, with values other than its defaults."""
    measure = {"methods": [m for m in ALL_METHODS if m in ("projection", "half_area", "tail_area", "foot_area")],
               "seeds": "peaks", "invert": True, "area_min": 5, "area_max": 300, "pixel_size": 21.0, "unit": "nm",
               "half_area_level": 0.45, "half_area_peak_pixels": 3.0, "tail_area_level": 0.25, "tail_window": 1.3,
               "n_angles": 16, "outlines": False, "metrics_only": True,
               "deblend_iterations": 55, "deblend_tolerance": 0.0002}
    hole_check = {"reference": str(reference.resolve()), "reference_page": 1, "reference_bytes": 0,
                  "reference_mtime": 0, "max_error": 0.25, "error_function": None, "mode": "centered"}
    return {"measure": measure, "hole_check": dict(hole_check, **changes)}


def without_file_stats(settings):
    check = settings["hole_check"]
    stats = ("reference_bytes", "reference_mtime", "reference_digest", "error_function_digest")
    return {"measure": settings["measure"],
            "hole_check": check and {key: value for key, value in check.items() if key not in stats}}


def test_the_batch_command_gives_the_runs_settings_back(tmp_path):
    import tifffile

    reference = tmp_path / "reference.tif"
    tifffile.imwrite(reference, np.zeros((2, 16, 16), np.uint8), photometric="minisblack")
    function = tmp_path / "my_error.py"
    function.write_text("def error(reference, image):\n    return 0.0\n", encoding="utf-8")
    tiff, output = tmp_path / "scan.tif", tmp_path / "out"
    for settings in (run_settings(reference), run_settings(reference, error_function=f"{function.resolve()}:error"),
                     dict(run_settings(reference), hole_check=None)):
        command = part2.batch_command(tiff, output, (100, 260), 3, settings)
        assert command[0] == sys.executable and Path(command[1]) == part2.ROOT / "scripts" / "cd_batch.py"
        args = cd_batch.argument_parser().parse_args(command[2:])
        assert (args.inputs, args.output, args.pages, args.workers_per_gpu) == ([str(tiff)], output, "100:260", 3)
        built = cd_batch.build_settings(args)
        again = json.loads(json.dumps({"measure": built["measure"], "hole_check": built["hole_check"]}))
        assert without_file_stats(again) == without_file_stats(settings)
    assert part2.settings_differences(again, settings) == 0


def test_settings_that_differ_are_counted_per_key():
    a = {"measure": {"unit": "nm", "pixel_size": 21.0}, "hole_check": {"max_error": 0.3, "reference_page": 0}}
    assert part2.settings_differences(a, json.loads(json.dumps(a))) == 0
    b = {"measure": {"unit": "nm", "pixel_size": 20.0, "new": 1},
         "hole_check": {"max_error": 0.25, "reference_page": 0}}
    assert part2.settings_differences(a, b) == 3
    assert part2.settings_differences(a, dict(a, hole_check=None)) == 1


KOREAN_ASPM = """전원 구성표 GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (균형 조정)
  GUID 별칭: SCHEME_BALANCED
  하위 그룹 GUID: 501a4d13-42af-4429-9fd1-a8218c268e20  (PCI Express)
    GUID 별칭: SUB_PCIEXPRESS
    전원 설정 GUID: ee12f906-d277-404b-b6da-e5fa1a576df5  (링크 상태 전원 관리)
      GUID 별칭: ASPM
      가능한 설정 색인: 000
      가능한 설정 이름: 해제
      가능한 설정 색인: 001
      가능한 설정 이름: 보통 수준의 절전
      가능한 설정 색인: 002
      가능한 설정 이름: 최대 절전
    현재 AC 전원 설정 색인: 0x00000001
    현재 DC 전원 설정 색인: 0x00000002
"""
ENGLISH_ASPM = """Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)
  Subgroup GUID: 501a4d13-42af-4429-9fd1-a8218c268e20  (PCI Express)
    Power Setting GUID: ee12f906-d277-404b-b6da-e5fa1a576df5  (Link State Power Management)
      Possible Setting Index: 000
      Possible Setting Friendly Name: Off
      Possible Setting Index: 001
      Possible Setting Friendly Name: Moderate power savings
      Possible Setting Index: 002
      Possible Setting Friendly Name: Maximum power savings
    Current AC Power Setting Index: 0x00000000
    Current DC Power Setting Index: 0x00000002
"""


def test_pcie_power_saving_is_read_from_its_number_not_its_words():
    assert part2.aspm_index(KOREAN_ASPM) == 1
    assert part2.aspm_index(ENGLISH_ASPM) == 0
    assert part2.aspm_index(ENGLISH_ASPM.replace("0x00000000", "0x0000000A")) == 10
    assert part2.aspm_index("The power setting does not exist.") is None and part2.aspm_index(None) is None


def test_power_plans_by_guid_whatever_the_language():
    assert part2.plan_guid("전원 구성표 GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (균형 조정)") \
        == "381b4222-f694-41f0-9685-ff5bb260df2e"
    letters = [part2.plan_letter(part2.plan_guid(text)) for text in (
        "전원 구성표 GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (균형 조정)",
        "Power Scheme GUID: 8C5E7FDA-E8BF-4A96-9A85-A6E23A8C635C  (High performance)",
        "Power Scheme GUID: e9a42b02-d5df-448d-aa00-03f14749eb61  (Ultimate Performance)",
        "Power Scheme GUID: 11111111-2222-3333-4444-555555555555  (My plan)", "")]
    assert letters == ["a", "h", "u", "-", "-"]
    assert part2.plan_guid(ENGLISH_ASPM) == part2.HIGH_PERFORMANCE  # the plan comes first in powercfg's output


NVIDIA_SMI = """+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3464    C+G   ...ms\\Microsoft VS Code\\Code.exe      N/A      |
|    0   N/A  N/A           11120    C+G   C:\\Windows\\explorer.exe               N/A      |
|    2   N/A  N/A            5120      C   ...\\Python311\\python.exe            3512MiB |
|    3      7788    C+G   ...\\dwm.exe   N/A |
+-----------------------------------------------------------------------------------------+
"""


def test_processes_per_gpu_from_nvidia_smis_table():
    found = part2.gpu_processes(NVIDIA_SMI, ["0", "1", "2", "3"])
    assert {gpu: len(names) for gpu, names in found.items()} == {"0": 2, "1": 0, "2": 1, "3": 1}
    assert found["0"] == ["...ms\\Microsoft VS Code\\Code.exe", "C:\\Windows\\explorer.exe"]
    assert found["2"] == ["...\\Python311\\python.exe"] and found["3"] == ["...\\dwm.exe"]
    assert part2.gpu_processes("|  No running processes found                     |", ["0", "1"]) == {"0": [], "1": []}
    assert part2.gpu_processes(NVIDIA_SMI, ["1"]) == {"1": []}  # GPUs this script does not use are left out


def test_memory_and_cpu_while_one_worker_count_ran():
    run = {"start": 0.0, "end": 100.0, "arrivals": [10.0 * k for k in range(1, 9)]}  # 8 rows; middle half: 30-70 s
    gpu = [(50.0, "0", 90.0, 2000.0, 2500.0, 3072.0), (60.0, "1", 90.0, 2000.0, 2500.0, 4096.0),
           (150.0, "0", 0.0, 200.0, 2500.0, 9999.0)]  # the last one after the run: left out
    windows = [(t, {"c": c, "l": 100.0, "a": a, "p": 0.0, "u": u, "g": [{"n": "x", "d": 0.0, "s": s}]})
               for t, c, a, u, s in ((20.0, 40.0, 900.0, 10.0, 0.0), (40.0, 60.0, 800.0, 30.0, 2**30),
                                     (60.0, 50.0, 700.0, 50.0, 2**29), (120.0, 99.0, 1.0, 99.0, 2**33))]
    used = part2.usage(run, 8, gpu, windows)
    assert used["gpu_gb"] == pytest.approx(4.0) and used["shared_gb"] == pytest.approx(1.0)
    assert used["commit_pct"] == pytest.approx(60.0) and used["cpu_pct"] == pytest.approx(40.0)
    assert used["available_mb_least"] == 700.0
    few = part2.usage(dict(run, arrivals=[50.0]), 1, [], [])
    assert few["cpu_pct"] is None and few["gpu_gb"] is None and few["shared_gb"] is None
    stalled = part2.usage(dict(run, arrivals=[10.0, 20.0, 30.0, 40.0, 50.0]), 8, gpu, windows)
    assert stalled["cpu_pct"] == pytest.approx(40.0)  # row 2 (30 s) to the stop (100 s): the samples at 40 and 60 s
    result = dict(used, hole_pages_per_s=1.234, median_page_s=3.46)
    assert part2.setting_token(result) == "1.23/3.5/4/1/60/40/0.7" and part2.setting_token(None) == "-"
    assert part2.setting_token(dict(result, available_mb_least=None)) == "1.23/3.5/4/1/60/40/-"


def test_the_lines_mark_what_is_missing_and_no_letter_looks_like_a_digit():
    lines = part2.token_lines({"v": "8ea8/e27d5/fc874", "a": "378/380/375/377", "k": "1.1/3.5/3.7/0/40/30+-"},
                              part2.LINES)
    assert lines == ["v8ea8/e27d5/fc874 a378/380/375/377 p- u- e- h- w-", "t- n-", "k1.1/3.5/3.7/0/40/30+-",
                     "y- d- m-", "c- x-", "r- f-"]
    labels = [key for line in part2.LINES for key in line]
    assert len(labels) == len(set(labels)) and all(key.isalpha() and key.islower() for key in labels)
    sessions = {part2.session_letter(name) for name in ("RDP-Tcp#0", "rdp-tcp#12", "Console", None, "Services")}
    assert sessions == {"r", "c", "-"}
    # Typed by hand from a screen: no label and no letter inside a value may look like a digit.
    letters = "".join(labels) + "".join(part2.PLANS.values()) + "".join(sessions)
    assert not set(letters) & set("bgilosqz")
    # Section 3's values hold digits, separators, "-" and d (the hole lists differ) only.
    found, compared = drift_answers()
    compared["frozen"] = comparison(same=False)
    values = "".join(part2.drift_tokens(found, compared).values()) + part2.spread_token({"gaussian": 0.25}, 3)
    assert "d" in values and set(values) <= set("0123456789./+-d")


# ------------------------------------------------------------------------------------------ section 3: deblend drift
def trace_rows(count, first_step, ratio=0.9, cos=(0.99,), bg_max=(0.02,), edge=(1.5, 1.0, 0.9)):
    """The trace rows of one deblend pass (scripts/deblend_variants.deblend's): the raw step starts at ``first_step``
    and shrinks by ``ratio`` per round; the centres move a quarter of it, the widths all of it, the three edge bands
    ``edge`` times it; the background mean falls 0.06 per round from 37.4; cos and bg_max go round their values (round
    1 has no round before it: NaN)."""
    rows = []
    for k in range(count):
        step = first_step * ratio ** k
        rows.append({"round": k + 1, "step_p99": step, "centre_p99": step / 4, "width_p99": step,
                     "edge_p99": [f * step for f in edge], "bg_mean": 37.4 - 0.06 * k,
                     "bg_max": math.nan if k == 0 else bg_max[k % len(bg_max)],
                     "cos": math.nan if k == 0 else cos[k % len(cos)]})
    return rows


def measured(rounds, passes=2, seconds=4.56, measured_tiles=0, rows=None):
    """A variant() answer: ``rounds`` over ``passes`` passes, the last one's trace ``rows``."""
    first = {"used": 1, "rounds": trace_rows(3, 40 * TIGHT)}
    return {"rounds": rounds, "passes": [first] * (passes - 1) + [{"used": 1, "rounds": rows or first["rounds"]}],
            "seconds": seconds, "settled": False, "measured_tiles": measured_tiles, "tiles": 2646,
            "source": "inferred", "holes": 39042, "means": {}}


def comparison(page_mean=0.123, same=True, **changes):
    """A compare() answer."""
    return dict({"same_holes": same, "holes": (39042, 39042), "page_mean": page_mean, "page_mean_cell": 0.1,
                 "page_mean_wide": page_mean, "hole_max": 2.5, "hole_p99": 0.514 if same else None,
                 "hole_median": 0.034 if same else None, "holes_above": 2800 if same else None, "valid_changed": 3},
                **changes)


def drift_answers():
    """Section 3's measurements and comparisons, all of them made."""
    found = {"reference": measured(200, rows=trace_rows(20, 40 * TIGHT)), "ref100": measured(100),
             "today": measured(30, rows=trace_rows(20, 300 * TIGHT, ratio=0.96, cos=(0.995,), bg_max=(0.14,))),
             "lm100": measured(110), "frozen": measured(60), "anderson": measured(32), "wider": measured(108),
             "tiles16": measured(110, measured_tiles=1323), "reference_s": measured(220, 3), "today_s": measured(46, 3),
             "continued_s": measured(120, 3, seconds=9.54), "cured_s": measured(40, 3, seconds=4.06)}
    compared = {"today": comparison(1.27), "ref100": comparison(0.054), "lm100": comparison(0.05),
                "frozen": comparison(3.2), "anderson": comparison(0.012), "wider": comparison(0.049),
                "tiles16": comparison(0.0), "today_s": comparison(1.2), "continued_s": comparison(0.04),
                "cured_s": comparison(0.02)}
    return found, compared


def test_what_moves_from_the_trace_rows():
    # The last 12 of 20 rounds: each step 90 % of the one before, cosines 0.95 / 0.97 / 0.99 four times each, the
    # background's largest change 0.01 and 0.03 six times each. The last round: step 40 * 0.9 ** 19 = 5.4 tight units,
    # centre 1.35, edge bands 8.1 / 5.4 / 4.86.
    rows = trace_rows(20, 40 * TIGHT, cos=(0.95, 0.97, 0.99), bg_max=(0.01, 0.03))
    assert part2.movement_token(rows) == "1/5/8/5/5/0.02/90/97"
    # Today's scale: steps of about 100 tight units, each 96 % of the one before.
    assert part2.movement_token(trace_rows(20, 300 * TIGHT, ratio=0.96)) == "35/138/207/138/124/0.02/96/99"
    # A short pass: round 1's NaNs are left out of the medians (4 rounds with a round before them).
    assert part2.movement_token(trace_rows(5, 40 * TIGHT, cos=(0.95, 0.97, 0.99), bg_max=(0.01, 0.03))) \
        == "7/26/39/26/24/0.02/90/97"
    # One round: nothing to compare it with; an empty edge band (NaN) shows "-".
    assert part2.movement_token(trace_rows(1, 40 * TIGHT, edge=(math.nan, 1.0, 0.9))) == "10/40/-/40/36/-/-/-"
    assert part2.movement_token([]) is None


def test_the_reference_from_its_last_pass():
    reference = measured(200, rows=trace_rows(20, 40 * TIGHT))
    # 200 rounds in 2 passes; last step 5.4 tight units; 0.054 from 100 rounds; no tile measured; the background
    # mean fell 19 * 0.06 = 1.14 grey levels over the last pass.
    assert part2.reference_token(reference, comparison(0.054)) == "200/2/5.4/0.05/0/-1.1"
    tiles = dict(reference, measured_tiles=1323)
    assert part2.reference_token(tiles, None) == "200/2/5.4/-/50/-1.1"  # 100 rounds not measured
    fixed = dict(reference, measured_tiles=None, tiles=None)  # a fixed background level has no tiles
    assert part2.reference_token(fixed, comparison(0.0)) == "200/2/5.4/0/-/-1.1"
    empty = dict(reference, passes=[{"used": 0, "rounds": []}])
    assert part2.reference_token(empty, comparison(0.0)) == "200/1/-/0/0/-"


def test_comparisons_with_missing_pieces_and_other_hole_lists():
    assert part2.change_token(comparison(1.27)) == "0.1/1.27/0.03/7/3"  # 2800 of 39042 holes moved over 1 unit: 7 %
    assert part2.change_token(comparison(1.27, same=False)) == "0.1/1.27/d/d/3"
    assert part2.change_token(comparison(page_mean_cell=None)) == "-/0.12/0.03/7/3"
    assert part2.change_token(comparison(holes=(0, 0), holes_above=0)) == "0.1/0.12/0.03/-/3"
    assert part2.candidate_token(measured(110), comparison(0.054)) == "110/0.05/0.51"
    assert part2.candidate_token(measured(110), comparison(same=False)) == "110/0.12/d"
    assert part2.candidate_token(measured(110), None) == "110/-/-"  # the reference is missing
    assert part2.candidate_token(None, comparison()) == "-"
    assert part2.candidate_token(measured(110, measured_tiles=1323), comparison(0.0), tiles=True) == "110/0/0.51/50"
    found, compared = drift_answers()
    assert part2.slowest_token(found, compared) == "220/3/1.2/120/3/9.5/0.04/40/3/4.1/0.02"
    only = {"continued_s": found["continued_s"]}
    assert part2.slowest_token(only, {}) == "-/-/-/120/3/9.5/-/-/-/-/-"


def test_drift_tokens_from_all_or_some_measurements():
    found, compared = drift_answers()
    tokens = part2.drift_tokens(found, compared)
    assert tokens == {"d": "200/2/5.4/0.05/0/-1.1", "m": "35/138/207/138/124/0.14/96/100",
                      "c": "0.1/1.27/0.03/7/3",
                      "x": "110/0.05/0.51+60/3.2/0.51+32/0.01/0.51+108/0.05/0.51+110/0/0.51/50",
                      "r": "220/3/1.2/120/3/9.5/0.04/40/3/4.1/0.02"}
    # m is today's loop (what the batch computes), not the reference's.
    assert tokens["m"] == part2.movement_token(part2.last_pass(found["today"]))
    # The median page's reference failed: no d, no comparison (c), candidates without page means; one candidate failed.
    del found["reference"], found["frozen"]
    some = {key: value for key, value in compared.items() if key.endswith("_s")}
    tokens = part2.drift_tokens(found, some)
    assert set(tokens) == {"m", "x", "r"}
    assert tokens["x"] == "110/-/-+-+32/-/-+108/-/-+110/-/-/50"
    assert part2.drift_tokens({"today_s": measured(46, 3)}, {}) == {}  # nothing to show without its reference
    lines = part2.token_lines(dict(tokens, y="1/0"), part2.LINES)
    assert lines[3:] == [f"y1/0 d- m{tokens['m']}", f"c- x{tokens['x']}", f"r{tokens['r']} f-"]


def test_page_to_page_spread_from_pages_csv(tmp_path):
    path = tmp_path / "pages.csv"
    path.write_text(
        "file,page,status,holes,gaussian_valid,gaussian_cd,gaussian_cd_x,projection_cd,tail_area_cd,foot_area_cd,"
        "edge_area_cd\n"
        "lot/scan.tif,0,ok,900,900,90.0,95.0,80.0,140.0,170.0,\n"
        "lot/scan.tif,1,ok,900,900,90.2,99.0,80.6,140.1,170.4,120.0\n"
        "lot/scan.tif,2,no_holes,,,,,,,,\n"
        "lot/scan.tif,3,ok,900,900,90.4,91.0,80.3,140.2,170.8,\n"
        "lot/other.tif,0,ok,900,900,10.0,10.0,10.0,10.0,10.0,10.0\n", encoding="utf-8")
    rows = part2.csv_rows(path, Path("D:/data/scan.tif"))  # another file's rows are left out
    spread = part2.page_spread(rows)
    assert set(spread) == {"gaussian", "projection", "tail_area", "foot_area", "edge_area"}  # _cd_x is no method
    assert spread["gaussian"] == pytest.approx(np.std([90.0, 90.2, 90.4], ddof=1))
    assert spread["edge_area"] is None  # one page with a value
    # Cell methods: gaussian 0.2, projection 0.3; wide methods: tail_area 0.1, foot_area 0.4.
    assert part2.spread_token(spread, sum(row["status"] == "ok" for row in rows)) == "0.3/0.4/3"
    assert part2.spread_token(dict(spread, foot_area=None, tail_area=None), 3) == "0.3/-/3"  # no wide method measured
    assert part2.spread_token(part2.page_spread([]), 0) == "-/-/0"


def fake_result(cds, valid=None, shift=0.0):
    """A measure_cds result as compare() reads it: per method every hole's CD in px (x and y widths the same), its
    valid flags (all valid unless given); 2 nm per px; the seeds on a line, moved by ``shift``."""
    n = len(next(iter(cds.values())))
    measurements = {name: SimpleNamespace(valid=np.asarray((valid or {}).get(name, [True] * n)),
                                          metrics=dict.fromkeys(("cd_equivalent_px", "cd_x_px", "cd_y_px"),
                                                                np.asarray(values, float)))
                    for name, values in cds.items()}
    seeds = np.column_stack((np.arange(n) + shift, np.zeros(n)))
    return SimpleNamespace(measurements=measurements, seeds_xy=seeds, pixel_size=2.0, unit="nm")


def test_compare_gives_cell_and_wide_page_means_and_per_hole_changes():
    a = fake_result({"gaussian": [5, 5, 5, 5], "half_area": [6, 6, 6, 6], "tail_area": [8, 8, 8, 8],
                     "foot_area": [9, 9, 9, 9]})
    b = fake_result({"gaussian": [5.1, 5.1, 5.6, 5.1], "half_area": [6, 6, 7, 6], "tail_area": [8.3] * 4,
                     "foot_area": [9, 9, 9, np.nan]}, valid={"half_area": [True, True, False, True]})
    found = compare(a, b)
    # Page means, nm: gaussian 10 -> 10.45, half_area 12 -> 12 (its changed hole is not valid in b), tail_area 16 ->
    # 16.6, foot_area 18 -> 18 (the NaN hole left out).
    assert found["page_mean"] == pytest.approx(0.6) and found["page_mean_wide"] == pytest.approx(0.6)
    assert found["page_mean_cell"] == pytest.approx(0.45)
    # Per hole, nm: gaussian 0.2, 0.2, 1.2, 0.2 (median 0.2, one hole over 1 nm); tail_area 0.6 each (median 0.6).
    assert found["hole_median"] == pytest.approx(0.6) and found["holes_above"] == 1
    assert found["hole_max"] == pytest.approx(1.2) and found["valid_changed"] == 1 and found["same_holes"]
    moved = compare(a, fake_result({name: [5, 5, 5, 5] for name in a.measurements}, shift=0.5))
    assert not moved["same_holes"] and moved["page_mean"] is not None
    assert moved["hole_median"] is None and moved["holes_above"] is None and moved["hole_p99"] is None
    cell = compare(fake_result({"gaussian": [5, 5]}), fake_result({"gaussian": [5.5, 5.5]}))
    assert cell["page_mean_cell"] == pytest.approx(1.0) and cell["page_mean_wide"] is None
    empty = compare(fake_result({"tail_area": [np.nan] * 2}), fake_result({"tail_area": [np.nan] * 2}))
    assert empty["page_mean"] is None and empty["page_mean_wide"] is None and empty["holes_above"] == 0
