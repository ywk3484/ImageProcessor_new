"""scripts/cd_diagnose.py: the pieces that decide what the three typed lines say (no GPU needed)."""
from pathlib import Path

import numpy as np
import pytest

from scripts import cd_diagnose as diagnose


def rows(*pages):
    """timings.csv rows: (page, seconds, worker)."""
    return [{"file": "lot/a.tif", "page": str(p), "status": "ok", "measure_s": str(s), "gpu": "0", "worker": str(w)}
            for p, s, w in pages]


def test_pages_are_the_runs_median_and_slowest_or_the_ones_named():
    measured = diagnose.measured(rows((0, 9.0, 0), (1, 2.0, 1), (2, 3.0, 0), (3, 1.0, 1), (4, 4.0, 0)))
    assert [r["page"] for r in measured] == ["3", "1", "2", "4", "0"]
    assert diagnose.pick_pages(measured, None) == [(2, 3.0), (0, 9.0)]
    named = diagnose.pick_pages(measured, "4,7")  # page 7 is not in the run: no run time for it
    assert named[0] == (4, 4.0) and named[1][0] == 7 and np.isnan(named[1][1])
    assert diagnose.pick_pages(measured, "1") == [(1, 2.0), (1, 2.0)]
    for bad in ("x", "-1", ","):
        with pytest.raises(SystemExit, match="--pages"):
            diagnose.pick_pages(measured, bad)
    with pytest.raises(SystemExit, match="no measured page"):
        diagnose.pick_pages([], None)


def test_rows_of_another_file_or_without_holes_are_not_used(tmp_path):
    path = tmp_path / "timings.csv"
    path.write_text("file,page,status,measure_s\nlot/a.tif,0,ok,1.5\nlot/b.tif,1,ok,2.5\nlot/A.TIF,2,no_holes,\n",
                    encoding="utf-8")
    picked = diagnose.measured(diagnose.csv_rows(path, Path("D:/data/a.tif")))
    assert [(r["file"], r["page"]) for r in picked] == [("lot/a.tif", "0")]


def test_slowest_pages_that_were_a_workers_first_page():
    # Worker 0 started with page 0, worker 1 with page 1; the four slowest are 0, 5, 1 and 6.
    measured = diagnose.measured(rows((0, 9.0, 0), (1, 8.0, 1), (2, 1.0, 0), (3, 1.0, 1), (5, 8.5, 0), (6, 7.0, 1)))
    assert diagnose.first_pages_among_slowest(measured) == 2
    assert diagnose.first_pages_among_slowest([]) is None


def test_fingerprint_ignores_line_endings_but_not_content(tmp_path):
    a = tmp_path / "a.py"
    a.write_bytes(b"x = 1\ny = 2\n")
    first, each = diagnose.fingerprint([a], tmp_path)
    a.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert diagnose.fingerprint([a], tmp_path)[0] == first
    a.write_bytes(b"x = 1\ny = 3\n")
    assert diagnose.fingerprint([a], tmp_path)[0] != first
    assert list(each) == ["a.py"] and len(first) == 4


def test_numbers_and_lines_stay_short_and_mark_what_is_missing():
    assert [diagnose.num(v) for v in (2.0, 0.3, 1.25, None, float("nan"), -0.01)] == ["2", "0.3", "1.2", "-", "-", "0"]
    assert diagnose.num(1.3e-13, 2) == "0"
    lines = diagnose.token_lines({"v": "761b/8526d/fc874", "x": "1.7/7/=", "e": "0/0/0"})
    assert lines[0].startswith("v761b/8526d/fc874 n- p- w- ")
    assert lines[1].startswith("x1.7/7/= f- ") and lines[2] == "r- e0/0/0 k-"
    assert len(lines) == 3 and all(key.isalpha() and key.islower() for line in diagnose.LINES for key in line)
    # Typed by hand from a screen: no label and no letter inside a value may look like a digit.
    letters = "".join("".join(line) for line in diagnose.LINES) + "".join(diagnose.VERSION_LETTERS.values()) \
        + "".join(diagnose.PLANS.values()) + "".join(diagnose.SLOW_SETTINGS.values())
    assert not set(letters) & set("bgilosqz")


def test_debug_settings_and_other_library_versions():
    names, slow = diagnose.debug_settings({"CUDA_PATH": "x", "CUDA_PATH_V12_9": "x", "CUDA_VISIBLE_DEVICES": "0",
                                           "CUDA_LAUNCH_BLOCKING": "1", "CUPY_ACCELERATORS": "cub",
                                           "PYTHONDEVMODE": "1", "PATH": "x"})
    assert names == ["CUDA_LAUNCH_BLOCKING", "CUPY_ACCELERATORS", "PYTHONDEVMODE"] and slow == "!ap"
    assert diagnose.debug_settings({"CUDA_LAUNCH_BLOCKING": "0"}) == (["CUDA_LAUNCH_BLOCKING"], "")
    assert diagnose.debug_settings({"CUDA_CACHE_DISABLE": "1", "CUDA_FORCE_PTX_JIT": "1"})[1] == "j"
    same = {"numpy": "2.4.6", "scipy": "1.17.0", "cupy": "14.2.1", "cv2": "5.0.0", "tifffile": "2026.3.3",
            "python": "3.11.2", "cuda_runtime": "12"}
    assert diagnose.other_versions(same) == "="
    assert diagnose.other_versions(dict(same, numpy="1.26.4", cuda_runtime="13", cv2="missing (ImportError)")) == "nvr"


def test_windows_facts_become_four_letters():
    facts = {"plan": "Power Scheme GUID: 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  (High performance)", "scheduling": 2,
             "maker": "Dell Inc.", "model": "PowerEdge R760", "integrity": False}
    assert diagnose.windows_token(facts) == "h1n0"
    assert diagnose.windows_token(dict(facts, model="Virtual Machine", maker="Microsoft Corporation",
                                       scheduling=None, integrity=True)) == "h-v1"
    balanced = "Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced)"
    assert diagnose.windows_token(dict(facts, plan=balanced, scheduling=1)) == "a0n0"
    assert diagnose.windows_token({}) == "-"


def test_cycle_length_finds_a_loop_going_round_and_not_a_settling_one():
    rng = np.random.default_rng(1)
    phases = [rng.normal(size=(50, 6)) for _ in range(4)]
    going_round = [phases[i % 4] + 1e-9 * i for i in range(20)]
    assert diagnose.cycle_length(going_round) == 4
    settling = [np.full((50, 6), 1.0) + 0.5 ** i for i in range(20)]
    assert diagnose.cycle_length(settling) == 0
    assert diagnose.cycle_length([np.zeros((5, 6))] * 10) == 0 and diagnose.cycle_length([]) == 0


def test_no_nvidia_smi_gives_no_gpu_table(monkeypatch):
    monkeypatch.setattr(diagnose, "nvidia_smi", lambda *arguments: None)
    assert diagnose.gpu_table() is None
