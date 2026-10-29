import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from subpx import analyze_pitch, measure_cds, plot_pitch_report, save_cd_results, save_pitch_report


@pytest.fixture
def source():
    r, c = np.mgrid[:5, :7]
    points = np.column_stack((c.ravel() * 11 + 15 + (r.ravel() % 2) * 5.5, r.ravel() * 12 + 15))
    yy, xx = np.mgrid[:80, :100]
    image = np.full(xx.shape, 8.0)
    for x, y in points:
        image += 130 * np.exp(-((xx - x - 0.23)**2 + (yy - y - 0.17)**2) / (2 * 1.35**2))
    result = measure_cds(image, methods=("gaussian", "local_projection"), area_max=100,
                         pixel_size=2.5, unit="nm", outlines=False, metrics_only=True)
    # Exercise nonconstant pitch and missing fits in both Python and JS.
    centers = result.measurements["local_projection"].centers_xy.copy()
    centers += np.column_stack((0.04 * c.ravel()**2, 0.05 * r.ravel()**2))
    valid = np.ones(len(centers), bool)
    valid[3] = False
    centers[9] = np.nan
    result.measurements["local_projection"] = replace(result.measurements["local_projection"], centers_xy=centers, valid=valid)
    return result


def test_report_exports_centers_profiles_and_rejects_overwrite(source, tmp_path):
    output = tmp_path / "report"
    page = save_pitch_report(source, output, average_rows=3, average_columns=4, title="test image")
    data = json.loads((output / "data.json").read_text(encoding="utf-8"))
    assert data["layout"] == "staggered"
    assert data["shape"] == [5, 7]
    assert data["methods"]["local_projection"]["centers"][9] == [None, None]
    assert data["methods"]["local_projection"]["valid"][9] is False
    assert data["average_rows"] == 3 and data["average_columns"] == 4
    content = page.read_text(encoding="utf-8")
    assert "fetch(" not in content
    assert "Integrated x distance" not in content and "Integrated y distance" not in content
    for filename in ("image.png", "report.png", "centers.csv", "summary.csv", "data.js"):
        assert (output / filename).stat().st_size > 0
    with (output / "profiles.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    computed = analyze_pitch(source, average_rows=3, average_columns=4)
    fig = plot_pitch_report(source.image, computed["gaussian"])
    try:
        assert len(fig.axes) == 5  # Image, two pitch profiles, two registration profiles.
        for axis, ax in (("x", fig.axes[2]), ("y", fig.axes[4])):
            assert ax.get_title() == f"{axis}: Registration error"
            np.testing.assert_allclose(ax.lines[0].get_ydata(), computed["gaussian"].profiles[axis].registration)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    for name, measured in computed.items():
        for axis, p in measured.profiles.items():
            matches = [row for row in rows if row["method"] == name and row["axis"] == axis]
            np.testing.assert_allclose([float(row["pitch_to_next"]) for row in matches[:-1]], p.pitch)
            assert matches[-1]["pitch_to_next"] == ""
    with pytest.raises(ValueError, match="empty"):
        save_pitch_report(source, output)


def test_live_viewer_selection_and_integrals_match_python(source, tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js unavailable for viewer diagnostic")
    folder = tmp_path / "viewer"
    save_pitch_report(source, folder, average_rows=2, average_columns=3)
    scenarios = []
    for reference, rows, columns, nominal in ((0, 1, 1, None), (17, 3, 4, None), (34, 5, 7, (27.5, 30.0))):
        for name, m in analyze_pitch(source, reference_xy=source.seeds_xy[reference], average_rows=rows,
                                     average_columns=columns, nominal_pitch_xy=nominal).items():
            expected = []
            for axis, p in m.profiles.items():
                for i in p.indices:
                    interval = [p.pitch[i], p.std[i], p.counts[i]] if i < len(p.pitch) else [None] * 3
                    expected.append([axis, int(i), p.position[i], *interval, p.distance[i], p.registration[i], p.nominal_pitch])
            from subpx._pattern_report import _json_value
            scenarios.append(_json_value(dict(reference=reference, rows=rows, columns=columns,
                                              nominal=nominal, method=name, csv=expected)))
    expected = tmp_path / "expected.json"
    expected.write_text(json.dumps(scenarios, allow_nan=False), encoding="utf-8")
    diagnostic = Path(__file__).with_name("diag_pitch_viewer.cjs")
    run = subprocess.run([node, str(diagnostic), str(folder), str(expected)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_cli_batch_and_saved_cd_reuse(source, tmp_path):
    from PIL import Image
    from scripts.pitch_report import main

    images = tmp_path / "in"
    images.mkdir()
    Image.fromarray(source.image.astype(np.uint8)).save(images / "staggered.png")
    (images / "broken.png").write_text("bad data")
    output = tmp_path / "out"
    assert main([str(images), "--output", str(output), "--backend", "cpu", "--methods", "gaussian", "local_projection",
                 "--average-rows", "2", "--average-columns", "3"]) == 1
    assert (output / "staggered" / "index.html").exists()
    assert "broken.png" in (output / "index.html").read_text(encoding="utf-8")
    with (output / "images.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["failed", "ok"]
    # Export a full CD result; the new pipeline can reuse every saved method.
    full = measure_cds(source.image, methods=("gaussian", "projection"), area_max=100)
    save_cd_results(full, tmp_path / "cd", compact=True)
    reused = tmp_path / "reuse"
    assert main([str(tmp_path / "cd"), "--from-cd", "--output", str(reused)]) == 0
    assert set(json.loads((reused / "data.json").read_text())["methods"]) == {"gaussian", "projection"}
