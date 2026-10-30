"""Real Qt event-loop tests; compatible with the offscreen Qt platform."""

import os
import json
import sys
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtTest import QTest
from subpx._workbench_data import LoadedImage, analyze_image
from subpx.workbench import InspectionWindow


@pytest.fixture
def window(monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    widget = InspectionWindow()
    widget.show()
    y, x = np.mgrid[:80, :100]
    image = np.full((80, 100), 10.)
    for cy in range(12, 73, 16):
        for cx in range(12, 93, 16):
            image += 160 * np.exp(-((x - cx - .25)**2 + (y - cy - .15)**2) / 8)
    widget.set_image(image.astype(np.float32))
    yield widget, app
    wait_for_job(widget, app)
    widget.close()
    app.processEvents()
    assert not errors, [str(error[1]) for error in errors]


def wait_for_job(widget, app, timeout=25):
    deadline = time.monotonic() + timeout
    while widget.busy and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert not widget.busy, "Background job did not finish"


def test_background_detection_selection_and_input_reset(window):
    widget, app = window
    widget.detect_button.click()
    assert widget.busy and not widget.cd_button.isEnabled()
    wait_for_job(widget, app)
    assert widget.review is not None, widget.log.toPlainText()
    assert widget.review.detection is not None
    assert widget.table_model.rowCount() == len(widget.review.seeds_xy)
    widget._select_at(*widget.review.seeds_xy[5])
    assert widget.selected == 5 and widget.feature_spin.value() == 5
    widget.set_image(np.zeros((40, 50), np.uint16))
    assert widget.review is None and widget.table_model.rowCount() == 0
    assert not widget.export_action.isEnabled()


def test_demo_loads_without_external_sample_files(window):
    widget, app = window
    widget.load_demo()
    wait_for_job(widget, app)
    assert widget.loaded.image.shape == (192, 320)
    assert widget.loaded.image.dtype == np.uint8
    widget.run_analysis("pitch")
    wait_for_job(widget, app)
    assert len(widget.review.seeds_xy) > 100
    assert widget.review.patterns, widget.log.toPlainText()


def test_all_stages_methods_roi_and_display_contrast(window):
    widget, app = window
    widget.roi_action.setChecked(True)
    widget.roi.setPos((7.5, 7.5))
    widget.roi.setSize((80, 64))
    review = analyze_image(widget.loaded.image, widget.parameters(), operation="pitch", bounds=widget._bounds())
    widget._accept_review(review)
    np.testing.assert_array_equal(review.origin_xy, [8, 8])
    for stage in range(widget.stage.count()):
        widget.stage.setCurrentIndex(stage)
        app.processEvents()
    for method in range(widget.method.count()):
        widget.method.setCurrentIndex(method)
        widget.select_feature(2)
        widget.ray.setValue(15)
        app.processEvents()
    widget.stage.setCurrentIndex(0)
    widget.canvas.histogram.setLevels(15, 150)
    widget.layer_checks["Voronoi edges"].setChecked(True)
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [15, 150])
    global_point = review.seeds_xy[3] + review.origin_xy
    widget._select_at(*global_point)
    assert widget.selected == 3
    widget.area_min.setValue(4)
    assert "Settings changed" in widget.state_label.text()
    widget.failed_only.setChecked(True)
    measured = review.measurements[widget.method.currentData()]
    assert widget.table_model.rowCount() == np.count_nonzero(~measured.valid)


def test_detection_pitch_reuse_and_worker_error_recovery(window):
    widget, app = window
    widget.pitch_source.setCurrentIndex(widget.pitch_source.findData("detection"))
    widget.run_analysis("pitch")
    wait_for_job(widget, app)
    assert widget.review.patterns, widget.log.toPlainText()
    original = widget.review.detection
    widget.average_rows.setValue(2)
    widget.run_analysis("pitch")
    wait_for_job(widget, app)
    assert widget.review.detection is original
    assert widget.review.patterns["detection"].meta["average_rows"] == 2
    widget.area_max.setValue(1)
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    assert "Action failed" in widget.state_label.text()
    assert widget.cd_button.isEnabled()


def test_registration_job_and_pixel_center_mapping(window):
    widget, app = window
    widget.moving = LoadedImage(np.roll(widget.loaded.image, (2, -3), axis=(0, 1)), "Moving")
    widget.run_registration()
    wait_for_job(widget, app)
    assert widget.registration is not None, widget.log.toPlainText()
    np.testing.assert_allclose(widget.registration.result.shift_yx, [-2, 3], atol=.1)
    for view in range(widget.registration_view.count()):
        widget.registration_view.setCurrentIndex(view)
        app.processEvents()
    item = widget.canvas.image_item
    point = item.mapToParent(QtCore.QPointF(.5, .5))
    np.testing.assert_allclose([point.x(), point.y()], [0, 0], atol=1e-12)
    widget.canvas.fit_image()
    bounds = widget.canvas.plot.getViewBox().viewRange()
    assert bounds[0][0] < 0 and bounds[0][1] >= widget.loaded.image.shape[1] - 1


def test_actual_canvas_click_and_empty_result_clear_inspector(window):
    widget, app = window
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    point = widget.review.seeds_xy[7]
    scene = widget.canvas.plot.getViewBox().mapViewToScene(QtCore.QPointF(*point))
    position = widget.canvas.plot.mapFromScene(scene)
    QTest.mouseClick(widget.canvas.plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
    app.processEvents()
    assert widget.selected == 7
    widget.area_max.setValue(3)
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    assert len(widget.review.seeds_xy) == 0
    assert "No features" in widget.feature_label.text()
    assert widget.comparison.rowCount() == 0


def test_local_and_wide_method_visuals_use_their_own_inputs(window):
    widget, app = window
    parameters = widget.parameters()
    parameters["methods"] = ("local_projection", "local_half_area", "tail_area", "edge_area", "foot_area")
    review = analyze_image(widget.loaded.image, parameters)
    widget._accept_review(review)
    for method in range(widget.method.count()):
        widget.method.setCurrentIndex(method)
        widget.projection_axis.setCurrentIndex(1)
        widget.select_feature(3)
        app.processEvents()
    assert widget.input_canvas.image_item.image.size


def test_tiff_page_navigation_and_gui_export(window, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    widget, app = window
    image = widget.loaded.image
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(image)
        writer.write(image + 5)
    widget.open_image(path=path)
    wait_for_job(widget, app)
    widget.page_spin.setValue(2)
    wait_for_job(widget, app)
    assert widget.loaded.page == 1
    np.testing.assert_array_equal(widget.loaded.image, image + 5)
    widget.open_moving(path=path)
    wait_for_job(widget, app)
    widget.moving_page_spin.setValue(2)
    wait_for_job(widget, app)
    assert widget.moving.page == 1
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    widget.export_results()
    wait_for_job(widget, app)
    directories = list(tmp_path.glob("inspection_*"))
    assert len(directories) == 1
    source = json.loads((directories[0] / "source.json").read_text())
    assert source["page"] == 1 and source["moving_page"] == 1
    assert (directories[0] / "centers_global_xy.npy").is_file()
    assert widget.state_label.text() == "Results exported"
