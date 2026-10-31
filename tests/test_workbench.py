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

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtTest import QTest
from subpx._workbench_data import LoadedImage, analyze_image
from subpx.workbench import InspectionWindow


def settings_file(path):
    # Never the user's own settings (the registry on Windows).
    return QtCore.QSettings(str(path / "workbench.ini"), QtCore.QSettings.Format.IniFormat)


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    widget = InspectionWindow(settings=settings_file(tmp_path))
    widget.backend.setCurrentIndex(widget.backend.findData("cpu"))  # the same numbers on every machine
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


def test_detection_pitch_reuse_and_worker_error_recovery(window, monkeypatch):
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
    review = widget.review
    widget.area_max.setValue(1)  # below area min: checked as typed, Run disabled with the reason under it
    assert not widget.detect_button.isEnabled() and not widget.cd_button.isEnabled()
    assert "Area min (3) is above area max (1)" in widget.run_reasons["detect"].text()
    assert widget.area_max.property("invalid")
    widget.run_analysis("detect")
    assert not widget.busy and widget.review is review
    assert "Area min" in widget.error_label.text()
    widget.area_max.setValue(200)
    assert widget.cd_button.isEnabled() and not widget.run_reasons["cd"].isVisible()

    def fail(*args, **kwargs):
        raise ValueError("Unreadable page")

    monkeypatch.setattr("subpx.workbench.analyze_image", fail)
    widget.diagnostics.setCurrentWidget(widget.table_tab)
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    # One plain line over the results, which stay; the traceback is in the log, which does not open by itself.
    assert widget.error_banner.isVisible() and widget.error_label.text() == "Finding holes failed: Unreadable page"
    assert "Traceback" in widget.log.toPlainText()
    assert widget.review is review and widget.diagnostics.currentWidget() is widget.table_tab
    assert widget.state_label.text() == "Failed" and widget.cd_button.isEnabled()


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
    assert "No holes" in widget.feature_label.text()
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
    widget.page_spin.setValue(1)  # pages are counted from 0, like the batch
    wait_for_job(widget, app)
    assert widget.loaded.page == 1
    np.testing.assert_array_equal(widget.loaded.image, image + 5)
    widget.open_moving(path=path)
    wait_for_job(widget, app)
    widget.moving_page_spin.setValue(1)
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


def test_task_workflow_keeps_controls_and_results_available(window):
    widget, app = window
    assert widget.controls.isVisible() and widget.diagnostics.isVisible()
    assert widget.canvas.histogram.isVisible() and widget.page_bar.isHidden()
    assert widget.recipe.isVisible() and widget.cd_button.isVisible()
    assert widget.diagnostics.currentWidget() is widget.table_tab
    widget.task.setCurrentIndex(widget.task.findData("detect"))
    assert widget.threshold.isVisible() and widget.area_min.isVisible()
    assert not widget.recipe.isVisible()
    assert widget.run_button is widget.detect_button
    widget.run_button.click()
    wait_for_job(widget, app)
    assert widget.review.detection is not None
    assert widget.diagnostics.currentWidget() is widget.table_tab
    assert widget.table_model.columnCount() == 4
    widget._select_at(*widget.review.seeds_xy[2])
    assert widget.inspector.isVisible() and widget.diagnostics.isVisible()
    assert widget.diagnostics.currentWidget() is widget.feature_tab
    assert widget.table.selectionModel().selectedRows()[0].row() == 2


def test_measurement_recipes_comparison_and_calibration_guidance(window):
    widget, app = window
    widget.recipe.setCurrentIndex(widget.recipe.findData("spot"))
    widget.run_button.click()
    wait_for_job(widget, app)
    assert tuple(widget.review.measurements) == ("gaussian", "logquad", "gradient", "halfmax")
    assert widget.table_model.columnCount() == 10
    assert widget.table_model.extra_methods == ["logquad", "gradient", "halfmax"]
    index = widget.table_model.index(0, 7)
    value = widget.review.measurements["logquad"].metrics["cd_equivalent_px"][0]
    assert widget.table_model.rows[0][7] == pytest.approx(value)
    assert "Gaussian FWHM" in widget.table_model.data(index, QtCore.Qt.ItemDataRole.ToolTipRole)
    widget._select_table(widget.table_model.index(3, 8))
    assert widget.method.currentData() == "gradient" and widget.selected == 3
    assert widget.diagnostics.currentWidget() is widget.table_tab  # the table stays after a row click

    widget.recipe.setCurrentIndex(widget.recipe.findData("local"))
    assert widget.parameters()["methods"] == ("local_projection", "local_half_area")
    radial_group = next(group for group, methods in widget.method_groups if methods == {"gradient", "halfmax"})
    assert radial_group.isHidden()
    widget.run_button.click()
    wait_for_job(widget, app)
    assert tuple(widget.review.measurements) == ("local_projection", "local_half_area")

    assert widget.recipe.findData("hole") < 0  # calibrated methods are in the full list only
    for i in range(widget.method_list.count()):
        item = widget.method_list.item(i)
        if item.data(QtCore.Qt.ItemDataRole.UserRole) in ("disk", "flux_ratio"):
            item.setCheckState(QtCore.Qt.CheckState.Checked)
    assert widget.recipe.currentData() == "custom"
    assert widget.hole_group.isVisible() and widget.hole_note.isVisible()
    assert not widget.cd_button.isEnabled()
    assert "nominal CD or PSF" in widget.cd_button.toolTip()
    assert "nominal CD or PSF" in widget.run_reasons["cd"].text() and widget.run_reasons["cd"].isVisible()
    widget.nominal_cd.setValue(5)
    assert widget.cd_button.isEnabled()
    assert widget.background.currentData() == "model"


def test_task_switches_preserve_results_and_ignore_unrelated_settings(window):
    widget, app = window
    widget.run_button.click()
    wait_for_job(widget, app)
    original = widget.review.cd
    widget.task.setCurrentIndex(widget.task.findData("pitch"))
    assert widget.review.cd is original
    assert widget.average_rows.isVisible() and widget.nominal_x.isVisible()
    widget.run_button.click()
    wait_for_job(widget, app)
    assert widget.review.cd is original and widget.review.patterns
    assert widget.diagnostics.currentWidget() is widget.pitch_tab
    widget.task.setCurrentIndex(widget.task.findData("detect"))
    widget.run_button.click()
    wait_for_job(widget, app)
    assert widget.review.detection is not None
    widget.task.setCurrentIndex(widget.task.findData("cd"))
    assert widget.review.cd is original
    widget.task.setCurrentIndex(widget.task.findData("pitch"))
    widget.nominal_x.setValue(5)  # An incomplete pitch setting must not prevent CD.
    widget.task.setCurrentIndex(widget.task.findData("cd"))
    widget.run_button.click()
    wait_for_job(widget, app)
    assert widget.review.cd is not None
    assert "Action failed" not in widget.state_label.text()
    widget.task.setCurrentIndex(widget.task.findData("register"))
    assert widget.moving_button.isVisible() and widget.register_button.isVisible()
    assert not widget.register_button.isEnabled()
    assert not widget.recipe.isVisible() and not widget.scale_group.isVisible()


def test_stack_navigation_analysis_roi_and_export(window, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    widget, app = window
    image = widget.loaded.image
    path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(image)
        writer.write(image + 5)
        writer.write(image[:60, :90] + 10)
    widget.open_image(path=path, page=1)
    wait_for_job(widget, app)
    assert widget.page_bar.isVisible()
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    review = widget.review
    widget.canvas.histogram.setLevels(15, 150)
    widget.roi_action.setChecked(True)
    widget.roi.setPos((7.5, 7.5))
    widget.roi.setSize((80, 64))
    widget.stack_pages.setChecked(True)
    assert widget.busy and not widget.page_bar.isEnabled()
    wait_for_job(widget, app)
    assert len(widget._pages) == 3
    assert widget.review is review
    offset = widget._page_offsets[1]
    assert offset > image.shape[0]
    assert widget._bounds() == [8, 8, 88, 72]
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [15, 150])
    assert len(widget.canvas.context_images) == 2
    assert widget.canvas._image_bounds.bottom() == widget._page_offsets[-1] + 60 - .5
    assert widget.canvas._image_bounds.top() < -.5  # Reserve room for the first page label.
    point = review.seeds_xy[3]
    widget._select_at(point[0], point[1] + offset)
    assert widget.selected == 3
    np.testing.assert_allclose(widget.canvas.overlays["selected"].getData()[1], [point[1] + offset])
    widget._hover_image(2, offset + 3)
    assert "Page 1" in widget.hover_label.text()
    assert "pixel [3, 2]" in widget.hover_label.text()
    widget.canvas.histogram.setLevels(20, 130)
    for item in widget.canvas.context_images:
        np.testing.assert_allclose(item.getLevels(), [20, 130])

    # A stack is a display layout; crop coordinates and exports stay page-local.
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    np.testing.assert_array_equal(widget.review.origin_xy, [8, 8])
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    widget.export_results()
    wait_for_job(widget, app)
    output = next(tmp_path.glob("inspection_*"))
    source = json.loads((output / "source.json").read_text())
    assert source["page"] == 1 and source["pages"] == 3 and source["display_mode"] == "stack"
    np.testing.assert_allclose(np.load(output / "centers_global_xy.npy"), widget.review.seeds_xy + [8, 8])

    widget.run_analysis("cd")
    wait_for_job(widget, app)
    review = widget.review
    contours = review.measurements[widget.method.currentData()].contours_xy
    xs, ys = widget.canvas.overlays["contours"].getData()
    first = np.column_stack((xs, ys))[:contours.shape[1]]
    np.testing.assert_allclose(first, contours[0] + review.origin_xy + [0, offset], equal_nan=True)
    widget.layer_checks["Voronoi edges"].setChecked(True)
    for stage in range(widget.stage.count()):
        widget.stage.setCurrentIndex(stage)
        app.processEvents()
        origin = review.origin_xy if stage else np.zeros(2)
        point = widget.canvas.image_item.mapToParent(QtCore.QPointF(.5, .5))
        np.testing.assert_allclose([point.x(), point.y()], origin + [0, offset])
    widget.stage.setCurrentIndex(0)
    widget.zoom_selected()
    center_y = np.mean(widget.canvas.plot.getViewBox().viewRange()[1])
    assert center_y == pytest.approx(review.seeds_xy[widget.selected, 1] + review.origin_xy[1] + offset)

    widget.stack_pages.setChecked(False)
    assert widget._bounds() == [8, 8, 88, 72]
    assert not widget.canvas.context_images
    widget.stack_pages.setChecked(True)
    assert not widget.busy  # Already decoded; toggling does not reload the file.
    view_range = widget.canvas.plot.getViewBox().viewRange()
    widget._select_at(2, widget._page_offsets[2] + 3)
    assert widget.loaded.page == 2 and widget.review is None
    assert widget.stack_pages.isChecked() and not widget.next_page.isEnabled()
    assert widget.loaded.image.shape == (60, 90)
    np.testing.assert_allclose(widget.canvas.plot.getViewBox().viewRange(), view_range)
    widget.previous_page.click()
    assert widget.loaded.page == 1 and widget.next_page.isEnabled()
    assert widget.review is review
    assert widget._bounds() == [8, 8, 88, 72]
    widget.set_image(image)
    assert widget._pages is None and widget.page_bar.isHidden()
    assert not widget.stack_pages.isChecked()


def test_stack_load_failure_restores_navigation_and_current_review(window, tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    widget, app = window
    path = tmp_path / "pages.tif"
    tifffile.imwrite(path, np.stack([widget.loaded.image] * 3), photometric="minisblack")
    widget.open_image(path=path)
    wait_for_job(widget, app)
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    review = widget.review

    def fail(path):
        raise ValueError("Unreadable TIFF page")

    monkeypatch.setattr("subpx.workbench.load_pages", fail)
    widget.stack_pages.setChecked(True)
    wait_for_job(widget, app)
    assert widget.review is review and widget.loaded.page == 0
    assert widget._pages is None and not widget.stack_pages.isChecked()
    assert widget.next_page.isEnabled()
    assert "Unreadable TIFF page" in widget.log.toPlainText()


def write_pages(path, *images):
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffWriter(path) as writer:
        for image in images:
            writer.write(image)
    return path


def test_every_method_has_one_display_name_and_errors_read_plainly():
    from subpx.cd import METHOD_LABELS
    from subpx.workbench import METHOD_NAMES, plain_error

    assert set(METHOD_LABELS) <= set(METHOD_NAMES)
    names = [METHOD_NAMES[key] for key in METHOD_LABELS]
    assert len(set(names)) == len(names)
    detail = 'Traceback (most recent call last):\n  File "x.py", line 1\ntifffile.TiffFileError: not a TIFF file\n'
    assert plain_error(detail) == "not a TIFF file"
    assert plain_error("Area min is above area max.") == "Area min is above area max."


def test_defaults_are_the_batch_defaults(window, tmp_path):
    import inspect
    from scripts.cd_batch import argument_parser, build_settings
    from subpx.cd import measure_cds

    widget, app = window
    widget.reset_settings()
    batch = build_settings(argument_parser().parse_args(["in", "--output", str(tmp_path), "--no-hole-check"]))["measure"]
    parameters = widget.parameters()
    assert widget.recipe.currentData() == "batch" and parameters["methods"] == tuple(batch["methods"])
    for key in ("seeds", "invert", "area_min", "area_max", "pixel_size", "unit", "half_area_level",
                "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations", "deblend_tolerance"):
        assert parameters[key] == batch[key], key
    library = inspect.signature(measure_cds).parameters  # the batch leaves these at measure_cds's defaults
    for key in ("threshold", "pad", "background"):
        assert parameters[key] == library[key].default, key
    assert parameters["backend"] == "auto"
    assert widget.recipe.findData("hole") < 0  # no calibrated preset: disk and flux_ratio need a nominal CD


def test_settings_survive_a_new_window_and_reset_restores_defaults(window, tmp_path):
    widget, app = window
    widget.area_max.setValue(150)
    widget.recipe.setCurrentIndex(widget.recipe.findData("spot"))
    widget.unit.setText("nm")
    widget.pixel_size.setValue(21)
    widget.layer_checks["Voronoi edges"].setChecked(True)
    widget.close()
    again = InspectionWindow(settings=settings_file(tmp_path))
    try:
        assert again.area_max.value() == 150 and again.recipe.currentData() == "spot"
        assert again.parameters()["methods"] == ("gaussian", "logquad", "gradient", "halfmax")
        assert again.unit.text() == "nm" and again.pixel_size.value() == 21
        assert again.layer_checks["Voronoi edges"].isChecked() and again.backend.currentData() == "cpu"
        again.reset_settings()
        assert again.area_max.value() == 200 and again.recipe.currentData() == "batch"
        assert again.backend.currentData() == "auto" and again.unit.text() == "px"
        assert not again.layer_checks["Voronoi edges"].isChecked()
    finally:
        again.close()


def test_disabled_controls_look_disabled(window):
    widget, app = window
    button = widget.cd_button

    def fill():
        app.processEvents()
        image = button.grab().toImage()
        return image.pixelColor(image.width() // 2, 4).name()

    enabled = fill()
    widget.area_max.setValue(1)
    assert not button.isEnabled() and fill() != enabled
    assert widget.run_reasons["cd"].isVisible()


def test_no_hole_is_selected_until_a_click_and_failures_are_counted_per_method(window):
    widget, app = window
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    assert widget.selected is None and "selected" not in widget.canvas.overlays
    widget._select_at(*widget.review.seeds_xy[4])
    assert widget.selected == 4 and "selected" in widget.canvas.overlays
    parameters = widget.parameters()
    parameters["methods"] = ("gaussian", "projection", "half_area")
    review = analyze_image(widget.loaded.image, parameters)
    assert "No method failed" in widget._failure_summary(review)
    review.measurements["half_area"].valid[:3] = False
    widget._accept_review(review)
    widget.method.setCurrentIndex(widget.method.findData("gaussian"))
    assert widget.selected == 4  # still the hole at the clicked position
    text = widget.failure_label.text()
    assert f"Failed: Half area 3 ({100 * 3 / len(review.seeds_xy):.0f} %)" in text and "Projection 0" not in text
    assert "Gaussian fit:" in widget.summary_label.text()


def test_table_keys_select_holes_keep_the_table_and_pan_to_them(window):
    widget, app = window
    widget.run_analysis("cd")
    wait_for_job(widget, app)
    review = widget.review
    view = widget.canvas.plot.getViewBox()
    view.setRange(xRange=(0, 20), yRange=(0, 16), padding=0)
    app.processEvents()
    width = np.diff(view.viewRange()[0])[0]
    widget.table.setFocus()
    widget.table.setCurrentIndex(widget.table_model.index(0, 0))
    assert widget.selected == widget.table_model.rows[0][0]
    QTest.keyClick(widget.table, QtCore.Qt.Key.Key_Down)
    assert widget.selected == widget.table_model.rows[1][0]
    assert widget.diagnostics.currentWidget() is widget.table_tab
    far = int(np.argmax(review.seeds_xy.sum(axis=1)))
    widget.table.setCurrentIndex(widget.table_model.index(widget.table_model.row_of[far], 0))
    (x0, x1), (y0, y1) = view.viewRange()
    point = review.seeds_xy[far] + review.origin_xy
    assert widget.selected == far and x0 <= point[0] <= x1 and y0 <= point[1] <= y1
    assert x1 - x0 == pytest.approx(width)  # panned, not zoomed
    assert not widget.feature_spin.keyboardTracking()
    assert not widget.canvas.plot.getAxis("bottom").autoSIPrefix


def test_page_keys_act_on_the_image_not_the_table(window, tmp_path):
    widget, app = window
    image = widget.loaded.image
    widget.open_image(path=write_pages(tmp_path / "pages.tif", image, image + 5, image + 10))
    wait_for_job(widget, app)
    assert widget.page_spin.minimum() == 0 and "page 0 of 3" in widget.source_label.text()
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    widget.activateWindow()
    widget.table.setFocus()
    QTest.keyClick(widget.table, QtCore.Qt.Key.Key_PageDown)
    app.processEvents()
    assert widget.loaded.page == 0 and not widget.busy
    widget.canvas.plot.setFocus()
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_PageDown)
    wait_for_job(widget, app)
    assert widget.loaded.page == 1


def test_page_change_keeps_view_region_tab_selection_contrast_and_moving_image(window, tmp_path):
    widget, app = window
    image = widget.loaded.image
    widget.open_image(path=write_pages(tmp_path / "pages.tif", image, image + 5))
    wait_for_job(widget, app)
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    review = widget.review
    widget._select_at(*review.seeds_xy[7])
    position = review.seeds_xy[7] + review.origin_xy
    widget.roi_action.setChecked(True)
    widget.roi.setPos((7.5, 7.5))
    widget.roi.setSize((80, 64))
    view = widget.canvas.plot.getViewBox()
    view.setRange(xRange=(10, 40), yRange=(10, 30), padding=0)
    app.processEvents()
    zoom = view.viewRange()
    widget.canvas.histogram.setLevels(20, 120)
    widget.lock_levels.setChecked(True)
    widget._use_this_file()
    wait_for_job(widget, app)
    assert widget.moving.page == 0
    widget.diagnostics.setCurrentWidget(widget.pitch_tab)
    widget.next_page.click()
    wait_for_job(widget, app)
    assert widget.loaded.page == 1
    np.testing.assert_allclose(view.viewRange(), zoom)
    assert widget._bounds() == [8, 8, 88, 72] and widget.roi_action.isChecked()
    assert widget.diagnostics.currentWidget() is widget.pitch_tab
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [20, 120])
    assert widget.moving is not None and widget.register_button.isEnabled()
    widget.run_analysis("detect")
    wait_for_job(widget, app)
    moved = widget.review.seeds_xy[widget.selected] + widget.review.origin_xy
    np.testing.assert_allclose(moved, position, atol=0.5)
    widget.run_registration()
    wait_for_job(widget, app)
    assert widget.registration.hole_check_error < 0.8
    assert "below 0.8" in widget.registration_label.text()


def test_handler_errors_reach_the_banner_and_dropped_files_open(window, tmp_path, monkeypatch):
    import subpx.workbench as workbench

    widget, app = window
    assert sys.excepthook is workbench._exception_hook
    monkeypatch.setattr(workbench, "_previous_hook", lambda *error: None)  # not the fixture's error check
    try:
        raise RuntimeError("redraw broke")
    except RuntimeError:
        workbench._exception_hook(*sys.exc_info())
    assert widget.error_label.text() == "Unexpected error: redraw broke"
    assert "redraw broke" in widget.log.toPlainText()
    path = tmp_path / "dropped.npy"
    np.save(path, widget.loaded.image + 1)
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path))])
    event = QtGui.QDropEvent(QtCore.QPointF(10, 10), QtCore.Qt.DropAction.CopyAction, mime,
                             QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier)
    widget.dropEvent(event)
    wait_for_job(widget, app)
    assert widget.loaded.name == "dropped.npy" and not widget.error_banner.isVisible()
