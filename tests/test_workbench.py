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


def spots():
    y, x = np.mgrid[:80, :100]
    image = np.full((80, 100), 10.)
    for cy in range(12, 73, 16):
        for cx in range(12, 93, 16):
            image += 160 * np.exp(-((x - cx - .25)**2 + (y - cy - .15)**2) / 8)
    return image.astype(np.float32)


def pick(combo, data):
    combo.setCurrentIndex(combo.findData(data))


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    widget = InspectionWindow(settings=settings_file(tmp_path))
    widget.ask_before_discard = False
    pick(widget.setup.backend, "cpu")  # the same numbers on every machine
    widget.setup_dock.show()  # small offscreen screens start with Setup collapsed
    widget.show()
    widget.set_image(spots())
    yield widget, app
    wait_for_job(widget, app)
    widget.ask_before_discard = False
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


def run(widget, app, step):
    widget.setup.run_step(step)
    wait_for_job(widget, app)


def accept(widget, review, step="cd", source=None):
    # A result made outside the window, as if a job had returned it.
    widget.session._accept(widget.loaded.page, step, review, source)


def focus_image(widget, app):
    # Offscreen, a window is active (and its shortcuts work) after one turn of the event loop.
    widget.activateWindow()
    app.processEvents()
    widget.canvas.plot.setFocus()
    app.processEvents()
    return widget.canvas.plot


def view_state(widget):
    """Center and zoom of the main image; a panel appearing beside it changes only how much is seen."""
    box = widget.canvas.plot.getViewBox()
    (x0, x1), (y0, y1) = box.viewRange()
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, *box.viewPixelSize()])


def open_step(widget, key):
    widget.setup.steps.setCurrentIndex(widget.setup.step_keys.index(key))


def drain_stack(widget, app, timeout=5):
    # Pages in view are decoded and drawn between events.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        stack = widget.stack
        if stack is None or not (stack._queue or stack._decode_timer.isActive() or stack._update_timer.isActive()):
            break
        time.sleep(.01)


def write_pages(path, *images):
    tifffile = pytest.importorskip("tifffile")
    with tifffile.TiffWriter(path) as writer:
        for image in images:
            writer.write(image)
    return path


def test_background_detection_selection_and_input_reset(window):
    widget, app = window
    widget.setup.run_step("detect")
    assert widget.busy and not widget.setup.run_button.isEnabled() and widget.cancel_button.isVisible()
    wait_for_job(widget, app)
    assert widget.review is not None, widget.log.toPlainText()
    assert widget.review.detection is not None and widget.session.method == "detection"
    assert widget.measure.method.currentText() == "Previewed holes"
    assert widget.holes.model.rowCount() == len(widget.review.seeds_xy)
    widget.measure._clicked(*widget.review.seeds_xy[5])
    assert widget.selected == 5 and widget.inspector.hole.value() == 5
    widget.set_image(np.zeros((40, 50), np.uint16))
    assert widget.review is None and widget.holes.model.rowCount() == 0
    assert not widget.export_action.isEnabled()


def test_demo_loads_without_external_sample_files(window):
    widget, app = window
    widget.load_demo()
    wait_for_job(widget, app)
    assert widget.loaded.image.shape == (192, 320)
    assert widget.loaded.image.dtype == np.uint8
    run(widget, app, "detect")
    pick(widget.setup.pitch_source, "detection")
    run(widget, app, "pitch")
    assert len(widget.review.seeds_xy) > 100
    assert widget.session.pitch_review.patterns, widget.log.toPlainText()


def test_all_stages_methods_roi_and_display_contrast(window):
    widget, app = window
    view = widget.measure
    view.region_action.setChecked(True)
    view.roi.setPos((7.5, 7.5))
    view.roi.setSize((80, 64))
    assert widget.session.bounds == [8, 8, 88, 72]
    review = analyze_image(widget.loaded.image, widget.session.current_parameters(), operation="pitch",
                           bounds=widget.session.bounds)
    accept(widget, review)
    accept(widget, review, "pitch", review)
    np.testing.assert_array_equal(review.origin_xy, [8, 8])
    for stage in range(view.stage.count()):
        view.stage.setCurrentIndex(stage)
        app.processEvents()
    for method in range(view.method.count()):
        view.method.setCurrentIndex(method)
        widget.select_feature(2)
        widget.inspector.ray.setValue(15)
        app.processEvents()
    view.stage.setCurrentIndex(0)
    widget.canvas.histogram.setLevels(15, 150)
    view.layer_checks["Voronoi edges"].setChecked(True)
    view.layer_checks["Pitch pairs"].setChecked(True)
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [15, 150])
    view.method.setCurrentIndex(view.method.findData("gaussian"))
    global_point = review.seeds_xy[3] + review.origin_xy
    view._clicked(*global_point)
    assert widget.selected == 3
    widget.setup.area_min.setValue(4)
    assert widget.stale_banner.isVisible() and "area min 3 -> 4" in widget.stale_label.text()
    pick(widget.holes.filter, "failed")
    measured = review.measurements[widget.session.method]
    assert widget.holes.model.rowCount() == np.count_nonzero(~measured.valid)


def test_detection_pitch_reuse_and_worker_error_recovery(window, monkeypatch):
    widget, app = window
    run(widget, app, "detect")
    pick(widget.setup.pitch_source, "detection")
    run(widget, app, "pitch")
    pitch = widget.session.pitch_review
    assert pitch.patterns, widget.log.toPlainText()
    original = widget.session.page_results.detect
    assert pitch.detection is original.detection  # pitch uses the result on screen, never a hidden re-run
    widget.setup.average_rows.setValue(2)
    assert widget.stale_banner.isVisible() and "average rows" in widget.stale_label.text()
    run(widget, app, "pitch")
    assert widget.session.pitch_review.detection is original.detection
    assert widget.session.pitch_review.patterns["detection"].meta["average_rows"] == 2
    assert not widget.stale_banner.isVisible()
    review = widget.review
    open_step(widget, "holes")
    widget.setup.area_max.setValue(1)  # below area min: checked as typed, Run disabled with the reason under it
    assert not widget.setup.run_button.isEnabled()
    assert "Area min (3) is above area max (1)" in widget.setup.run_reason.text()
    assert widget.setup.area_max.property("invalid")
    widget.setup.run_step("detect")
    assert not widget.busy and widget.review is review
    assert "Area min" in widget.error_label.text()
    widget.setup.area_max.setValue(200)
    assert widget.setup.run_button.isEnabled() and not widget.setup.run_reason.isVisible()

    def fail(*args, **kwargs):
        raise ValueError("Unreadable page")

    monkeypatch.setattr("subpx.workbench.session.analyze_image", fail)
    widget.bottom.setCurrentWidget(widget.holes)
    run(widget, app, "detect")
    # One plain line over the results, which stay; the traceback is in the log, which does not open by itself.
    assert widget.error_banner.isVisible() and widget.error_label.text() == "Previewing holes failed: Unreadable page"
    assert "Traceback" in widget.log.toPlainText()
    assert widget.review is review and widget.bottom.currentWidget() is widget.holes
    assert widget.message_label.text() == "Failed" and widget.setup.run_button.isEnabled()


def test_registration_job_and_pixel_center_mapping(window):
    widget, app = window
    session = widget.session
    session.moving = LoadedImage(np.roll(widget.loaded.image, (2, -3), axis=(0, 1)), "Moving")
    session.moving_changed.emit()
    widget.set_mode("align")
    assert widget.views.currentWidget() is widget.align_view and widget.setup.run_button.text() == "Align"
    widget.setup.run_button.click()
    wait_for_job(widget, app)
    assert widget.registration is not None, widget.log.toPlainText()
    np.testing.assert_allclose(widget.registration.result.shift_yx, [-2, 3], atol=.1)
    assert "below 0.8" in widget.align_panel.check.text() and "3.000 px right" in widget.align_panel.meaning.text()
    for button in widget.align_view.mode_buttons.values():
        button.setChecked(True)
        app.processEvents()
    item = widget.align_view.canvas.image_item
    point = item.mapToParent(QtCore.QPointF(.5, .5))
    np.testing.assert_allclose([point.x(), point.y()], [0, 0], atol=1e-12)
    widget.align_view.canvas.fit_image()
    bounds = widget.align_view.canvas.plot.getViewBox().viewRange()
    assert bounds[0][0] < 0 and bounds[0][1] >= widget.loaded.image.shape[1] - 1
    widget.align_view.mode_buttons["flicker"].setChecked(True)
    shown = widget.align_view.flicker_aligned
    widget.flicker_action.trigger()  # Space
    assert widget.align_view.flicker_aligned is not shown
    widget.set_mode("measure")
    assert not widget.flicker_action.isEnabled() and widget.views.currentWidget() is widget.measure


def test_actual_canvas_click_and_empty_result_clear_inspector(window):
    widget, app = window
    run(widget, app, "detect")
    point = widget.review.seeds_xy[7]
    scene = widget.canvas.plot.getViewBox().mapViewToScene(QtCore.QPointF(*point))
    position = widget.canvas.plot.mapFromScene(scene)
    QTest.mouseClick(widget.canvas.plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
    app.processEvents()
    assert widget.selected == 7
    widget.setup.area_max.setValue(3)
    run(widget, app, "detect")
    assert len(widget.review.seeds_xy) == 0
    assert "No holes" in widget.inspector.info.text()
    assert widget.inspector.comparison.rowCount() == 0
    assert "No holes" in widget.holes.empty.text()


def test_local_and_wide_method_visuals_use_their_own_inputs(window):
    widget, app = window
    parameters = widget.session.current_parameters()
    parameters["methods"] = ("local_projection", "local_half_area", "tail_area", "edge_area", "foot_area")
    review = analyze_image(widget.loaded.image, parameters)
    accept(widget, review)
    for method in range(widget.measure.method.count()):
        widget.measure.method.setCurrentIndex(method)
        widget.inspector.projection_axis.setCurrentIndex(1)
        widget.select_feature(3)
        app.processEvents()
    assert widget.inspector.input_canvas.image_item.image.size


def test_tiff_page_navigation_moving_page_and_export(window, tmp_path, monkeypatch):
    widget, app = window
    image = widget.loaded.image
    path = write_pages(tmp_path / "pages.tif", image, image + 5)
    widget.open_image(path=path)
    wait_for_job(widget, app)
    widget.page_spin.setValue(1)  # pages are counted from 0, like the batch
    wait_for_job(widget, app)
    assert widget.loaded.page == 1
    np.testing.assert_array_equal(widget.loaded.image, image + 5)
    widget.set_mode("align")
    widget.setup.moving_this_file.click()
    assert widget.setup.moving_list.count() == 2
    widget.setup.moving_list.itemClicked.emit(widget.setup.moving_list.item(1))  # read, then aligned at once
    wait_for_job(widget, app)
    assert widget.moving.page == 1 and widget.registration is not None
    widget.set_mode("measure")
    run(widget, app, "detect")
    # The dialog's defaults: a new folder beside the opened file, every part there is.
    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda self: QtWidgets.QDialog.DialogCode.Accepted)
    widget.export_results()
    wait_for_job(widget, app)
    directories = list(tmp_path.glob("inspection_*"))
    assert len(directories) == 1
    source = json.loads((directories[0] / "source.json").read_text())
    assert source["page"] == 1 and source["moving_page"] == 1 and source["method"] == "detection"
    assert (directories[0] / "centers_global_xy.npy").is_file()
    assert (directories[0] / "registration.npz").is_file()
    rows = (directories[0] / "holes_table.csv").read_text().splitlines()
    assert rows[0].startswith("ID,") and len(rows) == len(widget.review.seeds_xy) + 1
    assert widget.message_label.text().startswith("Saved")


def test_layout_docks_steps_and_run_follow_the_open_step(window):
    widget, app = window
    for dock in (widget.setup_dock, widget.inspector_dock, widget.bottom_dock):
        assert dock.isVisible()
    assert widget.canvas.histogram.isVisible() and not widget.page_spin.isVisible()  # one page: no page bar
    assert [widget.setup.steps.itemText(i).split("  ")[1] for i in range(4)] == ["Image", "Holes", "CD methods", "Pitch"]
    assert widget.bottom.currentWidget() is widget.holes
    focus_image(widget, app)
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_2, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert not widget.inspector_dock.isVisible()
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_2, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert widget.inspector_dock.isVisible()
    open_step(widget, "holes")
    assert widget.setup.threshold.isVisible() and widget.setup.area_min.isVisible()
    assert not widget.setup.recipe.isVisible() and widget.setup.run_button.text() == "Preview holes"
    widget.run_open_action.trigger()  # Ctrl+Enter
    wait_for_job(widget, app)
    assert widget.review.detection is not None
    assert widget.holes.model.columnCount() == 5  # ID, x, y, status, area
    assert "components" in widget.setup.area_note.text()
    widget.measure._clicked(*widget.review.seeds_xy[2])
    assert widget.inspector.isVisible() and widget.bottom.currentWidget() is widget.holes
    assert widget.holes.table.selectionModel().selectedRows()[0].row() == 2
    open_step(widget, "cd")
    assert widget.setup.recipe.isVisible() and widget.setup.run_button.text() == "Measure CD"
    assert "· changed" not in widget.setup.steps.itemText(1)


def test_measurement_recipes_comparison_and_calibration_guidance(window):
    widget, app = window
    setup = widget.setup
    open_step(widget, "cd")
    pick(setup.recipe, "spot")
    setup.run_button.click()
    wait_for_job(widget, app)
    assert tuple(widget.review.measurements) == ("gaussian", "logquad", "gradient", "halfmax")
    model = widget.holes.model
    # Fixed order: ID, x, y, status, CD x, CD y of the shown method, then one CD per method in Setup order.
    assert model.columnCount() == 10
    assert model.cd_methods == ["gaussian", "logquad", "gradient", "halfmax"]
    index = model.index(0, 7)
    value = widget.review.measurements["logquad"].metrics["cd_equivalent_px"][model.hole_at(0)]
    assert model.value(0, 7) == pytest.approx(value)
    assert "Gaussian FWHM" in model.data(index, QtCore.Qt.ItemDataRole.ToolTipRole)
    widget.holes._clicked(model.index(3, 8))
    assert widget.session.method == "gradient" and widget.selected == 3
    assert widget.bottom.currentWidget() is widget.holes  # the table stays after a row click
    comparison = widget.inspector.comparison
    assert comparison.rowCount() == 4 and comparison.item(2, 0).font().bold()
    comparison.cellClicked.emit(1, 0)  # a method in the hole inspector shows that method everywhere
    assert widget.session.method == "logquad" and widget.measure.method.currentData() == "logquad"

    pick(setup.recipe, "local")
    assert setup.parameters()["methods"] == ("local_projection", "local_half_area")
    radial_group = next(group for group, methods in setup.method_groups if methods == {"gradient", "halfmax"})
    assert radial_group.isHidden()
    setup.run_button.click()
    wait_for_job(widget, app)
    assert tuple(widget.review.measurements) == ("local_projection", "local_half_area")

    assert setup.recipe.findData("hole") < 0  # calibrated methods are in the full list only
    for name in ("disk", "flux_ratio"):
        setup.method_checks[name].setChecked(True)
    assert setup.recipe.currentData() == "custom"
    assert setup.calibration.isVisible()
    assert not setup.run_button.isEnabled()
    assert "nominal CD or PSF" in setup.run_button.toolTip()
    assert "nominal CD or PSF" in setup.run_reason.text() and setup.run_reason.isVisible()
    setup.nominal_cd.setValue(5)
    assert setup.run_button.isEnabled()
    assert setup.background.currentData() == "model"


def test_steps_keep_each_others_results_and_ignore_unrelated_settings(window):
    widget, app = window
    setup, session = widget.setup, widget.session
    run(widget, app, "cd")
    original = session.page_results.cd
    open_step(widget, "pitch")
    assert widget.review is original
    assert setup.average_rows.isVisible() and setup.nominal_x.isVisible()
    setup.run_button.click()
    wait_for_job(widget, app)
    assert session.page_results.cd is original and session.pitch_review.patterns
    assert session.pitch_review.cd is original.cd  # the CD result on screen, not measured again
    run(widget, app, "detect")
    assert session.method == "detection" and widget.review is session.page_results.detect
    assert session.page_results.cd is original and session.pitch_review is not None  # running never hides another result
    widget.measure.method.setCurrentIndex(widget.measure.method.findData("gaussian"))
    assert widget.review is original
    setup.nominal_x.setValue(5)  # An incomplete pitch setting must not prevent CD.
    assert "Set both design pitches" in setup.run_reason.text()
    open_step(widget, "cd")
    assert setup.run_button.isEnabled()
    setup.run_button.click()
    wait_for_job(widget, app)
    assert widget.review.cd is not None and widget.review is not original
    assert widget.message_label.text() != "Failed"
    widget.set_mode("align")
    assert setup.moving_other.isVisible() and setup.run_button.text() == "Align"
    assert not setup.run_button.isEnabled() and "moving page" in setup.run_reason.text()
    assert not setup.recipe.isVisible()


def test_stack_navigation_analysis_roi_and_export(window, tmp_path):
    widget, app = window
    image = widget.loaded.image
    view = widget.measure
    widget.open_image(path=write_pages(tmp_path / "stack.tif", image, image + 5, image[:60, :90] + 10), page=1)
    wait_for_job(widget, app)
    assert widget.page_spin.isVisible() and widget.page_spin.value() == 1
    run(widget, app, "detect")
    review = widget.review
    widget.canvas.histogram.setLevels(15, 150)
    view.region_action.setChecked(True)
    view.roi.setPos((7.5, 7.5))
    view.roi.setSize((80, 64))
    widget.pages.stack_pages.setChecked(True)
    assert not widget.busy and widget.stack_action.isChecked()  # the file is open: stacking reads nothing first
    stack = widget.stack
    assert len(stack.shapes) == 3
    assert widget.review is review
    offset = stack.offsets[1]
    assert offset > image.shape[0]
    assert view.bounds() == [8, 8, 88, 72]
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [15, 150])
    drain_stack(widget, app)
    assert sorted(stack.items) == [0, 2]  # the other pages; the active page is the canvas's image
    assert stack.bounds().bottom() == stack.offsets[-1] + 60 - .5
    assert stack.bounds().top() < -.5  # Reserve room for the first page label.
    point = review.seeds_xy[3]
    view._clicked(point[0], point[1] + offset)
    assert widget.selected == 3
    np.testing.assert_allclose(view.layers.selection.getData()[1], [point[1] + offset])
    view._hovered(2, offset + 3)
    assert "Page 1" in widget.hover_label.text()
    assert "pixel [3, 2]" in widget.hover_label.text()
    widget.canvas.histogram.setLevels(20, 130)
    for item in stack.items.values():
        np.testing.assert_allclose(item.getLevels(), [20, 130])

    # A stack is a display layout; crop coordinates and exports stay page-local.
    run(widget, app, "detect")
    np.testing.assert_array_equal(widget.review.origin_xy, [8, 8])
    widget.export_results(folder=tmp_path / "out")
    wait_for_job(widget, app)
    source = json.loads((tmp_path / "out" / "source.json").read_text())
    assert source["page"] == 1 and source["pages"] == 3 and source["display_mode"] == "stack"
    np.testing.assert_allclose(np.load(tmp_path / "out" / "centers_global_xy.npy"), widget.review.seeds_xy + [8, 8])

    run(widget, app, "cd")
    review = widget.review
    contours = review.measurements[widget.session.method].contours_xy
    xs, ys = view.layers.contours.getData()
    first = np.column_stack((xs, ys))[:contours.shape[1]]
    np.testing.assert_allclose(first, contours[0] + review.origin_xy + [0, offset], equal_nan=True)
    view.layer_checks["Voronoi edges"].setChecked(True)
    for stage in range(view.stage.count()):
        view.stage.setCurrentIndex(stage)
        app.processEvents()
        origin = review.origin_xy if stage else np.zeros(2)
        point = widget.canvas.image_item.mapToParent(QtCore.QPointF(.5, .5))
        np.testing.assert_allclose([point.x(), point.y()], origin + [0, offset])
    view.stage.setCurrentIndex(0)
    view.zoom_selected()
    center_y = np.mean(widget.canvas.plot.getViewBox().viewRange()[1])
    assert center_y == pytest.approx(review.seeds_xy[widget.selected, 1] + review.origin_xy[1] + offset)

    widget.pages.stack_pages.setChecked(False)
    assert view.bounds() == [8, 8, 88, 72]
    assert not stack.items
    widget.stack_action.trigger()  # View > Stack vertically: the same switch
    assert widget.pages.stack_pages.isChecked() and not widget.busy  # The file stays open; toggling does not reload it.
    drain_stack(widget, app)
    seen = view_state(widget)
    view._clicked(2, widget.stack.offsets[2] + 3)
    wait_for_job(widget, app)
    assert widget.loaded.page == 2 and widget.review is None
    assert widget.pages.list.currentRow() == 2 and widget.page_spin.value() == 2  # the page list follows
    assert widget.pages.stack_pages.isChecked() and not widget.next_page.isEnabled()
    assert widget.loaded.image.shape == (60, 90)
    np.testing.assert_allclose(view_state(widget), seen)
    widget.previous_page.click()
    wait_for_job(widget, app)
    assert widget.loaded.page == 1 and widget.next_page.isEnabled()
    assert widget.review is review
    assert view.bounds() == [8, 8, 88, 72]
    widget.set_image(image)
    app.processEvents()  # the header hides the page bar on its next layout
    assert widget.stack is None and not widget.page_spin.isVisible()
    assert not widget.pages.stack_pages.isChecked()


def test_unreadable_pages_and_files_leave_the_open_page_and_result(window, tmp_path, monkeypatch):
    import subpx.workbench.session as session_module

    widget, app = window
    path = write_pages(tmp_path / "pages.tif", *[widget.loaded.image] * 3)
    widget.open_image(path=path)
    wait_for_job(widget, app)
    run(widget, app, "detect")
    review = widget.review
    read = session_module.PageSource.read

    def fail(self, page):
        if page == 2:
            raise ValueError("Unreadable TIFF page")
        return read(self, page)

    monkeypatch.setattr(session_module.PageSource, "read", fail)
    widget.pages.stack_pages.setChecked(True)
    drain_stack(widget, app)
    assert widget.review is review and widget.loaded.page == 0
    assert widget.stack is not None and 2 in widget.stack.broken and widget.next_page.isEnabled()
    assert "Unreadable TIFF page" in widget.log.toPlainText()

    def broken(path):
        raise ValueError("not a TIFF file")

    monkeypatch.setattr(session_module, "PageReader", broken)
    widget.open_image(path=tmp_path / "other.tif")
    wait_for_job(widget, app)
    assert widget.error_label.text() == "Opening other.tif failed: not a TIFF file"
    assert widget.review is review and widget.loaded.page == 0 and widget.stack is not None


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
    parameters = widget.setup.parameters()
    assert widget.setup.recipe.currentData() == "batch" and parameters["methods"] == tuple(batch["methods"])
    for key in ("seeds", "invert", "area_min", "area_max", "pixel_size", "unit", "half_area_level",
                "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations", "deblend_tolerance"):
        assert parameters[key] == batch[key], key
    library = inspect.signature(measure_cds).parameters  # the batch leaves these at measure_cds's defaults
    for key in ("threshold", "pad", "background"):
        assert parameters[key] == library[key].default, key
    assert parameters["backend"] == "auto"
    assert widget.setup.recipe.findData("hole") < 0  # no calibrated preset: disk and flux_ratio need a nominal CD


def test_settings_and_layout_survive_a_new_window_and_reset_restores_defaults(window, tmp_path):
    widget, app = window
    setup = widget.setup
    setup.area_max.setValue(150)
    pick(setup.recipe, "spot")
    setup.unit.setCurrentText("nm")
    setup.pixel_size.setValue(21)
    widget.measure.layer_checks["Voronoi edges"].setChecked(True)
    widget.bottom.setCurrentWidget(widget.log)
    widget.inspector_dock.hide()
    widget.close()
    again = InspectionWindow(settings=settings_file(tmp_path))
    try:
        again.show()
        app.processEvents()
        assert again.setup.area_max.value() == 150 and again.setup.recipe.currentData() == "spot"
        assert again.setup.parameters()["methods"] == ("gaussian", "logquad", "gradient", "halfmax")
        assert again.setup.unit.currentText() == "nm" and again.setup.pixel_size.value() == 21
        assert again.measure.layer_checks["Voronoi edges"].isChecked() and again.setup.backend.currentData() == "cpu"
        assert again.session.parameters["area_max"] == 150  # the session has the restored settings, too
        assert not again.inspector_dock.isVisible() and again.bottom.currentWidget() is again.log
        again.reset_settings()
        assert again.setup.area_max.value() == 200 and again.setup.recipe.currentData() == "batch"
        assert again.setup.backend.currentData() == "auto" and again.setup.unit.currentText() == "px"
        assert not again.measure.layer_checks["Voronoi edges"].isChecked()
        assert again.session.parameters["area_max"] == 200
    finally:
        again.close()


def test_disabled_controls_look_disabled(window):
    widget, app = window
    button = widget.setup.run_button

    def fill():
        app.processEvents()
        image = button.grab().toImage()
        return image.pixelColor(image.width() // 2, 4).name()

    enabled = fill()
    widget.setup.area_max.setValue(1)
    assert not button.isEnabled() and fill() != enabled
    assert widget.setup.run_reason.isVisible()


def test_no_hole_is_selected_until_a_click_and_failures_are_counted_per_method(window):
    widget, app = window
    run(widget, app, "detect")
    assert widget.selected is None and not widget.measure.layers.selection.isVisible()
    widget.measure._clicked(*widget.review.seeds_xy[4])
    assert widget.selected == 4 and widget.measure.layers.selection.isVisible()
    parameters = widget.session.current_parameters()
    parameters["methods"] = ("gaussian", "projection", "half_area")
    review = analyze_image(widget.loaded.image, parameters)
    assert "No method failed" in widget.failure_summary(review)
    review.measurements["half_area"].valid[:3] = False
    accept(widget, review)
    widget.measure.method.setCurrentIndex(widget.measure.method.findData("gaussian"))
    assert widget.selected == 4  # still the hole at the clicked position
    text = widget.summary_label.text()
    assert f"Failed: Half area 3 ({100 * 3 / len(review.seeds_xy):.0f} %)" in text and "Projection 0" not in text
    assert "Gaussian fit:" in text


def test_table_keys_select_holes_keep_the_table_and_pan_to_them(window):
    widget, app = window
    run(widget, app, "cd")
    review = widget.review
    table, model = widget.holes.table, widget.holes.model
    view = widget.canvas.plot.getViewBox()
    view.setRange(xRange=(0, 20), yRange=(0, 16), padding=0)
    app.processEvents()
    width = np.diff(view.viewRange()[0])[0]
    table.setFocus()
    table.setCurrentIndex(model.index(0, 0))
    assert widget.selected == model.hole_at(0)
    QTest.keyClick(table, QtCore.Qt.Key.Key_Down)
    assert widget.selected == model.hole_at(1)
    assert widget.bottom.currentWidget() is widget.holes
    far = int(np.argmax(review.seeds_xy.sum(axis=1)))
    table.setCurrentIndex(model.index(model.row_of_hole(far), 0))
    (x0, x1), (y0, y1) = view.viewRange()
    point = review.seeds_xy[far] + review.origin_xy
    assert widget.selected == far and x0 <= point[0] <= x1 and y0 <= point[1] <= y1
    assert x1 - x0 == pytest.approx(width)  # panned, not zoomed
    assert not widget.inspector.hole.keyboardTracking()
    assert not widget.canvas.plot.getAxis("bottom").autoSIPrefix


def test_hole_keys_follow_the_table_order_and_step_through_failures(window):
    widget, app = window
    parameters = widget.session.current_parameters()
    parameters["methods"] = ("gaussian", "half_area")
    review = analyze_image(widget.loaded.image, parameters)
    review.measurements["half_area"].valid[[2, 5, 9]] = False
    accept(widget, review)
    widget.holes.table.sortByColumn(0, QtCore.Qt.SortOrder.DescendingOrder)
    focus_image(widget, app)
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_J)  # J = next hole in the table's order
    first = widget.selected
    assert first == len(review.seeds_xy) - 1
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_J)
    assert widget.selected == first - 1
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_K)
    assert widget.selected == first
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_N)
    assert widget.session.method == "gaussian" and "No failed holes for Gaussian fit" in widget.message_label.text()
    widget.session.set_method("half_area")
    visited = []
    for _ in range(4):
        QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_N)
        visited.append(widget.selected)
    assert visited == [9, 5, 2, 9]  # descending table order, wrapping
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_N, QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert widget.selected == 2
    assert widget.holes.table.currentIndex().row() == widget.holes.model.row_of_hole(2)
    widget.copy()  # Ctrl+C: the selected hole's row
    text = QtWidgets.QApplication.clipboard().text().splitlines()
    assert text[0].startswith("ID\t") and text[1].startswith("2\t") and len(text) == 2


def test_cd_range_and_pitch_points_pick_holes(window):
    widget, app = window
    run(widget, app, "cd")
    session, holes = widget.session, widget.holes
    cds = widget.review.measurements[session.method].metrics["cd_equivalent_px"]
    widget.bottom.setCurrentWidget(widget.distribution)
    widget.distribution.pick.setChecked(True)
    low = float(np.median(cds))
    widget.distribution.region.setRegion((low, float(cds.max()) + 1))
    expected = np.flatnonzero(cds >= low)
    np.testing.assert_array_equal(np.sort(session.highlight), expected)
    assert holes.model.rowCount() == len(expected) and not holes.clear_subset.isHidden()
    holes.clear_subset.click()
    assert session.highlight is None and holes.model.rowCount() == len(cds)

    open_step(widget, "pitch")
    widget.setup.run_button.click()
    wait_for_job(widget, app)
    widget.bottom.setCurrentWidget(widget.pitch)
    widget.pitch.pick("x", 1, "node")  # a placement point: the holes of logical column 1
    pattern = widget.pitch.pattern()
    column = np.flatnonzero(pattern.indices_rc[:, 1] == 1)
    assert set(session.highlight) == set(column) and "column 1" in session.highlight_label
    assert widget.selected in column and holes.model.rowCount() == len(column)
    widget.pitch.pick("y", 0, "interval")
    assert len(session.highlight) and "rows 0 and 1" in session.highlight_label


def test_stale_banner_reverts_and_reruns(window):
    widget, app = window
    run(widget, app, "detect")
    assert not widget.stale_banner.isVisible()
    widget.setup.area_max.setValue(150)
    assert widget.stale_banner.isVisible() and "area max 200 -> 150" in widget.stale_label.text()
    assert widget.setup.steps.itemText(1).endswith("· changed")
    widget.revert_button.click()
    assert widget.setup.area_max.value() == 200 and not widget.stale_banner.isVisible()
    widget.measure.region_action.setChecked(True)
    assert "region" in widget.stale_label.text()
    widget.revert_button.click()  # the region comes back as it was: off
    assert not widget.measure.region_action.isChecked() and not widget.stale_banner.isVisible()
    widget.setup.area_max.setValue(150)
    widget.rerun_button.click()
    wait_for_job(widget, app)
    assert not widget.stale_banner.isVisible() and widget.review.parameters["area_max"] == 150


def test_ask_before_discarding_results_cancel_and_close_while_busy(window, monkeypatch):
    widget, app = window
    run(widget, app, "detect")
    review = widget.review
    widget.ask_before_discard = True
    asked = []
    answer = [QtWidgets.QMessageBox.StandardButton.No]
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: (asked.append(args), answer[0])[1])
    widget.load_demo()
    assert asked and not widget.busy and widget.review is review
    event = QtGui.QCloseEvent()
    widget.closeEvent(event)
    assert not event.isAccepted() and len(asked) == 2
    widget.setup.run_step("cd")
    assert widget.busy
    event = QtGui.QCloseEvent()
    widget.closeEvent(event)
    assert not event.isAccepted() and "Esc" in widget.message_label.text()
    widget.cancel_action.trigger()  # Esc: the result is discarded
    assert not widget.busy and widget.review is review and "cancelled" in widget.log.toPlainText()
    answer[0] = QtWidgets.QMessageBox.StandardButton.Yes
    widget.load_demo()
    wait_for_job(widget, app)
    assert widget.review is None and widget.loaded.image.shape == (192, 320)


def test_page_keys_act_on_the_image_and_page_list_not_the_table(window, tmp_path):
    widget, app = window
    image = widget.loaded.image
    widget.open_image(path=write_pages(tmp_path / "pages.tif", image, image + 5, image + 10))
    wait_for_job(widget, app)
    assert widget.page_spin.minimum() == 0 and widget.page_count.text().startswith("of 3")
    assert widget.pages.list.count() == 3 and widget.pages.list.currentRow() == 0
    run(widget, app, "detect")
    assert "holes previewed" in widget.pages.list.item(0).text()
    focus_image(widget, app)
    widget.holes.table.setFocus()
    QTest.keyClick(widget.holes.table, QtCore.Qt.Key.Key_PageDown)
    app.processEvents()
    assert widget.loaded.page == 0 and not widget.busy
    widget.canvas.plot.setFocus()
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_PageDown)
    wait_for_job(widget, app)
    assert widget.loaded.page == 1 and widget.pages.list.currentRow() == 1
    widget.left_tabs.setCurrentWidget(widget.pages)
    widget.pages.list.setFocus()
    QTest.keyClick(widget.pages.list, QtCore.Qt.Key.Key_PageDown)
    wait_for_job(widget, app)
    assert widget.loaded.page == 2
    widget.pages.list.setCurrentRow(0)
    wait_for_job(widget, app)
    assert widget.loaded.page == 0 and widget.page_spin.value() == 0 and widget.review is not None


def test_page_change_keeps_view_region_tab_selection_contrast_and_moving_image(window, tmp_path):
    widget, app = window
    image = widget.loaded.image
    widget.open_image(path=write_pages(tmp_path / "pages.tif", image, image + 5))
    wait_for_job(widget, app)
    run(widget, app, "detect")
    review = widget.review
    widget.measure._clicked(*review.seeds_xy[7])
    position = review.seeds_xy[7] + review.origin_xy
    widget.measure.region_action.setChecked(True)
    widget.measure.roi.setPos((7.5, 7.5))
    widget.measure.roi.setSize((80, 64))
    view = widget.canvas.plot.getViewBox()
    view.setRange(xRange=(10, 40), yRange=(10, 30), padding=0)
    app.processEvents()
    zoom = view_state(widget)
    widget.canvas.histogram.setLevels(20, 120)
    widget.measure.lock_levels.setChecked(True)
    widget.session.open_moving(None, 0)
    wait_for_job(widget, app)
    assert widget.moving.page == 0
    widget.bottom.setCurrentWidget(widget.pitch)
    widget.next_page.click()
    wait_for_job(widget, app)
    assert widget.loaded.page == 1
    np.testing.assert_allclose(view_state(widget), zoom)
    assert widget.measure.bounds() == [8, 8, 88, 72] and widget.measure.region_action.isChecked()
    assert widget.session.bounds == [8, 8, 88, 72]
    assert widget.bottom.currentWidget() is widget.pitch
    np.testing.assert_allclose(widget.canvas.image_item.getLevels(), [20, 120])
    widget.set_mode("align")
    assert widget.moving is not None and widget.setup.run_button.isEnabled()
    widget.set_mode("measure")
    run(widget, app, "detect")
    moved = widget.review.seeds_xy[widget.selected] + widget.review.origin_xy
    np.testing.assert_allclose(moved, position, atol=0.5)
    widget.session.run_align()
    wait_for_job(widget, app)
    assert widget.registration.hole_check_error < 0.8
    assert "below 0.8" in widget.align_panel.check.text()


def test_handler_errors_reach_the_banner_and_dropped_files_open(window, tmp_path, monkeypatch):
    import subpx.workbench.window as workbench

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


def test_selecting_moves_only_the_marker_and_far_views_show_raster_maps(window):
    widget, app = window
    view = widget.measure
    view.layer_checks["CD colors"].setChecked(True)
    parameters = widget.session.current_parameters()
    parameters["methods"] = ("gaussian", "projection", "half_area")
    review = analyze_image(widget.loaded.image, parameters)
    review.measurements["half_area"].valid[:3] = False
    cds = review.measurements["gaussian"].metrics["cd_equivalent_px"]
    cds[:] = np.linspace(4, 5, len(cds))  # the sample's holes are all alike; give them a spread and an outlier
    cds[-1] = 50
    accept(widget, review)
    layers = view.layers
    # Each cell carries its hole's CD; the scale runs over 1-99 % of the shown method's CDs (not to the outlier).
    x, y = np.rint(review.seeds_xy[5]).astype(int)
    assert review.cd.cell_labels[y, x] == 5
    assert layers.cd_map.source[y, x] == pytest.approx(cds[5], rel=1e-6)
    np.testing.assert_allclose(view.cd_bar.levels(), np.percentile(cds, (1, 99)))
    assert view.cd_bar.levels()[1] < 50 and view.cd_scale.isVisible()

    redrawn = []
    show = layers.show
    layers.show = lambda *args, **kwargs: (redrawn.append(args), show(*args, **kwargs))
    try:
        view._clicked(*review.seeds_xy[5])
        assert widget.selected == 5 and not redrawn  # only the marker moves
        np.testing.assert_allclose(layers.selection.getData()[0], [review.seeds_xy[5, 0]])
        view.lock_cd.setChecked(True)
        levels = view.cd_bar.levels()
        view.method.setCurrentIndex(view.method.findData("half_area"))
        assert redrawn  # a method switch redraws the maps
    finally:
        del layers.show
    np.testing.assert_allclose(view.cd_bar.levels(), levels)  # the locked range stays for the other method
    failed = layers.failures.source
    assert layers.failures.isVisible()
    assert failed[review.cd.cell_labels == 0].all() and not failed[review.cd.cell_labels == 5].any()
    view.lock_cd.setChecked(False)
    values = review.measurements["half_area"].metrics["cd_equivalent_px"][3:]
    low, high = np.percentile(values, (1, 99))
    np.testing.assert_allclose(view.cd_bar.levels(), (low, high) if high > low else (low - .5, high + .5))

    # Far out the vector layers give way to the maps; closer in only the holes in view are drawn.
    layers.VECTOR_LIMIT = 20
    box = widget.canvas.plot.getViewBox()
    widget.canvas.fit_image()
    app.processEvents()
    assert layers.in_view == len(review.seeds_xy) and not layers.centers.isVisible()
    assert not layers.contours.isVisible() and "zoom in" in view.hint.text()
    box.setRange(xRange=(0, 20), yRange=(0, 16), padding=0)
    app.processEvents()
    assert layers.near and layers.centers.isVisible() and layers.contours.isVisible()
    shown = len(layers.centers.getData()[0]) + len(layers.failed.getData()[0])
    assert 0 < shown == layers.in_view < len(review.seeds_xy)
    assert "zoom in" not in view.hint.text()


def test_table_keeps_its_columns_and_sort_when_the_method_changes(window):
    widget, app = window
    open_step(widget, "cd")
    pick(widget.setup.recipe, "spot")
    widget.setup.run_button.click()
    wait_for_job(widget, app)
    table, model = widget.holes.table, widget.holes.model
    header = table.horizontalHeader()
    horizontal = QtCore.Qt.Orientation.Horizontal
    names = [model.headerData(column, horizontal) for column in range(model.columnCount())]
    assert names[:6] == ["ID", "x (px)", "y (px)", "Status", "CD x (px)", "CD y (px)"]
    assert names[6:] == [name + chr(10) + "CD (px)" for name in ("Gaussian fit", "Log fit", "Gradient edge", "Half-height")]
    widget.measure._clicked(*widget.review.seeds_xy[3])
    table.sortByColumn(7, QtCore.Qt.SortOrder.DescendingOrder)

    def column(index):
        values = np.array([model.value(row, index) for row in range(model.rowCount())], float)
        return values[np.isfinite(values)]

    assert (np.diff(column(7)) <= 0).all()
    assert table.currentIndex().row() == model.row_of_hole(3)  # the selection follows its hole
    widget.session.set_method("logquad")
    assert [model.headerData(c, horizontal) for c in range(model.columnCount())] == names
    assert model.headerData(7, horizontal, QtCore.Qt.ItemDataRole.FontRole).bold()
    assert header.sortIndicatorSection() == 7 and (np.diff(column(7)) <= 0).all()
    table.sortByColumn(4, QtCore.Qt.SortOrder.AscendingOrder)  # CD x of the shown method
    widget.session.set_method("gaussian")
    cd_x = widget.review.measurements["gaussian"].metrics["cd_x_px"]
    np.testing.assert_allclose(column(4), np.sort(cd_x[np.isfinite(cd_x)]))
    assert header.sectionSize(6) >= header.fontMetrics().horizontalAdvance("Gaussian fit")


def test_results_are_kept_for_the_last_pages_only(window, tmp_path, monkeypatch):
    widget, app = window
    monkeypatch.setattr("subpx.workbench.session.RESULT_PAGES", 2)
    image = widget.loaded.image
    widget.open_image(path=write_pages(tmp_path / "pages.tif", image, image + 5, image + 10))
    wait_for_job(widget, app)
    for page in range(3):
        widget.page_spin.setValue(page)
        wait_for_job(widget, app)
        run(widget, app, "detect")
    assert set(widget.session.results) == {1, 2}
    assert "Results of page 0 dropped" in widget.log.toPlainText()
    assert "holes previewed" in widget.pages.list.item(0).text()  # the page list still says what was done there
    widget.page_spin.setValue(1)
    wait_for_job(widget, app)
    assert widget.review is not None  # a recent page comes back with its result
    widget.page_spin.setValue(0)
    wait_for_job(widget, app)
    assert widget.review is None
    monkeypatch.setattr("subpx.workbench.session.RESULT_BYTES", 1)  # the memory limit: only the open page's result
    widget.page_spin.setValue(1)
    wait_for_job(widget, app)
    assert set(widget.session.results) == {1}


def test_stack_draws_pages_in_view_at_their_resolution_with_a_memory_cap(window, tmp_path, monkeypatch):
    from subpx.workbench.canvas import PageStack
    from subpx.workbench.session import PageSource

    widget, app = window
    rng = np.random.default_rng(1)
    pages = [rng.integers(0, 200, (40, 600), dtype=np.uint8) for _ in range(12)]
    monkeypatch.setattr(PageSource, "FULL_BYTES", 4 * pages[0].nbytes)
    widget.open_image(path=write_pages(tmp_path / "wide.tif", *pages), page=5)
    wait_for_job(widget, app)
    monkeypatch.setattr(PageStack, "FULL_SCALE", 1e-3)  # every view counts as zoomed out: previews
    widget.pages.stack_pages.setChecked(True)
    stack, view, source = widget.stack, widget.canvas.plot.getViewBox(), widget.session.source
    drain_stack(widget, app)
    assert sorted(stack.items) == [page for page in range(12) if page != 5]
    assert all(item.source.shape == (13, 200) for item in stack.items.values())  # block means of 3 x 3 pixels
    np.testing.assert_allclose(stack.items[0].source[0, 0], pages[0][:3, :3].mean(), atol=0.5)
    stack.FULL_SCALE = 1e3  # every view counts as zoomed in: full pages, for the pages in view only
    for page in (2, 8, 10):
        top = stack.offsets[page]
        view.setRange(xRange=(0, 600), yRange=(top, top + 40), padding=0)
        drain_stack(widget, app)
        assert stack.items[page].source.shape == (40, 600)
        assert set(stack.items) == set(stack._in_view()[0].tolist()) - {5}  # the pages in view only
    assert len(source.full) <= 4  # the memory cap
    assert stack.memory() <= 12 * 13 * 200 + 4 * pages[0].nbytes
    focus_image(widget, app)
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_F, QtCore.Qt.KeyboardModifier.ShiftModifier)
    (x0, x1), (y0, y1) = view.viewRange()
    assert y0 <= 0 and y1 >= stack.offsets[-1] + 40 - 1  # the whole stack
    QTest.keyClick(widget.canvas.plot, QtCore.Qt.Key.Key_F)
    (x0, x1), (y0, y1) = view.viewRange()
    assert (y0 + y1) / 2 == pytest.approx(stack.offsets[5] + 19.5, abs=1) and x1 - x0 < 700  # the active page
    widget.measure._clicked(10, stack.offsets[9] + 10)
    wait_for_job(widget, app)
    assert widget.loaded.page == 9 and widget.pages.list.currentRow() == 9
    np.testing.assert_array_equal(widget.loaded.image, pages[9])
