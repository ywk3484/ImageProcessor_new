"""Hole inspector (right, always visible): the selected hole in every method, its pixels and the current
method's profile."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..._workbench_data import feature_input, feature_origin, projection_diagnostic, radial_diagnostic
from ...cd import METHOD_LABELS
from ..canvas import ImageCanvas
from ..names import FAILED_COLOR, method_color, method_help, method_name, metric_label, status_text
from ..widgets import Fold, dot_icon, hint, plot_widget

CONTEXT_METHODS = ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")


def pixel_outline(mask, origin):
    """Line segments along the pixel edges around ``mask`` (a thin outline that hides no pixel)."""
    padded = np.pad(mask, 1)
    ys, xs = np.nonzero(padded[1:, 1:-1] != padded[:-1, 1:-1])  # horizontal edges at y - 0.5
    horizontal = np.stack(([xs - .5, ys - .5], [xs + .5, ys - .5]), axis=1)
    ys, xs = np.nonzero(padded[1:-1, 1:] != padded[1:-1, :-1])  # vertical edges at x - 0.5
    vertical = np.stack(([xs - .5, ys - .5], [xs - .5, ys + .5]), axis=1)
    segments = np.concatenate((horizontal, vertical), axis=2).transpose(2, 1, 0).reshape(-1, 2)
    return segments + np.asarray(origin, float)


def _legend(*entries):
    return " &nbsp; ".join(f"<span style='color:{color}'>&#9473; {text}</span>" for color, text in entries)


class HoleInspector(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._radial = None  # (review, method, hole) -> radial diagnostic
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Hole"))
        self.hole = QtWidgets.QSpinBox()
        self.hole.setRange(0, 0)
        self.hole.setKeyboardTracking(False)  # typing "150" selects once, not 1, 15 and 150
        self.hole.valueChanged.connect(lambda value: session.select(value, "inspector"))
        header.addWidget(self.hole)
        self.count = QtWidgets.QLabel("")
        self.count.setObjectName("hint")
        header.addWidget(self.count)
        header.addStretch()
        self.buttons = {}
        for key, text, tip, action in (
                ("previous", "◀", "Previous hole in table order (K or Up in the table)", lambda: session.step_hole(-1)),
                ("next", "▶", "Next hole in table order (J or Down in the table)", lambda: session.step_hole(1)),
                ("previous_failed", "◀ failed", "Previous failed hole of the current method (Shift+N)",
                 lambda: session.step_failed(-1)),
                ("next_failed", "failed ▶", "Next failed hole of the current method (N)", lambda: session.step_failed(1))):
            button = QtWidgets.QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            button.clicked.connect(action)
            header.addWidget(button)
            self.buttons[key] = button
        layout.addLayout(header)
        self.info = hint("Click a hole on the image or a row of the table.")
        layout.addWidget(self.info)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter = splitter
        self.comparison = QtWidgets.QTableWidget(0, 5)
        self.comparison.setHorizontalHeaderLabels(["Method", "CD", "x (px)", "y (px)", "Status"])
        header_view = self.comparison.horizontalHeader()
        header_view.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header_view.setStretchLastSection(True)
        self.comparison.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.comparison.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.comparison.verticalHeader().hide()
        self.comparison.verticalHeader().setDefaultSectionSize(21)
        self.comparison.setToolTip("Click a method to make it the current method everywhere")
        self.comparison.cellClicked.connect(self._method_clicked)
        splitter.addWidget(self.comparison)
        crops = QtWidgets.QWidget()
        crops_layout = QtWidgets.QVBoxLayout(crops)
        crops_layout.setContentsMargins(0, 0, 0, 0)
        crops_row = QtWidgets.QHBoxLayout()
        self.crop_title = QtWidgets.QLabel("Original pixels | method input")
        self.crop_title.setObjectName("hint")
        crops_row.addWidget(self.crop_title, 1)
        self.current_only = QtWidgets.QCheckBox("Current method only")
        self.current_only.setToolTip("Draw only the current method's contour on the method input")
        self.current_only.toggled.connect(self.draw)
        crops_row.addWidget(self.current_only)
        crops_layout.addLayout(crops_row)
        images = QtWidgets.QHBoxLayout()
        self.raw_canvas = ImageCanvas()
        self.input_canvas = ImageCanvas()
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.setMinimumSize(110, 110)
            canvas.plot.hideAxis("bottom")
            canvas.plot.hideAxis("left")
            images.addWidget(canvas)
        self.input_canvas.clicked.connect(self._input_clicked)
        crops_layout.addLayout(images, 1)
        splitter.addWidget(crops)
        profile = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile)
        profile_layout.setContentsMargins(0, 0, 0, 0)
        controls = QtWidgets.QHBoxLayout()
        self.profile_title = QtWidgets.QLabel("")
        self.profile_title.setObjectName("hint")
        self.profile_title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        controls.addWidget(self.profile_title, 1)
        self.projection_axis = QtWidgets.QComboBox()
        self.projection_axis.addItems(["x", "y"])
        self.projection_axis.setToolTip("Projection axis")
        self.projection_axis.currentIndexChanged.connect(self.draw)
        controls.addWidget(self.projection_axis)
        self.ray = QtWidgets.QSpinBox()
        self.ray.setPrefix("Ray ")
        self.ray.setRange(0, 95)
        self.ray.setToolTip("Ray shown in the profile; click a ray in the plot of all rays or on the input")
        self.ray.valueChanged.connect(self.draw)
        controls.addWidget(self.ray)
        profile_layout.addLayout(controls)
        plots = QtWidgets.QHBoxLayout()
        self.profile_plot = plot_widget()
        self.profile_plot.setMinimumHeight(120)
        plots.addWidget(self.profile_plot, 3)
        self.rays_plot = plot_widget(bottom="Angle (°)", left="Edge radius (px)")
        self.rays_plot.scene().sigMouseClicked.connect(self._rays_clicked)
        plots.addWidget(self.rays_plot, 2)
        profile_layout.addLayout(plots, 1)
        splitter.addWidget(profile)
        splitter.setSizes([200, 170, 200])
        layout.addWidget(splitter, 1)
        self.metrics = QtWidgets.QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.verticalHeader().hide()
        self.metrics.verticalHeader().setDefaultSectionSize(20)
        self.metrics.setMinimumHeight(160)
        self.metrics_fold = Fold("Metrics", self.metrics)
        layout.addWidget(self.metrics_fold)
        self.definition = hint("")
        self.definition_fold = Fold("Definition", self.definition)
        layout.addWidget(self.definition_fold)
        for widget in (self.projection_axis, self.ray, self.rays_plot):
            widget.hide()
        session.results_changed.connect(self._results_changed)
        session.method_changed.connect(self.refresh)
        session.selection_changed.connect(lambda reason: self.refresh())

    # ------------------------------------------------------------------------------------ updates
    def _results_changed(self):
        review = self.session.review
        holes = len(review.seeds_xy) if review is not None else 0
        with QtCore.QSignalBlocker(self.hole):
            self.hole.setRange(0, max(0, holes - 1))
        self.count.setText(f"of {holes:,}" if holes else "")
        for button in self.buttons.values():
            button.setEnabled(holes > 0)

    def clear(self, message):
        self.info.setText(message)
        self.comparison.setRowCount(0)
        self.metrics.setRowCount(0)
        self.profile_plot.clear()
        self.rays_plot.clear()
        self.rays_plot.hide()
        self.profile_title.clear()
        self.definition.setText("")
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.clear()

    def refresh(self):
        """The selected hole in every method, then the current method's pixels and profile."""
        session = self.session
        review, index, method = session.review, session.selected, session.method
        self._radial = None
        if review is None:
            self.clear("Preview holes or measure CD, then click a hole on the image or a row of the table.")
            return
        if not len(review.seeds_xy):
            self.clear("No holes found. Check the threshold mask, the area limits, polarity and region.")
            return
        if index is None or method not in review.measurements:
            self.clear("Click a hole on the image or a row of the table.")
            return
        with QtCore.QSignalBlocker(self.hole):
            self.hole.setValue(index)
        for widget, shown in ((self.ray, method in ("gradient", "halfmax")), (self.rays_plot, method in ("gradient", "halfmax")),
                              (self.projection_axis, method in ("projection", "local_projection"))):
            widget.setVisible(shown)
        measured = review.measurements[method]
        seed = review.seeds_xy[index] + review.origin_xy
        page = session.loaded.page if session.loaded is not None else 0
        state = status_text(measured.status[index])
        self.info.setText(f"x {seed[0]:.3f}, y {seed[1]:.3f} px | page {page} | {method_name(method)}: "
                          + (state if measured.valid[index] else f"<span style='color:{FAILED_COLOR}'>{state}</span>"))
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        self.comparison.setHorizontalHeaderLabels(["Method", f"CD ({unit})", "x (px)", "y (px)", "Status"])
        self.comparison.setRowCount(len(review.measurements))
        bold = QtGui.QFont(self.comparison.font())
        bold.setBold(True)
        for row, (name, measurement) in enumerate(review.measurements.items()):
            cd = measurement.metrics.get("cd_equivalent_px")
            center = measurement.centers_xy[index] + review.origin_xy
            values = [method_name(name),
                      f"{cd[index] * scale:.4g}" if cd is not None and np.isfinite(cd[index]) else "--",
                      f"{center[0]:.3f}" if np.isfinite(center[0]) else "--",
                      f"{center[1]:.3f}" if np.isfinite(center[1]) else "--",
                      status_text(measurement.status[index])]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setToolTip(f"{name}: {METHOD_LABELS.get(name, name)} | status {measurement.status[index]}")
                if column == 0:
                    item.setIcon(dot_icon(method_color(row)))
                if column == 4 and not measurement.valid[index]:
                    item.setForeground(QtGui.QColor(FAILED_COLOR))
                if name == method:
                    item.setFont(bold)
                self.comparison.setItem(row, column, item)
        self.comparison.resizeColumnToContents(0)
        rows = [(key, values) for key, values in measured.metrics.items()]
        if review.cd is not None:
            rows = [("cell_area_px2", review.cd.cell_areas_px2), ("mean_original_intensity", review.cd.mean_intensity)] + rows
        self.metrics.setRowCount(len(rows))
        for row, (key, values) in enumerate(rows):
            label, power = metric_label(key, unit)
            value = values[index] * scale ** power
            self.metrics.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            self.metrics.item(row, 0).setToolTip(key)
            self.metrics.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.6g}" if np.isfinite(value) else "--"))
        self.definition.setText(f"<b>{method_name(method)}</b> ({method}: {METHOD_LABELS.get(method, method)})<br>"
                                + method_help(method))
        self.definition.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.draw()

    def _method_clicked(self, row, column):
        item = self.comparison.item(row, 0)
        if item is not None:
            self.session.set_method(item.data(QtCore.Qt.ItemDataRole.UserRole))

    # ---------------------------------------------------------------------------------------- drawing
    def draw(self, *args):
        session = self.session
        review, index, method = session.review, session.selected, session.method
        if review is None or index is None or not len(review.seeds_xy) or method not in review.measurements:
            return
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.clear_overlays()
        self.profile_plot.clear()
        self.profile_plot.enableAutoRange()
        self.profile_plot.setLabel("bottom", "x", units="px")
        self.profile_plot.setLabel("left", "Intensity")
        if review.cd is None:
            self._draw_detection(review, index)
            return
        result = review.cd
        data, mask, description = feature_input(result, method, index)
        ox, oy = feature_origin(result, method, index)
        yy, xx = np.indices(mask.shape)
        in_image = ((yy + oy >= 0) & (yy + oy < review.image.shape[0]) &
                    (xx + ox >= 0) & (xx + ox < review.image.shape[1]))
        raw = review.image[np.clip(yy + oy, 0, review.image.shape[0] - 1),
                           np.clip(xx + ox, 0, review.image.shape[1] - 1)].astype(float)
        origin = np.array([ox, oy]) + review.origin_xy
        self.raw_canvas.set_image(np.where(in_image, raw, np.nan), origin)
        self.input_canvas.set_image(np.where(mask, data, np.nan), origin)
        self.crop_title.setText(f"Original pixels | method input: {description}")
        self.crop_title.setToolTip(description)
        labels = result.cell_labels[np.clip(yy + oy, 0, result.image.shape[0] - 1),
                                    np.clip(xx + ox, 0, result.image.shape[1] - 1)]
        outline = pixel_outline(in_image & (labels == index), origin)
        self.raw_canvas.add_overlay("cell", pg.PlotDataItem(outline[:, 0], outline[:, 1], connect="pairs",
                                                            pen=pg.mkPen("#56d6cf", width=1)))
        for i, (name, measurement) in enumerate(result.measurements.items()):
            if measurement.contours_xy.size and (name == method or not self.current_only.isChecked()):
                contour = measurement.contours_xy[index] + review.origin_xy
                self.input_canvas.add_overlay(name, pg.PlotDataItem(contour[:, 0], contour[:, 1], connect="finite",
                                              pen=pg.mkPen(method_color(i), width=2.5 if name == method else 1)))
        measured = result.measurements[method]
        center = measured.centers_xy[index]
        if not np.isfinite(center).all():
            center = result.seeds_xy[index]
        self.input_canvas.add_overlay("center", pg.ScatterPlotItem(pos=[center + review.origin_xy], symbol="+",
                                                                   size=12, pen="w"))
        if method in ("gradient", "halfmax"):
            self._draw_rays(review, result, method, index, origin)
            return
        color = method_color(list(result.measurements).index(method))
        row = int(np.clip(np.rint(center[1] - oy), 0, data.shape[0] - 1))
        xs = np.arange(data.shape[1]) + origin[0]
        if method in ("projection", "local_projection"):
            axis = self.projection_axis.currentText()
            diagnostic = projection_diagnostic(result, method, index, axis)
            if diagnostic is not None:
                grid, values, slope = diagnostic["grid"], diagnostic["values"], diagnostic["slope"]
                amplitude = max(float(np.max(np.abs(values))), 1e-12)
                slope_scale = max(float(np.max(np.abs(slope))), 1e-12)
                self.profile_title.setText(f"{axis.upper()} projection, width {diagnostic['right'] - diagnostic['left']:.4g} px | "
                                           + _legend((color, "projection"), ("#ffd166", "slope")))
                self.profile_plot.setLabel("bottom", f"{axis} offset in input", units="px")
                self.profile_plot.setLabel("left", "Normalized signal / slope")
                self.profile_plot.plot(grid, values / amplitude, pen=color)
                self.profile_plot.plot(np.arange(len(diagnostic["samples"])), diagnostic["samples"] / amplitude,
                                       pen=None, symbol="o", symbolSize=4, symbolBrush=color)
                self.profile_plot.plot(grid, slope / slope_scale, pen="#ffd166")
                for value in (diagnostic["left"], diagnostic["right"]):
                    self.profile_plot.addItem(pg.InfiniteLine(value, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine)))
            return
        title = [f"Input row y = {row + origin[1]:.0f} px", _legend((color, "samples"))]
        self.profile_plot.plot(xs, np.where(mask[row], data[row], np.nan), pen=color, symbol="o", symbolSize=4)
        if method == "gaussian" and "eigenvalues" in measured.geometry:
            values, axes = measured.geometry["eigenvalues"][index], measured.geometry["axes"][index]
            if np.isfinite(values).all() and (values > 0).all():
                dense_x = np.linspace(0, data.shape[1] - 1, 300)
                delta = np.column_stack((dense_x + ox - center[0], np.full(300, row + oy - center[1])))
                precision = (axes * values) @ axes.T
                peak = measured.metrics["peak_signal"][index]
                model = peak * np.exp(-.5 * np.einsum("ni,ij,nj->n", delta, precision, delta))
                self.profile_plot.plot(dense_x + origin[0], model, pen="#ffd166")
                sample_delta = np.column_stack((np.arange(data.shape[1]) + ox - center[0],
                                                np.full(data.shape[1], row + oy - center[1])))
                sample_model = peak * np.exp(-.5 * np.einsum("ni,ij,nj->n", sample_delta, precision, sample_delta))
                self.profile_plot.plot(xs, np.where(mask[row], data[row] - sample_model, np.nan), pen="#f78c6b")
                title.append(_legend(("#ffd166", "fitted Gaussian"), ("#f78c6b", "residual")))
        if method == "logquad":
            background = 0.0
            if result.meta.get("background") != "model" and mask.any():
                from ..._cpu.cd import _background

                background = _background(data, mask)
            peak = measured.metrics["peak_signal"][index]
            fit_mask = mask & (data - background > result.meta["fit_fraction"] * peak)
            rgba = np.zeros((*mask.shape, 4), np.uint8)
            rgba[fit_mask] = (255, 206, 115, 90)
            self.input_canvas.raster("fit_support", rgba, origin)
            self.profile_plot.addItem(pg.InfiniteLine(background + result.meta["fit_fraction"] * peak, angle=0,
                                                      pen=pg.mkPen("#ffd166", style=QtCore.Qt.PenStyle.DashLine)))
            title.append("gold pixels: log-fit support")
        if "threshold" in measured.metrics and method not in CONTEXT_METHODS:
            threshold = measured.metrics["threshold"][index]
            if np.isfinite(threshold):
                self.profile_plot.addItem(pg.InfiniteLine(threshold, angle=0,
                                                          pen=pg.mkPen("#ffd166", style=QtCore.Qt.PenStyle.DashLine)))
                title.append(f"threshold {threshold:.5g} (dashed)")
        self.profile_title.setText(" | ".join(title))
        self.input_canvas.add_overlay("profile_row", pg.PlotDataItem(
            [origin[0] - .5, origin[0] + data.shape[1] - .5], [origin[1] + row, origin[1] + row],
            pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DotLine)))

    def _draw_detection(self, review, index):
        x, y = np.rint(review.seeds_xy[index]).astype(int)
        radius = max(8, review.parameters["pad"] * 2)
        x0, x1 = max(0, x - radius), min(review.image.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(review.image.shape[0], y + radius + 1)
        origin = np.array([x0, y0]) + review.origin_xy
        self.raw_canvas.set_image(review.image[y0:y1, x0:x1], origin)
        self.input_canvas.set_image(review.detection.binary[y0:y1, x0:x1], origin)
        self.crop_title.setText("Original pixels | threshold mask")
        self.profile_title.setText(f"Center row y = {y + review.origin_xy[1]:.0f} px | " + _legend((method_color(0), "original pixels")))
        self.profile_plot.plot(np.arange(x0, x1) + review.origin_xy[0], review.image[y, x0:x1], pen=method_color(0),
                               symbol="o", symbolSize=3)

    def _draw_rays(self, review, result, method, index, origin):
        key = (id(review), method, index)
        if self._radial is None or self._radial[0] != key:
            self._radial = (key, radial_diagnostic(result, method, index))
        diagnostic = self._radial[1]
        self.rays_plot.clear()
        if diagnostic is None:
            self.profile_title.setText("Too little signal or cell support for radial samples")
            return
        angles = diagnostic["angles"]
        with QtCore.QSignalBlocker(self.ray):
            self.ray.setMaximum(len(angles) - 1)
        ray = self.ray.value()
        radii, profile = diagnostic["radii"], diagnostic["profiles"][ray]
        valid, edge = diagnostic["valid"][ray], diagnostic["edges"][ray]
        divisor = profile[0] if profile[0] > 0 else 1
        color = method_color(list(result.measurements).index(method))
        self.profile_title.setText(f"Ray {ray} at {np.degrees(angles[ray]):.1f}°: "
                                   + (f"edge {edge:.3f} px" if np.isfinite(edge) else "no edge found") + " | "
                                   + _legend((color, "I(r) / I(0)"), ("#ffd166", "falling slope")))
        self.profile_plot.setLabel("bottom", "Radius", units="px")
        self.profile_plot.setLabel("left", "Normalized signal / slope")
        self.profile_plot.plot(radii, np.where(valid, profile / divisor, np.nan), pen=color, symbol="o", symbolSize=3)
        slope = -np.gradient(profile, radii[1] - radii[0])
        slope_scale = max(float(np.max(np.abs(slope[valid]))) if valid.any() else 0, 1e-12)
        self.profile_plot.plot(radii, np.where(valid, slope / slope_scale, np.nan), pen="#ffd166")
        if method == "halfmax":
            self.profile_plot.addItem(pg.InfiniteLine(0.5, angle=0, pen=pg.mkPen("#b39dff", style=QtCore.Qt.PenStyle.DashLine)))
        if np.isfinite(edge):
            self.profile_plot.addItem(pg.InfiniteLine(edge, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine)))
        support = radii[valid][-1] if valid.any() else 1
        self.profile_plot.setXRange(0, max(support + 0.1, float(edge) + .2 if np.isfinite(edge) else 1), padding=0)
        self.profile_plot.setYRange(-0.15, 1.15)
        # Every ray's edge radius: the rays without an edge (what fails the hole) in red.
        edges = diagnostic["edges"]
        degrees = np.degrees(angles)
        found = np.isfinite(edges)
        self.rays_plot.plot(degrees[found], edges[found], pen=None, symbol="o", symbolSize=5, symbolBrush=color)
        if (~found).any():
            self.rays_plot.plot(degrees[~found], np.zeros((~found).sum()), pen=None, symbol="x", symbolSize=8,
                                symbolPen=FAILED_COLOR)
        self.rays_plot.plot([degrees[ray]], [edges[ray] if found[ray] else 0], pen=None, symbol="o", symbolSize=11,
                            symbolBrush=None, symbolPen="w")
        self.rays_plot.setXRange(0, 360, padding=0.02)
        self.rays_plot.setTitle(f"{found.sum()} of {len(edges)} rays with an edge", size="9pt")
        direction = np.array([np.cos(angles[ray]), np.sin(angles[ray])])
        ray_center = diagnostic["center"] + origin
        end = ray_center + direction * support
        self.input_canvas.add_overlay("ray", pg.PlotDataItem([ray_center[0], end[0]], [ray_center[1], end[1]],
                                                             pen=pg.mkPen("w", width=2)))
        if np.isfinite(edge):
            self.input_canvas.add_overlay("edge", pg.ScatterPlotItem(pos=[ray_center + direction * edge], size=10,
                                                                     brush="#ffd166", pen=None))
        self._ray_center = ray_center

    def _rays_clicked(self, event):
        if self._radial is None or self._radial[1] is None:
            return
        point = self.rays_plot.getViewBox().mapSceneToView(event.scenePos())
        angles = np.degrees(self._radial[1]["angles"])
        distance = np.abs((angles - point.x() + 180) % 360 - 180)
        self.ray.setValue(int(np.argmin(distance)))

    def _input_clicked(self, x, y):
        """A click on the input crop picks the ray pointing there (radial methods)."""
        if self._radial is None or self._radial[1] is None or not self.ray.isVisible():
            return
        dx, dy = x - self._ray_center[0], y - self._ray_center[1]
        angle = np.arctan2(dy, dx) % (2 * np.pi)
        angles = self._radial[1]["angles"] % (2 * np.pi)
        distance = np.abs((angles - angle + np.pi) % (2 * np.pi) - np.pi)
        self.ray.setValue(int(np.argmin(distance)))
