"""Interactive PyQt6/pyqtgraph inspection application.

Run ``python -m subpx.workbench [image]`` or call :func:`launch_workbench`.
Qt is deliberately imported only when this optional module is requested.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import replace
from html import escape as html_escape
import json
from pathlib import Path
import re
import sys
import time
import traceback
import weakref

import numpy as np

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:
    raise ImportError('The workbench requires PyQt6 and pyqtgraph. Install: pip install -e ".[analysis,viewer]"') from exc

from ._workbench_data import (
    HOLE_CHECK_MAX_ERROR, AnalysisReview, LoadedImage, PageReader, analyze_image, analyze_patterns, array_bytes,
    cell_boundary, cell_mask_map, cell_values_map, demo_image, export_review, feature_input, feature_origin,
    grayscale, load_image, page_preview, projection_diagnostic, radial_diagnostic, register_images,
)
from .cd import METHOD_LABELS


COLORS = ("#56d6cf", "#ffce73", "#b49aff", "#ff8faa", "#79b8ff", "#9be28a", "#ff9f62")
METHOD_HELP = {
    "gaussian": "Joint Gaussian model. Neighbour signals and a smooth background are removed. The contour is the fitted half-maximum ellipse. Inspect fit_rmse and deblend convergence; a Gaussian shape is an assumption.",
    "logquad": "Weighted quadratic fit to log(I - background), above the fit-fraction cutoff. The negative Hessian defines the half-maximum ellipse. Inspect log_fit_rmse, fit support and whether the contour stays inside the cell.",
    "gradient": "Cubic samples along radial rays. The edge is the maximum falling slope with parabolic refinement. The highlighted ray and its actual samples are shown below. Missing rays make a measurement invalid.",
    "halfmax": "Cubic samples along radial rays. Each edge is the first crossing of half the center signal, interpolated between samples. The highlighted ray shows the measured edge and half-height level.",
    "projection": "Summed x/y profiles of the cleaned cell. Width is the distance between the steepest rising and falling slopes of a zero-padded cubic spline. Select x or y below to inspect the actual samples, spline, slopes and edge picks.",
    "local_projection": "Summed x/y profiles of the raw Voronoi cell after subtracting its median edge level. Width is the distance between the steepest rise and fall. Neighbours are not subtracted.",
    "disk": "A round hole convolved with an imaging blur, fitted with free brightness and background. Requires a known PSF FWHM or nominal CD. Its diameter has a different definition from spot FWHM.",
    "flux_ratio": "Hole light divided by its own fitted background minus detector dark level. Requires a nominal CD or PSF calibration. A calibration derived per image should not be used to compare image means.",
}
AREA_HELP = "The peak-connected region above the method's threshold is measured on a 4x cubic enlargement. Equivalent CD = 2 sqrt(area / pi). The outline, threshold and local level depend on the selected method; x/y widths use area and second moments."
# The default methods of scripts/cd_batch.py (ALL_METHODS of scripts/cd_report.py); a test pins them.
BATCH_METHODS = ("gaussian", "projection", "half_area", "half_area_min", "half_area_min_avg", "tail_area",
                 "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
RECIPES = {
    "batch": ("Batch default", BATCH_METHODS),
    "spot": ("Spot widths", ("gaussian", "logquad", "gradient", "halfmax")),
    "local": ("Raw-cell widths", ("local_projection", "local_half_area")),
}
# One display name per method. Library keys and labels stay in tooltips and exports.
METHOD_NAMES = {
    "gaussian": "Gaussian fit", "logquad": "Log fit", "gradient": "Gradient edge", "halfmax": "Half-height",
    "disk": "Blurred-hole fit", "flux_ratio": "Flux ratio", "projection": "Projection", "half_area": "Half area",
    "half_area_min": "Half area, min level", "half_area_min_avg": "Half area, min level, avg peak",
    "tail_area": "Tail area", "tail_area_norm": "Tail area, no level", "edge_area": "Edge area",
    "foot_area": "Foot area", "foot_area_ratio": "Foot area, image ratio",
    "local_projection": "Raw-cell projection", "local_half_area": "Raw-cell half area",
    "local_half_area_min_avg": "Raw-cell half area, min level, avg peak", "local_tail_area": "Raw-cell tail area",
    "pixel_half_area": "Pixel half area", "pixel_half_area_min_avg": "Pixel half area, min level, avg peak",
    "pixel_tail_area": "Pixel tail area", "detection": "Detected centers",
}
CALIBRATED_METHODS = {"disk", "flux_ratio"}
COMMON_SETTINGS = ("threshold", "invert", "area_min", "area_max", "pad", "backend", "pixel_size", "unit")
CD_SETTINGS = ("methods", "background", "seeds", "n_angles", "radial_step", "smooth_sigma", "fit_fraction",
               "half_area_level", "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations",
               "deblend_tolerance", "nominal_cd", "psf_fwhm", "dark_level")
SETTINGS_KEY = "workbench/v1"
# Results are kept for the open page and the most recent other pages: at most this many pages and bytes.
RESULT_PAGES = 3
RESULT_BYTES = 3e9


def method_name(key):
    return METHOD_NAMES.get(key, key)


def plain_error(detail):
    """The last line of a traceback without the exception class, e.g. 'Area max is below area min.'"""
    lines = [line for line in str(detail).strip().splitlines() if line.strip()]
    line = lines[-1].strip() if lines else "Unknown error"
    match = re.match(r"^[\w.]*(?:Error|Exception|Exit)\w*: (.+)$", line)
    return match.group(1) if match else line


_WINDOWS = weakref.WeakSet()
_previous_hook = None


def _exception_hook(kind, value, tb):
    """Errors raised in click or redraw handlers reach the open windows' error banner and log."""
    detail = "".join(traceback.format_exception(kind, value, tb))
    for window in list(_WINDOWS):
        try:
            window._show_error(detail, context="Unexpected error")
        except Exception:
            pass
    (_previous_hook or sys.__excepthook__)(kind, value, tb)


def _arrow_images():
    """Spin-box arrow images drawn once per process (a styled spin box shows no arrows of its own)."""
    import tempfile

    folder = Path(tempfile.gettempdir()) / "subpx-workbench"
    folder.mkdir(exist_ok=True)
    paths = []
    for name, points in (("up", ((1, 6), (5, 2), (9, 6))), ("down", ((1, 2), (5, 6), (9, 2)))):
        path = folder / f"arrow_{name}.png"
        image = QtGui.QImage(10, 8, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#a9bbce"))
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*point) for point in points]))
        painter.end()
        image.save(str(path))
        paths.append(path.as_posix())
    return paths


def _install_exception_hook():
    global _previous_hook
    if sys.excepthook is not _exception_hook:
        _previous_hook, sys.excepthook = sys.excepthook, _exception_hook


def no_si_prefix(plot_item):
    """Pixel axes read 'x (px)', never 'x (kpx)' or 'y (mpx)'."""
    for name in ("bottom", "left"):
        plot_item.getAxis(name).enableAutoSIPrefix(False)


class TaskTabs(QtWidgets.QTabBar):
    def currentData(self):
        return self.tabData(self.currentIndex())

    def findData(self, value):
        return next((i for i in range(self.count()) if self.tabData(i) == value), -1)


class _Signals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class _Job(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.signals.finished.emit(self.function())
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class ImageCanvas(QtWidgets.QWidget):
    """Image coordinates use integer pixel centers, x right and y downward."""

    clicked = QtCore.pyqtSignal(float, float)
    hovered = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None, *, histogram=False):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="#0c121b")
        self.plot.setLabel("bottom", "x", units="px")
        self.plot.setLabel("left", "y", units="px")
        no_si_prefix(self.plot.getPlotItem())
        # Page keys act on the image only when it has focus; a click gives it focus.
        self.plot.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.plot.setAspectLocked(True)
        self.plot.invertY(True)
        self.plot.showGrid(x=True, y=True, alpha=0.08)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_item.setOpts(autoDownsample=True)
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot, 1)
        self.histogram = None
        if histogram:
            self.histogram = pg.HistogramLUTWidget()
            self.histogram.setMinimumWidth(100)
            self.histogram.setMaximumWidth(125)
            self.histogram.setImageItem(self.image_item)
            self.histogram.gradient.loadPreset("grey")
            self.side_layout = QtWidgets.QVBoxLayout()
            self.side_layout.setContentsMargins(0, 0, 0, 0)
            self.side_layout.addWidget(self.histogram, 1)
            layout.addLayout(self.side_layout)
        self.overlays = {}
        self._image_key = None
        self._image_bounds = None
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self._hover_proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self._hovered)

    def _clicked(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            point = self.plot.getViewBox().mapSceneToView(event.scenePos())
            self.clicked.emit(point.x(), point.y())

    def _hovered(self, event):
        if self.plot.sceneBoundingRect().contains(event[0]):
            point = self.plot.getViewBox().mapSceneToView(event[0])
            self.hovered.emit(point.x(), point.y())

    def set_image(self, data, origin=(0, 0), *, fit=True, categorical=False, keep_levels=False):
        key = (id(data), categorical)
        preserve_levels = key == self._image_key or keep_levels
        self._image_key = key
        data = np.asarray(data)
        if not data.size:
            self.image_item.clear()
            return
        if data.dtype == bool:
            data = data.astype(np.uint8)
        self.image_item.setImage(data, autoLevels=False)
        self._image_bounds = QtCore.QRectF(origin[0] - 0.5, origin[1] - 0.5, data.shape[1], data.shape[0])
        self.image_item.setRect(self._image_bounds)
        if data.ndim == 2 and not preserve_levels:
            sample = data[::max(1, data.shape[0] // 512), ::max(1, data.shape[1] // 512)]
            finite = sample[np.isfinite(sample)]
            low, high = np.percentile(finite, (1, 99.5)) if finite.size else (0, 1)
            if high <= low:
                high = low + 1
            self.image_item.setLevels((float(low), float(high)))
            if self.histogram:
                self.histogram.setLevels(float(low), float(high))
                self.histogram.gradient.loadPreset("viridis" if categorical else "grey")
            else:
                self.image_item.setLookupTable(pg.colormap.get("viridis").getLookupTable() if categorical else None)
        if fit:
            self.fit_image()

    def fit_image(self):
        if self._image_bounds is not None:
            self.plot.getViewBox().setRange(self._image_bounds, padding=0.03)

    def clear_overlays(self):
        for item in self.overlays.values():
            self.plot.removeItem(item)
        self.overlays.clear()

    def add_overlay(self, name, item):
        old = self.overlays.pop(name, None)
        if old is not None:
            self.plot.removeItem(old)
        self.overlays[name] = item
        self.plot.addItem(item)
        return item

    def raster(self, name, rgba, origin):
        item = pg.ImageItem(rgba, axisOrder="row-major")
        item.setRect(QtCore.QRectF(origin[0] - 0.5, origin[1] - 0.5, rgba.shape[1], rgba.shape[0]))
        return self.add_overlay(name, item)


class MeasurementTable(QtCore.QAbstractTableModel):
    """The holes table on NumPy columns: built once per result; a method switch, filter or sort reorders indices.

    Fixed column order: ID, x, y, Status, CD x and CD y of the shown method, then the CD of every method in
    Setup order (the shown one in bold). A sort stays when the method or the filter changes.
    """

    METHOD_COLUMNS = 6  # the first per-method CD column when the result has CDs

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review = self.method = None
        self.failed_only = False
        self.columns = []  # dicts: header, values (one per hole), tooltip, method (CD columns)
        self.cd_methods = []
        self.order = np.empty(0, int)  # hole per row
        self.position = np.empty(0, int)  # row per hole, -1 when filtered out
        self.bad = np.empty(0, bool)  # the shown method failed
        self.sort_column, self.sort_order = None, QtCore.Qt.SortOrder.AscendingOrder

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.order)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        column, hole = self.columns[index.column()], int(self.order[index.row()])
        Role = QtCore.Qt.ItemDataRole
        if role == Role.DisplayRole:
            value = column["values"][hole]
            if isinstance(value, (float, np.floating)):
                return f"{value:.5g}" if np.isfinite(value) else "--"
            return str(value)
        if role == Role.UserRole:
            return hole
        method = column.get("method")
        if role == Role.ForegroundRole:
            failed = self.review.measurements[method].status[hole] != "ok" if method else self.bad[hole]
            return QtGui.QColor("#ff8c96") if failed else None
        if role == Role.ToolTipRole and method:
            return (f"{method_name(method)} ({method}: {METHOD_LABELS.get(method, method)}): "
                    f"{self.review.measurements[method].status[hole]}")
        if role == Role.TextAlignmentRole and index.column() != 3:
            return QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation != QtCore.Qt.Orientation.Horizontal or not 0 <= section < len(self.columns):
            return None
        column = self.columns[section]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return column["header"]
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            return column.get("tooltip")
        if role == QtCore.Qt.ItemDataRole.FontRole and column.get("method") == self.method:
            font = QtGui.QFont()
            font.setBold(True)
            return font
        return None

    def set_review(self, review, method, failed_only=False):
        self.beginResetModel()
        self.review, self.method, self.failed_only = review, None, failed_only
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        self.cd_methods = [name for name, result in review.measurements.items() if "cd_equivalent_px" in result.metrics]
        shown = "Center and status of the shown method (the method box)"
        self.columns = [dict(header="ID", values=np.arange(len(review.seeds_xy)), tooltip="Hole number"),
                        dict(header="x (px)", tooltip=shown), dict(header="y (px)", tooltip=shown),
                        dict(header="Status", tooltip=shown)]
        if self.cd_methods:
            self.columns += [dict(header=f"CD x ({unit})", tooltip="CD x of the shown method"),
                             dict(header=f"CD y ({unit})", tooltip="CD y of the shown method")]
            self.columns += [dict(header=f"{method_name(name)}\nCD ({unit})", method=name,
                                  tooltip=f"{name}: {METHOD_LABELS.get(name, name)}",
                                  values=review.measurements[name].metrics["cd_equivalent_px"] * scale)
                             for name in self.cd_methods]
        self._set_method(method)
        self._filter_and_sort()
        self.endResetModel()

    def _set_method(self, method):
        review, scale = self.review, self.review.parameters["pixel_size"]
        measured = review.measurements[method]
        self.method = method
        points = measured.centers_xy + review.origin_xy
        self.columns[1]["values"], self.columns[2]["values"] = points[:, 0], points[:, 1]
        self.columns[3]["values"] = measured.status
        self.bad = measured.status != "ok"
        if self.cd_methods:
            missing = np.full(len(points), np.nan)
            for column, key in zip(self.columns[4:6], ("cd_x_px", "cd_y_px")):
                values = measured.metrics.get(key)
                column["values"] = values * scale if values is not None else missing

    def set_method(self, method, failed_only=False):
        """Show another method of the same result (or another filter); rows keep the sort."""
        if method == self.method and failed_only == self.failed_only:
            return
        self._set_method(method)
        self.failed_only = failed_only
        self._relayout()
        if self.order.size:
            self.dataChanged.emit(self.index(0, 1), self.index(len(self.order) - 1, len(self.columns) - 1))
        self.headerDataChanged.emit(QtCore.Qt.Orientation.Horizontal, 0, len(self.columns) - 1)

    def _filter_and_sort(self):
        rows = np.flatnonzero(self.bad) if self.failed_only else np.arange(len(self.bad))
        if self.sort_column is not None and self.sort_column < len(self.columns):
            values = self.columns[self.sort_column]["values"][rows]
            descending = self.sort_order == QtCore.Qt.SortOrder.DescendingOrder
            if values.dtype.kind in "fiu":
                # Missing values (NaN) last in both directions.
                rows = rows[np.argsort(-values if descending else values, kind="stable")]
            else:
                rows = rows[np.argsort(values, kind="stable")[::-1 if descending else 1]]
        self.order = rows
        self.position = np.full(len(self.bad), -1)
        self.position[rows] = np.arange(len(rows))

    def _relayout(self):
        """Reorder the rows; the selection and current row stay on their holes."""
        self.layoutAboutToBeChanged.emit()
        persistent = self.persistentIndexList()
        holes = [int(self.order[index.row()]) for index in persistent]
        self._filter_and_sort()
        self.changePersistentIndexList(persistent, [
            self.index(int(self.position[hole]), index.column()) if self.position[hole] >= 0 else QtCore.QModelIndex()
            for hole, index in zip(holes, persistent)])
        self.layoutChanged.emit()

    def row_of_hole(self, hole):
        """The table row of a hole, -1 when the filter hides it."""
        return int(self.position[hole]) if 0 <= hole < len(self.position) else -1

    def hole_at(self, row):
        return int(self.order[row])

    def value(self, row, column):
        return self.columns[column]["values"][self.order[row]]

    def method_of_column(self, column):
        return self.columns[column].get("method") if 0 <= column < len(self.columns) else None

    def clear(self):
        self.beginResetModel()
        self.review = self.method = None
        self.columns, self.cd_methods = [], []
        self.order = self.position = np.empty(0, int)
        self.bad = np.empty(0, bool)
        self.endResetModel()

    def sort(self, column, order=QtCore.Qt.SortOrder.AscendingOrder):
        self.sort_column, self.sort_order = column, order
        if self.review is not None:
            self._relayout()


def _mask_item(color, z):
    """An image item that paints the pixels equal to 1 in ``color`` (RGBA) and leaves 0 transparent."""
    item = pg.ImageItem(axisOrder="row-major")
    item.setLookupTable(np.array([(0, 0, 0, 0), color], np.uint8))
    item.setLevels((0, 1))
    item.setZValue(z)
    return item


def _set_raster(item, data, origin):
    if getattr(item, "source", None) is not data:  # re-render only for a new map
        item.source = data
        item.setImage(data, autoLevels=False)
    item.setRect(QtCore.QRectF(origin[0] - 0.5, origin[1] - 0.5, data.shape[1], data.shape[0]))
    item.show()


class HoleLayers(QtCore.QObject):
    """The overlays of one result on the main image (spec 4.4).

    Raster maps (CD colors, failed cells, Voronoi edges) are made once per result and method and shown at every
    zoom. Centers, contours, pitch pairs and rejected boxes are drawn for the holes in view only, and only while
    at most VECTOR_LIMIT holes are in view; farther out the raster maps carry the picture. The selection marker
    is an item of its own: selecting a hole redraws nothing else.
    """

    VECTOR_LIMIT = 2000
    changed = QtCore.pyqtSignal()  # vectors appeared or gave way to the rasters

    def __init__(self, plot):
        super().__init__(plot)
        self.plot, self.view = plot, plot.getViewBox()
        self.cd_map = pg.ImageItem(axisOrder="row-major")
        self.cd_map.setOpacity(0.6)
        self.cd_map.setZValue(5)
        self.failures = _mask_item((255, 116, 138, 150), 6)
        self.edges = _mask_item((86, 214, 207, 165), 7)
        self.centers = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#56d6cf", width=1), brush=pg.mkBrush(86, 214, 207, 60),
                                          symbol="+")
        self.failed = pg.ScatterPlotItem(size=10, pen=pg.mkPen("#ff748a", width=2), brush=None, symbol="x")
        self.contours = pg.PlotDataItem(connect="finite")
        self.pairs = {axis: pg.PlotDataItem(connect="pairs", pen=pg.mkPen(color, width=1))
                      for axis, color in (("x", "#ffce73"), ("y", "#b49aff"))}
        self.reference = pg.ScatterPlotItem(symbol="star", size=18, brush="#ffce73", pen="#fff3cf")
        self.rejected = pg.PlotDataItem(connect="finite", pen="#ff748a")
        self.selection = pg.ScatterPlotItem(symbol="o", size=20, brush=None, pen=pg.mkPen("#ffffff", width=2))
        self.vectors = [self.centers, self.failed, self.contours, *self.pairs.values(), self.rejected]
        for z, item in enumerate([*self.vectors, self.reference, self.selection], start=20):
            item.setZValue(z)
        for item in (self.cd_map, self.failures, self.edges, *self.vectors, self.reference, self.selection):
            item.hide()
            plot.addItem(item)
        self.state = None
        self._review = None
        self._maps = {}
        self.in_view, self.near = 0, True
        self._timer = QtCore.QTimer(self, singleShot=True, interval=0, timeout=self._update_vectors)
        self.view.sigRangeChanged.connect(lambda *args: self._timer.start())

    def _map(self, key, build):
        if key not in self._maps:
            self._maps[key] = build()
        return self._maps[key]

    def show(self, review, method, origin, layers):
        """Show ``review``'s ``method`` with the named ``layers`` on; ``origin``: display position of its pixel [0, 0].

        ``review=None`` hides every layer.
        """
        if review is not self._review:
            self._review, self._maps = review, {}
        if review is None:
            self.state = None
            for item in (self.cd_map, self.failures, self.edges, *self.vectors, self.reference):
                item.hide()
            self.changed.emit()
            return
        measured = review.measurements[method]
        origin = np.asarray(origin, float)
        points = measured.centers_xy + origin
        self.state = dict(review=review, method=method, origin=origin, layers=set(layers), points=points,
                          seeds=review.seeds_xy + origin, valid=measured.valid & np.isfinite(points).all(axis=1),
                          margin=np.sqrt(review.image.size / max(len(review.seeds_xy), 1)))
        cd = review.cd
        values = measured.metrics.get("cd_equivalent_px")
        if "CD colors" in layers and cd is not None and values is not None:
            scale = review.parameters["pixel_size"]
            _set_raster(self.cd_map, self._map(("cd", method), lambda: cell_values_map(
                cd.cell_labels, np.where(self.state["valid"], values * scale, np.nan))), origin)
        else:
            self.cd_map.hide()
        if "Failed cells" in layers and cd is not None and not measured.valid.all():
            _set_raster(self.failures, self._map(("failed", method), lambda: cell_mask_map(
                cd.cell_labels, ~measured.valid)), origin)
        else:
            self.failures.hide()
        if "Voronoi edges" in layers and cd is not None:
            _set_raster(self.edges, self._map("edges", lambda: cell_boundary(cd.cell_labels).view(np.uint8)), origin)
        else:
            self.edges.hide()
        self.contours.setPen(pg.mkPen(COLORS[list(review.measurements).index(method) % len(COLORS)], width=1.3))
        self._update_vectors()

    def _update_vectors(self):
        state = self.state
        if state is None:
            return
        review, layers, points, seeds = state["review"], state["layers"], state["points"], state["seeds"]
        (x0, x1), (y0, y1) = self.view.viewRange()
        margin = state["margin"]
        inside = ((seeds[:, 0] >= x0 - margin) & (seeds[:, 0] <= x1 + margin)
                  & (seeds[:, 1] >= y0 - margin) & (seeds[:, 1] <= y1 + margin))
        near, was = int(inside.sum()) <= self.VECTOR_LIMIT, self.near
        self.in_view, self.near = int(inside.sum()), near
        measured, valid = review.measurements[state["method"]], state["valid"]
        if near and "Centers" in layers:
            self.centers.setData(pos=points[inside & valid])
            self.failed.setData(pos=seeds[inside & ~valid])
        self.centers.setVisible(near and "Centers" in layers)
        self.failed.setVisible(near and "Centers" in layers)
        contours = near and "Contours" in layers and measured.contours_xy.size > 0
        if contours:
            shapes = measured.contours_xy[inside] + state["origin"]
            segments = np.concatenate((shapes, np.full((len(shapes), 1, 2), np.nan)), axis=1).reshape(-1, 2)
            self.contours.setData(segments[:, 0], segments[:, 1], connect="finite")
        self.contours.setVisible(contours)
        pattern = review.patterns.get(state["method"]) if "Pitch pairs" in layers else None
        for axis, item in self.pairs.items():
            pairs = pattern.profiles[axis].pairs if pattern is not None else np.empty((0, 2), int)
            shown = near and len(pairs) > 0
            if shown:
                segments = points[pairs[inside[pairs].any(axis=1)]].reshape(-1, 2)
                item.setData(segments[:, 0], segments[:, 1], connect="pairs")
            item.setVisible(shown)
        if pattern is not None:
            self.reference.setData(pos=[seeds[pattern.reference_index]])
        self.reference.setVisible(pattern is not None)
        rejected = near and "Rejected boxes" in layers and review.detection is not None
        if rejected:
            boxes = review.detection.boxes[~review.detection.kept]
            if len(boxes):
                x, y, w, h = (boxes[:, i].astype(float)[:, None] for i in range(4))
                corners = np.stack((np.hstack((x, x + w, x + w, x, x, x)) - .5,
                                    np.hstack((y, y, y + h, y + h, y, y)) - .5), axis=-1)
                corners[:, -1] = np.nan
                corners += state["origin"]
                middle = corners[:, 0] + np.column_stack((w[:, 0], h[:, 0])) / 2
                corners = corners[(middle[:, 0] >= x0) & (middle[:, 0] <= x1) & (middle[:, 1] >= y0) & (middle[:, 1] <= y1)]
                self.rejected.setData(corners[..., 0].ravel(), corners[..., 1].ravel(), connect="finite")
            rejected = len(boxes) > 0
        self.rejected.setVisible(rejected)
        if near != was:
            self.changed.emit()

    def hidden_vectors(self):
        """True when vector layers are on but give way to the rasters (too many holes in view)."""
        return (self.state is not None and not self.near
                and bool(self.state["layers"] & {"Centers", "Contours", "Pitch pairs", "Rejected boxes"}))

    def set_selection(self, point):
        if point is None:
            self.selection.hide()
        else:
            self.selection.setData(pos=[point])
            self.selection.show()


class PageStack(QtCore.QObject):
    """Every page of the open file one under another ("Stack vertically"), drawn as the pages come into view.

    Pages out of view are not drawn. Pages in view are drawn from a reduced preview when zoomed out and at full
    resolution when zoomed in; previews are kept for every page seen (small), full pages up to FULL_BYTES. The
    active page is the canvas's own image and is never drawn here. Decoding runs a page at a time between events,
    so the window stays responsive while a large file fills in.
    """

    FULL_BYTES = 128e6
    FULL_SCALE = 3.0  # image pixels per screen pixel below which pages in view are drawn at full resolution
    LABEL_PAGES = 12  # page labels while at most this many pages are in view
    failed = QtCore.pyqtSignal(str)

    def __init__(self, plot, reader):
        super().__init__(plot)
        self.plot, self.view, self.reader = plot, plot.getViewBox(), reader
        self.shapes = np.array(reader.shapes, int).reshape(-1, 2)
        gap = max(8, round(int(self.shapes[:, 1].max()) * 0.04))
        self.offsets = np.concatenate(([0], np.cumsum(self.shapes[:-1, 0] + gap)))
        self.active, self.levels, self.visible = None, None, True
        self.previews, self.full, self.items, self.labels = {}, OrderedDict(), {}, {}
        self.broken = set()
        self.borders = pg.PlotDataItem(connect="finite", pen=pg.mkPen("#354a60", width=1))
        self.borders.setZValue(10)
        corners = []
        for (h, w), y in zip(self.shapes, self.offsets):
            corners += [(-.5, y - .5), (w - .5, y - .5), (w - .5, y + h - .5), (-.5, y + h - .5), (-.5, y - .5),
                        (np.nan, np.nan)]
        corners = np.array(corners)
        self.borders.setData(corners[:, 0], corners[:, 1], connect="finite")
        self.active_border = QtWidgets.QGraphicsRectItem()
        self.active_border.setPen(pg.mkPen("#56d6cf", width=2))
        self.active_border.setZValue(11)
        plot.addItem(self.borders)
        plot.addItem(self.active_border)
        self._queue = []
        self._update_timer = QtCore.QTimer(self, singleShot=True, interval=0, timeout=self.update)
        self._decode_timer = QtCore.QTimer(self, singleShot=True, interval=0, timeout=self._decode)
        self.view.sigRangeChanged.connect(self._schedule)

    def _schedule(self, *args):
        if self.visible:
            self._update_timer.start()

    def bounds(self):
        """The whole stack, with room above the first page for its label."""
        height = int(self.offsets[-1] + self.shapes[-1, 0])
        headroom = max(16, height * .05)
        return QtCore.QRectF(-0.5, -0.5 - headroom, int(self.shapes[:, 1].max()), height + headroom)

    def page_at(self, x, y):
        """The page under the stack position (x, y), or None in a gap."""
        page = int(np.searchsorted(self.offsets, y + 0.5, side="right")) - 1
        if page >= 0:
            h, w = self.shapes[page]
            if -0.5 <= x < w - 0.5 and y < self.offsets[page] + h - 0.5:
                return page
        return None

    def page(self, page):
        """The page decoded at full resolution, if it is in memory."""
        if page in self.full:
            self.full.move_to_end(page)
            return self.full[page]
        return None

    def remember(self, loaded):
        """Keep a page that was decoded elsewhere (the page that was active), so it needs no decoding."""
        if loaded.page not in self.previews:
            self.previews[loaded.page] = page_preview(loaded.image)
        self.full[loaded.page] = loaded
        self.full.move_to_end(loaded.page)
        self._trim()

    def set_active(self, page):
        self.active = page
        h, w = self.shapes[page]
        self.active_border.setRect(-0.5, self.offsets[page] - 0.5, w, h)
        item = self.items.pop(page, None)
        if item is not None:
            self.plot.removeItem(item)
        self.update()

    def set_levels(self, levels):
        """Contrast of the active page, used for every page."""
        self.levels = tuple(levels)
        for item in self.items.values():
            item.setLevels(self.levels)

    def set_visible(self, visible):
        self.visible = visible
        for item in (self.borders, self.active_border):
            item.setVisible(visible)
        if not visible:
            for item in (*self.items.values(), *self.labels.values()):
                self.plot.removeItem(item)
            self.items, self.labels, self._queue = {}, {}, []
        else:
            self.update()

    def close(self):
        self.set_visible(False)
        self._update_timer.stop()
        self._decode_timer.stop()
        for item in (self.borders, self.active_border):
            self.plot.removeItem(item)
        self.view.sigRangeChanged.disconnect(self._schedule)
        self.reader.close()

    def _in_view(self):
        (x0, x1), (y0, y1) = self.view.viewRange()
        pages = np.flatnonzero((self.offsets + self.shapes[:, 0] >= y0) & (self.offsets <= y1) & (x0 <= self.shapes[:, 1]))
        full = self.view.viewPixelSize()[1] < self.FULL_SCALE
        return pages, full

    def update(self):
        if not self.visible or self.active is None:
            return
        pages, full = self._in_view()
        wanted = set(int(page) for page in pages) - {self.active}
        for page in [page for page in self.items if page not in wanted]:
            self.plot.removeItem(self.items.pop(page))
        missing = []
        for page in sorted(wanted):
            if not self._draw(page, full) and page not in self.broken:
                missing.append(page)
        middle = np.mean(self.view.viewRange()[1])
        self._queue = sorted(missing, key=lambda page: abs(self.offsets[page] + self.shapes[page, 0] / 2 - middle))
        if self._queue:
            self._decode_timer.start()
        labelled = set(int(page) for page in pages) if len(pages) <= self.LABEL_PAGES else set()
        for page in [page for page in self.labels if page not in labelled]:
            self.plot.removeItem(self.labels.pop(page))
        for page in labelled:
            active = page == self.active
            text = f"Page {page}" + (" · active" if active else " · click to select")
            label = self.labels.get(page)
            if label is None:
                label = self.labels[page] = pg.TextItem(anchor=(0, 1), fill="#151e2b")
                label.setPos(0, self.offsets[page] - .5)
                label.setZValue(12)
                self.plot.addItem(label)
            label.setText(text, color="#56d6cf" if active else "#a9bbce")

    def _draw(self, page, full):
        """Draw a page from memory; False when it must be decoded first."""
        loaded = self.full.get(page) if full else None
        if loaded is not None:
            data, factor = loaded.image, 1
        elif page in self.previews and not full:
            data, factor = self.previews[page]
        elif page in self.previews:  # full resolution wanted: the preview until it is decoded
            self._show(page, *self.previews[page])
            return False
        else:
            return False
        self._show(page, data, factor)
        return True

    def _show(self, page, data, factor):
        item = self.items.get(page)
        if item is None:
            item = self.items[page] = pg.ImageItem(axisOrder="row-major")
            item.setOpts(autoDownsample=True)  # averaged when small, as the active page
            item.setZValue(-1)
            self.plot.addItem(item)
        if getattr(item, "source", None) is not data:
            item.source = data
            item.setImage(data, autoLevels=False)
            if self.levels is not None:
                item.setLevels(self.levels)
            item.setRect(QtCore.QRectF(-0.5, self.offsets[page] - 0.5, data.shape[1] * factor, data.shape[0] * factor))

    def _decode(self):
        """Decode the pages in view nearest the middle first, for at most about 30 ms per turn of the event loop."""
        started = time.perf_counter()
        pages, full = self._in_view()
        wanted = set(int(page) for page in pages)
        while self._queue and time.perf_counter() - started < 0.03:
            page = self._queue.pop(0)
            if page not in wanted or page == self.active:
                continue
            try:
                loaded = self.reader.read(page)
            except Exception as exc:
                self.broken.add(page)
                self.failed.emit(f"Page {page} could not be read: {exc}")
                continue
            self.previews[page] = page_preview(loaded.image)
            if full or self.previews[page][1] == 1:
                self.full[page] = loaded
                self._trim()
            self._draw(page, full)
        if self._queue:
            self._decode_timer.start()

    def _trim(self):
        total = sum(loaded.image.nbytes for loaded in self.full.values())
        while len(self.full) > 1 and total > self.FULL_BYTES:
            page, loaded = self.full.popitem(last=False)
            total -= loaded.image.nbytes

    def memory(self):
        return sum(data.nbytes for data, _ in self.previews.values()) + sum(
            loaded.image.nbytes for loaded in self.full.values())


class InspectionWindow(QtWidgets.QMainWindow):
    """Desktop workbench for reviewing repository measurements and their inputs."""

    def __init__(self, image=None, *, title="subpx | Inspection workbench", settings=None):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)
        self.setAcceptDrops(True)
        # Settings and window layout persist here; tests pass their own QSettings file.
        self._settings = settings if settings is not None else QtCore.QSettings("subpx", "workbench")
        self.loaded = None
        self.moving = None
        self.registration = None
        self.review = None
        self.selected = None
        self._selected_xy = None  # loaded-page [x, y] of the selected hole, kept across pages and runs
        self._region = None  # (page-local position, size) of the analysis region as last set by the user
        self.busy = False
        self._job = None
        self._radial_cache = None
        self.stack = None  # PageStack while "Stack vertically" has been used for the open file
        self._page_reviews = OrderedDict()  # page -> result shown when the page was left, oldest first
        self._task_reviews = {}  # (page, task) -> result
        self._cd_levels = None  # CD color range while locked
        self._summary_text = ""
        self._syncing_table = False
        self._building = True
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._build_ui()
        self._apply_style()
        self._defaults = self._control_values()
        self._load_settings()
        self._building = False
        self._update_enabled()
        _install_exception_hook()
        _WINDOWS.add(self)
        if image is not None:
            self.set_image(image)

    def _build_ui(self):
        toolbar = self.addToolBar("Image and session")
        toolbar.setMovable(False)
        self.open_action = toolbar.addAction("Open image...", self.open_image)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open an image (Ctrl+O), or drop a file on the window")
        self.demo_action = toolbar.addAction("Load demo", self.load_demo)
        toolbar.addSeparator()
        fit = toolbar.addAction("Fit page", lambda: self.canvas.fit_image())
        fit.setShortcut("F")
        fit.setToolTip("Fit the active page (F); Shift+F fits the whole stack")
        fit_stack = QtGui.QAction("Fit stack", self, shortcut="Shift+F", triggered=self.fit_stack)
        self.addAction(fit_stack)
        zoom = toolbar.addAction("Zoom to hole", self.zoom_selected)
        zoom.setShortcut("Z")
        zoom.setToolTip("Zoom to the selected hole (Z)")
        self.roi_action = toolbar.addAction("Analysis region")
        self.roi_action.setCheckable(True)
        self.roi_action.toggled.connect(self._toggle_roi)
        toolbar.addSeparator()
        self.export_action = toolbar.addAction("Export results...", self.export_results)
        toolbar.addAction("Screenshot...", self.save_screenshot)
        self.reset_action = toolbar.addAction("Reset settings", self.reset_settings)
        self.reset_action.setToolTip("Every setting back to its default (the batch's defaults)")
        toolbar.addAction("Guide", self.show_guide)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(12, 8, 12, 6)
        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("subpx / Workbench")
        title.setObjectName("title")
        heading.addWidget(title)
        self.source_label = QtWidgets.QLabel("Open an image or try the demo")
        self.source_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        heading.addWidget(self.source_label, 1)
        self.state_label = QtWidgets.QLabel("Ready")
        self.state_label.setObjectName("state")
        heading.addWidget(self.state_label)
        layout.addLayout(heading)
        self.task = TaskTabs()
        self.task.setExpanding(False)
        for label, key in (("Find features", "detect"), ("Measure CD", "cd"),
                           ("Pitch + placement", "pitch"), ("Align images", "register")):
            self.task.setTabData(self.task.addTab(label), key)
        self.task.setCurrentIndex(self.task.findData("cd"))
        layout.addWidget(self.task)
        self.splitter = QtWidgets.QSplitter()
        layout.addWidget(self.splitter, 1)
        self.controls = self._build_controls()
        self.splitter.addWidget(self.controls)
        self.splitter.addWidget(self._build_workspace())
        self.splitter.setSizes([290, 1130])
        self.splitter.setStretchFactor(1, 1)
        self.task.currentChanged.connect(self._task_changed)
        self._task_changed()
        for keys in ("Ctrl+Return", "Ctrl+Enter"):  # main and keypad Enter
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, activated=self._run_current)
        self.hover_label = QtWidgets.QLabel("Wheel: zoom | Drag: pan | Click a hole: inspect | F: fit | Z: zoom to hole")
        self.hover_label.setObjectName("hint")
        self.hover_label.setMinimumWidth(460)
        # Permanent: messages (saved files, errors) appear on the left and are never overwritten by hovering.
        self.statusBar().addPermanentWidget(self.hover_label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)

    def _fit_detail_images(self):
        for canvas in (self.raw_canvas, self.input_canvas, *self.registration_canvases):
            canvas.fit_image()

    def _task_changed(self):
        operation = self.task.currentData()
        for key, page in self.task_widgets.items():
            page.setVisible(key == operation)
        self.feature_group.setVisible(operation in ("detect", "cd"))
        for key, row in self.feature_rows.items():
            self.feature_form.setRowVisible(row, operation in (("detect", "cd") if key == "polarity" else
                                                              ("cd",) if key in ("seeds", "filter") else ("detect",)))
        self.scale_group.setVisible(operation != "register")
        self._update_recipe_context()
        saved = self._task_reviews.get((self.loaded.page, operation)) if self.loaded is not None else None
        if saved is not None and saved is not self.review:
            self._accept_review(saved, restoring=True)
        target = self.pitch_tab if operation == "pitch" else self.alignment_tab if operation == "register" else self.table_tab
        self.diagnostics.setCurrentWidget(target)
        self.run_button = self.action_buttons[operation]
        self._render_image()
        self._update_enabled()

    def _run_current(self):
        self.run_button.click()

    def _combo(self, entries):
        widget = QtWidgets.QComboBox()
        widget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(6)
        for text, value in entries:
            widget.addItem(text, value)
        widget.currentIndexChanged.connect(self._settings_changed)
        return widget

    def _number(self, value, low, high, *, decimals=None, step=1):
        widget = QtWidgets.QSpinBox() if decimals is None else QtWidgets.QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.valueChanged.connect(self._settings_changed)
        return widget

    def _check(self, text, checked=False):
        widget = QtWidgets.QCheckBox(text)
        widget.setChecked(checked)
        widget.toggled.connect(self._settings_changed)
        return widget

    def _form_group(self, title, rows, layout):
        box = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(box)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setContentsMargins(10, 15, 10, 10)
        for label, widget in rows:
            form.addRow(label, widget)
        layout.addWidget(box)
        return box

    def _build_controls(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(345)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(6, 4, 8, 4)
        self.invert = self._check("Dark features on bright background")
        self.threshold = self._combo([("Otsu (automatic)", "otsu"), ("Triangle", "triangle")])
        self.area_min = self._number(3, 1, 100000000)
        self.area_max = self._number(200, 1, 100000000)
        limits = QtWidgets.QWidget()
        limits_layout = QtWidgets.QHBoxLayout(limits)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        limits_layout.addWidget(self.area_min)
        limits_layout.addWidget(QtWidgets.QLabel("to"))
        limits_layout.addWidget(self.area_max)
        self.pad = self._number(3, 1, 1000)
        self.refine = self._combo([(label, value) for label, value in
            (("Log-quadratic", "logquad"), ("Weighted center", "weighted"),
             ("Edge gradient", "edge_gradmoment"), ("Automatic", "auto"), ("None", "none"))])
        self.seeds = self._combo([("Separate spots", "threshold"), ("Touching / dense spots", "peaks")])
        self.seeds.setToolTip("Separate spots use threshold components; dense spots use local intensity peaks.")
        self.filter_button = QtWidgets.QPushButton()
        self.filter_button.clicked.connect(lambda: self.task.setCurrentIndex(self.task.findData("detect")))
        self.feature_group = self._form_group("Image features", [("", self.invert), ("Pattern", self.seeds),
            ("Feature filter", self.filter_button), ("Threshold", self.threshold), ("Area (px²)", limits),
            ("Refine center", self.refine), ("Padding (px)", self.pad)], layout)
        self.feature_form = self.feature_group.layout()
        self.feature_rows = dict(zip(("polarity", "seeds", "filter", "threshold", "area", "refine", "pad"), range(7)))
        self.threshold.currentIndexChanged.connect(self._update_feature_summary)
        self.area_min.valueChanged.connect(self._update_feature_summary)
        self.area_max.valueChanged.connect(self._update_feature_summary)
        self._update_feature_summary()

        self.recipe = self._combo([(label, key) for key, (label, methods) in RECIPES.items()] + [("Custom selection", "custom")])
        self.recipe_summary = self._hint("")
        self.methods_button = QtWidgets.QPushButton("Choose methods and parameters...")
        self.methods_button.clicked.connect(lambda: self.method_dialog.show())
        self.recipe_group = self._form_group("Measurement", [("", self.recipe), ("", self.recipe_summary),
                                                           ("", self.methods_button)], layout)
        self.pixel_size = self._number(1, 0.000001, 1e9, decimals=6, step=0.1)
        self.pixel_size.setToolTip("Length of one pixel, in the unit beside it")
        self.unit = QtWidgets.QLineEdit("px")
        self.unit.setMaximumWidth(65)
        self.unit.setToolTip("Length unit, for example nm")
        self.unit.textChanged.connect(self._settings_changed)
        scale = QtWidgets.QWidget()
        scale_layout = QtWidgets.QHBoxLayout(scale)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.addWidget(self.pixel_size, 1)
        scale_layout.addWidget(self.unit)
        self.backend = self._combo([("Auto (GPU when available)", "auto"), ("GPU", "gpu"), ("CPU", "cpu")])
        self.scale_group = self._form_group("Scale and compute", [("Pixel size", scale), ("Compute", self.backend)], layout)
        self.nominal_cd = self._number(0, 0, 1e9, decimals=4)
        self.psf_fwhm = self._number(0, 0, 1e9, decimals=4)
        for number in (self.nominal_cd, self.psf_fwhm):
            number.setSpecialValueText("Not specified")
        self.hole_group = self._form_group("Hole calibration (length unit)", [("Nominal CD", self.nominal_cd),
                                                                           ("PSF FWHM", self.psf_fwhm)], layout)
        self.hole_note = self._hint("Set a nominal CD or PSF FWHM before measuring calibrated diameters.")
        layout.addWidget(self.hole_note)
        self.task_pages = QtWidgets.QWidget()
        task_layout = QtWidgets.QVBoxLayout(self.task_pages)
        task_layout.setContentsMargins(0, 0, 0, 0)
        self.task_widgets = {}
        self.action_buttons = {}
        self.run_reasons = {}
        for operation, label in (("detect", "Find features"), ("cd", "Measure CD"),
                                 ("pitch", "Measure pitch"), ("register", "Align images")):
            page = QtWidgets.QWidget()
            inner = QtWidgets.QVBoxLayout(page)
            inner.setContentsMargins(0, 0, 0, 0)
            self.task_widgets[operation] = page
            task_layout.addWidget(page)
            button = QtWidgets.QPushButton(label)
            button.setObjectName("primary")
            button.setMinimumHeight(38)
            button.setToolTip("Run on the active page (Ctrl+Enter)")
            if operation == "register":
                button.clicked.connect(self.run_registration)
            else:
                button.clicked.connect(lambda checked=False, op=operation: self.run_analysis(op))
            self.action_buttons[operation] = button
            reason = QtWidgets.QLabel()
            reason.setObjectName("reason")
            reason.setWordWrap(True)
            reason.hide()
            self.run_reasons[operation] = reason
        self.detect_button, self.cd_button, self.pitch_button, self.register_button = (
            self.action_buttons[key] for key in ("detect", "cd", "pitch", "register"))
        self.task_widgets["detect"].layout().addWidget(self._hint("Check feature centers and the threshold mask before measuring. These settings also feed CD measurements."))
        self.task_widgets["cd"].layout().addWidget(self._hint("Compare definitions in the results table. Click a feature to see the actual pixels, contour and profile."))
        pitch = self.task_widgets["pitch"].layout()
        self.pitch_source = self._combo([("CD measurements", "cd"), ("Detected features", "detection")])
        self.pitch_source.currentIndexChanged.connect(self._update_recipe_context)
        self.pattern_layout = self._combo([("Automatic", "auto"), ("Square / rectangular", "square"), ("Staggered", "staggered")])
        self.stagger_axis = self._combo([("Offset rows", "row"), ("Offset columns", "column")])
        self.average_rows = self._number(0, 0, 1000000)
        self.average_columns = self._number(0, 0, 1000000)
        self.nominal_x = self._number(0, 0, 1e9, decimals=5)
        self.nominal_y = self._number(0, 0, 1e9, decimals=5)
        for number in (self.average_rows, self.average_columns):
            number.setSpecialValueText("All")
        for number in (self.nominal_x, self.nominal_y):
            number.setSpecialValueText("Measured mean")
        self.reference_selected = self._check("Reference at selected hole")
        self._form_group("Pitch and placement", [("Centers", self.pitch_source), ("Lattice", self.pattern_layout),
            ("Average rows", self.average_rows), ("Average columns", self.average_columns),
            ("Design pitch x", self.nominal_x), ("Design pitch y", self.nominal_y), ("", self.reference_selected)], pitch)
        self.pitch_hint_x = self._number(0, 0, 1e9, decimals=3)
        self.pitch_hint_y = self._number(0, 0, 1e9, decimals=3)
        self.index_dialog, index_layout = self._options_dialog("Pitch indexing")
        self._form_group("Indexing hints (0 = automatic)", [("Stagger direction", self.stagger_axis),
            ("Pitch hint x (px)", self.pitch_hint_x), ("Pitch hint y (px)", self.pitch_hint_y)], index_layout)
        index_button = QtWidgets.QPushButton("Indexing options...")
        index_button.clicked.connect(self.index_dialog.show)
        pitch.addWidget(index_button)
        align = self.task_widgets["register"].layout()
        moving_buttons = QtWidgets.QHBoxLayout()
        self.moving_button = QtWidgets.QPushButton("Open moving image...")
        self.moving_button.clicked.connect(self.open_moving)
        moving_buttons.addWidget(self.moving_button)
        self.moving_this_file = QtWidgets.QPushButton("Use this file")
        self.moving_this_file.setToolTip("Moving image = the page of the open file chosen in 'Moving page'")
        self.moving_this_file.clicked.connect(self._use_this_file)
        moving_buttons.addWidget(self.moving_this_file)
        align.addLayout(moving_buttons)
        self.moving_label = self._hint("Open a second image with the same dimensions as the active reference page.")
        align.addWidget(self.moving_label)
        self.moving_page_spin = self._number(0, 0, 0)
        self.moving_page_spin.setToolTip("Page of the moving file (0 = first)")
        self.moving_page_spin.setKeyboardTracking(False)
        self.moving_page_spin.valueChanged.connect(self._moving_page_changed)
        self.upsample = self._number(20, 1, 1000)
        self.hole_check_mode = self._combo([("Centered (batch default)", "centered"), ("Raw", "raw")])
        self.hole_check_mode.setToolTip("The batch's hole check: 'centered' removes each image's mean first "
                                        "(passes below 0.8), 'raw' does not (passes below 0.3)")
        self._form_group("Alignment", [("Moving page", self.moving_page_spin), ("Subpixel factor", self.upsample),
                                       ("Hole check", self.hole_check_mode)], align)
        self.registration_label = self._hint("Open the moving image, then Align images. Shifts are reported in [dy, dx] pixels.")
        align.addWidget(self.registration_label)
        for key, page in self.task_widgets.items():
            page.layout().addWidget(self.action_buttons[key])
            page.layout().addWidget(self.run_reasons[key])
            page.layout().addStretch()
        layout.addWidget(self.task_pages)
        layout.addStretch()
        self._build_method_options()
        self.seeds.currentIndexChanged.connect(self._update_recipe_context)
        self.recipe.currentIndexChanged.connect(self._recipe_changed)
        self._recipe_changed()
        return scroll

    def _options_dialog(self, title):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(550, 620)
        outer = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        inner = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(dialog.hide)
        outer.addWidget(close)
        return dialog, inner

    def _build_method_options(self):
        self.method_dialog, layout = self._options_dialog("Measurement methods and parameters")
        layout.addWidget(self._hint("Choose the definitions to compare. Only parameters used by those methods appear below. Changes apply on the next measurement."))
        self.method_list = QtWidgets.QListWidget()
        self.method_list.setMinimumHeight(210)
        self.method_list.setMaximumHeight(250)
        for name, label in METHOD_LABELS.items():
            item = QtWidgets.QListWidgetItem(method_name(name))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setToolTip(f"{name}: {label}. " + METHOD_HELP.get(name, AREA_HELP))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.method_list.addItem(item)
        layout.addWidget(self.method_list)
        self.background = self._combo([("Gaussian neighbour model", "model"), ("Cell border (legacy)", "border")])
        self.n_angles = self._number(96, 16, 720, step=16)
        self.radial_step = self._number(0.1, 0.01, 1, decimals=2, step=0.05)
        self.smooth_sigma = self._number(0, 0, 10, decimals=2, step=0.1)
        self.fit_fraction = self._number(0.2, 0.01, 0.95, decimals=2, step=0.05)
        self.half_level = self._number(0.5, 0.01, 0.99, decimals=2, step=0.05)
        self.peak_pixels = self._number(4, 1, 100, decimals=1)
        self.tail_level = self._number(0.2, 0.01, 0.99, decimals=2, step=0.05)
        self.tail_window = self._number(1.2, 0.5, 2, decimals=2, step=0.1)
        self.iterations = self._number(30, 1, 1000)
        self.tolerance = self._number(1e-3, 1e-6, 1, decimals=6, step=1e-4)
        self.tolerance.setToolTip("Deblending stops when the fit changes less than this per round "
                                  "(or at the iteration limit); the batch's --deblend-tolerance")
        self.dark_level = self._number(0, -1e9, 1e9, decimals=3)
        self.method_groups = []

        def group(title, rows, methods):
            self.method_groups.append((self._form_group(title, rows, layout), set(methods)))

        nonlocal_methods = set(METHOD_LABELS) - {name for name in METHOD_LABELS if name.startswith("local_")}
        group("Neighbour subtraction", [("Background", self.background), ("Fit iterations", self.iterations),
                                        ("Stop tolerance (px)", self.tolerance)], nonlocal_methods)
        group("Radial edge sampling", [("Rays", self.n_angles), ("Step (px)", self.radial_step),
              ("Smoothing (px)", self.smooth_sigma)], ("gradient", "halfmax"))
        group("Log-quadratic fit", [("Fit fraction", self.fit_fraction)], ("logquad",))
        group("Half-height area", [("Level", self.half_level)], [name for name in METHOD_LABELS if "half_area" in name])
        group("Peak averaging", [("Peak area (px²)", self.peak_pixels)],
              ("half_area_min_avg", "local_half_area_min_avg", "tail_area", "tail_area_norm", "local_tail_area"))
        group("Tail threshold", [("Level", self.tail_level)], ("tail_area", "tail_area_norm", "local_tail_area"))
        group("Wide measurement window", [("Window (pitch)", self.tail_window)],
              ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio"))
        group("Detector calibration", [("Dark level", self.dark_level)], ("flux_ratio",))
        self.method_list.itemChanged.connect(self._custom_methods_changed)
        layout.addStretch()

    def _checked_methods(self):
        return tuple(self.method_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                     for i in range(self.method_list.count())
                     if self.method_list.item(i).checkState() == QtCore.Qt.CheckState.Checked)

    def _recipe_changed(self):
        recipe = RECIPES.get(self.recipe.currentData())
        if recipe is not None:
            self.method_list.blockSignals(True)
            for i in range(self.method_list.count()):
                item = self.method_list.item(i)
                selected = item.data(QtCore.Qt.ItemDataRole.UserRole) in recipe[1]
                item.setCheckState(QtCore.Qt.CheckState.Checked if selected else QtCore.Qt.CheckState.Unchecked)
            self.method_list.blockSignals(False)
            self.background.setCurrentIndex(self.background.findData("model"))
        self._update_recipe_context()
        self._settings_changed()

    def _custom_methods_changed(self):
        self.recipe.blockSignals(True)
        self.recipe.setCurrentIndex(self.recipe.findData("custom"))
        self.recipe.blockSignals(False)
        self._update_recipe_context()
        self._settings_changed()

    def _update_recipe_context(self):
        if not hasattr(self, "method_list"):
            return
        names = self._checked_methods()
        selected = set(names)
        needs_model = bool(selected - {"logquad", "gradient", "halfmax"} -
                           {name for name in METHOD_LABELS if name.startswith("local_")}) or self.seeds.currentData() == "peaks"
        border = self.background.findData("border")
        self.background.model().item(border).setEnabled(not needs_model)
        if needs_model:
            self.background.setCurrentIndex(self.background.findData("model"))
        self.recipe_summary.setText(", ".join(method_name(name) for name in names) or "Select at least one measurement method.")
        for widget, methods in self.method_groups:
            widget.setVisible(bool(selected & methods))
        operation = self.task.currentData()
        use_cd = operation == "cd" or (operation == "pitch" and self.pitch_source.currentData() == "cd")
        self.recipe_group.setVisible(use_cd)
        calibrated = use_cd and bool(selected & CALIBRATED_METHODS)
        self.hole_group.setVisible(calibrated)
        self.hole_note.setVisible(calibrated)
        if not self._building:
            self._update_enabled()

    def _update_feature_summary(self):
        self.filter_button.setText(f"{self.area_min.value()} - {self.area_max.value()} px²  |  Adjust...")

    # ---------------------------------------------------------------------------------------------- settings
    def _persistent_controls(self):
        """Every setting that is saved between sessions and reset by 'Reset settings', by name."""
        controls = dict(
            invert=self.invert, threshold=self.threshold, area_min=self.area_min, area_max=self.area_max,
            pad=self.pad, refine=self.refine, seeds=self.seeds, recipe=self.recipe, background=self.background,
            n_angles=self.n_angles, radial_step=self.radial_step, smooth_sigma=self.smooth_sigma,
            fit_fraction=self.fit_fraction, half_level=self.half_level, peak_pixels=self.peak_pixels,
            tail_level=self.tail_level, tail_window=self.tail_window, iterations=self.iterations,
            tolerance=self.tolerance, dark_level=self.dark_level, pixel_size=self.pixel_size, unit=self.unit,
            backend=self.backend, nominal_cd=self.nominal_cd, psf_fwhm=self.psf_fwhm,
            pitch_source=self.pitch_source, pattern_layout=self.pattern_layout, stagger_axis=self.stagger_axis,
            average_rows=self.average_rows, average_columns=self.average_columns, nominal_x=self.nominal_x,
            nominal_y=self.nominal_y, reference_selected=self.reference_selected, pitch_hint_x=self.pitch_hint_x,
            pitch_hint_y=self.pitch_hint_y, upsample=self.upsample, hole_check_mode=self.hole_check_mode,
            failed_only=self.failed_only, lock_levels=self.lock_levels, lock_cd=self.lock_cd,
        )
        controls.update({f"layer:{name}": check for name, check in self.layer_checks.items()})
        return controls

    def _control_values(self):
        values = {}
        for name, widget in self._persistent_controls().items():
            if isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QtWidgets.QAbstractButton):
                values[name] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QLineEdit):
                values[name] = widget.text()
            else:
                values[name] = widget.value()
        values["methods"] = list(self._checked_methods())
        return values

    def _apply_values(self, values):
        building, self._building = self._building, True
        try:
            for name, widget in self._persistent_controls().items():
                if name not in values:
                    continue
                value = values[name]
                with QtCore.QSignalBlocker(widget):
                    if isinstance(widget, QtWidgets.QComboBox):
                        if widget.findData(value) >= 0:
                            widget.setCurrentIndex(widget.findData(value))
                    elif isinstance(widget, QtWidgets.QAbstractButton):
                        widget.setChecked(bool(value))
                    elif isinstance(widget, QtWidgets.QLineEdit):
                        widget.setText(str(value))
                    else:
                        widget.setValue(type(widget.value())(value))
            if "methods" in values:
                with QtCore.QSignalBlocker(self.method_list):
                    for i in range(self.method_list.count()):
                        item = self.method_list.item(i)
                        checked = item.data(QtCore.Qt.ItemDataRole.UserRole) in values["methods"]
                        item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            self._update_feature_summary()
            self._update_recipe_context()
        finally:
            self._building = building

    def _load_settings(self):
        settings = self._settings
        try:
            values = json.loads(settings.value(f"{SETTINGS_KEY}/controls", "{}") or "{}")
        except (TypeError, ValueError):
            values = {}
        self._apply_values(values)
        task = self.task.findData(settings.value(f"{SETTINGS_KEY}/task", "cd"))
        if task >= 0:
            self.task.setCurrentIndex(task)
        for key, restore in (("geometry", self.restoreGeometry), ("splitter", self.splitter.restoreState),
                             ("vertical_splitter", self.vertical_splitter.restoreState)):
            state = settings.value(f"{SETTINGS_KEY}/{key}")
            if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
                restore(state)

    def save_settings(self):
        settings = self._settings
        settings.setValue(f"{SETTINGS_KEY}/controls", json.dumps(self._control_values()))
        settings.setValue(f"{SETTINGS_KEY}/task", self.task.currentData())
        settings.setValue(f"{SETTINGS_KEY}/geometry", self.saveGeometry())
        settings.setValue(f"{SETTINGS_KEY}/splitter", self.splitter.saveState())
        settings.setValue(f"{SETTINGS_KEY}/vertical_splitter", self.vertical_splitter.saveState())
        settings.sync()

    def reset_settings(self):
        """Every setting back to its default; the image, results and window layout stay."""
        self._apply_values(self._defaults)
        self._settings_changed()
        self._render_image()
        self._update_table()
        self.statusBar().showMessage("Settings reset to their defaults (the batch's defaults).", 8000)

    def _last_folder(self):
        return str(self._settings.value(f"{SETTINGS_KEY}/last_folder", "") or "")

    def _remember_folder(self, path):
        self._settings.setValue(f"{SETTINGS_KEY}/last_folder", str(Path(path).parent))

    def _summary_row(self, text):
        label = QtWidgets.QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(False)
        label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _hint(text):
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("hint")
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _build_workspace(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        self.page_bar = QtWidgets.QWidget()
        pages = QtWidgets.QHBoxLayout(self.page_bar)
        pages.setContentsMargins(0, 0, 0, 4)
        self.previous_page = QtWidgets.QToolButton()
        self.previous_page.setText("Previous")
        self.previous_page.clicked.connect(lambda: self.page_spin.setValue(self.page_spin.value() - 1))
        self.previous_page.setToolTip("Previous page (Page Up on the image)")
        self.next_page = QtWidgets.QToolButton()
        self.next_page.setText("Next")
        self.next_page.clicked.connect(lambda: self.page_spin.setValue(self.page_spin.value() + 1))
        self.next_page.setToolTip("Next page (Page Down on the image)")
        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setRange(0, 0)
        self.page_spin.setKeyboardTracking(False)
        self.page_spin.setPrefix("Page ")
        self.page_spin.setToolTip("Page number, 0 = first (as in the batch's pages.csv and --pages)")
        self.page_spin.valueChanged.connect(self._page_changed)
        self.page_count = QtWidgets.QLabel("1 page")
        self.stack_pages = QtWidgets.QCheckBox("Stack vertically")
        self.stack_pages.setToolTip("Show every page; analysis uses only the active, outlined page")
        self.stack_pages.toggled.connect(self._stack_changed)
        for control in (self.previous_page, self.page_spin, self.page_count, self.next_page):
            pages.addWidget(control)
        pages.addStretch()
        pages.addWidget(self.stack_pages)
        layout.addWidget(self.page_bar)
        self.page_bar.hide()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("View"))
        self.stage = self._combo([(x, x) for x in ("Loaded image", "Threshold mask", "Component labels", "Voronoi cells", "Background model")])
        self.stage.currentIndexChanged.disconnect(self._settings_changed)
        self.stage.currentIndexChanged.connect(self._render_image)
        self.stage.setMinimumContentsLength(17)
        row.addWidget(self.stage)
        self.method_label = QtWidgets.QLabel("Method")
        row.addWidget(self.method_label)
        self.method = QtWidgets.QComboBox()
        self.method.setMinimumWidth(140)
        self.method.setMaximumWidth(280)
        self.method.currentIndexChanged.connect(self._method_changed)
        row.addWidget(self.method, 1)
        row.addStretch()
        layout.addLayout(row)
        layers = QtWidgets.QHBoxLayout()
        self.layer_checks = {}
        tips = {"Centers": "Shown while at most 2,000 holes are in view", "Contours": "Shown while at most 2,000 holes "
                "are in view", "Failed cells": "Cells of the holes the shown method failed on, in red",
                "CD colors": "Each cell colored by the shown method's CD (scale beside the image)"}
        for name, checked in (("Centers", True), ("Contours", True), ("Failed cells", True), ("Voronoi edges", False),
                              ("Rejected boxes", False), ("Pitch pairs", False), ("CD colors", False)):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(checked)
            check.setToolTip(tips.get(name, ""))
            check.toggled.connect(self._render_image)
            self.layer_checks[name] = check
            layers.addWidget(check)
        layers.addStretch()
        layout.addLayout(layers)
        # Two single-line rows of constant height: the image below never moves when the text changes.
        self.summary_label = self._summary_row("Load your image or choose Load demo to explore a synthetic staggered pattern.")
        self.failure_label = self._summary_row("")
        layout.addWidget(self.summary_label)
        layout.addWidget(self.failure_label)
        self.error_banner = QtWidgets.QFrame()
        self.error_banner.setObjectName("banner")
        banner = QtWidgets.QHBoxLayout(self.error_banner)
        banner.setContentsMargins(8, 4, 4, 4)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setObjectName("error")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        banner.addWidget(self.error_label, 1)
        details = QtWidgets.QToolButton()
        details.setText("Details")
        details.setToolTip("Show the full message in the Activity log")
        details.clicked.connect(lambda: self.diagnostics.setCurrentWidget(self.log))
        banner.addWidget(details)
        dismiss = QtWidgets.QToolButton()
        dismiss.setText("Dismiss")
        dismiss.clicked.connect(self._clear_error)
        banner.addWidget(dismiss)
        self.error_banner.hide()
        layout.addWidget(self.error_banner)
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(self.vertical_splitter, 1)
        self.canvas = ImageCanvas(histogram=True)
        self.canvas.clicked.connect(self._select_at)
        self.canvas.hovered.connect(self._hover_image)
        self.lock_levels = QtWidgets.QCheckBox("Lock")
        self.lock_levels.setToolTip("Keep this contrast on every page (otherwise each page is stretched to 1-99.5 %)")
        self.lock_levels.toggled.connect(self._lock_levels_changed)
        self.canvas.side_layout.addWidget(self.lock_levels)
        self.canvas.histogram.sigLevelsChanged.connect(self._levels_changed)
        self._locked_levels = None
        self.layers = HoleLayers(self.canvas.plot)
        self.layers.changed.connect(self._show_summary)
        # CD color scale beside the image: 1-99 % of the shown method, drag its ends to set, Lock to compare.
        self.cd_scale = QtWidgets.QWidget()
        self.cd_scale.setFixedWidth(92)
        scale_layout = QtWidgets.QVBoxLayout(self.cd_scale)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_view = pg.GraphicsLayoutWidget()
        scale_view.setBackground("#0c121b")
        self.cd_bar = pg.ColorBarItem(values=(0, 1), colorMap=pg.colormap.get("viridis"), width=14, rounding=1e-3)
        scale_view.addItem(self.cd_bar)
        self.cd_bar.setImageItem(self.layers.cd_map)
        self.cd_bar.sigLevelsChangeFinished.connect(self._cd_levels_changed)
        scale_layout.addWidget(scale_view, 1)
        self.lock_cd = QtWidgets.QCheckBox("Lock")
        self.lock_cd.setToolTip("Keep this CD color range for every method and page "
                                "(otherwise 1-99 % of the shown method's CDs)")
        self.lock_cd.toggled.connect(self._lock_cd_changed)
        scale_layout.addWidget(self.lock_cd)
        self.cd_scale.hide()
        self.canvas.layout().insertWidget(1, self.cd_scale)
        self.vertical_splitter.addWidget(self.canvas)
        for keys, button in (("PgUp", self.previous_page), ("PgDown", self.next_page)):
            for scope in (self.canvas, self.page_bar):  # never while typing in a table or field
                QtGui.QShortcut(QtGui.QKeySequence(keys), scope, activated=button.click,
                                context=QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#ffce73", width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.sigRegionChangeFinished.connect(self._roi_changed)
        self.canvas.plot.addItem(self.roi)
        self.roi.hide()
        self.diagnostics = QtWidgets.QTabWidget()
        self.expand_diagnostics = QtWidgets.QToolButton()
        self.expand_diagnostics.setText("Expand")
        self.expand_diagnostics.setCheckable(True)
        self.expand_diagnostics.setToolTip("Give the diagnostic plots more room")
        self.expand_diagnostics.toggled.connect(self._focus_diagnostics)
        self.diagnostics.setCornerWidget(self.expand_diagnostics)
        self.diagnostics.currentChanged.connect(lambda: QtCore.QTimer.singleShot(0, self._fit_detail_images))
        self.vertical_splitter.addWidget(self.diagnostics)
        self.vertical_splitter.setSizes([460, 320])
        self._build_table_tab()
        self._build_detail_tab()
        self._build_pitch_tab()
        self._build_registration_tab()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.diagnostics.addTab(self.log, "Activity")
        return widget

    def _focus_diagnostics(self, expanded):
        self.vertical_splitter.setSizes([100, 700] if expanded else [520, 330])
        self.expand_diagnostics.setText("Restore" if expanded else "Expand")

    def _build_detail_tab(self):
        tab = QtWidgets.QWidget()
        self.feature_tab = tab
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QtWidgets.QHBoxLayout()
        self.detail_label = self._hint("Click a detected feature to inspect its input and measurement.")
        controls.addWidget(self.detail_label, 1)
        self.projection_label = QtWidgets.QLabel("Projection")
        controls.addWidget(self.projection_label)
        self.projection_axis = QtWidgets.QComboBox()
        self.projection_axis.addItems(["x", "y"])
        self.projection_axis.currentIndexChanged.connect(self._draw_detail)
        controls.addWidget(self.projection_axis)
        self.ray_label = QtWidgets.QLabel("Ray")
        controls.addWidget(self.ray_label)
        self.ray = QtWidgets.QSpinBox()
        self.ray.setRange(0, 95)
        self.ray.valueChanged.connect(self._draw_detail)
        controls.addWidget(self.ray)
        for widget in (self.projection_label, self.projection_axis, self.ray_label, self.ray):
            widget.hide()
        layout.addLayout(controls)
        splitter = QtWidgets.QSplitter()
        self.inspector = self._build_inspector()
        splitter.addWidget(self.inspector)
        self.raw_canvas = ImageCanvas()
        self.input_canvas = ImageCanvas()
        self.profile_plot = pg.PlotWidget(background="#0c121b")
        self.profile_plot.addLegend(offset=(8, 8))
        self.profile_plot.showGrid(x=True, y=True, alpha=0.15)
        no_si_prefix(self.profile_plot.getPlotItem())
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.setMinimumWidth(100)
            splitter.addWidget(canvas)
        splitter.addWidget(self.profile_plot)
        splitter.setSizes([260, 180, 180, 350])
        layout.addWidget(splitter, 1)
        self.diagnostics.addTab(tab, "Selected feature")

    def _build_pitch_tab(self):
        tab = QtWidgets.QWidget()
        self.pitch_tab = tab
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.pitch_label = self._hint("Run Pitch + placement to inspect adjacent-pair pitch and its cumulative error.")
        layout.addWidget(self.pitch_label)
        graphics = pg.GraphicsLayoutWidget()
        graphics.setBackground("#0c121b")
        self.pitch_plots = {}
        for row, (key, title) in enumerate((("pitch", "pitch"), ("registration", "placement error"))):
            for column, axis in enumerate(("x", "y")):
                plot = graphics.addPlot(row=row, col=column, title=f"{axis.upper()} {title}")
                plot.showGrid(x=True, y=True, alpha=0.15)
                no_si_prefix(plot)
                self.pitch_plots[key, axis] = plot
        layout.addWidget(graphics, 1)
        self.diagnostics.addTab(tab, "Pitch + placement")

    def _build_registration_tab(self):
        tab = QtWidgets.QWidget()
        self.alignment_tab = tab
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.registration_view = QtWidgets.QComboBox()
        self.registration_view.addItems(["Before / after differences (same intensity scale)", "Reference / aligned moving image", "Reference / 50% aligned blend"])
        self.registration_view.currentIndexChanged.connect(self._draw_registration)
        layout.addWidget(self.registration_view)
        splitter = QtWidgets.QSplitter()
        self.registration_canvases = [ImageCanvas() for _ in range(3)]
        for canvas in self.registration_canvases:
            splitter.addWidget(canvas)
        layout.addWidget(splitter, 1)
        self.diagnostics.addTab(tab, "Align")

    def _build_table_tab(self):
        tab = QtWidgets.QWidget()
        self.table_tab = tab
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.review_notice = self._hint("")
        self.review_notice.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.review_notice.hide()
        layout.addWidget(self.review_notice)
        self.failed_only = QtWidgets.QCheckBox("Show failed measurements only")
        self.failed_only.toggled.connect(self._update_table)
        layout.addWidget(self.failed_only)
        self.table_model = MeasurementTable(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.table_model)
        header = self.table.horizontalHeader()
        header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)  # rows in ID order until a header click
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        # Compact columns the user can widen; fixed row height, so 40k rows cost nothing to lay out.
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setDefaultSectionSize(92)
        header.setMinimumSectionSize(36)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.clicked.connect(self._select_table)
        # Arrow keys, Page keys and clicks all move the current row; the selected hole follows it.
        self.table.selectionModel().currentRowChanged.connect(self._table_row_changed)
        layout.addWidget(self.table, 1)
        self.diagnostics.addTab(tab, "Results")

    def _build_inspector(self):
        widget = QtWidgets.QWidget()
        widget.setMinimumWidth(250)
        widget.setMaximumWidth(320)
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 0, 4, 0)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Hole"))
        self.feature_spin = QtWidgets.QSpinBox()
        self.feature_spin.setRange(0, 0)
        self.feature_spin.setKeyboardTracking(False)  # typing "150" selects once, not 1, 15 and 150
        self.feature_spin.valueChanged.connect(lambda value: self.select_feature(value, reveal=False, pan=True))
        controls.addWidget(self.feature_spin, 1)
        next_failed = QtWidgets.QPushButton("Next failed")
        next_failed.clicked.connect(self.next_failed)
        controls.addWidget(next_failed)
        layout.addLayout(controls)
        self.feature_label = self._hint("Click a feature or result row.")
        layout.addWidget(self.feature_label)
        self.comparison = QtWidgets.QTableWidget(0, 3)
        self.comparison.setHorizontalHeaderLabels(["Method", "CD", "Status"])
        self.comparison.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.comparison.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison.verticalHeader().hide()
        self.comparison.verticalHeader().setDefaultSectionSize(23)
        self.comparison.horizontalHeader().setFixedHeight(25)
        self.comparison.cellClicked.connect(
            lambda row, col: self.method.setCurrentIndex(self.method.findData(self.comparison.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole))))
        layout.addWidget(self.comparison, 1)
        buttons = QtWidgets.QHBoxLayout()
        definition = QtWidgets.QPushButton("Method definition")
        self.method_help = self._hint("")
        definition.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Method definition", self.method_help.text()))
        buttons.addWidget(definition)
        metrics_button = QtWidgets.QPushButton("Metrics and notes...")
        self.metrics_dialog, details = self._options_dialog("Selected feature metrics")
        self.metrics = QtWidgets.QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.verticalHeader().hide()
        self.metrics.setMinimumHeight(350)
        details.addWidget(self.metrics)
        self.notes_label = self._hint("No measurements yet.")
        details.addWidget(self.notes_label)
        metrics_button.clicked.connect(self.metrics_dialog.show)
        buttons.addWidget(metrics_button)
        layout.addLayout(buttons)
        return widget

    def _apply_style(self):
        up, down = _arrow_images()
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #151e2b; color: #dee7f2; font-family: 'Segoe UI'; font-size: 12px; }
            QToolBar { background: #1b2839; padding: 7px; spacing: 7px; border: 0; }
            QToolButton { padding: 6px 10px; border-radius: 4px; }
            QToolButton:hover, QToolButton:checked { background: #30465e; }
            QLabel#title { font-size: 20px; font-weight: 600; color: #f4f8ff; }
            QLabel#state { color: #56d6cf; padding: 6px 12px; background: #203c44; border-radius: 4px; }
            QLabel#state[level="warning"] { color: #ffce73; background: #3b3322; }
            QLabel#state[level="error"] { color: #ff9aa6; background: #42232b; }
            QLabel#section { color: #56d6cf; font-weight: 600; padding-bottom: 8px; }
            QLabel#hint { color: #a9bbce; line-height: 1.4; }
            QLabel#reason { color: #ffce73; }
            QLabel#error { color: #ffd3d8; }
            QFrame#banner { background: #42232b; border: 1px solid #8f3a46; border-radius: 4px; }
            QPushButton { background: #24384e; border: 1px solid #3c556e; border-radius: 4px; padding: 7px 9px; }
            QPushButton:hover { background: #31516b; border-color: #56d6cf; }
            QPushButton#primary { background: #246c72; border-color: #56d6cf; color: #ffffff; font-weight: 600; padding: 7px 24px; }
            QPushButton:disabled, QPushButton#primary:disabled { background: #1a2532; color: #5d6f82; border-color: #293a4c; }
            QCheckBox:disabled, QLabel:disabled, QToolButton:disabled { color: #5d6f82; }
            QGroupBox { border: 1px solid #304155; border-radius: 5px; margin-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; color: #90b2d2; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #0f1824; border: 1px solid #354a60; border-radius: 3px; padding: 4px; min-height: 20px; }
            QSpinBox, QDoubleSpinBox { padding-right: 18px; }
            QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] { border: 1px solid #ff748a; }
            QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled { color: #5d6f82; border-color: #293a4c; }
            QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; width: 16px; background: #1c2b3d; border-left: 1px solid #354a60; }
            QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 3px; }
            QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 3px; }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #31516b; }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(ARROW_UP); width: 10px; height: 8px; }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(ARROW_DOWN); width: 10px; height: 8px; }
            QComboBox QAbstractItemView { background: #1c2b3d; selection-background-color: #365571; }
            QTabWidget::pane { border: 1px solid #304155; }
            QTabBar::tab { background: #1b293a; padding: 8px 10px; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected { color: #56d6cf; border-bottom: 2px solid #56d6cf; background: #243447; }
            QTableView, QListWidget, QPlainTextEdit { background: #101925; alternate-background-color: #1b2a3b; border: 1px solid #304155; selection-background-color: #304f69; }
            QHeaderView::section { background: #26384c; color: #c3d3e5; border: 0; padding: 6px; }
            QScrollArea { border: 0; }
            QSplitter::handle { background: #28394c; }
            QStatusBar { color: #a9bbce; background: #101925; }
            QCheckBox { spacing: 5px; }
        """.replace("ARROW_UP", up).replace("ARROW_DOWN", down))

    def _settings_changed(self, *args):
        if self._building:
            return
        if (self.review is not None or self.registration is not None) and not self.busy:
            self._set_state("Settings changed | measure again to apply", "warning")
        self._update_enabled()

    def _set_state(self, text, level="info"):
        self.state_label.setText(text)
        self.state_label.setProperty("level", level)
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

    def _run_problems(self):
        """Why each task cannot run now ('' = it can), and the fields that cause it."""
        problems = dict.fromkeys(self.action_buttons, "")
        invalid = set()
        if self.loaded is None:
            return dict.fromkeys(self.action_buttons, "Open an image first."), invalid
        segmentation = ""
        if self.area_min.value() > self.area_max.value():
            segmentation = f"Area min ({self.area_min.value()}) is above area max ({self.area_max.value()})."
            invalid |= {self.area_min, self.area_max}
        selected = set(self._checked_methods())
        cd = segmentation
        if not cd and not selected:
            cd = "Tick at least one CD method (Choose methods and parameters)."
        elif not cd and selected & CALIBRATED_METHODS and not (self.nominal_cd.value() or self.psf_fwhm.value()):
            cd = ", ".join(method_name(m) for m in sorted(selected & CALIBRATED_METHODS)) + " needs a nominal CD or PSF FWHM."
            invalid |= {self.nominal_cd, self.psf_fwhm}
        pitch = cd if self.pitch_source.currentData() == "cd" else segmentation
        for (x, y), what in (((self.nominal_x, self.nominal_y), "design pitches"),
                             ((self.pitch_hint_x, self.pitch_hint_y), "pitch hints")):
            if (x.value() == 0) != (y.value() == 0):
                pitch = pitch or f"Set both {what} (x and y), or neither."
                invalid |= {x, y}
        register = ""
        if self.moving is None:
            register = "Open a moving image, or use a page of this file."
        elif self.moving.image.shape != self.loaded.image.shape:
            (mh, mw), (h, w) = self.moving.image.shape, self.loaded.image.shape
            register = f"The moving image is {mw} x {mh} px; this page is {w} x {h} px."
        problems.update(detect=segmentation, cd=cd, pitch=pitch, register=register)
        return problems, invalid

    def _update_enabled(self):
        available = self.loaded is not None and not self.busy
        problems, invalid = self._run_problems()
        for operation, button in self.action_buttons.items():
            button.setEnabled(available and not problems[operation])
            # The reason sits under the button; a busy job is not a reason.
            self.run_reasons[operation].setText(problems[operation])
            self.run_reasons[operation].setVisible(bool(problems[operation]) and not self.busy)
            button.setToolTip(problems[operation] or "Run on the active page (Ctrl+Enter)")
        for widget in (self.area_min, self.area_max, self.nominal_cd, self.psf_fwhm, self.nominal_x, self.nominal_y,
                       self.pitch_hint_x, self.pitch_hint_y):
            if widget.property("invalid") != (widget in invalid):
                widget.setProperty("invalid", widget in invalid)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        self.moving_this_file.setEnabled(available and bool(self.loaded.path))
        self.export_action.setEnabled(not self.busy and (self.review is not None or self.registration is not None))
        self.open_action.setEnabled(not self.busy)
        self.demo_action.setEnabled(not self.busy)
        self.roi_action.setEnabled(available)
        self.controls.setEnabled(not self.busy)
        self.task.setEnabled(not self.busy)
        self.invert.setEnabled(not self.busy)
        self.method_dialog.setEnabled(not self.busy)
        self.index_dialog.setEnabled(not self.busy)
        self.method.setEnabled(self.review is not None)
        analyzing = self.task.currentData() != "register"
        self.method.setVisible(analyzing)
        self.method_label.setVisible(analyzing)
        self.stage.setEnabled(analyzing and self.review is not None)
        self.page_bar.setEnabled(available)
        self.previous_page.setEnabled(available and self.loaded.page > 0)
        self.next_page.setEnabled(available and self.loaded.page < self.loaded.pages - 1)
        self._sync_page_controls()
        self._update_overlay_controls()

    def _sync_page_controls(self):
        if self.loaded is None:
            return
        self.page_spin.blockSignals(True)
        self.page_spin.setRange(0, self.loaded.pages - 1)
        self.page_spin.setValue(self.loaded.page)
        self.page_spin.blockSignals(False)
        self.page_count.setText(f"{self.loaded.pages} pages")
        self.page_bar.setVisible(self.loaded.pages > 1)

    def _run_job(self, message, function, on_result):
        if self.busy:
            return
        self.busy = True
        self.save_settings()
        self._set_state(message)
        self.log.appendPlainText(message)
        self.progress.setRange(0, 0)
        self.progress.show()
        self._update_enabled()
        self._job = _Job(function)
        context = message.rstrip(".") + " failed"

        def done(result):
            self.busy = False
            self.progress.hide()
            self._clear_error()
            try:
                on_result(result)
            except Exception:
                self._show_error(traceback.format_exc(), context=context)
            self._update_enabled()

        def failed(detail):
            self.busy = False
            self.progress.hide()
            self._show_error(detail, context=context)
            self._update_enabled()

        self._job.signals.finished.connect(done)
        self._job.signals.failed.connect(failed)
        self._pool.start(self._job)

    def _show_error(self, detail, *, context=""):
        """One plain line over the results (which stay visible); the full text goes to the Activity log."""
        if self.stack is None:
            self.stack_pages.blockSignals(True)
            self.stack_pages.setChecked(False)
            self.stack_pages.blockSignals(False)
        self.log.appendPlainText(detail)
        self.error_label.setText((f"{context}: " if context else "") + plain_error(detail))
        self.error_banner.show()
        self._set_state("Failed", "error")

    def _clear_error(self):
        self.error_banner.hide()
        self.error_label.clear()

    def open_image(self, checked=False, *, path=None, page=0):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open inspection image", self._last_folder(),
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy);;All files (*)")
        if path:
            self._remember_folder(path)
            self._run_job("Loading image", lambda: load_image(path, page), self._set_loaded)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths and not self.busy:
            event.acceptProposedAction()
            self.open_image(path=paths[0])

    def _page_changed(self, value):
        if self.loaded is not None and self.loaded.path and value != self.loaded.page and not self.busy:
            cached = self.stack.page(value) if self.stack is not None else None
            if cached is not None:
                self._set_loaded(cached, keep_pages=True)
                return
            # With the stack the file is open already; otherwise the page is read from the file.
            read = self.stack.reader.read if self.stack is not None else (lambda page, path=self.loaded.path:
                                                                         load_image(path, page))
            self._run_job("Loading page", lambda: read(value), lambda loaded: self._set_loaded(loaded, keep_pages=True))

    def _stacked(self):
        return self.stack is not None and self.stack_pages.isChecked()

    def _stack_changed(self, enabled):
        if self.loaded is None or self.busy:
            return
        if enabled and self.stack is None:
            path = self.loaded.path

            def accept(reader):
                self.stack = PageStack(self.canvas.plot, reader)
                self.stack.failed.connect(lambda text: self.log.appendPlainText(text))
                self.stack.remember(self.loaded)
                self.stack.set_levels(self.canvas.image_item.getLevels())
                self.stack.set_active(self.loaded.page)
                self._refresh_page_view(np.zeros(2))
                self._set_state("All pages visible | analysis uses the active page")

            self._run_job("Reading page sizes", lambda: PageReader(path), accept)
        elif self.stack is not None:
            # The toggle has already changed; recover the previous display offset.
            old_offset = np.zeros(2) if enabled else np.array([0, self.stack.offsets[self.loaded.page]])
            self.stack.set_visible(enabled)
            self._refresh_page_view(old_offset)

    def _display_offset(self):
        if self._stacked():
            return np.array([0, self.stack.offsets[self.loaded.page]])
        return np.zeros(2)

    def _refresh_page_view(self, old_offset):
        offset = self._display_offset()
        self.roi.blockSignals(True)
        self.roi.setPos(self.roi.pos() + QtCore.QPointF(*(offset - old_offset)))
        h, w = self.loaded.image.shape
        self.roi.maxBounds = QtCore.QRectF(-0.5, offset[1] - 0.5, w, h)
        self.roi.blockSignals(False)
        self._render_image()
        self._update_selection()
        self.fit_stack()  # the whole stack when it was turned on, the page when off

    def fit_stack(self):
        """Shift+F: the whole stack in view (the page when the stack is off)."""
        if self._stacked():
            self.canvas.plot.getViewBox().setRange(self.stack.bounds(), padding=0.02)
        else:
            self.canvas.fit_image()

    def _page_at(self, x, y):
        """(page number, page-local y) under a display position, or (None, y)."""
        if self.loaded is None:
            return None, y
        if self._stacked():
            page = self.stack.page_at(x, y)
            return (None, y) if page is None else (page, y - self.stack.offsets[page])
        h, w = self.loaded.image.shape
        return (self.loaded.page, y) if -0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5 else (None, y)

    def set_image(self, image, *, name="Array image"):
        """Load a finite grayscale/RGB NumPy array, resetting dependent results."""
        if self.busy:
            raise RuntimeError("Wait for the current job before replacing the image.")
        data, conversion = grayscale(image)
        self._set_loaded(LoadedImage(data, name, conversion=conversion))

    def _set_loaded(self, loaded, *, keep_pages=False):
        """Show a new image, or (keep_pages) another page of the open file.

        A page change keeps the zoom, the analysis region, the bottom tab, the image view, the selected position,
        the moving image and a locked contrast, so one area can be followed through the pages.
        """
        previous = self.loaded
        page_change = keep_pages and previous is not None
        old_offset = self._display_offset() if previous is not None else np.zeros(2)
        stacked = page_change and self._stacked()
        keep_view = page_change and (stacked or previous.image.shape == loaded.image.shape)
        if page_change:
            self._page_reviews.pop(previous.page, None)
            self._page_reviews[previous.page] = self.review  # the most recent page last
            if self.stack is not None:
                self.stack.remember(previous)  # drawn in the stack from memory, no decoding
        else:
            self._page_reviews.clear()
            self._task_reviews.clear()
            if self.stack is not None:
                self.stack.close()
                self.stack = None
            self.stack_pages.blockSignals(True)
            self.stack_pages.setChecked(False)
            self.stack_pages.blockSignals(False)
            self.moving = None
            self._selected_xy = None
        self.loaded = loaded
        if self.stack is not None:
            self.stack.set_active(loaded.page)
        self.review = self.registration = None
        self._radial_cache = None
        self.selected = None
        self.method.clear()
        if not page_change:
            self.stage.setCurrentIndex(0)
            self.diagnostics.setCurrentWidget(self.table_tab)
        self._sync_page_controls()
        h, w = loaded.image.shape
        page = f"  |  page {loaded.page} of {loaded.pages} (0 = first)" if loaded.pages > 1 else ""
        self.source_label.setText(f"{loaded.name}  |  {w:,} x {h:,} px  |  {loaded.image.dtype}{page}"
                                  + (f"  |  {loaded.conversion}" if loaded.conversion else ""))
        self.source_label.setToolTip(loaded.path or loaded.name)
        self._set_state("Image ready")
        self._set_summary("Measure the active page, then click a hole on the image or a row of the table.")
        self.failure_label.clear()
        self.feature_spin.setRange(0, 0)
        self._clear_detail("Click a hole after measuring.")
        self.table_model.clear()
        self.notes_label.setText("Review failed fits and method assumptions before accepting measurements.")
        self.review_notice.hide()
        for canvas in self.registration_canvases:
            canvas.image_item.clear()
            canvas.clear_overlays()
        for plot in self.pitch_plots.values():
            plot.clear()
        self.pitch_label.setText("Run Pitch + placement to calculate profiles.")
        self.registration_label.setText("No alignment calculated." + (
            " The moving image stays; align again for this page." if self.moving is not None else ""))
        if self.moving is None:
            self._sync_moving_controls()
        offset = self._display_offset()
        self.roi.blockSignals(True)
        if page_change:
            # The same region on the new page, moved inside it when the page is smaller.
            position, size = self._region or (self.roi.pos() - QtCore.QPointF(*old_offset), self.roi.size())
            size = QtCore.QPointF(min(size.x(), w), min(size.y(), h))
            x = min(max(position.x(), -0.5), w - 0.5 - size.x())
            y = min(max(position.y(), -0.5), h - 0.5 - size.y())
            self.roi.setSize((size.x(), size.y()))
            self.roi.setPos((x, y + offset[1]))
        else:
            self.roi_action.setChecked(False)
            self._region = None
            self.roi.setPos((w * 0.25 - 0.5, h * 0.25 - 0.5 + offset[1]))
            self.roi.setSize((w * 0.5, h * 0.5))
        self.roi.maxBounds = QtCore.QRectF(-0.5, offset[1] - 0.5, w, h)
        self.roi.blockSignals(False)
        self._render_image()
        if not keep_view:
            self.canvas.fit_image()
        self.log.appendPlainText(f"Loaded {loaded.name}, page {loaded.page}: shape={loaded.image.shape}, "
                                 f"dtype={loaded.image.dtype}. {loaded.conversion}")
        saved = self._page_reviews.get(loaded.page)
        if saved is not None:
            self._accept_review(saved, restoring=True)
        self._keep_results()
        self._update_enabled()

    def _keep_results(self):
        """Keep the results of the open page and of the most recent other pages, at most RESULT_PAGES pages and
        RESULT_BYTES of memory; older pages' results are dropped (measure the page again to see them)."""
        current = self.loaded.page if self.loaded is not None else None
        pages = [current] + [page for page in reversed(self._page_reviews) if page != current]
        pages += [page for page, _ in self._task_reviews if page not in pages]
        kept, total, dropped = set(), 0, []
        for page in pages:
            reviews = [review for (other, _), review in self._task_reviews.items() if other == page]
            size = array_bytes(self._page_reviews.get(page), *reviews)
            if page != current and not size:
                continue  # no result on this page
            if page == current or (len(kept) < RESULT_PAGES and total + size <= RESULT_BYTES):
                kept.add(page)
                total += size
            else:
                dropped.append(page)
        for page in [page for page in self._page_reviews if page not in kept]:
            del self._page_reviews[page]
        for key in [key for key in self._task_reviews if key[0] not in kept]:
            del self._task_reviews[key]
        if dropped:
            self.log.appendPlainText(
                f"Results of page {', '.join(map(str, sorted(dropped)))} dropped to limit memory (kept: "
                f"{len(kept)} page{'s' * (len(kept) > 1)}, {total / 1e9:.1f} GB; at most {RESULT_PAGES} pages and "
                f"{RESULT_BYTES / 1e9:.0f} GB).")

    def _clear_detail(self, message):
        self.feature_label.setText(message)
        self.metrics.setRowCount(0)
        self.comparison.setRowCount(0)
        self.profile_plot.clear()
        self.detail_label.setText("Click a hole to inspect its input and measurement.")
        self.method_help.setText("Select a method to see its definition and assumptions.")
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.image_item.clear()
            canvas.clear_overlays()

    def _lock_levels_changed(self, locked):
        self._locked_levels = tuple(self.canvas.image_item.getLevels()) if locked else None

    def _levels_changed(self):
        if self.stage.currentText() != "Loaded image":
            return
        if self.lock_levels.isChecked():
            self._locked_levels = tuple(self.canvas.histogram.getLevels())
        if self.stack is not None:
            self.stack.set_levels(self.canvas.histogram.getLevels())  # one contrast for every page

    def load_demo(self):
        def generate():
            return LoadedImage(demo_image(),
                               "Synthetic staggered contact holes (seed 42)")
        self._run_job("Generating demonstration image", generate, self._set_loaded)

    def _toggle_roi(self, enabled):
        self.roi.setVisible(enabled)
        self._settings_changed()

    def _roi_changed(self):
        # The region as the user set it (page-local), so a smaller page in between cannot shrink it.
        self._region = (self.roi.pos() - QtCore.QPointF(*self._display_offset()), self.roi.size())
        self._settings_changed()

    def _bounds(self):
        if not self.roi_action.isChecked():
            return None
        h, w = self.loaded.image.shape
        position, size = self.roi.pos(), self.roi.size()
        position = position - QtCore.QPointF(*self._display_offset())
        # Integer-centered pixels selected by the ROI's visual half-pixel edges.
        x0 = max(0, min(w, int(np.ceil(position.x()))))
        y0 = max(0, min(h, int(np.ceil(position.y()))))
        x1 = max(0, min(w, int(np.ceil(position.x() + size.x()))))
        y1 = max(0, min(h, int(np.ceil(position.y() + size.y()))))
        return [x0, y0, x1, y1]

    def parameters(self, *, validate_pitch=True):
        """Snapshot all controls so background jobs never access Qt widgets."""
        methods = tuple(self.method_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                        for i in range(self.method_list.count())
                        if self.method_list.item(i).checkState() == QtCore.Qt.CheckState.Checked)
        nominal = [self.nominal_x.value(), self.nominal_y.value()]
        hint = [self.pitch_hint_x.value(), self.pitch_hint_y.value()]
        if validate_pitch and (nominal[0] == 0) != (nominal[1] == 0):
            raise ValueError("Set both nominal x/y pitches, or leave both at zero for a measured-mean baseline.")
        if validate_pitch and (hint[0] == 0) != (hint[1] == 0):
            raise ValueError("Set both x/y pitch hints, or leave both at zero for automatic indexing.")
        reference = None
        if self.reference_selected.isChecked() and self.review is not None and self.selected is not None:
            # Convert via loaded-image coordinates if the analysis region changed.
            origin = np.array(self._bounds()[:2]) if self._bounds() is not None else np.zeros(2)
            reference = (self.review.seeds_xy[self.selected] + self.review.origin_xy - origin).tolist()
        return dict(
            threshold=self.threshold.currentData(), invert=self.invert.isChecked(),
            area_min=self.area_min.value(), area_max=self.area_max.value(), pad=self.pad.value(),
            refine=self.refine.currentData(), backend=self.backend.currentData(), methods=methods,
            background=self.background.currentData(), seeds=self.seeds.currentData(),
            pixel_size=self.pixel_size.value(), unit=self.unit.text().strip() or "px",
            n_angles=self.n_angles.value(), radial_step=self.radial_step.value(), smooth_sigma=self.smooth_sigma.value(),
            fit_fraction=self.fit_fraction.value(), half_area_level=self.half_level.value(),
            half_area_peak_pixels=self.peak_pixels.value(), tail_area_level=self.tail_level.value(),
            tail_window=self.tail_window.value(), deblend_iterations=self.iterations.value(),
            deblend_tolerance=self.tolerance.value(),
            nominal_cd=self.nominal_cd.value() or None, psf_fwhm=self.psf_fwhm.value() or None,
            dark_level=self.dark_level.value(), pitch_source=self.pitch_source.currentData(),
            pitch_options=dict(layout=self.pattern_layout.currentData(), stagger_axis=self.stagger_axis.currentData(),
                               average_rows=self.average_rows.value() or None, average_columns=self.average_columns.value() or None,
                               reference_xy=reference, nominal_pitch_xy=nominal if nominal[0] else None,
                               pitch_hint_xy=hint if hint[0] else None),
        )

    def run_analysis(self, operation):
        if self.loaded is None or self.busy:
            return
        problem = self._run_problems()[0][operation]
        if problem:
            self._show_error(problem)
            return
        try:
            parameters, bounds = self.parameters(validate_pitch=operation == "pitch"), self._bounds()
        except ValueError as exc:
            self._show_error(str(exc))
            return
        image = self.loaded.image
        previous = self.review
        source_operation = "detect" if operation == "pitch" and parameters["pitch_source"] == "detection" else operation
        # Reuse the displayed measurement only when all contributing settings
        # and the analysis region match. Changing pitch bands does not refit CD.
        matching = previous is not None and previous.parameters.get("bounds") == bounds
        if matching:
            relevant = COMMON_SETTINGS + (CD_SETTINGS if parameters["pitch_source"] == "cd" else ("refine",))
            matching = all(previous.parameters.get(key) == parameters[key] for key in relevant)
            matching &= (previous.cd is not None) if parameters["pitch_source"] == "cd" else (previous.detection is not None)

        def calculate():
            if operation == "pitch" and matching:
                patterns = analyze_patterns(previous, parameters["pitch_options"])
                return replace(previous, patterns=patterns, parameters=dict(parameters, bounds=bounds, operation="pitch"))
            result = analyze_image(image, parameters, operation=source_operation, bounds=bounds)
            if operation == "pitch" and source_operation == "detect":
                try:
                    result.patterns = analyze_patterns(result, parameters["pitch_options"])
                except ValueError as exc:
                    result.warnings.append(f"Pitch indexing failed: {exc}")
                result.parameters["operation"] = "pitch"
            return result

        labels = {"detect": "Finding holes", "cd": "Measuring CD", "pitch": "Calculating pitch and placement"}
        self._run_job(labels[operation], calculate, self._accept_review)

    def _accept_review(self, review, *, restoring=False):
        self.review = review
        self._task_reviews[self.loaded.page, review.parameters.get("operation", "cd")] = review
        if not restoring:
            self._keep_results()
        self._radial_cache = None
        old = self.method.currentData()
        self.method.blockSignals(True)
        self.method.clear()
        for name in review.measurements:
            self.method.addItem(method_name(name), name)
            self.method.setItemData(self.method.count() - 1, f"{name}: {METHOD_LABELS.get(name, name)}",
                                    QtCore.Qt.ItemDataRole.ToolTipRole)
        self.method.setCurrentIndex(max(0, self.method.findData(old)))
        self.method.blockSignals(False)
        self.feature_spin.blockSignals(True)
        self.feature_spin.setRange(0, max(0, len(review.seeds_xy) - 1))
        self.feature_spin.blockSignals(False)
        # No hole is selected until a click; a hole selected before stays selected at its position.
        index = self._hole_near(review, self._selected_xy)
        self.selected = index if index >= 0 else None
        warnings = review.warnings
        self._set_state(f"{len(review.seeds_xy):,} holes | {review.seconds:.2f} s"
                        + (f" | {len(warnings)} warning{'s' * (len(warnings) > 1)}" if warnings else ""),
                        "warning" if warnings else "info")
        self.notes_label.setText("\n\n".join([*(f"Warning: {text}" for text in warnings), *review.notes])
                                 or "No method failures reported. Inspect image support and fit assumptions "
                                    "before accepting a measurement.")
        lines = ([f"<span style='color:#ffce73'>&#9888; {html_escape(text)}</span>" for text in warnings]
                 + [f"<span style='color:#a9bbce'>{html_escape(text)}</span>" for text in review.notes])
        self.review_notice.setText("<br>".join(lines))
        self.review_notice.setVisible(bool(lines))
        if not restoring:
            self.log.appendPlainText(f"Analysis complete in {review.seconds:.3f} s. Region origin [x,y]={review.origin_xy.tolist()}.")
            if review.cd is not None:
                self.log.appendPlainText(f"Background: {review.cd.meta.get('background_definition')}")
                seconds = review.cd.meta.get("seconds", {})
                self.log.appendPlainText("Stage timings: " + ", ".join(
                    f"{stage} {value:.2f} s" for stage, value in seconds.items() if isinstance(value, (int, float))))
            for text in warnings:
                self.log.appendPlainText("Warning: " + text)
            for text in review.notes:
                self.log.appendPlainText(text)
        self._method_changed()
        if not len(review.seeds_xy):
            self._clear_detail("No holes found. Review segmentation, area limits, polarity and region.")
            self.detail_label.setText("No hole input is available for this calculation.")
        if not restoring:
            if review.patterns:
                self.diagnostics.setCurrentWidget(self.pitch_tab)
                self.layer_checks["Pitch pairs"].setChecked(True)
            elif self.diagnostics.currentWidget() in (self.log, self.alignment_tab):
                self.diagnostics.setCurrentWidget(self.table_tab)
        self._update_overlay_controls()
        current = self.parameters(validate_pitch=False)
        relevant = COMMON_SETTINGS + (("refine",) if review.detection is not None else CD_SETTINGS)
        if review.patterns:
            relevant += ("pitch_source", "pitch_options")
        if review.parameters.get("bounds") != self._bounds() or any(review.parameters.get(key) != current[key] for key in relevant):
            self._settings_changed()

    def _method_changed(self, *args):
        self._radial_cache = None
        for widget in (self.ray, self.ray_label):
            widget.setVisible(self.method.currentData() in ("gradient", "halfmax"))
        for widget in (self.projection_axis, self.projection_label):
            widget.setVisible(self.method.currentData() in ("projection", "local_projection"))
        self._render_image()
        self._update_table()
        self._draw_pitch()
        if self.selected is not None:
            self.select_feature(self.selected, reveal=False)
        elif self.review is not None and len(self.review.seeds_xy):
            self._clear_detail("Click a hole on the image or a row of the table.")

    def _update_overlay_controls(self):
        has_review = self.review is not None
        if self.task.currentData() == "register":
            has_review = False
        has_cd = has_review and self.review.cd is not None
        for name, check in self.layer_checks.items():
            relevant = has_review
            if name in ("Contours", "Voronoi edges", "CD colors"):
                relevant = has_cd
            elif name == "Rejected boxes":
                relevant = has_review and self.review.detection is not None
            elif name == "Pitch pairs":
                relevant = has_review and bool(self.review.patterns)
            check.setEnabled(relevant)
            check.setVisible(self.task.currentData() != "register" and (relevant or name in ("Centers", "Contours", "Voronoi edges")))

    def _render_image(self, *args):
        """The image (or pipeline stage) and the result's layers; cached maps make this cheap to call often."""
        if self.loaded is None:
            return
        canvas = self.canvas
        review = self.review if self.task.currentData() != "register" else None
        data, origin, categorical = self.loaded.image, np.zeros(2), False
        stage = self.stage.currentText()
        message = ""
        if stage != "Loaded image":
            if review is None:
                message = "Run an analysis to calculate this view. "
            elif stage == "Threshold mask":
                data = review.cd.binary if review.cd is not None else review.detection.binary
                origin = review.origin_xy
                if review.cd is not None and review.cd.meta.get("seeds") == "peaks":
                    message = "Peak seeding does not use a threshold mask. "
            elif stage == "Component labels":
                data = review.cd.component_labels if review.cd is not None else review.detection.labels
                origin, categorical = review.origin_xy, True
            elif review.cd is None:
                message = "Measure CD to calculate Voronoi cells and background. "
            elif stage == "Voronoi cells":
                data, origin, categorical = review.cd.cell_labels, review.origin_xy, True
            elif review.cd.background_map is not None:
                data, origin = review.cd.background_map, review.origin_xy
            else:
                message = "This background mode has no global background map. "
        offset = self._display_offset()
        display_origin = review.origin_xy + offset if review is not None else offset
        locked = self.lock_levels.isChecked() and self._locked_levels is not None and stage == "Loaded image"
        canvas.set_image(data, origin + offset, fit=False, categorical=categorical, keep_levels=locked)
        if locked:
            canvas.histogram.setLevels(*self._locked_levels)
        if self.stack is not None and stage == "Loaded image":
            self.stack.set_levels(canvas.image_item.getLevels())
        self.roi.setZValue(50)
        name = self.method.currentData()
        shown = review is not None and len(review.seeds_xy) and name in review.measurements
        layers = {key for key, check in self.layer_checks.items() if check.isChecked()}
        self.layers.show(review if shown else None, name, display_origin, layers)
        self._update_selection()
        self._update_cd_scale(review if shown and "CD colors" in layers else None, name)
        if not shown:
            self.failure_label.clear()
            if self.task.currentData() == "register":
                self._summary_text = "Reference: active page. Open a matching moving image, then Align images."
            elif review is not None and not len(review.seeds_xy):
                self._summary_text = "No holes found. Inspect the threshold mask, area limits, polarity and region."
            elif message:
                self._summary_text = message
            self._show_summary()
            return
        measured = review.measurements[name]
        valid = measured.valid & np.isfinite(measured.centers_xy).all(axis=1)
        page = f"Page {self.loaded.page} | " if self.loaded.pages > 1 else ""
        summary = f"{page}{len(valid):,} holes | {method_name(name)}: {valid.sum():,} measured"
        values = measured.metrics.get("cd_equivalent_px")
        if values is not None and valid.any():
            summary += f", mean CD {np.mean(values[valid]) * review.parameters['pixel_size']:.4g} {review.parameters['unit']}"
        self._summary_text = message + summary
        self._show_summary()
        self.failure_label.setText(self._failure_summary(review))

    def _set_summary(self, text):
        self._summary_text = text
        self._show_summary()

    def _show_summary(self):
        """The summary line, with a hint when centers and contours wait for a closer zoom."""
        hint = ""
        if self.layers.hidden_vectors():
            hint = (f" | {self.layers.in_view:,} holes in view: zoom in to see centers and contours "
                    f"(at most {self.layers.VECTOR_LIMIT:,})")
        self.summary_label.setText(self._summary_text + hint)

    def _update_cd_scale(self, review, method):
        """CD color scale: 1-99 % of the shown method's CDs, or the locked range."""
        values = review.measurements[method].metrics.get("cd_equivalent_px") if review is not None else None
        good = review.measurements[method].valid & np.isfinite(values) if values is not None else None
        if values is None or not good.any():
            self.cd_scale.hide()
            return
        unit = review.parameters["unit"]
        if self.lock_cd.isChecked() and self._cd_levels is not None:
            low, high = self._cd_levels
        else:
            low, high = np.percentile(values[good] * review.parameters["pixel_size"], (1, 99))
            if high <= low:
                low, high = low - 0.5, high + 0.5
            if self.lock_cd.isChecked():
                self._cd_levels = (float(low), float(high))
        self.cd_bar.setLevels((float(low), float(high)))
        self.cd_bar.axis.setLabel(f"CD ({unit})")
        self.cd_scale.show()

    def _cd_levels_changed(self):
        if self.lock_cd.isChecked():
            self._cd_levels = tuple(float(value) for value in self.cd_bar.levels())

    def _lock_cd_changed(self, locked):
        self._cd_levels = tuple(float(value) for value in self.cd_bar.levels()) if locked and self.cd_scale.isVisible() else None
        if not locked:
            self._render_image()

    def _update_selection(self):
        """Move the selection marker only; no other layer is redrawn."""
        review = self.review if self.task.currentData() != "register" else None
        if review is None or self.selected is None or not 0 <= self.selected < len(review.seeds_xy):
            self.layers.set_selection(None)
        else:
            self.layers.set_selection(review.seeds_xy[self.selected] + review.origin_xy + self._display_offset())

    @staticmethod
    def _failure_summary(review):
        """Failed holes of every method, largest first; always on screen (the shown method is only one of them)."""
        if review.cd is None:
            return "<span style='color:#a9bbce'>Detection only: measure CD to see failures per method.</span>"
        total = max(len(review.seeds_xy), 1)
        failed = sorted(((int((~measured.valid).sum()), name) for name, measured in review.measurements.items()),
                        reverse=True)
        parts = [f"{html_escape(method_name(name))} {count:,} ({100 * count / total:.0f} %)"
                 for count, name in failed if count]
        if not parts:
            return "<span style='color:#9be28a'>No method failed on any hole.</span>"
        return "<span style='color:#ffce73'>Failed: " + " &middot; ".join(parts) + "</span>"

    def _select_at(self, x, y):
        if self.loaded is None:
            return
        page, local_y = self._page_at(x, y)
        if page is None:
            return
        if page != self.loaded.page:
            if not self.busy:
                self.page_spin.setValue(page)
            return
        index = self._hole_near(self.review, (x, local_y))
        if index >= 0:
            self.select_feature(index)

    @staticmethod
    def _hole_near(review, xy):
        """The hole whose cell holds the loaded-page point xy, else the nearest within 3/4 of a pitch; -1: none."""
        if review is None or xy is None or not len(review.seeds_xy):
            return -1
        local = np.asarray(xy, float) - review.origin_xy
        h, w = review.image.shape
        if review.cd is not None and -0.5 <= local[0] < w - 0.5 and -0.5 <= local[1] < h - 0.5:
            index = int(review.cd.cell_labels[int(np.floor(local[1] + .5)), int(np.floor(local[0] + .5))])
            if index >= 0:
                return index
        distance = np.hypot(*(review.seeds_xy - local).T)
        nearest = int(np.argmin(distance))
        spacing = np.sqrt(review.image.size / len(review.seeds_xy))
        return nearest if distance[nearest] <= 0.75 * spacing else -1

    def _hover_image(self, x, y):
        if self.loaded is None:
            return
        page, local_y = self._page_at(x, y)
        if page is not None:
            ix, iy = int(np.floor(x + .5)), int(np.floor(local_y + .5))
            prefix = f"Page {page} | " if self.loaded.pages > 1 else ""
            # Other pages of the stack: the value when the page is in memory at full resolution.
            loaded = self.loaded if page == self.loaded.page else self.stack.page(page)
            value = f" = {loaded.image[iy, ix]:.6g}" if loaded is not None else ""
            self.hover_label.setText(f"{prefix}x={x:.2f}, y={local_y:.2f} px | pixel [{iy}, {ix}]{value}")

    def select_feature(self, index, *, reveal=True, pan=False):
        """Select hole ``index``. ``reveal`` opens the hole tab (image clicks); ``pan`` brings it into view."""
        review = self.review
        if index is None or review is None or not len(review.seeds_xy) or self.method.currentData() not in review.measurements:
            return
        self.selected = max(0, min(int(index), len(review.seeds_xy) - 1))
        self._selected_xy = review.seeds_xy[self.selected] + review.origin_xy
        if reveal:
            self.diagnostics.setCurrentWidget(self.feature_tab)
        self._sync_table_selection()
        self._update_selection()
        self.feature_spin.blockSignals(True)
        self.feature_spin.setValue(self.selected)
        self.feature_spin.blockSignals(False)
        self._radial_cache = None
        method = self.method.currentData()
        measured = review.measurements[method]
        point = measured.centers_xy[self.selected] + review.origin_xy
        seed = review.seeds_xy[self.selected] + review.origin_xy
        self.feature_label.setText(f"Hole {self.selected} | {measured.status[self.selected]}\n"
                                  f"Seed x={seed[0]:.4f}, y={seed[1]:.4f} px\n"
                                  f"{method_name(method)} x={point[0]:.4f}, y={point[1]:.4f} px")
        self.method_help.setText(f"{method_name(method)} ({method}). " + METHOD_HELP.get(
            method, METHOD_LABELS.get(method, "Subpixel center detection") + ". "
            + (AREA_HELP if method != "detection" else "Review the threshold mask and area limits before measuring CD.")))
        self.comparison.setRowCount(len(review.measurements))
        self.comparison.setVisible(review.cd is not None)
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        for row, (name, measurement) in enumerate(review.measurements.items()):
            cd = measurement.metrics.get("cd_equivalent_px")
            values = [method_name(name), f"{cd[self.selected] * scale:.5g} {unit}" if cd is not None and np.isfinite(cd[self.selected]) else "--",
                      str(measurement.status[self.selected])]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
                item.setToolTip(f"{name}: {METHOD_LABELS.get(name, name)}")
                item.setForeground(QtGui.QColor(COLORS[row % len(COLORS)] if measurement.valid[self.selected] else "#ff748a"))
                self.comparison.setItem(row, column, item)
        rows = list(measured.metrics.items())
        if review.cd is not None:
            rows = [("cell_area_px2", review.cd.cell_areas_px2),
                    ("mean_original_intensity", review.cd.mean_intensity)] + rows
        self.metrics.setRowCount(len(rows))
        for row, (name, values) in enumerate(rows):
            value = values[self.selected]
            label = name
            if name.endswith("_px2"):
                value, label = value * scale**2, name[:-4] + f" ({unit}²)"
            elif name.endswith("_px"):
                value, label = value * scale, name[:-3] + f" ({unit})"
            self.metrics.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            self.metrics.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{value:.6g}" if np.isfinite(value) else "--"))
        self._draw_detail()
        if pan:
            self._reveal(seed + self._display_offset())
        if self.reference_selected.isChecked() and method in review.patterns and not self.busy:
            if review.patterns[method].reference_index != self.selected:
                self._set_state("Reference changed | rerun Pitch + placement", "warning")

    def _reveal(self, point):
        """Pan, keeping the zoom, when the point lies outside the view."""
        view = self.canvas.plot.getViewBox()
        (x0, x1), (y0, y1) = view.viewRange()
        if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
            dx, dy = (x1 - x0) / 2, (y1 - y0) / 2
            view.setRange(xRange=(point[0] - dx, point[0] + dx), yRange=(point[1] - dy, point[1] + dy), padding=0)

    def next_failed(self):
        if self.review is None or self.method.currentData() not in self.review.measurements:
            return
        bad = np.flatnonzero(~self.review.measurements[self.method.currentData()].valid)
        if len(bad):
            later = bad[bad > (-1 if self.selected is None else self.selected)]
            self.select_feature(later[0] if len(later) else bad[0], reveal=False)
            self.zoom_selected()
        else:
            self.statusBar().showMessage(f"No failed holes for {method_name(self.method.currentData())}.", 5000)

    def zoom_selected(self):
        if self.review is None or self.selected is None or not len(self.review.seeds_xy):
            return
        point = self.review.seeds_xy[self.selected] + self.review.origin_xy + self._display_offset()
        radius = 15
        if self.review.cd is not None:
            radius = max(8, max(self.review.cd.masks.shape[1:]) * 1.5)
        self.canvas.plot.setRange(xRange=(point[0] - radius, point[0] + radius),
                                  yRange=(point[1] - radius, point[1] + radius), padding=0)

    def _update_table(self, *args):
        """A new result rebuilds the table; another method or filter only reorders it (the sort stays)."""
        review, method = self.review, self.method.currentData()
        if review is None or method not in review.measurements:
            return
        self._syncing_table = True
        try:
            if self.table_model.review is review:
                self.table_model.set_method(method, self.failed_only.isChecked())
            else:
                # The sort of the last table carries over to the new result.
                header = self.table.horizontalHeader()
                self.table_model.sort_column = max(0, header.sortIndicatorSection())
                self.table_model.sort_order = header.sortIndicatorOrder()
                self.table_model.set_review(review, method, self.failed_only.isChecked())
                # Compact: each column as wide as its name (bold when shown), never squeezed below it.
                font = QtGui.QFont(header.font())
                font.setBold(True)
                metrics = QtGui.QFontMetrics(font)
                for column in range(self.table_model.columnCount()):
                    text = self.table_model.headerData(column, QtCore.Qt.Orientation.Horizontal)
                    width = max(metrics.horizontalAdvance(line) for line in text.splitlines()) + 24
                    header.resizeSection(column, int(np.clip(width, 140 if column == 3 else 56, 280)))
                if self.table_model.sort_column >= self.table_model.columnCount():
                    self.table_model.sort_column = 0
                    header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
        finally:
            self._syncing_table = False
        self._sync_table_selection()

    def _sync_table_selection(self):
        """The table's current row follows the selected hole (no row when the filter hides it)."""
        row = self.table_model.row_of_hole(self.selected) if self.selected is not None else -1
        self._syncing_table = True
        try:
            if row < 0:
                self.table.clearSelection()
            elif self.table.currentIndex().row() != row:
                self.table.selectRow(row)
                self.table.scrollTo(self.table_model.index(row, 0))
        finally:
            self._syncing_table = False

    def _table_row_changed(self, current, previous):
        # Clicks, arrow keys and Page keys in the table select the hole of the current row; the table stays.
        if self._syncing_table or not current.isValid():
            return
        feature = int(self.table_model.data(current, QtCore.Qt.ItemDataRole.UserRole))
        if feature != self.selected:
            self.select_feature(feature, reveal=False, pan=True)

    def _select_table(self, index):
        # A click on another method's CD column shows that method too.
        method = self.table_model.method_of_column(index.column())
        if method is not None and method != self.method.currentData():
            self.selected = int(self.table_model.data(index, QtCore.Qt.ItemDataRole.UserRole))
            self.method.setCurrentIndex(self.method.findData(method))

    def _draw_detail(self, *args):
        review = self.review
        if review is None or self.selected is None or not len(review.seeds_xy):
            return
        index, method = self.selected, self.method.currentData()
        if method not in review.measurements:
            return
        self.raw_canvas.clear_overlays()
        self.input_canvas.clear_overlays()
        self.profile_plot.clear()
        self.profile_plot.enableAutoRange()
        self.profile_plot.setLabel("bottom", "x", units="px")
        self.profile_plot.setLabel("left", "Intensity")
        if review.cd is None:
            x, y = np.rint(review.seeds_xy[index]).astype(int)
            radius = max(8, review.parameters["pad"] * 2)
            x0, x1 = max(0, x - radius), min(review.image.shape[1], x + radius + 1)
            y0, y1 = max(0, y - radius), min(review.image.shape[0], y + radius + 1)
            origin = np.array([x0, y0]) + review.origin_xy
            self.raw_canvas.set_image(review.image[y0:y1, x0:x1], origin)
            self.input_canvas.set_image(review.detection.binary[y0:y1, x0:x1], origin)
            self.raw_canvas.plot.setTitle("Original pixels")
            self.input_canvas.plot.setTitle("Threshold mask")
            self.detail_label.setText("Detection: threshold > components > area filter > center refinement. CD provides cell diagnostics.")
            self.profile_plot.setTitle("Original center-row intensity")
            self.profile_plot.plot(np.arange(x0, x1) + review.origin_xy[0], review.image[y, x0:x1], pen=COLORS[0], symbol="o", symbolSize=3)
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
        self.raw_canvas.plot.setTitle("Original pixels")
        self.input_canvas.plot.setTitle("Method input")
        self.detail_label.setText(f"#{index} | {description}")
        outline = np.zeros((*mask.shape, 4), np.uint8)
        labels = result.cell_labels[np.clip(yy + oy, 0, result.image.shape[0] - 1),
                                    np.clip(xx + ox, 0, result.image.shape[1] - 1)]
        outline[cell_boundary((in_image & (labels == index)).astype(int))] = (86, 214, 207, 180)
        self.raw_canvas.raster("mask", outline, origin)
        for i, (name, measurement) in enumerate(result.measurements.items()):
            if measurement.contours_xy.size:
                contour = measurement.contours_xy[index] + review.origin_xy
                self.input_canvas.add_overlay(name, pg.PlotDataItem(contour[:, 0], contour[:, 1], connect="finite",
                    pen=pg.mkPen(COLORS[i % len(COLORS)], width=2.5 if name == method else 1)))
        measured = result.measurements[method]
        center = measured.centers_xy[index]
        if not np.isfinite(center).all():
            center = result.seeds_xy[index]
        self.input_canvas.add_overlay("center", pg.ScatterPlotItem(pos=[center + review.origin_xy], symbol="+", size=12, pen="w"))
        if method in ("gradient", "halfmax"):
            if self._radial_cache is None:
                self._radial_cache = radial_diagnostic(result, method, index)
            diagnostic = self._radial_cache
            if diagnostic is None:
                self.profile_plot.setTitle("Insufficient signal or cell support for radial samples")
                return
            self.ray.blockSignals(True)
            self.ray.setMaximum(len(diagnostic["angles"]) - 1)
            self.ray.blockSignals(False)
            ray = self.ray.value()
            radii, profile = diagnostic["radii"], diagnostic["profiles"][ray]
            valid, edge = diagnostic["valid"][ray], diagnostic["edges"][ray]
            divisor = profile[0] if profile[0] > 0 else 1
            self.profile_plot.setTitle(f"Ray {ray} | {np.degrees(diagnostic['angles'][ray]):.1f}° | "
                                       + (f"edge {edge:.4f} px" if np.isfinite(edge) else "no measured edge"))
            self.profile_plot.setLabel("bottom", "Radius", units="px")
            self.profile_plot.setLabel("left", "Normalized signal / slope")
            self.profile_plot.plot(radii, np.where(valid, profile / divisor, np.nan), pen=COLORS[0],
                                   symbol="o", symbolSize=3, name="I(r) / I(0)")
            slope = -np.gradient(profile, radii[1] - radii[0])
            slope_scale = max(float(np.max(np.abs(slope[valid]))) if valid.any() else 0, 1e-12)
            self.profile_plot.plot(radii, np.where(valid, slope / slope_scale, np.nan), pen=COLORS[1], name="Falling slope / max |slope|")
            if method == "halfmax":
                self.profile_plot.addItem(pg.InfiniteLine(0.5, angle=0, pen=pg.mkPen(COLORS[2], style=QtCore.Qt.PenStyle.DashLine)))
            if np.isfinite(edge):
                self.profile_plot.addItem(pg.InfiniteLine(edge, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine)))
            support = radii[valid][-1] if valid.any() else 1
            self.profile_plot.setXRange(0, max(support + 0.1, float(edge) + .2 if np.isfinite(edge) else 1), padding=0)
            self.profile_plot.setYRange(-0.15, 1.15)
            direction = np.array([np.cos(diagnostic["angles"][ray]), np.sin(diagnostic["angles"][ray])])
            ray_center = diagnostic["center"] + origin
            end = ray_center + direction * support
            self.input_canvas.add_overlay("ray", pg.PlotDataItem([ray_center[0], end[0]], [ray_center[1], end[1]], pen=pg.mkPen("w", width=2)))
            if np.isfinite(edge):
                self.input_canvas.add_overlay("edge", pg.ScatterPlotItem(pos=[ray_center + direction * edge], size=10, brush=COLORS[1], pen=None))
            return
        row = int(np.clip(np.rint(center[1] - oy), 0, data.shape[0] - 1))
        xs = np.arange(data.shape[1]) + origin[0]
        if method in ("projection", "local_projection"):
            axis = self.projection_axis.currentText()
            diagnostic = projection_diagnostic(result, method, index, axis)
            if diagnostic is not None:
                grid, values, slope = diagnostic["grid"], diagnostic["values"], diagnostic["slope"]
                amplitude = max(float(np.max(np.abs(values))), 1e-12)
                slope_scale = max(float(np.max(np.abs(slope))), 1e-12)
                self.profile_plot.setTitle(f"{axis.upper()} projection | width {diagnostic['right'] - diagnostic['left']:.4g} px")
                self.profile_plot.setLabel("bottom", f"{axis} offset in input", units="px")
                self.profile_plot.setLabel("left", "Normalized signal / slope")
                self.profile_plot.plot(grid, values / amplitude, pen=COLORS[0], name="Cubic projection / max")
                self.profile_plot.plot(np.arange(len(diagnostic["samples"])), diagnostic["samples"] / amplitude,
                                       pen=None, symbol="o", symbolSize=4, symbolBrush=COLORS[0])
                self.profile_plot.plot(grid, slope / slope_scale, pen=COLORS[1], name="Slope / max |slope|")
                for value in (diagnostic["left"], diagnostic["right"]):
                    self.profile_plot.addItem(pg.InfiniteLine(value, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine)))
                self.detail_label.setText(f"#{index} | {description} | measured {axis} edges: {diagnostic['left']:.3f}, {diagnostic['right']:.3f} px in input")
        else:
            self.profile_plot.setTitle(f"Input row at y={row + origin[1]:.0f} px (context)")
            self.profile_plot.plot(xs, np.where(mask[row], data[row], np.nan), pen=COLORS[0], symbol="o", symbolSize=4, name="Input samples")
            if method == "gaussian" and "eigenvalues" in measured.geometry:
                values, axes = measured.geometry["eigenvalues"][index], measured.geometry["axes"][index]
                if np.isfinite(values).all() and (values > 0).all():
                    dense_x = np.linspace(0, data.shape[1] - 1, 300)
                    delta = np.column_stack((dense_x + ox - center[0], np.full(300, row + oy - center[1])))
                    precision = (axes * values) @ axes.T
                    model = measured.metrics["peak_signal"][index] * np.exp(-.5 * np.einsum("ni,ij,nj->n", delta, precision, delta))
                    self.profile_plot.plot(dense_x + origin[0], model, pen=COLORS[1], name="Fitted Gaussian (same row)")
                    sample_delta = np.column_stack((np.arange(data.shape[1]) + ox - center[0], np.full(data.shape[1], row + oy - center[1])))
                    sample_model = measured.metrics["peak_signal"][index] * np.exp(-.5 * np.einsum("ni,ij,nj->n", sample_delta, precision, sample_delta))
                    self.profile_plot.plot(xs, np.where(mask[row], data[row] - sample_model, np.nan),
                                           pen=COLORS[3], name="Row residual")
            if method == "logquad":
                background = 0.0
                if result.meta.get("background") != "model" and mask.any():
                    from ._cpu.cd import _background

                    background = _background(data, mask)
                peak = measured.metrics["peak_signal"][index]
                fit_mask = mask & (data - background > result.meta["fit_fraction"] * peak)
                rgba = np.zeros((*mask.shape, 4), np.uint8)
                rgba[fit_mask] = (255, 206, 115, 90)
                self.input_canvas.raster("fit_support", rgba, origin)
                self.profile_plot.addItem(pg.InfiniteLine(background + result.meta["fit_fraction"] * peak, angle=0,
                    pen=pg.mkPen(COLORS[1], style=QtCore.Qt.PenStyle.DashLine)))
                self.detail_label.setText(f"#{index} | {description} | gold pixels: actual log-fit support")
            context_methods = ("tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio")
            if "threshold" in measured.metrics and method not in context_methods:
                threshold = measured.metrics["threshold"][index]
                if np.isfinite(threshold):
                    self.profile_plot.addItem(pg.InfiniteLine(threshold, angle=0, pen=pg.mkPen(COLORS[1], style=QtCore.Qt.PenStyle.DashLine)))
                    self.detail_label.setText(f"#{index} | {description} | measured threshold={threshold:.5g}; outline uses 4x samples")
        self.input_canvas.add_overlay("profile_row", pg.PlotDataItem([origin[0] - .5, origin[0] + data.shape[1] - .5],
            [origin[1] + row, origin[1] + row], pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DotLine)))

    def _draw_pitch(self):
        for plot in self.pitch_plots.values():
            plot.clear()
        if self.review is None or self.method.currentData() not in self.review.patterns:
            self.pitch_label.setText("Run Pitch + placement to inspect this method's pair profiles.")
            return
        pattern = self.review.patterns[self.method.currentData()]
        descriptions = []
        for axis, color in zip(("x", "y"), (COLORS[1], COLORS[2])):
            profile = pattern.profiles[axis]
            nodes = profile.indices
            positions = (nodes[:-1] + nodes[1:]) / 2
            plot = self.pitch_plots["pitch", axis]
            plot.setLabel("bottom", "Logical interval")
            plot.setLabel("left", "Pitch", units=pattern.unit)
            plot.plot(positions, profile.pitch, pen=color, symbol="o", symbolSize=5, connect="finite")
            plot.addItem(pg.ErrorBarItem(x=positions, y=profile.pitch, top=profile.std, bottom=profile.std, pen=pg.mkPen(color)))
            plot.addItem(pg.InfiniteLine(profile.nominal_pitch, angle=0, pen=pg.mkPen("#899caf", style=QtCore.Qt.PenStyle.DashLine)))
            plot = self.pitch_plots["registration", axis]
            plot.setLabel("bottom", "Logical node")
            plot.setLabel("left", "Placement error", units=pattern.unit)
            plot.plot(nodes, profile.registration, pen=color, symbol="o", symbolSize=5, connect="finite")
            plot.addItem(pg.InfiniteLine(0, angle=0, pen="#586d84"))
            plot.addItem(pg.InfiniteLine(nodes[profile.reference_node], pen=pg.mkPen("#899caf", style=QtCore.Qt.PenStyle.DashLine)))
            descriptions.append(f"{axis}: {profile.counts.sum():,} pairs, baseline {profile.nominal_pitch:.5g} {pattern.unit}")
        self.pitch_label.setText(f"{pattern.layout} | ref #{pattern.reference_index} | "
            f"{pattern.meta['average_rows']} rows / {pattern.meta['average_columns']} columns | " + " | ".join(descriptions)
            + f"\nError bars: spread across contributing pairs per interval. Baseline: {pattern.meta['nominal_source']}. "
              f"Unindexed: {pattern.meta['unindexed_seeds']}. Gaps interrupt integration.")

    def _sync_moving_controls(self):
        """Moving page box: the moving file's pages, or the open file's until a moving image is chosen."""
        moving = self.moving
        pages = moving.pages if moving is not None else self.loaded.pages if self.loaded is not None else 1
        with QtCore.QSignalBlocker(self.moving_page_spin):
            self.moving_page_spin.setRange(0, max(0, pages - 1))
            if moving is not None:
                self.moving_page_spin.setValue(moving.page)
        if moving is None:
            self.moving_label.setText("Open a moving image of the same size, or 'Use this file' for a page of the "
                                      "open file. The active page is the fixed reference.")
        else:
            self.moving_label.setText(f"Moving: {moving.name} | page {moving.page}"
                                      + (f"\n{moving.conversion}" if moving.conversion else ""))

    def _moving_page_changed(self, value):
        if self.moving is not None and self.moving.path and value != self.moving.page and not self.busy:
            self.open_moving(path=self.moving.path, page=value)

    def _use_this_file(self):
        if self.loaded is not None and self.loaded.path and not self.busy:
            self.open_moving(path=self.loaded.path, page=min(self.moving_page_spin.value(), self.loaded.pages - 1))

    def open_moving(self, checked=False, *, path=None, page=0):
        if self.loaded is None:
            self.statusBar().showMessage("Open the fixed reference image first.", 5000)
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open moving image", self._last_folder(),
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy);;All files (*)")
        if not path:
            return
        self._remember_folder(path)

        def accept(image):
            if image.image.shape != self.loaded.image.shape:
                (mh, mw), (h, w) = image.image.shape, self.loaded.image.shape
                raise ValueError(f"The moving image is {mw} x {mh} px; the reference page is {w} x {h} px.")
            self.moving = image
            self.registration = None
            self._sync_moving_controls()
            self._set_state("Moving image ready")
            self.registration_label.setText("Ready to align. The fixed image is the active page.")
            for canvas in self.registration_canvases:
                canvas.image_item.clear()
            self.registration_canvases[0].set_image(image.image)
            self.registration_canvases[0].plot.setTitle("Moving image before alignment")
            self.diagnostics.setCurrentWidget(self.alignment_tab)

        self._run_job("Loading moving image", lambda: load_image(path, page), accept)

    def run_registration(self):
        if self.loaded is None or self.moving is None or self.busy:
            return
        problem = self._run_problems()[0]["register"]
        if problem:
            self._show_error(problem)
            return
        reference, moving, factor = self.loaded.image, self.moving.image, self.upsample.value()
        mode = self.hole_check_mode.currentData()

        def accept(result):
            self.registration = result
            dy, dx = result.result.shift_yx
            limit = HOLE_CHECK_MAX_ERROR[result.hole_check_mode]
            passes = result.hole_check_error < limit
            verdict = (f"below {limit}: the batch would measure this page" if passes else
                       f"not below {limit}: the batch would skip it as 'no holes'; this shift may mean nothing")
            self.registration_label.setText(
                f"Move the moving image by [dy, dx] = [{dy:.4f}, {dx:.4f}] px\n"
                f"Overlap RMSE: {result.rmse_before:.4g} → {result.rmse_after:.4g} (intensity units)\n"
                f"Hole check ({result.hole_check_mode}, as the batch): {result.hole_check_error:.3f}, {verdict}\n"
                "A lower RMSE does not rule out a shift by whole pitches on a repeating pattern.")
            self._draw_registration()
            self.log.appendPlainText(self.registration_label.text())
            if passes:
                self._set_state(f"Aligned | dy={dy:.4f}, dx={dx:.4f} px")
            else:
                self._set_state(f"Aligned, hole check failed ({result.hole_check_error:.2f})", "warning")
            self.diagnostics.setCurrentWidget(self.alignment_tab)

        self._run_job("Aligning images", lambda: register_images(reference, moving, upsample_factor=factor,
                                                                   hole_check_mode=mode), accept)

    def _draw_registration(self, *args):
        result = self.registration
        if result is None:
            return
        before, after, correlation = self.registration_canvases
        view = self.registration_view.currentIndex()
        if view == 0:
            before.set_image(result.difference_before)
            before.plot.setTitle("Moving - reference | before")
            after.set_image(result.difference_after)
            after.plot.setTitle("Aligned - reference | after")
            limit = max(float(np.percentile(np.abs(result.difference_before[result.overlap]), 99)),
                        float(np.nanpercentile(np.abs(result.difference_after), 99)), 1e-9)
            for canvas in (before, after):
                canvas.image_item.setLevels((-limit, limit))
                canvas.image_item.setLookupTable(pg.colormap.get("CET-D1").getLookupTable())
        else:
            before.set_image(self.loaded.image)
            before.plot.setTitle("Fixed reference")
            preview = result.aligned if view == 1 else .5 * self.loaded.image + .5 * result.aligned
            after.set_image(preview)
            after.plot.setTitle("Aligned moving image" if view == 1 else "50% aligned blend")
            for canvas in (before, after):
                canvas.image_item.setLevels(self.canvas.image_item.getLevels())
                canvas.image_item.setLookupTable(None)
        h, w = result.correlation.shape
        correlation.set_image(result.correlation, origin=(-(w // 2), -(h // 2)), categorical=True)
        correlation.plot.setTitle("Phase correlation | integer-lag context")

    def export_results(self):
        if self.busy or (self.review is None and self.registration is None):
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export parent folder", self._last_folder())
        if not directory:
            return
        from datetime import datetime

        output = Path(directory) / ("inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        review, registration = self.review, self.registration
        source = dict(path=self.loaded.path, name=self.loaded.name, page=self.loaded.page,
                      pages=self.loaded.pages, display_mode="stack" if self.stack_pages.isChecked() else "page",
                      shape=list(self.loaded.image.shape), dtype=str(self.loaded.image.dtype),
                      conversion=self.loaded.conversion, moving_path=self.moving.path if self.moving else None,
                      moving_page=self.moving.page if self.moving else None,
                      moving_conversion=self.moving.conversion if self.moving else None)

        def save():
            output.mkdir(parents=True)
            if review is not None:
                export_review(review, output)
            if registration is not None:
                np.savez_compressed(output / "registration.npz", shift_yx=registration.result.shift_yx,
                    aligned=registration.aligned, overlap=registration.overlap, correlation=registration.correlation,
                    difference_before=registration.difference_before, difference_after=registration.difference_after)
                source["registration"] = dict(shift_yx=registration.result.shift_yx.tolist(),
                    rmse_before=registration.rmse_before, rmse_after=registration.rmse_after,
                    hole_check_error=registration.hole_check_error, hole_check_mode=registration.hole_check_mode,
                    **registration.result.meta)
            (output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
            return output

        def accept(path):
            self.log.appendPlainText(f"Exported results to {path}")
            self._set_state("Results exported")
            self.statusBar().showMessage(f"Saved: {path}", 20000)

        self._run_job("Exporting results", save, accept)

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save workbench screenshot", "inspection.png", "PNG (*.png)")
        if path:
            if not self.grab().save(path):
                self._show_error("Could not save screenshot: " + path)
            else:
                self.statusBar().showMessage(f"Screenshot saved: {path}", 10000)

    def show_guide(self):
        QtWidgets.QMessageBox.information(self, "Inspection workflow",
            "Choose a task tab. Its settings and Run button stay beside the image, with results below. "
            "A disabled Run button says why under it.\n\n"
            "Find features: polarity, threshold and area limits; check the holes before measuring.\n\n"
            "Measure CD: the default preset is the batch's ten methods with the batch's settings. "
            "'Choose methods and parameters' ticks other methods. The line above the image lists the failed "
            "holes of every method.\n\n"
            "The table lists the holes; click a row or use the arrow keys to select a hole, click another "
            "method's CD to show that method. Clicking a hole on the image opens 'Selected feature' with its "
            "pixels, contours and profile.\n\n"
            "Pitch + placement: centers, lattice, averaging and design pitch. Align images: open a moving image "
            "or use a page of this file; the shift, RMSE and the batch's hole check are shown.\n\n"
            "Pages are counted from 0, as in the batch's pages.csv. Page Up / Page Down on the image change the "
            "page; the zoom, analysis region, selected position and moving image stay. 'Lock' keeps the contrast. "
            "Stack vertically shows every page; click a page to make it active.\n\n"
            "Wheel zooms, drag pans, F fits, Z zooms to the selected hole, Ctrl+Enter runs the task. "
            "Drop a file on the window to open it. Settings are remembered; 'Reset settings' restores the defaults.")

    def closeEvent(self, event):
        if self.busy:
            self.statusBar().showMessage("The background job is still running. Close after it completes.", 8000)
            event.ignore()
        else:
            self.save_settings()
            if self.stack is not None:
                self.stack.close()
                self.stack = None
            _WINDOWS.discard(self)
            event.accept()


def launch_workbench(image=None, *, name="Array image", block=None):
    """Open the inspection UI; return the window for notebook use.

    Parameters
    ----------
    image : ndarray, optional
        Finite 2-D grayscale or RGB(A) image.
    name : str
        Image label displayed in the window.
    block : bool, optional
        Start the Qt event loop. Defaults to True when this function creates
        the application, False when an application already exists. In IPython
        enable ``%gui qt`` before using ``block=False``.
    """
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication(["subpx-workbench"])
    window = InspectionWindow()
    if image is not None:
        window.set_image(image, name=name)
    window.show()
    windows = getattr(application, "_subpx_workbenches", [])
    windows.append(window)
    application._subpx_workbenches = windows
    should_block = owns_application if block is None else block
    if should_block:
        application.exec()
    return window


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive subpx image inspection workbench")
    parser.add_argument("image", nargs="?", help="TIFF, PNG, JPEG, BMP or 2-D NumPy image")
    parser.add_argument("--demo", action="store_true", help="Open a reproducible synthetic contact-hole image")
    args = parser.parse_args(argv)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = InspectionWindow()
    window.show()
    if args.image:
        window.open_image(path=args.image)
    elif args.demo:
        window.load_demo()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
