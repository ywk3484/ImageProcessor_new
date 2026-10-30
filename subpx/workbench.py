"""Interactive PyQt6/pyqtgraph inspection application.

Run ``python -m subpx.workbench [image]`` or call :func:`launch_workbench`.
Qt is deliberately imported only when this optional module is requested.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import traceback

import numpy as np

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:
    raise ImportError('The workbench requires PyQt6 and pyqtgraph. Install: pip install -e ".[analysis,viewer]"') from exc

from ._workbench_data import (
    AnalysisReview, LoadedImage, analyze_image, analyze_patterns, cell_boundary, demo_image,
    export_review, feature_input, feature_origin, grayscale, load_image, projection_diagnostic,
    radial_diagnostic, register_images,
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
            layout.addWidget(self.histogram)
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

    def set_image(self, data, origin=(0, 0), *, fit=True, categorical=False):
        key = (id(data), tuple(origin), categorical)
        preserve_levels = key == self._image_key
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.headers = []
        self.rows = []

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        if role == QtCore.Qt.ItemDataRole.UserRole:
            return row[0]
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and row[3] != "ok":
            return QtGui.QColor("#ff8c96")
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = row[index.column()]
            if isinstance(value, (float, np.floating)):
                return f"{value:.5g}" if np.isfinite(value) else "--"
            return str(value)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return None

    def set_review(self, review, method, failed_only=False):
        self.beginResetModel()
        measured = review.measurements[method]
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        self.headers = ["ID", "x (px)", "y (px)", "Status", f"CD ({unit})", f"CD x ({unit})", f"CD y ({unit})"]
        self.rows = []
        columns = [measured.metrics.get(key) for key in ("cd_equivalent_px", "cd_x_px", "cd_y_px")]
        for i, point in enumerate(measured.centers_xy + review.origin_xy):
            if failed_only and measured.valid[i]:
                continue
            values = [column[i] * scale if column is not None else np.nan for column in columns]
            self.rows.append([i, *point, str(measured.status[i]), *values])
        self.endResetModel()

    def sort(self, column, order=QtCore.Qt.SortOrder.AscendingOrder):
        self.layoutAboutToBeChanged.emit()
        self.rows.sort(key=lambda row: (isinstance(row[column], float) and not np.isfinite(row[column]), row[column]),
                       reverse=order == QtCore.Qt.SortOrder.DescendingOrder)
        self.layoutChanged.emit()


class InspectionWindow(QtWidgets.QMainWindow):
    """Desktop workbench for reviewing repository measurements and their inputs."""

    def __init__(self, image=None, *, title="subpx | Inspection workbench"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1680, 1060)
        self.setMinimumSize(1120, 740)
        self.loaded = None
        self.moving = None
        self.registration = None
        self.review = None
        self.selected = 0
        self.busy = False
        self._job = None
        self._radial_cache = None
        self._building = True
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._build_ui()
        self._apply_style()
        self._building = False
        self._update_enabled()
        if image is not None:
            self.set_image(image)

    def _build_ui(self):
        toolbar = self.addToolBar("Image and session")
        toolbar.setMovable(False)
        self.open_action = toolbar.addAction("Open image...", self.open_image)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.demo_action = toolbar.addAction("Load demo", self.load_demo)
        toolbar.addSeparator()
        toolbar.addAction("Fit image", lambda: self.canvas.fit_image()).setShortcut("F")
        toolbar.addAction("Zoom to feature", self.zoom_selected).setShortcut("Z")
        self.roi_action = toolbar.addAction("Analysis region")
        self.roi_action.setCheckable(True)
        self.roi_action.toggled.connect(self._toggle_roi)
        toolbar.addSeparator()
        self.export_action = toolbar.addAction("Export results...", self.export_results)
        toolbar.addAction("Save screenshot...", self.save_screenshot)
        toolbar.addSeparator()
        toolbar.addAction("Guide", self.show_guide)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(16, 12, 16, 8)
        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("subpx  /  INSPECTION WORKBENCH")
        title.setObjectName("title")
        heading.addWidget(title)
        heading.addStretch()
        self.state_label = QtWidgets.QLabel("Open an image to begin")
        self.state_label.setObjectName("state")
        heading.addWidget(self.state_label)
        layout.addLayout(heading)
        self.source_label = QtWidgets.QLabel("Explore the image, compare measurements, inspect how each method works.")
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.source_label)
        steps = QtWidgets.QHBoxLayout()
        self.detect_button = QtWidgets.QPushButton("1  Find patterns")
        self.cd_button = QtWidgets.QPushButton("2  Measure CD")
        self.pitch_button = QtWidgets.QPushButton("3  Pitch + placement")
        self.register_button = QtWidgets.QPushButton("4  Register images")
        for button, operation in ((self.detect_button, "detect"), (self.cd_button, "cd"), (self.pitch_button, "pitch")):
            button.clicked.connect(lambda checked=False, op=operation: self.run_analysis(op))
            button.setMinimumHeight(36)
            steps.addWidget(button)
        self.register_button.clicked.connect(self.run_registration)
        self.register_button.setMinimumHeight(36)
        steps.addWidget(self.register_button)
        layout.addLayout(steps)
        self.splitter = QtWidgets.QSplitter()
        layout.addWidget(self.splitter, 1)
        self.controls = self._build_controls()
        self.splitter.addWidget(self.controls)
        self.splitter.addWidget(self._build_workspace())
        self.splitter.addWidget(self._build_inspector())
        self.splitter.setSizes([310, 1020, 320])
        self.splitter.setStretchFactor(1, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(180)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Wheel: zoom  |  Drag: pan  |  Click a feature: inspect  |  F: fit  |  Z: selected feature")

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
        scroll.setMinimumWidth(265)
        scroll.setMaximumWidth(430)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        self.page_spin = self._number(1, 1, 1)
        self.page_spin.valueChanged.connect(self._page_changed)
        self.backend = self._combo([("CPU", "cpu"), ("Auto (CPU / GPU)", "auto"), ("GPU", "gpu")])
        self.pixel_size = self._number(1, 0.000001, 1e9, decimals=6, step=0.1)
        self.unit = QtWidgets.QLineEdit("px")
        self.unit.textChanged.connect(self._settings_changed)
        self._form_group("IMAGE & CALIBRATION", [("Page (1-based)", self.page_spin), ("Compute", self.backend),
                                                 ("Length / pixel", self.pixel_size), ("Length unit", self.unit)], layout)
        self.threshold = self._combo([("Otsu", "otsu"), ("Triangle", "triangle")])
        self.invert = self._check("Dark features on bright field")
        self.area_min = self._number(3, 1, 100000000)
        self.area_max = self._number(500, 1, 100000000)
        self.pad = self._number(5, 1, 1000)
        self.refine = self._combo([(x, x) for x in ("logquad", "weighted", "edge_gradmoment", "auto", "none")])
        self._form_group("DETECTION", [("Threshold", self.threshold), ("Polarity", self.invert),
                                      ("Min area (px²)", self.area_min), ("Max area (px²)", self.area_max),
                                      ("Cell padding (px)", self.pad), ("Center refinement", self.refine)], layout)
        layout.addWidget(self._hint("Find patterns shows segmentation and area rejection. CD seeds are calculated independently by the selected CD pipeline."))
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        cd_tab = QtWidgets.QWidget()
        cd_layout = QtWidgets.QVBoxLayout(cd_tab)
        cd_layout.setContentsMargins(4, 8, 4, 4)
        self.background = self._combo([("Gaussian neighbour model", "model"), ("Cell border (legacy)", "border")])
        self.seeds = self._combo([("Threshold components", "threshold"), ("Local peaks (dense patterns)", "peaks")])
        self._form_group("CELL PREPARATION", [("Background", self.background), ("Seed source", self.seeds)], cd_layout)
        self.method_list = QtWidgets.QListWidget()
        self.method_list.setMinimumHeight(180)
        self.method_list.setMaximumHeight(245)
        self.method_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.method_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for name, label in METHOD_LABELS.items():
            item = QtWidgets.QListWidgetItem(f"{name}  |  {label}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            item.setToolTip(METHOD_HELP.get(name, label + ". " + AREA_HELP))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if name in ("gaussian", "logquad", "gradient", "halfmax") else QtCore.Qt.CheckState.Unchecked)
            self.method_list.addItem(item)
        self.method_list.itemChanged.connect(self._settings_changed)
        cd_layout.addWidget(QtWidgets.QLabel("Methods to compare"))
        cd_layout.addWidget(self.method_list)
        self.n_angles = self._number(96, 16, 720, step=16)
        self.radial_step = self._number(0.1, 0.01, 1, decimals=2, step=0.05)
        self.smooth_sigma = self._number(0, 0, 10, decimals=2, step=0.1)
        self.fit_fraction = self._number(0.2, 0.01, 0.95, decimals=2, step=0.05)
        self.half_level = self._number(0.5, 0.01, 0.99, decimals=2, step=0.05)
        self.peak_pixels = self._number(4, 1, 100, decimals=1)
        self.tail_level = self._number(0.2, 0.01, 0.99, decimals=2, step=0.05)
        self.tail_window = self._number(1.2, 0.5, 2, decimals=2, step=0.1)
        self.iterations = self._number(30, 1, 1000)
        self.nominal_cd = self._number(0, 0, 1e9, decimals=4, step=1)
        self.psf_fwhm = self._number(0, 0, 1e9, decimals=4, step=1)
        self.dark_level = self._number(0, -1e9, 1e9, decimals=3)
        self._form_group("METHOD PARAMETERS", [("Rays", self.n_angles), ("Ray step (px)", self.radial_step),
                                             ("Ray smoothing (px)", self.smooth_sigma), ("Log-fit fraction", self.fit_fraction),
                                             ("Half-area level", self.half_level), ("Peak area (px²)", self.peak_pixels),
                                             ("Tail level", self.tail_level), ("Tail window (pitch)", self.tail_window),
                                             ("Fit iterations", self.iterations), ("Nominal CD (unit)*", self.nominal_cd),
                                             ("PSF FWHM (unit)*", self.psf_fwhm), ("Dark level", self.dark_level)], cd_layout)
        cd_layout.addWidget(self._hint("* 0 = unspecified. Disk and flux-ratio need calibration. Other methods do not require it. Raw local methods skip deblending when selected alone."))
        tabs.addTab(cd_tab, "CD")
        pitch_tab = QtWidgets.QWidget()
        pitch_layout = QtWidgets.QVBoxLayout(pitch_tab)
        self.pitch_source = self._combo([("CD method centers", "cd"), ("Detection centers", "detection")])
        self.pattern_layout = self._combo([("Automatic", "auto"), ("Square / rectangular", "square"), ("Staggered", "staggered")])
        self.stagger_axis = self._combo([("Offset rows", "row"), ("Offset columns", "column")])
        self.average_rows = self._number(0, 0, 1000000)
        self.average_columns = self._number(0, 0, 1000000)
        self.nominal_x = self._number(0, 0, 1e9, decimals=5)
        self.nominal_y = self._number(0, 0, 1e9, decimals=5)
        self.pitch_hint_x = self._number(0, 0, 1e9, decimals=3)
        self.pitch_hint_y = self._number(0, 0, 1e9, decimals=3)
        self.reference_selected = self._check("Use selected feature as reference")
        self._form_group("PITCH & PLACEMENT", [("Center source", self.pitch_source), ("Lattice", self.pattern_layout),
                                             ("Stagger direction", self.stagger_axis), ("Rows averaged (0=all)", self.average_rows),
                                             ("Columns averaged", self.average_columns), ("Nominal x (unit)", self.nominal_x),
                                             ("Nominal y (unit)", self.nominal_y), ("Pitch hint x (px)", self.pitch_hint_x),
                                             ("Pitch hint y (px)", self.pitch_hint_y), ("Reference", self.reference_selected)], pitch_layout)
        pitch_layout.addWidget(self._hint("Nominal x/y = 0 uses each measured mean as the baseline. Set both to use a design pitch. Placement error is the cumulative sum of (pitch - baseline), zero at the reference. Missing intervals break the curve. Pitch hints are for indexing only."))
        pitch_layout.addStretch()
        tabs.addTab(pitch_tab, "Pitch")
        reg_tab = QtWidgets.QWidget()
        reg_layout = QtWidgets.QVBoxLayout(reg_tab)
        self.moving_button = QtWidgets.QPushButton("Open moving image...")
        self.moving_button.clicked.connect(self.open_moving)
        reg_layout.addWidget(self.moving_button)
        self.moving_label = self._hint("The loaded image is the fixed reference. Open a second image of the same dimensions.")
        reg_layout.addWidget(self.moving_label)
        self.upsample = self._number(20, 1, 1000)
        self.moving_page_spin = self._number(1, 1, 1)
        self.moving_page_spin.valueChanged.connect(self._moving_page_changed)
        self._form_group("TRANSLATION", [("Moving page", self.moving_page_spin), ("Subpixel factor", self.upsample)], reg_layout)
        reg_layout.addWidget(self._hint("CPU phase cross-correlation estimates the translation to apply to the moving image. Shifts use [dy, dx] in pixels. Before/after differences and RMSE use the same valid overlap. Periodic patterns can have ambiguous peaks."))
        self.registration_label = self._hint("No image registration calculated.")
        reg_layout.addWidget(self.registration_label)
        reg_layout.addStretch()
        tabs.addTab(reg_tab, "Register")
        layout.addStretch()
        return scroll

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
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("View"))
        self.stage = self._combo([(x, x) for x in ("Loaded image", "Threshold mask", "Component labels", "Voronoi cells", "Background model")])
        self.stage.currentIndexChanged.disconnect(self._settings_changed)
        self.stage.currentIndexChanged.connect(self._render_image)
        row.addWidget(self.stage)
        row.addWidget(QtWidgets.QLabel("Method"))
        self.method = QtWidgets.QComboBox()
        self.method.currentIndexChanged.connect(self._method_changed)
        row.addWidget(self.method, 1)
        layout.addLayout(row)
        layers = QtWidgets.QHBoxLayout()
        self.layer_checks = {}
        for name, checked in (("Centers", True), ("Contours", True), ("Voronoi edges", False),
                              ("Rejected boxes", False), ("Pitch pairs", False), ("CD colors", False)):
            check = QtWidgets.QCheckBox(name)
            check.setChecked(checked)
            check.toggled.connect(self._render_image)
            self.layer_checks[name] = check
            layers.addWidget(check)
        layers.addStretch()
        layout.addLayout(layers)
        self.summary_label = self._hint("Load your image or choose Load demo to explore a synthetic staggered pattern.")
        layout.addWidget(self.summary_label)
        self.vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(self.vertical_splitter, 1)
        self.canvas = ImageCanvas(histogram=True)
        self.canvas.clicked.connect(self._select_at)
        self.canvas.hovered.connect(self._hover_image)
        self.vertical_splitter.addWidget(self.canvas)
        self.roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#ffce73", width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.sigRegionChangeFinished.connect(self._settings_changed)
        self.canvas.plot.addItem(self.roi)
        self.roi.hide()
        self.diagnostics = QtWidgets.QTabWidget()
        self.expand_diagnostics = QtWidgets.QToolButton()
        self.expand_diagnostics.setText("Expand")
        self.expand_diagnostics.setCheckable(True)
        self.expand_diagnostics.setToolTip("Give the diagnostic plots more room")
        self.expand_diagnostics.toggled.connect(self._focus_diagnostics)
        self.diagnostics.setCornerWidget(self.expand_diagnostics)
        self.vertical_splitter.addWidget(self.diagnostics)
        self.vertical_splitter.setSizes([520, 330])
        self._build_detail_tab()
        self._build_pitch_tab()
        self._build_registration_tab()
        self._build_table_tab()
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
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        controls = QtWidgets.QHBoxLayout()
        self.detail_label = self._hint("Click a detected feature to inspect its input and measurement.")
        controls.addWidget(self.detail_label, 1)
        controls.addWidget(QtWidgets.QLabel("Projection"))
        self.projection_axis = QtWidgets.QComboBox()
        self.projection_axis.addItems(["x", "y"])
        self.projection_axis.currentIndexChanged.connect(self._draw_detail)
        controls.addWidget(self.projection_axis)
        controls.addWidget(QtWidgets.QLabel("Ray"))
        self.ray = QtWidgets.QSpinBox()
        self.ray.setRange(0, 95)
        self.ray.valueChanged.connect(self._draw_detail)
        controls.addWidget(self.ray)
        layout.addLayout(controls)
        splitter = QtWidgets.QSplitter()
        self.raw_canvas = ImageCanvas()
        self.input_canvas = ImageCanvas()
        self.profile_plot = pg.PlotWidget(background="#0c121b")
        self.profile_plot.addLegend(offset=(8, 8))
        self.profile_plot.showGrid(x=True, y=True, alpha=0.15)
        for canvas in (self.raw_canvas, self.input_canvas):
            canvas.setMinimumWidth(100)
            splitter.addWidget(canvas)
        splitter.addWidget(self.profile_plot)
        splitter.setSizes([240, 240, 440])
        layout.addWidget(splitter, 1)
        self.diagnostics.addTab(tab, "Feature / CD steps")

    def _build_pitch_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.pitch_label = self._hint("Run Pitch + placement to inspect adjacent-pair pitch and its cumulative error.")
        layout.addWidget(self.pitch_label)
        graphics = pg.GraphicsLayoutWidget()
        graphics.setBackground("#0c121b")
        self.pitch_plots = {}
        for row, key in enumerate(("pitch", "registration")):
            for column, axis in enumerate(("x", "y")):
                plot = graphics.addPlot(row=row, col=column, title=f"{axis.upper()} {key}")
                plot.showGrid(x=True, y=True, alpha=0.15)
                self.pitch_plots[key, axis] = plot
        layout.addWidget(graphics, 1)
        self.diagnostics.addTab(tab, "Pitch & placement")

    def _build_registration_tab(self):
        tab = QtWidgets.QWidget()
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
        self.diagnostics.addTab(tab, "Image registration")

    def _build_table_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        self.failed_only = QtWidgets.QCheckBox("Show failed measurements only")
        self.failed_only.toggled.connect(self._update_table)
        layout.addWidget(self.failed_only)
        self.table_model = MeasurementTable(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.table_model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.clicked.connect(lambda index: self.select_feature(int(self.table_model.data(index, QtCore.Qt.ItemDataRole.UserRole))))
        layout.addWidget(self.table, 1)
        self.diagnostics.addTab(tab, "Measurements")

    def _build_inspector(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(440)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(8, 4, 4, 4)
        label = QtWidgets.QLabel("FEATURE INSPECTOR")
        label.setObjectName("section")
        layout.addWidget(label)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("ID"))
        self.feature_spin = QtWidgets.QSpinBox()
        self.feature_spin.setRange(0, 0)
        self.feature_spin.valueChanged.connect(self.select_feature)
        controls.addWidget(self.feature_spin, 1)
        next_failed = QtWidgets.QPushButton("Next failed")
        next_failed.clicked.connect(self.next_failed)
        controls.addWidget(next_failed)
        layout.addLayout(controls)
        self.feature_label = self._hint("No feature selected.")
        layout.addWidget(self.feature_label)
        self.method_help = self._hint("Select a method to see its definition and assumptions.")
        layout.addWidget(self.method_help)
        layout.addWidget(QtWidgets.QLabel("METHOD COMPARISON  |  calibrated CD"))
        self.comparison = QtWidgets.QTableWidget(0, 3)
        self.comparison.setHorizontalHeaderLabels(["Method", "CD", "Status"])
        self.comparison.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.comparison.horizontalHeader().setStretchLastSection(True)
        self.comparison.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison.verticalHeader().hide()
        self.comparison.setMinimumHeight(155)
        self.comparison.setMaximumHeight(245)
        self.comparison.cellClicked.connect(lambda row, col: self.method.setCurrentIndex(row))
        layout.addWidget(self.comparison)
        layout.addWidget(QtWidgets.QLabel("SELECTED METHOD DETAILS"))
        self.metrics = QtWidgets.QTableWidget(0, 2)
        self.metrics.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.metrics.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.metrics.verticalHeader().hide()
        self.metrics.setMinimumHeight(210)
        layout.addWidget(self.metrics)
        layout.addWidget(QtWidgets.QLabel("REVIEW NOTES"))
        self.notes_label = self._hint("Failed measurements remain visible. Inspect their status; a failed fit is never replaced with a successful seed measurement.")
        layout.addWidget(self.notes_label)
        layout.addStretch()
        return scroll

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #151e2b; color: #dee7f2; font-family: 'Segoe UI'; font-size: 12px; }
            QToolBar { background: #1b2839; padding: 7px; spacing: 7px; border: 0; }
            QToolButton { padding: 6px 10px; border-radius: 4px; }
            QToolButton:hover, QToolButton:checked { background: #30465e; }
            QLabel#title { font-size: 20px; font-weight: 600; color: #f4f8ff; }
            QLabel#state { color: #56d6cf; padding: 6px 12px; background: #203c44; border-radius: 4px; }
            QLabel#section { color: #56d6cf; font-weight: 600; padding-bottom: 8px; }
            QLabel#hint { color: #a9bbce; line-height: 1.4; }
            QPushButton { background: #24384e; border: 1px solid #3c556e; border-radius: 4px; padding: 7px 9px; }
            QPushButton:hover { background: #31516b; border-color: #56d6cf; }
            QPushButton:disabled { color: #6b7f92; border-color: #293a4c; }
            QGroupBox { border: 1px solid #304155; border-radius: 5px; margin-top: 12px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 9px; color: #90b2d2; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #0f1824; border: 1px solid #354a60; border-radius: 3px; padding: 4px; min-height: 20px; }
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
        """)

    def _settings_changed(self, *args):
        if not self._building and (self.review is not None or self.registration is not None) and not self.busy:
            self.state_label.setText("Settings changed | run analysis to apply")

    def _update_enabled(self):
        available = self.loaded is not None and not self.busy
        for button in (self.detect_button, self.cd_button, self.pitch_button):
            button.setEnabled(available)
        self.register_button.setEnabled(available and self.moving is not None)
        self.export_action.setEnabled(not self.busy and (self.review is not None or self.registration is not None))
        self.open_action.setEnabled(not self.busy)
        self.demo_action.setEnabled(not self.busy)
        self.roi_action.setEnabled(available)
        self.controls.setEnabled(not self.busy)

    def _run_job(self, message, function, on_result):
        if self.busy:
            return
        self.busy = True
        self.state_label.setText(message)
        self.log.appendPlainText(message)
        self.progress.setRange(0, 0)
        self.progress.show()
        self._update_enabled()
        self._job = _Job(function)

        def done(result):
            self.busy = False
            self.progress.hide()
            try:
                on_result(result)
            except Exception:
                self._show_error(traceback.format_exc())
            self._update_enabled()

        def failed(detail):
            self.busy = False
            self.progress.hide()
            self._show_error(detail)
            self._update_enabled()

        self._job.signals.finished.connect(done)
        self._job.signals.failed.connect(failed)
        self._pool.start(self._job)

    def _show_error(self, detail):
        self.log.appendPlainText(detail)
        self.state_label.setText("Action failed | see Activity")
        self.statusBar().showMessage(detail.strip().splitlines()[-1], 20000)
        self.diagnostics.setCurrentWidget(self.log)

    def open_image(self, checked=False, *, path=None, page=0):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open inspection image", "",
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy);;All files (*)")
        if path:
            self._run_job("Loading image...", lambda: load_image(path, page), self._set_loaded)

    def _page_changed(self, value):
        if self.loaded is not None and self.loaded.path and value - 1 != self.loaded.page and not self.busy:
            self.open_image(path=self.loaded.path, page=value - 1)

    def set_image(self, image, *, name="Array image"):
        """Load a finite grayscale/RGB NumPy array, resetting dependent results."""
        if self.busy:
            raise RuntimeError("Wait for the current job before replacing the image.")
        data, conversion = grayscale(image)
        self._set_loaded(LoadedImage(data, name, conversion=conversion))

    def _set_loaded(self, loaded):
        self.loaded = loaded
        self.review = self.registration = self.moving = None
        self._radial_cache = None
        self.selected = 0
        self.method.clear()
        self.stage.setCurrentIndex(0)
        self.page_spin.blockSignals(True)
        self.page_spin.setRange(1, loaded.pages)
        self.page_spin.setValue(loaded.page + 1)
        self.page_spin.blockSignals(False)
        self.page_spin.setEnabled(loaded.pages > 1)
        h, w = loaded.image.shape
        self.source_label.setText(f"{loaded.name}  |  {w:,} x {h:,} px  |  {loaded.image.dtype}  |  page {loaded.page + 1}/{loaded.pages}"
                                 + (f"  |  {loaded.conversion}" if loaded.conversion else ""))
        self.source_label.setToolTip(loaded.path or loaded.name)
        self.state_label.setText("Image ready")
        self.summary_label.setText("Choose Find patterns to review segmentation, Measure CD to compare methods, or Pitch + placement to run both.")
        self.feature_label.setText("Click a feature after analysis.")
        self.feature_spin.setRange(0, 0)
        self.metrics.setRowCount(0)
        self.comparison.setRowCount(0)
        self.table_model.beginResetModel()
        self.table_model.rows = []
        self.table_model.endResetModel()
        self.profile_plot.clear()
        self.detail_label.setText("Click a detected feature to inspect its input and measurement.")
        self.method_help.setText("Select a method to see its definition and assumptions.")
        self.notes_label.setText("Review failed fits and method assumptions before accepting measurements.")
        for canvas in (self.raw_canvas, self.input_canvas, *self.registration_canvases):
            canvas.image_item.clear()
            canvas.clear_overlays()
        for plot in self.pitch_plots.values():
            plot.clear()
        self.pitch_label.setText("Run Pitch + placement to calculate profiles.")
        self.registration_label.setText("No image registration calculated.")
        self.moving_page_spin.setRange(1, 1)
        self.moving_label.setText("Open a moving image of the same dimensions. The loaded image is the fixed reference.")
        self.roi_action.setChecked(False)
        self.roi.setPos((w * 0.25 - 0.5, h * 0.25 - 0.5))
        self.roi.setSize((w * 0.5, h * 0.5))
        self.roi.maxBounds = QtCore.QRectF(-0.5, -0.5, w, h)
        self._render_image()
        self.canvas.fit_image()
        self.log.appendPlainText(f"Loaded {loaded.name}: shape={loaded.image.shape}, dtype={loaded.image.dtype}. {loaded.conversion}")
        self._update_enabled()

    def load_demo(self):
        def generate():
            return LoadedImage(demo_image(),
                               "Synthetic staggered contact holes (seed 42)")
        self._run_job("Generating demonstration image...", generate, self._set_loaded)

    def _toggle_roi(self, enabled):
        self.roi.setVisible(enabled)
        self._settings_changed()

    def _bounds(self):
        if not self.roi_action.isChecked():
            return None
        h, w = self.loaded.image.shape
        position, size = self.roi.pos(), self.roi.size()
        # Integer-centered pixels selected by the ROI's visual half-pixel edges.
        x0 = max(0, min(w, int(np.ceil(position.x()))))
        y0 = max(0, min(h, int(np.ceil(position.y()))))
        x1 = max(0, min(w, int(np.ceil(position.x() + size.x()))))
        y1 = max(0, min(h, int(np.ceil(position.y() + size.y()))))
        return [x0, y0, x1, y1]

    def parameters(self):
        """Snapshot all controls so background jobs never access Qt widgets."""
        methods = tuple(self.method_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                        for i in range(self.method_list.count())
                        if self.method_list.item(i).checkState() == QtCore.Qt.CheckState.Checked)
        nominal = [self.nominal_x.value(), self.nominal_y.value()]
        hint = [self.pitch_hint_x.value(), self.pitch_hint_y.value()]
        if (nominal[0] == 0) != (nominal[1] == 0):
            raise ValueError("Set both nominal x/y pitches, or leave both at zero for a measured-mean baseline.")
        if (hint[0] == 0) != (hint[1] == 0):
            raise ValueError("Set both x/y pitch hints, or leave both at zero for automatic indexing.")
        reference = None
        if self.reference_selected.isChecked() and self.review is not None and len(self.review.seeds_xy):
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
        try:
            parameters, bounds = self.parameters(), self._bounds()
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
            relevant = ("threshold", "invert", "area_min", "area_max", "pad", "backend", "pixel_size", "unit")
            if parameters["pitch_source"] == "cd":
                relevant += ("methods", "background", "seeds", "n_angles", "radial_step", "smooth_sigma", "fit_fraction",
                             "half_area_level", "half_area_peak_pixels", "tail_area_level", "tail_window", "deblend_iterations",
                             "nominal_cd", "psf_fwhm", "dark_level")
            else:
                relevant += ("refine",)
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
                    result.notes.append(f"Pitch indexing failed: {exc}")
                result.parameters["operation"] = "pitch"
            return result

        labels = {"detect": "Finding patterns...", "cd": "Measuring CD...", "pitch": "Calculating pitch and placement..."}
        self._run_job(labels[operation], calculate, self._accept_review)

    def _accept_review(self, review):
        self.review = review
        self._radial_cache = None
        old = self.method.currentData()
        self.method.blockSignals(True)
        self.method.clear()
        for name in review.measurements:
            self.method.addItem(name, name)
        self.method.setCurrentIndex(max(0, self.method.findData(old)))
        self.method.blockSignals(False)
        self.feature_spin.blockSignals(True)
        self.feature_spin.setRange(0, max(0, len(review.seeds_xy) - 1))
        self.feature_spin.blockSignals(False)
        self.selected = min(self.selected, max(0, len(review.seeds_xy) - 1))
        self.state_label.setText(f"{len(review.seeds_xy):,} features | {review.seconds:.2f} s | "
                                 + ("review notes" if review.notes else "ready"))
        self.notes_label.setText("\n\n".join(review.notes) or "No method failures reported. Inspect image support and fit assumptions before accepting a measurement.")
        self.log.appendPlainText(f"Analysis complete in {review.seconds:.3f} s. Region origin [x,y]={review.origin_xy.tolist()}.")
        if review.cd is not None:
            self.log.appendPlainText(f"Background: {review.cd.meta.get('background_definition')}")
            self.log.appendPlainText("Stage timings (s): " + str(review.cd.meta.get("seconds", {})))
        for note in review.notes:
            self.log.appendPlainText(note)
        self._method_changed()
        if not len(review.seeds_xy):
            self.feature_label.setText("No features found. Review segmentation, area limits, polarity and region.")
            self.metrics.setRowCount(0)
            self.comparison.setRowCount(0)
            self.detail_label.setText("No feature input is available for this calculation.")
            self.profile_plot.clear()
            for canvas in (self.raw_canvas, self.input_canvas):
                canvas.image_item.clear()
                canvas.clear_overlays()
        if review.patterns:
            self.diagnostics.setCurrentIndex(1)
            self.layer_checks["Pitch pairs"].setChecked(True)
        else:
            self.diagnostics.setCurrentIndex(0)

    def _method_changed(self, *args):
        self._radial_cache = None
        self.ray.setEnabled(self.method.currentData() in ("gradient", "halfmax"))
        self.projection_axis.setEnabled(self.method.currentData() in ("projection", "local_projection"))
        self._render_image()
        self._update_table()
        self._draw_pitch()
        self.select_feature(self.selected)

    def _render_image(self, *args):
        if self.loaded is None:
            return
        canvas, review = self.canvas, self.review
        canvas.clear_overlays()
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
        canvas.set_image(data, origin, fit=False, categorical=categorical)
        self.roi.setZValue(50)
        if review is None or not len(review.seeds_xy) or self.method.currentData() not in review.measurements:
            if review is not None and not len(review.seeds_xy):
                self.summary_label.setText("No features found. Inspect the threshold mask, area limits, polarity and region.")
            elif message:
                self.summary_label.setText(message)
            return
        name = self.method.currentData()
        measured = review.measurements[name]
        points = measured.centers_xy + review.origin_xy
        seeds = review.seeds_xy + review.origin_xy
        valid = measured.valid & np.isfinite(points).all(axis=1)
        flags = self.layer_checks
        if flags["Voronoi edges"].isChecked() and review.cd is not None:
            edge = cell_boundary(review.cd.cell_labels)
            rgba = np.zeros((*edge.shape, 4), np.uint8)
            rgba[edge] = (86, 214, 207, 165)
            canvas.raster("cells", rgba, review.origin_xy)
        color_message = ""
        if flags["CD colors"].isChecked() and "cd_equivalent_px" in measured.metrics:
            values = measured.metrics["cd_equivalent_px"] * review.parameters["pixel_size"]
            good = valid & np.isfinite(values)
            if good.any():
                low, high = np.min(values[good]), np.max(values[good])
                colors = pg.colormap.get("viridis").map((values[good] - low) / max(high - low, 1e-12), mode="byte")
                canvas.add_overlay("values", pg.ScatterPlotItem(pos=points[good], size=10,
                    brush=[pg.mkBrush(c) for c in colors], pen=None))
                color_message = f" | CD color: {low:.4g} (purple) to {high:.4g} (yellow) {review.parameters['unit']}"
        if flags["Centers"].isChecked():
            canvas.add_overlay("centers", pg.ScatterPlotItem(pos=points[valid], size=6,
                pen=pg.mkPen("#56d6cf", width=1), brush=pg.mkBrush(86, 214, 207, 60), symbol="+"))
            canvas.add_overlay("failed", pg.ScatterPlotItem(pos=seeds[~valid], size=10,
                pen=pg.mkPen("#ff748a", width=2), symbol="x"))
        if flags["Contours"].isChecked() and measured.contours_xy.size:
            contours = measured.contours_xy + review.origin_xy
            segments = np.concatenate((contours, np.full((len(contours), 1, 2), np.nan)), axis=1).reshape(-1, 2)
            canvas.add_overlay("contours", pg.PlotDataItem(segments[:, 0], segments[:, 1], connect="finite",
                pen=pg.mkPen(COLORS[list(review.measurements).index(name) % len(COLORS)], width=1.3)))
        if flags["Rejected boxes"].isChecked() and review.detection is not None:
            segments = []
            for x, y, w, h, area in review.detection.boxes[~review.detection.kept]:
                segments.extend([(x - .5, y - .5), (x + w - .5, y - .5), (x + w - .5, y + h - .5),
                                 (x - .5, y + h - .5), (x - .5, y - .5), (np.nan, np.nan)])
            if segments:
                segments = np.array(segments) + review.origin_xy
                canvas.add_overlay("rejected", pg.PlotDataItem(segments[:, 0], segments[:, 1], connect="finite", pen="#ff748a"))
        if flags["Pitch pairs"].isChecked() and name in review.patterns:
            pattern = review.patterns[name]
            for axis, color in zip(("x", "y"), ("#ffce73", "#b49aff")):
                pairs = pattern.profiles[axis].pairs
                if len(pairs):
                    segments = points[pairs].reshape(-1, 2)
                    canvas.add_overlay(f"pairs_{axis}", pg.PlotDataItem(segments[:, 0], segments[:, 1], connect="pairs", pen=pg.mkPen(color, width=1)))
            ref = seeds[pattern.reference_index]
            canvas.add_overlay("reference", pg.ScatterPlotItem(pos=[ref], symbol="star", size=18, brush="#ffce73", pen="#fff3cf"))
        if 0 <= self.selected < len(seeds):
            canvas.add_overlay("selected", pg.ScatterPlotItem(pos=[seeds[self.selected]], symbol="o", size=20,
                brush=None, pen=pg.mkPen("#ffffff", width=2)))
        self.summary_label.setText(message + f"{valid.sum():,}/{len(valid):,} valid | {name} | "
            f"region origin ({int(review.origin_xy[0])}, {int(review.origin_xy[1])}) px | red x: failed" + color_message)

    def _select_at(self, x, y):
        if self.review is None or not len(self.review.seeds_xy):
            return
        local = np.array([x, y]) - self.review.origin_xy
        h, w = self.review.image.shape
        if not (-0.5 <= local[0] < w - 0.5 and -0.5 <= local[1] < h - 0.5):
            return
        index = -1
        if self.review.cd is not None:
            index = int(self.review.cd.cell_labels[int(np.floor(local[1] + .5)), int(np.floor(local[0] + .5))])
        if index < 0:
            index = int(np.argmin(np.sum((self.review.seeds_xy - local) ** 2, axis=1)))
        self.select_feature(index)

    def _hover_image(self, x, y):
        if self.loaded is None:
            return
        ix, iy = int(np.floor(x + .5)), int(np.floor(y + .5))
        if 0 <= ix < self.loaded.image.shape[1] and 0 <= iy < self.loaded.image.shape[0]:
            self.statusBar().showMessage(f"x={x:.2f}, y={y:.2f} px | pixel [{iy}, {ix}] = {self.loaded.image[iy, ix]:.6g} | wheel: zoom, drag: pan, click: inspect")

    def select_feature(self, index):
        review = self.review
        if review is None or not len(review.seeds_xy) or self.method.currentData() not in review.measurements:
            return
        self.selected = max(0, min(int(index), len(review.seeds_xy) - 1))
        self.feature_spin.blockSignals(True)
        self.feature_spin.setValue(self.selected)
        self.feature_spin.blockSignals(False)
        self._radial_cache = None
        method = self.method.currentData()
        measured = review.measurements[method]
        point = measured.centers_xy[self.selected] + review.origin_xy
        seed = review.seeds_xy[self.selected] + review.origin_xy
        self.feature_label.setText(f"Feature {self.selected} | {measured.status[self.selected]}\n"
                                  f"Seed x={seed[0]:.4f}, y={seed[1]:.4f} px\n"
                                  f"Method x={point[0]:.4f}, y={point[1]:.4f} px")
        self.method_help.setText(METHOD_HELP.get(method, METHOD_LABELS.get(method, "Subpixel center detection") + ". "
                                               + (AREA_HELP if method != "detection" else "Review the threshold mask and area limits before measuring CD.")))
        self.comparison.setRowCount(len(review.measurements))
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        for row, (name, measurement) in enumerate(review.measurements.items()):
            cd = measurement.metrics.get("cd_equivalent_px")
            values = [name, f"{cd[self.selected] * scale:.5g} {unit}" if cd is not None and np.isfinite(cd[self.selected]) else "--",
                      str(measurement.status[self.selected])]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
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
        self._render_image()
        self._draw_detail()
        if self.reference_selected.isChecked() and method in review.patterns and not self.busy:
            if review.patterns[method].reference_index != self.selected:
                self.state_label.setText("Reference changed | rerun Pitch + placement")

    def next_failed(self):
        if self.review is None or self.method.currentData() not in self.review.measurements:
            return
        bad = np.flatnonzero(~self.review.measurements[self.method.currentData()].valid)
        if len(bad):
            later = bad[bad > self.selected]
            self.select_feature(later[0] if len(later) else bad[0])
            self.zoom_selected()
        else:
            self.statusBar().showMessage("No failed measurements for the selected method.", 5000)

    def zoom_selected(self):
        if self.review is None or not len(self.review.seeds_xy):
            return
        point = self.review.seeds_xy[self.selected] + self.review.origin_xy
        radius = 15
        if self.review.cd is not None:
            radius = max(8, max(self.review.cd.masks.shape[1:]) * 1.5)
        self.canvas.plot.setRange(xRange=(point[0] - radius, point[0] + radius),
                                  yRange=(point[1] - radius, point[1] + radius), padding=0)

    def _update_table(self, *args):
        if self.review is not None and self.method.currentData() in self.review.measurements:
            self.table_model.set_review(self.review, self.method.currentData(), self.failed_only.isChecked())

    def _draw_detail(self, *args):
        review = self.review
        if review is None or not len(review.seeds_xy):
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
        self.raw_canvas.plot.setTitle("Original pixels / cell boundary")
        self.input_canvas.plot.setTitle("Method input / contours")
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

    def _moving_page_changed(self, value):
        if self.moving is not None and self.moving.path and value - 1 != self.moving.page and not self.busy:
            self.open_moving(path=self.moving.path, page=value - 1)

    def open_moving(self, checked=False, *, path=None, page=0):
        if self.loaded is None:
            self.statusBar().showMessage("Open the fixed reference image first.", 5000)
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open moving image", "",
                "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy);;All files (*)")
        if not path:
            return

        def accept(image):
            if image.image.shape != self.loaded.image.shape:
                raise ValueError(f"Moving image shape {image.image.shape} must match reference {self.loaded.image.shape}.")
            self.moving = image
            self.registration = None
            self.moving_page_spin.blockSignals(True)
            self.moving_page_spin.setRange(1, image.pages)
            self.moving_page_spin.setValue(image.page + 1)
            self.moving_page_spin.blockSignals(False)
            self.moving_label.setText(f"Moving: {image.name} | page {image.page + 1}\n{image.conversion}")
            self.state_label.setText("Moving image ready")
            self.registration_label.setText("Ready to register. The fixed image is the loaded reference.")
            for canvas in self.registration_canvases:
                canvas.image_item.clear()
            self.registration_canvases[0].set_image(image.image)
            self.registration_canvases[0].plot.setTitle("Moving image before alignment")
            self.diagnostics.setCurrentIndex(2)

        self._run_job("Loading moving image...", lambda: load_image(path, page), accept)

    def run_registration(self):
        if self.loaded is None or self.moving is None or self.busy:
            return
        reference, moving, factor = self.loaded.image, self.moving.image, self.upsample.value()

        def accept(result):
            self.registration = result
            dy, dx = result.result.shift_yx
            self.registration_label.setText(f"Apply [dy, dx] = [{dy:.5f}, {dx:.5f}] px\n"
                f"Overlap RMSE: {result.rmse_before:.5g} > {result.rmse_after:.5g} intensity units\n"
                f"Phase-correlation normalized error: {result.result.error:.5g}\n"
                "RMSE compares the same overlap before and after translation. A lower RMSE does not resolve periodic-peak ambiguity.")
            self._draw_registration()
            self.log.appendPlainText(self.registration_label.text())
            self.state_label.setText(f"Registered | dy={dy:.4f}, dx={dx:.4f} px")
            self.diagnostics.setCurrentIndex(2)

        self._run_job("Registering images...", lambda: register_images(reference, moving, upsample_factor=factor), accept)

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
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose export parent folder")
        if not directory:
            return
        from datetime import datetime
        import json

        output = Path(directory) / ("inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        review, registration = self.review, self.registration
        source = dict(path=self.loaded.path, name=self.loaded.name, page=self.loaded.page,
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
                    normalized_error=registration.result.error, **registration.result.meta)
            (output / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
            return output

        def accept(path):
            self.log.appendPlainText(f"Exported results to {path}")
            self.state_label.setText("Results exported")
            self.statusBar().showMessage(f"Saved: {path}", 20000)

        self._run_job("Exporting results...", save, accept)

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save workbench screenshot", "inspection.png", "PNG (*.png)")
        if path:
            if not self.grab().save(path):
                self._show_error("Could not save screenshot: " + path)
            else:
                self.statusBar().showMessage(f"Screenshot saved: {path}", 10000)

    def show_guide(self):
        QtWidgets.QMessageBox.information(self, "Inspection workflow",
            "1. Open an image or load the synthetic demo. Set polarity, component area limits and calibration.\n\n"
            "2. Find patterns. Review the threshold mask and rejected boxes. Detection refinement only applies to this step.\n\n"
            "3. Select CD methods and Measure CD. Toggle contours and Voronoi edges. Click the image, a table row, or a feature ID. "
            "Compare definitions, statuses, fit inputs and individual radial samples. Gold pixels mark log-fit support. "
            "Context plots are labelled when they are not the complete method input.\n\n"
            "4. Pitch + placement indexes the selected center source. Choose averaging bands, a reference and a nominal pitch in the Pitch settings. "
            "Gold/purple links show actual x/y pairs. Placement error is the cumulative pitch deviation, not two-image alignment.\n\n"
            "5. Register images compares the full loaded reference with a second image, independently of the analysis region. "
            "Inspect same-scale differences and the phase-correlation peak.\n\n"
            "Wheel zooms, drag pans. F fits the image; Z zooms to the selected feature. Drag Analysis region before running to measure a crop. "
            "The right histogram adjusts display contrast only. Parameter changes require another run. "
            "Export includes parameters, failures, coordinate origins, arrays and tables.")

    def closeEvent(self, event):
        if self.busy:
            self.statusBar().showMessage("The background job is still running. Close after it completes.", 8000)
            event.ignore()
        else:
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
