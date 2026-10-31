"""Setup (left): the steps Image, Holes, CD methods and Pitch in pipeline order, each with a mark (not run, ok,
changed), and the Run button of the open step pinned under them; in the Align view, the alignment settings."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from ...cd import METHOD_LABELS
from ..names import CALIBRATED_METHODS, METHOD_GROUPS, RECIPES, method_help, method_line, method_name
from ..settings import STEP_TITLES
from ..widgets import BACKGROUND, Fold, dot_icon, hint

MARKS = {"none": ("#5d6f82", "not run"), "ok": ("#9be28a", "up to date"), "changed": ("#ffce73", "changed")}
RUNS = {"image": ("Measure CD", "cd"), "holes": ("Preview holes", "detect"), "cd": ("Measure CD", "cd"),
        "pitch": ("Measure pitch", "pitch"), "align": ("Align", "register")}
# Analysis parameter -> Setup control (Revert puts a result's parameters back into the controls).
PARAMETER_CONTROLS = dict(
    invert="invert", threshold="threshold", area_min="area_min", area_max="area_max", pad="pad", refine="refine",
    seeds="seeds", background="background", n_angles="n_angles", radial_step="radial_step",
    smooth_sigma="smooth_sigma", fit_fraction="fit_fraction", half_area_level="half_level",
    half_area_peak_pixels="peak_pixels", tail_area_level="tail_level", tail_window="tail_window",
    deblend_iterations="iterations", deblend_tolerance="tolerance", dark_level="dark_level",
    pixel_size="pixel_size", unit="unit", backend="backend", nominal_cd="nominal_cd", psf_fwhm="psf_fwhm",
    pitch_source="pitch_source", reference_selected="reference_selected",
)
INVALID_CONTROLS = dict(area_min="area_min", area_max="area_max", nominal_cd="nominal_cd", psf_fwhm="psf_fwhm",
                        nominal_x="nominal_x", nominal_y="nominal_y", hint_x="pitch_hint_x", hint_y="pitch_hint_y")


def _form(rows):
    widget = QtWidgets.QWidget()
    form = QtWidgets.QFormLayout(widget)
    form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setContentsMargins(4, 4, 4, 4)
    for label, field in rows:
        form.addRow(label, field)
    return widget


def _pair(*widgets, between=None):
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for i, item in enumerate(widgets):
        if i and between:
            layout.addWidget(QtWidgets.QLabel(between))
        layout.addWidget(item, 1)
    return widget


class SetupPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._building = True
        self.controls = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.stacked = QtWidgets.QStackedWidget()
        layout.addWidget(self.stacked, 1)
        self.steps = QtWidgets.QToolBox()
        self.stacked.addWidget(self.steps)
        self.step_keys = []
        self._build_image()
        self._build_holes()
        self._build_methods()
        self._build_pitch()
        self.align_page = self._build_align()
        self.stacked.addWidget(self.align_page)
        # Pinned under the steps: never scrolls away.
        self.run_button = QtWidgets.QPushButton()
        self.run_button.setObjectName("primary")
        self.run_button.setMinimumHeight(34)
        self.run_button.clicked.connect(self.run_current)
        layout.addWidget(self.run_button)
        self.run_reason = QtWidgets.QLabel()
        self.run_reason.setObjectName("reason")
        self.run_reason.setWordWrap(True)
        self.run_reason.hide()
        layout.addWidget(self.run_reason)
        self.steps.currentChanged.connect(self.update_state)
        self.mode = "measure"
        self._building = False
        self._recipe_changed()
        session.settings_changed.connect(self.update_state)
        session.results_changed.connect(self._results_changed)
        session.busy_changed.connect(self.update_state)
        session.moving_changed.connect(self._moving_changed)
        session.image_changed.connect(self._image_changed)
        session.registration_changed.connect(self.update_state)

    # ------------------------------------------------------------------------------------- controls
    def _combo(self, name, entries):
        widget = QtWidgets.QComboBox()
        widget.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        widget.setMinimumContentsLength(6)
        for text, value in entries:
            widget.addItem(text, value)
        widget.currentIndexChanged.connect(self._changed)
        self.controls[name] = widget
        return widget

    def _number(self, name, value, low, high, *, decimals=None, step=1, special=None):
        widget = QtWidgets.QSpinBox() if decimals is None else QtWidgets.QDoubleSpinBox()
        if decimals is not None:
            widget.setDecimals(decimals)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        widget.setMinimumWidth(56)  # wide ranges must not widen the panel; the field scrolls its text
        if special:
            widget.setSpecialValueText(special)
        widget.valueChanged.connect(self._changed)
        self.controls[name] = widget
        return widget

    def _check(self, name, text, checked=False):
        widget = QtWidgets.QCheckBox(text)
        widget.setChecked(checked)
        widget.toggled.connect(self._changed)
        self.controls[name] = widget
        return widget

    def _add_step(self, key, content):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        self.steps.addItem(scroll, f"{len(self.step_keys) + 1}  {STEP_TITLES[key]}")
        self.step_keys.append(key)

    def _build_image(self):
        self.pixel_size = self._number("pixel_size", 1, 0.000001, 1e9, decimals=6, step=0.1)
        self.pixel_size.setToolTip("Length of one pixel, in the unit beside it")
        self.unit = QtWidgets.QComboBox()
        self.unit.setEditable(True)
        self.unit.addItems(["px", "nm", "µm", "mm"])
        self.unit.setMinimumContentsLength(4)
        self.unit.setToolTip("Length unit of the pixel size and of every length shown")
        self.unit.currentTextChanged.connect(self._changed)
        self.controls["unit"] = self.unit
        self.invert = self._check("invert", "Dark holes on a bright background")
        self.backend = self._combo("backend", [("Auto (GPU when available)", "auto"), ("GPU", "gpu"), ("CPU", "cpu")])
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_form([("Pixel size", _pair(self.pixel_size, self.unit)), ("", self.invert),
                                ("Compute", self.backend)]))
        layout.addStretch()
        self._add_step("image", content)

    def _build_holes(self):
        self.seeds = self._combo("seeds", [("Separate holes", "threshold"), ("Touching / dense holes", "peaks")])
        self.seeds.setToolTip("Separate holes: threshold components; touching holes: local intensity peaks")
        self.threshold = self._combo("threshold", [("Otsu (automatic)", "otsu"), ("Triangle", "triangle")])
        self.area_min = self._number("area_min", 3, 1, 100000000)
        self.area_max = self._number("area_max", 200, 1, 100000000)
        self.pad = self._number("pad", 3, 1, 1000)
        self.refine = self._combo("refine", [("Log-quadratic", "logquad"), ("Weighted center", "weighted"),
                                             ("Edge gradient", "edge_gradmoment"), ("Automatic", "auto"),
                                             ("None", "none")])
        self.refine.setToolTip("Center refinement of Preview holes only; CD methods find their own centers")
        self.area_plot = pg.PlotWidget(background=BACKGROUND)
        self.area_plot.setFixedHeight(110)
        self.area_plot.setMouseEnabled(x=False, y=False)
        self.area_plot.hideButtons()
        self.area_plot.setLabel("bottom", "component area (px²)")
        self.area_plot.getAxis("left").setStyle(showValues=False)
        self.area_plot.getPlotItem().getAxis("bottom").enableAutoSIPrefix(False)
        self.area_bars = pg.BarGraphItem(x=[], height=[], width=1, brush="#4d6f8f", pen=None)
        self.area_plot.addItem(self.area_bars)
        self.area_lines = []
        for widget in (self.area_min, self.area_max):
            line = pg.InfiniteLine(widget.value(), movable=True, pen=pg.mkPen("#ffce73", width=2))
            line.sigPositionChangeFinished.connect(lambda line, widget=widget: widget.setValue(int(round(line.value()))))
            self.area_plot.addItem(line)
            self.area_lines.append(line)
        self.area_note = hint("Preview holes to see the component areas against the limits.")
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_form([("Pattern", self.seeds), ("Threshold", self.threshold),
                                ("Area (px²)", _pair(self.area_min, self.area_max, between="to")),
                                ("Padding (px)", self.pad), ("Preview centers", self.refine)]))
        layout.addWidget(self.area_plot)
        layout.addWidget(self.area_note)
        layout.addWidget(hint("Preview holes finds the holes only (fast). Measure CD uses these settings too."))
        layout.addStretch()
        self._add_step("holes", content)

    def _build_methods(self):
        self.recipe = self._combo("recipe", [(label, key) for key, (label, methods) in RECIPES.items()]
                                  + [("Custom", "custom")])
        self.recipe.currentIndexChanged.connect(self._recipe_changed)
        buttons = QtWidgets.QHBoxLayout()
        for text, state in (("All", True), ("None", False)):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(f"Tick {text.lower()} of the methods")
            button.clicked.connect(lambda checked=False, state=state: self._tick_all(state))
            buttons.addWidget(button)
        buttons.addStretch()
        self.method_checks, self.group_checks = {}, []
        methods = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout(methods)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(1)
        listed = set()
        groups = list(METHOD_GROUPS) + [("Other", tuple(m for m in METHOD_LABELS
                                                        if m not in {k for _, ks in METHOD_GROUPS for k in ks}))]
        for title, names in groups:
            names = [name for name in names if name in METHOD_LABELS and name not in listed]
            if not names:
                continue
            listed |= set(names)
            group = QtWidgets.QCheckBox(title)
            group.setTristate(True)
            group.setStyleSheet("font-weight: 600; color: #90b2d2;")
            group.clicked.connect(lambda checked=False, names=names, group=group: self._tick_group(names, group))
            list_layout.addWidget(group)
            self.group_checks.append((group, names))
            for name in names:
                check = QtWidgets.QCheckBox(method_name(name))
                check.setToolTip(f"{name}: {METHOD_LABELS[name]}. {method_help(name)}")
                check.toggled.connect(self._methods_ticked)
                line = hint(method_line(name))
                line.setStyleSheet("font-size: 11px; padding-left: 22px;")
                line.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.NoTextInteraction)
                item = QtWidgets.QWidget()
                item_layout = QtWidgets.QVBoxLayout(item)
                item_layout.setContentsMargins(16, 0, 0, 2)
                item_layout.setSpacing(0)
                item_layout.addWidget(check)
                item_layout.addWidget(line)
                list_layout.addWidget(item)
                self.method_checks[name] = check
        self.background = self._combo("background", [("Gaussian neighbour model", "model"),
                                                     ("Cell border (legacy)", "border")])
        self.n_angles = self._number("n_angles", 96, 16, 720, step=16)
        self.radial_step = self._number("radial_step", 0.1, 0.01, 1, decimals=2, step=0.05)
        self.smooth_sigma = self._number("smooth_sigma", 0, 0, 10, decimals=2, step=0.1)
        self.fit_fraction = self._number("fit_fraction", 0.2, 0.01, 0.95, decimals=2, step=0.05)
        self.half_level = self._number("half_level", 0.5, 0.01, 0.99, decimals=2, step=0.05)
        self.peak_pixels = self._number("peak_pixels", 4, 1, 100, decimals=1)
        self.tail_level = self._number("tail_level", 0.2, 0.01, 0.99, decimals=2, step=0.05)
        self.tail_window = self._number("tail_window", 1.2, 0.5, 2, decimals=2, step=0.1)
        self.iterations = self._number("iterations", 30, 1, 1000)
        self.tolerance = self._number("tolerance", 1e-3, 1e-6, 1, decimals=6, step=1e-4)
        self.tolerance.setToolTip("Deblending stops when the fit changes less than this per round "
                                  "(or at the iteration limit); the batch's --deblend-tolerance")
        self.dark_level = self._number("dark_level", 0, -1e9, 1e9, decimals=3)
        self.nominal_cd = self._number("nominal_cd", 0, 0, 1e9, decimals=4, special="Not specified")
        self.psf_fwhm = self._number("psf_fwhm", 0, 0, 1e9, decimals=4, special="Not specified")
        self.calibration = QtWidgets.QGroupBox("Calibration (length unit)")
        calibration = QtWidgets.QFormLayout(self.calibration)
        calibration.addRow("Nominal CD", self.nominal_cd)
        calibration.addRow("PSF FWHM", self.psf_fwhm)
        nonlocal_methods = set(METHOD_LABELS) - {name for name in METHOD_LABELS if name.startswith("local_")}
        groups = [
            ("Neighbour subtraction", [("Background", self.background), ("Fit iterations", self.iterations),
                                       ("Stop tolerance (px)", self.tolerance)], nonlocal_methods),
            ("Radial edge sampling", [("Rays", self.n_angles), ("Step (px)", self.radial_step),
                                      ("Smoothing (px)", self.smooth_sigma)], {"gradient", "halfmax"}),
            ("Log-quadratic fit", [("Fit fraction", self.fit_fraction)], {"logquad"}),
            ("Half-height area", [("Level", self.half_level)], {name for name in METHOD_LABELS if "half_area" in name}),
            ("Peak averaging", [("Peak area (px²)", self.peak_pixels)],
             {"half_area_min_avg", "local_half_area_min_avg", "tail_area", "tail_area_norm", "local_tail_area"}),
            ("Tail threshold", [("Level", self.tail_level)], {"tail_area", "tail_area_norm", "local_tail_area"}),
            ("Wide measurement window", [("Window (pitch)", self.tail_window)],
             {"tail_area", "tail_area_norm", "edge_area", "foot_area", "foot_area_ratio"}),
            ("Detector calibration", [("Dark level", self.dark_level)], {"flux_ratio"}),
        ]
        parameters = QtWidgets.QWidget()
        parameter_layout = QtWidgets.QVBoxLayout(parameters)
        parameter_layout.setContentsMargins(0, 0, 0, 0)
        self.method_groups = []
        for title, rows, names in groups:
            box = QtWidgets.QGroupBox(title)
            inner = QtWidgets.QVBoxLayout(box)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.addWidget(_form(rows))
            parameter_layout.addWidget(box)
            self.method_groups.append((box, set(names)))
        self.parameters_fold = Fold("Parameters of the ticked methods", parameters)
        self.methods_summary = hint()
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(_form([("Preset", self.recipe)]))
        layout.addLayout(buttons)
        layout.addWidget(methods)
        layout.addWidget(self.methods_summary)
        layout.addWidget(self.calibration)
        layout.addWidget(self.parameters_fold)
        layout.addStretch()
        self._add_step("cd", content)

    def _build_pitch(self):
        self.pitch_source = self._combo("pitch_source", [("CD measurement", "cd"), ("Previewed holes", "detection")])
        self.pitch_source.setToolTip("Pitch uses these centers of the active page; it never measures again")
        self.pattern_layout = self._combo("pattern_layout", [("Automatic", "auto"), ("Square / rectangular", "square"),
                                                             ("Staggered", "staggered")])
        self.average_rows = self._number("average_rows", 0, 0, 1000000, special="All")
        self.average_columns = self._number("average_columns", 0, 0, 1000000, special="All")
        self.nominal_x = self._number("nominal_x", 0, 0, 1e9, decimals=5, special="Measured mean")
        self.nominal_y = self._number("nominal_y", 0, 0, 1e9, decimals=5, special="Measured mean")
        self.reference_selected = self._check("reference_selected", "Reference at the selected hole")
        self.stagger_axis = self._combo("stagger_axis", [("Offset rows", "row"), ("Offset columns", "column")])
        self.pitch_hint_x = self._number("pitch_hint_x", 0, 0, 1e9, decimals=3, special="Automatic")
        self.pitch_hint_y = self._number("pitch_hint_y", 0, 0, 1e9, decimals=3, special="Automatic")
        advanced = _form([("Stagger direction", self.stagger_axis), ("Pitch hint x (px)", self.pitch_hint_x),
                          ("Pitch hint y (px)", self.pitch_hint_y)])
        self.pitch_advanced = Fold("Advanced: indexing hints", advanced)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_form([("Centers", self.pitch_source), ("Lattice", self.pattern_layout),
                                ("Average rows", self.average_rows), ("Average columns", self.average_columns),
                                ("Design pitch x", self.nominal_x), ("Design pitch y", self.nominal_y),
                                ("", self.reference_selected)]))
        layout.addWidget(self.pitch_advanced)
        layout.addStretch()
        self._add_step("pitch", content)

    def _build_align(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QtWidgets.QLabel("Align a moving page to the active page")
        title.setStyleSheet("font-weight: 600; color: #90b2d2;")
        layout.addWidget(title)
        buttons = QtWidgets.QHBoxLayout()
        self.moving_this_file = QtWidgets.QPushButton("This file")
        self.moving_this_file.setToolTip("List the pages of the open file")
        self.moving_this_file.clicked.connect(self._moving_this_file)
        self.moving_other = QtWidgets.QPushButton("Other file...")
        self.moving_other.setToolTip("Open another file and list its pages")
        buttons.addWidget(self.moving_this_file)
        buttons.addWidget(self.moving_other)
        layout.addLayout(buttons)
        self.moving_label = hint("Pick a page: it is read and aligned to the active page.")
        layout.addWidget(self.moving_label)
        self.moving_list = QtWidgets.QListWidget()
        self.moving_list.setToolTip("Click a page to align it to the active page")
        self.moving_list.itemActivated.connect(self._moving_picked)
        self.moving_list.itemClicked.connect(self._moving_picked)
        layout.addWidget(self.moving_list, 1)
        self.upsample = self._number("upsample", 20, 1, 1000)
        self.upsample.setToolTip("Subpixel factor of the phase correlation (20 = 1/20 px)")
        self.hole_check_mode = self._combo("hole_check_mode", [("Centered (batch default)", "centered"),
                                                               ("Raw", "raw")])
        self.hole_check_mode.setToolTip("The batch's hole check: 'centered' removes each image's mean first "
                                        "(passes below 0.8), 'raw' does not (passes below 0.3)")
        layout.addWidget(_form([("Subpixel factor", self.upsample), ("Hole check", self.hole_check_mode)]))
        self._list_source = None  # the file listed for the moving page; None: the moving image's file
        return page

    # -------------------------------------------------------------------------------------- methods
    def checked_methods(self):
        return tuple(name for name, check in self.method_checks.items() if check.isChecked())

    def _set_methods(self, names):
        for name, check in self.method_checks.items():
            with QtCore.QSignalBlocker(check):
                check.setChecked(name in names)

    def _tick_all(self, state):
        self._set_methods(self.method_checks if state else ())
        self._methods_ticked()

    def _tick_group(self, names, group):
        state = not all(self.method_checks[name].isChecked() for name in names)
        for name in names:
            with QtCore.QSignalBlocker(self.method_checks[name]):
                self.method_checks[name].setChecked(state)
        self._methods_ticked()

    def _recipe_changed(self, *args):
        recipe = RECIPES.get(self.recipe.currentData())
        if recipe is not None:
            self._set_methods(recipe[1])
            with QtCore.QSignalBlocker(self.background):
                self.background.setCurrentIndex(self.background.findData("model"))
        self._update_methods()
        self._changed()

    def _methods_ticked(self, *args):
        names = self.checked_methods()
        recipe = next((key for key, (label, methods) in RECIPES.items() if tuple(methods) == names), "custom")
        with QtCore.QSignalBlocker(self.recipe):
            self.recipe.setCurrentIndex(self.recipe.findData(recipe))
        self._update_methods()
        self._changed()

    def _update_methods(self):
        names = self.checked_methods()
        selected = set(names)
        for group, members in self.group_checks:
            ticked = sum(name in selected for name in members)
            state = (QtCore.Qt.CheckState.Checked if ticked == len(members) else
                     QtCore.Qt.CheckState.PartiallyChecked if ticked else QtCore.Qt.CheckState.Unchecked)
            with QtCore.QSignalBlocker(group):
                group.setCheckState(state)
        needs_model = bool(selected - {"logquad", "gradient", "halfmax"} -
                           {name for name in METHOD_LABELS if name.startswith("local_")}) or self.seeds.currentData() == "peaks"
        border = self.background.findData("border")
        self.background.model().item(border).setEnabled(not needs_model)
        if needs_model and self.background.currentData() != "model":
            with QtCore.QSignalBlocker(self.background):
                self.background.setCurrentIndex(self.background.findData("model"))
        for widget, methods in self.method_groups:
            widget.setVisible(bool(selected & methods))
        self.calibration.setVisible(bool(selected & CALIBRATED_METHODS))
        self.methods_summary.setText(f"{len(names)} method{'s' * (len(names) != 1)} ticked." if names else
                                     "Tick at least one method.")

    # ------------------------------------------------------------------------------------- values
    def values(self):
        """Every persistent setting by name (saved between sessions, reset by Reset settings)."""
        values = {}
        for name, widget in self.controls.items():
            if name == "unit":
                values[name] = widget.currentText()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentData()
            elif isinstance(widget, QtWidgets.QAbstractButton):
                values[name] = widget.isChecked()
            else:
                values[name] = widget.value()
        values["methods"] = list(self.checked_methods())
        values["parameters_open"] = self.parameters_fold.is_open()
        values["step"] = self.steps.currentIndex()
        return values

    def apply_values(self, values):
        building, self._building = self._building, True
        try:
            for name, widget in self.controls.items():
                if name not in values or values[name] is None:
                    continue
                value = values[name]
                with QtCore.QSignalBlocker(widget):
                    if name == "unit":
                        widget.setCurrentText(str(value))
                    elif isinstance(widget, QtWidgets.QComboBox):
                        if widget.findData(value) >= 0:
                            widget.setCurrentIndex(widget.findData(value))
                    elif isinstance(widget, QtWidgets.QAbstractButton):
                        widget.setChecked(bool(value))
                    else:
                        widget.setValue(type(widget.value())(value))
            if "methods" in values:
                self._set_methods(values["methods"])
            if "parameters_open" in values:
                self.parameters_fold.set_open(bool(values["parameters_open"]))
            if isinstance(values.get("step"), int) and 0 <= values["step"] < self.steps.count():
                self.steps.setCurrentIndex(values["step"])
            for line, widget in zip(self.area_lines, (self.area_min, self.area_max)):
                line.setValue(widget.value())
            self._update_methods()
        finally:
            self._building = building
        self._changed()

    def apply_parameters(self, parameters):
        """Put a result's parameters back into the controls (Revert)."""
        values = {control: parameters[key] for key, control in PARAMETER_CONTROLS.items() if key in parameters}
        for key in ("nominal_cd", "psf_fwhm"):
            if key in values:
                values[key] = values[key] or 0
        options = parameters.get("pitch_options") or {}
        if options:
            values.update(pattern_layout=options.get("layout"), stagger_axis=options.get("stagger_axis"),
                          average_rows=options.get("average_rows") or 0,
                          average_columns=options.get("average_columns") or 0)
            for key, (x, y) in (("nominal", options.get("nominal_pitch_xy") or (0, 0)),
                                ("pitch_hint", options.get("pitch_hint_xy") or (0, 0))):
                values[f"{key}_x"], values[f"{key}_y"] = x, y
        if "methods" in parameters:
            values["methods"] = list(parameters["methods"])
            values["recipe"] = next((key for key, (label, methods) in RECIPES.items()
                                     if tuple(methods) == tuple(parameters["methods"])), "custom")
        self.apply_values(values)

    def parameters(self):
        """The analysis parameters of the current settings (jobs never read widgets)."""
        nominal = [self.nominal_x.value(), self.nominal_y.value()]
        hints = [self.pitch_hint_x.value(), self.pitch_hint_y.value()]
        return dict(
            threshold=self.threshold.currentData(), invert=self.invert.isChecked(),
            area_min=self.area_min.value(), area_max=self.area_max.value(), pad=self.pad.value(),
            refine=self.refine.currentData(), backend=self.backend.currentData(), methods=self.checked_methods(),
            background=self.background.currentData(), seeds=self.seeds.currentData(),
            pixel_size=self.pixel_size.value(), unit=self.unit.currentText().strip() or "px",
            n_angles=self.n_angles.value(), radial_step=self.radial_step.value(), smooth_sigma=self.smooth_sigma.value(),
            fit_fraction=self.fit_fraction.value(), half_area_level=self.half_level.value(),
            half_area_peak_pixels=self.peak_pixels.value(), tail_area_level=self.tail_level.value(),
            tail_window=self.tail_window.value(), deblend_iterations=self.iterations.value(),
            deblend_tolerance=self.tolerance.value(),
            nominal_cd=self.nominal_cd.value() or None, psf_fwhm=self.psf_fwhm.value() or None,
            dark_level=self.dark_level.value(), pitch_source=self.pitch_source.currentData(),
            reference_selected=self.reference_selected.isChecked(),
            pitch_options=dict(layout=self.pattern_layout.currentData(), stagger_axis=self.stagger_axis.currentData(),
                               average_rows=self.average_rows.value() or None,
                               average_columns=self.average_columns.value() or None, reference_xy=None,
                               nominal_pitch_xy=nominal if nominal[0] and nominal[1] else None,
                               pitch_hint_xy=hints if hints[0] and hints[1] else None),
            nominal_x=nominal[0], nominal_y=nominal[1], hint_x=hints[0], hint_y=hints[1],
            upsample=self.upsample.value(), hole_check_mode=self.hole_check_mode.currentData(),
        )

    def _changed(self, *args):
        if self._building:
            return
        for line, widget in zip(self.area_lines, (self.area_min, self.area_max)):
            if line.value() != widget.value():
                line.setValue(widget.value())
        if self.sender() is self.seeds:
            self._update_methods()
        self.session.set_parameters(self.parameters())

    # ---------------------------------------------------------------------------------------- state
    def set_mode(self, mode):
        self.mode = mode
        self.stacked.setCurrentWidget(self.align_page if mode == "align" else self.steps)
        if mode == "align":
            self._moving_changed()
        self.update_state()

    def current_step(self):
        return "align" if self.mode == "align" else self.step_keys[self.steps.currentIndex()]

    def run_current(self):
        operation = RUNS[self.current_step()][1]
        if operation == "register":
            self.session.run_align()
        else:
            self.session.run(operation)

    def run_step(self, operation):
        """Run 'detect', 'cd', 'pitch' or 'register' whichever step is open."""
        self.session.run_align() if operation == "register" else self.session.run(operation)

    def update_state(self, *args):
        session = self.session
        problems, invalid = session.problems()
        text, operation = RUNS[self.current_step()]
        problem = problems[operation]
        busy = session.busy
        self.run_button.setText(text)
        self.run_button.setEnabled(not busy and not problem)
        self.run_button.setToolTip(problem or f"{text} on the active page (Ctrl+Enter)")
        self.run_reason.setText(problem)
        self.run_reason.setVisible(bool(problem) and not busy)
        for key, control in INVALID_CONTROLS.items():
            widget = self.controls[control]
            bad = key in invalid
            if widget.property("invalid") != bad:
                widget.setProperty("invalid", bad)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
        self.stacked.setEnabled(not busy)
        states = session.step_states()
        for index, key in enumerate(self.step_keys):
            color, word = MARKS[states[key]]
            self.steps.setItemIcon(index, dot_icon(color))
            self.steps.setItemText(index, f"{index + 1}  {STEP_TITLES[key]}" + (" · changed" if states[key] == "changed" else ""))
            self.steps.setItemToolTip(index, f"{STEP_TITLES[key]}: {word}")
        self.moving_this_file.setEnabled(not busy and session.source is not None)
        self.moving_other.setEnabled(not busy and session.loaded is not None)

    def _results_changed(self):
        """The area histogram of the page's components against the area limits."""
        results = self.session.page_results
        areas = None
        if results is not None and results.detect is not None:
            areas = results.detect.detection.boxes[:, 4]
        elif results is not None and results.cd is not None and results.cd.cd.component_labels is not None:
            labels = results.cd.cd.component_labels
            areas = np.bincount(labels[labels > 0].ravel())[1:] if labels.max() > 0 else np.empty(0)
            areas = areas[areas > 0]
        if areas is None or not len(areas):
            self.area_bars.setOpts(x=[], height=[])
            self.area_note.setText("Preview holes to see the component areas against the limits.")
        else:
            top = max(float(np.percentile(areas, 99.5)), self.area_max.value() * 1.3, 10)
            edges = np.linspace(0, top, 61)
            counts, _ = np.histogram(np.clip(areas, 0, top), edges)
            self.area_bars.setOpts(x=(edges[:-1] + edges[1:]) / 2, height=counts, width=edges[1] - edges[0])
            self.area_plot.setXRange(0, top, padding=0.02)
            inside = (areas >= self.area_min.value()) & (areas <= self.area_max.value())
            self.area_note.setText(f"{len(areas):,} components: {inside.sum():,} inside the limits, "
                                   f"{(areas < self.area_min.value()).sum():,} smaller, "
                                   f"{(areas > self.area_max.value()).sum():,} larger. Drag a line to set a limit.")
        self.update_state()

    # --------------------------------------------------------------------------------------- moving
    def _image_changed(self, new_file):
        if new_file:
            self._list_source = None
        self._moving_changed()

    def _moving_this_file(self):
        self._list_source = self.session.source
        self._moving_changed()

    def _moving_changed(self):
        session = self.session
        source = self._list_source or session.moving_file()
        self.moving_list.clear()
        moving = session.moving
        if moving is not None:
            self.moving_label.setText(f"Moving: {moving.name}, page {moving.page}" +
                                      (f"\n{moving.conversion}" if moving.conversion else ""))
        else:
            self.moving_label.setText("Pick a page: it is read and aligned to the active page.")
        if source is None:
            return
        for page, (h, w) in enumerate(source.shapes):
            item = QtWidgets.QListWidgetItem(f"Page {page}   {w} x {h} px")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, (source.path, page))
            if moving is not None and moving.path == source.path and moving.page == page:
                item.setIcon(dot_icon("#56d6cf"))
            self.moving_list.addItem(item)
        self.update_state()

    def _moving_picked(self, item):
        path, page = item.data(QtCore.Qt.ItemDataRole.UserRole)
        session = self.session
        if session.busy:
            return
        own = session.source is not None and path == session.source.path
        self._list_source = None
        session.open_moving(None if own else path, page, align=True)
