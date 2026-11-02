"""The main window: menus, the header line, the docks around the image, banners and the status bar, all wired
to one Session. Run ``python -m subpx.workbench [image]`` or call :func:`launch_workbench`."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
import weakref

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .._workbench_data import HOLE_CHECK_MAX_ERROR
from .._workbench_pages import PAGES_PER_GPU, devices
from .canvas import MeasureView
from .names import RECIPES, method_name
from .panels.align import AlignPanel, AlignView
from .panels.all_pages import AllPagesPanel
from .panels.average import AveragePanel
from .panels.distribution import DistributionPanel
from .panels.holes_table import HolesPanel
from .panels.inspector import HoleInspector
from .panels.log import LogPanel
from .panels.pages import PagesPanel
from .panels.pitch import PitchPanel
from .panels.setup import SetupPanel
from .session import Session, failure_counts
from .settings import preprocess_text
from .settings import SETTINGS_KEY, describe
from .widgets import apply_style, hint

IMAGE_FILTER = "Images (*.tif *.tiff *.png *.jpg *.jpeg *.bmp *.npy);;All files (*)"
RECENT_FILES = 8
GUIDE = """
<h3>Workflow</h3>
<p><b>Setup</b> (left) holds the steps in pipeline order: Image, Preprocess, Holes, CD methods, Pitch. Each step
shows a mark
(grey: not run, green: up to date, amber: changed since the result) and the Run button of the open step stays
under them. <b>Pages</b> (the tab next to Setup) lists the file's pages, counted from 0 like the batch's
pages.csv; <b>Stack vertically</b> shows them all.</p>
<p>The image shows the result of the method in the Method box: the previewed holes or a CD method. The
<b>hole inspector</b> (right) compares the selected hole in every method; the bottom tabs hold the holes
table, the CD distribution, the pitch and the log. When settings change after a run, a banner offers
<b>Re-run</b> and <b>Revert</b>.</p>
<p><b>Preprocessed</b> (above the image, P) flips between the raw page and the preprocessed page (Setup >
Preprocess) in the same place, with the same contrast. Each version keeps its own results (holes, CD, pitch); runs
measure the version on screen. The inspector, the CD distribution and the summary line also show the other
version's CD. <b>Show > Removed by preprocessing</b> maps raw minus preprocessed; the <b>Average hole</b> tab
stacks every hole on a 4x finer grid, raw and preprocessed. The stack's other pages and Align use raw pages.</p>
<p><b>Measure all pages</b> (Pages tab or Run menu) measures CD and pitch on every page of the file with the
current settings, on the GPU in the batch's fast mode (the same CDs as Measure CD), beside the window: pages can be
browsed meanwhile. Pages failing the hole check against the page on screen are skipped. Each page's means are in the
<b>All pages</b> tab; pages.csv, one hole CSV per page and run.json go to a folder.</p>
<p><b>Align</b> (header): pick a moving page of this or another file; it is aligned to the active page and
shown on the image (Space flickers).</p>
<h3>Keys</h3>
<table cellpadding="3">
<tr><td>Ctrl+O, drop a file</td><td>open</td></tr>
<tr><td>Ctrl+Enter</td><td>run the open Setup step</td></tr>
<tr><td>P</td><td>raw / preprocessed page (and their results)</td></tr>
<tr><td>F / Shift+F / Z</td><td>fit the page / fit the whole stack / zoom to the selected hole</td></tr>
<tr><td>J / K, Down / Up in the table</td><td>next / previous hole in table order</td></tr>
<tr><td>N / Shift+N</td><td>next / previous failed hole of the current method</td></tr>
<tr><td>Page Down / Page Up</td><td>next / previous page (image or page list)</td></tr>
<tr><td>Esc</td><td>cancel the running job (its result is discarded), else stop Measure all pages</td></tr>
<tr><td>Ctrl+C</td><td>copy the selected rows, or the selected hole's row</td></tr>
<tr><td>Space</td><td>flicker reference / aligned (Align view)</td></tr>
<tr><td>Ctrl+1 / Ctrl+2 / Ctrl+3</td><td>show or hide Setup / Inspector / bottom panel</td></tr>
</table>
<p>Mouse: click a hole to select it, wheel to zoom, drag to pan.</p>
"""

_WINDOWS = weakref.WeakSet()
_previous_hook = None
_GPU = None


def _exception_hook(kind, value, tb):
    """Errors raised in click or redraw handlers reach the open windows' error banner and log."""
    detail = "".join(traceback.format_exception(kind, value, tb))
    for window in list(_WINDOWS):
        try:
            window.session.fail("Unexpected error", detail)
        except Exception:
            pass
    (_previous_hook or sys.__excepthook__)(kind, value, tb)


def _install_exception_hook():
    global _previous_hook
    if sys.excepthook is not _exception_hook:
        _previous_hook, sys.excepthook = sys.excepthook, _exception_hook


def gpu_available():
    global _GPU
    if _GPU is None:
        try:
            from ..backends import is_gpu_available

            _GPU = bool(is_gpu_available())
        except Exception:
            _GPU = False
    return _GPU


class InspectionWindow(QtWidgets.QMainWindow):
    """Desktop workbench for inspecting the repository's measurements on sample images."""

    def __init__(self, image=None, *, title="subpx | Inspection workbench", settings=None):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1440, 900)
        self.setMinimumSize(1024, 680)
        self.setAcceptDrops(True)
        # Settings and window layout persist here; tests pass their own QSettings file.
        self._settings = settings if settings is not None else QtCore.QSettings("subpx", "workbench")
        self.ask_before_discard = True
        self.mode = "measure"
        self.session = session = Session(self)
        self._build_central()
        self._build_docks()
        self._build_header()
        self._build_status()
        self._build_menus()
        apply_style(self)
        self._defaults = self._values()
        self._load_settings()
        session.set_parameters(self.setup.parameters())
        session.error.connect(self._show_error)
        session.status.connect(self._show_status)
        session.busy_changed.connect(self._busy_changed)
        session.jobs.progress.connect(self.job_label.setText)
        session.image_changed.connect(self._image_changed)
        session.results_changed.connect(self._update_summary)
        session.method_changed.connect(self._update_summary)
        session.settings_changed.connect(self._update_stale)
        session.results_changed.connect(self._update_stale)
        session.settings_changed.connect(self._update_compute)
        session.pages_progress.connect(self._update_pages_progress)
        session.pages_changed.connect(self._update_pages_progress)
        session.pages_changed.connect(lambda: self._busy_changed(self.busy))
        self._busy_changed(False)
        self._update_stale()
        _install_exception_hook()
        _WINDOWS.add(self)
        QtCore.QTimer.singleShot(0, self._update_compute)
        if image is not None:
            self.set_image(image)

    # ----------------------------------------------------------------------------------------- build
    def _build_central(self):
        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)
        self.error_banner = QtWidgets.QFrame()
        self.error_banner.setObjectName("banner")
        banner = QtWidgets.QHBoxLayout(self.error_banner)
        banner.setContentsMargins(8, 3, 4, 3)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setObjectName("bannertext")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        banner.addWidget(self.error_label, 1)
        details = QtWidgets.QToolButton()
        details.setText("Details")
        details.setToolTip("Show the full message in the Log tab")
        details.clicked.connect(self.show_log)
        banner.addWidget(details)
        dismiss = QtWidgets.QToolButton()
        dismiss.setText("Dismiss")
        dismiss.clicked.connect(self._clear_error)
        banner.addWidget(dismiss)
        self.error_banner.hide()
        layout.addWidget(self.error_banner)
        self.stale_banner = QtWidgets.QFrame()
        self.stale_banner.setObjectName("stale")
        stale = QtWidgets.QHBoxLayout(self.stale_banner)
        stale.setContentsMargins(8, 3, 4, 3)
        self.stale_label = QtWidgets.QLabel()
        self.stale_label.setObjectName("bannertext")
        self.stale_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        stale.addWidget(self.stale_label, 1)
        self.rerun_button = QtWidgets.QPushButton("Re-run")
        self.rerun_button.setToolTip("Run the out-of-date step again with the current settings")
        self.rerun_button.clicked.connect(self.session.rerun)
        stale.addWidget(self.rerun_button)
        self.revert_button = QtWidgets.QPushButton("Revert")
        self.revert_button.setToolTip("Put back the settings the results on screen were made with")
        self.revert_button.clicked.connect(self.revert_settings)
        stale.addWidget(self.revert_button)
        self.stale_banner.hide()
        layout.addWidget(self.stale_banner)
        self.views = QtWidgets.QStackedWidget()
        self.measure = MeasureView(self.session)
        self.align_view = AlignView(self.session)
        self.views.addWidget(self.measure)
        self.views.addWidget(self.align_view)
        layout.addWidget(self.views, 1)
        self.setCentralWidget(root)

    def _dock(self, title, name, widget, area):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    def _build_docks(self):
        session = self.session
        Area = QtCore.Qt.DockWidgetArea
        # The bottom panel sits under the image only; Setup and the inspector run full height.
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, Area.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, Area.RightDockWidgetArea)
        self.setup = SetupPanel(session)
        self.pages = PagesPanel(session)
        self.left_tabs = QtWidgets.QTabWidget()
        self.left_tabs.addTab(self.setup, "Setup")
        self.left_tabs.addTab(self.pages, "Pages")
        self.setup_dock = self._dock("Setup", "setup", self.left_tabs, Area.LeftDockWidgetArea)
        self.inspector = HoleInspector(session)
        self.align_panel = AlignPanel(session)
        self.right = QtWidgets.QStackedWidget()
        self.right.addWidget(self.inspector)
        self.right.addWidget(self.align_panel)
        self.inspector_dock = self._dock("Hole inspector", "inspector", self.right, Area.RightDockWidgetArea)
        self.holes = HolesPanel(session)
        self.distribution = DistributionPanel(session)
        self.pitch = PitchPanel(session)
        self.average = AveragePanel(session)
        self.all_pages = AllPagesPanel(session)
        self.log = LogPanel(session)
        self.bottom = QtWidgets.QTabWidget()
        for widget, title in ((self.holes, "Holes"), (self.distribution, "CD distribution"), (self.pitch, "Pitch"),
                              (self.average, "Average hole"), (self.all_pages, "All pages"), (self.log, "Log")):
            self.bottom.addTab(widget, title)
        self.bottom_dock = self._dock("Results", "bottom", self.bottom, Area.BottomDockWidgetArea)
        self.resizeDocks([self.setup_dock, self.inspector_dock], [280, 380], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.bottom_dock], [260], QtCore.Qt.Orientation.Vertical)

    def _build_header(self):
        bar = self.addToolBar("Header")
        bar.setObjectName("header")
        bar.setMovable(False)
        bar.setFloatable(False)
        self.view_buttons = QtWidgets.QButtonGroup(self)
        for i, (text, tip) in enumerate((("Measure", "Measure holes on the active page"),
                                         ("Align", "Align a moving page to the active page"))):
            button = QtWidgets.QToolButton()
            button.setObjectName("view")
            button.setText(text)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.view_buttons.addButton(button, i)
            bar.addWidget(button)
        self.view_buttons.idToggled.connect(lambda i, checked: checked and self.set_mode(("measure", "align")[i]))
        bar.addSeparator()
        self.source_label = QtWidgets.QLabel("Open an image (Ctrl+O), drop a file here, or File > Load demo")
        self.source_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label.setMinimumWidth(160)
        self.source_label.setMaximumWidth(420)
        bar.addWidget(self.source_label)
        self.previous_page = QtWidgets.QToolButton()
        self.previous_page.setText("◀")
        self.previous_page.setToolTip("Previous page (Page Up on the image or the page list)")
        self.previous_page.clicked.connect(lambda: self._page_step(-1))
        self.page_spin = QtWidgets.QSpinBox()
        self.page_spin.setRange(0, 0)
        self.page_spin.setKeyboardTracking(False)
        self.page_spin.setPrefix("Page ")
        self.page_spin.setToolTip("Page number, 0 = first (as in the batch's pages.csv and --pages)")
        self.page_spin.valueChanged.connect(self.session.set_page)
        self.page_count = QtWidgets.QLabel("")
        self.next_page = QtWidgets.QToolButton()
        self.next_page.setText("▶")
        self.next_page.setToolTip("Next page (Page Down on the image or the page list)")
        self.next_page.clicked.connect(lambda: self._page_step(1))
        self.page_widgets = [bar.addWidget(widget) for widget in
                             (self.previous_page, self.page_spin, self.page_count, self.next_page)]
        bar.addSeparator()
        self.image_label = QtWidgets.QLabel("")
        self.image_label.setObjectName("hint")
        bar.addWidget(self.image_label)
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        self.compute_label = QtWidgets.QLabel("")
        self.compute_label.setObjectName("hint")
        bar.addWidget(self.compute_label)
        for action in self.page_widgets:
            action.setVisible(False)

    def _build_status(self):
        status = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(status)
        layout.setContentsMargins(6, 0, 6, 0)
        self.message_label = QtWidgets.QLabel("Ready")
        self.message_label.setObjectName("status")
        self.message_label.setMinimumWidth(160)
        self.message_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.message_label, 2)
        self.hover_label = QtWidgets.QLabel("Wheel: zoom | drag: pan | click a hole")
        self.hover_label.setObjectName("hint")
        self.hover_label.setMinimumWidth(240)
        self.measure.hovered.connect(self.hover_label.setText)
        layout.addWidget(self.hover_label)
        self.job_label = QtWidgets.QLabel("")
        layout.addWidget(self.job_label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(110)
        self.progress.setMaximumHeight(14)
        layout.addWidget(self.progress)
        self.cancel_button = QtWidgets.QToolButton()
        self.cancel_button.setText("Cancel")
        self.cancel_button.setToolTip("Stop waiting for the job and discard its result (Esc)")
        self.cancel_button.clicked.connect(self.session.cancel)
        layout.addWidget(self.cancel_button)
        self.pages_label = QtWidgets.QLabel("")
        self.pages_label.setToolTip("Measure all pages: pages done, speed and time left (All pages tab for results)")
        layout.addWidget(self.pages_label)
        self.pages_bar = QtWidgets.QProgressBar()
        self.pages_bar.setMaximumWidth(110)
        self.pages_bar.setMaximumHeight(14)
        self.pages_bar.setTextVisible(False)
        layout.addWidget(self.pages_bar)
        self.pages_stop = QtWidgets.QToolButton()
        self.pages_stop.setText("Stop")
        self.pages_stop.setToolTip("Stop Measure all pages: the pages being measured are dropped, the rest stays (Esc)")
        self.pages_stop.clicked.connect(self.session.stop_all_pages)
        layout.addWidget(self.pages_stop)
        for widget in (self.pages_label, self.pages_bar, self.pages_stop):
            widget.hide()
        self.summary_label = QtWidgets.QLabel("")
        self.summary_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.summary_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.summary_label, 5)
        self.summary_label.hide()
        self.statusBar().addPermanentWidget(status, 1)

    def _action(self, menu, text, slot=None, shortcut=None, tip=None, checkable=False):
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcuts([QtGui.QKeySequence(key) for key in
                                 (shortcut if isinstance(shortcut, (list, tuple)) else [shortcut])])
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        if slot is not None:
            (action.toggled if checkable else action.triggered).connect(slot)
        menu.addAction(action)
        return action

    def _build_menus(self):
        session, menus = self.session, self.menuBar()
        file = menus.addMenu("&File")
        self.open_action = self._action(file, "&Open...", self.open_image, QtGui.QKeySequence.StandardKey.Open,
                                        "Open an image or a multipage TIFF (Ctrl+O); or drop a file on the window")
        self.recent_menu = file.addMenu("Open &recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        self.demo_action = self._action(file, "Load &demo", self.load_demo, tip="A synthetic staggered hole pattern")
        file.addSeparator()
        self.export_action = self._action(file, "&Export...", self.export_results, "Ctrl+E",
                                          "Save the holes table, the result arrays and the alignment (Ctrl+E)")
        self._action(file, "Save &screenshot...", self.save_screenshot)
        file.addSeparator()
        self._action(file, "&Quit", self.close, QtGui.QKeySequence.StandardKey.Quit)
        edit = menus.addMenu("&Edit")
        self.copy_action = self._action(edit, "&Copy rows", self.copy, QtGui.QKeySequence.StandardKey.Copy,
                                        "Copy the selected table rows, or the selected hole's row (Ctrl+C)")
        self.reset_action = self._action(edit, "&Reset settings", self.reset_settings,
                                         tip="Every setting back to its default (the batch's defaults)")
        view = menus.addMenu("&View")
        for dock, key in ((self.setup_dock, "Ctrl+1"), (self.inspector_dock, "Ctrl+2"), (self.bottom_dock, "Ctrl+3")):
            action = dock.toggleViewAction()
            action.setShortcut(key)
            view.addAction(action)
        view.addSeparator()
        self.fit_action = self._action(view, "&Fit page", self.measure.fit_page, "F", "Fit the active page (F)")
        self.fit_stack_action = self._action(view, "Fit &stack", self.measure.fit_stack, "Shift+F",
                                             "Fit the whole stack (Shift+F)")
        self.zoom_action = self._action(view, "&Zoom to hole", self.measure.zoom_selected, "Z",
                                        "Zoom to the selected hole (Z)")
        self.preprocessed_action = self._action(
            view, "&Preprocessed page", lambda on: session.set_version("pre" if on else "raw"), "P",
            "Show and measure the preprocessed page; off: the raw page (P)", checkable=True)
        session.version_changed.connect(lambda: self._set_checked(self.preprocessed_action, session.version == "pre"))
        self.stack_action = self._action(view, "Stack &vertically", session.set_stacked, checkable=True,
                                         tip="Every page in one view; analysis uses the active, outlined page")
        session.stacked_changed.connect(lambda stacked: self._set_checked(self.stack_action, stacked))
        view.addSeparator()
        layers = view.addMenu("&Layers")
        for action in self.measure.layer_checks.values():
            layers.addAction(action)
        layers.addSeparator()
        layers.addAction(self.measure.region_action)
        run = menus.addMenu("&Run")
        self.run_actions = {}
        for key, text in (("detect", "&Preview holes"), ("cd", "Measure &CD"), ("pitch", "Measure p&itch"),
                          ("register", "&Align")):
            self.run_actions[key] = self._action(run, text, lambda checked=False, key=key: self.setup.run_step(key))
        self.all_pages_action = self._action(
            run, "Measure &all pages...", self.measure_all_pages, "Ctrl+Shift+A",
            "Measure CD and pitch on every page with the current settings, on the GPU (Ctrl+Shift+A)")
        run.addSeparator()
        self.run_open_action = self._action(run, "Run the open &step", self.setup.run_current,
                                            ["Ctrl+Return", "Ctrl+Enter"], "Run the open Setup step (Ctrl+Enter)")
        self.cancel_action = self._action(run, "Ca&ncel", session.cancel, "Esc",
                                          "Stop waiting for the job and discard its result (Esc)")
        holes = menus.addMenu("&Holes")
        self._action(holes, "&Next hole", lambda: session.step_hole(1), "J", "Next hole in table order (J)")
        self._action(holes, "&Previous hole", lambda: session.step_hole(-1), "K", "Previous hole in table order (K)")
        self._action(holes, "Next &failed", lambda: session.step_failed(1), "N",
                     "Next failed hole of the current method (N)")
        self._action(holes, "Previous fai&led", lambda: session.step_failed(-1), "Shift+N",
                     "Previous failed hole of the current method (Shift+N)")
        holes.addSeparator()
        self._action(holes, "Next page\tPage Down (image or page list)", lambda: self._page_step(1))
        self._action(holes, "Previous page\tPage Up (image or page list)", lambda: self._page_step(-1))
        self.flicker_action = self._action(holes, "Flicker reference / aligned", self.align_view.flicker, "Space",
                                           "Align view: swap the reference and the aligned page (Space)")
        help_menu = menus.addMenu("&Help")
        self._action(help_menu, "&Keys and workflow", self.show_guide, "F1")
        for menu in (file, edit, view, run, holes, help_menu):
            menu.setToolTipsVisible(True)
        self.setup.moving_other.clicked.connect(self.open_moving)
        self.pages.measure_all.clicked.connect(self.measure_all_pages)

    @staticmethod
    def _set_checked(action, checked):
        with QtCore.QSignalBlocker(action):
            action.setChecked(checked)

    # ------------------------------------------------------------------------------- convenience
    @property
    def busy(self):
        return self.session.busy

    @property
    def loaded(self):
        return self.session.loaded

    @property
    def review(self):
        return self.session.review

    @property
    def selected(self):
        return self.session.selected

    @property
    def registration(self):
        return self.session.registration

    @property
    def moving(self):
        return self.session.moving

    @property
    def canvas(self):
        return self.measure.canvas

    def select_feature(self, index):
        """Select hole ``index`` of the result on screen (as typed in the inspector: the image pans to it)."""
        self.session.select(index, "inspector")

    @property
    def stack(self):
        return self.measure.stack

    # -------------------------------------------------------------------------------------- settings
    def _values(self):
        values = self.setup.values()
        values.update({f"layer:{name}": action.isChecked() for name, action in self.measure.layer_checks.items()})
        values.update(lock_levels=self.measure.lock_levels.isChecked(), lock_cd=self.measure.lock_cd.isChecked(),
                      holes_filter=self.holes.filter.currentData())
        return values

    def _apply_values(self, values):
        self.setup.apply_values(values)
        for name, action in self.measure.layer_checks.items():
            if f"layer:{name}" in values:
                action.setChecked(bool(values[f"layer:{name}"]))
        for key, check in (("lock_levels", self.measure.lock_levels), ("lock_cd", self.measure.lock_cd)):
            if key in values:
                check.setChecked(bool(values[key]))
        if self.holes.filter.findData(values.get("holes_filter")) >= 0:
            self.holes.filter.setCurrentIndex(self.holes.filter.findData(values["holes_filter"]))

    def _load_settings(self):
        settings = self._settings
        try:
            values = json.loads(settings.value(f"{SETTINGS_KEY}/controls", "{}") or "{}")
        except (TypeError, ValueError):
            values = {}
        self._apply_values(values)
        geometry = settings.value(f"{SETTINGS_KEY}/geometry")
        if isinstance(geometry, QtCore.QByteArray) and not geometry.isEmpty():
            self.restoreGeometry(geometry)
        state = settings.value(f"{SETTINGS_KEY}/docks")
        if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
            self.restoreState(state)
        else:
            screen = self.screen().availableGeometry() if self.screen() is not None else None
            if screen is not None and (screen.width() < 1300 or screen.height() < 800):
                self.resize(min(1440, screen.width()), min(900, screen.height()))
                self.setup_dock.hide()  # small screens: Setup starts collapsed (Ctrl+1)
        tab = settings.value(f"{SETTINGS_KEY}/bottom_tab")
        if tab is not None and str(tab).isdigit():
            self.bottom.setCurrentIndex(int(tab))

    def save_settings(self):
        settings = self._settings
        settings.setValue(f"{SETTINGS_KEY}/controls", json.dumps(self._values()))
        settings.setValue(f"{SETTINGS_KEY}/geometry", self.saveGeometry())
        settings.setValue(f"{SETTINGS_KEY}/docks", self.saveState())
        settings.setValue(f"{SETTINGS_KEY}/bottom_tab", self.bottom.currentIndex())
        settings.sync()

    def reset_settings(self):
        """Every setting back to its default; the image, results and window layout stay."""
        self._apply_values(self._defaults)
        self.session.set_parameters(self.setup.parameters())
        self._show_status("Settings reset to their defaults (the batch's defaults).", "info")

    def revert_settings(self):
        """Put back the settings the results on screen were made with (the stale banner's Revert)."""
        parameters = self.session.revert_parameters()
        if not parameters:
            return
        self.setup.apply_parameters(parameters)
        if "bounds" in parameters:
            self.measure.set_region(parameters["bounds"])
        self._show_status("Settings reverted to those of the results on screen.", "info")

    def _last_folder(self):
        return str(self._settings.value(f"{SETTINGS_KEY}/last_folder", "") or "")

    def _remember(self, path):
        path = str(Path(path).resolve())
        self._settings.setValue(f"{SETTINGS_KEY}/last_folder", str(Path(path).parent))
        recent = [item for item in self._recent() if item != path]
        self._settings.setValue(f"{SETTINGS_KEY}/recent", json.dumps([path] + recent[:RECENT_FILES - 1]))

    def _recent(self):
        try:
            return [str(item) for item in json.loads(self._settings.value(f"{SETTINGS_KEY}/recent", "[]") or "[]")]
        except (TypeError, ValueError):
            return []

    def _fill_recent(self):
        self.recent_menu.clear()
        recent = self._recent()
        for path in recent:
            self.recent_menu.addAction(path, lambda path=path: self.open_image(path=path))
        if not recent:
            self.recent_menu.addAction("No recent files").setEnabled(False)

    # ------------------------------------------------------------------------------------ open, close
    def _confirm_discard(self, what):
        """Ask before results are discarded (another file, closing); True to go on."""
        if not self.ask_before_discard or not self.session.has_results():
            return True
        pages = sum(1 for results in self.session.results.values() if results.reviews())
        answer = QtWidgets.QMessageBox.question(
            self, "Discard results?", f"{what} discards the results of {pages} page{'s' * (pages != 1)}"
            + (" and the alignment" if self.session.registration is not None else "") + ". Continue?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _refuse_while_pages(self, what):
        """True (and a hint) while Measure all pages runs."""
        if self.session.pages_running:
            self._show_status(f"Measure all pages is running: stop it (Esc) before {what}.", "warning")
            return True
        return False

    def open_image(self, checked=False, *, path=None, page=0):
        if self.busy or self._refuse_while_pages("opening another file"):
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open image", self._last_folder(), IMAGE_FILTER)
        if path and self._confirm_discard("Opening another file"):
            self._remember(path)
            self.session.open_file(path, page)

    def open_moving(self, checked=False, *, path=None, page=0):
        """Open another file for the moving page; its pages are listed in Setup."""
        if self.busy or self.session.loaded is None:
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open moving file", self._last_folder(), IMAGE_FILTER)
        if path:
            self._settings.setValue(f"{SETTINGS_KEY}/last_folder", str(Path(path).resolve().parent))
            self.session.open_moving(path, page, align=self.mode == "align")

    def load_demo(self):
        if self._refuse_while_pages("loading the demo"):
            return
        if not self.busy and self._confirm_discard("Loading the demo"):
            self.session.load_demo()

    def set_image(self, image, *, name="Array image"):
        """Load a finite grayscale/RGB NumPy array, resetting dependent results."""
        self.session.set_array(image, name)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths and not self.busy and not self._refuse_while_pages("opening another file"):
            event.acceptProposedAction()
            self.open_image(path=paths[0])

    def closeEvent(self, event):
        if self.busy:
            self._show_status(f"{self.session.jobs.label} is running: press Esc to cancel it, then close.", "warning")
            event.ignore()
            return
        if self._refuse_while_pages("closing"):
            event.ignore()
            return
        if not self._confirm_discard("Closing"):
            event.ignore()
            return
        self.save_settings()
        if self.measure.stack is not None:
            self.measure.stack.close()
        self.session.close()
        _WINDOWS.discard(self)
        event.accept()

    # ---------------------------------------------------------------------------------------- view
    def set_mode(self, mode):
        """Measure or Align: the main area, the right panel and Setup follow."""
        if mode == self.mode:
            return
        view = (self.measure if self.mode == "measure" else self.align_view).canvas.plot.getViewBox()
        ranges = view.viewRange()
        self.mode = mode
        align = mode == "align"
        self.views.setCurrentWidget(self.align_view if align else self.measure)
        self.right.setCurrentWidget(self.align_panel if align else self.inspector)
        self.inspector_dock.setWindowTitle("Alignment" if align else "Hole inspector")
        self.setup.set_mode(mode)
        self.left_tabs.setCurrentWidget(self.setup)
        self.flicker_action.setEnabled(align)
        target = (self.align_view if align else self.measure).canvas.plot.getViewBox()
        if self.session.loaded is not None and self.measure.stack is None:
            target.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)  # the same part of the page
        with QtCore.QSignalBlocker(self.view_buttons):
            self.view_buttons.button(1 if align else 0).setChecked(True)

    def _page_step(self, step):
        if self.session.loaded is not None:
            self.session.set_page(self.session.loaded.page + step)

    def show_log(self):
        self.bottom_dock.show()
        self.bottom.setCurrentWidget(self.log)

    def show_guide(self):
        dialog = getattr(self, "_guide", None)
        if dialog is None:
            dialog = self._guide = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Keys and workflow")
            dialog.resize(560, 560)
            layout = QtWidgets.QVBoxLayout(dialog)
            text = QtWidgets.QTextBrowser()
            text.setHtml(GUIDE)
            layout.addWidget(text)
        dialog.show()
        dialog.raise_()

    def copy(self):
        self.holes.copy()

    # ---------------------------------------------------------------------------------- session events
    def _image_changed(self, new_file):
        loaded, source = self.session.loaded, self.session.source
        pages = source.pages if source is not None else 1
        with QtCore.QSignalBlocker(self.page_spin):
            self.page_spin.setRange(0, max(0, pages - 1))
            self.page_spin.setValue(loaded.page)
        self.page_count.setText(f"of {pages} (0 = first)")
        for action in self.page_widgets:
            action.setVisible(pages > 1)
        h, w = loaded.image.shape
        self.source_label.setText(loaded.name)
        self.source_label.setToolTip(loaded.path or loaded.name)
        self.image_label.setText(f"{w:,} x {h:,} px | {loaded.image.dtype}"
                                 + (f" | {loaded.conversion}" if loaded.conversion else ""))
        self.stack_action.setEnabled(pages > 1)
        self._busy_changed(self.busy)
        if new_file:
            self._clear_error()
            self._show_status("Image ready: open a Setup step and run it (Ctrl+Enter).", "info")

    def _busy_changed(self, busy):
        session = self.session
        loaded = session.loaded is not None
        for widget in (self.job_label, self.progress, self.cancel_button):
            widget.setVisible(busy)
        if not busy:
            self.job_label.clear()
        running = session.pages_running
        self.cancel_action.setEnabled(busy or running)
        for action in (self.open_action, self.demo_action, self.reset_action, self.recent_menu.menuAction()):
            action.setEnabled(not busy and not running)
        problem = session.all_pages_problem()
        self.all_pages_action.setEnabled(not problem)
        self.pages.measure_all.setEnabled(not problem)
        self.pages.measure_all.setToolTip(problem or "Measure CD and pitch on every page with the current settings, "
                                          "on the GPU in the batch's fast mode (Ctrl+Shift+A)")
        self.export_action.setEnabled(not busy and loaded and (session.review is not None
                                                               or session.registration is not None))
        for action in self.run_actions.values():
            action.setEnabled(not busy and loaded and not running)
        self.run_open_action.setEnabled(not busy and loaded and not running)
        pages = session.source.pages if session.source is not None else 1
        page = session.loaded.page if loaded else 0
        self.page_spin.setEnabled(not busy and pages > 1)
        self.previous_page.setEnabled(not busy and page > 0)
        self.next_page.setEnabled(not busy and page < pages - 1)
        self.rerun_button.setEnabled(not busy)
        self.revert_button.setEnabled(not busy)

    def _show_error(self, text):
        """One plain line over the results (which stay visible); the full text is in the Log tab."""
        self.error_label.setText(text)
        self.error_banner.show()

    def _clear_error(self):
        self.error_banner.hide()
        self.error_label.clear()

    def _show_status(self, text, level="info"):
        self.message_label.setText(text)
        self.message_label.setToolTip(text)
        self.message_label.setProperty("level", level)
        self.message_label.style().unpolish(self.message_label)
        self.message_label.style().polish(self.message_label)
        if level != "error" and text != "Failed":
            if text.startswith(("Image ready", "Aligned", "Pitch measured")) or " holes | " in text:
                self._clear_error()

    def _update_stale(self):
        found = self.session.stale()
        self.stale_banner.setVisible(bool(found))
        if found:
            self.stale_label.setText("Results are from earlier settings: " + describe(found))
            self.stale_label.setToolTip("\n".join(text for _, text in found))
        self._busy_changed(self.busy)

    def _update_compute(self):
        backend = self.session.parameters.get("backend", "auto")
        if backend == "auto":
            text = "Compute: GPU (auto)" if gpu_available() else "Compute: CPU (auto, no GPU found)"
        else:
            text = f"Compute: {backend.upper()}"
        self.compute_label.setText(text)

    def _update_summary(self):
        """Holes, the current method's mean CD and every method's failures (always visible)."""
        session = self.session
        review, method = session.review, session.method
        if review is None or method not in review.measurements:
            self.summary_label.setText("")
            self.summary_label.hide()  # the status message gets the room
            return
        self.summary_label.show()
        measured = review.measurements[method]
        valid = measured.valid & np.isfinite(measured.centers_xy).all(axis=1)
        page = f"Page {session.loaded.page} | " if session.loaded.pages > 1 else ""
        version = "<b>Preprocessed</b> | " if session.version == "pre" else ""
        summary = f"{version}{page}{len(valid):,} holes | {method_name(method)}: {valid.sum():,} measured"
        values = measured.metrics.get("cd_equivalent_px")
        if values is not None and valid.any():
            summary += f", mean CD {np.mean(values[valid]) * review.parameters['pixel_size']:.4g} {review.parameters['unit']}"
            other = session.other_review()
            twin = other.measurements.get(method) if other is not None and other.cd is not None else None
            cds = twin.metrics.get("cd_equivalent_px") if twin is not None else None
            if cds is not None and (twin.valid & np.isfinite(cds)).any():
                good = twin.valid & np.isfinite(cds)
                summary += (f" ({'raw' if session.version == 'pre' else 'preprocessed'} page: "
                            f"{np.mean(cds[good]) * other.parameters['pixel_size']:.4g})")
        self.summary_label.setText(f"<span style='color:#dee7f2'>{summary}</span> &nbsp; " + self.failure_summary(review))
        self.summary_label.setToolTip(self.failure_summary(review, limit=None))

    @staticmethod
    def failure_summary(review, limit=4):
        """Failed holes of every method, largest first (the shown method is only one of them)."""
        if review.cd is None:
            return "<span style='color:#a9bbce'>Previewed holes: measure CD to see failures per method.</span>"
        total = max(len(review.seeds_xy), 1)
        failed = failure_counts(review)
        if not failed:
            return "<span style='color:#9be28a'>No method failed on any hole.</span>"
        parts = [f"{method_name(name)} {count:,} ({100 * count / total:.0f} %)" for count, name in failed]
        more = len(parts) - limit if limit is not None else 0
        shown = parts[:limit] if limit is not None else parts
        return ("<span style='color:#ffce73'>Failed: " + " &middot; ".join(shown)
                + (f" &middot; +{more} more" if more > 0 else "") + "</span>")

    # -------------------------------------------------------------------------------------- export
    def export_results(self, checked=False, *, folder=None, table=True, arrays=True, alignment=True):
        """Export dialog: folder and contents; ``folder`` given = no dialog."""
        session = self.session
        if self.busy or session.loaded is None or (session.review is None and session.registration is None):
            return
        if folder is None:
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Export")
            form = QtWidgets.QFormLayout(dialog)
            default = Path(self._last_folder() or Path.home()) / ("inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            path = QtWidgets.QLineEdit(str(default))
            browse = QtWidgets.QPushButton("Browse...")
            browse.clicked.connect(lambda: path.setText(QtWidgets.QFileDialog.getExistingDirectory(
                dialog, "Export into", str(Path(path.text()).parent)) or path.text()))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(path, 1)
            row.addWidget(browse)
            form.addRow("Folder", row)
            table_check = QtWidgets.QCheckBox("Holes table as shown (filter and sort), CSV")
            table_check.setChecked(session.review is not None)
            table_check.setEnabled(session.review is not None)
            arrays_check = QtWidgets.QCheckBox("All result arrays (NumPy) and pitch profiles")
            arrays_check.setChecked(session.review is not None)
            arrays_check.setEnabled(session.review is not None)
            alignment_check = QtWidgets.QCheckBox("Alignment (shift, aligned page, differences)")
            alignment_check.setChecked(session.registration is not None)
            alignment_check.setEnabled(session.registration is not None)
            for check in (table_check, arrays_check, alignment_check):
                form.addRow("", check)
            form.addRow("", hint("source.json (file, page, settings) is always written."))
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                                 | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            dialog.resize(560, 220)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not path.text().strip():
                return
            folder, table, arrays, alignment = (path.text().strip(), table_check.isChecked(),
                                                arrays_check.isChecked(), alignment_check.isChecked())
            self._settings.setValue(f"{SETTINGS_KEY}/last_folder", str(Path(folder).parent))
        rows = self.holes.model.export_rows() if table and self.holes.model.review is not None else None
        session.export(folder, arrays=arrays, table=rows, alignment=alignment)

    # --------------------------------------------------------------------------------- all pages
    def measure_all_pages(self, checked=False, *, folder=None, first=None, last=None, skip_no_holes=True,
                          finders=True):
        """The Measure all pages dialog (pages, hole check, GPU centers, folder), then the run; ``folder`` given =
        no dialog."""
        session = self.session
        problem = session.all_pages_problem()
        if problem:
            self._show_status(problem, "warning")
            return False
        pages = session.source.pages
        first = 0 if first is None else first
        last = pages - 1 if last is None else last
        if folder is None:
            parameters = session.all_pages_parameters()
            try:
                backend, gpus = devices(parameters)
            except ValueError as exc:
                self._show_error(str(exc))
                return False
            dialog = QtWidgets.QDialog(self)
            dialog.setWindowTitle("Measure all pages")
            form = QtWidgets.QFormLayout(dialog)
            start, stop = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
            for spin, value in ((start, first), (stop, last)):
                spin.setRange(0, pages - 1)
                spin.setValue(value)
            span = QtWidgets.QHBoxLayout()
            span.addWidget(start)
            span.addWidget(QtWidgets.QLabel("to"))
            span.addWidget(stop)
            span.addWidget(hint(f"of {pages} pages (0 = first)", wrap=False))
            span.addStretch()
            form.addRow("Pages", span)
            limit = HOLE_CHECK_MAX_ERROR[session.parameters.get("hole_check_mode", "centered")]
            skip = QtWidgets.QCheckBox(f"Skip pages without holes: compare each page with page {session.loaded.page} "
                                       f"(this page)")
            skip.setToolTip(f"The batch's hole check ({session.parameters.get('hole_check_mode', 'centered')}): a "
                            f"page is measured when its correlation error is below {limit:g}")
            skip.setChecked(skip_no_holes)
            form.addRow("", skip)
            ticked = tuple(parameters.get("pitch_finders") or ())
            finder_check = QtWidgets.QCheckBox("GPU centers too for pitch: " + ", ".join(
                method_name(f"gpu_{key}").removeprefix("GPU ").removesuffix(" centers") for key in ticked))
            finder_check.setToolTip("The GPU center finders ticked in Setup > Pitch. The three defaults take about "
                                    "2 s more per 39,000-hole page.")
            finder_check.setChecked(finders and bool(ticked))
            finder_check.setVisible(bool(ticked) and backend == "gpu")
            form.addRow("", finder_check)
            stem = Path(session.source.path).stem
            default = Path(session.source.path).parent / f"{stem}_all_pages_{datetime.now():%Y%m%d_%H%M%S}"
            path = QtWidgets.QLineEdit(str(default))
            browse = QtWidgets.QPushButton("Browse...")
            browse.clicked.connect(lambda: path.setText(QtWidgets.QFileDialog.getExistingDirectory(
                dialog, "Results into", str(Path(path.text()).parent)) or path.text()))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(path, 1)
            row.addWidget(browse)
            form.addRow("Folder", row)
            methods = tuple(parameters["methods"])
            preset = next((label for label, names in RECIPES.values() if tuple(names) == methods), None)
            compute = (f"{len(gpus)} GPU{'s' * (len(gpus) != 1)} ({', '.join(name or gpu for gpu, name in gpus)}), "
                       f"{PAGES_PER_GPU} pages at a time on each" if backend == "gpu" else "the CPU, one page at a time")
            what = (f"Measures {len(methods)} CD method{'s' * (len(methods) != 1)}"
                    + (f" ({preset})" if preset else "") + " with the current settings on the "
                    + ("preprocessed pages (" + preprocess_text(parameters.get("preprocess")) + ")"
                       if parameters.get("preprocess") else "raw pages")
                    + (", in the analysis region" if parameters.get("bounds") else ", whole pages")
                    + ", then the pitch of every method (Pitch settings, each page's default reference hole). "
                    f"On {compute}, in the batch's fast mode: the same CDs as Measure CD, without contours. "
                    "Writes pages.csv (cd_batch's columns, then pitch), holes/page_NNNN.csv and run.json. Pages can "
                    "be browsed meanwhile; other runs wait.")
            form.addRow("", hint(what))
            buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
            buttons.addButton("Start", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            form.addRow(buttons)
            dialog.resize(620, 300)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or not path.text().strip():
                return False
            folder, first, last = path.text().strip(), start.value(), stop.value()
            skip_no_holes, finders = skip.isChecked(), finder_check.isChecked()
        if first > last:
            first, last = last, first
        started = session.run_all_pages(first, last, folder, skip_no_holes=skip_no_holes, finders=finders)
        if started:
            self.bottom_dock.show()
            self.bottom.setCurrentWidget(self.all_pages)
        return started

    def _update_pages_progress(self):
        run = self.session.pages_run
        running = run is not None and run.running
        for widget in (self.pages_label, self.pages_bar, self.pages_stop):
            widget.setVisible(running)
        if not running:
            return
        done, total, rate, left = run.progress()
        counts = run.counts
        self.pages_bar.setRange(0, total)
        self.pages_bar.setValue(done)
        self.pages_label.setText(
            f"All pages {done} of {total} | {counts['no_holes']} no holes, {counts['failed']} failed | "
            + (f"{1 / rate:.2f} s/page, about {max(left, 0) / 60:.1f} min left" if rate > 0 else
               f"starting {run.elapsed():.0f} s"))

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save workbench screenshot", "inspection.png", "PNG (*.png)")
        if path:
            if not self.grab().save(path):
                self._show_error("Could not save the screenshot: " + path)
            else:
                self._show_status(f"Screenshot saved: {path}", "info")


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
    parser.add_argument("--page", type=int, default=0, help="Page to open, 0 = first")
    parser.add_argument("--demo", action="store_true", help="Open a reproducible synthetic contact-hole image")
    args = parser.parse_args(argv)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    window = InspectionWindow()
    window.show()
    if args.image:
        window.open_image(path=args.image, page=args.page)
    elif args.demo:
        window.load_demo()
    return application.exec()
