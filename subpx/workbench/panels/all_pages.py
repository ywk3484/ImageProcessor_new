"""All pages (bottom): the results of Measure all pages, one row per page: each method's mean CD, holes measured or
failed, pitch or placement error, and a plot of the chosen method's value by page. A click on a row or on a point of
the plot shows that page (Measure CD on it to inspect its holes)."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..names import method_name
from ..widgets import hint, plot_widget
from .pitch import _line

# (key, text): what the table and the plot show
VALUES = (("cd", "Mean CD"), ("cd_x", "Mean CD x"), ("cd_y", "Mean CD y"), ("valid", "Holes measured"),
          ("failed", "Holes failed"), ("pitch_x", "Pitch x"), ("pitch_y", "Pitch y"),
          ("placement_x", "Placement error x"), ("placement_y", "Placement error y"))
PITCH_VALUES = {"pitch_x", "pitch_y", "placement_x", "placement_y"}
COUNT_VALUES = {"valid", "failed"}
STATUS_TEXT = {"ok": "measured", "no_holes": "no holes", "failed": "failed"}
STATUS_COLORS = {"ok": "#dee7f2", "no_holes": "#899caf", "failed": "#ff5c74", None: "#5d6f82"}
LINE_COLOR = "#4dd0e1"
NAN = float("nan")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def page_value(row, method, key):
    """One page's value of ``method``: a number, or (lowest, highest) for placement error; None for a page without
    a measurement (not measured yet, no holes, failed)."""
    if row is None or row.get("status") != "ok":
        return None
    if key == "failed":
        return _number(row.get("holes")) - _number(row.get(f"{method}_valid"))
    if key.startswith("placement"):
        axis = key[-1]
        return (_number(row.get(f"{method}_placement_{axis}_min")), _number(row.get(f"{method}_placement_{axis}_max")))
    return _number(row.get(f"{method}_{key}"))


def value_text(value, key):
    if value is None:
        return ""
    if key.startswith("placement"):
        low, high = value
        return f"{low:+.3g} to {high:+.3g}" if np.isfinite(low) else ""
    if not np.isfinite(value):
        return ""
    return f"{int(value):,}" if key in COUNT_VALUES else f"{value:.5g}"


class PagesModel(QtCore.QAbstractTableModel):
    """Rows: the pages of the run; columns: page, status, holes, then the chosen value of every method."""

    FIXED = ("Page", "Status", "Holes")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.run, self.key, self.method, self.methods = None, "cd", None, []

    def configure(self, run, key, method, methods):
        self.beginResetModel()
        self.run, self.key, self.method, self.methods = run, key, method, list(methods)
        self.endResetModel()

    def refresh(self):
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() or self.run is None else len(self.run.pages)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() or self.run is None else len(self.FIXED) + len(self.methods)

    def page(self, row):
        return self.run.pages[row]

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.run is None:
            return None
        page = self.run.pages[index.row()]
        row = self.run.rows.get(page)
        status = row.get("status") if row is not None else None
        column = index.column()
        Role = QtCore.Qt.ItemDataRole
        if role == Role.DisplayRole:
            if column == 0:
                return str(page)
            if column == 1:
                return STATUS_TEXT.get(status, "waiting" if self.run.running else "not measured")
            if column == 2:
                return f"{int(row['holes']):,}" if status == "ok" else ""
            return value_text(page_value(row, self.methods[column - 3], self.key), self.key)
        if role == Role.ForegroundRole:
            return QtGui.QBrush(QtGui.QColor(STATUS_COLORS.get(status, STATUS_COLORS[None])))
        if role == Role.TextAlignmentRole and column != 1:
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == Role.FontRole and column >= 3 and self.methods[column - 3] == self.method:
            font = QtGui.QFont()
            font.setBold(True)
            return font
        if role == Role.ToolTipRole and status == "failed":
            return str(row.get("error", ""))
        if role == Role.ToolTipRole and status == "no_holes":
            return f"Hole check {_number(row.get('pcc_error')):.3f}: passes below {self.run.max_error:g}"
        if role == Role.ToolTipRole and status == "ok" and row.get("pitch_error"):
            return "Pitch: " + str(row["pitch_error"])
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if orientation != QtCore.Qt.Orientation.Horizontal or self.run is None:
            return None
        method = self.methods[section - 3] if section >= 3 else None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return self.FIXED[section] if method is None else method_name(method)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole and method is not None:
            return f"{method}: click to plot this method"
        if role == QtCore.Qt.ItemDataRole.FontRole and method is not None and method == self.method:
            font = QtGui.QFont()
            font.setBold(True)
            return font
        return None


class AllPagesPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Show"))
        self.value = QtWidgets.QComboBox()
        for key, text in VALUES:
            self.value.addItem(text, key)
        self.value.setToolTip("The value of every method per page in the table; the chosen method's in the plot")
        self.value.currentIndexChanged.connect(self._configure)
        row.addWidget(self.value)
        row.addWidget(QtWidgets.QLabel("Method"))
        self.method = QtWidgets.QComboBox()
        self.method.setMinimumContentsLength(14)
        self.method.setToolTip("The method plotted by page (or click its column header)")
        self.method.currentIndexChanged.connect(self._method_changed)
        row.addWidget(self.method)
        row.addStretch()
        self.open_folder = QtWidgets.QPushButton("Open folder")
        self.open_folder.setToolTip("The run's pages.csv, holes/page_NNNN.csv and run.json")
        self.open_folder.clicked.connect(self._open_folder)
        row.addWidget(self.open_folder)
        layout.addLayout(row)
        self.summary = hint("Measure all pages (Pages tab or Run menu) measures CD and pitch on every page; the "
                            "results appear here.")
        layout.addWidget(self.summary)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.model = PagesModel(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().sectionClicked.connect(self._header_clicked)
        self.table.clicked.connect(lambda index: self._show_page(self.model.page(index.row())))
        self.table.setToolTip("Click a page to show it; Measure CD on it to inspect its holes")
        splitter.addWidget(self.table)
        self.plot = plot_widget()
        self.plot.setLabel("bottom", "Page")
        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)
        splitter.addWidget(self.plot)
        splitter.setSizes([520, 480])
        layout.addWidget(splitter, 1)
        self._points = (np.empty(0), np.empty(0))
        self._run = None
        session.pages_changed.connect(self._pages_changed)
        session.pages_progress.connect(self._update_summary)
        session.image_changed.connect(self._follow_page)
        self._configure()

    # ------------------------------------------------------------------------------------------ state
    def key(self):
        return self.value.currentData()

    def _methods(self):
        run = self.session.pages_run
        if run is None:
            return []
        return run.pitch_methods if self.key() in PITCH_VALUES else run.methods

    def _pages_changed(self):
        run = self.session.pages_run
        if run is not self._run:
            self._run = run
            self._configure()
        else:
            self.model.refresh()
            self.draw()
            self._update_summary()
        self._follow_page(scroll=False)  # rows arriving never scroll the table away from where it is

    def _configure(self, *args):
        """Another run or value: the method box, the table's columns and the plot."""
        methods = self._methods()
        old = self.method.currentData()
        with QtCore.QSignalBlocker(self.method):
            self.method.clear()
            for name in methods:
                self.method.addItem(method_name(name), name)
            wanted = old if old in methods else self.session.method if self.session.method in methods else None
            self.method.setCurrentIndex(max(0, self.method.findData(wanted)) if methods else -1)
        self.model.configure(self.session.pages_run, self.key(), self.method.currentData(), methods)
        self._size_columns()
        self.open_folder.setEnabled(self.session.pages_run is not None)
        self.draw()
        self._update_summary()
        self._follow_page()

    def _size_columns(self):
        """Each column as wide as its name in bold or its widest value, whichever is wider (values arrive later)."""
        header = self.table.horizontalHeader()
        font = QtGui.QFont(header.font())
        font.setBold(True)
        bold, plain = QtGui.QFontMetrics(font), QtGui.QFontMetrics(self.table.font())
        key = self.key()
        sample = ("-0.0412 to +0.0711" if key.startswith("placement") else "39,042" if key in COUNT_VALUES else
                  "-1234.56")
        for column in range(self.model.columnCount()):
            text = str(self.model.headerData(column, QtCore.Qt.Orientation.Horizontal))
            values = {0: "0000", 1: "not measured", 2: "39,042"}.get(column, sample)
            header.resizeSection(column, max(bold.horizontalAdvance(text), bold.horizontalAdvance(values)) + 20)

    def _method_changed(self, *args):
        self.model.configure(self.model.run, self.key(), self.method.currentData(), self.model.methods)
        self.draw()
        self._follow_page()

    def _header_clicked(self, section):
        if section >= 3:
            self.method.setCurrentIndex(self.method.findData(self.model.methods[section - 3]))

    def _follow_page(self, *args, scroll=True):
        """The table's current row is the page on screen (scrolled into view when the page changes)."""
        loaded, run = self.session.loaded, self.model.run
        if loaded is None or run is None or loaded.page not in run.pages:
            self.table.clearSelection()
            return
        row = run.pages.index(loaded.page)
        if self.table.currentIndex().row() != row:
            self.table.selectRow(row)
        if scroll:
            self.table.scrollTo(self.model.index(row, 0))

    def _show_page(self, page):
        self.session.set_page(int(page))

    def _open_folder(self):
        run = self.session.pages_run
        if run is not None:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(run.output)))

    # ------------------------------------------------------------------------------------------- views
    def _update_summary(self):
        run = self.session.pages_run
        if run is None:
            self.summary.setText("Measure all pages (Pages tab or Run menu) measures CD and pitch on every page; the "
                                 "results appear here.")
            return
        done, total, rate, left = run.progress()
        counts = run.counts
        found = f"{counts['ok']} measured, {counts['no_holes']} no holes, {counts['failed']} failed"
        speed = f"{1 / rate:.2f} s per page" if rate > 0 else "the first pages take longer (workers start)"
        if run.running:
            text = (f"Measuring pages {run.pages[0]} to {run.pages[-1]}: {done} of {total} done ({found}), {speed}"
                    + (f", about {_duration(left)} left" if np.isfinite(left) else "") + ".")
        else:
            state = {"done": "", "stopped": " (stopped)", "failed": " (stopped by an error)"}[run.state]
            text = (f"{done} of {total} pages in {_duration(run.elapsed())}{state}: {found}"
                    + (f", {1 / rate:.2f} s per page" if rate > 0 else "") + f". Results: {run.output}.")
        unit = run.parameters.get("unit", "px")
        key = self.key()
        if key not in COUNT_VALUES:
            text += f" Values in {unit}" + ("; placement from the default reference hole of each page." if
                                             key.startswith("placement") else "; means over the holes each "
                                             "method measured." if not key.startswith("pitch") else ".")
        self.summary.setText(text)

    def draw(self):
        self.plot.clear()
        run, method, key = self.model.run, self.method.currentData(), self.key()
        self._points = (np.empty(0), np.empty(0))
        if run is None or method is None:
            return
        pages = np.asarray(run.pages, float)
        values = [page_value(run.rows.get(page), method, key) for page in run.pages]
        text = dict(VALUES)[key]
        unit = "" if key in COUNT_VALUES else run.parameters.get("unit", "px")
        self.plot.setLabel("left", f"{method_name(method)}: {text}" + (f" ({unit})" if unit else ""))
        if key.startswith("placement"):
            lows = np.array([NAN if v is None else v[0] for v in values])
            highs = np.array([NAN if v is None else v[1] for v in values])
            _line(self.plot, pages, lows, LINE_COLOR)
            _line(self.plot, pages, highs, LINE_COLOR)
            self._points = (np.r_[pages, pages], np.r_[lows, highs])
        else:
            ys = np.array([NAN if v is None else v for v in values])
            _line(self.plot, pages, ys, LINE_COLOR)
            self._points = (pages, ys)

    def _plot_clicked(self, event):
        xs, ys = self._points
        finite = np.isfinite(ys)
        if not finite.any():
            return
        point = self.plot.getViewBox().mapSceneToView(event.scenePos())
        candidates = np.flatnonzero(finite)
        self._show_page(int(xs[candidates[np.argmin(np.abs(xs[candidates] - point.x()))]]))


def _duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d} min" if seconds >= 60 else f"{seconds} s"
