"""Holes table (bottom): one row per hole, fixed columns, sortable, filtered by failures or by the holes picked
in CD distribution or Pitch; the selection follows the keyboard and the mouse both ways."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ...cd import METHOD_LABELS
from ..names import FAILED_COLOR, method_name, status_text
from ..widgets import hint

FILTERS = (("All holes", "all"), ("Failed: current method", "failed"), ("Failed: any method", "any"))


class MeasurementTable(QtCore.QAbstractTableModel):
    """The holes table on NumPy columns: built once per result; a method switch, filter or sort reorders indices.

    Fixed column order: ID, x, y, Status, CD x and CD y of the shown method, then the CD of every method in
    Setup order (the shown one in bold). Previewed holes: ID, x, y, Status and the component area. A sort stays
    when the method or the filter changes.
    """

    METHOD_COLUMNS = 6  # the first per-method CD column when the result has CDs

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review = self.method = None
        self.mode, self.subset = "all", None
        self.columns = []  # dicts: header, values (one per hole), tooltip, method (CD columns)
        self.cd_methods = []
        self.order = np.empty(0, int)  # hole per row
        self.position = np.empty(0, int)  # row per hole, -1 when filtered out
        self.bad = np.empty(0, bool)  # the shown method failed
        self.any_bad = np.empty(0, bool)  # some method failed
        self.sort_column, self.sort_order = None, QtCore.Qt.SortOrder.AscendingOrder

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.order)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.columns)

    def text(self, row, column):
        value = self.columns[column]["values"][self.order[row]]
        if column == 3:
            return status_text(value)
        if isinstance(value, (float, np.floating)):
            return f"{value:.5g}" if np.isfinite(value) else "--"
        return str(value)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        column, hole = self.columns[index.column()], int(self.order[index.row()])
        Role = QtCore.Qt.ItemDataRole
        if role == Role.DisplayRole:
            return self.text(index.row(), index.column())
        if role == Role.UserRole:
            return hole
        method = column.get("method")
        if role == Role.ForegroundRole:
            failed = not self.review.measurements[method].valid[hole] if method else self.bad[hole]
            return QtGui.QColor(FAILED_COLOR) if failed else None
        if role == Role.ToolTipRole:
            if method:
                return (f"{method_name(method)} ({method}: {METHOD_LABELS.get(method, method)}): "
                        f"{status_text(self.review.measurements[method].status[hole])}")
            if index.column() == 3:
                return str(column["values"][hole])
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

    def set_review(self, review, method, mode="all", subset=None):
        self.beginResetModel()
        self.review, self.method, self.mode, self.subset = review, None, mode, subset
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
        elif review.detection is not None:
            areas = review.detection.boxes[review.detection.kept, 4]
            if len(areas) == len(review.seeds_xy):
                self.columns.append(dict(header="Area (px²)", values=areas.astype(float),
                                         tooltip="Component area within the area limits"))
        failed = [~measured.valid for measured in review.measurements.values()]
        self.any_bad = np.any(failed, axis=0) if failed else np.zeros(len(review.seeds_xy), bool)
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
        self.bad = ~measured.valid  # as the summary and N / Shift+N count failures
        if self.cd_methods:
            missing = np.full(len(points), np.nan)
            for column, key in zip(self.columns[4:6], ("cd_x_px", "cd_y_px")):
                values = measured.metrics.get(key)
                column["values"] = values * scale if values is not None else missing

    def set_method(self, method, mode=None, subset=False):
        """Show another method of the same result, another filter or subset (False: keep); rows keep the sort."""
        mode = self.mode if mode is None else mode
        subset = self.subset if subset is False else subset
        if method == self.method and mode == self.mode and subset is self.subset:
            return
        self._set_method(method)
        self.mode, self.subset = mode, subset
        self._relayout()
        if self.order.size:
            self.dataChanged.emit(self.index(0, 1), self.index(len(self.order) - 1, len(self.columns) - 1))
        self.headerDataChanged.emit(QtCore.Qt.Orientation.Horizontal, 0, len(self.columns) - 1)

    def _filter_and_sort(self):
        keep = self.bad if self.mode == "failed" else self.any_bad if self.mode == "any" else np.ones(len(self.bad), bool)
        if self.subset is not None:
            inside = np.zeros(len(self.bad), bool)
            inside[self.subset] = True
            keep = keep & inside
        rows = np.flatnonzero(keep)
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
        return int(self.position[hole]) if hole is not None and 0 <= hole < len(self.position) else -1

    def hole_at(self, row):
        return int(self.order[row])

    def value(self, row, column):
        return self.columns[column]["values"][self.order[row]]

    def method_of_column(self, column):
        return self.columns[column].get("method") if 0 <= column < len(self.columns) else None

    def headers(self):
        return [column["header"].replace("\n", " ") for column in self.columns]

    def rows_text(self, rows):
        return [[self.text(row, column) for column in range(len(self.columns))] for row in rows]

    def export_rows(self):
        """Every row as shown (filter and sort), full precision."""
        rows = []
        for hole in self.order:
            row = []
            for column in self.columns:
                value = column["values"][hole]
                row.append(f"{value:.9g}" if isinstance(value, (float, np.floating)) else str(value))
            rows.append(row)
        return self.headers(), rows

    def clear(self):
        self.beginResetModel()
        self.review = self.method = None
        self.subset = None
        self.columns, self.cd_methods = [], []
        self.order = self.position = np.empty(0, int)
        self.bad = self.any_bad = np.empty(0, bool)
        self.endResetModel()

    def sort(self, column, order=QtCore.Qt.SortOrder.AscendingOrder):
        self.sort_column, self.sort_order = column, order
        if self.review is not None:
            self._relayout()


class HolesPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.filter = QtWidgets.QComboBox()
        for text, key in FILTERS:
            self.filter.addItem(text, key)
        self.filter.setToolTip("Which holes the table lists")
        self.filter.currentIndexChanged.connect(self._filter_changed)
        row.addWidget(self.filter)
        self.find = QtWidgets.QLineEdit()
        self.find.setPlaceholderText("Hole ID")
        self.find.setMaximumWidth(90)
        self.find.setValidator(QtGui.QIntValidator(0, 10**9))
        self.find.setToolTip("Type a hole number and press Enter to select it")
        self.find.returnPressed.connect(self._find)
        row.addWidget(self.find)
        self.count = QtWidgets.QLabel("")
        self.count.setObjectName("hint")
        row.addWidget(self.count)
        self.subset_label = QtWidgets.QLabel("")
        self.subset_label.setObjectName("warning")
        row.addWidget(self.subset_label)
        self.clear_subset = QtWidgets.QToolButton()
        self.clear_subset.setText("Show all")
        self.clear_subset.setToolTip("Clear the holes picked in CD distribution or Pitch")
        self.clear_subset.clicked.connect(lambda: session.set_highlight(None))
        self.clear_subset.hide()
        row.addWidget(self.clear_subset)
        row.addStretch()
        layout.addLayout(row)
        self.empty = hint("Preview holes or measure CD to list the holes.")
        layout.addWidget(self.empty)
        self.model = MeasurementTable(self)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        header = self.table.horizontalHeader()
        header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)  # rows in ID order until a header click
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # Compact columns the user can widen; fixed row height, so 40k rows cost nothing to lay out.
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setDefaultSectionSize(92)
        header.setMinimumSectionSize(36)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setDefaultSectionSize(21)
        self.table.clicked.connect(self._clicked)
        self.model.layoutChanged.connect(self._publish)  # a header click re-sorts: J / K and N follow the new order
        # Arrow keys, Page keys and clicks all move the current row; the selected hole follows it.
        self.table.selectionModel().currentRowChanged.connect(self._row_changed)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.table, 1)
        self.table.hide()
        self._syncing = False
        session.results_changed.connect(self._results_changed)
        session.method_changed.connect(self._method_changed)
        session.selection_changed.connect(self._selection_changed)
        session.highlight_changed.connect(self._highlight_changed)

    # -------------------------------------------------------------------------------- session events
    def _results_changed(self):
        session, model = self.session, self.model
        review, method = session.review, session.method
        if review is None or method not in review.measurements or not len(review.seeds_xy):
            model.clear()
            self.table.hide()
            self.empty.setText("Preview holes or measure CD to list the holes." if review is None else
                               "No holes found on this page.")
            self.empty.show()
            self._publish()
            return
        self.empty.hide()
        self.table.show()
        if model.review is not review:
            self._syncing = True
            try:
                # The sort of the last table carries over to the new result.
                header = self.table.horizontalHeader()
                model.sort_column = max(0, header.sortIndicatorSection())
                model.sort_order = header.sortIndicatorOrder()
                model.set_review(review, method, self.filter.currentData(), session.highlight)
                # Compact: each column as wide as its name (bold when shown), never squeezed below it.
                font = QtGui.QFont(header.font())
                font.setBold(True)
                metrics = QtGui.QFontMetrics(font)
                for column in range(model.columnCount()):
                    text = model.headerData(column, QtCore.Qt.Orientation.Horizontal)
                    width = max(metrics.horizontalAdvance(line) for line in text.splitlines()) + 24
                    header.resizeSection(column, int(np.clip(width, 150 if column == 3 else 56, 280)))
                if model.sort_column >= model.columnCount():
                    model.sort_column = 0
                    header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)
            finally:
                self._syncing = False
        self._publish()
        self._sync_selection()

    def _method_changed(self):
        review, method = self.session.review, self.session.method
        if self.model.review is not review or review is None or method not in review.measurements:
            return
        self._syncing = True
        try:
            self.model.set_method(method)
        finally:
            self._syncing = False
        self._publish()
        self._sync_selection()

    def _filter_changed(self):
        if self.model.review is None:
            return
        self._syncing = True
        try:
            self.model.set_method(self.model.method, self.filter.currentData())
        finally:
            self._syncing = False
        self._publish()
        self._sync_selection()

    def _highlight_changed(self):
        session = self.session
        if self.model.review is not None and self.model.review is session.review:
            self._syncing = True
            try:
                self.model.set_method(self.model.method, subset=session.highlight)
            finally:
                self._syncing = False
            self._publish()
            self._sync_selection()
        picked = session.highlight is not None
        self.subset_label.setText(f"{len(session.highlight):,} holes picked: {session.highlight_label}" if picked else "")
        self.clear_subset.setVisible(picked)

    def _publish(self):
        model = self.model
        self.session.set_order(model.order if model.review is not None else None)
        total = len(model.review.seeds_xy) if model.review is not None else 0
        self.count.setText(f"{len(model.order):,} of {total:,} holes" if total else "")

    def _selection_changed(self, reason):
        if reason != "table":
            self._sync_selection()

    def _sync_selection(self):
        """The current row follows the selected hole (no row when the filter hides it)."""
        row = self.model.row_of_hole(self.session.selected)
        self._syncing = True
        try:
            if row < 0:
                self.table.clearSelection()
            elif self.table.currentIndex().row() != row:
                # Exactly this row: selectRow would extend the selection while Shift is held (Shift+N).
                flags = QtCore.QItemSelectionModel.SelectionFlag
                self.table.selectionModel().setCurrentIndex(self.model.index(row, 0), flags.ClearAndSelect | flags.Rows)
                self.table.scrollTo(self.model.index(row, 0))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------------------------ user input
    def _row_changed(self, current, previous):
        # Clicks, arrow keys and Page keys in the table select the hole of the current row; the table stays.
        if self._syncing or not current.isValid():
            return
        hole = self.model.hole_at(current.row())
        if hole != self.session.selected:
            self.session.select(hole, "table")

    def _clicked(self, index):
        # A click on another method's CD column shows that method too.
        method = self.model.method_of_column(index.column())
        if method is not None and method != self.session.method:
            hole = self.model.hole_at(index.row())
            self.session.select(hole, "table")
            self.session.set_method(method)

    def _find(self):
        text = self.find.text().strip()
        review = self.session.review
        if text and review is not None:
            hole = int(text)
            if 0 <= hole < len(review.seeds_xy):
                self.session.select(hole, "inspector")
            else:
                self.session.status.emit(f"No hole {hole}: holes are numbered 0 to {len(review.seeds_xy) - 1}.", "warning")

    def selected_rows(self):
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def copy(self, rows=None):
        """Copy rows (default: the selected rows, else the selected hole's row) as tab-separated text."""
        model = self.model
        if model.review is None:
            return 0
        if rows is None:
            rows = self.selected_rows()
            if not rows:
                row = model.row_of_hole(self.session.selected)
                rows = [row] if row >= 0 else []
        if not rows:
            return 0
        lines = ["\t".join(model.headers())] + ["\t".join(values) for values in model.rows_text(rows)]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines) + "\n")
        self.session.status.emit(f"Copied {len(rows):,} row{'s' * (len(rows) != 1)}", "info")
        return len(rows)

    def _context_menu(self, position):
        index = self.table.indexAt(position)
        menu = QtWidgets.QMenu(self)
        if index.isValid():
            value = self.model.text(index.row(), index.column())
            menu.addAction(f"Copy value ({value})", lambda: QtWidgets.QApplication.clipboard().setText(value))
        menu.addAction("Copy selected rows (Ctrl+C)", self.copy)
        menu.addAction(f"Copy all {len(self.model.order):,} rows shown",
                       lambda: self.copy(list(range(len(self.model.order)))))
        method = self.model.method_of_column(index.column()) if index.isValid() else None
        if method is not None and method != self.session.method:
            menu.addAction(f"Show {method_name(method)}", lambda: self.session.set_method(method))
        menu.exec(self.table.viewport().mapToGlobal(position))
