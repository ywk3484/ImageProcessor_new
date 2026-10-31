"""Pitch (bottom): the pitch result of the active page per method, in plain words and two large plots; a click
on a point picks its holes in the table and on the image."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..names import method_name
from ..widgets import hint, plot_widget

AXIS_COLORS = {"x": "#ffd166", "y": "#b39dff"}


class PitchPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Method"))
        self.method = QtWidgets.QComboBox()
        self.method.setMinimumContentsLength(14)
        self.method.setToolTip("Whose centers the pitch shown comes from")
        self.method.currentIndexChanged.connect(self.draw)
        row.addWidget(self.method)
        self.view = QtWidgets.QButtonGroup(self)
        for i, text in enumerate(("Pitch", "Placement error")):
            button = QtWidgets.QToolButton()
            button.setObjectName("view")
            button.setText(text)
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.view.addButton(button, i)
            row.addWidget(button)
        self.view.idToggled.connect(lambda i, checked: checked and self.draw())
        row.addStretch()
        layout.addLayout(row)
        self.summary = hint("Measure pitch to see it here. Pitch uses the CD result on screen (Setup, step 4).")
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.summary)
        plots = QtWidgets.QHBoxLayout()
        self.plots = {}
        for axis in ("x", "y"):
            plot = plot_widget()
            plot.scene().sigMouseClicked.connect(lambda event, axis=axis: self._clicked(axis, event))
            plots.addWidget(plot)
            self.plots[axis] = plot
        layout.addLayout(plots, 1)
        self._points = {}  # axis -> (x values, y values, kind) of the drawn points
        session.results_changed.connect(self._results_changed)
        session.method_changed.connect(self._follow_method)

    def _results_changed(self):
        pitch = self.session.pitch_review
        names = list(pitch.patterns) if pitch is not None else []
        old = self.method.currentData()
        with QtCore.QSignalBlocker(self.method):
            self.method.clear()
            for name in names:
                self.method.addItem(method_name(name), name)
            wanted = self.session.method if self.session.method in names else old
            self.method.setCurrentIndex(max(0, self.method.findData(wanted)))
        self.draw()

    def _follow_method(self):
        index = self.method.findData(self.session.method)
        if index >= 0 and index != self.method.currentIndex():
            self.method.setCurrentIndex(index)

    def pattern(self):
        pitch = self.session.pitch_review
        name = self.method.currentData()
        return pitch.patterns.get(name) if pitch is not None and name is not None else None

    def draw(self, *args):
        for plot in self.plots.values():
            plot.clear()
        self._points = {}
        pattern = self.pattern()
        if pattern is None:
            results = self.session.page_results
            self.summary.setText("Measure pitch to see it here. Pitch uses the CD result on screen (Setup, step 4)."
                                 if results is None or results.pitch is None else
                                 "Pitch indexing gave no result for this method.")
            return
        placement = self.view.checkedId() == 1
        unit = pattern.unit
        parts = []
        for axis, plot in self.plots.items():
            profile = pattern.profiles[axis]
            color = AXIS_COLORS[axis]
            nodes = profile.indices
            if placement:
                plot.setLabel("bottom", f"{'Column' if axis == 'x' else 'Row'} (logical node)")
                plot.setLabel("left", f"{axis.upper()} placement error", units=unit)
                plot.plot(nodes, profile.registration, pen=color, symbol="o", symbolSize=6, symbolBrush=color,
                          connect="finite")
                plot.addItem(pg.InfiniteLine(0, angle=0, pen="#586d84"))
                plot.addItem(pg.InfiniteLine(nodes[profile.reference_node],
                                             pen=pg.mkPen("#899caf", style=QtCore.Qt.PenStyle.DashLine)))
                self._points[axis] = (nodes.astype(float), profile.registration, "node")
            else:
                positions = (nodes[:-1] + nodes[1:]) / 2
                plot.setLabel("bottom", f"Between {'columns' if axis == 'x' else 'rows'} (logical interval)")
                plot.setLabel("left", f"{axis.upper()} pitch", units=unit)
                plot.plot(positions, profile.pitch, pen=color, symbol="o", symbolSize=6, symbolBrush=color,
                          connect="finite")
                plot.addItem(pg.ErrorBarItem(x=positions, y=profile.pitch, top=profile.std, bottom=profile.std,
                                             pen=pg.mkPen(color)))
                plot.addItem(pg.InfiniteLine(profile.nominal_pitch, angle=0,
                                             pen=pg.mkPen("#899caf", style=QtCore.Qt.PenStyle.DashLine)))
                self._points[axis] = (positions, profile.pitch, "interval")
            for item in plot.getPlotItem().axes.values():
                item["item"].enableAutoSIPrefix(False)
            pairs = int(profile.counts.sum())
            mean = float(np.nansum(profile.pitch * profile.counts) / max(pairs, 1))
            error = profile.registration[np.isfinite(profile.registration)]
            span = f", placement error {error.min():+.3g} to {error.max():+.3g} {unit}" if error.size else ""
            parts.append(f"<b>{axis} pitch {mean:.5g} {unit}</b> ({pairs:,} pairs; design "
                         f"{profile.nominal_pitch:.5g} {unit}{span})")
        self.summary.setText(
            f"{method_name(pattern.method)}: " + " &nbsp; ".join(parts)
            + f"<br>{pattern.layout} lattice, reference hole {pattern.reference_index}, "
              f"{pattern.meta.get('average_rows')} rows / {pattern.meta.get('average_columns')} columns averaged, "
              f"{pattern.meta.get('unindexed_seeds', 0)} holes not indexed. Click a point to pick its holes.")

    def _clicked(self, axis, event):
        pattern = self.pattern()
        if pattern is None or axis not in self._points:
            return
        plot = self.plots[axis]
        point = plot.getViewBox().mapSceneToView(event.scenePos())
        xs, ys, kind = self._points[axis]
        finite = np.isfinite(ys)
        if not finite.any():
            return
        candidates = np.flatnonzero(finite)
        target = int(candidates[np.argmin(np.abs(xs[candidates] - point.x()))])
        self.pick(axis, target, kind)

    def pick(self, axis, target, kind):
        """Pick the holes of a placement node (a column or row) or of a pitch interval (its pairs)."""
        pattern = self.pattern()
        profile = pattern.profiles[axis]
        dim = 1 if axis == "x" else 0
        if kind == "node":
            holes = profile.selected_indices[pattern.indices_rc[profile.selected_indices, dim] == target]
            label = f"{'column' if axis == 'x' else 'row'} {target} ({axis} placement)"
        else:
            pairs = profile.pairs[pattern.indices_rc[profile.pairs[:, 0], dim] == target]
            holes = np.unique(pairs)
            label = f"{axis} pitch between {'columns' if axis == 'x' else 'rows'} {target} and {target + 1}"
        session = self.session
        pitch = session.pitch_review
        shown = session.review
        if shown is not None and pitch is not None and (shown.cd is pitch.cd and shown.detection is pitch.detection):
            found = np.asarray(holes, int)
        else:  # pitch from another result: the holes of the result on screen at those positions
            found = session.holes_at(pattern.centers_xy[holes] + pitch.origin_xy)
        session.set_highlight(found, label)
        if len(found):
            middle = np.mean(shown.seeds_xy[found], axis=0)
            nearest = found[int(np.argmin(np.hypot(*(shown.seeds_xy[found] - middle).T)))]
            session.select(nearest, "pitch")
