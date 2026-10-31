"""CD distribution (bottom): the current method's CDs as a histogram; dragging a range picks those holes in the
table and on the image."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..names import method_color, method_name
from ..widgets import hint, plot_widget


class DistributionPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.summary = hint("Measure CD to see the distribution of the current method.")
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        row.addWidget(self.summary, 1)
        self.pick = QtWidgets.QPushButton("Pick a range")
        self.pick.setCheckable(True)
        self.pick.setToolTip("Drag the shaded range: the table and the image show only those holes")
        self.pick.toggled.connect(self._pick_toggled)
        row.addWidget(self.pick)
        layout.addLayout(row)
        self.plot = plot_widget(left="Holes")
        self.plot.setMouseEnabled(x=True, y=False)
        layout.addWidget(self.plot, 1)
        self.bars = pg.BarGraphItem(x=[], height=[], width=1, brush="#4d6f8f", pen=None)
        self.plot.addItem(self.bars)
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(255, 255, 255, 40))
        self.region.setZValue(10)
        self.region.sigRegionChangeFinished.connect(self._range_changed)
        self.plot.addItem(self.region)
        self.region.hide()
        self.mean_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ffffff", style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.mean_line)
        self._values = None  # (CDs in the length unit, NaN where failed)
        session.results_changed.connect(self.refresh)
        session.method_changed.connect(self.refresh)
        session.highlight_changed.connect(self._highlight_changed)

    def refresh(self):
        session = self.session
        review, method = session.review, session.method
        measured = review.measurements.get(method) if review is not None else None
        values = measured.metrics.get("cd_equivalent_px") if measured is not None else None
        if values is None or not len(values):
            self._values = None
            self.bars.setOpts(x=[], height=[])
            self.mean_line.hide()
            self.region.hide()
            self.pick.setEnabled(False)
            self.summary.setText("Measure CD to see the distribution of the current method.")
            return
        scale, unit = review.parameters["pixel_size"], review.parameters["unit"]
        good = measured.valid & np.isfinite(values)
        cds = np.where(good, values * scale, np.nan)
        self._values = cds
        self.pick.setEnabled(bool(good.any()))
        failed = int((~good).sum())
        if not good.any():
            self.bars.setOpts(x=[], height=[])
            self.mean_line.hide()
            self.summary.setText(f"{method_name(method)}: every hole failed ({failed:,}).")
            return
        valid = cds[good]
        low, high = np.percentile(valid, (0.5, 99.5))
        if high <= low:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, 81)
        counts, _ = np.histogram(np.clip(valid, low, high), edges)
        color = method_color(list(review.measurements).index(method))
        self.bars.setOpts(x=(edges[:-1] + edges[1:]) / 2, height=counts, width=edges[1] - edges[0], brush=color)
        self.plot.setLabel("bottom", f"{method_name(method)} CD ({unit})")
        self.plot.setXRange(low, high, padding=0.03)
        self.plot.enableAutoRange(axis="y")
        mean, spread = float(np.mean(valid)), float(np.std(valid))
        self.mean_line.setValue(mean)
        self.mean_line.show()
        outside = int(((valid < low) | (valid > high)).sum())
        self.summary.setText(
            f"<b>{method_name(method)}</b>: {len(valid):,} measured, "
            + (f"<span style='color:#ff5c74'>{failed:,} failed</span>" if failed else "none failed")
            + f" | mean {mean:.4g} {unit}, sd {spread:.3g} {unit} | 1-99 %: "
              f"{np.percentile(valid, 1):.4g} to {np.percentile(valid, 99):.4g} {unit}"
            + (f" | {outside:,} beyond the plot, counted at its ends" if outside else ""))
        if self.pick.isChecked():
            self.region.setRegion(np.percentile(valid, (25, 75)))

    def _pick_toggled(self, picking):
        if picking and self._values is not None:
            valid = self._values[np.isfinite(self._values)]
            if valid.size:
                with QtCore.QSignalBlocker(self.region):
                    self.region.setRegion(np.percentile(valid, (25, 75)))
            self.region.show()
            self._range_changed()
        else:
            self.region.hide()
            if self.session.highlight_label.startswith("CD "):
                self.session.set_highlight(None)

    def _range_changed(self):
        if self._values is None or not self.pick.isChecked():
            return
        low, high = self.region.getRegion()
        holes = np.flatnonzero((self._values >= low) & (self._values <= high))
        unit = self.session.review.parameters["unit"]
        self.session.set_highlight(holes, f"CD {low:.4g} to {high:.4g} {unit}")

    def _highlight_changed(self):
        # Another pick (a pitch point) or "Show all" ends the range pick.
        if self.pick.isChecked() and not self.session.highlight_label.startswith("CD "):
            with QtCore.QSignalBlocker(self.pick):
                self.pick.setChecked(False)
            self.region.hide()
