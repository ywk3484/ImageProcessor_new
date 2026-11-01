"""Average hole (bottom): every hole of the result on screen stacked at its own subpixel position on a finer grid,
on the raw and on the preprocessed page, with the profiles through the centre."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..._workbench_data import half_height_width
from ...preprocess import average_hole
from ..names import method_name
from ..widgets import BACKGROUND, hint, no_si_prefix, plot_widget

COLORS = {"raw": "#a9bbce", "pre": "#4dd0e1"}
NAMES = {"raw": "raw page", "pre": "preprocessed page"}


class AveragePanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        row = QtWidgets.QHBoxLayout()
        self.make_button = QtWidgets.QPushButton("Average the holes")
        self.make_button.setToolTip("Stack every hole of the result on screen at its own centre, on the raw page "
                                    "and on the preprocessed page (when made)")
        self.make_button.clicked.connect(self.make)
        row.addWidget(self.make_button)
        row.addWidget(QtWidgets.QLabel("Grid"))
        self.upsample = QtWidgets.QSpinBox()
        self.upsample.setRange(1, 8)
        self.upsample.setValue(4)
        self.upsample.setPrefix("1/")
        self.upsample.setSuffix(" px")
        self.upsample.setToolTip("Bin width of the average: 1/4 px is 4 times finer than the pixels")
        row.addWidget(self.upsample)
        row.addStretch()
        layout.addLayout(row)
        self.summary = hint("Preview holes or measure CD, then average the holes: the mean hole shape on a grid finer "
                            "than the pixels, on the raw and on the preprocessed page.")
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.summary)
        body = QtWidgets.QHBoxLayout()
        self.image_plot = pg.PlotWidget(background=BACKGROUND)
        self.image_plot.setAspectLocked(True)
        self.image_plot.invertY(True)
        self.image_plot.setLabel("bottom", "x offset", units="px")
        self.image_plot.setLabel("left", "y offset", units="px")
        no_si_prefix(self.image_plot.getPlotItem())
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.image_item)
        body.addWidget(self.image_plot, 2)
        self.profiles = {}
        for axis in ("x", "y"):
            plot = plot_widget(bottom=f"{axis} offset from the centre (px)", left="Mean intensity")
            body.addWidget(plot, 3)
            self.profiles[axis] = plot
        layout.addLayout(body, 1)
        self._made = None  # dict(averages, review, method, page, holes)
        session.image_changed.connect(lambda new_file: self.clear())
        session.version_changed.connect(self.draw)
        session.results_changed.connect(self._update_button)
        session.method_changed.connect(self._update_button)
        session.busy_changed.connect(self._update_button)
        self._update_button()

    def clear(self):
        self._made = None
        self.image_item.clear()
        for plot in self.profiles.values():
            plot.clear()
        self.summary.setText("Preview holes or measure CD, then average the holes: the mean hole shape on a grid "
                             "finer than the pixels, on the raw and on the preprocessed page.")
        self._update_button()

    def _update_button(self, *args):
        session = self.session
        self.make_button.setEnabled(not session.busy and session.review is not None and len(session.review.seeds_xy) > 1)

    def centers(self):
        """Page [x, y] of the holes on screen: the current method's centres where it succeeded, else the seeds."""
        review, method = self.session.review, self.session.method
        measured = review.measurements.get(method)
        points = review.seeds_xy
        if measured is not None and review.cd is not None:
            good = measured.valid & np.isfinite(measured.centers_xy).all(axis=1)
            points = measured.centers_xy[good]
        return points + review.origin_xy

    def make(self):
        session = self.session
        if session.busy or session.review is None or session.loaded is None:
            return
        review, method, page = session.review, session.method, session.loaded.page
        centers, upsample = self.centers(), self.upsample.value()
        images = {"raw": session.loaded.image}
        if session.processed is not None:
            images["pre"] = session.processed.image

        def calculate():
            return {name: average_hole(image, centers, upsample=upsample) for name, image in images.items()}

        def accept(averages):
            self._made = dict(averages=averages, review=review, method=method, page=page,
                              version=session.version, holes=len(centers))
            self.draw()

        session.run_job("Averaging the holes", calculate, accept)

    def draw(self, *args):
        made = self._made
        if made is None:
            return
        averages = made["averages"]
        shown = self.session.version if self.session.version in averages else "raw"
        average = averages[shown]
        step = 1 / average.upsample
        start = average.offsets[0] - step / 2
        finite = average.image[np.isfinite(average.image)]
        self.image_item.setImage(average.image, autoLevels=False)
        if finite.size:
            self.image_item.setLevels((float(finite.min()), float(finite.max())))
        self.image_item.setRect(QtCore.QRectF(start, start, step * len(average.offsets), step * len(average.offsets)))
        self.image_plot.setTitle(f"Average hole, {NAMES[shown]}", color="#a9bbce", size="9pt")
        center = len(average.offsets) // 2
        widths = {}
        for axis, plot in self.profiles.items():
            plot.clear()
            for name, result in averages.items():
                values = result.image[center] if axis == "x" else result.image[:, center]
                filled = np.isfinite(values)  # empty bins (no hole at that subpixel phase) are left out
                if filled.any():
                    plot.plot(result.offsets[filled], values[filled], symbol="o", symbolSize=3,
                              pen=pg.mkPen(COLORS[name], width=2 if name == shown else 1),
                              symbolBrush=COLORS[name], symbolPen=None)
                widths[name, axis] = half_height_width(values[filled], result.offsets[filled])
        raw = averages["raw"].meta
        where = "preprocessed" if made["version"] == "pre" else "raw"
        parts = [f"{made['holes']:,} holes of {method_name(made['method'])} ({where} page result), page {made['page']}",
                 f"grid 1/{average.upsample} px, fewest pixels per bin {raw['fewest_samples']:,}"
                 + (f", {raw['empty_bins']:,} empty bins" if raw["empty_bins"] else ""),
                 f"brightness spread {100 * raw['brightness_spread']:.1f} %"]
        lines = " &nbsp; ".join(
            f"<span style='color:{COLORS[name]}'>&#9473; {NAMES[name]}: width at half height x "
            f"{widths[name, 'x']:.3f}, y {widths[name, 'y']:.3f} px</span>" for name in averages)
        if "pre" not in averages:
            lines += " &nbsp; (preprocess this page and average again to compare)"
        self.summary.setText(" | ".join(parts) + "<br>" + lines)
