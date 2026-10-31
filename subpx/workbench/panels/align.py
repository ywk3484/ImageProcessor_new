"""Align view: the comparison on the main image area (flicker, differences, checkerboard, blend) and, on the
right, the shift, RMSE, the batch's hole check and the correlation peak with its runner-up."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets
import pyqtgraph as pg

from ..._workbench_data import HOLE_CHECK_MAX_ERROR
from ..canvas import ImageCanvas
from ..widgets import BACKGROUND, hint

MODES = (("Flicker", "flicker"), ("Difference before", "before"), ("Difference after", "after"),
         ("Checkerboard", "checker"), ("Blend", "blend"))
PEAK_WINDOW = 32  # lags shown around the correlation peak


class AlignView(QtWidgets.QWidget):
    """The comparison of the active page (reference) and the moving page at full size."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        self.mode = QtWidgets.QButtonGroup(self)
        self.mode_buttons = {}
        for i, (text, key) in enumerate(MODES):
            button = QtWidgets.QToolButton()
            button.setObjectName("view")
            button.setText(text + (" (Space)" if key == "flicker" else ""))
            button.setCheckable(True)
            button.setChecked(i == 0)
            self.mode.addButton(button, i)
            self.mode_buttons[key] = button
            row.addWidget(button)
        self.mode.idToggled.connect(lambda i, checked: checked and self.render())
        self.block = QtWidgets.QSpinBox()
        self.block.setRange(4, 1024)
        self.block.setValue(64)
        self.block.setPrefix("Block ")
        self.block.setSuffix(" px")
        self.block.setKeyboardTracking(False)
        self.block.valueChanged.connect(self.render)
        row.addWidget(self.block)
        self.showing = QtWidgets.QLabel("")
        self.showing.setStyleSheet("font-weight: 600;")
        # what is on screen; it may be cut on a narrow window, never pushing the buttons
        self.showing.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        row.addWidget(self.showing, 1)
        layout.addLayout(row)
        self.empty = hint("Pick a moving page in Setup: it is read and aligned to the active page (the reference).")
        layout.addWidget(self.empty)
        body = QtWidgets.QHBoxLayout()
        self.canvas = ImageCanvas(keep_zoom=True)
        body.addWidget(self.canvas, 1)
        self.scale = pg.GraphicsLayoutWidget()
        self.scale.setBackground(BACKGROUND)
        self.scale.setFixedWidth(88)
        self.bar = pg.ColorBarItem(values=(-1, 1), colorMap=pg.colormap.get("CET-D1"), width=14, interactive=False)
        self.scale.addItem(self.bar)
        self.bar.axis.setLabel("moving - reference")
        body.addWidget(self.scale)
        layout.addLayout(body, 1)
        self.flicker_aligned = True  # flicker shows the aligned moving page (False: the reference)
        self._levels = None
        self._fitted = False
        session.registration_changed.connect(self.render)
        session.moving_changed.connect(self.render)
        session.image_changed.connect(self._image_changed)

    def _image_changed(self, new_file):
        self._levels = None
        if new_file:
            self._fitted = False
        self.render()

    def current_mode(self):
        return MODES[self.mode.checkedId()][1]

    def flicker(self):
        """Space: swap the reference and the aligned moving page (and show flicker)."""
        if self.current_mode() != "flicker":
            self.mode_buttons["flicker"].setChecked(True)
        else:
            self.flicker_aligned = not self.flicker_aligned
            self.render()

    def images(self):
        """(reference, moving as aligned, aligned?) or None; before alignment the moving page as it is."""
        session = self.session
        if session.loaded is None or session.moving is None:
            return None
        reference = np.asarray(session.loaded.image, float)
        if session.registration is not None:
            return reference, session.registration.aligned, True
        moving = np.asarray(session.moving.image, float)
        return (reference, moving, False) if moving.shape == reference.shape else None

    def render(self, *args):
        images = self.images()
        mode = self.current_mode()
        self.block.setVisible(mode == "checker")
        if images is None:
            self.canvas.clear()
            self.scale.hide()
            self.showing.clear()
            self.empty.show()
            return
        self.empty.hide()
        reference, moving, aligned = images
        registration = self.session.registration
        if self._levels is None:
            sample = reference[::max(1, reference.shape[0] // 512), ::max(1, reference.shape[1] // 512)]
            low, high = np.percentile(sample[np.isfinite(sample)], (1, 99.5))
            self._levels = (float(low), float(high if high > low else low + 1))
        name = "aligned moving page" if aligned else "moving page (not aligned yet)"
        difference = mode in ("before", "after")
        if mode == "flicker":
            data = moving if self.flicker_aligned else reference
            self.showing.setText(name.capitalize() if self.flicker_aligned else "Reference (active page)")
        elif difference:
            if registration is not None:
                data = registration.difference_before if mode == "before" else registration.difference_after
            else:
                data = moving - reference
            self.showing.setText("Moving - reference, " + ("before" if mode == "before" or not aligned else "after")
                                 + " alignment")
        elif mode == "checker":
            block = self.block.value()
            yy, xx = np.indices(reference.shape)
            data = np.where(((yy // block + xx // block) % 2) == 0, reference, moving)
            self.showing.setText(f"Checkerboard: reference / {name}")
        else:
            data = 0.5 * reference + 0.5 * moving
            self.showing.setText(f"Blend: reference + {name}")
        self.showing.setToolTip(self.showing.text())
        self.canvas.set_image(data, fit=not self._fitted, keep_levels=True)
        self._fitted = True
        if difference:
            if registration is not None:
                overlap = registration.overlap
                spread = [np.percentile(np.abs(registration.difference_before[overlap]), 99),
                          np.nanpercentile(np.abs(registration.difference_after), 99)]
            else:
                spread = [np.percentile(np.abs(data), 99)]
            limit = max(max(float(value) for value in spread), 1e-9)  # one scale for before and after
            self.canvas.image_item.setLookupTable(pg.colormap.get("CET-D1").getLookupTable())
            self.canvas.image_item.setLevels((-limit, limit))
            self.bar.setLevels((-limit, limit))
            self.scale.show()
        else:
            self.canvas.image_item.setLookupTable(None)
            self.canvas.image_item.setLevels(self._levels)
            self.scale.hide()


class AlignPanel(QtWidgets.QWidget):
    """The alignment result in plain words (right, in the Align view)."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.pages = hint("")
        layout.addWidget(self.pages)
        self.shift = QtWidgets.QLabel("No alignment yet")
        self.shift.setObjectName("big")
        self.shift.setWordWrap(True)
        layout.addWidget(self.shift)
        self.meaning = hint("")
        layout.addWidget(self.meaning)
        self.rmse = QtWidgets.QLabel("")
        self.rmse.setWordWrap(True)
        layout.addWidget(self.rmse)
        self.check = QtWidgets.QLabel("")
        self.check.setWordWrap(True)
        layout.addWidget(self.check)
        title = QtWidgets.QLabel("Phase correlation around its peak")
        title.setStyleSheet("font-weight: 600; color: #90b2d2;")
        layout.addWidget(title)
        self.correlation = ImageCanvas()
        self.correlation.plot.setLabel("bottom", "dx lag", units="px")
        self.correlation.plot.setLabel("left", "dy lag", units="px")
        self.correlation.setMinimumHeight(200)
        layout.addWidget(self.correlation, 1)
        self.peaks = hint("")
        self.peaks.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.peaks)
        session.registration_changed.connect(self.refresh)
        session.moving_changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        session = self.session
        result, moving, loaded = session.registration, session.moving, session.loaded
        self.pages.setText(
            (f"Reference: page {loaded.page} of {loaded.name}" if loaded is not None else "No image")
            + (f"\nMoving: page {moving.page} of {moving.name}" if moving is not None else "\nMoving: none picked"))
        self.correlation.clear()
        if result is None:
            self.shift.setText("No alignment yet")
            self.meaning.setText("Pick a moving page in Setup; it is aligned to the reference right away.")
            for label in (self.rmse, self.check, self.peaks):
                label.clear()
            return
        dy, dx = result.result.shift_yx
        self.shift.setText(f"dy {dy:+.3f} px\ndx {dx:+.3f} px")
        self.meaning.setText(f"Shifting the moving page {abs(dx):.3f} px {'right' if dx >= 0 else 'left'} and "
                             f"{abs(dy):.3f} px {'down' if dy >= 0 else 'up'} puts it onto the reference.")
        self.rmse.setText(f"Overlap RMSE {result.rmse_before:.4g} → {result.rmse_after:.4g} (intensity units)")
        limit = HOLE_CHECK_MAX_ERROR[result.hole_check_mode]
        passes = result.hole_check_error < limit
        self.check.setObjectName("good" if passes else "warning")
        self.check.style().unpolish(self.check)
        self.check.style().polish(self.check)
        self.check.setText(f"Hole check ({result.hole_check_mode}, as the batch): {result.hole_check_error:.3f}, "
                           + (f"below {limit}: the batch would measure this page." if passes else
                              f"not below {limit}: the batch would skip this page as 'no holes'; "
                              "this shift may mean nothing."))
        self._draw_peaks(result.correlation)

    def _draw_peaks(self, correlation):
        h, w = correlation.shape
        center = np.array([h // 2, w // 2])
        best = np.array(np.unravel_index(int(np.argmax(correlation)), correlation.shape))
        masked = correlation.copy()
        y0, y1 = max(0, best[0] - 2), min(h, best[0] + 3)
        x0, x1 = max(0, best[1] - 2), min(w, best[1] + 3)
        masked[y0:y1, x0:x1] = -np.inf
        second = np.array(np.unravel_index(int(np.argmax(masked)), correlation.shape))
        top, bottom = max(0, best[0] - PEAK_WINDOW), min(h, best[0] + PEAK_WINDOW + 1)
        left, right = max(0, best[1] - PEAK_WINDOW), min(w, best[1] + PEAK_WINDOW + 1)
        crop = correlation[top:bottom, left:right]
        origin = (left - center[1], top - center[0])
        self.correlation.set_image(crop, origin=origin, categorical=True)
        self.correlation.image_item.setLevels((float(crop.min()), float(crop.max())))
        lag_best, lag_second = best - center, second - center
        self.correlation.add_overlay("best", pg.ScatterPlotItem(pos=[lag_best[::-1]], symbol="o", size=16,
                                                                brush=None, pen=pg.mkPen("w", width=2)))
        self.correlation.add_overlay("second", pg.ScatterPlotItem(pos=[lag_second[::-1]], symbol="x", size=14,
                                                                  pen=pg.mkPen("#f78c6b", width=2)))
        ratio = float(correlation[tuple(second)] / correlation[tuple(best)]) if correlation[tuple(best)] > 0 else 0
        warning = ratio > 0.8
        self.peaks.setText(
            f"Best peak (white) at lag dy {lag_best[0]:+d}, dx {lag_best[1]:+d} px; runner-up (orange) at "
            f"dy {lag_second[0]:+d}, dx {lag_second[1]:+d} px, {ratio:.2f} of the best."
            + ("<br><span style='color:#ffce73'>The runner-up is nearly as high: on a repeating pattern the shift "
               "may be off by whole pitches.</span>" if warning else ""))
