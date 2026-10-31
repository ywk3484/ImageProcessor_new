"""The image view: the page (or every page, stacked), pipeline stages, the result's layers with a level of
detail, the CD colour scale and the analysis region."""

from __future__ import annotations

import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .._workbench_data import cell_boundary, cell_mask_map, cell_values_map
from .names import FAILED_COLOR, method_color, method_name, method_line
from .widgets import BACKGROUND, no_si_prefix


class ImageCanvas(QtWidgets.QWidget):
    """Image coordinates use integer pixel centers, x right and y downward.

    With ``keep_zoom`` a resize (a panel beside the image shown or hidden, the window resized) keeps the center
    and the zoom and shows more or less of the image, instead of rescaling the image to the old range.
    """

    clicked = QtCore.pyqtSignal(float, float)
    hovered = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None, *, histogram=False, keep_zoom=False):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.plot = pg.PlotWidget(background=BACKGROUND)
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
            self.histogram.setMinimumWidth(96)
            self.histogram.setMaximumWidth(110)
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
        self._view_state = self._view_size = None  # (center x, center y, data units per pixel), view size
        if keep_zoom:
            self.plot.getViewBox().sigRangeChanged.connect(self._remember_view)
            self.plot.getViewBox().sigResized.connect(self._keep_view)

    def _remember_view(self, *args):
        box = self.plot.getViewBox()
        size = (box.width(), box.height())
        if size == self._view_size and min(size) > 1:  # not the range change a resize makes
            (x0, x1), (y0, y1) = box.viewRange()
            self._view_state = ((x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / size[0])

    def _keep_view(self, *args):
        box = self.plot.getViewBox()
        width, height = box.width(), box.height()
        state, self._view_size = self._view_state, (width, height)
        if state is None or min(width, height) <= 1:
            self._remember_view()
            return
        x, y, scale = state
        box.setRange(xRange=(x - width * scale / 2, x + width * scale / 2),
                     yRange=(y - height * scale / 2, y + height * scale / 2), padding=0)

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

    def clear(self):
        self.image_item.clear()
        self._image_key = None
        self.clear_overlays()

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


def _segments(shapes):
    """(N, K, 2) outlines as one NaN-separated line."""
    return np.concatenate((shapes, np.full((len(shapes), 1, 2), np.nan)), axis=1).reshape(-1, 2)


class HoleLayers(QtCore.QObject):
    """The overlays of one result on the main image (spec 4.4).

    Raster maps (CD colors, failed cells, Voronoi edges, highlighted holes) are made once per result and method
    and shown at every zoom. Centers, contours, pitch pairs and rejected boxes are drawn for the holes in view
    only, and only while at most VECTOR_LIMIT holes are in view (all methods' contours: VECTOR_LIMIT in total);
    farther out the raster maps carry the picture. The selection marker is an item of its own: selecting a
    hole redraws nothing else.
    """

    VECTOR_LIMIT = 2000
    changed = QtCore.pyqtSignal()  # vectors appeared or gave way to the rasters

    def __init__(self, plot):
        super().__init__(plot)
        self.plot, self.view = plot, plot.getViewBox()
        self.cd_map = pg.ImageItem(axisOrder="row-major")
        self.cd_map.setOpacity(0.6)
        self.cd_map.setZValue(5)
        self.failures = _mask_item((255, 92, 116, 150), 6)
        self.edges = _mask_item((86, 214, 207, 165), 7)
        self.highlight = _mask_item((255, 255, 255, 110), 8)
        self.highlight_points = pg.ScatterPlotItem(size=14, pen=pg.mkPen("#ffffff", width=1.5), brush=None)
        self.centers = pg.ScatterPlotItem(size=6, pen=pg.mkPen("#56d6cf", width=1), brush=pg.mkBrush(86, 214, 207, 60),
                                          symbol="+")
        self.failed = pg.ScatterPlotItem(size=10, pen=pg.mkPen(FAILED_COLOR, width=2), brush=None, symbol="x")
        self.contours = pg.PlotDataItem(connect="finite")
        self.all_contours = {}  # method -> line, made when "All methods' contours" is first shown
        self.pairs = {axis: pg.PlotDataItem(connect="pairs", pen=pg.mkPen(color, width=1))
                      for axis, color in (("x", "#ffce73"), ("y", "#b49aff"))}
        self.reference = pg.ScatterPlotItem(symbol="star", size=18, brush="#ffce73", pen="#fff3cf")
        self.rejected = pg.PlotDataItem(connect="finite", pen=FAILED_COLOR)
        self.selection = pg.ScatterPlotItem(symbol="o", size=20, brush=None, pen=pg.mkPen("#ffffff", width=2))
        self.vectors = [self.centers, self.failed, self.contours, *self.pairs.values(), self.rejected]
        for z, item in enumerate([*self.vectors, self.highlight_points, self.reference, self.selection], start=20):
            item.setZValue(z)
        for item in (self.cd_map, self.failures, self.edges, self.highlight, *self.vectors, self.highlight_points,
                     self.reference, self.selection):
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

    def show(self, review, method, origin, layers, *, pitch=None, pitch_origin=None):
        """Show ``review``'s ``method`` with the named ``layers`` on; ``origin``: display position of its pixel [0, 0].

        ``pitch`` is the page's pitch result (pairs and reference); ``review=None`` hides every layer.
        """
        if review is not self._review:
            self._review, self._maps = review, {}
        if review is None:
            self.state = None
            for item in (self.cd_map, self.failures, self.edges, *self.vectors, self.reference,
                         *self.all_contours.values()):
                item.hide()
            self.changed.emit()
            return
        measured = review.measurements[method]
        origin = np.asarray(origin, float)
        points = measured.centers_xy + origin
        pattern = pitch.patterns.get(method) if pitch is not None and "Pitch pairs" in layers else None
        self.state = dict(review=review, method=method, origin=origin, layers=set(layers), points=points,
                          seeds=review.seeds_xy + origin, valid=measured.valid & np.isfinite(points).all(axis=1),
                          margin=np.sqrt(review.image.size / max(len(review.seeds_xy), 1)), pattern=pattern,
                          pattern_points=(pattern.centers_xy + np.asarray(pitch_origin, float)
                                          if pattern is not None else None))
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
        self.contours.setPen(pg.mkPen(method_color(list(review.measurements).index(method)), width=1.3))
        self._update_vectors()

    def show_highlight(self, holes):
        """Highlight holes of the shown result (a CD range, a pitch point); None clears."""
        state = self.state
        self.highlight.hide()
        self.highlight_points.hide()
        if state is None or holes is None or not len(holes):
            return
        review = state["review"]
        if review.cd is not None:
            selected = np.zeros(len(review.seeds_xy), bool)
            selected[holes] = True
            _set_raster(self.highlight, cell_mask_map(review.cd.cell_labels, selected), state["origin"])
        else:
            self.highlight_points.setData(pos=state["seeds"][holes])
            self.highlight_points.show()

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
            segments = _segments(measured.contours_xy[inside] + state["origin"])
            self.contours.setData(segments[:, 0], segments[:, 1], connect="finite")
        self.contours.setVisible(contours)
        self._update_all_contours(inside, near and "All methods' contours" in layers)
        pattern, pattern_points = state["pattern"], state["pattern_points"]
        for axis, item in self.pairs.items():
            pairs = pattern.profiles[axis].pairs if pattern is not None else np.empty((0, 2), int)
            shown = near and len(pairs) > 0
            if shown:
                ends = pattern_points[pairs]
                keep = ((ends[..., 0] >= x0) & (ends[..., 0] <= x1) & (ends[..., 1] >= y0) & (ends[..., 1] <= y1)).any(axis=1)
                segments = ends[keep].reshape(-1, 2)
                item.setData(segments[:, 0], segments[:, 1], connect="pairs")
            item.setVisible(shown)
        if pattern is not None:
            self.reference.setData(pos=[pattern_points[pattern.reference_index]])
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

    def _update_all_contours(self, inside, wanted):
        """Every method's contour in its colour, while at most VECTOR_LIMIT contours are drawn in all."""
        state = self.state
        names = [name for name, measured in state["review"].measurements.items() if measured.contours_xy.size]
        wanted = wanted and int(inside.sum()) * max(len(names), 1) <= self.VECTOR_LIMIT
        for name, item in self.all_contours.items():
            item.setVisible(wanted and name in names and name != state["method"])
        if not wanted:
            return
        for index, name in enumerate(state["review"].measurements):
            if name not in names or name == state["method"]:
                continue
            item = self.all_contours.get(name)
            if item is None:
                item = self.all_contours[name] = pg.PlotDataItem(connect="finite")
                item.setZValue(19)
                self.plot.addItem(item)
            item.setPen(pg.mkPen(method_color(index), width=0.8))
            segments = _segments(state["review"].measurements[name].contours_xy[inside] + state["origin"])
            item.setData(segments[:, 0], segments[:, 1], connect="finite")
            item.show()

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
    resolution when zoomed in; the session's PageSource keeps the previews of every page seen (small) and full
    pages up to its FULL_BYTES. The active page is the canvas's own image and is never drawn here. Decoding runs
    a page at a time between events, so the window stays responsive while a large file fills in.
    """

    FULL_SCALE = 3.0  # image pixels per screen pixel below which pages in view are drawn at full resolution
    LABEL_PAGES = 12  # page labels while at most this many pages are in view
    failed = QtCore.pyqtSignal(str)

    def __init__(self, plot, source):
        super().__init__(plot)
        self.plot, self.view, self.source = plot, plot.getViewBox(), source
        self.shapes = np.array(source.shapes, int).reshape(-1, 2)
        gap = max(8, round(int(self.shapes[:, 1].max()) * 0.04))
        self.offsets = np.concatenate(([0], np.cumsum(self.shapes[:-1, 0] + gap)))
        self.active, self.levels = None, None
        self.items, self.labels = {}, {}
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
        self._update_timer.start()

    def busy(self):
        return bool(self._queue or self._decode_timer.isActive() or self._update_timer.isActive())

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

    def close(self):
        self._update_timer.stop()
        self._decode_timer.stop()
        self.view.sigRangeChanged.disconnect(self._schedule)
        for item in (*self.items.values(), *self.labels.values(), self.borders, self.active_border):
            self.plot.removeItem(item)
        self.items, self.labels, self._queue = {}, {}, []

    def _in_view(self):
        (x0, x1), (y0, y1) = self.view.viewRange()
        pages = np.flatnonzero((self.offsets + self.shapes[:, 0] >= y0) & (self.offsets <= y1) & (x0 <= self.shapes[:, 1]))
        full = self.view.viewPixelSize()[1] < self.FULL_SCALE
        return pages, full

    def update(self):
        if self.active is None:
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
        loaded = self.source.cached(page) if full else None
        preview = self.source.preview(page)
        if loaded is not None:
            data, factor = loaded.image, 1
        elif preview is not None and (not full or preview[1] == 1):
            data, factor = preview
        elif preview is not None:  # full resolution wanted: the preview until it is decoded
            self._show(page, *preview)
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
                self.source.read(page)
            except Exception as exc:
                self.broken.add(page)
                self.failed.emit(f"Page {page} could not be read: {exc}")
                continue
            self._draw(page, full)
        if self._queue:
            self._decode_timer.start()

    def memory(self):
        return self.source.memory()


LAYERS = (
    ("Centers", True, "Hole centers; failed holes as red crosses (while at most 2,000 holes are in view)"),
    ("Contours", True, "The shown method's contours (while at most 2,000 holes are in view)"),
    ("All methods' contours", False, "Every method's contour in its colour (while at most 2,000 contours fit)"),
    ("Failed cells", True, "Cells of the holes the shown method failed on, in red"),
    ("Voronoi edges", False, "The cell boundaries"),
    ("Rejected boxes", False, "Components outside the area limits (previewed holes)"),
    ("Pitch pairs", False, "The hole pairs of the pitch result and its reference hole"),
    ("CD colors", False, "Each cell colored by the shown method's CD (scale beside the image)"),
)
STAGES = ("Image", "Threshold mask", "Component labels", "Voronoi cells", "Background model")
PAN_REASONS = {"table", "key", "failed", "inspector", "pitch", "distribution"}


class MeasureView(QtWidgets.QWidget):
    """The main image with the shown result's layers, linked to the session."""

    hovered = QtCore.pyqtSignal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(4, 0, 4, 0)
        row.addWidget(QtWidgets.QLabel("Show"))
        self.stage = QtWidgets.QComboBox()
        self.stage.addItems(STAGES)
        self.stage.setToolTip("The image or a stage of the hole finding and measurement")
        self.stage.currentIndexChanged.connect(self.render)
        row.addWidget(self.stage)
        row.addWidget(QtWidgets.QLabel("Method"))
        self.method = QtWidgets.QComboBox()
        self.method.setMinimumContentsLength(16)
        self.method.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method.setToolTip("The method shown on the image, in the table and in the inspector")
        self.method.currentIndexChanged.connect(self._method_picked)
        row.addWidget(self.method)
        self.layers_button = QtWidgets.QToolButton()
        self.layers_button.setText("Layers")
        self.layers_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.layers_menu = QtWidgets.QMenu(self.layers_button)
        self.layers_button.setMenu(self.layers_menu)
        self.layer_checks = {}
        for name, checked, tip in LAYERS:
            action = self.layers_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(checked)
            action.setToolTip(tip)
            action.toggled.connect(self.render_layers)
            self.layer_checks[name] = action
        self.layers_menu.setToolTipsVisible(True)
        self.layers_menu.addSeparator()
        self.region_action = self.layers_menu.addAction("Analysis region")
        self.region_action.setCheckable(True)
        self.region_action.setToolTip("Measure inside a rectangle only; drag its corners")
        self.region_action.toggled.connect(self._toggle_region)
        row.addWidget(self.layers_button)
        self.hint = QtWidgets.QLabel()
        self.hint.setObjectName("hint")
        self.hint.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        row.addWidget(self.hint, 1)
        layout.addLayout(row)

        self.canvas = ImageCanvas(histogram=True, keep_zoom=True)
        self.canvas.clicked.connect(self._clicked)
        self.canvas.hovered.connect(self._hovered)
        self.lock_levels = QtWidgets.QCheckBox("Lock")
        self.lock_levels.setToolTip("Keep this contrast on every page (otherwise each page is stretched to 1-99.5 %)")
        self.lock_levels.toggled.connect(self._lock_levels_changed)
        self.canvas.side_layout.addWidget(self.lock_levels)
        self.canvas.histogram.sigLevelsChanged.connect(self._levels_changed)
        self._locked_levels = None
        self.layers = HoleLayers(self.canvas.plot)
        self.layers.changed.connect(self._update_hint)
        # CD color scale beside the image: 1-99 % of the shown method, drag its ends to set, Lock to compare.
        self.cd_scale = QtWidgets.QWidget()
        self.cd_scale.setFixedWidth(88)
        scale_layout = QtWidgets.QVBoxLayout(self.cd_scale)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_view = pg.GraphicsLayoutWidget()
        scale_view.setBackground(BACKGROUND)
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
        self._cd_levels = None
        self.canvas.layout().insertWidget(1, self.cd_scale)
        layout.addWidget(self.canvas, 1)

        self.roi = pg.RectROI((10, 10), (100, 100), pen=pg.mkPen("#ffce73", width=2, style=QtCore.Qt.PenStyle.DashLine))
        for corner, anchor in (((0, 0), (1, 1)), ((0, 1), (1, 0)), ((1, 0), (0, 1))):
            self.roi.addScaleHandle(corner, anchor)
        self.roi.setZValue(50)
        self.roi.sigRegionChangeFinished.connect(self._region_changed)
        self.canvas.plot.addItem(self.roi)
        self.roi.hide()
        self._region = None  # (page-local position, size) of the region as last set by the user
        self.stack = None
        for keys, step in (("PgUp", -1), ("PgDown", 1)):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, activated=lambda step=step: self._page_step(step),
                            context=QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        session.image_changed.connect(self._image_changed)
        session.results_changed.connect(self._results_changed)
        session.method_changed.connect(self._method_changed)
        session.selection_changed.connect(self._selection_changed)
        session.highlight_changed.connect(self._highlight_changed)
        session.stacked_changed.connect(self._stacked_changed)

    # ------------------------------------------------------------------------------------------ pages
    def offset(self):
        if self.stack is not None and self.session.loaded is not None:
            return np.array([0, self.stack.offsets[self.session.loaded.page]])
        return np.zeros(2)

    def _page_step(self, step):
        if self.session.loaded is not None:
            self.session.set_page(self.session.loaded.page + step)

    def _image_changed(self, new_file):
        loaded = self.session.loaded
        previous_offset = getattr(self, "_shown_offset", np.zeros(2))
        if self.stack is not None:
            if new_file:
                self.stack.close()
                self.stack = None
            else:
                self.stack.set_active(loaded.page)
        h, w = loaded.image.shape
        offset = self.offset()
        shape_kept = getattr(self, "_shown_shape", None) == loaded.image.shape
        with QtCore.QSignalBlocker(self.roi):
            if not new_file:
                # The same region on the new page, moved inside it when the page is smaller.
                position, size = self._region or (self.roi.pos() - QtCore.QPointF(*previous_offset), self.roi.size())
                size = QtCore.QPointF(min(size.x(), w), min(size.y(), h))
                x = min(max(position.x(), -0.5), w - 0.5 - size.x())
                y = min(max(position.y(), -0.5), h - 0.5 - size.y())
                self.roi.setSize((size.x(), size.y()))
                self.roi.setPos((x, y + offset[1]))
            else:
                with QtCore.QSignalBlocker(self.region_action):
                    self.region_action.setChecked(False)
                self.roi.hide()
                self._region = None
                self.roi.setPos((w * 0.25 - 0.5, h * 0.25 - 0.5 + offset[1]))
                self.roi.setSize((w * 0.5, h * 0.5))
            self.roi.maxBounds = QtCore.QRectF(-0.5, offset[1] - 0.5, w, h)
        if new_file:
            with QtCore.QSignalBlocker(self.stage):
                self.stage.setCurrentIndex(0)
        self._shown_offset, self._shown_shape = offset, loaded.image.shape
        self.render_base()
        if new_file or not (shape_kept or self.stack is not None):
            self.canvas.fit_image()
        if not new_file and self.region_action.isChecked():
            self.session.set_bounds(self.bounds())

    def _stacked_changed(self, stacked):
        session = self.session
        if session.loaded is None:
            return
        old_offset = self.offset()
        if stacked and self.stack is None and session.source is not None:
            self.stack = PageStack(self.canvas.plot, session.source)
            self.stack.failed.connect(session.log)
            self.stack.set_levels(self.canvas.image_item.getLevels())
            self.stack.set_active(session.loaded.page)
        elif not stacked and self.stack is not None:
            self.stack.close()
            self.stack = None
        offset = self.offset()
        with QtCore.QSignalBlocker(self.roi):
            self.roi.setPos(self.roi.pos() + QtCore.QPointF(*(offset - old_offset)))
            h, w = session.loaded.image.shape
            self.roi.maxBounds = QtCore.QRectF(-0.5, offset[1] - 0.5, w, h)
        self._shown_offset = offset
        self.render_base()
        self.fit_stack()  # the whole stack when it was turned on, the page when off

    def fit_page(self):
        self.canvas.fit_image()

    def fit_stack(self):
        """Shift+F: the whole stack in view (the page when the stack is off)."""
        if self.stack is not None:
            self.canvas.plot.getViewBox().setRange(self.stack.bounds(), padding=0.02)
        else:
            self.canvas.fit_image()

    def page_at(self, x, y):
        """(page number, page-local y) under a display position, or (None, y)."""
        loaded = self.session.loaded
        if loaded is None:
            return None, y
        if self.stack is not None:
            page = self.stack.page_at(x, y)
            return (None, y) if page is None else (page, y - self.stack.offsets[page])
        h, w = loaded.image.shape
        return (loaded.page, y) if -0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5 else (None, y)

    # -------------------------------------------------------------------------------------- drawing
    def active_layers(self):
        return {name for name, action in self.layer_checks.items() if action.isChecked()}

    def render(self, *args):
        self.render_base()

    def render_base(self, *args):
        """The image or pipeline stage, then the layers."""
        session, loaded = self.session, self.session.loaded
        if loaded is None:
            return
        review = session.review
        data, origin, categorical = loaded.image, np.zeros(2), False
        stage = self.stage.currentText()
        message = ""
        if stage != "Image":
            if review is None:
                message = "Preview holes or measure CD to see this stage."
            elif stage == "Threshold mask":
                data = review.cd.binary if review.cd is not None else review.detection.binary
                origin = review.origin_xy
                if review.cd is not None and review.cd.meta.get("seeds") == "peaks":
                    message = "Touching holes are found from peaks, not from a threshold mask."
            elif stage == "Component labels":
                data = review.cd.component_labels if review.cd is not None else review.detection.labels
                origin, categorical = review.origin_xy, True
            elif review.cd is None:
                message = "Measure CD to see Voronoi cells and the background."
            elif stage == "Voronoi cells":
                data, origin, categorical = review.cd.cell_labels, review.origin_xy, True
            elif review.cd.background_map is not None:
                data, origin = review.cd.background_map, review.origin_xy
            else:
                message = "This background mode has no background map."
        offset = self.offset()
        locked = self.lock_levels.isChecked() and self._locked_levels is not None and stage == "Image"
        self.canvas.set_image(data, origin + offset, fit=False, categorical=categorical, keep_levels=locked)
        if locked:
            self.canvas.histogram.setLevels(*self._locked_levels)
        if self.stack is not None and stage == "Image":
            self.stack.set_levels(self.canvas.image_item.getLevels())
        self._stage_message = message
        self.render_layers()

    def render_layers(self, *args):
        session, loaded = self.session, self.session.loaded
        if loaded is None:
            return
        review, method = session.review, session.method
        shown = review is not None and len(review.seeds_xy) > 0 and method in review.measurements
        offset = self.offset()
        pitch = session.pitch_review
        layers = self.active_layers()
        self.layers.show(review if shown else None, method, (review.origin_xy + offset) if shown else offset, layers,
                         pitch=pitch, pitch_origin=(pitch.origin_xy + offset) if pitch is not None else None)
        self.layers.show_highlight(session.highlight if shown else None)
        self._update_selection()
        self._update_cd_scale(review if shown and "CD colors" in layers else None, method)
        self._update_hint()

    def _update_hint(self):
        text = getattr(self, "_stage_message", "")
        if self.layers.hidden_vectors():
            text = (text + " " if text else "") + (
                f"{self.layers.in_view:,} holes in view: zoom in to see centers and contours "
                f"(at most {self.layers.VECTOR_LIMIT:,})")
        self.hint.setText(text)

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
            self.render_layers()

    def _lock_levels_changed(self, locked):
        self._locked_levels = tuple(self.canvas.image_item.getLevels()) if locked else None

    def _levels_changed(self):
        if self.stage.currentText() != "Image":
            return
        if self.lock_levels.isChecked():
            self._locked_levels = tuple(self.canvas.histogram.getLevels())
        if self.stack is not None:
            self.stack.set_levels(self.canvas.histogram.getLevels())  # one contrast for every page

    def _update_selection(self):
        """Move the selection marker only; no other layer is redrawn."""
        session = self.session
        review = session.review
        if review is None or session.selected is None or not 0 <= session.selected < len(review.seeds_xy):
            self.layers.set_selection(None)
        else:
            self.layers.set_selection(review.seeds_xy[session.selected] + review.origin_xy + self.offset())

    # -------------------------------------------------------------------------------- session events
    def _results_changed(self):
        names = self.session.methods()
        with QtCore.QSignalBlocker(self.method):
            self.method.clear()
            for name in names:
                self.method.addItem(method_name(name), name)
                self.method.setItemData(self.method.count() - 1, f"{name}: {method_line(name)}",
                                        QtCore.Qt.ItemDataRole.ToolTipRole)
            self.method.setCurrentIndex(max(0, self.method.findData(self.session.method)))
        self.method.setEnabled(bool(names))
        self.render_base()

    def _method_changed(self):
        index = self.method.findData(self.session.method)
        if index != self.method.currentIndex():
            with QtCore.QSignalBlocker(self.method):
                self.method.setCurrentIndex(index)
        self.render_layers()

    def _method_picked(self, index):
        name = self.method.itemData(index)
        if name is not None:
            self.session.set_method(name)

    def _selection_changed(self, reason):
        self._update_selection()
        session = self.session
        if reason in PAN_REASONS and session.selected is not None:
            review = session.review
            self.reveal(review.seeds_xy[session.selected] + review.origin_xy + self.offset())

    def _highlight_changed(self):
        session = self.session
        self.layers.show_highlight(session.highlight if self.layers.state is not None else None)

    def reveal(self, point):
        """Pan, keeping the zoom, when the point lies outside the view."""
        view = self.canvas.plot.getViewBox()
        (x0, x1), (y0, y1) = view.viewRange()
        if not (x0 <= point[0] <= x1 and y0 <= point[1] <= y1):
            dx, dy = (x1 - x0) / 2, (y1 - y0) / 2
            view.setRange(xRange=(point[0] - dx, point[0] + dx), yRange=(point[1] - dy, point[1] + dy), padding=0)

    def zoom_selected(self):
        session = self.session
        review = session.review
        if review is None or session.selected is None or not len(review.seeds_xy):
            return
        point = review.seeds_xy[session.selected] + review.origin_xy + self.offset()
        radius = max(8, max(review.cd.masks.shape[1:]) * 1.5) if review.cd is not None else 15
        self.canvas.plot.setRange(xRange=(point[0] - radius, point[0] + radius),
                                  yRange=(point[1] - radius, point[1] + radius), padding=0)

    def _clicked(self, x, y):
        session = self.session
        if session.loaded is None:
            return
        page, local_y = self.page_at(x, y)
        if page is None:
            return
        if page != session.loaded.page:
            session.set_page(page)
        else:
            session.select_at(x, local_y)

    def _hovered(self, x, y):
        session = self.session
        if session.loaded is None:
            return
        page, local_y = self.page_at(x, y)
        if page is None:
            return
        ix, iy = int(np.floor(x + .5)), int(np.floor(local_y + .5))
        prefix = f"Page {page} | " if session.loaded.pages > 1 else ""
        # Other pages of the stack: the value when the page is in memory at full resolution.
        loaded = session.loaded if page == session.loaded.page else session.source.cached(page)
        value = f" = {loaded.image[iy, ix]:.6g}" if loaded is not None else ""
        self.hovered.emit(f"{prefix}x={x:.2f}, y={local_y:.2f} px | pixel [{iy}, {ix}]{value}")

    # ---------------------------------------------------------------------------------------- region
    def _toggle_region(self, enabled):
        self.roi.setVisible(enabled)
        self.session.set_bounds(self.bounds())

    def _region_changed(self):
        # The region as the user set it (page-local), so a smaller page in between cannot shrink it.
        self._region = (self.roi.pos() - QtCore.QPointF(*self.offset()), self.roi.size())
        self.session.set_bounds(self.bounds())

    def bounds(self):
        """The analysis region as page-local [x0, y0, x1, y1], or None when it is off."""
        loaded = self.session.loaded
        if not self.region_action.isChecked() or loaded is None:
            return None
        h, w = loaded.image.shape
        position = self.roi.pos() - QtCore.QPointF(*self.offset())
        size = self.roi.size()
        # Integer-centered pixels selected by the ROI's visual half-pixel edges.
        x0 = max(0, min(w, int(np.ceil(position.x()))))
        y0 = max(0, min(h, int(np.ceil(position.y()))))
        x1 = max(0, min(w, int(np.ceil(position.x() + size.x()))))
        y1 = max(0, min(h, int(np.ceil(position.y() + size.y()))))
        return [x0, y0, x1, y1]

    def set_region(self, bounds):
        """Show the analysis region [x0, y0, x1, y1] (None: off), e.g. when settings are reverted."""
        if bounds is None:
            self.region_action.setChecked(False)
            return
        x0, y0, x1, y1 = bounds
        offset = self.offset()
        with QtCore.QSignalBlocker(self.roi):
            self.roi.setPos((x0 - 0.5, y0 - 0.5 + offset[1]))
            self.roi.setSize((x1 - x0, y1 - y0))
        self._region = (self.roi.pos() - QtCore.QPointF(*offset), self.roi.size())
        if self.region_action.isChecked():
            self.session.set_bounds(self.bounds())
        else:
            self.region_action.setChecked(True)
