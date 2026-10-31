"""Pages (a tab next to Setup): every page of the open file, counted from 0, with a thumbnail made when it
scrolls into view and a mark for what was measured there; linked both ways with "Stack vertically"."""

from __future__ import annotations

import time

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..widgets import hint

THUMB_WIDTH = 72


def thumbnail(image, levels=None):
    """A small grey QPixmap of a page (or its preview)."""
    step = max(1, int(np.ceil(image.shape[1] / THUMB_WIDTH)))
    small = np.asarray(image[::step, ::step], float)
    low, high = levels if levels is not None else np.percentile(small, (1, 99.5))
    if high <= low:
        high = low + 1
    data = np.ascontiguousarray(np.clip((small - low) * (255 / (high - low)), 0, 255).astype(np.uint8))
    picture = QtGui.QImage(data.data, data.shape[1], data.shape[0], data.strides[0],
                           QtGui.QImage.Format.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(picture.copy())


class PagesPanel(QtWidgets.QWidget):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.stack_pages = QtWidgets.QCheckBox("Stack vertically")
        self.stack_pages.setToolTip("Every page in one view; analysis uses the active, outlined page. "
                                    "Shift+F fits the stack, F the active page.")
        self.stack_pages.toggled.connect(session.set_stacked)
        layout.addWidget(self.stack_pages)
        self.note = hint("Open a multipage file to list its pages.")
        layout.addWidget(self.note)
        self.list = QtWidgets.QListWidget()
        self.list.setIconSize(QtCore.QSize(THUMB_WIDTH, THUMB_WIDTH))
        self.list.setUniformItemSizes(True)  # fast with thousands of pages: every row three lines, one icon size
        self.list.setToolTip("Click a page to make it active (Page Up / Page Down step through the pages)")
        self.list.currentRowChanged.connect(self._row_changed)
        self.list.verticalScrollBar().valueChanged.connect(lambda *args: self._timer.start())
        layout.addWidget(self.list, 1)
        for keys, step in (("PgUp", -1), ("PgDown", 1)):
            QtGui.QShortcut(QtGui.QKeySequence(keys), self.list, activated=lambda step=step: self._step(step),
                            context=QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._thumbs = set()
        self._timer = QtCore.QTimer(self, singleShot=True, interval=30, timeout=self._make_thumbnails)
        session.image_changed.connect(self._image_changed)
        session.results_changed.connect(self._update_marks)
        session.stacked_changed.connect(self._stacked_changed)
        session.busy_changed.connect(lambda busy: self.setEnabled(not busy))

    def _step(self, step):
        if self.session.loaded is not None:
            self.session.set_page(self.session.loaded.page + step)

    def _image_changed(self, new_file):
        session = self.session
        source, loaded = session.source, session.loaded
        if new_file:
            self._thumbs = set()
            with QtCore.QSignalBlocker(self.list):
                self.list.clear()
                if source is not None:
                    # A blank thumbnail until the page's own is made, so rows line up from the start.
                    h, w = source.shapes[0]
                    size = QtCore.QSize(THUMB_WIDTH, max(8, round(THUMB_WIDTH * h / max(w, 1))))
                    self.list.setIconSize(size)
                    blank = QtGui.QPixmap(size)
                    blank.fill(QtGui.QColor("#151e2b"))
                    for page in range(source.pages):
                        item = QtWidgets.QListWidgetItem(QtGui.QIcon(blank), self._text(page))
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, page)
                        self.list.addItem(item)
            pages = source.pages if source is not None else 1
            self.note.setText(f"{pages} pages, counted from 0 as in the batch's pages.csv." if pages > 1 else
                              "This image has one page.")
            self.stack_pages.setEnabled(pages > 1)
            with QtCore.QSignalBlocker(self.stack_pages):
                self.stack_pages.setChecked(session.stacked)
        with QtCore.QSignalBlocker(self.list):
            if loaded is not None and 0 <= loaded.page < self.list.count():
                self.list.setCurrentRow(loaded.page)
                self.list.scrollToItem(self.list.item(loaded.page))
        self._update_marks()
        self._timer.start()

    def _stacked_changed(self, stacked):
        with QtCore.QSignalBlocker(self.stack_pages):
            self.stack_pages.setChecked(stacked)

    def _row_changed(self, row):
        if row >= 0:
            self.session.set_page(row)
            loaded = self.session.loaded
            if loaded is not None and loaded.page != row and not self.session.busy:
                # Not changed (e.g. a job is running): the list stays on the active page.
                with QtCore.QSignalBlocker(self.list):
                    self.list.setCurrentRow(loaded.page)

    def _text(self, page):
        """Three lines: page, size, what was measured there (blank when nothing)."""
        h, w = self.session.source.shapes[page]
        short, _ = self.session.marks.get(page, (" ", ""))
        return f"Page {page}\n{w} x {h} px\n{short}"

    def _update_marks(self):
        if self.session.source is None:
            return
        for page in range(self.list.count()):
            item = self.list.item(page)
            text = self._text(page)
            if item.text() != text:
                mark = self.session.marks.get(page)
                item.setText(text)
                item.setToolTip(mark[1] if mark else "")
                item.setForeground(QtGui.QColor("#56d6cf") if mark else QtGui.QColor("#dee7f2"))

    def _make_thumbnails(self):
        """Thumbnails of the rows in view, a few per turn of the event loop."""
        source = self.session.source
        if source is None or not self.list.count() or not self.isVisible():
            return
        top = self.list.indexAt(QtCore.QPoint(4, 4)).row()
        bottom = self.list.indexAt(QtCore.QPoint(4, self.list.viewport().height() - 4)).row()
        top = max(top, 0)
        bottom = self.list.count() - 1 if bottom < 0 else bottom
        started = time.perf_counter()
        for page in range(top, bottom + 1):
            if page in self._thumbs:
                continue
            preview = source.preview(page)
            try:
                image = preview[0] if preview is not None else source.read(page).image
            except Exception as exc:
                self._thumbs.add(page)
                self.session.log.emit(f"Page {page} could not be read for its thumbnail: {exc}")
                continue
            self.list.item(page).setIcon(QtGui.QIcon(thumbnail(image)))
            self._thumbs.add(page)
            if time.perf_counter() - started > 0.03:
                self._timer.start()
                break

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()
