"""Log (bottom): activity, timings and full error texts; it never opens by itself."""

from __future__ import annotations

import time

from PyQt6 import QtWidgets


class LogPanel(QtWidgets.QPlainTextEdit):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        session.log.connect(self.add)

    def add(self, text):
        self.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")
