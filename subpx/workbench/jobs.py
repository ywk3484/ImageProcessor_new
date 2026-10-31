"""One background worker for loading, measuring, aligning and exporting, with elapsed time and Cancel."""

from __future__ import annotations

import traceback

from PyQt6 import QtCore


class _Signals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)


class _Job(QtCore.QRunnable):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = _Signals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.signals.finished.emit(self.function())
        except Exception:
            self.signals.failed.emit(traceback.format_exc())


class JobRunner(QtCore.QObject):
    """Runs one job at a time on a worker thread; the window never waits for it.

    Cancel stops waiting and discards the job's result: the library call itself cannot be interrupted, so it
    finishes in the background (a new job starts after it). ``progress`` reports the job and its elapsed time.
    """

    started = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)
    stopped = QtCore.pyqtSignal()  # finished, failed or cancelled: no job is running any more
    failed = QtCore.pyqtSignal(str, str)  # context ("Measuring CD failed"), traceback
    log = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._current = None
        self._label = ""
        self._clock = QtCore.QElapsedTimer()
        self._timer = QtCore.QTimer(self, interval=250, timeout=self._tick)

    @property
    def busy(self):
        return self._current is not None

    @property
    def label(self):
        return self._label if self.busy else ""

    def elapsed(self):
        return self._clock.elapsed() / 1000 if self.busy else 0.0

    def run(self, label, function, on_result):
        """Start ``function`` on the worker; ``on_result(result)`` runs in the window when it returns."""
        if self.busy:
            return False
        job = self._current = _Job(function)
        self._label = label
        context = label.rstrip(".") + " failed"
        self._clock.start()
        self._timer.start()

        def done(result):
            if job is not self._current:
                return  # cancelled: the result is discarded
            self._finish()
            try:
                on_result(result)
            except Exception:
                self.failed.emit(context, traceback.format_exc())

        def failed(detail):
            if job is not self._current:
                return
            self._finish()
            self.failed.emit(context, detail)

        job.signals.finished.connect(done)
        job.signals.failed.connect(failed)
        self.started.emit(label)
        self._tick()
        self._pool.start(job)
        return True

    def cancel(self):
        if not self.busy:
            return
        label, seconds = self._label, self.elapsed()
        self._finish()
        self.log.emit(f"{label} cancelled after {seconds:.1f} s; the calculation finishes in the background and "
                      "its result is discarded.")

    def running(self):
        """Worker threads still busy, including a cancelled job that has not returned yet."""
        return self._pool.activeThreadCount()

    def wait(self, msecs=-1):
        return self._pool.waitForDone(msecs)

    def _finish(self):
        self._current = None
        self._timer.stop()
        self.stopped.emit()

    def _tick(self):
        if self.busy:
            self.progress.emit(f"{self._label} | {self.elapsed():.1f} s")
