"""Small shared widgets and the window style."""

from __future__ import annotations

from pathlib import Path
import tempfile

from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

BACKGROUND = "#0c121b"


def no_si_prefix(plot_item):
    """Pixel axes read 'x (px)', never 'x (kpx)' or 'y (mpx)'."""
    for name in ("bottom", "left"):
        plot_item.getAxis(name).enableAutoSIPrefix(False)


def plot_widget(**labels):
    plot = pg.PlotWidget(background=BACKGROUND)
    plot.showGrid(x=True, y=True, alpha=0.15)
    no_si_prefix(plot.getPlotItem())
    for side, text in labels.items():
        plot.setLabel(side, text)
    return plot


def hint(text="", *, wrap=True):
    label = QtWidgets.QLabel(text)
    label.setWordWrap(wrap)
    label.setObjectName("hint")
    label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def dot_icon(color, size=10):
    """A filled circle (method colours, step marks)."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QtGui.QIcon(pixmap)


class Fold(QtWidgets.QWidget):
    """A titled section that folds open and closed (instead of a dialog)."""

    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title, content=None, *, opened=False, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.button = QtWidgets.QToolButton()
        self.button.setObjectName("fold")
        self.button.setText(title)
        self.button.setCheckable(True)
        self.button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.button.toggled.connect(self._toggled)
        layout.addWidget(self.button)
        self.content = content if content is not None else QtWidgets.QWidget()
        layout.addWidget(self.content)
        self.button.setChecked(opened)
        self._toggled(opened)

    def _toggled(self, opened):
        self.button.setArrowType(QtCore.Qt.ArrowType.DownArrow if opened else QtCore.Qt.ArrowType.RightArrow)
        self.content.setVisible(opened)
        self.toggled.emit(opened)

    def is_open(self):
        return self.button.isChecked()

    def set_open(self, opened):
        self.button.setChecked(opened)


def _arrow_images():
    """Spin-box arrow images drawn once per process (a styled spin box shows no arrows of its own)."""
    folder = Path(tempfile.gettempdir()) / "subpx-workbench"
    folder.mkdir(exist_ok=True)
    paths = []
    for name, points in (("up", ((1, 6), (5, 2), (9, 6))), ("down", ((1, 2), (5, 6), (9, 2)))):
        path = folder / f"arrow_{name}.png"
        image = QtGui.QImage(10, 8, QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#a9bbce"))
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(*point) for point in points]))
        painter.end()
        image.save(str(path))
        paths.append(path.as_posix())
    return paths


STYLE = """
    QMainWindow, QWidget { background: #151e2b; color: #dee7f2; font-family: 'Segoe UI'; font-size: 12px; }
    QMenuBar { background: #1b2839; } QMenuBar::item:selected, QMenu::item:selected { background: #30465e; }
    QMenu { background: #1c2b3d; border: 1px solid #304155; } QMenu::item:disabled { color: #5d6f82; }
    QToolBar { background: #1b2839; padding: 3px 6px; spacing: 6px; border: 0; }
    QToolButton { padding: 4px 8px; border-radius: 4px; }
    QToolButton:hover, QToolButton:checked { background: #30465e; }
    QToolButton#view { padding: 4px 14px; border: 1px solid #3c556e; }
    QToolButton#view:checked { background: #246c72; border-color: #56d6cf; color: #ffffff; font-weight: 600; }
    QToolButton#fold { border: 0; padding: 4px 2px; font-weight: 600; color: #90b2d2; text-align: left; }
    QDockWidget { color: #90b2d2; font-weight: 600; }
    QDockWidget::title { background: #1b2839; padding: 4px 8px; }
    QLabel#hint { color: #a9bbce; }
    QLabel#reason { color: #ffce73; }
    QLabel#big { font-size: 20px; font-weight: 600; color: #f4f8ff; }
    QLabel#warning { color: #ffce73; }
    QLabel#good { color: #9be28a; }
    QLabel#status[level="warning"] { color: #ffce73; } QLabel#status[level="error"] { color: #ff9aa6; }
    QFrame#banner { background: #42232b; border: 1px solid #8f3a46; border-radius: 4px; }
    QFrame#stale { background: #3b3322; border: 1px solid #7a6330; border-radius: 4px; }
    QLabel#bannertext { color: #ffe3c2; }
    QPushButton { background: #24384e; border: 1px solid #3c556e; border-radius: 4px; padding: 5px 9px; }
    QPushButton:hover { background: #31516b; border-color: #56d6cf; }
    QPushButton#primary { background: #246c72; border-color: #56d6cf; color: #ffffff; font-weight: 600; padding: 7px 18px; }
    QPushButton:disabled, QPushButton#primary:disabled { background: #1a2532; color: #5d6f82; border-color: #293a4c; }
    QCheckBox:disabled, QLabel:disabled, QToolButton:disabled, QRadioButton:disabled { color: #5d6f82; }
    QGroupBox { border: 1px solid #304155; border-radius: 5px; margin-top: 12px; font-weight: 600; }
    QGroupBox::title { subcontrol-origin: margin; left: 9px; color: #90b2d2; }
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #0f1824; border: 1px solid #354a60; border-radius: 3px; padding: 3px; min-height: 19px; }
    QSpinBox, QDoubleSpinBox { padding-right: 18px; }
    QSpinBox[invalid="true"], QDoubleSpinBox[invalid="true"] { border: 1px solid #ff748a; }
    QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled { color: #5d6f82; border-color: #293a4c; }
    QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
        subcontrol-origin: border; width: 16px; background: #1c2b3d; border-left: 1px solid #354a60; }
    QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 3px; }
    QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 3px; }
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #31516b; }
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url(ARROW_UP); width: 10px; height: 8px; }
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url(ARROW_DOWN); width: 10px; height: 8px; }
    QComboBox QAbstractItemView { background: #1c2b3d; selection-background-color: #365571; }
    QTabWidget::pane { border: 1px solid #304155; }
    QTabBar::tab { background: #1b293a; padding: 6px 10px; border-bottom: 2px solid transparent; }
    QTabBar::tab:selected { color: #56d6cf; border-bottom: 2px solid #56d6cf; background: #243447; }
    QToolBox::tab { background: #1b293a; border: 1px solid #304155; border-radius: 3px; padding: 4px 6px; font-weight: 600; }
    QToolBox::tab:selected { color: #56d6cf; background: #243447; }
    QTableView, QTreeView, QListView, QPlainTextEdit, QTextBrowser { background: #101925; alternate-background-color: #1b2a3b; border: 1px solid #304155; selection-background-color: #304f69; }
    QHeaderView::section { background: #26384c; color: #c3d3e5; border: 0; padding: 4px 6px; }
    QScrollArea { border: 0; }
    QSplitter::handle { background: #28394c; }
    QMainWindow::separator { background: #28394c; width: 3px; height: 3px; }
    QStatusBar { color: #a9bbce; background: #101925; }
    QCheckBox { spacing: 5px; }
"""


def apply_style(widget):
    up, down = _arrow_images()
    widget.setStyleSheet(STYLE.replace("ARROW_UP", up).replace("ARROW_DOWN", down))
