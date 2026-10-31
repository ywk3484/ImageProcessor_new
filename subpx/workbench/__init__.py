"""Optional PyQt6/pyqtgraph inspection workbench for the repository's measurements on sample images.

Run ``python -m subpx.workbench [image]`` or ``subpx-workbench``, or call :func:`launch_workbench` from a
notebook. The window is built from one :class:`~subpx.workbench.session.Session` (image, pages, settings,
results, selection, jobs) and panels that listen to it.
"""

from .names import METHOD_NAMES, plain_error
from .session import Session
from .window import InspectionWindow, launch_workbench, main

__all__ = ["InspectionWindow", "METHOD_NAMES", "Session", "launch_workbench", "main", "plain_error"]
