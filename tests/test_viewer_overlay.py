"""Smoke tests for viewer overlay API.

These test the pure-Python logic (color parsing, array construction)
without requiring a display server or pyqtgraph.  GL-dependent tests
are skipped if pyqtgraph is not installed.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _parse_color is importable even without pyqtgraph because it only uses
# builtins — but the module-level 'import pyqtgraph.opengl' will fail.
# So we import selectively.
# ---------------------------------------------------------------------------

_viewer_available = False
try:
    from subpx.viewer import (
        _parse_color,
        _reset_gl_shader_caches,
        _NAMED_COLORS,
        GLTiledImshow,
    )
    _viewer_available = True
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    not _viewer_available,
    reason="pyqtgraph / PyQt not installed",
)


# ---------------------------------------------------------------------------
# _parse_color
# ---------------------------------------------------------------------------

class TestParseColor:
    def test_named_red(self):
        assert _parse_color("red") == (1.0, 0.0, 0.0, 1.0)

    def test_named_case_insensitive(self):
        assert _parse_color("RED") == _parse_color("red")

    def test_rgb_tuple(self):
        assert _parse_color((0.5, 0.5, 0.5)) == (0.5, 0.5, 0.5, 1.0)

    def test_rgba_tuple(self):
        assert _parse_color((0.1, 0.2, 0.3, 0.4)) == (0.1, 0.2, 0.3, 0.4)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown color"):
            _parse_color("chartreuse")

    def test_bad_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _parse_color((1.0, 2.0))

    def test_all_named_colors_are_rgba(self):
        for name, rgba in _NAMED_COLORS.items():
            assert len(rgba) == 4, f"{name} should be 4-tuple"


# ---------------------------------------------------------------------------
# _reset_gl_shader_caches (should not raise even without GL context)
# ---------------------------------------------------------------------------

def test_reset_shader_caches_no_crash():
    _reset_gl_shader_caches()


# ---------------------------------------------------------------------------
# Overlay array construction (requires QApplication — skip if headless)
# ---------------------------------------------------------------------------

_has_qapp = False
try:
    from subpx.viewer import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _has_qapp = True
except Exception:
    pass


@pytest.mark.skipif(not _has_qapp, reason="No QApplication / display")
class TestOverlayMethods:
    @pytest.fixture()
    def viewer(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        v = GLTiledImshow(img, show_colorbar=False, show_copy_button=False)
        yield v
        v.clear_overlays()

    def test_add_points_returns_name(self, viewer):
        pts = np.array([[10.0, 20.0], [30.0, 40.0]])
        name = viewer.add_points(pts)
        assert isinstance(name, str)
        assert name in viewer._overlays

    def test_add_points_bad_shape_raises(self, viewer):
        with pytest.raises(ValueError, match="N, 2"):
            viewer.add_points(np.array([1.0, 2.0, 3.0]))

    def test_add_lines_returns_name(self, viewer):
        segs = np.array([[[0, 0], [10, 10]], [[5, 5], [15, 15]]], dtype=np.float32)
        name = viewer.add_lines(segs)
        assert name in viewer._overlays

    def test_add_lines_bad_shape_raises(self, viewer):
        with pytest.raises(ValueError, match="M, 2, 2"):
            viewer.add_lines(np.zeros((3, 3)))

    def test_add_rects_returns_name(self, viewer):
        rects = np.array([[5, 5, 10, 10], [20, 20, 5, 5]], dtype=np.float32)
        name = viewer.add_rects(rects)
        assert name in viewer._overlays

    def test_add_rects_bad_shape_raises(self, viewer):
        with pytest.raises(ValueError, match="K, 4"):
            viewer.add_rects(np.zeros((2, 3)))

    def test_remove_overlay(self, viewer):
        name = viewer.add_points(np.array([[1.0, 2.0]]))
        viewer.remove_overlay(name)
        assert name not in viewer._overlays

    def test_remove_nonexistent_is_noop(self, viewer):
        viewer.remove_overlay("does-not-exist")  # should not raise

    def test_clear_overlays(self, viewer):
        viewer.add_points(np.array([[1.0, 2.0]]))
        viewer.add_points(np.array([[3.0, 4.0]]))
        assert len(viewer._overlays) == 2
        viewer.clear_overlays()
        assert len(viewer._overlays) == 0

    def test_custom_name(self, viewer):
        name = viewer.add_points(np.array([[1.0, 2.0]]), name="my-centers")
        assert name == "my-centers"

    def test_duplicate_name_replaces(self, viewer):
        viewer.add_points(np.array([[1.0, 2.0]]), name="pts")
        viewer.add_points(np.array([[3.0, 4.0]]), name="pts")
        assert len(viewer._overlays) == 1
        assert "pts" in viewer._overlays
