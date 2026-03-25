"""
OpenGL Tiled + LOD Huge Image Viewer (PyQtGraph)

Features implemented:
- Orthographic 2D pan/zoom (pixel coordinates: x=col, y=row, y-down)
- Tiled rendering with LOD (power-of-two downsample factor based on zoom)
- Supports grayscale (H,W) with matplotlib colormap LUT + optional colorbar
- Supports RGB/RGBA (H,W,3/4) tiles (no colormap/colorbar by default)
- Vertical divider lines every N pixels (e.g., 32 or 64)
- Hover status: x, y, stripe (default stripe_width=64), intensity (gray) or RGB tuple (RGB)
- Throttled tile updates and throttled hover reads (memmap-friendly)
- Right-click context menu: Copy View / Copy Window to clipboard
- Optional "Copy View" button in bottom status bar

Dependencies:
  pip install pyqtgraph PyQt6 PyOpenGL matplotlib
(or PyQt5 instead of PyQt6)
"""

import math
import numpy as np
import pyqtgraph.opengl as gl

from matplotlib import cm

try:
    from PyQt6 import QtCore, QtWidgets, QtGui
    from PyQt6.QtCore import Qt
except Exception:
    from PyQt5 import QtCore, QtWidgets, QtGui
    from PyQt5.QtCore import Qt


# -----------------------------
# GL shader cache reset
# -----------------------------

def _reset_gl_shader_caches():
    """Reset class-level shader program caches on pyqtgraph GL items.

    PyQtGraph caches compiled shaders at the class level. When a GL context
    is destroyed (viewer window closed) and a new one is created, the stale
    cache causes items to silently fail to render.  Calling this before
    creating a new viewer forces recompilation in the new context.
    """
    for cls_path in (
        "pyqtgraph.opengl.items.GLScatterPlotItem.GLScatterPlotItem",
        "pyqtgraph.opengl.items.GLLinePlotItem.GLLinePlotItem",
        "pyqtgraph.opengl.items.GLImageItem.GLImageItem",
        "pyqtgraph.opengl.items.GLVolumeItem.GLVolumeItem",
    ):
        try:
            parts = cls_path.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls = getattr(mod, parts[1])
            if hasattr(cls, "_shaderProgram"):
                cls._shaderProgram = None
        except Exception:
            pass
    try:
        from pyqtgraph.opengl import shaders as pg_shaders
        if hasattr(pg_shaders, "initShaders"):
            pg_shaders.initShaders()
    except Exception:
        pass


# -----------------------------
# Color parsing
# -----------------------------

_NAMED_COLORS = {
    "red":     (1.0, 0.0, 0.0, 1.0),
    "green":   (0.0, 1.0, 0.0, 1.0),
    "blue":    (0.0, 0.0, 1.0, 1.0),
    "cyan":    (0.0, 1.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0, 1.0),
    "yellow":  (1.0, 1.0, 0.0, 1.0),
    "white":   (1.0, 1.0, 1.0, 1.0),
    "orange":  (1.0, 0.5, 0.0, 1.0),
}


def _parse_color(color):
    """Convert a color spec to (R, G, B, A) float tuple.

    Accepts: named string ("red"), (R,G,B) tuple, or (R,G,B,A) tuple.
    """
    if isinstance(color, str):
        c = _NAMED_COLORS.get(color.lower())
        if c is None:
            raise ValueError(f"Unknown color name: {color!r}. Use one of {list(_NAMED_COLORS)}")
        return c
    c = tuple(float(x) for x in color)
    if len(c) == 3:
        return (*c, 1.0)
    if len(c) == 4:
        return c
    raise ValueError(f"Color must be a name, (R,G,B), or (R,G,B,A) — got length {len(c)}")


# -----------------------------
# Utility functions
# -----------------------------

def robust_levels_sample_gray(img, p=(1, 99), step_y=512, step_x=64):
    """Percentiles on a strided sample for huge grayscale images."""
    samp = img[::step_y, ::step_x]
    samp = np.asarray(samp)
    samp = samp[np.isfinite(samp)]
    if samp.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(samp.astype(np.float32, copy=False), p)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(img))
        hi = float(np.nanmax(img)) if float(np.nanmax(img)) > lo else lo + 1.0
    return float(lo), float(hi)


def robust_levels_sample_rgb(img, p=(1, 99), step_y=512, step_x=64):
    """Per-channel percentiles on a strided sample for huge RGB(A) images."""
    samp = img[::step_y, ::step_x, :3].astype(np.float32, copy=False)
    lo = np.zeros(3, dtype=np.float32)
    hi = np.ones(3, dtype=np.float32)
    for c in range(3):
        v = samp[..., c]
        v = v[np.isfinite(v)]
        if v.size == 0:
            lo[c], hi[c] = 0.0, 1.0
            continue
        lo_c, hi_c = np.percentile(v, p)
        if not np.isfinite(lo_c) or not np.isfinite(hi_c) or hi_c <= lo_c:
            lo_c = float(np.nanmin(v))
            hi_c = float(np.nanmax(v)) if float(np.nanmax(v)) > lo_c else lo_c + 1.0
        lo[c], hi[c] = float(lo_c), float(hi_c)
    return lo, hi


def make_lut_rgba_u8(cmap_name="gray"):
    """256x4 uint8 LUT using matplotlib colormap."""
    cmap = cm.get_cmap(cmap_name)
    lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)  # RGBA
    return lut


def rgb_to_uint8(tile_rgb, lo_rgb=None, hi_rgb=None):
    """
    Convert HxWx3 or HxWx4 tile to uint8.
    - uint8 returned as-is.
    - float in [0,1] scaled to 0..255.
    - otherwise robust per-channel scaling if lo_rgb/hi_rgb provided.
    """
    if tile_rgb.dtype == np.uint8:
        return tile_rgb

    t = tile_rgb.astype(np.float32, copy=False)

    tmin = float(np.nanmin(t)) if np.isfinite(np.nanmin(t)) else 0.0
    tmax = float(np.nanmax(t)) if np.isfinite(np.nanmax(t)) else 1.0

    # 0..1 float fast path
    if (tmin >= 0.0) and (tmax <= 1.0):
        return np.clip(t * 255.0, 0, 255).astype(np.uint8)

    if lo_rgb is None or hi_rgb is None:
        lo_rgb = np.array([tmin, tmin, tmin], dtype=np.float32)
        hi_rgb = np.array([tmax if tmax > tmin else tmin + 1.0] * 3, dtype=np.float32)

    out = np.empty_like(t, dtype=np.uint8)
    ch = min(3, t.shape[2])
    for c in range(ch):
        lo = float(lo_rgb[c])
        hi = float(hi_rgb[c])
        if not (hi > lo):
            hi = lo + 1.0
        u = (t[..., c] - lo) / (hi - lo)
        u = np.clip(u, 0.0, 1.0)
        out[..., c] = (u * 255.0).astype(np.uint8)

    # Preserve alpha if present
    if t.shape[2] == 4:
        a = t[..., 3]
        # If alpha is 0..65535, scale down
        if np.nanmax(a) > 255.0:
            a = np.clip(a / 257.0, 0, 255)
        out[..., 3] = np.clip(a, 0, 255).astype(np.uint8)

    return out


def tile_to_rgba(tile, mode, lut=None, lo=None, hi=None, lo_rgb=None, hi_rgb=None):
    """
    Convert a tile to HxWx4 uint8 RGBA for upload:
      - mode="gray": tile is HxW -> LUT mapping
      - mode="rgb":  tile is HxWx3/4 -> direct RGBA (uint8)
    """
    if mode == "gray":
        t = tile.astype(np.float32, copy=False)
        den = (hi - lo) if (hi is not None and lo is not None and hi > lo) else 1.0
        u = (t - lo) / den
        u = np.clip(u, 0.0, 1.0)
        u8 = (u * 255.0).astype(np.uint8)
        rgba = lut[u8]  # HxWx4
        if np.isnan(t).any():
            a = rgba[..., 3]
            a[np.isnan(t)] = 0
            rgba[..., 3] = a
        return rgba

    # RGB/RGBA
    t8 = rgb_to_uint8(tile, lo_rgb=lo_rgb, hi_rgb=hi_rgb)
    if t8.shape[2] == 3:
        alpha = np.full((t8.shape[0], t8.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate([t8, alpha], axis=2)
    else:
        rgba = t8  # already RGBA

    # Optional NaN transparency
    t3 = tile[..., :3].astype(np.float32, copy=False)
    if not np.isfinite(t3).all():
        mask = np.any(~np.isfinite(t3), axis=2)
        rgba[mask, 3] = 0

    return rgba


def glimage_set_data(item, rgba_wh4):
    """Handle GLImageItem API differences across pyqtgraph versions."""
    try:
        item.setData(rgba_wh4)
    except TypeError:
        item.setData(image=rgba_wh4)


# -----------------------------
# Colorbar widget (for grayscale)
# -----------------------------

class ColorBarWidget(QtWidgets.QWidget):
    """Paint-only vertical colorbar based on a 256x4 RGBA LUT and [vmin, vmax]."""
    def __init__(self, lut_rgba_u8, vmin, vmax, n_ticks=5, parent=None):
        super().__init__(parent)
        self.lut = lut_rgba_u8  # (256,4) uint8
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n_ticks = int(n_ticks)

        self.setMinimumWidth(90)
        self.setMaximumWidth(130)

        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.setSizePolicy(sp)

    def set_range(self, vmin, vmax):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.update()

    def paintEvent(self, ev):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)

        r = self.rect()
        margin = 8
        bar_w = 18
        text_pad = 6

        bar_rect = QtCore.QRect(
            margin,
            margin,
            bar_w,
            max(1, r.height() - 2 * margin),
        )

        # Gradient top=vmax, bottom=vmin
        grad = QtGui.QLinearGradient(bar_rect.left(), bar_rect.top(), bar_rect.left(), bar_rect.bottom())
        for i in range(256):
            rr, gg, bb, aa = map(int, self.lut[255 - i])
            grad.setColorAt(i / 255.0, QtGui.QColor(rr, gg, bb, aa))
        painter.fillRect(bar_rect, grad)

        painter.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180, 255), 1))
        painter.drawRect(bar_rect)

        vmin, vmax = self.vmin, self.vmax
        if not (vmax > vmin):
            vmax = vmin + 1.0

        painter.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220, 255), 1))
        n = max(2, self.n_ticks)
        for k in range(n):
            t = k / (n - 1)  # 0..1 bottom->top
            y = int(round(bar_rect.bottom() - t * bar_rect.height()))
            painter.drawLine(bar_rect.right() + 1, y, bar_rect.right() + 6, y)
            val = vmin + t * (vmax - vmin)
            painter.drawText(bar_rect.right() + 6 + text_pad, y + 4, f"{val:.4g}")

        painter.drawText(margin, margin - 2, "Value")


# -----------------------------
# Orthographic GL 2D view
# -----------------------------

class GLOrtho2D(gl.GLViewWidget):
    """
    Orthographic 2D view with explicit pan/zoom in pixel coordinates.
    x = column index, y = row index, y increases downward (image convention).
    """
    viewChanged = QtCore.pyqtSignal()
    mouseWorldMoved = QtCore.pyqtSignal(float, float)
    rightClicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setMouseTracking(True)

        self._center = np.array([0.0, 0.0], dtype=np.float64)
        self._half_height = 100.0
        self._last_pos = None
        self._panning = False

        # Optional bounds clamp (set by set_image_bounds)
        self._bounds_enabled = False
        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0

    def set_image_bounds(self, H, W):
        """Enable bounds clamp so pan/zoom cannot drift away from the image."""
        self._bounds_enabled = True
        self._x_min, self._x_max = 0.0, float(W)
        self._y_min, self._y_max = 0.0, float(H)

    def _clamp_view(self):
        if not self._bounds_enabled:
            return

        # Clamp zoom-out (half height cannot exceed image half-height)
        H = self._y_max - self._y_min
        max_half_h = max(1.0, 0.5 * H)
        self._half_height = float(min(self._half_height, max_half_h))
        self._half_height = float(max(self._half_height, 1e-6))

        # Clamp center so viewport stays within image bounds
        w = max(1, float(self.width()))
        h = max(1, float(self.height()))
        aspect = w / h
        hh = float(self._half_height)
        hw = hh * aspect

        cx, cy = float(self._center[0]), float(self._center[1])

        cx = min(max(cx, self._x_min + hw), self._x_max - hw)
        cy = min(max(cy, self._y_min + hh), self._y_max - hh)

        self._center[0], self._center[1] = cx, cy

    def viewMatrix(self):
        return QtGui.QMatrix4x4()  # identity

    def projectionMatrix(self, region=None):
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        aspect = w / h

        cx, cy = float(self._center[0]), float(self._center[1])
        hh = float(self._half_height)
        hw = hh * aspect

        left, right = cx - hw, cx + hw
        top, bottom = cy - hh, cy + hh

        m = QtGui.QMatrix4x4()
        # bottom, top swapped => y-down
        m.ortho(left, right, bottom, top, -1e6, 1e6)
        return m

    def screen_to_world_xy(self, px, py):
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        aspect = w / h

        cx, cy = float(self._center[0]), float(self._center[1])
        hh = float(self._half_height)
        hw = hh * aspect

        x = (px / w) * (2 * hw) + (cx - hw)
        y = (py / h) * (2 * hh) + (cy - hh)
        return float(x), float(y)

    def world_units_per_pixel(self):
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        aspect = w / h
        hh = float(self._half_height)
        hw = hh * aspect
        return (2 * hw) / w, (2 * hh) / h

    def view_bounds(self):
        w = max(1.0, float(self.width()))
        h = max(1.0, float(self.height()))
        aspect = w / h

        cx, cy = float(self._center[0]), float(self._center[1])
        hh = float(self._half_height)
        hw = hh * aspect
        return (cx - hw, cx + hw, cy - hh, cy + hh)

    def set_view_full_image(self, H, W, pad=0.02):
        self._center[:] = (W / 2.0, H / 2.0)
        self._half_height = (H / 2.0) * (1.0 + pad)
        self._clamp_view()
        self.update()
        self.viewChanged.emit()

    def mousePressEvent(self, ev):
        btn = ev.button() if hasattr(ev, "button") else None
        pos = ev.position() if hasattr(ev, "position") else ev.pos()

        right_btn = (btn == Qt.MouseButton.RightButton) if hasattr(Qt, "MouseButton") else (btn == Qt.RightButton)
        if right_btn:
            self.rightClicked.emit(int(pos.x()), int(pos.y()))
            ev.accept()
            return

        left_btn = (btn == Qt.MouseButton.LeftButton) if hasattr(Qt, "MouseButton") else (btn == Qt.LeftButton)
        self._last_pos = pos
        self._panning = bool(left_btn)
        ev.accept()

    def mouseReleaseEvent(self, ev):
        self._panning = False
        self._last_pos = None
        ev.accept()

    def mouseMoveEvent(self, ev):
        pos = ev.position() if hasattr(ev, "position") else ev.pos()

        if self._last_pos is None:
            self._last_pos = pos

        if self._panning:
            dx_pix = float(pos.x() - self._last_pos.x())
            dy_pix = float(pos.y() - self._last_pos.y())
            ux, uy = self.world_units_per_pixel()

            self._center[0] -= dx_pix * ux
            self._center[1] -= dy_pix * uy

            self._last_pos = pos
            self._clamp_view()
            self.update()
            self.viewChanged.emit()

        xw, yw = self.screen_to_world_xy(float(pos.x()), float(pos.y()))
        self.mouseWorldMoved.emit(xw, yw)
        ev.accept()

    def wheelEvent(self, ev):
        delta = ev.angleDelta().y() if hasattr(ev, "angleDelta") else ev.delta()
        if delta == 0:
            ev.accept()
            return

        pos = ev.position() if hasattr(ev, "position") else ev.pos()
        mx, my = float(pos.x()), float(pos.y())
        before = np.array(self.screen_to_world_xy(mx, my), dtype=np.float64)

        factor = 0.85 if delta > 0 else 1.15
        self._half_height = float(max(1e-6, self._half_height * factor))

        after = np.array(self.screen_to_world_xy(mx, my), dtype=np.float64)
        self._center += (before - after)

        self._clamp_view()
        self.update()
        self.viewChanged.emit()
        ev.accept()


# -----------------------------
# Main viewer widget
# -----------------------------

class GLTiledImshow(QtWidgets.QWidget):
    """
    OpenGL tiled + LOD image viewer for huge arrays (ndarray or memmap).
    Supports:
      - Gray: HxW, LUT colormap + optional colorbar
      - RGB/RGBA: HxWx3/4, direct display
    Overlay:
      - Vertical divider lines every divider_step pixels
    UX:
      - Hover status with stripe index (stripe_width pixels) + intensity/RGB
      - Right-click menu: copy view/window
      - Optional "Copy View" button
    """
    def __init__(
        self,
        img,
        cmap="gray",
        tile_w=4096,
        tile_h=8192,
        max_ds=256,
        divider_step=32,
        divider_alpha=0.20,
        stripe_width=64,
        show_colorbar=True,
        show_copy_button=True,
        tile_update_interval_ms=35,
        hover_update_interval_ms=60,
        parent=None,
    ):
        super().__init__(parent)

        if img.ndim == 2:
            self._mode = "gray"
            self.H, self.W = int(img.shape[0]), int(img.shape[1])
        elif img.ndim == 3 and img.shape[2] in (3, 4):
            self._mode = "rgb"
            self.H, self.W = int(img.shape[0]), int(img.shape[1])
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}. Use HxW, HxWx3, or HxWx4.")

        self.img = img
        self.cmap = str(cmap)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.max_ds = int(max_ds)

        self.stripe_width = int(stripe_width)
        self._divider_step = None
        self._divider_item = None

        # LUT always created; used only in gray mode
        self.lut = make_lut_rgba_u8(self.cmap)

        if self._mode == "gray":
            self.lo, self.hi = robust_levels_sample_gray(self.img, p=(1, 99), step_y=512, step_x=64)
            self.rgb_lo = None
            self.rgb_hi = None
        else:
            self.lo = None
            self.hi = None
            self.rgb_lo, self.rgb_hi = robust_levels_sample_rgb(self.img, p=(1, 99), step_y=512, step_x=64)

        # ---------- UI layout ----------
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top row: GL view + colorbar
        top_row = QtWidgets.QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        self.view = GLOrtho2D()
        self.view.set_image_bounds(self.H, self.W)
        top_row.addWidget(self.view, stretch=1)

        self.colorbar = None
        if show_colorbar and (self._mode == "gray"):
            self.colorbar = ColorBarWidget(self.lut, self.lo, self.hi, n_ticks=5)
            top_row.addWidget(self.colorbar, stretch=0)

        layout.addLayout(top_row, stretch=1)

        # Bottom bar: status + optional copy button
        bottom = QtWidgets.QHBoxLayout()
        bottom.setContentsMargins(6, 0, 6, 0)
        bottom.setSpacing(8)

        self.info = QtWidgets.QLabel()
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info.setWordWrap(False)
        self.info.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        sp = self.info.sizePolicy()
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.Expanding)
        self.info.setSizePolicy(sp)
        self.info.setMinimumHeight(18)
        self.info.setMaximumHeight(22)

        bottom.addWidget(self.info, stretch=1)

        self.btn_copy = None
        if show_copy_button:
            self.btn_copy = QtWidgets.QPushButton("Copy View")
            spb = self.btn_copy.sizePolicy()
            spb.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
            self.btn_copy.setSizePolicy(spb)
            self.btn_copy.setToolTip("Copy the current OpenGL viewport to clipboard")
            self.btn_copy.clicked.connect(self.copy_view_to_clipboard)
            bottom.addWidget(self.btn_copy, stretch=0)

        layout.addLayout(bottom, stretch=0)

        # ---------- Overlays ----------
        self._overlays = {}        # name -> GLGraphicsItem
        self._overlay_counter = 0

        # ---------- Tiles ----------
        self._tiles = {}           # key=(ds, ty, tx) -> GLImageItem
        self._current_ds = None

        # Divider overlay
        self.set_divider_step(divider_step, alpha=divider_alpha)

        # ---------- Timers ----------
        # Tile update throttle
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(tile_update_interval_ms))
        self._timer.timeout.connect(self._update_tiles)

        self.view.viewChanged.connect(lambda: self._timer.start())

        # Hover throttle (for memmap friendliness)
        self._hover_timer = QtCore.QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(int(hover_update_interval_ms))
        self._hover_timer.timeout.connect(self._update_hover_label)

        self._pending_hover_xy = None
        self._last_hover_pixel = None

        self.view.mouseWorldMoved.connect(self._on_mouse_world_moved)
        self.view.rightClicked.connect(self._show_context_menu)

        # Initialize view and tiles
        self.view.set_view_full_image(self.H, self.W)
        self._update_tiles()
        self._update_info_static()

    # -----------------------------
    # Public configuration methods
    # -----------------------------

    def set_divider_step(self, step, alpha=0.20):
        """step=None disables dividers. step=int draws vertical lines at x=0,step,2*step,..."""
        if self._divider_item is not None:
            try:
                self.view.removeItem(self._divider_item)
            except Exception:
                pass
            self._divider_item = None

        if step is None:
            self._divider_step = None
            return

        step = int(step)
        if step <= 0:
            self._divider_step = None
            return

        self._divider_step = step

        xs = np.arange(0, self.W + 1, step, dtype=np.float32)
        pos = np.zeros((xs.size * 2, 3), dtype=np.float32)
        pos[0::2, 0] = xs
        pos[0::2, 1] = 0.0
        pos[1::2, 0] = xs
        pos[1::2, 1] = float(self.H)

        color = (1.0, 1.0, 1.0, float(alpha))
        item = gl.GLLinePlotItem(pos=pos, color=color, width=1.0, mode="lines")
        item.setDepthValue(20)
        self.view.addItem(item)
        self._divider_item = item

        self._update_info_static()

    def set_stripe_width(self, stripe_width):
        self.stripe_width = int(stripe_width)
        self._update_info_static()

    def set_gray_levels(self, lo, hi):
        """Update grayscale scaling (and colorbar) and refresh tiles."""
        if self._mode != "gray":
            return
        self.lo, self.hi = float(lo), float(hi)
        if self.colorbar is not None:
            self.colorbar.set_range(self.lo, self.hi)
        self._current_ds = None  # force tile rebuild
        self._update_tiles()

    # -----------------------------
    # Overlay API
    # -----------------------------

    def _next_overlay_name(self, prefix):
        self._overlay_counter += 1
        return f"{prefix}-{self._overlay_counter}"

    def add_points(self, xy, color="red", size=5, name=None):
        """Add scatter markers at (N, 2) pixel coordinates [x, y].

        Parameters
        ----------
        xy : array_like, shape (N, 2)
            Point positions in [x, y] (column, row) image pixel coordinates.
        color : str or tuple
            Named color string or (R, G, B) / (R, G, B, A) float tuple.
        size : float
            Marker diameter in pixels.
        name : str, optional
            Overlay name for later removal.  Auto-generated if not given.

        Returns
        -------
        str
            The overlay name (use with ``remove_overlay``).
        """
        pts = np.asarray(xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"xy must be (N, 2), got shape {pts.shape}")

        rgba = _parse_color(color)
        pos3 = np.zeros((pts.shape[0], 3), dtype=np.float32)
        pos3[:, 0] = pts[:, 0]
        pos3[:, 1] = pts[:, 1]

        item = gl.GLScatterPlotItem(pos=pos3, color=rgba, size=size, pxMode=True)
        item.setDepthValue(10)
        self.view.addItem(item)

        name = name or self._next_overlay_name("points")
        if name in self._overlays:
            self.remove_overlay(name)
        self._overlays[name] = item
        return name

    def add_lines(self, segments, color="cyan", width=1.5, name=None):
        """Add line segments.

        Parameters
        ----------
        segments : array_like, shape (M, 2, 2)
            M line segments, each [[x0, y0], [x1, y1]].
        color : str or tuple
            Named color string or (R, G, B) / (R, G, B, A) float tuple.
        width : float
            Line width in pixels.
        name : str, optional
            Overlay name for later removal.

        Returns
        -------
        str
            The overlay name.
        """
        segs = np.asarray(segments, dtype=np.float32)
        if segs.ndim != 3 or segs.shape[1:] != (2, 2):
            raise ValueError(f"segments must be (M, 2, 2), got shape {segs.shape}")

        rgba = _parse_color(color)
        # Flatten to (2*M, 3) for mode="lines"
        pos3 = np.zeros((segs.shape[0] * 2, 3), dtype=np.float32)
        pos3[0::2, 0] = segs[:, 0, 0]  # x0
        pos3[0::2, 1] = segs[:, 0, 1]  # y0
        pos3[1::2, 0] = segs[:, 1, 0]  # x1
        pos3[1::2, 1] = segs[:, 1, 1]  # y1

        item = gl.GLLinePlotItem(pos=pos3, color=rgba, width=width, mode="lines")
        item.setDepthValue(10)
        self.view.addItem(item)

        name = name or self._next_overlay_name("lines")
        if name in self._overlays:
            self.remove_overlay(name)
        self._overlays[name] = item
        return name

    def add_rects(self, rects, color="yellow", width=1.0, name=None):
        """Add rectangular outlines.

        Parameters
        ----------
        rects : array_like, shape (K, 4)
            Each row is [x, y, w, h] — top-left corner and size.
        color : str or tuple
            Named color string or (R, G, B) / (R, G, B, A) float tuple.
        width : float
            Line width in pixels.
        name : str, optional
            Overlay name for later removal.

        Returns
        -------
        str
            The overlay name.
        """
        r = np.asarray(rects, dtype=np.float32)
        if r.ndim != 2 or r.shape[1] != 4:
            raise ValueError(f"rects must be (K, 4), got shape {r.shape}")

        # Each rect becomes 4 line segments (12 coordinate pairs, but we
        # need 4 segment pairs = 8 points per rect)
        K = r.shape[0]
        x, y, w, h = r[:, 0], r[:, 1], r[:, 2], r[:, 3]

        # 4 segments per rect: top, right, bottom, left
        segs = np.zeros((K * 4, 2, 2), dtype=np.float32)
        # top:    (x, y) -> (x+w, y)
        segs[0::4, 0, 0] = x;       segs[0::4, 0, 1] = y
        segs[0::4, 1, 0] = x + w;   segs[0::4, 1, 1] = y
        # right:  (x+w, y) -> (x+w, y+h)
        segs[1::4, 0, 0] = x + w;   segs[1::4, 0, 1] = y
        segs[1::4, 1, 0] = x + w;   segs[1::4, 1, 1] = y + h
        # bottom: (x+w, y+h) -> (x, y+h)
        segs[2::4, 0, 0] = x + w;   segs[2::4, 0, 1] = y + h
        segs[2::4, 1, 0] = x;       segs[2::4, 1, 1] = y + h
        # left:   (x, y+h) -> (x, y)
        segs[3::4, 0, 0] = x;       segs[3::4, 0, 1] = y + h
        segs[3::4, 1, 0] = x;       segs[3::4, 1, 1] = y

        return self.add_lines(segs, color=color, width=width,
                              name=name or self._next_overlay_name("rects"))

    def remove_overlay(self, name):
        """Remove a named overlay from the view.

        Parameters
        ----------
        name : str
            The name returned by ``add_points``, ``add_lines``, or ``add_rects``.
        """
        item = self._overlays.pop(name, None)
        if item is not None:
            try:
                self.view.removeItem(item)
            except Exception:
                pass

    def clear_overlays(self):
        """Remove all overlays (points, lines, rects) from the view."""
        for item in self._overlays.values():
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._overlays.clear()

    # -----------------------------
    # Copy to clipboard
    # -----------------------------

    def copy_view_to_clipboard(self):
        cb = QtWidgets.QApplication.clipboard()
        if hasattr(self.view, "grabFramebuffer"):
            qimg = self.view.grabFramebuffer()
            pm = QtGui.QPixmap.fromImage(qimg)
        else:
            pm = self.view.grab()
        cb.setPixmap(pm)

    def copy_window_to_clipboard(self):
        cb = QtWidgets.QApplication.clipboard()
        pm = self.window().grab()
        cb.setPixmap(pm)

    def _show_context_menu(self, x, y):
        menu = QtWidgets.QMenu(self)
        act_copy_view = menu.addAction("Copy View to Clipboard")
        act_copy_win = menu.addAction("Copy Window to Clipboard")

        chosen = menu.exec(self.view.mapToGlobal(QtCore.QPoint(int(x), int(y))))
        if chosen == act_copy_view:
            self.copy_view_to_clipboard()
        elif chosen == act_copy_win:
            self.copy_window_to_clipboard()

    # -----------------------------
    # Hover handling
    # -----------------------------

    def _on_mouse_world_moved(self, xw, yw):
        self._pending_hover_xy = (float(xw), float(yw))
        if not self._hover_timer.isActive():
            self._hover_timer.start()

    def _update_hover_label(self):
        if self._pending_hover_xy is None:
            return

        xw, yw = self._pending_hover_xy
        xi = int(np.floor(xw))
        yi = int(np.floor(yw))

        if self._last_hover_pixel == (xi, yi):
            return
        self._last_hover_pixel = (xi, yi)

        if not (0 <= xi < self.W and 0 <= yi < self.H):
            self.info.setText(f"(out of bounds) x={xi} y={yi} | div={self._divider_step} stripe_w={self.stripe_width}")
            return

        stripe = (xi // self.stripe_width) + 1  # 1-based
        x_in = xi % self.stripe_width

        try:
            val = self.img[yi, xi]
        except Exception as e:
            self.info.setText(f"x={xi} y={yi} stripe={stripe} | read error: {e}")
            return

        if self._mode == "gray":
            # scalar
            try:
                v = float(val) if isinstance(val, np.generic) else float(val)
            except Exception:
                v = val
            if isinstance(v, (float, int)) and np.isfinite(v) and (self.hi is not None) and (self.lo is not None) and (self.hi > self.lo):
                norm = (float(v) - float(self.lo)) / (float(self.hi) - float(self.lo))
                norm = float(np.clip(norm, 0.0, 1.0))
                self.info.setText(
                    f"x={xi} y={yi}  I={float(v):.6g}  norm={norm:.3f}  "
                    f"stripe={stripe} (x_in={x_in})  | ds={self._current_ds} div={self._divider_step}"
                )
            else:
                self.info.setText(
                    f"x={xi} y={yi}  I={v}  stripe={stripe} (x_in={x_in})  | ds={self._current_ds} div={self._divider_step}"
                )
        else:
            # RGB(A)
            try:
                v = np.asarray(val).tolist()
            except Exception:
                v = val
            if isinstance(v, list) and len(v) >= 3:
                self.info.setText(
                    f"x={xi} y={yi}  RGB={tuple(v[:3])}  stripe={stripe} (x_in={x_in})  | ds={self._current_ds} div={self._divider_step}"
                )
            else:
                self.info.setText(
                    f"x={xi} y={yi}  val={v}  stripe={stripe} (x_in={x_in})  | ds={self._current_ds} div={self._divider_step}"
                )

    def _update_info_static(self):
        if self._mode == "gray":
            self.info.setText(
                f"img={self.H}x{self.W} | gray cmap={self.cmap} | div={self._divider_step} | stripe_w={self.stripe_width}"
            )
        else:
            self.info.setText(
                f"img={self.H}x{self.W} | RGB(A) | div={self._divider_step} | stripe_w={self.stripe_width}"
            )

    # -----------------------------
    # Tiles / LOD
    # -----------------------------

    def _choose_ds(self):
        ux, uy = self.view.world_units_per_pixel()
        scale = max(float(ux), float(uy))
        scale = max(1.0, scale)
        ds = 2 ** int(math.floor(math.log2(scale)))
        ds = int(max(1, min(ds, self.max_ds)))
        return ds

    def _clear_tiles(self):
        for item in self._tiles.values():
            try:
                self.view.removeItem(item)
            except Exception:
                pass
        self._tiles.clear()

    def _update_tiles(self):
        ds = self._choose_ds()
        xmin, xmax, ymin, ymax = self.view.view_bounds()

        # Clamp to image bounds
        xmin = int(max(0, math.floor(xmin)))
        xmax = int(min(self.W, math.ceil(xmax)))
        ymin = int(max(0, math.floor(ymin)))
        ymax = int(min(self.H, math.ceil(ymax)))

        # If ds changed, flush tiles (simple, stable)
        if self._current_ds != ds:
            self._clear_tiles()
            self._current_ds = ds

        # Degenerate view (should be prevented by clamping in GLOrtho2D)
        if xmax <= xmin or ymax <= ymin:
            return

        tx0 = xmin // self.tile_w
        tx1 = (max(xmin, xmax - 1)) // self.tile_w
        ty0 = ymin // self.tile_h
        ty1 = (max(ymin, ymax - 1)) // self.tile_h

        needed = set((ds, ty, tx) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1))

        # Remove tiles not needed
        for k in list(self._tiles.keys()):
            if k not in needed:
                try:
                    self.view.removeItem(self._tiles[k])
                except Exception:
                    pass
                del self._tiles[k]

        # Create missing tiles
        for (ds_k, ty, tx) in needed:
            if (ds_k, ty, tx) in self._tiles:
                continue

            x0 = tx * self.tile_w
            x1 = min(self.W, (tx + 1) * self.tile_w)
            y0 = ty * self.tile_h
            y1 = min(self.H, (ty + 1) * self.tile_h)

            tile = self.img[y0:y1:ds_k, x0:x1:ds_k]

            rgba = tile_to_rgba(
                tile,
                mode=self._mode,
                lut=self.lut,
                lo=self.lo, hi=self.hi,
                lo_rgb=self.rgb_lo, hi_rgb=self.rgb_hi,
            )

            # GLImageItem expects (W,H,4) orientation; swap axes
            rgba_gl = np.swapaxes(rgba, 0, 1)

            dummy = np.zeros((2, 2, 4), dtype=np.uint8)
            try:
                item = gl.GLImageItem(dummy)
            except TypeError:
                item = gl.GLImageItem(data=dummy)

            try:
                item.setGLOptions("translucent")
            except Exception:
                pass

            glimage_set_data(item, rgba_gl)

            # Map downsampled tile coords back to original pixel coords
            item.resetTransform()
            item.translate(float(x0), float(y0), 0.0)
            item.scale(float(ds_k), float(ds_k), 1.0)

            item.setDepthValue(0)
            self.view.addItem(item)
            self._tiles[(ds_k, ty, tx)] = item


# -----------------------------
# Notebook-facing convenience wrappers
# -----------------------------

def imshow(img, title=None, window_size=(1200, 800), **kwargs):
    """Show a huge image in an interactive GL viewer.

    Creates a ``QMainWindow`` containing a :class:`GLTiledImshow` widget,
    calls ``.show()``, and returns the viewer widget.  Intended for use in
    Jupyter notebooks with ``%gui qt`` enabled.

    Parameters
    ----------
    img : ndarray or memmap
        Image array — ``(H, W)`` grayscale or ``(H, W, 3/4)`` RGB/RGBA.
    title : str, optional
        Window title.  Defaults to image shape description.
    window_size : tuple of (int, int)
        Initial window width and height in pixels.
    **kwargs
        Forwarded to :class:`GLTiledImshow` (``cmap``, ``divider_step``,
        ``stripe_width``, ``tile_w``, ``tile_h``, etc.).

    Returns
    -------
    GLTiledImshow
        The viewer widget.  Access the window via ``viewer.window()``.

    Examples
    --------
    ::

        %gui qt
        import subpx
        v = subpx.imshow(img, cmap="gray", divider_step=64)
        v.add_points(centers_xy, color="red", size=4)
        v.add_rects(bboxes, color="cyan")
    """
    _reset_gl_shader_caches()

    viewer = GLTiledImshow(img, **kwargs)

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(viewer)

    if title is None:
        shape_str = "x".join(str(s) for s in img.shape)
        title = f"subpx — {shape_str}"
    win.setWindowTitle(title)

    w, h = int(window_size[0]), int(window_size[1])
    win.resize(w, h)
    win.show()

    # Keep a reference so the window isn't garbage collected
    viewer._main_window = win

    return viewer


def imshow_huge(img, **kwargs):
    """Create and return a GLTiledImshow widget for a huge image.

    .. deprecated:: Use :func:`imshow` instead for managed window creation.
    """
    return GLTiledImshow(img, **kwargs)


def show_image(img, **kwargs):
    """Alias for :func:`imshow_huge`."""
    return imshow_huge(img, **kwargs)


__all__ = [
    "robust_levels_sample_gray",
    "robust_levels_sample_rgb",
    "make_lut_rgba_u8",
    "rgb_to_uint8",
    "tile_to_rgba",
    "glimage_set_data",
    "ColorBarWidget",
    "GLOrtho2D",
    "GLTiledImshow",
    "imshow",
    "imshow_huge",
    "show_image",
]
