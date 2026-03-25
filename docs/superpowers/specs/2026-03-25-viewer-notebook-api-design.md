# Viewer Notebook API Design

## Problem

`GLTiledImshow` renders huge images but has no overlay support and no convenient single-call entry point. Users must manually manage widget creation, window showing, and have no way to visualize detected centers, pitch grids, or ROI boxes on top of the GL viewer.

## Design

### Entry point: `imshow()`

```python
v = subpx.imshow(img, cmap="gray", divider_step=64, title="my image", window_size=(1200, 800))
```

- Creates `QMainWindow` wrapping `GLTiledImshow`, calls `.show()`
- Returns the `GLTiledImshow` instance (window accessible via `v.window()`)
- Resets pyqtgraph GL shader caches before creating the widget (fixes the stale-context bug)
- `title=` sets window title; `window_size=` sets initial geometry
- `imshow_huge()` and `show_image()` become aliases

### Overlay methods on `GLTiledImshow`

All coordinates in `[x, y]` pixel space. All methods return a string overlay name for later removal.

#### `add_points(xy, color="red", size=5, name=None)`

- `xy`: (N, 2) array in [x, y] order
- Uses `GLScatterPlotItem` positioned at z=10 (above image tiles at z=0)
- `color`: named string or (R, G, B, A) float tuple
- `size`: point diameter in pixels

#### `add_lines(segments, color="cyan", width=1.5, name=None)`

- `segments`: (M, 2, 2) array — M line segments, each [[x0,y0],[x1,y1]]
- Uses `GLLinePlotItem` with `mode="lines"`

#### `add_rects(rects, color="yellow", width=1.0, name=None)`

- `rects`: (K, 4) array — each row [x, y, w, h]
- Converts to line segments internally, uses `GLLinePlotItem`

#### `remove_overlay(name)` / `clear_overlays()`

- Tracked in `self._overlays: dict[str, GLGraphicsItem]`
- `remove_overlay` removes by name; `clear_overlays` removes all

### Color parsing

Single utility `_parse_color(color)` → (R, G, B, A) float tuple.
Supports: named strings via a small lookup table ("red", "green", "blue", "cyan", "magenta", "yellow", "white"), or direct (3,) / (4,) tuples.

### Shader cache reset

`_reset_gl_shader_caches()` module-level function, called at the top of `imshow()`. Resets `_shaderProgram = None` on `GLScatterPlotItem`, `GLLinePlotItem`, `GLImageItem`, `GLVolumeItem`.

## Files changed

- `subpx/viewer.py` — all additions (overlay methods, `imshow()`, shader reset)
- `subpx/__init__.py` — export `imshow`

## Not in scope

- Event loop management (user runs `%gui qt`)
- CenterResult/PitchResult unwrapping
- Matplotlib integration
