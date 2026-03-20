from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

from .centers import detect_centers, filter_centers, dedupe_centers

try:
    import tifffile as tiff  # type: ignore
except Exception:  # pragma: no cover
    tiff = None


@dataclass(frozen=True)
class TiffKey:
    stripe: int
    vertical: int
    board: int


_fname_re = re.compile(r"(?P<s>\d+)_(?P<v>\d+)_(?P<b>\d+)\.tiff?$", re.IGNORECASE)


def parse_tiff_key(path: str | Path) -> TiffKey:
    p = Path(path)
    m = _fname_re.match(p.name)
    if not m:
        raise ValueError(f"Bad filename: {p.name}")
    return TiffKey(int(m['s']), int(m['v']), int(m['b']))


def build_tiff_index(tiff_paths: list[str | Path]) -> dict[tuple[int, int], list[Path]]:
    buckets: dict[tuple[int, int], list[tuple[int, Path]]] = {}
    for pp in tiff_paths:
        p = Path(pp)
        k = parse_tiff_key(p)
        buckets.setdefault((k.stripe, k.board), []).append((k.vertical, p))
    out: dict[tuple[int, int], list[Path]] = {}
    for sb, items in buckets.items():
        out[sb] = [p for _v, p in sorted(items, key=lambda x: x[0])]
    return out


def _require_tiff():
    if tiff is None:
        raise RuntimeError('tifffile is required for mosaic page iteration.')


def iter_pages_across_vertical(files_v: list[str | Path]):
    _require_tiff()
    k = 0
    for fp in files_v:
        with tiff.TiffFile(str(fp)) as tf:
            for page in tf.pages:
                yield k, page.asarray()
                k += 1


def iter_pages_canonical(files_v: list[str | Path], *, reverse_pages: bool = False):
    _require_tiff()
    if not reverse_pages:
        yield from iter_pages_across_vertical(files_v)
        return
    pages = []
    for fp in files_v:
        tf = tiff.TiffFile(str(fp))
        for page in tf.pages:
            pages.append((tf, page))
    try:
        for k_canon, (_tf, page) in enumerate(reversed(pages)):
            yield k_canon, page.asarray()
    finally:
        seen = set()
        for tf, _page in pages:
            if id(tf) not in seen:
                tf.close()
                seen.add(id(tf))


def map_centers_to_global(centers_xy, stripe_meta: dict, board_meta: dict, *, k_canon: int, H: int, crop_top: int = 0, flipud: bool = False) -> np.ndarray:
    c = np.asarray(centers_xy, dtype=np.float64)
    if c.size == 0:
        return c.reshape(0, 2)
    Hprime = int(stripe_meta['Hprime'])
    x0s = float(stripe_meta.get('x0', 0.0))
    y0s = float(stripe_meta.get('y0', 0.0))
    x0b = float(board_meta.get('x0', 0.0))
    x = c[:, 0].copy()
    y = c[:, 1].copy()
    if flipud:
        y = (int(H) - 1) - y
    y_eff = (int(k_canon) * int(H)) + y - float(crop_top)
    keep = (y_eff >= 0.0) & (y_eff < float(Hprime))
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float64)
    xg = x0s + x0b + x[keep]
    yg = y0s + y_eff[keep]
    return np.column_stack([xg, yg]).astype(np.float64)


def true_runs(mask) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return []
    runs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if start is not None and ((not v) or (i == m.size - 1 and v)):
            end = i + 1 if v and i == m.size - 1 else i
            runs.append((start, end))
            start = None
    return runs


def cell_runs_to_global_rects(cell_only_mask, stripe_meta: dict, board_meta: dict, *, H: int, crop_top: int = 0, board_width: float | None = None):
    Hprime = float(stripe_meta['Hprime'])
    x0 = float(stripe_meta.get('x0', 0.0)) + float(board_meta.get('x0', 0.0))
    if board_width is None:
        board_width = float(board_meta.get('width', 0.0))
    rects = []
    for k0, k1 in true_runs(cell_only_mask):
        y0 = float(stripe_meta.get('y0', 0.0)) + (k0 * int(H) - float(crop_top))
        y1 = float(stripe_meta.get('y0', 0.0)) + (k1 * int(H) - float(crop_top))
        y0 = max(float(stripe_meta.get('y0', 0.0)), y0)
        y1 = min(float(stripe_meta.get('y0', 0.0)) + Hprime, y1)
        if y1 > y0:
            rects.append((x0, y0, x0 + float(board_width), y1))
    return rects


def process_stripe_board(
    files_v: list[str | Path],
    stripe_meta: dict,
    board_meta: dict,
    cell_only_mask,
    *,
    H: int,
    reverse_pages: bool = False,
    flipud: bool = False,
    crop_top: int = 0,
    detect_kwargs: dict | None = None,
    margin: float = 5.0,
    dedupe_eps: float = 1.5,
):
    detect_kwargs = {} if detect_kwargs is None else dict(detect_kwargs)
    out = []
    for k_canon, img in iter_pages_canonical(files_v, reverse_pages=reverse_pages):
        if not bool(np.asarray(cell_only_mask, dtype=bool)[k_canon]):
            continue
        centers = detect_centers(img, **detect_kwargs).centers_xy
        centers = filter_centers(centers, img.shape, margin=margin)
        centers = dedupe_centers(centers, eps=dedupe_eps)
        g = map_centers_to_global(centers, stripe_meta, board_meta, k_canon=k_canon, H=H, crop_top=crop_top, flipud=flipud)
        if g.size:
            out.append(g)
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(out)
