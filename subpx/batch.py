"""Notebook-friendly batch processing helpers."""

from __future__ import annotations

from collections.abc import Iterable, Callable
from typing import Any


def process_frames(frames: Iterable, operation: Callable[..., Any], /, *args, progress: bool = False, **kwargs):
    out = []
    for i, frame in enumerate(frames):
        out.append(operation(frame, *args, **kwargs))
        if progress and (i + 1) % 50 == 0:
            print(f"processed {i + 1} frames")
    return out


def process_stack(stack, operation: Callable[..., Any], /, *args, progress: bool = False, **kwargs):
    return process_frames(stack, operation, *args, progress=progress, **kwargs)


def detect_centers_batch(frames: Iterable, /, *args, progress: bool = False, **kwargs):
    from .centers import detect_centers
    return process_frames(frames, detect_centers, *args, progress=progress, **kwargs)


def detect_centers_tiled_batch(frames: Iterable, /, *args, tile_h: int = 8192, progress: bool = False, **kwargs):
    from .centers import detect_centers
    def _tiled(frame, *a, **kw):
        return detect_centers(frame, *a, tile_h=tile_h, **kw)
    return process_frames(frames, _tiled, *args, progress=progress, **kwargs)


def estimate_shift_batch(ref_frames: Iterable, mov_frames: Iterable, /, *args, progress: bool = False, **kwargs):
    from .registration import estimate_shift
    out = []
    for i, (ref, mov) in enumerate(zip(ref_frames, mov_frames)):
        out.append(estimate_shift(ref, mov, *args, **kwargs))
        if progress and (i + 1) % 50 == 0:
            print(f"processed {i + 1} pairs")
    return out
