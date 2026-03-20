# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**maskproc** (v0.2.0) — notebook-friendly photomask image-processing utilities. Python 3.10+, numpy-only core dependency. Used in Jupyter notebooks for photomask center detection, pitch estimation, and image registration.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_smoke.py::test_detect_centers_cpu_smoke -v
```

No linter or formatter is configured. No CI pipeline exists.

## Architecture

### Backend Dispatch Pattern

The central design pattern: public API modules dispatch to `_cpu/` or `_gpu/` implementations via `backends.py`.

```
maskproc/centers.py  (public API)
    → maskproc/_cpu/centers.py   (OpenCV-based)
    → maskproc/_gpu/centers.py   (CuPy-based, hybrid CPU segmentation + GPU refinement)
```

Same pattern for `components.py` and `registration.py`. Backend selected via `backend="auto"|"cpu"|"gpu"` parameter on public functions. `"auto"` checks CuPy availability.

GPU implementations are **hybrid**: segmentation stays on CPU (OpenCV), expensive per-ROI refinement runs on GPU (CuPy). All public API outputs are NumPy arrays — GPU code must convert before returning.

### Module Roles

| Module | Role |
|---|---|
| `centers.py` | Center detection: Otsu → connected components → subpixel refinement |
| `pitch.py` | Pitch estimation: KNN or index_regression, global and per-line |
| `registration.py` | Phase cross-correlation shift estimation |
| `spectra.py` | FFT-based periodic error analysis |
| `calibration.py` | Row/column clustering, residual maps, distortion field fitting |
| `mosaic.py` | TIFF stripe/board parsing and global coordinate mapping |
| `batch.py` | Thin wrappers for processing image stacks |
| `viewer.py` | OpenGL tiled image viewer (PyQt5/6 + pyqtgraph) |
| `legacy.py` | Deprecation wrappers mapping old function names → new API |
| `types.py` | Result dataclasses: `CenterResult`, `PitchResult`, `ShiftResult`, etc. |

### Conventions

- **Coordinates**: Points are `(N, 2)` arrays in `[x, y]` order (x=column, y=row). Image shape is `(H, W)`.
- **Naming**: Public functions use task-based names (`detect_centers`, not `find_centers_hybrid_gpu_cpu_logquad`). Algorithm choice via keyword args (`refine="logquad"`).
- **Return types**: Dataclasses with NumPy arrays + metadata dict. See `types.py`.
- **Optional deps**: opencv-python, scikit-image, cupy, PyQt5/6, scipy, tifffile — all wrapped in try/except for graceful degradation.

### Key Refinement Methods

`detect_centers()` supports `refine=` parameter:
- `"weighted"` — weighted centroid (fast, good for small blobs)
- `"logquad"` — log-quadratic subpixel fitting (better accuracy)
- `"edge_gradmoment"` — gradient-moment edge localization (best for large features)

`recommend_refine_method(bbox_size)` provides heuristic selection.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Use plan mode for verification steps, not just building

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Review lessons at session start

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Ask: "Would a staff engineer approve this?"

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- Skip this for simple, obvious fixes — don't over-engineer

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
- **Be Critical**: Do not try to be a 'helpful assistant' too much. Saying good things on everything when it actually isn't is a sin and not helpful.
