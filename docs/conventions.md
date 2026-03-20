
# Conventions

## Coordinates
- Points are `(N, 2)` arrays in `[x, y]`
- `x` is image column
- `y` is image row
- image shape is `(H, W)`

## Output policy
- Public API returns NumPy arrays or dataclasses containing NumPy arrays
- Internal GPU implementations may use CuPy freely
- Future GPU code should convert outputs to NumPy before returning from public APIs

## Naming
Use short task-based public names:
- `detect_centers`
- `refine_centers`
- `dedupe_centers`
- `estimate_pitch`
- `estimate_shift`

Avoid algorithm-history names such as:
- `find_centers_hybrid_gpu_cpu_logquad`
- `find_subpixel_centers_tiled_logquad_gpu`
