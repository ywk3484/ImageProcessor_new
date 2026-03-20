
# Overview

This cleanup converts a chat-grown collection of utility functions into a real package.
The package design intentionally separates:

1. stable notebook-facing public API
2. backend implementation detail
3. legacy compatibility wrappers
4. documentation of conventions and theory

The public naming rule is:
- function name = user task
- keyword arguments = algorithm choice / backend choice

Examples:
- `detect_centers(...)`
- `estimate_pitch(...)`
- `estimate_shift(...)`
- `process_stack(...)`


## Internal layout added in v0.2
- `subpx._cpu.*`: backend-specific CPU implementations
- `subpx._gpu.*`: backend-specific GPU implementations
- public modules dispatch to one of the above while preserving one stable API


## Added after GPU migration
- `components.py`: connected-component stats with CPU/GPU backends
- `spectra.py`: FFT-based periodic error analysis
- `calibration.py`: row/column clustering and residual-cloud helpers
- `detect_centers_tiled(...)`: cleaned tiled detection API
