# Mosaic and stripe-board utilities

This module contains correctness-first helpers for bringing stripe/board/page TIFF data into one canonical global coordinate system.

Key ideas:
- canonical global coordinates: x right, y down
- backward stripes can be normalized with `reverse_pages`, `flipud`, and `crop_top`
- page-wise cell-only classification can be converted into global rectangles before full stitching
