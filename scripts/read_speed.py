"""How fast TIFF files read from their disk: a plain read of the file (the disk's own speed), the reader of
scripts/cd_batch.py (pages in file order) and tifffile.imread (the whole file into RAM).

    python scripts/read_speed.py /data/dataset1/stripe_001/scan_001.tif

Before each read the file is dropped from the operating system's cache (Linux: posix_fadvise; Windows: opening it
unbuffered), so every read comes from the disk, as in a real run. An HDD reads about 50-250 MB/s; several GB/s mean
the file came from RAM. The two plain reads should agree: if the second is much faster, the cache could not be
dropped and the numbers are not the disk's.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cd_batch import iter_frames


def drop_from_cache(path) -> bool:
    """Make the next read of ``path`` come from the disk; False when this system offers no way to."""
    if hasattr(os, "posix_fadvise"):
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    if sys.platform == "win32":  # opening a file unbuffered makes Windows drop its cached pages
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileW.restype = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        generic_read, share_read_write, open_existing, no_buffering = 0x80000000, 0x3, 3, 0x20000000
        handle = kernel32.CreateFileW(str(path), generic_read, share_read_write, None, open_existing, no_buffering,
                                      None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), f"cannot open {path}")
        kernel32.CloseHandle(handle)
        return True
    return False


def plain_read(path) -> None:
    with open(path, "rb", buffering=0) as handle:
        while handle.read(8 << 20):
            pass


def batch_reader(path) -> int:
    """The pages cd_batch.py reads from the file, in its order; the number of pages."""
    pages = 0
    for _ in iter_frames(path):
        pages += 1
    return pages


def imread(path) -> None:
    import tifffile

    tifffile.imread(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="TIFF files (each is read four times)")
    args = parser.parse_args(argv)
    import tifffile

    for path in args.files:
        size = path.stat().st_size
        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            print(f"{path}: {size / 1e9:.2f} GB, pages of {page.imagewidth} x {page.imagelength} px, {page.dtype}, "
                  f"{'uncompressed' if page.compression == 1 else f'compression {page.compression}'}", flush=True)
        for label, read in (("plain read of the file (the disk)", plain_read),
                            ("cd_batch.py's reader (pages in order)", batch_reader),
                            ("tifffile.imread (whole file)", imread),
                            ("plain read again", plain_read)):
            if not drop_from_cache(path):
                print("  (this system cannot drop the file from its cache: a second read may come from RAM)")
            start = time.perf_counter()
            pages = read(path)
            seconds = time.perf_counter() - start
            print(f"  {label:39s} {seconds:7.1f} s {size / 1e6 / seconds:7.0f} MB/s"
                  + (f"   ({pages} pages)" if pages is not None else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
