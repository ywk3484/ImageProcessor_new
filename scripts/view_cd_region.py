"""Write the interactive contour viewer for one region of saved CD results.

Images with ~1e5 holes are too large for a single offline HTML viewer; this
writes one for a crop, from a results folder of analyze_contact_holes.py
(full or ``--outputs compact``). Cell IDs and coordinates shown are those of
the full image, matching measurements.csv.

    python scripts/view_cd_region.py artifacts/contact_holes_cd --region 2000 800 1000 1000
    python scripts/view_cd_region.py results/img_01 --region 0 0 512 512 --output region.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subpx import load_cd_results, save_cd_viewer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, help="Results folder (or its raw_data.npz)")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True,
                        help="Crop in full-image pixels: left, top, width, height")
    parser.add_argument("--metric", choices=("cd_equivalent_px", "cd_x_px", "cd_y_px", "area_px2"), default="cd_equivalent_px")
    parser.add_argument("--output", type=Path, help="Default: <results>/region_X_Y_W_H.html")
    args = parser.parse_args(argv)
    folder = args.results if args.results.is_dir() else args.results.parent
    output = args.output or folder / ("region_" + "_".join(map(str, args.region)) + ".html")
    start = time.perf_counter()
    result = load_cd_results(args.results)
    try:
        path = save_cd_viewer(result, output, metric=args.metric, region=args.region)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
