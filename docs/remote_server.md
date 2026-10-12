# Run the contact-hole CD pipeline

```powershell
# TIFF: select page 10 (the 11th image).
python scripts/analyze_contact_holes.py "D:\data\images.tiff" --page 10 --output "D:\results\page_10"

# PNG:
python scripts/analyze_contact_holes.py "D:\data\image.png" --output "D:\results\image"
```

Run from the repository root on the server, using your Python environment.
Replace the input and output paths. This runs all four CD methods on the CPU.

For many large images (~1e5 holes each), use the GPU and compact outputs:

```bash
python scripts/analyze_contact_holes.py /data/run_01/*.png --backend gpu --outputs compact --output /results/run_01
python scripts/view_cd_region.py /results/run_01/img_007 --region 2000 800 1000 1000   # viewer for one crop
python scripts/benchmark_cd.py --backends gpu                                        # this machine's speed
```

Each image gets its own subfolder; `batch_summary.csv` lists all images.
See [Large images and batches](contact_holes_cd.md#large-images-and-batches---backend-gpu---outputs-compact).

## Input parameters

| Parameter | Default | Meaning |
|---|---|---|
| Image arguments | Supply your image path(s) | One grayscale image, or several files, wildcards or folders (batch: one output subfolder per image). |
| `--backend` | `cpu` | `gpu` runs everything on the GPU (needs CuPy and `--background model`), 40–70x faster on large images; `auto` uses it when available. |
| `--outputs` | `full` | `compact` skips contours.csv/contours.png/contours.html and writes a lossless uncompressed NPZ (~0.4 GB instead of ~7 GB per 1e5-hole image). |
| `--page` | `0` | TIFF page index: `0` is the first image; `1499` is the last in a 1,500-page file. |
| `--output` | `artifacts/contact_holes_cd` | Output directory. Existing named files are overwritten. |
| `--methods` | `gaussian logquad gradient halfmax` | Methods to run: deblended Gaussian FWHM, log-fit Gaussian FWHM, radial maximum gradient, and half-height contours. |
| `--background` | `model` | `model`: fitted background map with neighbouring holes subtracted. `border`: previous per-cell 10th percentile (reproduces old results). |
| `--seeds` | `threshold` | `peaks` uses local maxima instead of thresholding; use it when pitch/FWHM is below about 1.6. |
| `--background-level` | none | Known constant background, for images with no hole-free area. |
| `--invert` | Off | Enable for dark holes on a bright background. |
| `--threshold` | `otsu` | Segmentation threshold: `otsu` or `triangle`. |
| `--area-min`, `--area-max` | `3`, `50` | Accepted thresholded component area, in pixels. Adjust for your hole sizes. |
| `--pad` | `3` | Padding around each detected component, in pixels. |
| `--pixel-size`, `--unit` | `1`, `px` | Calibration, e.g. `--pixel-size 2.5 --unit nm` for 2.5 nm/pixel. |

All available options: `python scripts/analyze_contact_holes.py --help`.

## How images are read

TIFF files use `tifffile.TiffFile(...).pages[page].asarray()`: only the selected
page is decoded into RAM. The file checksum is computed in chunks. PNG and
other formats use Pillow (`PIL.Image.open`); omit `--page` for these files.
The selected image must be **2D grayscale** with finite pixel values; export
a grayscale image first for RGB data. Original intensities are retained.

## Where to find the results

Open `contours.html` in the output directory for interactive inspection.
That directory also contains CD/intensity PNG maps, measurements and contours
in CSV files, all raw arrays in `raw_data.npz`, and run settings in `manifest.json`
(including the TIFF index as `input_page`). With `--outputs compact` there is
no `contours.html` or `contours.csv`: write a viewer for a region with
`scripts/view_cd_region.py`, and load the arrays with `subpx.load_cd_results`.
