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

## Whole dataset: average CD of every page (`scripts/cd_batch.py`)

Runs the report's 10 CD methods on every page of every TIFF of one dataset, on all GPUs, and writes one row per
page with each method's average CD. Pages without holes are skipped by comparing each page with a reference image
that has holes.

```bash
# Trial: the first 20 pages of one file (prints pages per second; see timings.csv)
python scripts/cd_batch.py /data/dataset1/scan_001.tif --reference /data/reference.tif \
    --output /results/dataset1 --pixel-size 21 --unit nm --pages 0:20

# The whole dataset (folders are searched for .tif/.tiff); the trial's pages are kept and not measured again
python scripts/cd_batch.py /data/dataset1 --reference /data/reference.tif --output /results/dataset1 \
    --pixel-size 21 --unit nm
```

Keep the reference outside the dataset folder, or it is measured as a page too.

| Parameter | Default | Meaning |
|---|---|---|
| inputs | | TIFF files, folders (searched recursively) or wildcards. All pages of all files are measured. |
| `--reference`, `--reference-page` | required, `0` | An image with holes. A page counts as having holes when its phase cross-correlation error with it is below `--max-error`. |
| `--max-error` | `0.3` | scikit-image's `phase_cross_correlation(reference, page, normalization=None)` error: 0 = the same pattern at some shift. Synthetic test: hole pages 0.16-0.26, noise / lines / pads 0.47-0.61. |
| `--error-function` | none | `FILE.py:NAME`: your own `NAME(reference, image) -> error` instead (NumPy arrays in, a number out; it may use the GPU). |
| `--no-hole-check` | off | Measure every page. |
| `--pages` | all | e.g. `0:20` (first 20 pages of every file), `100:` (from page 100). Page 0 is the first. |
| `--gpus`, `--workers-per-gpu` | all GPUs, `2` | Each worker measures one page at a time, all its holes at once on its GPU. More workers per GPU help when `timings.csv` shows much CPU time per page. |
| `--page-timeout` | `600` | Seconds after which a page counts as hung: it is listed in `failed.csv` and its worker restarted. |
| method options | as `cd_report.py` | `--methods`, `--half-level`, `--peak-pixels`, `--tail-level`, `--tail-window`, `--seeds`, `--area-min`, `--area-max`, `--invert`, `--pixel-size`, `--unit`. |

Outputs in `--output`:

- `pages.csv`: one row per page. `file` (path below the input folder), `page`, `status` (`ok` or `no_holes`),
  `pcc_error`, `holes`, `image_intensity` (mean of all pixels), `cell_intensity` (mean of tail_area's cell
  intensity over the holes). For each method: `<method>_valid` (holes measured) and `<method>_cd`, `_cd_x`, `_cd_y`
  (their mean CD and mean x / y width, in `--unit`). No other statistics are computed.
- `timings.csv`: seconds per stage, the cell stack size, and the worker's peak memory for every page.
- `failed.csv`: pages that failed, with the error. Running the same command again retries them.
- `run.json`: the settings. Running again into the same `--output` resumes; other method or hole-check settings
  are refused.

Ctrl+C (or `docker stop`) lets the pages being measured finish, then stops; run the same command again to continue.
In Docker the container needs the GPUs (`--gpus all`, or compose's GPU reservation).
