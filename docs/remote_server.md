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

Run the commands in PowerShell from the repository folder, with your Python environment active. Each command is one
line, so it pastes as it is; put a path with spaces in double quotes.

1. How fast your disk reads one TIFF (each read comes from the disk, not from RAM):

```powershell
python scripts\read_speed.py D:\data\dataset1\stripe_001\scan_001.tif
```

2. Once per machine: the first run compiles the GPU code (about 30 s before the first page) and keeps it for later
   runs.

```powershell
python scripts\cd_batch.py D:\data\dataset1\stripe_001\scan_001.tif --reference D:\data\reference.tif --reference-page 12 --output D:\results\warmup --pixel-size 21 --unit nm --pages 0:10
```

3. Trials: one whole TIFF (mostly pages with holes) with 1, 2 and 3 workers per GPU, each into its own folder. The
   three lines run one after another. Compare their last lines (`This run: N pages in T s`). Step 1 left the file in
   RAM, so they time the GPUs, not the disk.

```powershell
python scripts\cd_batch.py D:\data\dataset1\stripe_001\scan_001.tif --reference D:\data\reference.tif --reference-page 12 --output D:\results\trial_w1 --pixel-size 21 --unit nm --workers-per-gpu 1
python scripts\cd_batch.py D:\data\dataset1\stripe_001\scan_001.tif --reference D:\data\reference.tif --reference-page 12 --output D:\results\trial_w2 --pixel-size 21 --unit nm --workers-per-gpu 2
python scripts\cd_batch.py D:\data\dataset1\stripe_001\scan_001.tif --reference D:\data\reference.tif --reference-page 12 --output D:\results\trial_w3 --pixel-size 21 --unit nm --workers-per-gpu 3
```

4. The whole dataset: every .tif / .tiff in the folder and in all its subfolders, with the best worker count from
   step 3.

```powershell
python scripts\cd_batch.py D:\data\dataset1 --reference D:\data\reference.tif --reference-page 12 --output D:\results\dataset1 --pixel-size 21 --unit nm --workers-per-gpu 2
```

The reference is one page: `--reference D:\data\reference.tif --reference-page 12` uses page 12 of that TIFF and
nothing else of it (0 is the first page, so ImageJ's slice 13 is page 12). The run reads it once, at the start, and
its first line names it: `Reference: page 12 of D:\data\reference.tif (2048 x 1378 px)`. A page number the file does
not have stops the run before anything is written. The page may also be one of the dataset's own pages
(`--reference D:\data\dataset1\stripe_001\scan_001.tif --reference-page 37`): that file is measured as usual. A
separate reference file belongs outside the dataset folder, or its pages are measured as data too.

The `file` column is each file's path below the dataset folder (`stripe_001/scan_001.tif`), and a resumed run skips
the (file, page) pairs already in `pages.csv`. So run a dataset with one command and resume it with the same
command; give trials their own `--output` (a file given alone is named without its folder). Running one subfolder at
a time into the same `--output` would take same-named files of different subfolders as already measured.

A run takes hours. Start it in a PowerShell window on the server; over Remote Desktop, leave with Disconnect, not Sign
out (signing out stops the run). Do not click into that window while it runs: in the standard console window a click
starts a text selection, and the run waits until you press Esc.

### Reading and run time

One reader thread reads the files one after another, each front to back, and hands the pages to the workers, which
measure while the next pages are read. It reads each page's header together with its pixels: one pass through the
file. `tifffile.imread` (also `len(tif.pages)` and `tif.series`) first reads every page's header, spread over the
whole file, and only then the pixels; on an HDD that is one seek per page. Cold on this PC's HDD, a 2.09 GB file of
740 pages of 2048 x 1378 px read in 13-14 s with a plain read, 16.7 s with this reader and 20.5 s with
`tifffile.imread` (a file of 1500 smaller pages: 15 s, 17.7 s, 28.6 s). Uncompressed pages are read one at a time
either way, so `maxworkers` makes no difference. If `scripts/read_speed.py`'s plain read takes as long as your usual
load, the disk itself is the limit: no reader is faster than it.

Measuring is the slower part: on an RTX 4060 Ti a synthetic 2048 x 1378 page with 30,000 holes takes 1.6 s (about
0.8 s on the GPU, the rest on the CPU); a page without holes only takes the hole check, ~0.01 s. Real pages are
slower: on 4 x RTX 6000 Ada a real page with holes took 3.5 s per GPU (2026-10-26); `scripts/cd_diagnose.py` (below)
finds out why on your machine. Workers that share a GPU take turns on it (Windows has no
NVIDIA multi-process service to let them run together), so a second worker helps only by doing its CPU part while
another uses the GPU; on this Windows PC it did not help. The trials decide: take the worker count after which the
pages per second stop rising, or the disk's speed is reached (its MB/s divided by 2.8 MB per page).

| Parameter | Default | Meaning |
|---|---|---|
| inputs | | TIFF files, folders (searched recursively) or wildcards. All pages of all files are measured. |
| `--reference`, `--reference-page` | required, `0` | The reference: one page with holes, page `--reference-page` of the `--reference` TIFF (0 = the first page). A page counts as having holes when its phase cross-correlation error with it is below `--max-error`. |
| `--max-error` | `0.3` | scikit-image's `phase_cross_correlation(reference, page, normalization=None)` error: 0 = the same pattern at some shift. Synthetic test: hole pages 0.16-0.26, noise / lines / pads 0.47-0.61. |
| `--error-function` | none | `FILE.py:NAME`: your own `NAME(reference, image) -> error` instead (NumPy arrays in, a number out; it may use the GPU). |
| `--no-hole-check` | off | Measure every page. |
| `--pages` | all | e.g. `0:20` (first 20 pages of every file), `100:` (from page 100). Page 0 is the first; no negative numbers. |
| `--gpus`, `--workers-per-gpu` | all GPUs, `2` | Each worker measures one page at a time, all its holes at once on its GPU. More workers per GPU help while the CPU part of a page is larger than its GPU part (see above). |
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

Ctrl+C lets the pages being measured finish, then stops (Ctrl+C again: at once); run the same command again to
continue. At the end of a run `pages.csv` and `timings.csv` are sorted by file and page; one that is open in another
program then (Excel keeps the files it opens locked) stays in the order measured, with all its rows.

## Why pages take as long as they do (`scripts/cd_diagnose.py`)

One command after a trial (step 3 above), with that trial's `--output` folder and its TIFF, while nothing else uses
the GPUs:

```powershell
python scripts\cd_diagnose.py D:\results\trial_w1 D:\data\dataset1\stripe_001\scan_001.tif
```

It takes 2-5 minutes and changes nothing. It measures the trial's median and slowest page with holes again, in
workers started as the batch starts them: one alone on a GPU, two on one GPU, one on every GPU, then one that times
every deblend step. It also times the machine (CPU, GPU round trip, a fixed synthetic page) and checks the numbers
(same page twice, against `pages.csv`, GPU against CPU code, 1-3 more deblend rounds). It ends with a summary and
three short lines; type only those three lines back. On this PC (one RTX 4060 Ti, synthetic test file) they read:

```
v8ea8/8526d+1/fc874 n0 p= wa-n0 u1w/1/1 c51/51/12 a30/401
x1.6/7/= f1.9 t1.6/2.7/1.6/1.6 m0.5/0.3/0.6/0.2 h61/87 d7/1/56/35/5 j1/12x13 y2/0/0/5
r2.9/30/2/2/13x13/1 e0/0/0 k0/0
```

Each token is a letter and its values, `-` where a value could not be measured; the top of the script says what each
one is. Everything measured is also written to `diagnosis.txt` in the trial's folder. `--pages 94,46` uses those two
pages instead of the trial's median and slowest.
