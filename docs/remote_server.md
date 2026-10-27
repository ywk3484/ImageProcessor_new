# Run the contact-hole CD pipeline

## Test throughput with the native-pixel methods

For the large-dataset speed trial, use the same two paths as `cd_compare.py`:

```powershell
python scripts\cd_speed.py D:\results\trial_w1 D:\data\dataset1\stripe_001\scan_001.tif
```

This measures up to 512 consecutive pages with all visible GPUs, one worker
each, using the new `pixel_half_area`, `pixel_half_area_min_avg` and
`pixel_tail_area` methods. They measure connected areas at original pixel
resolution and skip the 4x enlargement. The command reports actual pages/s,
including startup and I/O, and conditional dataset-time estimates. It reuses
the prior run's numerical and screening settings and creates a fresh output
folder. See [method definitions and local timing](contact_holes_cd.md#fast-page-averages-and-an-actual-batch-timing).

For a full dataset, use these three names in `cd_batch.py --methods`, retain
your other input/calibration/screening options, and choose a new output folder.

## Compare the faster experimental methods on your existing trial

The goal is to compare speed, coverage and CD changes before another whole-dataset run. From the repository root
in the remote PC's Python environment, run one command with the previous trial folder and its TIFF:

```powershell
python scripts\cd_compare.py D:\results\trial_w1 D:\data\dataset1\stripe_001\scan_001.tif
```

This reuses the trial's calibration and measurement settings. It selects up to eight previously measured pages
across the recorded runtime range, including the slowest. On one GPU it compares the model methods with three
new methods, `local_half_area`, `local_half_area_min_avg` and `local_tail_area`. The local methods measure raw
Voronoi cells and skip Gaussian deblending. Both paths are warmed before timing. Two selected pages also get a
tighter model fit (at least 60 rounds allowed and tolerance at most 1e-4). This is a sensitivity check, **not a
ground-truth reference**, and its convergence flag is saved. For exact corresponding pages across repeated
datasets, add `--pages 94,46` to each run. Page indices start at zero.

The last lines give seconds per page for each scheme and the extra-round change in page means. No output needs
to be typed back to use the results. Everything is saved in a new `comparison_<date>_<time>` folder under the trial:

- `pages.csv`: timings, coverage, convergence and each method's page means.
- `changes.csv`: model/local differences, both over each method's valid population and over mutually matched
  positions; unmatched positions and changed valid flags are counted separately.
- `stability.csv`: ordinary versus tighter model fits, including whether the tighter fit converged.
- `screen.csv`: raw and centered hole-check errors on selected hole pages and up to two formerly skipped pages.
  The acceptance columns use the new defaults, explicitly named; they do not reproduce a custom old threshold.
- `holes/*.npz`: seed positions, measured centers, CD/x/y values in pixels, calibration, validity and statuses.
  These small files support matching the same physical holes across repeated datasets without saving image stacks.
- `summary.json`: the settings, code fingerprint, prior-run provenance and timing summary.

The reported speed is **one-device computation on selected hole pages**, excluding disk reads and warm-up.
It is not whole-dataset throughput or a repeatability ranking. The three area definitions are different from
the model-based definitions; in particular, local tail area is confined to its cell. A region touching its cell
boundary is invalid. Independent gain/offset invariance holds for fixed cell territories; detection and cell
boundaries are still recomputed, and overlapping neighbours remain uncorrected. Compare coverage and corresponding
pages from all repeated datasets, as well as sensitivity to real size changes. `--local-methods` can additionally
select `local_projection`, but steepest-edge selection can switch on concave/multilobed shapes, so it is optional.

Local validation on an RTX 4060 Ti, a 1378 x 2048 drifting synthetic page with 39,042 holes: the ten model methods
took 3.56 s, and all four local methods including projection took 0.50 s (median of three warmed runs). The model
fit used 30 rounds and did not converge. Local tail area measured 38,037 holes versus 39,035 for model tail area;
their mean CDs were 133.02 versus 143.40 nm at 21 nm/pixel. This demonstrates speed and changed coverage/definition,
not better real-data precision. On that page, model metrics-only output took 3.56 s versus 3.67 s retaining display
arrays; the larger gain comes from bypassing deblending. Real remote gains must come from the comparison above.

After inspecting the pilot, run the local candidates through the normal multi-GPU batch tool into a **new** folder:

```powershell
python scripts\cd_batch.py D:\data\dataset1 --reference D:\data\reference.tif --reference-page 12 --output D:\results\dataset1_local --pixel-size 21 --unit nm --workers-per-gpu 1 --methods local_half_area local_half_area_min_avg local_tail_area
```

Existing method names and the batch's default ten methods remain available. Mixing local and model methods in
one call aligns them to the model's final seed list but still performs deblending; use local-only methods for the
fast path. Local-only `cell_intensity` is blank because that existing column belongs to model `tail_area`.

New batch outputs include `deblend_converged`, `deblend_iterations` (total over all passes) and `deblend_passes`.
Local-only rows have blank convergence and zero rounds. Unconverged model rows remain available, explicitly
flagged. `--deblend-iterations` and `--deblend-tolerance` control the model solve; seed revision can add rounds
beyond the final-pass budget. Stronger convergence alone does not validate the Gaussian model.

Resume now checks measurement-code and input identities. Older output folders, changed implementations, or changed
input lists are refused: use a new folder so the comparison cannot silently mix results. Copying the update over an
existing checkout does not delete or convert earlier results. The numerical fixes correct winding-region growth
and statuses of accepted fits; pathological cases can therefore change even for an existing method name.

The default hole screen now subtracts each image's mean before correlation and rejects constant images. Its
initial threshold is `--max-error 0.8`, equivalent to an absolute normalized correlation above 0.6. This is a
practical initial threshold, not a validated classifier for every dataset; inspect `screen.csv` and skipped-page
coverage. `--hole-check-mode raw` retains the old calculation and default 0.3 threshold. An explicitly supplied
`--max-error` overrides either default. For a preselected set known to contain holes, use `--no-hole-check`.

## Single-image inspection

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
| `--hole-check-mode` | `centered` | Subtract each image's mean before correlation, rejecting zero contrast. `raw` retains the previous uncentered calculation. |
| `--max-error` | `0.8` centered; `0.3` raw | Accept errors below this value. Centered 0.8 requires absolute normalized correlation above 0.6; inspect acceptance on your data. |
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
  or measurement-code/input identities are refused.
- `invocations.jsonl`: elapsed time and completed-page counts for each invocation, excluding pauses between runs.

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

## What part 1 left open (`scripts/cd_diagnose.py --part 2`)

After part 1's three lines, the same command with `--part 2`, again while nothing else uses the GPUs:

```powershell
python scripts\cd_diagnose.py D:\results\trial_w1 D:\data\dataset1\stripe_001\scan_001.tif --part 2
```

It takes about 20 minutes. First it looks at the GPUs and Windows: other programs on each GPU, Remote Desktop's GPU
setting, PCIe power saving and each GPU's round trip. It then times the median page on the current power plan and on
High performance. Switching the plan is the only change it makes, for less than a minute, and it then sets back the
plan it found (also after Ctrl+C or an error). Next it runs the real `cd_batch.py` with 1, 2, 3 and 4 workers per GPU
on the 160 pages of the TIFF that hold the most hole pages, each into a temporary folder that is removed afterwards;
a worker count that gets no new page for 10 minutes is stopped. Last, it measures how the deblending drifts on the
median and the slowest page and tries cures for it, five on the median page and two on the slowest (a few minutes).
Do not click into its window while it runs (a click pauses it until you press Esc). It ends with a summary and six
short lines; type only those six lines back. Everything measured is also written to `diagnosis2.txt` in the trial's
folder. On this PC (one RTX 4060 Ti, the synthetic test file of 48 pages) it took 22 minutes, and the lines read:

```
v2e59/75323+4/75323 a26 p26 u-c e1 h28/27/52/51/3.5/3.6 wa
t1/0.35/3.4/96/1.5 n0
k0.34/3.5/5/0.3/41/23/12.5+0.39/6.4/9.3/0.6/46/32/11.1+0.37/10.4/12.1/0.9/50/49/9.4+0.03/28/15.5/3.6/54/45/5.2
y1/0 d200/2/1.3/0.07/0/-3.8 m31/125/143/127/123/0.17/96/100
c0.99/1.27/1.26/71/946 x110/0.05/0.6+15/3.25/20.58+32/0.01/0.4+96/0.04/0.59+110/0.05/0.6/0
r200/2/1.24/105/2/9.1/0.06/33/2/4.5/0 f2.3/3.05/40
```

The last group of `k` is 4 workers per GPU on this 16 GB card: 15.5 GB of GPU memory in use, 3.6 GB of it moved to
system RAM, 28 s per page instead of 3.5 and two pages failed (a PROBLEM line of the summary says so); in another
run they finished no page in 10 minutes and were stopped, which the summary reports the same way.
