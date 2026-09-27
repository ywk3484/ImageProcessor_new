# Run the contact-hole CD pipeline

```powershell
# TIFF: select page 10 (the 11th image).
python scripts/analyze_contact_holes.py "D:\data\images.tiff" --page 10 --output "D:\results\page_10"

# PNG:
python scripts/analyze_contact_holes.py "D:\data\image.png" --output "D:\results\image"
```

Run from the repository root on the server, using your Python environment.
Replace the input and output paths. This runs all three CD methods on the CPU.

## Input parameters

| Parameter | Default | Meaning |
|---|---|---|
| First argument | Supply your image path | One grayscale image to analyze. |
| `--page` | `0` | TIFF page index: `0` is the first image; `1499` is the last in a 1,500-page file. |
| `--output` | `artifacts/contact_holes_cd` | Output directory. Existing named files are overwritten. |
| `--methods` | `logquad gradient halfmax` | Methods to run: Gaussian FWHM, radial maximum gradient, and half-height contours. |
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
(including the TIFF index as `input_page`).
