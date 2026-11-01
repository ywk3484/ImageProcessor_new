# Preprocessing: noise reduction and the average hole

The noise-reduction and resolution "Do"s of the ACTIS preprocessing proposal
(`docs/actis_preprocessing_proposal.html`), built so they can be checked on real
pages. The functions are general: any 2-D image, any feature layout (square,
staggered, irregular) and any feature shape. NumPy only. Built: fixed Gaussian
smoothing, the Wiener filter tuned by Noise2Self, the mean-hole denoiser and the
super-resolved average hole. Not built: blur matching, deconvolution, BM3D and
AI denoisers.

**The short version.** On a synthetic page with known truth, the settings that
remove the most pixel noise also shrink real CD differences. Gaussian and Wiener
smoothing shrink both the page-wide step and the hole-to-hole differences. The
mean hole with an automatic K wipes out most of a real +5 % CD step. With K = 32
it keeps the CD differences, but then it removes little noise from the CDs. So a
smaller CD spread after preprocessing is **not** better precision unless the CDs
still follow the true size one to one. Check that before trusting a
preprocessed result.

## Functions (`subpx.preprocess`)

```python
from subpx import smooth_gaussian, denoise_wiener, denoise_mean_hole, average_hole

smooth = smooth_gaussian(image, sigma=0.6)            # PreprocessResult: .image (float32), .parameters, .meta
wiener = denoise_wiener(image)                         # noise="auto" (Noise2Self) or a standard deviation
clean = denoise_mean_hole(image, centers_xy)           # components="auto" or K, radius="auto" or px
mean = average_hole(image, centers_xy, upsample=4)     # AverageHoleResult: .image on a 1/4 px grid, .counts, .offsets
```

| Function | What it does | Settings |
| --- | --- | --- |
| `smooth_gaussian` | Averages every pixel with its neighbours | `sigma` px (the proposal's 0.5 to 0.7) |
| `denoise_wiener` | Takes out the noise level the page's own spectrum shows | `noise`: `"auto"` picks the level with the least self-predicted pixel error (Noise2Self); `meta["loss"]` holds the curve, `meta["at_edge"]` flags a pick at the end of the search |
| `denoise_mean_hole` | Rebuilds every feature from the average feature plus its K main shape differences. Pixels farther than the box from every feature stay unchanged | `components`: `"auto"` (Noise2Self, the least pixel error) or K; `radius` of the rebuilt box (auto: 3/4 of the median spacing). `meta["brightness"]` is each feature's fitted brightness |
| `average_hole` | Stacks every feature at its own subpixel centre on a grid finer than the pixels, after matching each feature's brightness and background | `upsample` (4 = 1/4 px bins), `radius` (auto: the median spacing). Bins no pixel fell into are NaN |

`centers_xy` is an `(N, 2)` array of `[x, y]`, e.g. from `detect_centers`. All
outputs keep the image's grey levels, and brighter or dimmer features are
handled by a per-feature brightness fit.

## Does it keep real CD differences? (synthetic truth)

`scripts/preprocess_check.py` makes a page with known truth and measures it raw
and preprocessed. The page has pitch 8.2 px, 4.8 px holes and a Gaussian PSF of
σ 1.4 px; hole sizes vary 4 % from hole to hole, holes in the right half are
5 % larger, brightness flickers 10 %, and noise is 4 % of the peak. Every
measurement is compared with the same method on the noise-free page. Half-area
CD, three layouts (square, staggered, mixed shapes):

| Version | +5 % step measured as | Hole-to-hole slope (1 = full size) | Scatter (px) |
| --- | --- | --- | --- |
| Raw | +4.7 to +4.9 % | 0.85 to 0.88 | 0.18 to 0.19 |
| Gaussian 0.6 px | +3.8 to +4.4 % | 0.77 to 0.80 | 0.12 to 0.13 |
| Wiener, auto | +4.2 to +4.9 % | 0.50 to 0.54 | 0.06 to 0.07 |
| Mean hole, auto K (3 to 5) | +1.0 to +2.2 % | 0.63 to 0.74 | 0.09 to 0.12 |
| Mean hole, K = 16 | +3.7 to +3.8 % | 0.82 to 0.85 | 0.16 to 0.17 |
| Mean hole, K = 32 | +4.3 to +4.6 % | 0.87 to 0.89 | 0.16 to 0.17 |

Read it this way: Wiener's scatter is a third of raw, but hole-to-hole
differences come out at half their size. That is smoothing, not precision. The
projection method is noisier and less regular in every version (see the script's
output).

The library tests (`tests/test_preprocess.py`) check that Gaussian equals SciPy's
filter; that Wiener's automatic noise level lands near the true one and is close
to the best fixed level; that the mean hole lowers pixel error on square,
staggered and mixed-shape layouts and leaves open areas unchanged; that its
brightness readout follows the planted brightness (r > 0.98); and that the
average hole matches the true shape within 0.6 grey levels RMS.

## On a real ACTIS page

`artifacts/cd_test/data/stripe1_board1.tif`, page 0 (2048 x 1378, 39,042
holes, spacing 8.2 px), CPU:

| Version | Time (whole page) | Found |
| --- | --- | --- |
| Gaussian 0.6 px | 0.1 to 0.2 s | removes 6.2 grey levels (sd) |
| Wiener, auto | 2.1 to 2.3 s | noise 7.3 grey levels; 11 % of the frequencies kept |
| Mean hole, auto | 5.0 to 5.4 s (includes finding the holes) | K = 2, box radius 7 px, all 39,042 holes rebuilt |
| Mean hole, K = 32 | 2.2 s | |
| Average hole | 0.4 s for 1,610 holes | fewest 66 pixels per 1/4 px bin, brightness spread 1.5 % |

CD in the region x 600-1000, y 400-700 (1,610 holes). "Per-hole r" is the
correlation of each hole's CD with its CD on the raw page:

| Method | Version | Mean CD (px) | sd (px) | Failed | Per-hole r with raw |
| --- | --- | --- | --- | --- | --- |
| Half area | raw | 4.40 | 0.161 | 0 | 1 |
| | Gaussian 0.6 | 4.50 | 0.122 | 0 | 0.87 |
| | Wiener auto | 4.47 | 0.075 | 0 | 0.52 |
| | Mean hole auto (K = 2) | 4.45 | 0.066 | 0 | 0.33 |
| | Mean hole K = 32 | 4.41 | 0.156 | 0 | 0.94 |
| Gaussian fit | raw | 4.52 | 0.105 | 0 | 1 |
| | Mean hole auto (K = 2) | 4.48 | 0.070 | 0 | 0.85 |
| Projection | raw | 4.14 | 0.450 | 80 | 1 |
| | Gaussian 0.6 | 4.18 | 0.332 | 197 | 0.71 |
| | Mean hole auto (K = 2) | 4.03 | 0.061 | 50 | 0.08 |

The automatic settings cut the spread by 1.5 to 7 times. They also weaken the
link to each hole's raw CD (r 0.85 down to 0.08, lowest for the noisiest
method). There is no truth here, so this page
alone cannot tell whether the raw hole-to-hole differences were noise or real.
The synthetic test says the automatic settings erase real differences too. The
test that decides it on real data is **repeatability**: measure two exposures
of the same holes. Preprocessing helps only if each hole's CD agrees *better*
between the two exposures than it does raw.

Gaussian smoothing also raises the mean CD (half area +0.10 px) and more than
doubles the projection method's failures.

## In the workbench

Setup step **2 Preprocess** picks the method and its settings; 0 means
automatic. The **Preprocessed** switch above the image (key **P**) flips
between the raw page and the preprocessed page in the same place, with the same
contrast. Each version keeps its own previewed holes, CD and pitch, and runs
measure the version on screen. Turning the switch on makes the preprocessed
page when the page has none yet; so does showing another page while it is on.
The step reports what the method found (K or noise level, holes rebuilt, grey
levels removed) and says when the pick is at the end of its range.

Comparisons with the other version:

- the summary line shows the other version's mean CD;
- **CD distribution** draws the other version as an outline and gives its mean
  and sd;
- the **hole inspector** adds a "CD raw" / "CD pre" column for the same hole
  (matched by position).

**Show > Removed by preprocessing** maps raw minus preprocessed (red: lowered,
blue: raised). Hole-shaped structure there means preprocessing changed the
holes, not only the noise. The **Average hole** tab stacks every hole of the
result on screen on a 1/4 px grid, on the raw and on the preprocessed page,
with x and y profiles and the width at half height of each.

Changing a Preprocess setting marks only the preprocessed version as out of
date. Its banner offers Re-run (makes the preprocessed page again, then
measures) and Revert. Export writes `preprocessed_image.npy`,
`preprocess_meta.npz` (curves, per-hole brightness) and the settings in
`source.json`. The stack's other pages and Align use raw pages.
