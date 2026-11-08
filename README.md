# Sudoku Scanner

Detects a printed sudoku in a photo, rectifies it, reads the 81 cells and checks whether the grid is a valid sudoku. The four puzzle corners come from `cv2.cornerHarris`, refined with `cv2.cornerSubPix`.

## Setup

```bash
cd sudoku_scan
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tesseract OCR is a separate program and must be installed on the system:

```bash
brew install tesseract              # macOS
sudo apt install tesseract-ocr      # Ubuntu / Debian
tesseract --version                 # check: version 4.1 or newer
```

## Usage

```bash
python sudoku_scan.py --input photo.jpg --outdir out/
python sudoku_scan.py --batch images_dir/ --outdir results/
```

- `--input` processes one image.
- `--batch` processes every `.jpg`, `.jpeg` and `.png` file in a folder.
- `--outdir` is where the output files go (default: `out`).
- Run without arguments, the program processes `data/dev/images/dev_01.jpg`.

If no sudoku is found, the result file is still written with `"verdict": "not_found"`.

## Output files

For an input `<name>.jpg`:

| File | Content |
|---|---|
| `<name>_grid.txt` | The recognised grid: 9 lines of 9 digits, `0` for an empty cell |
| `<name>_result.json` | Corners (TL, TR, BR, BL, original pixels), grid, A3, A4, H4, A3 fourth-corner error, A4 RMS residual, verdict, runtime |
| `<name>_harris.png` | Harris response heat map with candidate points and the four selected corners |
| `<name>_rect_A3.png` | 450 × 450 rectification with the exact affine from TL, TR, BL |
| `<name>_rect_A4.png` | 450 × 450 rectification with the least-squares affine from all four corners |
| `<name>_rect_H4.png` | 450 × 450 rectification with the homography from all four corners |
| `<name>_debug.png` | H4 image with cell borders, recognised digits, empty-cell marks and conflicting cells in red |

When the verdict is `not_found`, only `<name>_result.json` and `<name>_harris.png` are written.

## Project layout

### `pipeline/`

| Module | Role |
|---|---|
| `preprocess.py` | Downscale to 1200 px long side, grey, Gaussian blur, illumination normalisation |
| `corners.py` | Harris response, puzzle quadrilateral selection, snapping to Harris peaks, `cornerSubPix` refinement |
| `grid_check.py` | Rejects quadrilaterals that are not a 9 × 9 grid or contain fewer than 17 digits |
| `orient.py` | Picks the corner order (0°, 90°, 180°, 270°) with the most confident digit reading |
| `rectify.py` | A3, A4 and H4 fitting, their errors, and warping |
| `cells.py` | Grid-line removal, 81 cell windows, empty / filled decision, digit mask |
| `ocr.py` | Digit preparation and Tesseract reading (six reads per cell) |
| `digit_rules.py` | Shape rules per digit (holes, hole position, left-side ink) that check Tesseract's answers |
| `validate.py` | Row, column and box conflicts, verdict |
| `io_utils.py` | Image, annotation, result and grid file input and output |
| `visualize.py` | Harris and debug images |
| `metrics.py` | Corner error, cell accuracy, F1 and confusion matrix |

### `analysis/`

| Script | Role |
|---|---|
| `annotate.py` | Annotation tool: click the four corners, type the grid, writes Appendix A JSON |
| `convert_heic.py` | Converts the personal HEIC photos to JPEG |
| `diagnose_harris.py` | Compares quadrilateral corners, Harris peaks and ground truth per corner |
| `eval_corners.py` | Corner error on the dev set |
| `eval_rectify.py` | A3, A4, H4 errors on the dev set and an A4 check against `np.linalg.lstsq` |
| `eval_cells.py` | Empty / filled F1 for different centre bands and window sizes |
| `eval_ocr.py` | Digit recognition accuracy and confusion matrix on the dev set |
| `evaluate.py` | Scores a folder of result files against annotations, writes CSVs and misread cell crops |
| `obliqueness.py` | Obliqueness measure (plane tilt from the homography) and its check against the dev set angles |
| `compare_models.py` | A3 / A4 / H4 comparison: fit error, grid-line regularity and OCR accuracy versus tilt |
| `harris_params.py` | Harris parameter study, one parameter at a time |
| `ocr_compare.py` | OCR configurations: pipeline, single psm modes, prepared versus raw cells |
| `stress_test.py` | Robustness on rotated, downscaled, blurred, compressed, darkened and noisy copies |
| `tables.py` | Builds `report/tables.md` from the CSVs |
| `plots.py` | Builds the figures in `report/figs/` from the CSVs |
| `walkthrough.py` | Saves every pipeline stage of one image |

### Data and report

| Folder | Content |
|---|---|
| `data/dev/` | Development set: 20 images, annotations, `index.csv` |
| `data/personal/` | 12 personal photos (HEIC originals and JPEG), annotations in `annotations/`, see its README |
| `report/` | Figures (`figs/`) and tables (`tables.md`) used in the report |

## Reproducing the evaluation

```bash
python sudoku_scan.py --batch data/dev/images --outdir out/dev
python sudoku_scan.py --batch data/personal --outdir out/personal
python -m analysis.evaluate --results out/dev --annotations data/dev/annotations --images data/dev/images --out out/eval/dev
python -m analysis.evaluate --results out/personal --annotations data/personal/annotations --images data/personal --out out/eval/personal
python -m analysis.ocr_compare
python -m analysis.compare_models --images data/dev/images --annotations data/dev/annotations --out out/eval/compare_dev.csv
python -m analysis.compare_models --images data/personal --annotations data/personal/annotations --out out/eval/compare_personal.csv
python -m analysis.harris_params
python -m analysis.stress_test
python -m analysis.tables
python -m analysis.plots
```
