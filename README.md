# Sudoku Scanner

Python pipeline for detecting a Sudoku grid, recognizing digits, and validating the extracted result.

## Setup

```bash
cd sudokuscan
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Tesseract OCR must also be installed separately, for example with `brew install tesseract` on macOS.

## Usage

```bash
python sudoku_scan.py data/dev/your-image.jpg --output-dir out --debug
```
