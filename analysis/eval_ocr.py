import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from analysis.eval_cells import rectify_dev
from pipeline.cells import split_cells
from pipeline.io_utils import load_annotation, save_image
from pipeline.metrics import cell_accuracy, confusion_matrix, f1_score
from pipeline.ocr import RAW_PSMS, READ_PSMS, prepare, recognize

ANNOTATION_DIR = Path("data/dev/annotations")


def parse_args():
    parser = argparse.ArgumentParser(description="digit recognition accuracy on the dev set")
    parser.add_argument("--psm", type=int, help="use only this psm (default: the pipeline set)")
    parser.add_argument("--raw", action="store_true", help="give the unprepared gray cell to tesseract")
    parser.add_argument("--errors", type=Path, help="folder to save misread cells")
    return parser.parse_args()


def save_errors(folder, name, cells, pred, truth, prepared):
    for row, col in np.argwhere(pred != truth):
        cell = cells[row][col]
        if cell.digit is not None and prepared:
            image = prepare(cell)
        else:
            image = cell.gray
        save_image(folder / f"{name}_r{row}c{col}_true{truth[row, col]}_pred{pred[row, col]}.png", image)


def main():
    args = parse_args()
    preds, truths, accuracies, seconds = [], [], [], []
    for name, rectified, _ in rectify_dev():
        _, grid = load_annotation(ANNOTATION_DIR / f"{name}.json")
        truth = np.array(grid)
        cells = split_cells(rectified)
        start = time.perf_counter()
        psms = (args.psm,) if args.psm else (RAW_PSMS if args.raw else READ_PSMS)
        pred = recognize(cells, psms, prepared=not args.raw)
        seconds.append(time.perf_counter() - start)
        preds.append(pred)
        truths.append(truth)
        accuracies.append(cell_accuracy(pred, truth))
        wrong = [(int(r), int(c), int(truth[r, c]), int(pred[r, c])) for r, c in np.argwhere(pred != truth)]
        print(f"{name}: {accuracies[-1]:.4f}  {seconds[-1]:.2f}s  {wrong}")
        if args.errors:
            save_errors(args.errors, name, cells, pred, truth, not args.raw)

    pred_all, truth_all = np.array(preds), np.array(truths)
    exact = sum(np.array_equal(p, t) for p, t in zip(preds, truths))
    print(f"psm {args.psm or 'pipeline set'} {'raw' if args.raw else 'prepared'}")
    print(f"cell accuracy (mean over images): {np.mean(accuracies):.4f}")
    print(f"exact grids: {exact}/{len(preds)}")
    print(f"F1 empty: {f1_score(pred_all == 0, truth_all == 0):.4f}")
    print(f"OCR time per image: mean {np.mean(seconds):.2f}s, max {np.max(seconds):.2f}s")
    matrix = confusion_matrix(preds, truths)
    print("confusion (row = true, col = predicted, 0 = empty):")
    print(matrix)


if __name__ == "__main__":
    main()
