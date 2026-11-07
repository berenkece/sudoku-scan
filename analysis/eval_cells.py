from pathlib import Path

import numpy as np

import pipeline.cells as cells
from pipeline.cells import filled_mask, split_cells, warp_for_cells
from pipeline.corners import harris_response, locate_corners
from pipeline.io_utils import load_annotation, load_image
from pipeline.metrics import f1_score
from pipeline.preprocess import preprocess

IMAGE_DIR = Path("data/dev/images")
ANNOTATION_DIR = Path("data/dev/annotations")
MARGINS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18)
BANDS = (0.10, 0.15, 0.20, 0.25)


def rectify_dev():
    images = []
    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        _, grid = load_annotation(ANNOTATION_DIR / f"{image_path.stem}.json")
        image = load_image(image_path)
        pre = preprocess(image)
        corners = locate_corners(image, pre, harris_response(pre.gray))
        if corners is None:
            continue
        images.append((image_path.stem, warp_for_cells(image, corners), np.array(grid) > 0))
    return images


def evaluate(images, margin):
    preds, truths, errors = [], [], []
    for name, rectified, truth in images:
        pred = filled_mask(split_cells(rectified, margin))
        preds.append(pred)
        truths.append(truth)
        wrong = np.argwhere(pred != truth)
        if len(wrong):
            errors.append((name, [(int(r), int(c), "FP" if pred[r, c] else "FN") for r, c in wrong]))
    pred, truth = np.array(preds), np.array(truths)
    return f1_score(~pred, ~truth), f1_score(pred, truth), int(np.sum(pred & ~truth)), int(np.sum(~pred & truth)), errors


def report(images):
    print("band  margin  F1(empty)  F1(filled)  empty->filled  filled->empty")
    default_band = cells.CENTER_BAND
    for band in BANDS:
        cells.CENTER_BAND = band
        for margin in MARGINS:
            f1_empty, f1_filled, false_filled, false_empty, _ = evaluate(images, margin)
            print(f"{band:4.2f} {margin:7.2f} {f1_empty:10.4f} {f1_filled:11.4f} {false_filled:14d} {false_empty:14d}")
    cells.CENTER_BAND = default_band
    print(f"errors at band {cells.CENTER_BAND}, margin {cells.MARGIN_RATIO}:")
    for name, wrong in evaluate(images, cells.MARGIN_RATIO)[4]:
        print(f"  {name}: {wrong}")


if __name__ == "__main__":
    report(rectify_dev())
