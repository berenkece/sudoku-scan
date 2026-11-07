import csv
from pathlib import Path

import numpy as np

from pipeline.corners import harris_response, locate_corners
from pipeline.io_utils import load_annotation, load_image
from pipeline.preprocess import preprocess
from pipeline.rectify import apply_affine, fit_a3, fit_a4, fit_h4, target_corners

IMAGE_DIR = Path("data/dev/images")
ANNOTATION_DIR = Path("data/dev/annotations")
INDEX_PATH = Path("data/dev/index.csv")


def load_side_ratios():
    with open(INDEX_PATH) as f:
        return {row["image"]: float(row["side_ratio"]) for row in csv.DictReader(f)}


def lstsq_a4(corners, target):
    design = np.column_stack((corners, np.ones(len(corners))))
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return solution.T


def lstsq_rms(matrix, corners, target):
    residuals = apply_affine(matrix, corners) - target
    return np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))


def evaluate():
    side_ratios = load_side_ratios()
    target = target_corners()
    rows = []
    for image_path in sorted(IMAGE_DIR.glob("*.jpg")):
        gt, _ = load_annotation(ANNOTATION_DIR / f"{image_path.stem}.json")
        image = load_image(image_path)
        pre = preprocess(image)
        pred = locate_corners(image, pre, harris_response(pre.gray))

        for source, corners in (("gt", np.array(gt, dtype=float)), ("pred", pred)):
            if corners is None:
                continue
            _, a3_error = fit_a3(corners, target)
            a4, a4_rms, inliers = fit_a4(corners, target)
            _, h4_error = fit_h4(corners, target)
            exact = lstsq_a4(corners, target)
            rows.append((image_path.name, source, side_ratios[image_path.name], a3_error, a4_rms,
                         lstsq_rms(exact, corners, target), np.abs(a4 - exact).max(), int(inliers.sum()), h4_error))
    return rows


def report(rows):
    print("image       src  ratio  a3_err  a4_rms  lstsq_rms  max|a4-lstsq|  inliers  h4_max")
    for name, source, ratio, a3, a4, exact, diff, inliers, h4 in sorted(rows, key=lambda r: (r[1], r[2])):
        print(f"{name:11} {source:4} {ratio:5.3f} {a3:7.3f} {a4:7.3f} {exact:10.3f} {diff:14.2e} {inliers:8d} {h4:7.1e}")


if __name__ == "__main__":
    report(evaluate())
