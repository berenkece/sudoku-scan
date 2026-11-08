import csv
import json
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from pipeline.corners import harris_response, locate_corners
from pipeline.grid_check import has_enough_digits, is_sudoku_grid
from pipeline.io_utils import load_image
from pipeline.metrics import cell_accuracy, corner_error
from pipeline.ocr import recognize
from pipeline.orient import orient
from pipeline.preprocess import preprocess
from pipeline.validate import find_conflicts, verdict

SETS = (("data/dev/images", "data/dev/annotations"), ("data/personal", "data/personal/annotations"))
OUT_PATH = Path("out/eval/stress_test.csv")
WORKERS = 3


def move_points(points, matrix):
    points = np.asarray(points, dtype=float)
    return points @ matrix[:, :2].T + matrix[:, 2]


def rotate(image, corners, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(round(height * sin + width * cos)), int(round(height * cos + width * sin))
    matrix[0, 2] += (new_w - width) / 2
    matrix[1, 2] += (new_h - height) / 2
    rotated = cv2.warpAffine(image, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    return rotated, move_points(corners, matrix)


def scale(image, corners, factor):
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return small, (np.asarray(corners) + 0.5) * factor - 0.5


def blur(image, corners):
    return cv2.GaussianBlur(image, (0, 0), 2.0 * max(image.shape[:2]) / 1200), corners


def jpeg(image, corners):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 20])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR), corners


def darken(image, corners):
    table = (np.linspace(0, 1, 256) ** 2.5 * 255).astype(np.uint8)
    return cv2.LUT(image, table), corners


def noise(image, corners):
    rng = np.random.default_rng(0)
    noisy = image.astype(np.float32) + rng.normal(0, 15, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), corners


PERTURBATIONS = {
    "original": lambda im, c: (im, np.asarray(c, dtype=float)),
    "rot90": lambda im, c: rotate(im, c, 90),
    "rot180": lambda im, c: rotate(im, c, 180),
    "rot270": lambda im, c: rotate(im, c, 270),
    "rot45": lambda im, c: rotate(im, c, 45),
    "scale50": lambda im, c: scale(im, c, 0.5),
    "scale25": lambda im, c: scale(im, c, 0.25),
    "blur": blur,
    "jpeg20": jpeg,
    "dark": darken,
    "noise": noise,
}


def run_pipeline(image):
    pre = preprocess(image)
    corners = locate_corners(image, pre, harris_response(pre.gray))
    if corners is None or not is_sudoku_grid(image, corners) or not has_enough_digits(image, corners):
        return None, None, "not_found"
    corners, cells, _ = orient(image, corners)
    grid = recognize(cells)
    return corners, grid, verdict(find_conflicts(grid))


def run_case(case):
    image_path, annotation_path, name = case
    annotation = json.loads(Path(annotation_path).read_text())
    row = {"image": Path(image_path).name, "perturbation": name, "found": False, "crashed": False,
           "corner_err_pct": "", "cell_acc": "", "exact": False, "verdict": "", "error": ""}
    try:
        image, truth = PERTURBATIONS[name](load_image(Path(image_path)), annotation["corners"])
        corners, grid, result = run_pipeline(image)
        row["verdict"] = result
        if corners is not None:
            truth_grid = np.array(annotation["grid"])
            row.update(found=True, corner_err_pct=corner_error(corners, truth),
                       cell_acc=cell_accuracy(grid, truth_grid), exact=bool(np.array_equal(grid, truth_grid)))
    except Exception:
        row.update(crashed=True, error=traceback.format_exc(limit=1).strip().splitlines()[-1])
    return row


def main():
    cases = [(str(image_path), str(Path(annotation_dir) / f"{image_path.stem}.json"), name)
             for image_dir, annotation_dir in SETS for image_path in sorted(Path(image_dir).glob("*.jpg"))
             for name in PERTURBATIONS]
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(run_case, cases))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in PERTURBATIONS:
        chosen = [r for r in rows if r["perturbation"] == name]
        found = [r for r in chosen if r["found"]]
        errors = [r["corner_err_pct"] for r in found]
        bad = [r["image"] for r in chosen if not r["exact"]]
        print(f"{name:9} found {len(found):2d}/{len(chosen)}  crashed {sum(r['crashed'] for r in chosen)}  "
              f"exact {sum(r['exact'] for r in chosen):2d}  corner err mean {np.mean(errors) if errors else float('nan'):.3f}%  "
              f"max {max(errors) if errors else float('nan'):.3f}%  not exact: {bad}")
    for r in rows:
        if r["crashed"]:
            print("CRASH", r["image"], r["perturbation"], r["error"])


if __name__ == "__main__":
    main()
