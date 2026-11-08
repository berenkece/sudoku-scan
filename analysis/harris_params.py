import csv
import json
from pathlib import Path

import numpy as np

import pipeline.corners as corners
from pipeline.io_utils import load_image
from pipeline.metrics import corner_error
from pipeline.preprocess import preprocess, to_gray

IMAGES = [
    ("data/dev/images/dev_01.jpg", "data/dev/annotations/dev_01.json"),
    ("data/dev/images/dev_07.jpg", "data/dev/annotations/dev_07.json"),
    ("data/dev/images/dev_12.jpg", "data/dev/annotations/dev_12.json"),
    ("data/dev/images/dev_15.jpg", "data/dev/annotations/dev_15.json"),
    ("data/dev/images/dev_17.jpg", "data/dev/annotations/dev_17.json"),
    ("data/dev/images/dev_20.jpg", "data/dev/annotations/dev_20.json"),
    ("data/personal/s042401168_04.jpg", "data/personal/annotations/s042401168_04.json"),
    ("data/personal/s042401168_08.jpg", "data/personal/annotations/s042401168_08.json"),
]
SWEEPS = {
    "HARRIS_BLOCK_RATIO": (0.003, 0.004, 0.005, 0.007, 0.012),
    "HARRIS_KSIZE": (3, 5, 7),
    "HARRIS_K": (0.02, 0.04, 0.06, 0.08, 0.12),
    "THRESH_REL": (0.001, 0.005, 0.01, 0.05, 0.1),
    "HARRIS_PEAK_REL": (0.01, 0.05, 0.1, 0.3, 0.5),
    "HARRIS_SEARCH_PX": (3, 6, 10, 15),
    "SUBPIX_WIN_PX": (0, 2, 3, 5, 8, 11),
}
OUT_PATH = Path("out/eval/harris_params.csv")
ALL_SETS = (("data/dev/images", "data/dev/annotations"), ("data/personal", "data/personal/annotations"))
ALL_OUT_PATH = Path("out/eval/harris_block_all.csv")


def run(pre, gray):
    response = corners.harris_response(pre.gray)
    candidates = len(corners.find_candidates(response))
    quad = corners.select_quad(corners.find_quads(pre.gray), pre.gray.size)
    if quad is None:
        return candidates, None, None
    peaks = corners.snap_to_harris(response, corners.order_corners(quad))
    if peaks is None:
        return candidates, None, None
    points = corners.to_original(peaks, pre.scale)
    if corners.SUBPIX_WIN_PX == 0:
        return candidates, points, points
    return candidates, points, corners.refine_subpix(gray, points, pre.scale)


def main():
    data = []
    for image_path, annotation_path in IMAGES:
        image = load_image(Path(image_path))
        data.append((Path(image_path).name, preprocess(image), to_gray(image),
                     json.loads(Path(annotation_path).read_text())["corners"]))
    rows = []
    for name, values in SWEEPS.items():
        default = getattr(corners, name)
        for value in values:
            setattr(corners, name, value)
            for image_name, pre, gray, truth in data:
                candidates, peaks, found = run(pre, gray)
                rows.append({"param": name, "value": value, "default": value == default, "image": image_name,
                             "candidates": candidates, "found": found is not None,
                             "harris_err_pct": corner_error(peaks, truth) if found is not None else "",
                             "corner_err_pct": corner_error(found, truth) if found is not None else ""})
            chosen = [r for r in rows if r["param"] == name and r["value"] == value]
            errors = [r["corner_err_pct"] for r in chosen if r["found"]]
            harris = [r["harris_err_pct"] for r in chosen if r["found"]]
            print(f"{name:20} {value:<6} {'*' if value == default else ' '} candidates {np.mean([r['candidates'] for r in chosen]):6.0f}"
                  f"  not_found {sum(not r['found'] for r in chosen)}  harris err {np.mean(harris) if harris else float('nan'):.3f}%  final err mean {np.mean(errors) if errors else float('nan'):.3f}%"
                  f"  max {max(errors) if errors else float('nan'):.3f}%")
        setattr(corners, name, default)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def block_on_all_images():
    data = []
    for image_dir, annotation_dir in ALL_SETS:
        for image_path in sorted(Path(image_dir).glob("*.jpg")):
            image = load_image(image_path)
            truth = np.array(json.loads((Path(annotation_dir) / f"{image_path.stem}.json").read_text())["corners"])
            data.append((image_path.name, preprocess(image), to_gray(image), truth))
    default = corners.HARRIS_BLOCK_RATIO
    rows = []
    for value in SWEEPS["HARRIS_BLOCK_RATIO"]:
        corners.HARRIS_BLOCK_RATIO = value
        for image_name, pre, gray, truth in data:
            _, _, found = run(pre, gray)
            error = min(corner_error(np.roll(found, -k, axis=0), truth) for k in range(4)) if found is not None else ""
            rows.append({"value": value, "default": value == default, "image": image_name, "corner_err_pct": error})
        errors = [r["corner_err_pct"] for r in rows if r["value"] == value and r["corner_err_pct"] != ""]
        print(f"all images, block ratio {value}: mean {np.mean(errors):.3f}%  max {max(errors):.3f}%")
    corners.HARRIS_BLOCK_RATIO = default
    with open(ALL_OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
    block_on_all_images()
