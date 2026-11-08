import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.obliqueness import exif_focal_px, homography_tilt
from pipeline.cells import CELL_IMAGE_SIZE, CELLS, split_cells
from pipeline.io_utils import load_image
from pipeline.metrics import cell_accuracy
from pipeline.ocr import recognize
from pipeline.grid_check import line_masks
from pipeline.rectify import RECT_SIZE, fit_a3, fit_a4, fit_h4, target_corners, warp_affine, warp_perspective

PEAK_GAP_RATIO = 0.5
STRIPS = 3
FIELDS = ["image", "conditions", "tilt_deg", "focal_px", "focal_source", "model", "fit_err_px",
          "spacing_mean_px", "spacing_max_px", "straight_mean_px", "straight_max_px", "lines_found", "cell_acc"]


def parse_args():
    parser = argparse.ArgumentParser(description="compare A3, A4 and H4 rectification")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="CSV file to write")
    return parser.parse_args()


def find_lines(profile, gap):
    profile = profile.astype(float).copy()
    peaks = []
    for _ in range(CELLS + 1):
        best = int(np.argmax(profile))
        if profile[best] <= 0:
            break
        peaks.append(best)
        profile[max(best - gap, 0):best + gap + 1] = 0
    return sorted(peaks)


def track_line(profile, position, gap):
    start = max(position - gap // 2, 0)
    window = profile[start:position + gap // 2 + 1]
    return start + int(np.argmax(window))


def grid_measures(rectified, valid):
    size = rectified.shape[0]
    gap = round(PEAK_GAP_RATIO * size / CELLS)
    horizontal, vertical = line_masks(rectified, valid)
    spacings, bends, found = [], [], 0
    for mask, axis in ((horizontal, 1), (vertical, 0)):
        lines = find_lines(mask.sum(axis=axis, dtype=float), gap)
        found += len(lines)
        if len(lines) != CELLS + 1:
            continue
        spacings.extend(np.abs(np.diff(lines) - size / CELLS))
        strip_positions = []
        for strip in np.array_split(np.arange(size), STRIPS):
            part = mask[:, strip] if axis == 1 else mask[strip, :]
            profile = part.sum(axis=axis, dtype=float)
            strip_positions.append([track_line(profile, line, gap) for line in lines])
        positions = np.array(strip_positions)
        bends.extend(positions.max(axis=0) - positions.min(axis=0))
    spacing = (np.mean(spacings), np.max(spacings)) if spacings else (np.nan, np.nan)
    bend = (np.mean(bends), np.max(bends)) if bends else (np.nan, np.nan)
    return spacing, bend, found


def models(image, corners):
    target = target_corners()
    big = target_corners(CELL_IMAGE_SIZE, CELL_IMAGE_SIZE)
    blank = np.full(image.shape[:2], 255, np.uint8)
    a3, a3_err = fit_a3(corners, target)
    a4, a4_rms, _ = fit_a4(corners, target)
    h4, h4_err = fit_h4(corners, target)
    yield ("A3", a3_err, warp_affine(image, a3), warp_affine(blank, a3),
           warp_affine(image, fit_a3(corners, big)[0], CELL_IMAGE_SIZE, CELL_IMAGE_SIZE))
    yield ("A4", a4_rms, warp_affine(image, a4), warp_affine(blank, a4),
           warp_affine(image, fit_a4(corners, big)[0], CELL_IMAGE_SIZE, CELL_IMAGE_SIZE))
    yield ("H4", h4_err, warp_perspective(image, h4), warp_perspective(blank, h4),
           warp_perspective(image, fit_h4(corners, big)[0], CELL_IMAGE_SIZE, CELL_IMAGE_SIZE))


def main():
    args = parse_args()
    rows = []
    for annotation_path in sorted(args.annotations.glob("*.json")):
        annotation = json.loads(annotation_path.read_text())
        image_path = args.images / annotation["image"]
        image = load_image(image_path)
        height, width = image.shape[:2]
        corners = np.array(annotation["corners"], dtype=float)
        focal = exif_focal_px(image_path, width, height)
        tilt, focal = homography_tilt(corners, width, height, focal)
        source = "estimated" if np.isnan(exif_focal_px(image_path, width, height)) else "exif"
        truth = np.array(annotation["grid"])
        for name, fit_err, rectified, valid, big in models(image, corners):
            (spacing_mean, spacing_max), (straight_mean, straight_max), found = grid_measures(rectified, valid)
            accuracy = cell_accuracy(recognize(split_cells(big)), truth)
            rows.append({"image": annotation["image"], "conditions": " ".join(annotation["conditions"]),
                         "tilt_deg": tilt, "focal_px": focal, "focal_source": source, "model": name,
                         "fit_err_px": fit_err, "spacing_mean_px": spacing_mean, "spacing_max_px": spacing_max,
                         "straight_mean_px": straight_mean, "straight_max_px": straight_max,
                         "lines_found": found, "cell_acc": accuracy})
            print(f"{annotation['image']:20} {tilt:5.1f} {name} fit {fit_err:8.2f}  spacing {spacing_mean:6.2f}/{spacing_max:6.2f}"
                  f"  straight {straight_mean:6.2f}/{straight_max:6.2f}  lines {found:2d}  acc {accuracy:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
