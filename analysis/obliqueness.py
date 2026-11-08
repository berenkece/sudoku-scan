import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

INDEX_PATH = Path("data/dev/index.csv")
ANNOTATION_DIR = Path("data/dev/annotations")
FOCAL_35MM_TAG = 41989
FILM_LONG_SIDE_MM = 36.0
UNIT_SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def estimate_focal(centered):
    h1, h2 = centered[:, 0], centered[:, 1]
    focal_sq = np.nan
    if abs(h1[2] * h2[2]) > 1e-12:
        focal_sq = -(h1[0] * h2[0] + h1[1] * h2[1]) / (h1[2] * h2[2])
    if not focal_sq > 0 and abs(h2[2] ** 2 - h1[2] ** 2) > 1e-12:
        focal_sq = (h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2) / (h2[2] ** 2 - h1[2] ** 2)
    return float(np.sqrt(focal_sq)) if focal_sq > 0 else np.nan


def homography_tilt(corners, width, height, focal=None):
    homography, _ = cv2.findHomography(UNIT_SQUARE, np.asarray(corners, dtype=float))
    centered = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]) @ homography
    if focal is None or np.isnan(focal):
        focal = estimate_focal(centered)
    if np.isnan(focal):
        return np.nan, np.nan
    inverse_k = np.diag([1 / focal, 1 / focal, 1.0])
    r1, r2 = inverse_k @ centered[:, 0], inverse_k @ centered[:, 1]
    normal = np.cross(r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2))
    tilt = np.degrees(np.arccos(abs(normal[2]) / np.linalg.norm(normal)))
    return float(tilt), float(focal)


def exif_focal_px(path, width, height):
    focal_35mm = Image.open(path).getexif().get_ifd(0x8769).get(FOCAL_35MM_TAG)
    if not focal_35mm:
        return np.nan
    return float(focal_35mm) * max(width, height) / FILM_LONG_SIDE_MM


def true_tilt(row):
    tilt, pan = np.radians(float(row["tilt_deg"])), np.radians(float(row["pan_deg"]))
    return float(np.degrees(np.arccos(np.cos(tilt) * np.cos(pan))))


def validate_on_dev(out_path):
    rows = []
    with open(INDEX_PATH) as f:
        for row in csv.DictReader(f):
            corners = json.loads((ANNOTATION_DIR / row["image"].replace(".jpg", ".json")).read_text())["corners"]
            tilt, focal = homography_tilt(corners, int(row["width"]), int(row["height"]))
            rows.append({"image": row["image"], "true_deg": true_tilt(row), "estimated_deg": tilt, "focal_px": focal})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    errors = np.array([abs(r["estimated_deg"] - r["true_deg"]) for r in rows])
    for r, e in zip(rows, errors):
        print(f"{r['image']}: true {r['true_deg']:5.1f}  estimated {r['estimated_deg']:5.1f}  focal {r['focal_px']:6.0f}  error {e:5.2f}")
    print(f"median error {np.median(errors):.2f} deg, max {errors.max():.2f} deg")


if __name__ == "__main__":
    validate_on_dev(Path("out/eval/obliqueness_dev.csv"))
