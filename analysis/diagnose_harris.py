from pathlib import Path
import numpy as np
from pipeline.io_utils import load_annotation, load_image
from pipeline.preprocess import preprocess
from pipeline.corners import find_quads, select_quad, order_corners, to_original, harris_response, find_candidates

IMAGE_DIR = Path("data/dev/images")
ANNOTATION_DIR = Path("data/dev/annotations")
NAMES = ["TL", "TR", "BR", "BL"]

for img_path in sorted(IMAGE_DIR.glob("*.jpg")):
    gt, _ = load_annotation(ANNOTATION_DIR / f"{img_path.stem}.json")
    gt = np.array(gt, dtype=float)

    gray, scale = preprocess(load_image(img_path))
    quad = select_quad(find_quads(gray), gray.size)
    anchors = order_corners(to_original(quad, scale))
    candidates = to_original(find_candidates(harris_response(gray)), scale)

    for name, anchor, true_corner in zip(NAMES, anchors, gt):
        distances = np.linalg.norm(candidates - anchor, axis=1)
        harris = candidates[np.argmin(distances)]

        approx_gt = np.linalg.norm(anchor - true_corner)
        harris_gt = np.linalg.norm(harris - true_corner)
        approx_harris = np.linalg.norm(anchor - harris)

        print(f"{img_path.stem} {name} approx_gt={approx_gt:.1f} harris_gt={harris_gt:.1f} approx_harris={approx_harris:.1f}")
