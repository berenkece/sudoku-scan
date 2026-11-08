import argparse
from pathlib import Path

import cv2
import numpy as np

from pipeline.cells import CELLS, binarize, remove_lines, split_cells, warp_for_cells
from pipeline.corners import (count_inside, find_candidates, find_quads, harris_response, order_corners,
                              refine_subpix, select_quad, snap_to_harris, to_original)
from pipeline.io_utils import load_image, save_image
from pipeline.ocr import prepare, recognize
from pipeline.orient import orient, orientation_score
from pipeline.preprocess import preprocess, to_gray
from pipeline.rectify import fit_models, warp_perspective
from pipeline.validate import find_conflicts, verdict
from pipeline.visualize import draw_debug, draw_harris


def parse_args():
    parser = argparse.ArgumentParser(description="save every pipeline stage of one image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/walkthrough"))
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out / args.image.stem
    image = load_image(args.image)
    print("1 load:", image.shape)
    save_image(out / "01_original.png", image)

    pre = preprocess(image)
    print("2 preprocess: scale", round(pre.scale, 3), "working size", pre.gray.shape)
    save_image(out / "02_preprocessed.png", pre.gray)

    response = harris_response(pre.gray)
    candidates = find_candidates(response)
    print("3 harris: R.max", f"{response.max():.3g}", "candidates", len(candidates))
    save_image(out / "03_harris.png", draw_harris(pre.gray, response, candidates))

    binary = cv2.adaptiveThreshold(pre.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    quads = find_quads(pre.gray)
    quad = select_quad(quads, pre.gray.size)
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [q.reshape(-1, 1, 2) for q in quads], -1, (0, 200, 255), 1)
    cv2.drawContours(vis, [quad.reshape(-1, 1, 2)], -1, (0, 0, 255), 3)
    print("4 quads:", len(quads), "found; chosen quad area", int(cv2.contourArea(quad)), "with", count_inside(quad, quads), "quads inside")
    save_image(out / "04_quads.png", vis)

    anchors = order_corners(quad)
    peaks = snap_to_harris(response, anchors)
    corners = refine_subpix(to_gray(image), to_original(peaks, pre.scale), pre.scale)
    print("5 corners: quad->harris shift (working px)", np.round(np.linalg.norm(peaks - anchors, axis=1), 1).tolist(),
          "| subpix shift (original px)", np.round(np.linalg.norm(corners - to_original(peaks, pre.scale), axis=1), 1).tolist())
    save_image(out / "05_corners.png", draw_harris(pre.gray, response, candidates, (corners + 0.5) * pre.scale - 0.5))

    scores = [round(orientation_score(split_cells(warp_for_cells(image, np.roll(corners, -k, axis=0)))), 1) for k in range(4)]
    corners, cells, turn = orient(image, corners)
    print("6 orient: scores per turn", scores, "-> turn", turn)

    rect = fit_models(corners)
    rectified = warp_perspective(image, rect.h4)
    print("7 rectify: A3 corner error", round(rect.a3_corner_err_px, 1), "px, A4 RMS", round(rect.a4_rms_px, 1), "px, H4 max", f"{rect.h4_max_err_px:.1e}")
    save_image(out / "07_h4.png", rectified)

    big = warp_for_cells(image, corners)
    gray = to_gray(big)
    cell_binary = binarize(gray)
    ink = remove_lines(cell_binary, gray.shape[0] / CELLS)
    save_image(out / "08_binary.png", cell_binary)
    save_image(out / "09_lines_removed.png", ink)
    filled = [(r, c) for r in range(CELLS) for c in range(CELLS) if cells[r][c].digit is not None]
    print("8 cells:", len(filled), "filled of 81")

    prepared = [cv2.copyMakeBorder(cv2.resize(prepare(cells[r][c]), (80, 80)), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=128) for r, c in filled]
    while len(prepared) % 9:
        prepared.append(np.full((84, 84), 255, np.uint8))
    save_image(out / "10_prepared.png", np.vstack([np.hstack(prepared[i:i + 9]) for i in range(0, len(prepared), 9)]))

    grid = recognize(cells)
    conflicts = find_conflicts(grid)
    print("9 ocr: grid rows", ["".join(map(str, row)) for row in grid])
    print("10 validate:", verdict(conflicts), "conflicts", conflicts)
    save_image(out / "11_debug.png", draw_debug(rectified, grid, [[c.digit is not None for c in row] for row in cells], conflicts))
    print("saved to", out)


if __name__ == "__main__":
    main()
