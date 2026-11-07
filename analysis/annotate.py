import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from pipeline.corners import to_original
from pipeline.io_utils import load_annotation, load_image
from pipeline.rectify import RECT_SIZE, fit_h4, target_corners, warp_perspective
from pipeline.visualize import CORNER_NAMES

DEFAULT_FOLDER = Path("data/personal")
WINDOW = "annotate"
DISPLAY_LONG_SIDE = 800
ZOOM = 8
ZOOM_HALF_PX = 50
CELLS = 9
KEY_ACCEPT = (10, 13)
KEY_UNDO = ord("u")
KEY_REDO = ord("r")
KEY_ROTATE = ord("o")
KEY_QUIT = ord("q")


class Mouse:
    def __init__(self):
        self.position = (0, 0)
        self.click = None

    def __call__(self, event, x, y, flags, param):
        self.position = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    def take_click(self):
        click, self.click = self.click, None
        return click


def show(view, banner, position=None):
    frame = view.copy()
    if position is not None:
        cv2.drawMarker(frame, position, (0, 255, 255), cv2.MARKER_CROSS, 2000, 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, frame)


def wait_for_input(view, banner, mouse, crosshair=True):
    while True:
        show(view, banner, mouse.position if crosshair else None)
        key = cv2.waitKey(20) & 0xFF
        click = mouse.take_click()
        if click is not None and crosshair:
            return "click", click
        if key != 0xFF:
            return "key", key


def zoom_crop(image, center):
    height, width = image.shape[:2]
    size = 2 * ZOOM_HALF_PX + 1
    x0 = int(np.clip(round(center[0]) - ZOOM_HALF_PX, 0, max(width - size, 0)))
    y0 = int(np.clip(round(center[1]) - ZOOM_HALF_PX, 0, max(height - size, 0)))
    crop = image[y0:y0 + size, x0:x0 + size]
    return cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST), np.array([x0, y0], dtype=float)


def pick_exact(image, rough, name, mouse):
    view, origin = zoom_crop(image, rough)
    banner = f"{name}: click the OUTER corner, where the black border meets the background | u back"
    while True:
        action, value = wait_for_input(view, banner, mouse)
        if action == "click":
            return origin + to_original(np.array(value, dtype=float), ZOOM)
        if value == KEY_UNDO:
            return None


def draw_points(overview, corners, scale):
    view = overview.copy()
    for name, corner in zip(CORNER_NAMES, corners):
        point = tuple(int(v) for v in np.round((corner + 0.5) * scale - 0.5))
        cv2.circle(view, point, 5, (0, 0, 255), 2)
        cv2.putText(view, name, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return view


def pick_corners(image, mouse):
    scale = DISPLAY_LONG_SIDE / max(image.shape[:2])
    overview = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    corners = []
    while len(corners) < len(CORNER_NAMES):
        name = CORNER_NAMES[len(corners)]
        banner = f"{name} of the puzzle: click near it | u undo, q quit"
        action, value = wait_for_input(draw_points(overview, corners, scale), banner, mouse)
        if action == "click":
            exact = pick_exact(image, to_original(np.array(value, dtype=float), scale), name, mouse)
            if exact is not None:
                corners.append(exact)
        elif value == KEY_UNDO and corners:
            corners.pop()
        elif value == KEY_QUIT:
            return None
    return np.array(corners)


def sort_clockwise(corners):
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    order = np.argsort(angles)
    first_click = np.where(order == 0)[0][0]
    return corners[np.roll(order, -first_click)]


def build_preview(image, corners):
    matrix, _ = fit_h4(corners, target_corners())
    preview = warp_perspective(image, matrix)
    step = RECT_SIZE // CELLS
    for i in range(CELLS + 1):
        line = min(i * step, RECT_SIZE - 1)
        cv2.line(preview, (line, 0), (line, RECT_SIZE - 1), (0, 200, 0), 1)
        cv2.line(preview, (0, line), (RECT_SIZE - 1, line), (0, 200, 0), 1)
    return cv2.resize(preview, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)


def annotate_corners(image, mouse):
    while True:
        corners = pick_corners(image, mouse)
        if corners is None:
            return None, None
        corners = sort_clockwise(corners)
        preview = build_preview(image, corners)
        banner = "digits upright, lines on the grid? Enter accept | o rotate | r redo | q quit"
        while True:
            _, key = wait_for_input(preview, banner, mouse, crosshair=False)
            if key in KEY_ACCEPT:
                return corners, preview
            if key == KEY_QUIT:
                return None, None
            if key == KEY_ROTATE:
                corners = np.roll(corners, -1, axis=0)
                preview = build_preview(image, corners)
            if key == KEY_REDO:
                break


def parse_row(text):
    row = text.replace(" ", "")
    if len(row) == CELLS and row.isdigit():
        return [int(digit) for digit in row]
    return None


def print_grid(grid):
    for row in grid:
        print(" ".join(str(value) for value in row))


def read_grid(previous):
    if previous is not None:
        print("previous grid:")
        print_grid(previous)
        print("press Enter on row 1 to reuse it")
    while True:
        grid = []
        while len(grid) < CELLS:
            text = input(f"row {len(grid) + 1} (9 digits, 0 = empty): ")
            if not text.strip() and not grid and previous is not None:
                grid = [list(row) for row in previous]
                break
            row = parse_row(text)
            if row is None:
                print("need exactly 9 digits")
            else:
                grid.append(row)
        print_grid(grid)
        if input("correct? [y/n]: ").strip().lower() == "y":
            return grid


def load_conditions(folder):
    conditions = {}
    readme = folder / "README.md"
    if not readme.exists():
        return conditions
    for line in readme.read_text().splitlines():
        cells = [cell.strip(" `") for cell in line.split("|")]
        if len(cells) > 4:
            conditions[cells[2]] = re.findall(r"C[1-5]", cells[4])
    return conditions


def previous_grid(outdir, name):
    earlier = sorted(path for path in outdir.glob("*.json") if path.stem < name)
    if not earlier:
        return None
    return load_annotation(earlier[-1])[1]


def save_annotation(path, image_name, conditions, corners, grid):
    path.parent.mkdir(parents=True, exist_ok=True)
    points = [[round(float(x), 2), round(float(y), 2)] for x, y in corners]
    rows = ",\n".join(f"    {json.dumps(row)}" for row in grid)
    text = (
        "{\n"
        f'  "image": {json.dumps(image_name)},\n'
        f'  "conditions": {json.dumps(conditions)},\n'
        f'  "corners": {json.dumps(points)},\n'
        f'  "grid": [\n{rows}\n  ]\n'
        "}\n"
    )
    with open(path, "w") as f:
        f.write(text)


def list_images(folder, only):
    images = sorted(path for path in folder.glob("*.jpg"))
    if only:
        return [path for path in images if path.stem == only]
    return images


def parse_args():
    parser = argparse.ArgumentParser(description="click puzzle corners and type the grid to write annotation JSON files")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="image folder (default: %(default)s)")
    parser.add_argument("--outdir", type=Path, help="annotation folder (default: <folder>/annotations)")
    parser.add_argument("--only", help="annotate one image by base name, overwriting its file")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir or args.folder / "annotations"
    conditions = load_conditions(args.folder)
    mouse = Mouse()
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, mouse)

    for image_path in list_images(args.folder, args.only):
        out_path = outdir / f"{image_path.stem}.json"
        if out_path.exists() and not args.only:
            continue
        print(image_path.name, conditions.get(image_path.name, []))
        image = load_image(image_path)
        corners, preview = annotate_corners(image, mouse)
        if corners is None:
            break
        show(preview, "type the grid in the terminal")
        cv2.waitKey(1)
        grid = read_grid(previous_grid(outdir, image_path.stem))
        save_annotation(out_path, image_path.name, conditions.get(image_path.name, []), corners, grid)
        print("saved", out_path)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
