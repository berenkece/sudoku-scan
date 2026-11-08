from typing import NamedTuple, Optional

import cv2
import numpy as np

from pipeline.preprocess import normalize_illumination, to_gray
from pipeline.rectify import RECT_SIZE, fit_h4, target_corners, warp_perspective

CELLS = 9
CELL_IMAGE_SIZE = 2 * RECT_SIZE
MARGIN_RATIO = 0.08
MIN_DIGIT_HEIGHT = 0.3
MIN_DIGIT_AREA = 0.02
MIN_PART_AREA = 0.005
CENTER_BAND = 0.15
GROW_RATIO = 0.15
LINE_LENGTH_RATIO = 1.0
MAX_PART_GAP = 0.06


class Cell(NamedTuple):
    gray: np.ndarray
    binary: np.ndarray
    digit: Optional[np.ndarray]


def warp_for_cells(image, corners, size=CELL_IMAGE_SIZE):
    matrix, _ = fit_h4(corners, target_corners(size, size))
    return warp_perspective(image, matrix, size, size)


def binarize(gray):
    normalized = normalize_illumination(gray)
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def cell_bounds(shape, row, col, margin_ratio):
    height, width = shape[:2]
    cell_h, cell_w = height / CELLS, width / CELLS
    margin_y, margin_x = round(cell_h * margin_ratio), round(cell_w * margin_ratio)
    y0, y1 = round(row * cell_h) + margin_y, round((row + 1) * cell_h) - margin_y
    x0, x1 = round(col * cell_w) + margin_x, round((col + 1) * cell_w) - margin_x
    return y0, y1, x0, x1


def remove_lines(binary, cell_size):
    length = round(LINE_LENGTH_RATIO * cell_size)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    lines = cv2.dilate(cv2.max(horizontal, vertical), np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def box_gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return float(np.hypot(dx, dy))


def find_digit(ink, cell_box, cell_h, cell_w):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = ink.shape
    top, bottom, left, right = cell_box
    band_y, band_x = CENTER_BAND * (bottom - top), CENTER_BAND * (right - left)
    parts = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < MIN_PART_AREA * cell_h * cell_w or w == width or h == height:
            continue
        cx, cy = centroids[label]
        if left + band_x <= cx <= right - band_x and top + band_y <= cy <= bottom - band_y:
            parts.append((area, label, (x, y, w, h)))
    if not parts:
        return None
    _, main_label, main_box = max(parts)
    max_gap = MAX_PART_GAP * min(cell_h, cell_w)
    digit = np.zeros_like(ink)
    for _, label, box in parts:
        if label == main_label or box_gap(main_box, box) <= max_gap:
            digit[labels == label] = 255
    ys, xs = np.nonzero(digit)
    if ys.max() - ys.min() + 1 < MIN_DIGIT_HEIGHT * cell_h or len(ys) < MIN_DIGIT_AREA * cell_h * cell_w:
        return None
    return digit


def split_cells(rectified, margin_ratio=MARGIN_RATIO, grow_ratio=GROW_RATIO):
    gray = to_gray(rectified)
    height, width = gray.shape
    cell_h, cell_w = height / CELLS, width / CELLS
    ink = remove_lines(binarize(gray), min(cell_h, cell_w))
    grid = []
    for row in range(CELLS):
        cells = []
        for col in range(CELLS):
            y0, y1, x0, x1 = cell_bounds(gray.shape, row, col, margin_ratio)
            wy0, wy1, wx0, wx1 = cell_bounds(gray.shape, row, col, -grow_ratio)
            wy0, wx0, wy1, wx1 = max(wy0, 0), max(wx0, 0), min(wy1, height), min(wx1, width)
            cy0, cy1, cx0, cx1 = cell_bounds(gray.shape, row, col, 0.0)
            window = ink[wy0:wy1, wx0:wx1]
            cell_box = (cy0 - wy0, cy1 - wy0, cx0 - wx0, cx1 - wx0)
            cells.append(Cell(gray[y0:y1, x0:x1], window, find_digit(window, cell_box, cell_h, cell_w)))
        grid.append(cells)
    return grid


def filled_mask(cells):
    return np.array([[cell.digit is not None for cell in row] for row in cells])
