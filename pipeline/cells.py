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


def find_digit(binary, cell_h, cell_w):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    digit = np.zeros_like(binary)
    for label in range(1, count):
        _, _, w, h, area = stats[label]
        if area < MIN_PART_AREA * cell_h * cell_w or w == width or h == height:
            continue
        cx, cy = centroids[label]
        if CENTER_BAND * width <= cx <= (1 - CENTER_BAND) * width and CENTER_BAND * height <= cy <= (1 - CENTER_BAND) * height:
            digit[labels == label] = 255
    ys, xs = np.nonzero(digit)
    if len(ys) == 0:
        return None
    if ys.max() - ys.min() + 1 < MIN_DIGIT_HEIGHT * cell_h or len(ys) < MIN_DIGIT_AREA * cell_h * cell_w:
        return None
    return digit


def split_cells(rectified, margin_ratio=MARGIN_RATIO):
    gray = to_gray(rectified)
    binary = binarize(gray)
    cell_h, cell_w = gray.shape[0] / CELLS, gray.shape[1] / CELLS
    grid = []
    for row in range(CELLS):
        cells = []
        for col in range(CELLS):
            y0, y1, x0, x1 = cell_bounds(gray.shape, row, col, margin_ratio)
            cell_binary = binary[y0:y1, x0:x1]
            cells.append(Cell(gray[y0:y1, x0:x1], cell_binary, find_digit(cell_binary, cell_h, cell_w)))
        grid.append(cells)
    return grid


def filled_mask(cells):
    return np.array([[cell.digit is not None for cell in row] for row in cells])
