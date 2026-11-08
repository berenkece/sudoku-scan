import cv2
import numpy as np

from pipeline.cells import CELLS, filled_mask, split_cells, warp_for_cells
from pipeline.preprocess import normalize_illumination, to_gray
from pipeline.rectify import fit_h4, target_corners, warp_perspective

LINE_KERNEL_RATIO = 0.4
LINE_BAND_RATIO = 0.2
INK_LEVEL = 10
LINE_COVERAGE = 0.7
MIN_GRID_LINES = 11
MIN_GIVENS = 17


def line_masks(rectified, valid=None):
    darkness = 255 - normalize_illumination(to_gray(rectified))
    if valid is not None:
        darkness[valid == 0] = 0
    length = round(LINE_KERNEL_RATIO * rectified.shape[0] / CELLS)
    horizontal = cv2.morphologyEx(darkness, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
    vertical = cv2.morphologyEx(darkness, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
    return horizontal, vertical


def grid_line_count(rectified):
    horizontal, vertical = line_masks(rectified)
    cell = rectified.shape[0] / CELLS
    band = round(LINE_BAND_RATIO * cell)
    count = 0
    for k in range(CELLS + 1):
        center = round(k * cell)
        lo, hi = max(center - band, 0), center + band + 1
        count += (horizontal[lo:hi, :] > INK_LEVEL).any(axis=0).mean() >= LINE_COVERAGE
        count += (vertical[:, lo:hi] > INK_LEVEL).any(axis=1).mean() >= LINE_COVERAGE
    return int(count)


def is_sudoku_grid(image, corners):
    matrix, _ = fit_h4(corners, target_corners())
    return grid_line_count(warp_perspective(image, matrix)) >= MIN_GRID_LINES


def has_enough_digits(image, corners):
    counts = [int(filled_mask(split_cells(warp_for_cells(image, np.roll(corners, -turn, axis=0)))).sum()) for turn in (0, 1)]
    return max(counts) >= MIN_GIVENS
