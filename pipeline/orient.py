from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pipeline.cells import split_cells, warp_for_cells
from pipeline.ocr import OCR_WORKERS, digit_confidence, prepare

ORIENT_SAMPLES = 12
ORIENT_MARGIN = 10.0


def orientation_score(cells):
    filled = [cell for row in cells for cell in row if cell.digit is not None][:ORIENT_SAMPLES]
    if not filled:
        return 0.0
    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        return float(np.mean(list(pool.map(lambda cell: digit_confidence(prepare(cell)), filled))))


def orient(image, corners):
    options = []
    for turn in range(4):
        rolled = np.roll(corners, -turn, axis=0)
        cells = split_cells(warp_for_cells(image, rolled))
        options.append((orientation_score(cells), rolled, cells))
    best = max(range(4), key=lambda turn: options[turn][0])
    if options[best][0] < options[0][0] + ORIENT_MARGIN:
        best = 0
    _, corners, cells = options[best]
    return corners, cells, best
