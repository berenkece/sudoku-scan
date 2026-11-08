from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract

from pipeline.cells import CELLS
from pipeline.digit_rules import choose_digit, digit_features

DEFAULT_PSM = 10
READ_PSMS = (10, 8)
RAW_PSMS = (10, 8, 7)
WHITELIST = "123456789"
DIGIT_HEIGHTS_PX = (24, 32, 40)
PAD_RATIO = 0.5
OCR_WORKERS = 8


def tesseract_config(psm):
    return f"--psm {psm} -c tessedit_char_whitelist={WHITELIST}"


def prepare(cell, digit_height=DIGIT_HEIGHTS_PX[-1]):
    ys, xs = np.nonzero(cell.digit)
    digit = cell.digit[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = digit_height / digit.shape[0]
    digit = cv2.resize(digit, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    _, digit = cv2.threshold(digit, 127, 255, cv2.THRESH_BINARY)
    height, width = digit.shape
    pad = round(digit_height * PAD_RATIO)
    size = max(height, width) + 2 * pad
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - height) // 2, (size - width) // 2
    canvas[y0:y0 + height, x0:x0 + width] = digit
    return 255 - canvas


def read_digit(image, psm):
    text = pytesseract.image_to_string(image, config=tesseract_config(psm))
    digits = [char for char in text if char in WHITELIST]
    if len(digits) != 1:
        return None
    return int(digits[0])


def digit_confidence(image, psm=DEFAULT_PSM):
    data = pytesseract.image_to_data(image, config=tesseract_config(psm), output_type=pytesseract.Output.DICT)
    confidences = [float(conf) for conf, text in zip(data["conf"], data["text"]) if text.strip()]
    return max(confidences) if confidences else 0.0


def read_candidate(image, psm):
    data = pytesseract.image_to_data(image, config=tesseract_config(psm), output_type=pytesseract.Output.DICT)
    words = [(text.strip(), float(conf)) for text, conf in zip(data["text"], data["conf"]) if text.strip()]
    if not words or len(words[0][0]) != 1:
        return None
    return int(words[0][0]), words[0][1]


def recognize_cell(cell, psms=READ_PSMS, prepared=True):
    if not prepared:
        for psm in psms:
            digit = read_digit(cell.gray, psm)
            if digit is not None:
                return digit
        return 0
    candidates = []
    for height in DIGIT_HEIGHTS_PX:
        image = prepare(cell, height)
        for psm in psms:
            candidate = read_candidate(image, psm)
            if candidate is not None:
                candidates.append(candidate)
    return choose_digit(candidates, digit_features(cell.digit))


def recognize(cells, psms=READ_PSMS, prepared=True):
    grid = np.zeros((CELLS, CELLS), dtype=int)
    filled = [(row, col) for row in range(CELLS) for col in range(CELLS) if cells[row][col].digit is not None]
    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        digits = pool.map(lambda pos: recognize_cell(cells[pos[0]][pos[1]], psms, prepared), filled)
        for (row, col), digit in zip(filled, digits):
            grid[row, col] = digit
    return grid
