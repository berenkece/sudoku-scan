from typing import NamedTuple, Optional

import cv2
import numpy as np

MIN_HOLE_AREA = 0.01
LEFT_UPPER = (0.2, 0.45, 0.3)
LEFT_MIDDLE = (0.45, 0.6, 0.25)


class DigitFeatures(NamedTuple):
    holes: int
    hole_y: Optional[float]
    left_upper: float
    left_middle: float


def left_ink(digit, band):
    top, bottom, right = band
    height, width = digit.shape
    y0, y1 = int(top * height), max(int(bottom * height), int(top * height) + 1)
    return float(digit[y0:y1, :max(int(right * width), 1)].any(axis=1).mean())


def digit_features(mask):
    ys, xs = np.nonzero(mask)
    digit = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0
    height, width = digit.shape
    contours, hierarchy = cv2.findContours(digit.astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hole_ys = []
    if hierarchy is not None:
        for contour, links in zip(contours, hierarchy[0]):
            if links[3] != -1 and cv2.contourArea(contour) > MIN_HOLE_AREA * height * width:
                _, y, _, h = cv2.boundingRect(contour)
                hole_ys.append((y + h / 2) / height)
    hole_y = float(np.mean(hole_ys)) if hole_ys else None
    return DigitFeatures(len(hole_ys), hole_y, left_ink(digit, LEFT_UPPER), left_ink(digit, LEFT_MIDDLE))


def allowed(digit, f):
    if digit in (1, 2, 7):
        return f.holes == 0 and f.left_middle <= 0.2
    if digit == 3:
        return f.holes == 0 and f.left_middle <= 0.3 and f.left_upper <= 0.8
    if digit == 4:
        return f.holes <= 1 and f.left_middle >= 0.5
    if digit == 5:
        return f.holes == 0 and f.left_upper >= 0.9
    if digit == 6:
        return f.holes == 0 or (f.holes == 1 and f.hole_y >= 0.55)
    if digit == 8:
        return f.holes == 2
    if digit == 9:
        return f.holes == 0 or (f.holes == 1 and f.hole_y <= 0.4)
    return False


def choose_digit(candidates, features):
    ranked = sorted(candidates, key=lambda candidate: -candidate[1])
    for digit, _ in ranked:
        if allowed(digit, features):
            return digit
    return ranked[0][0] if ranked else 0
