from typing import NamedTuple

import cv2
import numpy as np

RECT_SIZE = 450
A3_CORNERS = (0, 1, 3)
A4_RANSAC_THRESH = 1e6


class Rectification(NamedTuple):
    a3: np.ndarray
    a4: np.ndarray
    h4: np.ndarray
    a3_corner_err_px: float
    a4_rms_px: float
    h4_max_err_px: float


def target_corners(width=RECT_SIZE, height=RECT_SIZE):
    return np.array([
        [-0.5, -0.5],
        [width - 0.5, -0.5],
        [width - 0.5, height - 0.5],
        [-0.5, height - 0.5],
    ], dtype=float)


def apply_affine(matrix, points):
    return points @ matrix[:, :2].T + matrix[:, 2]


def fit_a3(corners, target, used=A3_CORNERS):
    used = list(used)
    left_out = ({0, 1, 2, 3} - set(used)).pop()
    matrix = cv2.getAffineTransform(corners[used].astype(np.float32), target[used].astype(np.float32))
    error = np.linalg.norm(apply_affine(matrix, corners[left_out]) - target[left_out])
    return matrix, float(error)


def fit_a4(corners, target):
    matrix, inliers = cv2.estimateAffine2D(corners, target, method=cv2.RANSAC, ransacReprojThreshold=A4_RANSAC_THRESH)
    residuals = apply_affine(matrix, corners) - target
    rms = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
    return matrix, float(rms), inliers


def fit_h4(corners, target):
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target.astype(np.float32))
    start = corners.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(start, matrix).reshape(-1, 2)
    max_error = np.max(np.linalg.norm(mapped - target, axis=1))
    return matrix, float(max_error)


def fit_models(corners, width=RECT_SIZE, height=RECT_SIZE):
    target = target_corners(width, height)
    a3, a3_error = fit_a3(corners, target)
    a4, a4_rms, _ = fit_a4(corners, target)
    h4, h4_error = fit_h4(corners, target)
    return Rectification(a3, a4, h4, a3_error, a4_rms, h4_error)


def warp_affine(image, matrix, width=RECT_SIZE, height=RECT_SIZE):
    return cv2.warpAffine(image, matrix, (width, height))


def warp_perspective(image, matrix, width=RECT_SIZE, height=RECT_SIZE):
    return cv2.warpPerspective(image, matrix, (width, height))
