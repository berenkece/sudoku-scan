import cv2
import numpy as np


def rectify(image: np.ndarray, corners: np.ndarray, size: int = 900) -> np.ndarray:
    """Perspective-transform the detected grid to a square image."""
    if corners is None or len(corners) != 4:
        raise ValueError("Exactly four grid corners are required")
    destination = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (size, size))
