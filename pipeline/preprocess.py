import cv2
import numpy as np


def preprocess(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale and reduce noise while preserving Sudoku edges."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(image, (5, 5), 0)
