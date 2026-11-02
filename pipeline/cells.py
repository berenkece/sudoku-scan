import numpy as np


def split_cells(rectified: np.ndarray, margin_ratio: float = 0.08) -> list[np.ndarray]:
    """Split a square rectified grid into 81 cell images."""
    cells = []
    height, width = rectified.shape[:2]
    cell_height, cell_width = height / 9, width / 9
    for row in range(9):
        for col in range(9):
            y1 = int((row + margin_ratio) * cell_height)
            y2 = int((row + 1 - margin_ratio) * cell_height)
            x1 = int((col + margin_ratio) * cell_width)
            x2 = int((col + 1 - margin_ratio) * cell_width)
            cells.append(rectified[y1:y2, x1:x2])
    return cells
