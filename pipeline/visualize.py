import cv2
import numpy as np

CORNER_NAMES = ["TL", "TR", "BR", "BL"]


def draw_harris(gray, response, candidates, corners=None):

    positive = np.clip(response, 0, None)
    compressed = np.sqrt(positive)
    norm = cv2.normalize(compressed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis = cv2.addWeighted(base, 0.5, heat, 0.5, 0)

    for x, y in candidates:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), 1)

    cv2.putText(vis, f"candidates: {len(candidates)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if corners is not None:
        for name, (x, y) in zip(CORNER_NAMES, corners):
            center = (int(round(x)), int(round(y)))
            cv2.circle(vis, center, 8, (0, 0, 255), 2)
            cv2.putText(vis, name, (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(vis, name, (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def draw_debug(rectified, grid, filled, conflicts):
    vis = rectified.copy()
    height, width = vis.shape[:2]
    cells = len(grid)
    cell_h, cell_w = height / cells, width / cells
    for i in range(cells + 1):
        y, x = min(round(i * cell_h), height - 1), min(round(i * cell_w), width - 1)
        cv2.line(vis, (0, y), (width - 1, y), (0, 200, 0), 1)
        cv2.line(vis, (x, 0), (x, height - 1), (0, 200, 0), 1)
    for row in range(cells):
        for col in range(cells):
            x0, y0 = round(col * cell_w), round(row * cell_h)
            if grid[row][col]:
                cv2.putText(vis, str(grid[row][col]), (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            elif filled[row][col]:
                cv2.putText(vis, "?", (x0 + 3, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 2)
            else:
                cv2.circle(vis, (x0 + 6, y0 + 6), 2, (160, 160, 160), -1)
    for row, col in conflicts:
        x0, y0 = round(col * cell_w), round(row * cell_h)
        cv2.rectangle(vis, (x0 + 1, y0 + 1), (round((col + 1) * cell_w) - 2, round((row + 1) * cell_h) - 2), (0, 0, 255), 2)
    return vis
