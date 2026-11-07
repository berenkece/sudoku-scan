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
