import cv2
import numpy as np


def draw_harris(gray, response, candidates):

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
    return vis
