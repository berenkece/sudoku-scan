import numpy as np

def corner_error(pred, gt):
    pred = np.array(pred, dtype=float)
    gt = np.array(gt, dtype=float)

    distances = np.linalg.norm(pred - gt, axis=1)
    mean_distance = np.mean(distances)

    next_corners = np.roll(gt, -1, axis=0)

    edge_lengths = np.linalg.norm(gt - next_corners, axis=1)
    mean_edge_lengths = np.mean(edge_lengths)

    return mean_distance / mean_edge_lengths * 100

"""test:
python -c "
from pipeline.metrics import corner_error
gt = [[0, 0], [100, 0], [100, 100], [0, 100]]
pred = [[3, 4], [103, 4], [103, 104], [3, 104]]
print(corner_error(pred, gt))  # 5.0
print(corner_error(gt, gt))    # 0.0
"
"""
