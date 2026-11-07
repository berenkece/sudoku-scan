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


def cell_accuracy(pred, truth):
    return float(np.mean(np.asarray(pred) == np.asarray(truth)))


def confusion_matrix(preds, truths, classes=10):
    matrix = np.zeros((classes, classes), dtype=int)
    for pred, truth in zip(preds, truths):
        np.add.at(matrix, (np.asarray(truth).ravel(), np.asarray(pred).ravel()), 1)
    return matrix


def f1_score(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    true_pos = np.sum(pred & truth)
    if true_pos == 0:
        return 0.0
    precision = true_pos / np.sum(pred)
    recall = true_pos / np.sum(truth)
    return float(2 * precision * recall / (precision + recall))

"""test:
python -c "
from pipeline.metrics import corner_error
gt = [[0, 0], [100, 0], [100, 100], [0, 100]]
pred = [[3, 4], [103, 4], [103, 104], [3, 104]]
print(corner_error(pred, gt))  # 5.0
print(corner_error(gt, gt))    # 0.0
"
"""
