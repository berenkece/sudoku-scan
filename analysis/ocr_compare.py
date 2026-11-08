import csv
import json
from pathlib import Path

import numpy as np

from pipeline.cells import split_cells, warp_for_cells
from pipeline.io_utils import load_image
from pipeline.metrics import cell_accuracy, f1_score
from concurrent.futures import ThreadPoolExecutor

from pipeline.cells import CELLS
from pipeline.ocr import OCR_WORKERS, prepare, read_digit, recognize

SETS = {
    "dev": (Path("data/dev/images"), Path("data/dev/annotations"), Path("out/dev")),
    "personal": (Path("data/personal"), Path("data/personal/annotations"), Path("out/personal")),
}
CONFIGS = [("pipeline", "10+8"), ("single", 10), ("single", 7), ("single", 8), ("single", 13), ("single raw", 10)]
OUT_PATH = Path("out/eval/ocr_compare.csv")


def load_cells(images, annotations, results):
    items = []
    for annotation_path in sorted(annotations.glob("*.json")):
        annotation = json.loads(annotation_path.read_text())
        result = json.loads((results / f"{annotation_path.stem}_result.json").read_text())
        if result["corners"] is None:
            continue
        image = load_image(images / annotation["image"])
        items.append((split_cells(warp_for_cells(image, np.array(result["corners"]))), np.array(annotation["grid"])))
    return items


def single_pass(cells, psm, prepared):
    grid = np.zeros((CELLS, CELLS), dtype=int)
    filled = [(r, c) for r in range(CELLS) for c in range(CELLS) if cells[r][c].digit is not None]
    def read(pos):
        cell = cells[pos[0]][pos[1]]
        return read_digit(prepare(cell) if prepared else cell.gray, psm) or 0
    with ThreadPoolExecutor(OCR_WORKERS) as pool:
        for (r, c), digit in zip(filled, pool.map(read, filled)):
            grid[r, c] = digit
    return grid


def predict(cells, mode, psm):
    if mode == "pipeline":
        return recognize(cells)
    return single_pass(cells, psm, prepared=mode == "single")


def main():
    rows = []
    for name, folders in SETS.items():
        items = load_cells(*folders)
        for mode, psm in CONFIGS:
            preds = [predict(cells, mode, psm) for cells, _ in items]
            truths = [truth for _, truth in items]
            filled = np.array([[[cell.digit is not None for cell in row] for row in cells] for cells, _ in items])
            pred_all, truth_all = np.array(preds), np.array(truths)
            row = {"set": name, "mode": mode, "psm": psm,
                   "cell_acc": np.mean([cell_accuracy(p, t) for p, t in zip(preds, truths)]),
                   "exact": sum(np.array_equal(p, t) for p, t in zip(preds, truths)),
                   "images": len(items),
                   "digit_acc": float(np.mean(pred_all[truth_all > 0] == truth_all[truth_all > 0])),
                   "unread": int(np.sum(filled & (pred_all == 0))),
                   "f1_empty": f1_score(pred_all == 0, truth_all == 0)}
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
