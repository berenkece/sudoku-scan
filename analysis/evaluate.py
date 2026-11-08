import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.cells import split_cells, warp_for_cells
from pipeline.io_utils import load_image, save_image
from pipeline.metrics import cell_accuracy, confusion_matrix, corner_error, f1_score
from pipeline.ocr import prepare

FIELDS = ["image", "conditions", "found", "crashed", "corner_err_pct", "cell_acc", "f1_empty", "exact", "verdict", "runtime_s"]
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]


def parse_args():
    parser = argparse.ArgumentParser(description="score sudoku_scan result files against annotations")
    parser.add_argument("--results", type=Path, required=True, help="folder with <name>_result.json files")
    parser.add_argument("--annotations", type=Path, required=True, help="folder with <name>.json annotations")
    parser.add_argument("--images", type=Path, required=True, help="folder with the input images, for misread crops")
    parser.add_argument("--out", type=Path, required=True, help="folder for CSV files and misread crops")
    return parser.parse_args()


def save_misread(folder, image, corners, pred, truth, stem):
    cells = split_cells(warp_for_cells(image, np.array(corners)))
    for row, col in np.argwhere(pred != truth):
        cell = cells[row][col]
        gray = cv2.resize(cell.gray, (100, 100))
        crop = gray if cell.digit is None else np.hstack((gray, cv2.resize(prepare(cell), (100, 100))))
        save_image(folder / f"{stem}_r{row}c{col}_true{truth[row, col]}_pred{pred[row, col]}.png", crop)


def evaluate_image(annotation_path, args, crops):
    annotation = json.loads(annotation_path.read_text())
    stem = annotation_path.stem
    result_path = args.results / f"{stem}_result.json"
    crashed = (args.results / f"{stem}_error.txt").exists() or not result_path.exists()
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    row = {"image": annotation["image"], "conditions": " ".join(annotation["conditions"]),
           "found": result.get("corners") is not None, "crashed": crashed,
           "verdict": result.get("verdict"), "runtime_s": result.get("runtime_s")}
    if not row["found"]:
        return row, None, None
    truth = np.array(annotation["grid"])
    pred = np.array(result["grid"])
    row["corner_err_pct"] = corner_error(result["corners"], annotation["corners"])
    row["cell_acc"] = cell_accuracy(pred, truth)
    row["f1_empty"] = f1_score(pred == 0, truth == 0)
    row["exact"] = bool(np.array_equal(pred, truth))
    if not row["exact"]:
        save_misread(crops, load_image(args.images / annotation["image"]), result["corners"], pred, truth, stem)
    return row, pred, truth


def summarize(rows, label):
    found = [r for r in rows if r["found"]]
    line = {"group": label, "images": len(rows), "found": len(found), "crashed": sum(r["crashed"] for r in rows)}
    if found:
        line["corner_err_pct"] = np.mean([r["corner_err_pct"] for r in found])
        line["cell_acc"] = np.mean([r["cell_acc"] for r in found])
        line["f1_empty"] = np.mean([r["f1_empty"] for r in found])
        line["exact"] = sum(r["exact"] for r in found)
    line["runtime_mean_s"] = np.mean([r["runtime_s"] for r in rows if r["runtime_s"] is not None])
    return line


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    crops = args.out / "misread"
    rows, preds, truths = [], [], []
    for annotation_path in sorted(args.annotations.glob("*.json")):
        row, pred, truth = evaluate_image(annotation_path, args, crops)
        rows.append(row)
        if pred is not None:
            preds.append(pred)
            truths.append(truth)

    summary = [summarize(rows, "all")]
    summary += [summarize([r for r in rows if c in r["conditions"].split()], c) for c in CONDITIONS]
    summary = [s for s in summary if s["images"]]
    matrix = confusion_matrix(preds, truths)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per_image.csv", rows, FIELDS)
    write_csv(args.out / "summary.csv", summary, ["group", "images", "found", "crashed", "corner_err_pct", "cell_acc", "f1_empty", "exact", "runtime_mean_s"])
    np.savetxt(args.out / "confusion.csv", matrix, fmt="%d", delimiter=",")

    for s in summary:
        values = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in s.items()}
        print(values)
    print("confusion (row = true, col = predicted, 0 = empty):")
    print(matrix)
    off = [(int(t), int(p), int(matrix[t, p])) for t, p in zip(*np.nonzero(matrix)) if t != p]
    print("confusions (true, predicted, count):", sorted(off, key=lambda x: -x[2]))


if __name__ == "__main__":
    main()
