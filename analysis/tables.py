import csv
from pathlib import Path

import numpy as np

EVAL_DIR = Path("out/eval")
OUT_PATH = Path("report/tables.md")
SETS = ("dev", "personal")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def summary_tables():
    overall, by_condition = [], []
    for name in SETS:
        for row in read_csv(EVAL_DIR / name / "summary.csv"):
            values = [row["images"], f"{row['found']}/{row['images']}", row["crashed"], fmt(row["corner_err_pct"], 3),
                      fmt(row["cell_acc"]), fmt(row["f1_empty"]), f"{row['exact']}/{row['found']}", fmt(row["runtime_mean_s"], 2)]
            target = overall if row["group"] == "all" else by_condition
            target.append([name] + ([] if row["group"] == "all" else [row["group"]]) + values)
    header = ["images", "found", "crashed", "corner err (%)", "cell acc", "empty F1", "exact", "runtime (s)"]
    return table(["set"] + header, overall), table(["set", "condition"] + header, by_condition)


def ocr_table():
    rows = []
    for row in read_csv(EVAL_DIR / "ocr_compare.csv"):
        rows.append([row["set"], row["mode"], row["psm"], fmt(row["cell_acc"]), fmt(row["digit_acc"]),
                     f"{row['exact']}/{row['images']}", row["unread"], fmt(row["f1_empty"])])
    return table(["set", "mode", "psm", "cell acc", "digit acc", "exact", "unread", "empty F1"], rows)


def confusion_table():
    matrix = sum(np.loadtxt(EVAL_DIR / name / "confusion.csv", delimiter=",", dtype=int) for name in SETS)
    pairs = [(matrix[t, p], t, p) for t in range(10) for p in range(10) if t != p and matrix[t, p]]
    rows = [[t, p if p else "0 (empty)", count] for count, t, p in sorted(pairs, reverse=True)]
    if not rows:
        return "No misread cells."
    return table(["true", "predicted", "count"], rows)


def harris_table():
    rows = read_csv(EVAL_DIR / "harris_params.csv")
    out = []
    for param in dict.fromkeys(r["param"] for r in rows):
        for value in dict.fromkeys(r["value"] for r in rows if r["param"] == param):
            chosen = [r for r in rows if r["param"] == param and r["value"] == value]
            found = [r for r in chosen if r["found"] == "True"]
            out.append([param, value + (" (chosen)" if chosen[0]["default"] == "True" else ""),
                        f"{np.mean([float(r['candidates']) for r in chosen]):.0f}", f"{len(chosen) - len(found)}",
                        fmt(np.mean([float(r["harris_err_pct"]) for r in found]), 3),
                        fmt(np.mean([float(r["corner_err_pct"]) for r in found]), 3),
                        fmt(np.max([float(r["corner_err_pct"]) for r in found]), 3)])
    block = read_csv(EVAL_DIR / "harris_block_all.csv")
    block_rows = []
    for value in dict.fromkeys(r["value"] for r in block):
        errors = [float(r["corner_err_pct"]) for r in block if r["value"] == value]
        block_rows.append([value, fmt(np.mean(errors), 3), fmt(np.max(errors), 3), sum(e > 0.4 for e in errors)])
    return (table(["parameter", "value", "candidates", "not found", "Harris peak err (%)", "final err (%)", "worst final (%)"], out),
            table(["blockSize ratio", "mean err (%)", "worst err (%)", "images > 0.4 %"], block_rows))


def stress_table():
    rows = read_csv(EVAL_DIR / "stress_test.csv")
    out = []
    for name in dict.fromkeys(r["perturbation"] for r in rows):
        chosen = [r for r in rows if r["perturbation"] == name]
        found = [r for r in chosen if r["found"] == "True"]
        errors = [float(r["corner_err_pct"]) for r in found]
        out.append([name, f"{len(found)}/{len(chosen)}", sum(r["crashed"] == "True" for r in chosen),
                    fmt(np.mean(errors), 3) if errors else "-", fmt(max(errors), 3) if errors else "-",
                    fmt(np.mean([float(r["cell_acc"]) for r in found]), 4) if found else "-",
                    f"{sum(r['exact'] == 'True' for r in chosen)}/{len(chosen)}"])
    return table(["perturbation", "found", "crashed", "corner err mean (%)", "corner err max (%)", "cell acc", "exact"], out)


def main():
    overall, by_condition = summary_tables()
    text = "\n\n".join([
        "## Section 7 metrics",
        overall,
        "Cell accuracy, empty F1 and corner error are means over images where the puzzle was found. "
        "Empty F1 treats an empty cell as the positive class.",
        "## By condition",
        by_condition,
        "An image with several conditions is counted in each of its rows.",
        "## OCR configurations",
        ocr_table(),
        "pipeline: psm 10 and 8 at 24, 32 and 40 px digit height (six reads); the most confident reading that passes "
        "the digit shape rules (pipeline/digit_rules.py) is kept. "
        "single: one call per filled cell, no fallback. raw: the margin-cropped gray cell without preparation. "
        "unread: filled cells for which OCR returned no digit (written as 0).",
        "## Confusions (dev + personal, pipeline)",
        confusion_table(),
        "## Robustness: perturbed copies of all 32 annotated images",
        stress_table() if (EVAL_DIR / "stress_test.csv").exists() else "Not run.",
        "rot90/180/270/45: in-plane rotation of the whole photo; scale50/25: downscaled; blur: Gaussian, sigma 2 px at "
        "1200 px long side; jpeg20: JPEG quality 20; dark: gamma 2.5; noise: Gaussian, sigma 15 grey levels. "
        "Ground-truth corners are moved with the same transform.",
        "## Harris parameters (8 images, one parameter at a time)",
        harris_table()[0],
        "Images: dev_01, dev_07, dev_12, dev_15, dev_17, dev_20, s042401168_04, s042401168_08. "
        "Harris peak err is measured before cornerSubPix, final err after it.",
        "## blockSize on all 32 annotated images",
        harris_table()[1],
    ])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
