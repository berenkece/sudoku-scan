import csv
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.io_utils import load_image
from pipeline.rectify import fit_a3, fit_a4, fit_h4, target_corners, warp_affine, warp_perspective

EVAL_DIR = Path("out/eval")
FIG_DIR = Path("report/figs")
WIDTH_IN = 3.5
DPI = 300
MODELS = {"A3": ("#2a78d6", "o"), "A4": ("#eb6834", "s"), "H4": ("#1baf7a", "^")}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
FOCAL_FAILURES = {"dev_14.jpg"}


def load_rows():
    rows = []
    for name in ("dev", "personal"):
        with open(EVAL_DIR / f"compare_{name}.csv") as f:
            for row in csv.DictReader(f):
                row["set"] = name
                rows.append(row)
    return rows


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=7, labelcolor=INK_SECONDARY)
    ax.set_xlabel(xlabel, fontsize=8, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_SECONDARY)


def scatter(ax, rows, models, column):
    for model in models:
        color, marker = MODELS[model]
        for subset, filled in (("dev", False), ("personal", True)):
            points = [r for r in rows if r["model"] == model and r["set"] == subset and r["image"] not in FOCAL_FAILURES]
            x = [float(r["tilt_deg"]) for r in points]
            y = [float(r[column]) for r in points]
            ax.scatter(x, y, s=16, marker=marker, linewidths=0.9,
                       facecolors=color if filled else "none", edgecolors=color,
                       label=f"{model} ({'personal' if filled else 'dev'})")


def label_personal(ax, rows, model, column, min_tilt=15):
    for r in rows:
        if r["set"] == "personal" and r["model"] == model and float(r["tilt_deg"]) >= min_tilt:
            ax.annotate(r["image"][-6:-4], (float(r["tilt_deg"]), float(r[column])), xytext=(3, 2),
                        textcoords="offset points", fontsize=6, color=INK_SECONDARY)


def finish(fig, ax, path, legend_loc="upper left"):
    ax.legend(fontsize=6, frameon=False, loc=legend_loc, labelcolor=INK_SECONDARY, handletextpad=0.3)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("saved", path)


def residual_plot(rows):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.6))
    scatter(ax, rows, ("A3", "A4"), "fit_err_px")
    for model in ("A3", "A4"):
        label_personal(ax, rows, model, "fit_err_px")
    style(ax, "Obliqueness: plane tilt from H (deg)", "Residual (px, 450 px target)")
    ax.set_title("A3 fourth-corner error and A4 RMS", fontsize=8, color=INK, loc="left")
    finish(fig, ax, FIG_DIR / "residual_vs_tilt.png")


def grid_plot(rows):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.6))
    scatter(ax, rows, ("A3", "A4", "H4"), "spacing_mean_px")
    style(ax, "Obliqueness: plane tilt from H (deg)", "Mean line-spacing error (px)")
    ax.set_title("Grid regularity after rectification", fontsize=8, color=INK, loc="left")
    finish(fig, ax, FIG_DIR / "grid_vs_tilt.png")


def ocr_plot(rows):
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.6))
    scatter(ax, rows, ("A3", "A4", "H4"), "cell_acc")
    style(ax, "Obliqueness: plane tilt from H (deg)", "Cell accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("OCR accuracy with the same OCR", fontsize=8, color=INK, loc="left")
    finish(fig, ax, FIG_DIR / "ocr_vs_tilt.png", legend_loc="lower left")


def side_by_side(rows):
    personal = [r for r in rows if r["set"] == "personal" and r["model"] == "H4"]
    most = max(personal, key=lambda r: float(r["tilt_deg"]))
    image = load_image(Path("data/personal") / most["image"])
    corners = np.array(json.loads((Path("data/personal/annotations") / most["image"].replace(".jpg", ".json")).read_text())["corners"])
    target = target_corners()
    panels = [("A3", warp_affine(image, fit_a3(corners, target)[0])),
              ("A4", warp_affine(image, fit_a4(corners, target)[0])),
              ("H4", warp_perspective(image, fit_h4(corners, target)[0]))]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_IN * 2, WIDTH_IN * 2 / 3 + 0.3))
    for ax, (name, rectified) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9, color=INK)
        ax.axis("off")
    fig.suptitle(f"{most['image']}, tilt {float(most['tilt_deg']):.1f} deg", fontsize=8, color=INK_SECONDARY)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "models_side_by_side.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("saved", FIG_DIR / "models_side_by_side.png")


def harris_example():
    source = Path("out/personal/s042401168_07_harris.png")
    if source.exists():
        image = cv2.imread(str(source))
        fig, ax = plt.subplots(figsize=(WIDTH_IN, WIDTH_IN * image.shape[0] / image.shape[1]))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(FIG_DIR / "harris_example.png", dpi=DPI)
        plt.close(fig)
        print("saved", FIG_DIR / "harris_example.png")


def confusions_figure():
    crops = sorted((EVAL_DIR / "dev" / "misread").glob("*.png")) + sorted((EVAL_DIR / "personal" / "misread").glob("*.png"))
    if not crops:
        (FIG_DIR / "confusions.png").unlink(missing_ok=True)
        print("no misread cells, confusions.png not written")
        return
    cols = 4
    rows = (len(crops) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH_IN * 2, rows * 0.95))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, path in zip(np.ravel(axes), crops):
        name = path.stem
        true, pred = name.split("_true")[1].split("_pred")
        ax.imshow(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name.split('_r')[0]}: {true} read as {pred if pred != '0' else 'empty'}", fontsize=6, color=INK_SECONDARY)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusions.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("saved", FIG_DIR / "confusions.png")


HARRIS_PANELS = [
    ("HARRIS_BLOCK_RATIO", "blockSize (ratio of long side)"),
    ("HARRIS_KSIZE", "Sobel ksize"),
    ("HARRIS_K", "k"),
    ("HARRIS_PEAK_REL", "peak threshold in window"),
    ("HARRIS_SEARCH_PX", "search window (px)"),
    ("SUBPIX_WIN_PX", "cornerSubPix win (px, 0=off)"),
]


def harris_figure():
    path = EVAL_DIR / "harris_params.csv"
    if not path.exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    with open(EVAL_DIR / "harris_block_all.csv") as f:
        block_all = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH_IN * 2, 3.8))
    harris_color, final_color = MODELS["A3"][0], MODELS["A4"][0]
    for ax, (param, label) in zip(axes.ravel(), HARRIS_PANELS):
        values = sorted({r["value"] for r in rows if r["param"] == param}, key=float)
        harris = [np.mean([float(r["harris_err_pct"]) for r in rows if r["param"] == param and r["value"] == v]) for v in values]
        final = [np.mean([float(r["corner_err_pct"]) for r in rows if r["param"] == param and r["value"] == v]) for v in values]
        x = np.arange(len(values))
        ax.plot(x, harris, color=harris_color, marker="o", markersize=4, linewidth=1.5, label="Harris peak")
        ax.plot(x, final, color=final_color, marker="s", markersize=4, linewidth=1.5, label="after cornerSubPix")
        chosen = [i for i, v in enumerate(values) if any(r["default"] == "True" for r in rows if r["param"] == param and r["value"] == v)]
        for i in chosen:
            ax.axvline(i, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_xticks(x, values)
        style(ax, label, "corner error (%)")
    ax = axes.ravel()[6]
    values = sorted({r["value"] for r in rows if r["param"] == "THRESH_REL"}, key=float)
    counts = [np.mean([float(r["candidates"]) for r in rows if r["param"] == "THRESH_REL" and r["value"] == v]) for v in values]
    ax.plot(np.arange(len(values)), counts, color=harris_color, marker="o", markersize=4, linewidth=1.5)
    ax.set_xticks(np.arange(len(values)), values)
    style(ax, "candidate threshold (x R.max)", "candidates per image")
    ax = axes.ravel()[7]
    values = sorted({r["value"] for r in block_all}, key=float)
    means = [np.mean([float(r["corner_err_pct"]) for r in block_all if r["value"] == v]) for v in values]
    maxima = [np.max([float(r["corner_err_pct"]) for r in block_all if r["value"] == v]) for v in values]
    ax.plot(np.arange(len(values)), means, color=final_color, marker="s", markersize=4, linewidth=1.5, label="mean")
    ax.plot(np.arange(len(values)), maxima, color=MODELS["H4"][0], marker="^", markersize=4, linewidth=1.5, label="worst")
    ax.set_xticks(np.arange(len(values)), values)
    style(ax, "blockSize, all 32 images", "final corner error (%)")
    ax.legend(fontsize=6, frameon=False, labelcolor=INK_SECONDARY)
    axes.ravel()[0].legend(fontsize=6, frameon=False, labelcolor=INK_SECONDARY)
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "harris_params.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    print("saved", FIG_DIR / "harris_params.png")


def main():
    rows = load_rows()
    residual_plot(rows)
    grid_plot(rows)
    ocr_plot(rows)
    side_by_side(rows)
    harris_example()
    confusions_figure()
    harris_figure()


if __name__ == "__main__":
    main()
