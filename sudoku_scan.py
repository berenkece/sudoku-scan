import argparse
import time
from pathlib import Path

from pipeline.io_utils import load_image, save_grid, save_image, save_result
from pipeline.preprocess import preprocess
from pipeline.corners import harris_response, find_candidates, locate_corners, to_working
from pipeline.rectify import fit_models, warp_affine, warp_perspective
from pipeline.cells import filled_mask
from pipeline.orient import orient
from pipeline.ocr import recognize
from pipeline.validate import find_conflicts, verdict
from pipeline.visualize import draw_debug, draw_harris

DEV_IMAGE = Path("data/dev/images/dev_01.jpg")
DEFAULT_OUTDIR = Path("out")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}   


def parse_args():

    parser = argparse.ArgumentParser(description="sudoku scan: detect a printed sudoku in a photo and return its 9x9 grid.")
    group = parser.add_mutually_exclusive_group()

    group.add_argument("--input", type=Path, help="single image file to process")
    group.add_argument("--batch", type=Path, help="folder of images: every .jpg/.jpeg/.png file in it is processed" )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="directory where output files are written (default: %(default)s)")

    return parser.parse_args()


def list_images(folder):
    list_of_images = sorted(pth for pth in folder.iterdir() if pth.suffix.lower() in IMAGE_EXTS)

    return list_of_images


def process_image(image_path, outdir):

    start = time.perf_counter()
    name = image_path.stem

    image = load_image(image_path)
    print(name, "image shape", image.shape)

    pre = preprocess(image)
    response = harris_response(pre.gray)
    candidates = find_candidates(response)
    print(name, "number of candidates", len(candidates))

    corners = locate_corners(image, pre, response)
    if corners is not None:
        corners, cells, turn = orient(image, corners)
        if turn:
            print(name, "rotated corner order by", turn)

    corners_working = None if corners is None else to_working(corners, pre.scale)
    vis = draw_harris(pre.gray, response, candidates, corners_working)
    save_image(outdir / f"{name}_harris.png", vis)

    rect = grid = result = None
    if corners is not None:
        rect = fit_models(corners)
        rectified = warp_perspective(image, rect.h4)
        save_image(outdir / f"{name}_a3.png", warp_affine(image, rect.a3))
        save_image(outdir / f"{name}_a4.png", warp_affine(image, rect.a4))
        save_image(outdir / f"{name}_h4.png", rectified)

        grid = recognize(cells)
        conflicts = find_conflicts(grid)
        result = verdict(conflicts)
        save_grid(outdir / f"{name}_grid.txt", grid)
        save_image(outdir / f"{name}_debug.png", draw_debug(rectified, grid, filled_mask(cells), conflicts))

    save_result(outdir / f"{name}_result.json", image_path.name, corners, time.perf_counter() - start, rect, grid, result)
    print(name, "not_found" if corners is None else result)

def run_batch(folder, outdir):
    images = list_images(folder)
    n_ok = 0
    for img in images:
        try:
            process_image(img, outdir)
            n_ok += 1

        except Exception as e:
            print(img.name, e)
            save_result(outdir / f"{img.stem}_result.json", img.name, None, 0.0)

    print(f"{n_ok}/{len(images)} processed.")


def main():

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.input:
        process_image(args.input, args.outdir)
    elif args.batch:
        run_batch(args.batch, args.outdir)
    else:
        process_image(DEV_IMAGE, args.outdir)


if __name__ == "__main__":
    main()
