import argparse
import time
from pathlib import Path

from pipeline.io_utils import load_image, save_image, save_result
from pipeline.preprocess import preprocess
from pipeline.corners import harris_response, find_candidates, locate_corners, to_working
from pipeline.visualize import draw_harris

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

    corners_working = None if corners is None else to_working(corners, pre.scale)
    vis = draw_harris(pre.gray, response, candidates, corners_working)
    save_image(outdir / f"{name}_harris.png", vis)

    save_result(outdir / f"{name}_result.json", image_path.name, corners, time.perf_counter() - start)
    print(name, "not_found" if corners is None else "corners found")

def run_batch(folder, outdir):
    images = list_images(folder)
    n_ok = 0
    for img in images:
        try:
            process_image(img, outdir)
            n_ok += 1

        except Exception as e:
            print(img.name, e)

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
