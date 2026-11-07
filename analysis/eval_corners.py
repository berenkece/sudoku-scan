from pathlib import Path
from pipeline.io_utils import load_annotation, load_image
from pipeline.metrics import corner_error
from pipeline.corners import harris_response, locate_corners
from pipeline.preprocess import preprocess

IMAGE_DIR = Path("data/dev/images")
ANNOTATION_DIR = Path("data/dev/annotations")

def predict_corners(image_path):
    image = load_image(image_path)
    pre = preprocess(image)
    return locate_corners(image, pre, harris_response(pre.gray))

def evaluate():
    errors = []
    not_found = []

    for img_path in sorted(IMAGE_DIR.glob("*.jpg")):
        annotation_path = ANNOTATION_DIR/f"{img_path.stem}.json"
        gt, _ = load_annotation(annotation_path)
        pred = predict_corners(img_path)

        if pred is None:
            not_found.append(img_path.stem)
        else:
            error = corner_error(pred, gt)
            errors.append((img_path.stem, error))
            print(f"{img_path.stem}: {error:.2f}%")

    return errors, not_found

def report(errors, not_found):
    if errors:
        mean_error = sum(e for _, e in errors) / len(errors)
        worst = sorted(errors, key=lambda item: item[1], reverse=True)[:5]
        print(f"mean error: {mean_error:.2f}%")
        print("worst 5:")
        for name, error in worst:
            print(f"{name}: {error:.2f}%")

    print(f"not found: {len(not_found)}")
    print(not_found)

if __name__ == "__main__":
    errors, not_found = evaluate()
    report(errors, not_found)