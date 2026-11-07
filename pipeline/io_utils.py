import cv2
import json

def load_image(path):
    
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"could not read image {path}")
    return image


def save_image(path, image):

    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), image)
    if not success:
        raise OSError(f"path could not be written {path}")

def load_annotation(path):
    with open(path, "r") as f:
        annotation = json.load(f)
    corners = annotation["corners"]
    grids = annotation["grid"]
    return (corners, grids)


def save_result(path, image_name, corners, runtime_s, rect=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "image": image_name,
        "conditions": [],
        "corners": None if corners is None else corners.tolist(),
        "grid": None,
        "A3": None if rect is None else rect.a3.tolist(),
        "A4": None if rect is None else rect.a4.tolist(),
        "H4": None if rect is None else rect.h4.tolist(),
        "a3_corner_err_px": None if rect is None else rect.a3_corner_err_px,
        "a4_rms_px": None if rect is None else rect.a4_rms_px,
        "verdict": "not_found" if corners is None else None,
        "runtime_s": runtime_s,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
