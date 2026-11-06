import cv2

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
