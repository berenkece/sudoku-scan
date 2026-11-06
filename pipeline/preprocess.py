
import cv2
import numpy as np
from typing import NamedTuple


WORK_LONG_SIDE = 1200  # uzun kenar bu boyuta iner; dikey/yatay fotoğrafta ızgara benzer boyutta kalır
BLUR_RATIO = 0.005     # uzun kenarın oranı (~7 px); gürültüyü azaltır, kalın çerçeve köşelerini silmez
BG_RATIO = 0.12        # uzun kenarın oranı (~145 px); bir hücreden büyük olmalı, yoksa arka plan ızgarayı ve rakamları da takip eder


class Preprocessed(NamedTuple):
    gray: np.ndarray
    scale: float


def odd_kernel(size, ratio):
    wr_int = int(size*ratio)
    if not (wr_int % 2):
        wr_int+=1

    if wr_int < 3:
        return 3

    return wr_int


def resize_long_side(image, target_long_side):
    long_side = max(image.shape[:2])
    scale = target_long_side / long_side
    if scale >= 1.0 :
        return (image, 1.0)
    else:
        small_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return (small_image, scale)


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray):
    k = odd_kernel(max(gray.shape), BLUR_RATIO)
    return cv2.GaussianBlur(gray, (k, k), 0)


def normalize_illumination(gray):
    k = odd_kernel(max(gray.shape), BG_RATIO)
    gaus_blur = cv2.GaussianBlur(gray, (k,k), 0)
    return cv2.divide(gray, gaus_blur, scale=255)


def preprocess(image):
    small, scale = resize_long_side(image, WORK_LONG_SIDE)
    gray_image = to_gray(small)
    den_img = denoise(gray_image)
    preprcssd = normalize_illumination(den_img)
    return Preprocessed(preprcssd, scale)

"""test:
python -c "
from pathlib import Path
from pipeline.io_utils import load_image, save_image
from pipeline.preprocess import preprocess
pre = preprocess(load_image(Path('data/dev/images/dev_01.jpg')))
print(pre.gray.shape, pre.gray.dtype, pre.scale)
save_image(Path('out/dev_01_pre.png'), pre.gray)
"
"""
    
    
