import cv2
import numpy as np
from pipeline.preprocess import odd_kernel

HARRIS_BLOCK_RATIO = 0.005  # uzun kenarın oranı (~7 px); köşe arama penceresi, blur ile aynı ölçekte
HARRIS_KSIZE = 3            # Sobel kernel'ı; blur zaten yapıldığı için en küçüğü yeterli
HARRIS_K = 0.04             # önerilen 0.04-0.06 aralığının altı; köşe kaçırmamayı öne alır
THRESH_REL = 0.01           # R.max()'ın oranı; dev set'te 20/20 görüntüde dış köşeler aday, 108-432 aday
NMS_RATIO = 0.01            # uzun kenarın oranı (~13 px); yakın adayları tek noktaya indirir


def harris_response(gray):
    block = odd_kernel(max(gray.shape), HARRIS_BLOCK_RATIO)
    float_gray = gray.astype(np.float32)
    return cv2.cornerHarris(float_gray, block, HARRIS_KSIZE, HARRIS_K)

def find_candidates(response):
    strong = response > THRESH_REL * response.max()

    nkernel = odd_kernel(max(response.shape), NMS_RATIO)
    core = np.ones((nkernel, nkernel), np.uint8)
    dilated = cv2.dilate(response, core)
    mask = strong & (response == dilated)

    ys, xs = np.nonzero(mask)
    return np.column_stack((xs, ys))

def to_original(points, scale):
    return (points / scale)

