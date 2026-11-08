import cv2
import numpy as np
from pipeline.preprocess import odd_kernel, to_gray

HARRIS_BLOCK_RATIO = 0.004  # uzun kenarın oranı (~5 px); köşe arama penceresi
HARRIS_KSIZE = 3            # Sobel kernel'ı; blur zaten yapıldığı için en küçüğü yeterli
HARRIS_K = 0.04             # önerilen 0.04-0.06 aralığının altı; köşe kaçırmamayı öne alır
THRESH_REL = 0.01           # R.max()'ın oranı; dev set'te 20/20 görüntüde dış köşeler aday, 108-432 aday
NMS_RATIO = 0.01            # uzun kenarın oranı (~13 px); yakın adayları tek noktaya indirir
R_MIN = 1e4  # dev set'te en küçük R.max 8e6, gürültülü görüntüde ~2.7; ikisinin arasında, ızgarayı elemeyecek kadar düşük
MIN_INSIDE = 3
MIN_AREA_RATIO = 0.02
CORNER_SUPPORT_PX = 6
HARRIS_SEARCH_PX = 6
HARRIS_PEAK_REL = 0.1
SUBPIX_WIN_PX = 3
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)


def harris_response(gray):
    block = odd_kernel(max(gray.shape), HARRIS_BLOCK_RATIO)
    float_gray = gray.astype(np.float32)
    return cv2.cornerHarris(float_gray, block, HARRIS_KSIZE, HARRIS_K)

def find_candidates(response):
    if response.max() < R_MIN:
        return np.empty((0, 2), dtype=float)

    strong = response > THRESH_REL * response.max()

    nkernel = odd_kernel(max(response.shape), NMS_RATIO)
    core = np.ones((nkernel, nkernel), np.uint8)
    dilated = cv2.dilate(response, core)
    mask = strong & (response == dilated)

    mask_uint8 = mask.astype(np.uint8)
    mask_n, mask_labels, mask_stats, mask_centroids = cv2.connectedComponentsWithStats(mask_uint8)
    return mask_centroids[1:]  


def to_original(points, scale):
    return ((points + 0.5) / scale - 0.5)

def to_working(points, scale):
    return (points + 0.5) * scale - 0.5

def order_corners(points):
    points = np.array(points, dtype=float)

    center = points.mean(axis=0)
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    angles = np.arctan2(dy, dx)

    clockwise = points[np.argsort(angles)]

    tl_index = np.argmin(clockwise[:, 0] + clockwise[:, 1])
    return np.roll(clockwise, -tl_index, axis=0)

def find_quads(gray):
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2))
    return quads

def count_inside(big, quads):
    big_float = big.astype(np.float32)
    big_area = cv2.contourArea(big_float)
    count = 0
    for quad in quads:
        area = cv2.contourArea(quad)
        center = quad.mean(axis=0)
        if 0.003 * big_area < area < 0.5 * big_area:
            if cv2.pointPolygonTest(big_float, (float(center[0]), float(center[1])), False) > 0:
                count += 1
    return count

def select_quad(quads, image_area):
    best = None
    for quad in quads:
        if cv2.contourArea(quad) < MIN_AREA_RATIO * image_area:
            continue
        if count_inside(quad, quads) < MIN_INSIDE:
            continue
        if best is None or cv2.contourArea(quad) < cv2.contourArea(best):
            best = quad
    return best

def has_corner_support(quad, candidates):
    if len(candidates) == 0:
        return False

    for corner in quad:
        candidate_distances = np.linalg.norm(candidates - corner, axis=1)
        if np.min(candidate_distances) > CORNER_SUPPORT_PX:
            return False
    return True


def snap_to_harris(response, anchors):
    height, width = response.shape
    core = np.ones((3, 3), np.uint8)
    peaks = []
    for anchor in anchors:
        x, y = int(round(anchor[0])), int(round(anchor[1]))
        x0 = max(x - HARRIS_SEARCH_PX, 0)
        x1 = min(x + HARRIS_SEARCH_PX + 1, width)
        y0 = max(y - HARRIS_SEARCH_PX, 0)
        y1 = min(y + HARRIS_SEARCH_PX + 1, height)
        window = response[y0:y1, x0:x1]

        strong = window > HARRIS_PEAK_REL * window.max()
        local_max = window == cv2.dilate(window, core)
        peak_ys, peak_xs = np.where(strong & local_max)
        if len(peak_xs) == 0:
            return None
        window_peaks = np.column_stack((peak_xs + x0, peak_ys + y0)).astype(float)

        distances = np.linalg.norm(window_peaks - anchor, axis=1)
        peaks.append(window_peaks[np.argmin(distances)])
    return np.array(peaks)


def refine_subpix(gray, points, scale):
    win = max(1, round(SUBPIX_WIN_PX / scale))
    start = points.astype(np.float32).reshape(-1, 1, 2)
    refined = cv2.cornerSubPix(gray, start, (win, win), (-1, -1), SUBPIX_CRITERIA)
    return refined.reshape(-1, 2).astype(float)


def locate_corners(image, pre, response):
    quad = select_quad(find_quads(pre.gray), pre.gray.size)
    if quad is None:
        return None
    peaks = snap_to_harris(response, order_corners(quad))
    if peaks is None:
        return None
    return refine_subpix(to_gray(image), to_original(peaks, pre.scale), pre.scale)
