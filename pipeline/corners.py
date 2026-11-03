"""Stage 2a – Harris köşe adayları.

Bu modül yalnızca ADAY üretir. Dış 4 köşenin seçimi Faz 2'nin işi.

Koordinat kuralı: noktalar her zaman (x, y) sırasında tutulur, OpenCV'nin
istediği sıra budur. NumPy indekslemesi ise img[y, x] şeklindedir, dikkat.
"""

# TODO(1): importlar – cv2, numpy as np


# ---------------------------------------------------------------------------
# Parametreler
# ---------------------------------------------------------------------------

# TODO(2): değerleri seç; her birinin etkisini Öğren bölümünde not et
HARRIS_BLOCK_RATIO = ...  # blockSize = genişlik * oran  (komşuluk penceresi)
HARRIS_KSIZE = ...        # Sobel kernel'ı: 3, 5 veya 7
HARRIS_K = ...            # genelde 0.04–0.06
THRESH_REL = ...          # R > THRESH_REL * R.max() olan noktalar aday olur
NMS_RATIO = ...           # NMS penceresi = genişlik * oran; iki aday en az bu kadar uzak olur


def harris_response(gray):
    """Harris R haritasını döndürür (float32, gray ile aynı boyutta)."""
    # TODO(3):
    #   - cornerHarris float32 girdi ister
    #   - blockSize'ı görüntü genişliğinden hesapla (sabit piksel kullanma)
    #   - cv2.cornerHarris(src, blockSize, ksize, k)
    ...


def find_candidates(response):
    """Eşik + NMS ile aday noktaları (N, 2) dizisi olarak, (x, y) sırasında döndürür."""
    # TODO(4): eşik
    #   mask = response > THRESH_REL * response.max()
    # TODO(5): NMS (dilate yöntemi)
    #   - dilated = cv2.dilate(response, kernel)  -> her piksel komşuluğunun maksimumu olur
    #   - yerel maksimum: response == dilated
    #   - adaylar = eşik VE yerel maksimum
    # TODO(6): koordinatlar
    #   - np.nonzero / np.argwhere (satır, sütun) yani (y, x) verir -> (x, y) yap
    # Düşün: düz bir platoda (eşit R değerleri) aynı köşe için birden fazla aday çıkabilir mi?
    ...


def to_original(points, scale):
    """Küçük görüntü koordinatlarını orijinal görüntü koordinatlarına çevirir."""
    # TODO(7): tek satır. 4000 -> 1000 px örneğini burada kontrol et.
    ...
