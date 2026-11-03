"""Stage 1 – Ön işleme.

Girdi : orijinal BGR görüntü (örn. 2400x1800)
Çıktı : Harris'e verilecek küçük, gri, temiz görüntü + ölçek oranı

Kural: kernel boyutları SABİT PİKSEL olmaz, çalışma genişliğine oranla hesaplanır.
"""

# TODO(1): importlar – cv2, numpy as np, typing'den NamedTuple


# ---------------------------------------------------------------------------
# Parametreler (oranlar). Değerleri deneyerek sen belirleyeceksin.
# ---------------------------------------------------------------------------

# TODO(2): değerleri seç ve neden seçtiğini yanına not et
WORK_WIDTH = ...     # piksel; örn. ~1000. Hız ile ayrıntı arasındaki denge.
BLUR_RATIO = ...     # Gaussian kernel = genişlik * BLUR_RATIO  (küçük olmalı)
BG_RATIO = ...       # arka plan tahmini kernel'ı; bir hücreden BÜYÜK olmalı, neden?


class Preprocessed(NamedTuple):
    """preprocess() çıktısı.

    gray  : küçültülmüş, düzeltilmiş tek kanallı görüntü (uint8)
    scale : küçük / orijinal oranı.  orijinal_koordinat = küçük_koordinat / scale
    """
    gray: np.ndarray
    scale: float


# ---------------------------------------------------------------------------
# Yardımcılar – her biri tek bir iş yapar
# ---------------------------------------------------------------------------

def odd_kernel(width, ratio):
    """width * ratio değerini en yakın TEK tamsayıya çevirir (en az 3)."""
    # TODO(3): GaussianBlur/medianBlur tek sayı kernel ister. Neden?
    ...


def resize_to_width(image, target_width):
    """En-boy oranını koruyarak küçültür. (küçük_görüntü, scale) döndürür."""
    # TODO(4):
    #   - scale = target_width / orijinal_genişlik
    #   - görüntü zaten küçükse büyütme (scale = 1.0)
    #   - küçültmede hangi interpolation? (cv2.INTER_AREA neden iyi?)
    # Düşün: dikey fotoğraflarda (1800x2400) genişlik mi, uzun kenar mı baz alınmalı?
    ...


def to_gray(image):
    """BGR -> tek kanal."""
    # TODO(5): cvtColor yeterli mi? Mavi tükenmez kalemle yazılmış rakamlar
    #   gri tonda neredeyse kayboluyorsa tek bir kanal (B/G/R) daha mı iyi?
    #   Kararını Araştırdıklarım bölümüne yaz.
    ...


def denoise(gray):
    """Gürültüyü azaltır (Gaussian blur). Kernel genişliğe oranla belirlenir."""
    # TODO(6): odd_kernel(...) + cv2.GaussianBlur
    ...


def normalize_illumination(gray):
    """Gölge/ışık farkını düzeltir: görüntü / arka_plan_tahmini."""
    # TODO(7):
    #   - arka plan tahmini: büyük kernel'lı blur (veya morfolojik closing)
    #   - gray / arka_plan  -> float işlem, sonra 0..255 aralığına normalize, uint8
    #   - sıfıra bölmeye dikkat
    # Not: burada adaptive threshold YAPMIYORUZ. Harris gri görüntü ister (neden?)
    ...


# ---------------------------------------------------------------------------
# Modülün dışarıya açık tek fonksiyonu
# ---------------------------------------------------------------------------

def preprocess(image):
    """Stage 1'in tamamı: küçült -> gri -> gürültü azalt -> aydınlatma düzelt."""
    # TODO(8): yukarıdaki yardımcıları sırayla çağır, Preprocessed(gray, scale) döndür
    ...
