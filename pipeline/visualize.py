"""Debug görselleri. Burada sadece ÇİZİM yapılır; hesaplama ve dosya yazma yapılmaz."""

# TODO(1): importlar – cv2, numpy as np


def draw_harris(gray, response, candidates):
    """<name>_harris.png içeriği: R haritası + adaylar işaretli, BGR görüntü döndürür."""
    # TODO(2): R haritasını görünür yap
    #   - R çok geniş bir aralıkta değişir ve negatif değerler alır. Doğrudan uint8'e çevirmek işe yaramaz.
    #   - Seçenekler: sadece pozitif kısmı al, log/sqrt ile sıkıştır, 0..255'e normalize et
    #   - cv2.applyColorMap ile renklendir (örn. COLORMAP_JET)
    # TODO(3): okunabilirlik için gri görüntüyle karıştır (cv2.addWeighted) – isteğe bağlı
    # TODO(4): her aday için cv2.circle; koordinatlar int ve (x, y) sırasında olmalı
    # TODO(5): sol üste aday sayısını yaz (cv2.putText)
    ...
