"""Dosya okuma/yazma yardımcıları. Görüntü işleme burada YAPILMAZ."""

# TODO(1): importlar – cv2, numpy as np, pathlib.Path


def load_image(path):
    """Görüntüyü BGR NumPy dizisi olarak okur.

    Dosya okunamazsa None döndürmez, hata fırlatır.
    """
    # TODO(2):
    #   - cv2.imread Path değil str ister
    #   - dönen değer None ise raise (hangi hata türü? FileNotFoundError mantıklı)
    #   - Hata mesajına yolu koy ki batch çıktısında hangi dosya olduğu görünsün
    ...


def save_image(path, image):
    """Görüntüyü yazar; klasör yoksa oluşturur."""
    # TODO(3):
    #   - path.parent.mkdir(parents=True, exist_ok=True)
    #   - cv2.imwrite(str(path), image)
    #   - imwrite başarısız olursa False döner -> kontrol edip raise et
    ...
