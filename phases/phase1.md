# Faz 1 – Ön işleme ve Harris (Stage 1, Stage 2a)

**Amaç:** Her fotoğraf için temiz bir aday köşe listesi ve `<name>_harris.png`.
Dış 4 köşenin *seçimi* Faz 2'nin konusu; bu fazda yalnızca adayları üretiyoruz.

---

## 1. CLI iskeleti (`sudoku_scan.py`)

Dosyanın görevi görüntü işlemek değil: komut satırını okur, hangi dosyaların işleneceğine ve çıktının nereye yazılacağına karar verir, her görüntüyü `process_image`'a verir. Asıl iş `pipeline/` içinde yapılır.

| Komut | Davranış |
|---|---|
| `--input foto.jpg` | Tek görüntü, `try/except` yok (traceback görünsün) |
| `--batch klasör/` | Klasördeki her görüntü, her biri `try/except` içinde |
| argümansız | `data/dev/images/dev_01.jpg` (geliştirme modu) |
| `--outdir` | Her modda geçerli; varsayılan `out/` |

**Yapı**
- Sabitler: `DEV_IMAGE`, `DEFAULT_OUTDIR`, `IMAGE_EXTS` (küçük harfli uzantı kümesi)
- `parse_args()`: `ArgumentParser`, `add_mutually_exclusive_group()` (`required` verilmez), `type=Path`, `--outdir` için `default`
- `process_image(image_path, outdir)`: şimdilik görüntüyü okur ve `shape` ile `stem` değerini yazdırır. `cv2.imread` `str` ister ve dosya yoksa `None` döner, bu durumda hata fırlatılır.
- `list_images(folder)`: `iterdir()` + `suffix.lower() in IMAGE_EXTS` + `sorted()`
- `main()`: `outdir.mkdir(parents=True, exist_ok=True)`, sonra moda göre çağrı. Batch'te `except Exception` kullanılır (çıplak `except:` değil) ve sonda "N/M işlendi" özeti yazdırılır.
- `if __name__ == "__main__": main()`

**Testler**
- [ ] Argümansız çalışıyor, dev_01'in boyutunu yazıyor
- [ ] `--input` ve `--batch` birlikte verilince argparse hata veriyor
- [ ] `--batch data/dev/images` 20 görüntüyü işliyor
- [ ] Bir görüntüde kasten `raise` edilince batch devam ediyor (test sonrası o satır silindi)
- [ ] `--input yok.jpg` traceback ile duruyor

---

## 2. Öğren

Her konudan sonra kontrol sorusu cevaplanır.

- [ ] Görüntü = NumPy dizisi; `shape` = (H, W, 3); BGR; `img[y, x]` ile OpenCV'nin `(x, y)` sırası arasındaki fark
- [ ] Gri ton (mavi mürekkep notu), Gaussian blur, adaptive threshold, arka plana bölme: her biri hangi sorunu çözer, hangisi Harris girdisi için uygun?
- [ ] Parametreleri görüntü boyutuna göre ölçekleme; küçült → çalış → koordinatları geri ölçekle
- [ ] Gradyan (Sobel) ve yapı tensörü M
- [ ] R = det(M) − k·trace(M)²: köşe / kenar / düz bölgede R'nin işareti ve büyüklüğü
- [ ] `blockSize`, `ksize`, `k` parametrelerinin etkisi
- [ ] Maksimuma göre eşik + NMS (dilate ile karşılaştırma)
- [ ] `cornerSubPix` ne yapar, ne zaman işe yarar (kullanımı Faz 2'de)

---

## 3. Yap

- [ ] `pipeline/preprocess.py`: küçültme + ölçek oranı döndürme, gri ton, gürültü azaltma, aydınlatma düzeltme. Kernel boyutları görüntü genişliğine oranla belirlenir; sabit piksel kullanılmaz.
- [ ] `pipeline/corners.py`: `cv2.cornerHarris` ile R haritası
- [ ] Eşik + NMS ile aday noktaların `(x, y)` listesi (küçültülmüş koordinatlar; orijinale dönüşüm ölçek oranıyla yapılır)
- [ ] `<name>_harris.png`: R haritası görselleştirmesi + tüm adaylar işaretli
- [ ] `process_image` bu adımları çağırıyor
- [ ] Dev set'te 5–6 görüntüde görsel kontrol: aday sayısı makul mü, dış köşeler aday listesinde var mı? (Anotasyondaki köşelerle karşılaştır; dev set'te köşe = kalın çerçevenin **dış** köşesi.)
- [ ] Batch modunda 20 görüntü < 20 sn/görüntü

---

## 4. Review

- [ ] Kontrol soruları: ikilileştirilmiş ve gri görüntü farkı (Harris'e hangisi verilmeli, neden?); 4000→1000 px koordinat dönüşümü
- [ ] Kod review edildi
- [ ] Ayrı commit'ler: notlar / dev set / iskelet temizliği / CLI iskeleti / preprocess / corners
- [ ] Takip dokümanında maddeler tiklendi, Günlük log'a AI kullanımı yazıldı

---

## Araştırdıklarım


## Review notu
