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
- [x] Argümansız çalışıyor, dev_01'in boyutunu yazıyor
- [x] `--input` ve `--batch` birlikte verilince argparse hata veriyor
- [x] `--batch data/dev/images` 20 görüntüyü işliyor
- [x] Bir görüntüde kasten `raise` edilince batch devam ediyor (test sonrası o satır silindi)
- [x] `--input yok.jpg` traceback ile duruyor

---

## 2. Öğren

Her konudan sonra kontrol sorusu cevaplanır.

- [x] Görüntü = NumPy dizisi; `shape` = (H, W, 3); BGR; `img[y, x]` ile OpenCV'nin `(x, y)` sırası arasındaki fark
- [x] Gri ton (mavi mürekkep notu), Gaussian blur, adaptive threshold, arka plana bölme: her biri hangi sorunu çözer, hangisi Harris girdisi için uygun?
- [x] Parametreleri görüntü boyutuna göre ölçekleme; küçült → çalış → koordinatları geri ölçekle
- [x] Gradyan (Sobel) ve yapı tensörü M
- [x] R = det(M) − k·trace(M)²: köşe / kenar / düz bölgede R'nin işareti ve büyüklüğü
- [x] `blockSize`, `ksize`, `k` parametrelerinin etkisi
- [x] Maksimuma göre eşik + NMS (dilate ile karşılaştırma)
- [x] `cornerSubPix` ne yapar, ne zaman işe yarar (kullanımı Faz 2'de)

---

## 3. Yap

- [x] `pipeline/preprocess.py`: küçültme + ölçek oranı döndürme, gri ton, gürültü azaltma, aydınlatma düzeltme. Kernel boyutları görüntü genişliğine oranla belirlenir; sabit piksel kullanılmaz.
- [x] `pipeline/corners.py`: `cv2.cornerHarris` ile R haritası
- [x] Eşik + NMS ile aday noktaların `(x, y)` listesi (küçültülmüş koordinatlar; orijinale dönüşüm ölçek oranıyla yapılır)
- [x] `<name>_harris.png`: R haritası görselleştirmesi + tüm adaylar işaretli
- [x] `process_image` bu adımları çağırıyor
- [x] Dev set'te 5–6 görüntüde görsel kontrol: aday sayısı makul mü, dış köşeler aday listesinde var mı? (Anotasyondaki köşelerle karşılaştır; dev set'te köşe = kalın çerçevenin **dış** köşesi.)
- [x] Batch modunda 20 görüntü < 20 sn/görüntü

---

## 4. Review

- [x] Kontrol soruları: ikilileştirilmiş ve gri görüntü farkı (Harris'e hangisi verilmeli, neden?); 4000→1000 px koordinat dönüşümü
- [x] Kod review edildi
- [ ] Günlük commit'ler atıldı (hocanın kuralı: proje tek commit'te gönderilmez)
- [ ] Takip dokümanında maddeler tiklendi, genel log'a AI kullanımı yazıldı

---

## Araştırdıklarım

1. NumPy img[y, x], OpenCV (x, y); imread BGR verir.
2. Harris'e ikili değil, gölgesi düzeltilmiş gri görüntü verilir.
3. Küçült, çalış, koordinatları s'ye bölerek orijinale geri çevir.
4. M, penceredeki gradyan yönlerini özetler; iki büyük özdeğer köşe demektir.
5. R pozitifse köşe, negatifse kenar, sıfıra yakınsa düz bölge.
6. blockSize hangi büyüklükte köşe, k ne kadar seçici, ksize gradyan ne kadar pürüzsüz.
7. Önce maksimuma göre eşik, sonra dilate ile NMS; her köşe tek nokta.
8. cornerSubPix köşeyi orijinal görüntüde ondalıklı hassasiyete inceltir.
9. Mavi mürekkep kararı: şimdilik standart gri ton (`COLOR_BGR2GRAY`), OCR'da sorun çıkarsa dönülecek.

## Review notu

**Sonuç:** Dış 4 köşe dev set'in 20 görüntüsünün hepsinde aday listesinde (küçültülmüş görüntüde en fazla 5,5 px uzakta). Aday sayısı 108–432. Batch: 20 görüntü toplam 1,3 sn. Görsel kontrol: dev_01, 05, 07, 11, 13, 15.

**Dokümandan sapma:** Küçültme ve kernel boyutları genişliğe değil uzun kenara göre hesaplanıyor. Dev set'te dikey ve yatay fotoğraflar bir arada; genişliği baz alınca aynı ızgara dikey fotoğrafta daha büyük kalıyor.

**Öğren bölümü:** Notlar "Araştırdıklarım" bölümünde. Kontrol soruları ayrıca cevaplanmadı.

**Faz 2'de ele alınacaklar:**
- İçe doğru kayma: Adaylar gerçek köşeden ızgaranın merkezine doğru tipik olarak ~3,5 px kayıyor (ölçek 0,375'te orijinalde ~9 px). `cornerSubPix` bunu çözüyor mu, ölçülecek; çözmezse çerçeve kırpılabilir.
- Çift adaylar: R değeri eşit komşu pikseller aynı köşe için iki aday üretebiliyor (dev_18'de 7 px'ten yakın bir çift var).
- Köşesiz görüntü: Eşik sadece `R.max()` değerine bağlı. Yapay gürültülü görüntü 2683 aday üretti; taban eşik ya da kontrol gerekecek.
- Doygunluk: Işık düzeltmesi sonrası piksellerin %62–79'u 255 oluyor. Şu an köşe kaçırmıyor; soluk çerçeveli görüntüde köşe kaçarsa önce buraya bakılacak.
- Köşe seçimi: Kağıt köşeleri de aday (dev_05, dev_11), Notes kutusu ve yan kağıttaki tablo da köşe üretiyor (dev_13), metin çok aday üretiyor (dev_15). "En dıştaki 4 aday" ya da "herhangi bir dörtgen" yöntemi yetmez.
- İnce çizgiler: dev_05 ve dev_11'de bazı iç kesişimlerde aday yok. Gerekirse `THRESH_REL` ve `BLUR_RATIO` değerlerine bakılacak.
