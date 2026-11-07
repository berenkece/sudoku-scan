# Faz 2 – Köşe seçimi (Stage 2b) — yapılacaklar
Kural: seçim mantığını (bölge, 4 köşe seçimi, sıralama, not_found kriteri) Beren yazar.
Claude Code yalnızca takip eder, yardımcı araçlarda yardım eder, review yapar.

## Hazırlık / araçlar
- [x] io_utils.load_annotation(path) → corners, grid
- [x] pipeline/metrics.py: corner_error(pred, gt) (ort. Öklid / GT kenar ortalaması × 100)
- [x] analysis/eval_corners.py: dev set'te görüntü başına %, ortalama, en kötü 5, not_found listesi
- [x] visualize.draw_harris: seçilen 4 köşe farklı renkte, TL/TR/BR/BL etiketli
- [x] corners.to_original → (p + 0.5) / s − 0.5

## Aday temizliği (Faz 1 notundan)
- [x] Çift adaylar (dev_18, eşit R değerli komşu pikseller)
- [x] Köşesiz görüntü için taban eşik / kontrol

## Köşe seçimi (Beren)
- [x] Bölge bulma: kâğıt kenarı ve ikinci dikdörtgenden (dev_05, 11, 13) ayırt etme
- [x] Adayları bölgeyle sınırlama (toleranslı maske / pointPolygonTest)
      → Maske yerine dörtgen köşesinin çevresindeki 6 px'lik pencerede Harris tepesi aranıyor.
- [x] 4 köşenin seçimi; kalın çerçevede dış köşenin garanti edilmesi
      → Dış köşe kişisel karelerde gözle ve kayma ölçümüyle kontrol edildi, garanti değil.
- [x] order_corners: TL, TR, BR, BL (atan2 + başlangıç noktası)
- [x] not_found yolu (kriterler + result JSON)
- [x] cornerSubPix (orijinal görüntüde); ~9 px içe kayma düzeliyor mu, ölç
- [x] Kuralın gerekçesi phases/phase2.md'ye (rapora girecek)

## Ölçüm
- [x] Dev set'in tamamında köşe hatası %; başarısız görüntüler listesi
      (dev_19, 20 kıvrımlı; dev_12 dönmüş + gazete; dev_15 yoğun metin)
      → Başarısız görüntü yok: 20/20, ortalama %0,17, en kötü %0,28.

## Gerçek fotoğraflar (bu akşam)
- [x] 3 kaynak: sudoku kitabı + 2 farklı görünümlü bulmaca
      → Şartname kitabı zorunlu kılmıyor; 3 farklı çıktı kullanıldı.
- [x] 12 kare, her karede öğrenci no + tarih kartı (grid dışında)
- [x] C1 · C2 ×3 (biri ≥40°) · C3 · C4 · C5
      → s042401168_07 bilerek 40°'den fazla eğilerek çekildi; açı ölçülmedi.
- [x] Orijinaller (EXIF korunarak) → data/personal/
- [x] Kuralı 3–4 gerçek karede dene
      → 12 karenin hepsinde denendi.

## Kapanış
- [ ] Kuralı kendi cümlelerimle 3 dakikada anlatabiliyorum (savunma provası)
- [ ] Parça parça commit'ler (araçlar / sıralama / kural / subpix)
- [ ] phase2.md: Araştırdıklarım + Review notu; takip dokümanında tikler; AI kullanım logu
      → Araştırdıklarım, Review notu ve tikler tamam.

---

## Kuralın gerekçesi

Bulmaca, içinde çok sayıda küçük dörtgen barındıran en küçük büyük dörtgendir. Kâğıt ve gri çerçeve de aynı küçük dörtgenleri içerir ama bulmacadan büyüktür. Notes kutusunun ve tek bir hücrenin içinde ise küçük dörtgen yoktur. Kitap gibi ikinci bir ızgara, görüntüde asıl bulmacadan küçük kalırsa %2 alan sınırına takılır ya da yeterli iç dörtgen üretmez; bu `s042401168_12`'de gözlendi, genel bir garanti değil.

Dörtgen yalnızca köşenin nerede aranacağını gösterir. Şartnameye göre son 4 köşe `cornerHarris`'ten gelmelidir. Bu yüzden dörtgen köşesinin yakınındaki güçlü Harris tepelerinden en yakını seçilir ve orijinal görüntüde `cornerSubPix` ile iyileştirilir.

Uygun dörtgen yoksa veya bir köşenin çevresinde Harris tepesi yoksa sonuç `not_found` olur.

## Araştırdıklarım

1. Otsu tüm görüntüye tek eşik uygular. Görüntüde çok koyu bir bölge varsa eşik aşağı iner ve soluk çizgiler kaybolabilir. `adaptiveThreshold` her pikseli kendi çevresiyle karşılaştırır.
2. `findContours` beyaz şekilleri arar, bu yüzden `THRESH_BINARY_INV` ile çizgiler beyaz yapılır. `RETR_LIST` iç içe şekilleri de getirir.
3. `approxPolyDP` konturu daha az köşeli bir şekle indirir; `epsilon` büyüdükçe şekil daha çok sadeleşir.
4. `pointPolygonTest` bir noktanın çokgenin içinde olup olmadığını söyler; pozitif sonuç "içinde" demektir.
5. `np.arctan2(dy, dx)` merkez etrafındaki açıyı verir. Görüntüde y aşağı arttığı için artan açı saat yönüdür.
6. `np.roll` ile bir elemanı başa almak için negatif kaydırma gerekir: `np.roll(a, -i, axis=0)`. `axis=-1` köşe sırasını değil, x ve y sütunlarını kaydırır.
7. Harris tepesi köşeyi kalın çizgide içe doğru kaydırır. `cornerSubPix` gradyanlara bakarak noktayı ondalıklı hassasiyetle düzeltir; pencere boyutu sonucu güçlü etkiler.

## Review notu

**Sonuç:** Dev set'te 20/20 köşe bulundu. Ortalama hata %0,17, en kötü %0,28 (`dev_10`). Kişisel 12 karenin hepsinde köşe bulundu; anotasyon olmadığı için hata yüzdesi yok. Boş, gürültülü, içi boş dikdörtgenli ve bulmacasız kırpılmış görüntüde sonuç `not_found`.

**Dokümandan sapmalar:**
- Köşeler aday listesinden değil, dörtgen köşesi çevresindeki Harris yanıtından seçiliyor. Genel listede `s042401168_07`, `10` ve `12` için en yakın aday 36–65 px uzaktaydı.
- "Toleranslı maske" yerine küçük arama penceresi kullanılıyor.
- 3 kaynağın hepsi çıktı, kitap yok.

**Faz 3'e devreden konular:**
- `order_corners` 45°'den fazla dönmüş görüntüde gerçek sol üstü bilemez. `s042401168_09` ve `s042401168_11`'de rakamlar yaklaşık 90° yan duruyor, ızgara bu karelerde dönük çıkacak.
- Sonuç JSON'unda `grid`, `A3`, `A4`, `H4`, `a3_corner_err_px`, `a4_rms_px` ve bulunan karelerde `verdict` şimdilik `null`.
- `conditions` için `[]` ve `not_found` karelerde `null` alanlar varsayılan seçim; şartname bunları açık bırakıyor.

**Bilinen sınırlar:**
- Eşikler aynı karelere bakılarak seçildi, ayrı bir test seti yok. Yeni görüntülerde sonuç düşebilir.
- Sudoku olmayan ama ızgaralı bir görüntü (ör. tek başına bir tablo) denenmedi.
- `has_corner_support` ve `CORNER_SUPPORT_PX` artık kullanılmıyor. `find_candidates` yalnızca hata ayıklama görseli için çalışıyor.
