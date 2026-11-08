# Faz 4 – Hücreler, OCR ve doğrulama (Stage 4)

## Araştırdıklarım

**81 hücre ve margin**
- Hücre (r, c), y'de r·h/9 … (r+1)·h/9, x'te c·w/9 … (c+1)·w/9 aralığındadır. NumPy'da önce y gelir: `img[y1:y2, x1:x2]`. Hücre boyutu hedef boyuttan türetiliyor.
- Margin, hücre kenarına taşan ızgara çizgisini dışarıda bırakır. Az olursa çizgi kalıntısı "mürekkep" sayılır, fazla olursa rakamın ucu kesilir.

**Boş/dolu kararı**
- Her hücreye ayrı Otsu uygulanmaz: Otsu her zaman iki sınıf arar, boş hücrede eşiği kâğıt dokusunun ortasına koyar. İkilileştirme H4 görüntüsünün tamamında bir kez yapılır (arka plana bölme + global Otsu), sonra hücrelere bölünür.
- `connectedComponentsWithStats(bin, connectivity=8)`: `stats` satırları `[x, y, w, h, area]`, etiket 0 arka plan. Ön plan beyaz olmalı, bu yüzden ters ikilileştirme kullanılır.

**Tesseract**
- `--psm 10` tek karakter, `8` tek kelime, `7` tek satır, `13` ham satır.
- `-c tessedit_char_whitelist=123456789`: 0 yok, boş hücre OCR'dan değil boş/dolu kararından gelir. Sürüm 5.5.4, whitelist LSTM motorunda çalışıyor.
- Tesseract belgeler için eğitildiği için beyaz zemin üstünde siyah yazı, etrafında boşluk ve makul bir karakter boyu bekler. Hazırlık adımlarının her biri bu beklentilerden birini karşılar.

**Metrikler**
- Hücre doğruluğu: 81 hücreden doğru olanların oranı (0 = boş da bir sınıf), görüntü başına hesaplanıp ortalaması alınır.
- Boş hücre F1: yalnızca boş/dolu kararını ölçer. Boş hücre çok olduğu için doğruluk yüksek görünürken dolu hücrelerdeki hata gizlenebilir; F1 bunu ortaya çıkarır.
- Confusion matrisi 10×10: satır gerçek, sütun tahmin, 0 = boş.

## Pipeline

1. `orient`: köşe sırası 4 şekilde döndürülür, her yönde ilk 12 dolu hücre Tesseract'a okutulur, ortalama güven puanı en yüksek yön seçilir. Başka bir yön ancak düz yönü 10 puan geçerse seçilir.
2. `warp_for_cells`: OCR için H4 900×900 px'e warp edilir. JSON'daki `H4` yine 450×450 hedefe göre.
3. `split_cells`: arka plana bölme + global Otsu. `remove_lines` ızgara çizgilerini bir hücre boyundaki yatay ve dikey açmayla bulup siler; çizgiler birçok hücre boyunca uzandığı için kalır, rakamlar bir hücreden kısa olduğu için silinmez. Her hücre için hücreyi her yönde %15 aşan bir pencereye bakılır, böylece hücre sınırını taşan rakam da kesilmez. Ham OCR yolu için ayrıca her kenardan %8 margin ile kırpılmış gri hücre tutulur.
4. `find_digit`: penceredeki bağlı bileşenlerden şunlar atılır:
   - alanı hücrenin %0,5'inden küçük olanlar (gürültü),
   - pencereyi kenardan kenara geçenler (silinmeden kalan çizgi parçası),
   - ağırlık merkezi kendi hücresinin orta bandının (%15–85) dışında kalanlar; komşu hücrenin rakamı bu yüzden karışmaz.

   Kalan parçalar birleştirilir. Birleşimin yüksekliği hücrenin %30'undan, alanı %2'sinden büyükse hücre doludur.
5. `prepare`: yalnızca rakam bileşeni alınır, bounding box'a kırpılır, 40 px yüksekliğe büyütülüp yeniden eşiklenir, yarım yükseklik kadar beyaz boşlukla kare tuvalin ortasına konur. Son görüntü beyaz zemin üstünde siyah rakam.
6. `recognize`: psm 10. Rakam dönmezse psm 8 ve 7 denenir. Yine dönmezse aynı sıra 24 ve 32 px rakam yüksekliğiyle denenir. Hiçbiri okuyamazsa hücreye 0 yazılır. OCR çağrıları 8 iş parçacığında paralel.
7. `validate`: satır, sütun ve 3×3 kutu kontrolü, 0'lar atlanır. Tekrar eden rakamın geçtiği bütün hücreler çakışma listesine girer. Çakışma varsa `invalid`, yoksa `valid`.
8. Çıktılar: `_grid.txt` (9 satır, ayraçsız 9 rakam), `_debug.png` (H4 üstünde okunan rakamlar mavi, dolu ama okunamayan hücreler turuncu "?", boş hücreler gri nokta, çakışmalar kırmızı çerçeve), result JSON'da `grid` ve `verdict`. `not_found` karelerde `grid` `null` kalıyor ve `_grid.txt` yazılmıyor. Batch sırasında bir görüntü hata verirse o görüntü için `not_found` JSON'u yazılıyor.

## Review notu

**Sonuç:** Dev set'te boş hücre F1 1,00 (1620 hücre, hiç hata yok), hücre doğruluğu %99,94, 20 grid'in 19'u tam doğru, 20 karenin hepsi `valid`. Kişisel 12 karede boş hücre F1 1,00, hücre doğruluğu %99,79, 10 grid tam doğru, 10 `valid`, 2 `invalid`. Süre dev set'te kare başına ortalama 0,9 sn; kişisel sette ortalama 1,1 sn.

**Pencere ve band seçimi** (`analysis/eval_cells.py`, dev set): band 0,10 ve 0,15'te pencere büyütmesinin 0–0,25 arasındaki her değerinde boş hücre F1 1,0000. Band 0,20'de 1, band 0,25'te 5 dolu hücre boş sanılıyor (hepsi kıvrık `dev_20`'de kenara kaymış rakamlar). Hiçbir ayarda boş hücre dolu çıkmadı. Band 0,15 ve büyütme 0,15 seçildi. OCR sonucu da büyütmenin 0–0,25 arasındaki her değerinde iki sette aynı.

**İlk yöntem ve neden değişti:** İlk sürümde çizgiler silinmiyordu; hücre her kenardan %8 margin ile kırpılıyordu. Dev set'te bu da hatasızdı (margin 0,06–0,12, band 0,10–0,15). Ama gri çerçeveli bulmacada (`s042401168_10`–`12`) rakamlar hücrenin üst sınırını taşıyor ve margin rakamın üstünü kesiyordu: 3 ve 8, 2 okunuyor; 7 okunamıyor; 5, 4 okunuyordu. Kişisel sette hücre doğruluğu %99,28 ve 7/12 tam doğruydu. Çizgileri silip hücreden geniş bir pencereye bakınca bu 5 hatanın hepsi gitti, dev set değişmedi.

**psm karşılaştırması** (`analysis/eval_ocr.py`, dev set):

Güncel tablo `report/tables.md`'de (`analysis/ocr_compare.py`, iki set). Fallback olmadan tek okumada, dev set:

| Yol | Hücre doğruluğu | Rakam doğruluğu | Tam doğru grid | Okunamayan |
|---|---|---|---|---|
| Ana akış (psm 10 + fallback) | 0,9994 | 0,9982 | 19/20 | 0 |
| psm 10, tek okuma | 0,9821 | 0,9487 | 10/20 | 28 |
| psm 7, tek okuma | 0,9821 | 0,9487 | 10/20 | 28 |
| psm 8, tek okuma | 0,9944 | 0,9841 | 16/20 | 4 |
| psm 13, tek okuma | 0,9944 | 0,9841 | 16/20 | 4 |
| psm 10, ham hücre | 0,9265 | 0,7894 | 0/20 | 106 |

- psm 10 ile 7, psm 8 ile 13 birebir aynı sonucu veriyor. Bu, Tesseract 5'in LSTM motorunun tek karakter modunu tek satır gibi işlediğini düşündürüyor (gözlem, belgeyle doğrulanmadı).
- psm 8 tek başına daha çok hücre okuyor ama okuduğunda daha sık yanılıyor; psm 10 emin olmadığında cevap vermiyor. Zinciri psm 10 ile başlatıp okunamayanı fallback'e bırakmak en iyi sonucu veriyor: psm 8 ile başlayan zincir dev set'te 17/20, psm 10 ile başlayan 19/20.
- Hazırlık, aynı psm 10 tek okumada rakam doğruluğunu dev set'te 0,79'dan 0,95'e, kişisel sette 0,67'den 0,91'e çıkarıyor. Ham yol yalnızca ölçüm scriptinde çalışıyor, ana akışta değil.

**Plandan sapmalar:**
- "Kenara değmeyen en büyük bileşen" kuralı yetmedi. `dev_13`'te 5'in üst çizgisi ikilileştirmede gövdeden kopuyor ve iki parça da tek başına kısa kalıyordu; parçalar birleştirilerek çözüldü. `dev_20`'de kıvrım rakamları kırpma kenarına itiyordu; kenar yerine ağırlık merkezine bakılarak çözüldü. Gri çerçeveli bulmacada (`s042401168_09`–`12`) ızgara çizgileri hücreye taşıyıp L şeklinde kalıntı bırakıyordu; önce kenardan kenara geçen bileşen atılarak, sonra çizgiler silinerek çözüldü.
- Hücreler 450 yerine 900 px'lik H4'ten kesiliyor. 450 px'de rakam ~25 px kalıyor, büyütülünce tırtıklı oluyor ve 5→3 karışması artıyordu. Dolu hücrelerde doğru okuma 552/565'ten 560/565'e çıktı.
- Tesseract rakam boyuna çok duyarlı: aynı "1" 24 px'de okunurken 40 px'de okunmuyor. Boyut fallback'i eklenince okunamayan hücre kalmadı. Fallback yalnızca okunamayan hücrede çalıştığı için doğru okunanları bozamaz.
- Bulanıklaştırarak ve gri tonlu hazırlık denendi, kazanç olmadı. 60 px rakam yüksekliği sonucu kötüleştirdi.
- Mavi mürekkep: dev set'te anotasyondaki dolu hücre sayısı `index.csv`'deki basılı rakam sayısıyla 20 karenin hepsinde aynı. Mavi rakamlar da basılı ve dolu sayılıyor, renk ayrımı gerekmedi.

**Yön düzeltmesi:** Yön her hücre için ayrı değil, bütün grid için tek seferde seçiliyor; ters duran tek bir 6, 9'a çevrilemez. 90° ve 270° yönleri en fazla 3,8 puan, 180° yönü 8–56 puan alıyor; doğru yön 62–96. Doğru yön ile ters yön arasındaki en küçük fark 38,6 puan (`s042401168_10`). İlk hücre yönteminde `10`'da rakamların üstü kesik olduğu için bu fark 7,8 puandı (47,9 / 40,1); güvenlik payı o yüzden eklendi. 32 karede yanlış yön seçilmedi. `s042401168_09` ve `_11` artık düz okunuyor, köşe sırası da anotasyonla aynı (köşe hatası %0,18 ve %0,15). Bedeli: kare başına ~0,5 sn.

**Kalan hatalar:**
- Yalnızca 3 hata kaldı, hepsi temiz hazırlanmış rakamlarda Tesseract hatası: `dev_18` 5→3, `s042401168_06` 9→2, `s042401168_08` 8→3.
- `validate` yalnızca tekrar oluşturan hataları görebilir. `dev_18` yanlış okunduğu halde `valid`; `06` ve `08`'deki hatalar tekrar oluşturduğu için `invalid`.

**Bilinen sınırlar:**
- Eşikler dev set'e bakılarak seçildi. Kişisel set bağımsız bir kontrol ama sadece 12 kare ve 3 bulmaca.
- Okunamayan dolu hücre 0 yazılıyor; grid'de boş görünüyor ve `validate` onu atlıyor. İki sette de şu an böyle bir hücre yok.
- Çizgi silme, H4 sonrası çizgilerin yatay ve dikey olduğunu varsayıyor. Kıvrık sayfada eğri kalan çizgi parçaları silinmeyebilir; bunlar kenardan kenara ve ağırlık merkezi kurallarına kalıyor.
- Yön seçimi, örneklenen hücrelerin çoğu 180° dönünce yine rakama benzeyen 1, 8, 6, 9'lardan oluşursa zorlanabilir.
- `eval_cells` ve `eval_ocr` yön kontrolünü çağırmıyor. Dev set'te hep düz yön seçildiği için sonuç değişmiyor; kişisel sette tam akış `sudoku_scan.py` çıktılarıyla ölçülmeli.
