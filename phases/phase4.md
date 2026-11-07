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
3. `split_cells`: arka plana bölme + global Otsu, 81 hücre, her kenardan hücre boyunun %8'i margin.
4. `find_digit`: hücredeki bağlı bileşenlerden şunlar atılır:
   - alanı hücrenin %0,5'inden küçük olanlar (gürültü),
   - kırpılmış hücreyi kenardan kenara geçenler (ızgara çizgisi),
   - ağırlık merkezi hücrenin orta bandının (%15–85) dışında kalanlar.

   Kalan parçalar birleştirilir. Birleşimin yüksekliği hücrenin %30'undan, alanı %2'sinden büyükse hücre doludur.
5. `prepare`: yalnızca rakam bileşeni alınır, bounding box'a kırpılır, 40 px yüksekliğe büyütülüp yeniden eşiklenir, yarım yükseklik kadar beyaz boşlukla kare tuvalin ortasına konur. Son görüntü beyaz zemin üstünde siyah rakam.
6. `recognize`: psm 10. Rakam dönmezse psm 8 ve 7 denenir. Yine dönmezse aynı sıra 24 ve 32 px rakam yüksekliğiyle denenir. Hiçbiri okuyamazsa hücreye 0 yazılır. OCR çağrıları 8 iş parçacığında paralel.
7. `validate`: satır, sütun ve 3×3 kutu kontrolü, 0'lar atlanır. Tekrar eden rakamın geçtiği bütün hücreler çakışma listesine girer. Çakışma varsa `invalid`, yoksa `valid`.
8. Çıktılar: `_grid.txt` (9 satır, ayraçsız 9 rakam), `_debug.png` (H4 üstünde okunan rakamlar mavi, dolu ama okunamayan hücreler turuncu "?", boş hücreler gri nokta, çakışmalar kırmızı çerçeve), result JSON'da `grid` ve `verdict`. `not_found` karelerde `grid` `null` kalıyor ve `_grid.txt` yazılmıyor. Batch sırasında bir görüntü hata verirse o görüntü için `not_found` JSON'u yazılıyor.

## Review notu

**Sonuç:** Dev set'te boş hücre F1 1,00 (1620 hücre, hiç hata yok), hücre doğruluğu %99,94, 20 grid'in 19'u tam doğru, 20 karenin hepsi `valid`. Kişisel 12 karede hücre doğruluğu %99,28, 7 grid tam doğru, 9 `valid`, 3 `invalid`. Süre dev set'te kare başına ortalama 0,9 sn, en fazla 1,2 sn; kişisel sette ortalama 1,1 sn, en fazla 1,5 sn.

**Margin ve band seçimi** (`analysis/eval_cells.py`, dev set, boş hücre F1):

| band \ margin | 0,04 | 0,06 | 0,08 | 0,10 | 0,12 | 0,15 | 0,18 |
|---|---|---|---|---|---|---|---|
| 0,10 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 1,0000 | 0,9995 | 0,9976 |
| 0,15 | 1,0000 | 1,0000 | **1,0000** | 1,0000 | 1,0000 | 0,9995 | 0,9976 |
| 0,20 | 0,9995 | 0,9991 | 0,9991 | 0,9991 | 0,9986 | 0,9981 | 0,9976 |
| 0,25 | 0,9948 | 0,9943 | 0,9943 | 0,9943 | 0,9943 | 0,9939 | 0,9939 |

Hiçbir ayarda boş hücre dolu çıkmadı; hatalar hep dolu hücrenin boş sanılması. Margin büyüdükçe rakamın ucu kesiliyor, band büyüdükçe kıvrık `dev_20`'de kenara kaymış rakamlar kaçıyor. Hatasız bölgenin ortası seçildi: margin 0,08, band 0,15. Kişisel sette (eşikler ona bakılarak seçilmedi) margin 0,06–0,10 aralığında da hatasız.

**psm karşılaştırması** (`analysis/eval_ocr.py`, dev set):

| Yol | Hücre doğruluğu | Tam doğru grid | Boş hücre F1 |
|---|---|---|---|
| psm 10, hazırlıklı | 0,9994 | 19/20 | 1,0000 |
| psm 7, hazırlıklı | 0,9994 | 19/20 | 1,0000 |
| psm 8, hazırlıklı | 0,9963 | 17/20 | 1,0000 |
| psm 13, hazırlıklı | 0,9963 | 17/20 | 1,0000 |
| psm 10, ham hücre | 0,9444 | 0/20 | 0,9666 |

psm 10 seçildi: mantıken doğru mod bu ve en iyi sonuçla berabere. Ham hücrede F1 düşüyor, çünkü OCR'ın okuyamadığı dolu hücreler 0 yazılıyor ve boş görünüyor. Ham yol yalnızca ölçüm scriptinde çalışıyor, ana akışta değil.

**Plandan sapmalar:**
- "Kenara değmeyen en büyük bileşen" kuralı yetmedi. `dev_13`'te 5'in üst çizgisi ikilileştirmede gövdeden kopuyor ve iki parça da tek başına kısa kalıyordu; parçalar birleştirilerek çözüldü. `dev_20`'de kıvrım rakamları kırpma kenarına itiyordu; kenar yerine ağırlık merkezine bakılarak çözüldü. Gri çerçeveli bulmacada (`s042401168_09`–`12`) ızgara çizgileri hücreye taşıyıp L şeklinde kalıntı bırakıyordu; kenardan kenara geçen bileşen atılarak çözüldü.
- Hücreler 450 yerine 900 px'lik H4'ten kesiliyor. 450 px'de rakam ~25 px kalıyor, büyütülünce tırtıklı oluyor ve 5→3 karışması artıyordu. Dolu hücrelerde doğru okuma 552/565'ten 560/565'e çıktı.
- Tesseract rakam boyuna çok duyarlı: aynı "1" 24 px'de okunurken 40 px'de okunmuyor. Boyut fallback'i eklenince okunamayan hücre kalmadı. Fallback yalnızca okunamayan hücrede çalıştığı için doğru okunanları bozamaz.
- Bulanıklaştırarak ve gri tonlu hazırlık denendi, kazanç olmadı. 60 px rakam yüksekliği sonucu kötüleştirdi.
- Mavi mürekkep: dev set'te anotasyondaki dolu hücre sayısı `index.csv`'deki basılı rakam sayısıyla 20 karenin hepsinde aynı. Mavi rakamlar da basılı ve dolu sayılıyor, renk ayrımı gerekmedi.

**Yön düzeltmesi:** Yön her hücre için ayrı değil, bütün grid için tek seferde seçiliyor; ters duran tek bir 6, 9'a çevrilemez. 90° ve 270° yönleri hep ~0 puan, 180° yönü 8–56 puan alıyor. Doğru yön `s042401168_10` hariç 62–96. `10`'da düz yön 47,9, ters yön 40,1 aldı; güvenlik payı bu yüzden var. 32 karede yanlış yön seçilmedi. `s042401168_09` ve `_11` artık düz okunuyor, köşe sırası da anotasyonla aynı (köşe hatası %0,18 ve %0,15). Bedeli: kare başına ~0,5 sn.

**Kalan hatalar:**
- Tesseract hataları, hazırlanmış görüntüler temiz: `dev_18` 5→3, `s042401168_06` 9→2, `s042401168_08` 8→3.
- Gri çerçeveli bulmacada yazı tipi büyük ve rakamlar hücrenin üstüne yapışık, rakamın üstü margin'de kesiliyor: `s042401168_10` 3→2 ve 8→2, `_11` 3→2, `_12` 7→0 ve 5→4.
- `validate` yalnızca tekrar oluşturan hataları görebilir. `dev_18` ve `s042401168_12` yanlış okunduğu halde `valid`.

**Bilinen sınırlar:**
- Eşikler dev set'e bakılarak seçildi. Kişisel set bağımsız bir kontrol ama sadece 12 kare ve 3 bulmaca.
- Okunamayan dolu hücre 0 yazılıyor; grid'de boş görünüyor ve `validate` onu atlıyor.
- Yön seçimi, örneklenen hücrelerin çoğu 180° dönünce yine rakama benzeyen 1, 8, 6, 9'lardan oluşursa zorlanabilir.
- `eval_cells` ve `eval_ocr` yön kontrolünü çağırmıyor. Dev set'te hep düz yön seçildiği için sonuç değişmiyor; kişisel sette tam akış `sudoku_scan.py` çıktılarıyla ölçülmeli.
