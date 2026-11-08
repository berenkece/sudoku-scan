# Faz 5 – Analiz (Stage 4, Section 7)

## Yapılanlar

| Adım | Dosya | Durum |
|---|---|---|
| 1. Anotasyon | `analysis/annotate.py`, `data/personal/annotations/` | 12 kare (Faz 3'te) |
| 2. Değerlendirme | `analysis/evaluate.py` | Dev ve kişisel set |
| 3. Harris parametreleri | `analysis/harris_params.py` | 8 kare + blockSize için 32 kare |
| 4. A3 / A4 / H4 karşılaştırması | `analysis/obliqueness.py`, `analysis/compare_models.py` | 32 kare |
| 5. Figürler | `analysis/plots.py` → `report/figs/` | 8 figür |
| 6. Section 7 tabloları | `analysis/ocr_compare.py`, `analysis/tables.py` → `report/tables.md` | Tamam |
| Ek | `analysis/stress_test.py`, `analysis/walkthrough.py` | Dayanıklılık testi, aşama aşama görüntüler |
| 7. Hata analizi | – | Örnekler seçilecek |
| 8. Affine paragrafı | – | Yazılacak |

Ham sonuçlar `out/eval/` altındaki CSV'lerde; tablolar ve figürler bu CSV'lerden üretiliyor.

## Araştırdıklarım

**Eğiklik ölçüsü: homografiden kâğıt düzleminin eğimi**
- Düzlemsel bir nesne için H = K [r1 r2 t]. Birim kareden fotoğraftaki köşelere giden H bulunur, görüntü merkezi asal nokta kabul edilir.
- r1 ve r2 dik ve aynı uzunlukta olmalı; bu iki koşuldan odak uzaklığı f çözülür. Sonra K⁻¹h1 ve K⁻¹h2'nin vektörel çarpımı kâğıdın normalini verir. Eğiklik, normal ile kamera ekseni arasındaki açıdır.
- Neredeyse düz bakışta perspektif bilgisi neredeyse yok, f tahmini bozuluyor (kişisel sette `01`, `04`, `05`, `08`, `12`). Bu yüzden kişisel fotoğraflarda f, EXIF'teki 35 mm eşdeğer odak uzaklığından hesaplanıyor: f_px = f35 · uzun kenar / 36. Eğik karelerde tahmin EXIF ile uyuşuyor (`07`: tahmin 2694 px, EXIF 2688 px).
- Dev set'te gerçek açı arccos(cos tilt · cos pan) ile karşılaştırıldı: ortanca hata 0,02°. Tek istisna `dev_14`: f 4201 px tahmin ediliyor (diğerleri ~1900–2560), açı 18,0° yerine 35,4° çıkıyor. Bu kare eğiklik grafiklerinde yok.
- Diğer adaylar da ölçüldü (gerçek açıyla Spearman): kenar oranı 0,961, köşe açısı sapması 0,980, paralelkenardan sapma 0,968, homografi açısı 0,991.

**Izgara düzgünlüğü: projection profile**
- Gri tonlu "koyuluk" görüntüsünde (aydınlatma düzeltilmiş, ters çevrilmiş) hücre boyunun 0,4'ü uzunluğunda yatay ve dikey gri tonlu açma. İkili görüntü kullanılmadı, çünkü ince gri ızgara çizgileri global Otsu'da kopuk çıkıyordu.
- Profilden 10 + 10 tepe bulunur. Aralık sapması |aralık − hücre boyu|, düzlük ise her çizginin 3 şeritteki konum farkı.
- A3/A4'te fotoğrafın dışına taşan siyah bölgeler maskeleniyor, yoksa çizgi gibi görünüyorlar.

**Harris ve cornerSubPix**
- `blockSize` büyüdükçe Harris tepesi kalın çerçevenin içine kayıyor (Faz 2'deki ~9 px içe kayma).
- `cornerSubPix` köşeyi pencere içindeki gradyanlardan çözer; başlangıç noktası pencerede kaldığı sürece Harris tepesinin tam yeri önemini yitirir. Pencere çok küçükse köşeye ulaşamıyor, çok büyükse çerçevenin iç kenarı da hesaba giriyor.
- İki parametre birbirine bağlı: küçük blok tepeyi köşeye yaklaştırdığı için daha küçük pencere yetiyor.

**Tesseract**
- `image_to_data` her okumanın güven puanını veriyor.
- Fallback olmadan tek başına ölçülünce psm 10 ile 7, psm 8 ile 13 birebir aynı sonucu veriyor. LSTM motorunun tek karakter modunu tek satır gibi işlediğini düşündürüyor (gözlem, belgeyle doğrulanmadı).
- psm 8 daha çok hücre okuyor ama okuduğunda daha sık yanılıyor; psm 10 emin olmadığında cevap vermiyor.

## Review notu

**Sonuç (Section 7):**

| Set | Bulunan | Çöken | Köşe hatası | Hücre doğruluğu | Boş hücre F1 | Tam doğru | Süre |
|---|---|---|---|---|---|---|---|
| dev | 20/20 | 0 | %0,063 | 0,9994 | 0,9995 | 19/20 | 2,1 sn |
| kişisel | 12/12 | 0 | %0,111 | 1,0000 | 1,0000 | 12/12 | 2,3 sn |

Tek hata `dev_20`'nin (7, 5) hücresi: sayfa kıvrımı 7'yi hücrenin en üstüne itiyor, ağırlık merkezi orta bandın dışına düşüyor, hücre boş sanılıyor. Boş, gürültülü, içi boş kutulu, bulmacasız kırpılmış görüntüler, rakamsız sudoku ızgarası, 6×4 tablo ve XOX tahtası `not_found`.

**Stage 4: A3 / A4 / H4** (`dev_14` hariç):

| Set, eğiklik | Kare | Model | Uyum hatası (px) | Çizgi aralığı sapması (px) | Düzlük (px) | Hücre doğruluğu |
|---|---|---|---|---|---|---|
| kişisel, < 10° | 8 | A3 / A4 / H4 | 11,8 / 2,9 / 0 | 1,75 / 1,26 / 1,12 | 2,69 / 1,68 / 0,75 | 0,997 / 1,000 / 1,000 |
| kişisel, ≥ 24° | 4 | A3 / A4 / H4 | 166–391 / 34–63 / 0 | 12–21 / 12–16 / 0,5–1,8 | 9–12 / 7–13 / 0,7–1,4 | 0,51–0,72 / 0,56–0,70 / 1,000 |
| dev, < 10° | 5 | A3 / A4 / H4 | 25,4 / 6,4 / 0 | 4,43 / 1,61 / 0,68 | 5,80 / 2,86 / 0,13 | 0,960 / 1,000 / 1,000 |
| dev, 10–30° | 5 | A3 / A4 / H4 | 67,7 / 16,9 / 0 | 10,40 / 5,86 / 0,71 | 9,95 / 7,10 / 0,07 | 0,788 / 0,943 / 1,000 |
| dev, ≥ 30° | 9 | A3 / A4 / H4 | 163 / 41 / 0 | 15,81 / 12,99 / 0,85 | 9,56 / 8,58 / 0,20 | 0,661 / 0,634 / 0,999 |

- Düz karelerde üç model neredeyse aynı. Eğiklik arttıkça A3 ve A4 bozuluyor, H4 her açıda sabit.
- A4'ün uyum hatası A3'ünkünün yaklaşık dörtte biri, ama eğik karelerde OCR'ı kurtarmıyor: en küçük kareler hatayı köşelere dağıtıyor, ızgaranın içindeki aralık bozulmasını düzeltemiyor.
- A3 hatası gerçek açıyla Spearman 0,929, görüntüdeki paralelkenardan sapmayla 0,976 uyuşuyor. Hata açıya ve kameranın kâğıda uzaklığına birlikte bağlı (weak perspective, ΔZ/Z₀): `dev_18` 42°'de 211 px, daha eğik `dev_20` 55°'de 176 px.

**Bu fazda pipeline'da alınan kararlar:**

| Değişiklik | Neden | Ölçüm |
|---|---|---|
| Hücreler: ızgara çizgileri silinip hücreden %15 geniş pencereye bakılıyor | Gri çerçeveli bulmacada rakam hücre sınırını taşıyor, margin rakamın üstünü kesiyordu | Kişisel 7/12 → 10/12 tam doğru, dev değişmedi |
| OCR: 3 boyut × psm 10 ve 8 = 6 okuma; kurala uyan en yüksek güvenli cevap | Tesseract 3 temiz rakamı yanlış okuyordu, ama doğru cevabı daha yüksek güvenle de veriyordu | Kişisel 10/12 → 12/12, dev 19/20 → 20/20; süre ~1 sn → ~2,2 sn |
| Rakam kuralları (`digit_rules.py`): delik sayısı ve yeri, sol-üst ve sol-orta mürekkep | Tesseract'ın imkânsız cevaplarını elemek | 905 rakamın hiçbiri kendi kuralını bozmuyor; 3 hatalı cevabın üçü de kuralla reddediliyor. Bu veride sonucu güven puanı tek başına da düzeltiyor, kurallar güvenlik ağı |
| Harris `blockSize` 7 → 5 px | Tepe köşeye daha yakın (Harris tepesi hatası 8 karede %0,78 → %0,51) | 8 karede 3 px'in tepesi en isabetliydi (%0,40), ama 32 karede `s042401168_07`'de yanlış tepe seçti (%0,55); 5 px'te sorun yok |
| `cornerSubPix` penceresi 5 → 3 px | 5 px blokla daha küçük pencere yetiyor | Köşe hatası %0,168 → %0,081 (32 kare), en kötü %0,357 → %0,183 |
| Rakam parçaları arası mesafe sınırı (hücre boyunun 0,06'sı) | Yeni köşelerle kıvrık `dev_20`'de eğri çizgi kırıntıları rakama karışıyordu | `dev_20`'deki 3 hatanın 2'si düzeldi, başka bir şey değişmedi |
| 9×9 ızgara kontrolü (`grid_check.py`): beklenen 20 çizginin en az 11'i genişliğin %70'ini kaplamalı | Tablo ve XOX tahtası sudoku sanılıyordu, XOX `valid` çıkıyordu | Gerçek karelerin 31'inde 20/20, `dev_20`'de 13/20; tablo 6, XOX 8 → `not_found` |
| En az 17 dolu hücre, OCR'dan önce | Rakamsız ızgara `valid` çıkıyordu; 17, tek çözümlü sudoku için gereken en az ipucu | Boş ızgara `not_found`, OCR'a girmiyor |
| Sayım iki yönde yapılıp büyüğü alınıyor | Sayım yön düzeltmesinden önce çalıştığı için 90° dönmüş fotoğraflarda yan rakamlar sayılmıyordu (stress testinde 32'den 22'ye düşüş) | 90° ve 270°'de yine 32/32 |
| `--input` modunda da hata yakalama | Tek görüntüde hata olursa program çöküyordu | Olmayan ve bozuk dosyada `not_found` + `_error.txt` |
| Çıktı adları `_rect_A3/A4/H4.png` | README ile uyum | – |

**Denenip vazgeçilenler:**
- `cornerSubPix` öncesi Gaussian blur: 3 px pencerede kişisel set %0,111 → %0,104, dev set %0,063 → %0,090. Fark 0,1–0,2 piksel; pencere zaten yüzlerce gradyanı ortaladığı için eklenmedi.
- Band'i 0,15'ten küçültmek: `dev_20`'deki son hücreyi düzeltiyor ama kenardaki çizgi kırıntıları boş hücreleri "1" diye dolduruyor (`dev_05`, `s042401168_08`).
- Bulanık rakamlar için rakam başına yerel yeniden eşikleme (Otsu ve mürekkep-kâğıt arası oranlar): delikler açılmadı, eşik mürekkebe yaklaştıkça net görüntüler bozuldu (905 hücrede 1 → 5 hata). Bulanıklık deliğin içini çizgi kadar koyulaştırdığı için bilgi görüntüde yok.
- 6 okumada sadece çoğunluk oylaması: 905 hücrede 1 hata (`dev_18`'deki 5'te 4 okuma "3", 2 okuma "5"); en yüksek güven daha iyi.

**Dayanıklılık (stress testi, 32 kare):**

| Bozma | Bulunan | Tam doğru | Not |
|---|---|---|---|
| 90°, 180°, 270° döndürme | 32/32 | 31/32 | Yön düzeltmesi çalışıyor (bonus B4) |
| 45° döndürme | 32/32 | 32/32 | |
| %50 / %25 küçültme | 32 / 31 | 30 / 28 | OCR, birer hücre |
| JPEG 20 | 32/32 | 28/32 | OCR, birer hücre |
| Karartma (gamma 2,5) | 31/32 | 30/32 | `s042401168_12` bulunamıyor |
| Gürültü (σ 15) | 31/32 | 27/32 | OCR, birer hücre; `dev_19` bulunamıyor |
| Bulanıklık (σ 2 px, 1200 px ölçeğinde) | 27/32 | 14/32 | En zayıf nokta: delikler kapanıyor; `dev_13`'te yön 180° yanlış seçiliyor |

Hiçbir durumda çökme yok.

**Hata analizi için adaylar** (en az 4 seçilecek):
- `dev_20` sayfa kıvrımı: 3×3 dönüşümün sınırı, kalan tek hücre.
- `s042401168_09` / `_11` yön: köşe sırası rakamlara bakılarak düzeltiliyor.
- Kurallar öncesi Tesseract karışıklıkları (5→3, 9→2, 8→3): `report/figs/ocr_errors_before_rules.png`.
- Gri çerçeveli bulmacada kesilen rakamlar (3→2, 8→2, 7→boş, 5→4): çizgi silme ile çözüldü.
- Harris 3 px blok, `s042401168_07`'de yanlış tepe: küçük set yanıltıcı olabiliyor.
- Bulanıklık: kapanan delikler, denenen ve işe yaramayan yerel eşikleme.

**Bilinen sınırlar:**
- Eşikler ve kurallar aynı 32 kareye bakılarak seçildi; ayrı bir test seti yok. %100'e yakın sonuçlar "bu veride" diye okunmalı.
- `dev_14`'te odak tahmini başarısız; eğiklik ölçüsü düz ve neredeyse düz karelerde EXIF olmadan güvenilmez.
- Kişisel sette eğik kare sadece 4 (24°, 37°, 42°, 43°); aradaki açıları dev set dolduruyor.
- Süre görüntü başına ~2,2 sn (sınır 20 sn); en çok zamanı 6 okumalı OCR alıyor.
- CSV'ler `out/eval/` altında ve `.gitignore` yüzünden repoya girmiyor.
