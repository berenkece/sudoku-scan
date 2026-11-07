# Faz 2 raporu: kod ne yapıyor?

Bu dosya, köşe bulma kodunun çalışmasını adım adım anlatır. Sayılar 07.11.2026 tarihli çalıştırmalardan alındı.

## 1. Program ne yapıyor?

Program bir fotoğraf alır, içindeki bulmacanın dış 4 köşesini bulur ve TL, TR, BR, BL sırasıyla, orijinal piksel koordinatında JSON dosyasına yazar. Bulmaca bulamazsa durmaz, `not_found` yazar.

```
fotoğraf
  1. load_image        dosyayı oku
  2. preprocess        küçült, griye çevir, bulanıklaştır, ışığı düzelt
  3. harris_response   her piksel için köşe puanı (R haritası) hesapla
  4. find_quads        görüntüdeki tüm dörtgenleri bul
  5. select_quad       bulmaca olan dörtgeni seç
  6. order_corners     köşeleri TL, TR, BR, BL sırasına koy
  7. snap_to_harris    her köşeyi yakındaki Harris tepesine taşı
  8. to_original       küçültülmüş koordinatı orijinale çevir
  9. refine_subpix     orijinal görüntüde köşeyi hassaslaştır
 10. save_result       <ad>_result.json dosyasını yaz
```

5\. adımda dörtgen bulunmazsa veya 7. adımda bir köşenin çevresinde Harris tepesi yoksa sonuç `not_found` olur.

**Temel fikir:** Dörtgen, köşenin **nerede aranacağını** söyler. Köşenin **kesin yerini** Harris ve `cornerSubPix` belirler, çünkü şartname son köşelerin Harris'ten gelmesini istiyor.

## 2. Örnek: `dev_01`

| Adım | Sonuç |
|---|---|
| Fotoğraf | 2400 × 1800 piksel |
| `preprocess` | 1200 × 900 gri görüntü, `scale = 0.5` |
| `find_quads` | 74 dörtgen |
| `select_quad` | Görüntünün %27,4'ü kadar, içinde 14 küçük dörtgen olan dörtgen |
| Son TL köşesi | `[576,5, 365,7]` (gerçek köşe: `[575,3, 364,8]`) |
| `corner_error` | %0,16 |

## 3. Görüntüyü hazırlama: `pipeline/preprocess.py`

### `preprocess(image)`
- **Girdi:** Renkli fotoğraf (BGR).
- **Çıktı:** `gray` (işlenmiş gri görüntü) ve `scale` (küçültme oranı).
- **Ne yapıyor:** Aşağıdaki dört işlemi sırayla uygular.

### `resize_long_side(image, target_long_side)`
Uzun kenarı 1200 piksele indirir ve oranı döndürür. Görüntü zaten küçükse dokunmaz, oran 1,0 olur. Böylece fotoğraf 1600 ya da 5712 piksel de olsa sonraki adımlar aynı boyutta çalışır.

### `to_gray(image)`
Renkli görüntüyü griye çevirir.

### `denoise(gray)`
Gauss bulanıklığıyla gürültüyü azaltır. Pencere, uzun kenarın %0,5'i kadardır (7 px).

### `normalize_illumination(gray)`
Gölgeyi düzeltir. Görüntüyü çok bulanık haliyle böler (pencere 145 px). Bulanık hali arka plan ışığını gösterir, bölünce kâğıt her yerde beyaza yaklaşır, koyu çizgiler ise koyu kalır.

### `odd_kernel(size, ratio)`
Boyutun belli bir oranını alır ve tek sayıya çevirir, en az 3 döndürür. OpenCV'nin bazı pencereleri tek sayı olmak zorundadır.

## 4. Köşe bulma: `pipeline/corners.py`

### `harris_response(gray)`
- **Girdi:** İşlenmiş gri görüntü.
- **Çıktı:** Görüntüyle aynı boyutta R haritası.
- **Ne yapıyor:** `cv2.cornerHarris` ile her piksele köşe puanı verir. R pozitif ve büyükse köşe, negatifse kenar, sıfıra yakınsa düz bölgedir.
- **Parametreler:** Pencere 7 px, Sobel 3, k = 0,04.

### `find_quads(gray)`
- **Girdi:** İşlenmiş gri görüntü.
- **Çıktı:** Dörtgenlerin listesi. Her biri 4 × 2 boyutunda köşe dizisidir.
- **Ne yapıyor:**
  1. **Yerel eşik:** `adaptiveThreshold` her pikseli 31 × 31'lik çevresinin ağırlıklı ortalamasıyla karşılaştırır. Pikselin değeri ortalama eksi 10'dan büyük değilse beyaz, büyükse siyah olur. Yani koyu çizgiler beyaz, kâğıt siyah görünür.
  2. **Konturlar:** `findContours` beyaz şekillerin sınırlarını bulur. `RETR_LIST` iç içe şekilleri de getirir, bu önemli çünkü bulmaca kâğıdın içindedir.
  3. **Sadeleştirme:** `approxPolyDP` her sınırı daha az köşeli bir şekle indirir. Tolerans, çevrenin %2'sidir.
  4. **Eleme:** Yalnızca tam 4 köşeli ve dışbükey olanlar tutulur.

### `count_inside(big, quads)`
- **Girdi:** Bir büyük dörtgen ve tüm dörtgenler.
- **Çıktı:** Büyük dörtgenin içindeki küçük dörtgen sayısı.
- **Ne yapıyor:** Her dörtgen için iki şeye bakar:
  1. **Boyutu uygun mu?** Alanı büyük dörtgenin %0,3'ü ile %50'si arasında olmalı. Daha küçükler gürültüdür. Daha büyükler sayılmaz, böylece dörtgen kendini ya da neredeyse aynı büyüklükteki bir kopyasını saymaz.
  2. **İçinde mi?** Dört köşesinin ortalaması olan merkez noktası `pointPolygonTest` ile kontrol edilir.

### `select_quad(quads, image_area)`
- **Girdi:** Tüm dörtgenler ve görüntünün alanı.
- **Çıktı:** Bulmaca olarak seçilen dörtgen ya da `None`.
- **Ne yapıyor:** Her dörtgen için:
  1. Alanı görüntünün %2'sinden küçükse geçer.
  2. İçinde 3'ten az küçük dörtgen varsa geçer.
  3. Kalanların **en küçüğünü** tutar.
- **Neden en küçüğü?** Kâğıt ve gri çerçeve de aynı küçük dörtgenleri içerir ama bulmacadan büyüktür.

### `order_corners(points)`
- **Girdi:** Karışık sırada 4 nokta.
- **Çıktı:** TL, TR, BR, BL sırasında 4 nokta.
- **Ne yapıyor:**
  1. Dört noktanın merkezini bulur.
  2. `arctan2` ile her noktanın merkeze göre açısını hesaplar.
  3. Açıya göre sıralar. Görüntüde y aşağı arttığı için bu sıra saat yönüdür.
  4. `x + y` değeri en küçük olan noktayı TL sayar ve `np.roll` ile başa alır.

### `snap_to_harris(response, anchors)`
- **Girdi:** R haritası ve dörtgenin sıralı köşeleri.
- **Çıktı:** Harris'ten gelen 4 nokta (küçültülmüş koordinat) ya da `None`.
- **Ne yapıyor:** Her köşe için:
  1. Çevresinde en fazla 13 × 13'lük bir pencere keser (her yöne 6 px).
  2. R değeri penceredeki en yüksek değerin %10'unu geçen noktaları güçlü sayar.
  3. `dilate` ile yerel tepeleri, yani komşularından düşük olmayan noktaları bulur.
  4. Hem güçlü hem tepe olan noktalardan köşeye **en yakınını** seçer.
  5. Hiç nokta yoksa `None` döndürür.
- **Neden en güçlüsü değil?** Kalın çerçevede en güçlü tepe bazen iç köşe oluyor. Dörtgen köşesi dış kenarda olduğu için en yakını dış köşedir.

### `to_original(points, scale)`
Küçültülmüş koordinatı orijinale çevirir: `(p + 0.5) / s − 0.5`. `0.5`'ler piksel merkezlerini hesaba katar.

### `to_working(points, scale)`
`to_original`'ın tersidir. Köşeleri küçültülmüş görsele çizmek için kullanılır.

### `refine_subpix(gray, points, scale)`
- **Girdi:** Orijinal gri görüntü, orijinal koordinatta 4 nokta ve oran.
- **Çıktı:** Ondalıklı hassasiyette 4 nokta.
- **Ne yapıyor:** `cv2.cornerSubPix` çevredeki gradyanlara bakarak noktayı gerçek köşeye doğru taşır. En fazla 40 tur döner veya hareket 0,01 px'in altına inince durur.
- **Pencere:** Küçültülmüş ölçekte 5 px, orijinalde `5 / scale`. OpenCV bu değeri yarıçap gibi kullandığı için `scale = 0.5` iken yarıçap 10, pencere 21 × 21 olur.
- **Neden gerekli?** Harris tepesi kalın çizgide içe doğru kayıyor, bu adım bunu düzeltiyor.

### `locate_corners(image, pre, response)`
Tüm akışı birleştirir: `find_quads` → `select_quad` → `order_corners` → `snap_to_harris` → `to_original` → `refine_subpix`. Son 4 köşeyi ya da `None` döndürür.

### Son akışta kullanılmayanlar
- **`find_candidates(response)`:** Eşik ve NMS ile Harris aday listesi çıkarır, çift adayları `connectedComponentsWithStats` ile birleştirir. R haritası `R_MIN`'in altındaysa boş liste döner. Artık yalnızca hata ayıklama görselinde ve `diagnose_harris.py`'de kullanılıyor.
- **`has_corner_support(quad, candidates)`:** Dörtgenin her köşesinin 6 px yakınında aday var mı diye bakar. Doğru dörtgenleri de elediği için hiçbir yerde kullanılmıyor.

## 5. Dosya ve görsel işleri

### `pipeline/io_utils.py`
- **`load_image(path)`:** Görüntüyü okur, okuyamazsa `FileNotFoundError` verir.
- **`save_image(path, image)`:** Klasörü oluşturur ve görüntüyü kaydeder, yazamazsa `OSError` verir.
- **`load_annotation(path)`:** Anotasyon dosyasından gerçek köşeleri ve ızgarayı okur.
- **`save_result(path, image_name, corners, runtime_s)`:** Şartnamedeki anahtarlarla sonuç JSON'unu yazar. Köşe yoksa `corners: null` ve `verdict: "not_found"` yazar. `grid`, `A3`, `A4`, `H4` ve iki hata alanı Faz 3-4'e kadar `null` kalır.

### `pipeline/visualize.py`
**`draw_harris(gray, response, candidates, corners=None)`:** Hata ayıklama görselini üretir. R haritasını renkli ısı haritası olarak griyle karıştırır, adayları yeşil daire çizer, seçilen köşeleri kırmızı daire ve TL, TR, BR, BL etiketiyle gösterir.

### `pipeline/metrics.py`
**`corner_error(pred, gt)`:** Tahmin ile gerçek köşeler arasındaki ortalama uzaklığı, ızgaranın ortalama kenar uzunluğuna böler ve yüzde verir. Böylece görüntü boyutundan bağımsız bir hata elde edilir.

### `sudoku_scan.py`
- **`parse_args()`:** `--input`, `--batch` ve `--outdir` seçeneklerini okur.
- **`list_images(folder)`:** Klasördeki `.jpg`, `.jpeg` ve `.png` dosyalarını sıralı listeler.
- **`process_image(image_path, outdir)`:** Bir görüntü için tüm akışı çalıştırır, görseli ve JSON'u yazar, süreyi ölçer.
- **`run_batch(folder, outdir)`:** Klasördeki her görüntüyü işler. Biri hata verirse diğerlerine devam eder.
- **`main()`:** Seçeneğe göre tek görüntü, klasör ya da varsayılan `dev_01` çalıştırır.

### `analysis/`
- **`eval_corners.py`:** Dev set'te her görüntü için `predict_corners` çalıştırır, hatayı gerçek köşelerle karşılaştırır, ortalama, en kötü 5 ve `not_found` listesini yazar.
- **`diagnose_harris.py`:** Her köşe için dörtgen, Harris adayı ve gerçek köşe arasındaki uzaklıkları yazar. İçe kaymayı bu betikle gördük.
- **`convert_heic.py`:** `data/personal` içindeki HEIC dosyalarını `sips` ile `s042401168_XX.jpg` yapar, var olan dosyanın üstüne yazmaz.

## 6. Neden böyle yaptık?

| Karar | Neden | Kanıt |
|---|---|---|
| Otsu yerine yerel eşik | Otsu `dev_10`'da çerçevenin altını kaybetti, dörtgen kapanmadı | Otsu ile en yakın dörtgen 253 px uzakta; yerel eşikle 20 karenin hepsinde en fazla 1,5 px (küçültülmüş) |
| İçinde küçük dörtgen olan | Bulmaca iç dörtgenlerle dolu, Notes kutusunun içi boş | Bulmacada 3–53 iç dörtgen; incelenen diğer büyük dörtgenlerde 0–1 |
| Bunların en küçüğü | Kâğıt ve gri çerçeve de aynı dörtgenleri içerir | `s042401168_09`: üç dörtgenin üçü de 59 içeriyor; doğrusu en küçüğü |
| En az 3 iç dörtgen ve %2 alan | Tek eşik yetmedi: 3 ile `s042401168_05`'te çok küçük bir şekil seçildi, 4 ile `dev_19`'da şekil kalmadı | En küçük bulmaca görüntünün %10,6'sı |
| Son köşeler Harris'ten | Şartnamenin 5. bölümü zorunlu kılıyor | Şartname repoda yok, yapıştırılan metinden alındı |
| Genel liste yerine yerel arama | Genel eşik bazı köşeleri listeye sokmuyor | `s042401168_07`, `10`, `12`'de en yakın aday 36–65 px uzakta |
| Pencerede en yakın tepe | En güçlü tepe kalın çerçevede iç köşe olabiliyor, "en uzak" ise gürültüye gidebilir | `s042401168_10`: en büyük kayma 9,2 px'ten 3,4 px'e indi |
| `cornerSubPix` | Harris tepesi sistematik olarak içe kayıyor | Bölüm 7'deki tablolar |
| Harris desteği eleme kuralı olmadı | Doğru dörtgenleri eliyordu | Dev set'te 20/20, kişisel karelerde 9/12 geçti |

## 7. Ölçümler

**Son sonuç (dev set, 20 kare):** Ortalama %0,17. En kötü 5: `dev_10` %0,28, `dev_04` %0,27, `dev_11` %0,26, `dev_08` %0,24, `dev_13` %0,21. Kıvrımlı (`dev_19`, `dev_20`), dönmüş (`dev_12`) ve yoğun metinli (`dev_15`) kareler en kötü 5'e girmedi.

**Kişisel kareler (12):** Hepsinde köşe bulundu. Anotasyon olmadığı için hata yüzdesi yok. Son köşeler dörtgen köşesinden en fazla 3,4 px (küçültülmüş) uzakta.

**Bulmacasız görüntüler (4):** Boş, gürültülü, içi boş dikdörtgenli ve bulmacasız kırpılmış görüntünün hepsi `not_found` verdi.

**Köşe kaynaklarının karşılaştırması**

| Köşe kaynağı | Ortalama | En kötü |
|---|---|---|
| Dörtgen köşesi (son sonuç olamaz) | %0,18 | %0,31 |
| En yakın Harris adayı, `cornerSubPix` yok | %0,74 | %0,94 |
| Yerel Harris + `cornerSubPix` (son hali) | %0,17 | %0,28 |

**Harris neden kötü çıktı? (80 köşe, orijinal px)**

| Uzaklık | Ortalama | En büyük |
|---|---|---|
| Dörtgen köşesi – gerçek köşe | 1,7 | 5,2 |
| Harris adayı – gerçek köşe | 7,0 | 13,4 |
| Dörtgen köşesi – Harris adayı | 6,3 | 9,6 |

Hata her köşede aşağı yukarı aynı ve hücre boyu kadar bir sıçrama yok. Sebep yanlış aday değil, sistematik içe kayma.

**`cornerSubPix` pencere boyutu (küçültülmüş px)**

| Pencere | Ortalama | En kötü |
|---|---|---|
| yok | %0,74 | %0,94 |
| 3 | %0,36 | %0,91 |
| 4 | %0,13 | %0,39 |
| **5 (seçilen)** | %0,17 | %0,28 |
| 6 | %0,24 | %0,34 |
| 8 | %0,33 | %0,42 |

4 daha düşük ortalama, 5 daha düşük en kötü hata veriyor. En kötü durumu öne aldığımız için 5 seçildi.

## 8. Bilinen sınırlar

- **Ayrı test seti yok:** Eşikler aynı karelere bakılarak seçildi, bu yüzden %0,17 iyimser olabilir.
- **Kişisel karelerde doğruluk ölçülmedi:** Anotasyon yok. Son haliyle yakından yalnızca `s042401168_07` ve `10` incelendi.
- **Dönüş:** `order_corners` 45°'den fazla dönmüş görüntüde gerçek sol üstü bilemez. `s042401168_09` ve `s042401168_11`'de ızgara yan çıkacak.
- **İkinci ızgara:** `s042401168_12`'deki kitap seçilmedi, ama iki bulmaca da kuralı geçerse küçük olan seçilir. Bu test edilmedi.
- **Sudoku olmayan ızgaralar:** Tek başına bir tablo denenmedi, `not_found` vermeyebilir.
- **JSON:** `conditions: []` ve `null` alanlar şartnamenin açık bıraktığı yerlerde varsayılan seçim.

## 9. Üç dakikalık anlatım

Aşağıdaki metni kendi cümlelerine çevirerek prova et.

> Amacım fotoğraftaki bulmacanın dış dört köşesini bulmaktı. Zorluk, görüntüde başka dörtgenlerin de olması: kâğıdın kenarı, not kutuları, tablolar, hatta başka bir sudoku.
>
> Önce görüntüyü küçültüp griye çeviriyorum ve ışığı düzeltiyorum. Sonra yerel eşikle çizgileri çıkarıyorum. Önce Otsu denedim ama tek eşik kullandığı için `dev_10`'da çerçevenin altı kayboldu, dörtgen kapanmadı. Yerel eşik her pikseli kendi çevresiyle karşılaştırdığı için bu sorunu çözdü.
>
> Sonra konturları bulup yalnızca dört köşeli, dışbükey olanları tutuyorum. Seçim kuralım şu: bulmaca, içinde çok sayıda küçük dörtgen barındıran en küçük dörtgendir. Not kutusunun içi boştur, o elenir. Kâğıt ve gri çerçeve de aynı küçük dörtgenleri içerir ama bulmacadan büyüktür, en küçüğü seçtiğim için onlar da elenir. Çok küçük gürültü şekilleri seçilmesin diye görüntünün en az yüzde ikisi kadar alan da istiyorum.
>
> Şartnameye göre son köşeler Harris'ten gelmeli, bu yüzden dörtgen köşeleri son cevap değil, sadece nerede arayacağımı gösteriyor. Her köşenin çevresindeki küçük pencerede güçlü Harris tepelerine bakıp en yakınını alıyorum. En yakını almamın sebebi, en güçlü tepenin kalın çerçevede bazen iç köşe olması.
>
> Harris tepesi kalın çizgide içe doğru kayıyor. Bunu köşe köşe ölçtüm: hata her köşede benzerdi, yani yanlış aday değil, sistematik kaymaydı. Bu yüzden orijinal görüntüde `cornerSubPix` kullanıyorum. Hata yüzde 0,74'ten yüzde 0,17'ye indi.
>
> Sıralamayı da merkez etrafındaki açıyla yapıyorum, `x + y` en küçük olan köşeyi başa alıyorum. Bulmaca yoksa program durmuyor, `not_found` yazıyor.
>
> Sonuç: dev set'te yirmi karenin yirmisi, ortalama yüzde 0,17, en kötü yüzde 0,28. Kendi on iki fotoğrafımda da köşeler bulundu ama onları ölçemedim, çünkü anotasyon yok. Sınırlarım da var: eşikleri aynı karelerde seçtim, ayrı test setim yok. Ayrıca 45 dereceden fazla dönmüş bir bulmacada hangi köşenin gerçek sol üst olduğunu köşelerden anlayamıyorum.
