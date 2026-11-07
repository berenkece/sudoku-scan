# Faz 3 – Rektifikasyon ve çıktılar (Stage 3)

## Araştırdıklarım

**Homojen koordinatlar**
- (x, y) → (x, y, 1). Öteleme de matris çarpımına girer, dönüşümler çarpılarak art arda bağlanabilir.
- (x, y, w) ile (kx, ky, kw) aynı noktadır, gerçek nokta (x/w, y/w). w = 0 sonsuzdaki nokta, yani bir yön; kaçış noktaları buradan gelir.

**Affine (2×3, 6 DoF)**
- 6 bilinmeyen var, her nokta çifti 2 denklem verir, yani 3 nokta yeter (aynı doğru üzerinde olmamalı).
- 3×3 yazılınca alt satır [0 0 1], w hep 1 kalır. Paralellik ve bir doğru üzerindeki oranlar korunur. Kare → paralelkenar. Kaçış noktası oluşturamaz.

**Homografi (3×3, 8 DoF)**
- 9 eleman var ama H ile kH aynı dönüşüm, bu yüzden 8 DoF. 4 nokta gerekir (hiçbir üçü aynı doğru üzerinde olmamalı).
- h31, h32 paydayı noktaya bağlı yapar: paralel doğrular yakınsayabilir, kare herhangi bir dörtgene gidebilir.
- Düzlemsel bir nesnenin perspektif görüntüsü tam olarak bir homografi; affine ise bunun yaklaşıklığı.

**A3 dördüncü köşe hatası**
- A3, TL/TR/BL ile kurulur. BR aynı matristen geçirilir, hedef BR'ye Öklid uzaklığı hesaplanır.
- Affine, fotoğraftaki dörtgenin paralelkenar olduğunu varsayar (BR = TR + BL − TL). Perspektifte uzak kenar kısalır, dörtgen paralelkenardan sapar ve bu sapmanın tamamı BR'de görünür. Bu yüzden hata eğiklikle büyür.
- Hata, kullandığım hedef konvansiyonuna göre ölçülmeli: hedef köşeler −0.5 … 449.5 ise BR hedefi 449.5. Aksi halde sabit ~0.7 px hata görülür.

**A4: en küçük kareler**
- 4 nokta 8 denklem, 6 bilinmeyen: aşırı belirli sistem. Hata 4 köşeye dağılır.
- RMS = √( (1/4) · Σ ‖A·pᵢ − qᵢ‖² )
- `estimateAffine2D` varsayılan olarak RANSAC kullanır. 4 noktanın da inlier olduğunu maskeden kontrol et ve `np.linalg.lstsq` ile karşılaştırarak doğrula.

**OpenCV notları**
- `getAffineTransform` / `getPerspectiveTransform`: float32, (3,2) / (4,2).
- `estimateAffine2D`: (matris, inlier maskesi) döndürür.
- `warpAffine` / `warpPerspective`: dsize = (genişlik, yükseklik).
- `perspectiveTransform`: girdi şekli (N, 1, 2), bölmeyi kendisi yapar.
- Daha fazla nokta için `findHomography` (savunma sorusu).

## Review notu

**Sonuç:** Dev set'te 20 karenin hepsi düzeltildi (`analysis/eval_rectify.py`, pipeline'ın köşeleriyle). A3 köşe hatası 15,9–260,8 px, A4 RMS 4,1–54,6 px. H4 dört köşeyi en fazla 2,8e-5 px sapmayla eşliyor. A4'te 20 karenin hepsinde 4 köşe de inlier ve `np.linalg.lstsq` ile fark en fazla 4e-12, yani `estimateAffine2D` gerçekten en küçük kareler çözümü veriyor. Hedef köşeler −0.5 … 449.5; `dev_07`'de 0 … 450 ile de denendi, A3 hatası ve RMS değişmedi. Kıvrımlı `dev_20`'de H4 sonrası çizgiler hâlâ dalgalı.

**Kişisel set:** 12 kare `analysis/annotate.py` ile işaretlendi. Pipeline'ın köşe hatası bu anotasyonlara göre ortalama %0,17, en kötü %0,36 (`s042401168_08`), yani dev set ile aynı seviyede. `s042401168_09` ve `_11`'de köşeler doğru bulunuyor (%0,18 ve %0,15), ama sıra bir köşe kaydığı için doğrudan ölçülünce hata %100 çıkıyor.

**Anotasyonda yaşananlar:**
- `10`–`12`'de ilk seferde kalın çerçevenin iç köşesi tıklandı ve hata %2,4–2,7 çıktı. Dış köşeyle yeniden yapıldı. Kural araca ve `data/personal/README.md`'ye yazıldı.
- `s042401168_10`'un sağ alt köşesi gölgede kaldığı için bir kez daha işaretlendi (hata %0,28 → %0,20).
- `06` ve `07`'de grid'in 8. satırı eksik yazılmıştı, `05` ile karşılaştırılarak düzeltildi.
- Köşeler çapraz sırada tıklanınca önizleme bozuluyordu. Araç artık köşeleri merkez etrafında açıya göre sıralıyor; doğru başlangıç köşesi önizlemede `o` tuşuyla seçiliyor.

- **Yan duran kareler (`s042401168_09`, `_11`):** rakamlar yan çıkıyor ve köşe sırası bulmacanın gerçek TL'sinden başlamıyor. Bu yüzden JSON'daki köşeler ve matrisler de yanlış yönde. Kontrol edilecek: EXIF Orientation mı (pipeline ile anotasyon aracı aynı şekilde okuyor mu), yoksa fotoğraf gerçekten 90° dönük mü? Gerçekten dönükse şartnamenin "roughly upright, ±30°" kapsamı dışında kalıyor (90°+ dönüş bonus B4). Anotasyonda TL gerçek sol üst olarak işaretlenecek. Bu iki kare Failure analysis için örnek olacak; C1–C5 kapsamı bunlara dayanmamalı.
- **EXIF kontrolünün sonucu ve anotasyon aracı:** `s042401168_09`'da EXIF Orientation 6. Ham piksellerde bulmaca düz, EXIF uygulanınca yan duruyor. `s042401168_11`'de EXIF 1, yani fotoğrafın kendisi dönük. EXIF'i yok saymak çözüm değil, çünkü `01–08` de EXIF 6 ile kayıtlı ve ancak EXIF uygulanınca düz görünüyor. Anotasyon koordinatlarının pipeline ile aynı olması için `analysis/annotate.py` yazıldı: görüntüyü pipeline'daki gibi `load_image` (`cv2.imread`) ile açıyor, köşeler 8× yakınlaştırılmış görünümde elle tıklanıyor ve kontrol için H4 önizlemesi gösteriliyor. Araç Harris veya `locate_corners` kullanmıyor, böylece anotasyon pipeline'ın sonucundan bağımsız kalıyor. Pipeline kodu bu araç için değiştirilmedi.
- **Eğiklik ölçüsü:** A3 hatası eğiklikle büyüyor ama `side_ratio` ile birebir gitmiyor (`dev_20`: oran 1,64 → 177 px; `dev_07`: oran 1,57 → 261 px). Olası nedenler: tek eksene bakan oran tilt + pan birleşimini görmüyor, daralmanın yönü bırakılan köşeye göre önemli. Kendi ölçümü Faz 5'te tanımlayacağım (Stage 4).
- **JSON alanları:** `grid` ve `verdict` Faz 4'e kadar `null`; Faz 4 sonunda `verdict` her zaman valid / invalid / not_found olacak. `conditions` pipeline'ın bilebileceği bir şey değil, result dosyasında boş liste kalıyor. H4 sapması şemada yok, JSON'a eklenmedi. Ölçüm scriptinde kontrol ediliyor, rapora bir cümle olarak girecek.
