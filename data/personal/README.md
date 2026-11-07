# Kişisel fotoğraflar

HEIC dosyaları telefondan gelen orijinallerdir, adları değiştirilmedi. JPEG'ler bunlardan `analysis/convert_heic.py` ile üretildi.

| Orijinal | JPEG | Kaynak | Koşul |
|---|---|---|---|
| `IMG_5307.HEIC` | `s042401168_01.jpg` | Sudoku Mega | C1 |
| `IMG_5308.HEIC` | `s042401168_02.jpg` | Sudoku Mega | C2 |
| `IMG_5309.HEIC` | `s042401168_03.jpg` | Sudoku Mega | C3 |
| `IMG_5310.HEIC` | `s042401168_04.jpg` | Sudoku Mega | C4 |
| `IMG_5311.HEIC` | `s042401168_05.jpg` | SUDOKU 9x9 Medium | C1 |
| `IMG_5312.HEIC` | `s042401168_06.jpg` | SUDOKU 9x9 Medium | C2 |
| `IMG_5313.HEIC` | `s042401168_07.jpg` | SUDOKU 9x9 Medium | C2 (40°'den fazla, ölçülmedi) |
| `IMG_5314.HEIC` | `s042401168_08.jpg` | SUDOKU 9x9 Medium | C5 |
| `IMG_5315.HEIC` | `s042401168_09.jpg` | Gri çerçeveli | C1 |
| `IMG_5316.HEIC` | `s042401168_10.jpg` | Gri çerçeveli | C2 |
| `IMG_5317.HEIC` | `s042401168_11.jpg` | Gri çerçeveli | C3 + C4 |
| `IMG_5318.HEIC` | `s042401168_12.jpg` | Gri çerçeveli | C5 |

## Anotasyon

`annotations/` içindeki dosyalar dev set ile aynı kurala göre hazırlandı:

- **corners:** Bulmacanın TL, TR, BR, BL köşeleri, orijinal görüntü pikselinde. Köşe, kalın dış çerçevenin **dış** köşesidir, yani siyah çerçevenin arka planla birleştiği nokta. Sol üst pikselin merkezi (0, 0) kabul edilir.
- **Yön:** TL, görüntünün değil bulmacanın sol üst köşesidir. `s042401168_09` ve `s042401168_11`'de bulmaca yan durduğu için bu ikisi farklıdır.
- **Görüntü koordinatları:** EXIF Orientation uygulanmış görüntüye göredir, `cv2.imread` varsayılan olarak bunu yapar. `s042401168_01`–`09` arası kareler EXIF 6 ile kayıtlı.
- **grid:** Üst satır ilk, boş hücre `0`.

Köşeler `analysis/annotate.py` ile elle, 8× yakınlaştırılmış görünümde işaretlendi.
