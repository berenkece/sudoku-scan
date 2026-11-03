# Faz 0 – Kurulum ve veri

## Araştırdıklarım

### GitHub Classroom ve remote

- Classroom reposu davet linki kabul edilince oluşur.
- Link gelene kadar kendi private repomda çalışıyorum; sonra `git remote add classroom <url>` + `git push classroom main` ile geçmişiyle taşınır.

### venv ve Tesseract

- `python3 -m venv venv`, `source venv/bin/activate`
- `pytesseract` sadece arayüz; Tesseract ayrıca kurulur: `brew install tesseract`, kontrol `tesseract --version`.

### Dev set formatı

- `images/dev_XX.jpg` ↔ `annotations/dev_XX.json` (Appendix A formatı).
- Köşe = kalın dış çerçevenin dış köşesi.
- `index.csv`: görüntü boyutu, koşullar, gerçek açılar (`tilt/pan/roll_deg`), `side_ratio`.

## Review notu

- Repo, klasör yapısı, ortam ve dev set hazır. Faz 0 tamam.
- Kişisel fotoğraflar Faz 2'nin sonuna taşındı.
