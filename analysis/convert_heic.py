import subprocess
from pathlib import Path

PERSONAL_DIR = Path("data/personal")
JPEG_QUALITY = "95"
PHOTO_PREFIX = "s042401168"

def convert_all():
    heics = sorted(p for p in PERSONAL_DIR.iterdir() if p.suffix.lower() == ".heic")

    for i, heic in enumerate(heics, start=1):
        jpg = PERSONAL_DIR / f"{PHOTO_PREFIX}_{i:02d}.jpg"
        if jpg.exists():
            print(f"{jpg.name} zaten var, atlandı")
            continue

        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", JPEG_QUALITY, str(heic), "--out", str(jpg)],
            check=True,
            capture_output=True,
        )
        print(f"{heic.name} -> {jpg.name}")


if __name__ == "__main__":
    convert_all()
