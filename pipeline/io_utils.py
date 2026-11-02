import json
from pathlib import Path


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data: dict) -> None:
    ensure_output_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    ensure_output_dir(path.parent)
    path.write_text(text, encoding="utf-8")
