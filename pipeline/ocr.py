import numpy as np
import pytesseract


def recognize_cell(cell: np.ndarray) -> int | None:
    """Prepare one cell and recognize a single digit with Tesseract."""
    text = pytesseract.image_to_string(
        cell,
        config="--psm 10 -c tessedit_char_whitelist=123456789",
    ).strip()
    return int(text) if len(text) == 1 and text.isdigit() else None
