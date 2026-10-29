"""Project paths independent of the process working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PDF = PROJECT_ROOT / "input" / "pdf"
INPUT_OCR = INPUT_PDF / "ocr"
OUTPUT = PROJECT_ROOT / "output"
