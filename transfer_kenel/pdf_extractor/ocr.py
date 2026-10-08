"""Lazy, entirely local Chinese/English OCR. Models ship with RapidOCR."""
from pathlib import Path


class LocalOCR:
    def __init__(self):
        self._engine = None

    def recognize(self, path: Path):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise RuntimeError("OCR 依赖未安装，请运行 pip install rapidocr-onnxruntime 或选择关闭 OCR。") from exc
            self._engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
        data, _ = self._engine(str(path))
        return [{"bbox": [min(p[0] for p in quad), min(p[1] for p in quad), max(p[0] for p in quad), max(p[1] for p in quad)],
                 "text": str(text), "confidence": float(score)} for quad, text, score in (data or [])]


def detect_grid_tables(path, lines):
    """Recover only regular ruled grids; retain uncertain layouts as OCR text."""
    import cv2
    import numpy as np
    # imdecode handles non-ASCII paths on Windows.
    gray = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return [], set()
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    height, width = gray.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 30), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 30))))
    contours, _ = cv2.findContours(horizontal | vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def centers(values):
        groups = []
        for v in values:
            if not groups or v - groups[-1][-1] > 3:
                groups.append([int(v)])
            else:
                groups[-1].append(int(v))
        return [round(sum(group) / len(group)) for group in groups]

    tables, consumed = [], set()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 100 or h < 60:
            continue
        xs = centers(np.flatnonzero((vertical[y:y+h, x:x+w] > 0).sum(axis=0) >= h * .8))
        ys = centers(np.flatnonzero((horizontal[y:y+h, x:x+w] > 0).sum(axis=1) >= w * .8))
        if len(xs) < 3 or len(ys) < 3 or len(xs) > 40 or len(ys) > 150:
            continue
        rows = [[[] for _ in range(len(xs) - 1)] for _ in range(len(ys) - 1)]
        used, confidence = set(), []
        for i, line in enumerate(lines):
            x0, y0, x1, y1 = line["bbox"]
            cx, cy = (x0 + x1) / 2 - x, (y0 + y1) / 2 - y
            col = next((j for j in range(len(xs) - 1) if xs[j] <= cx < xs[j+1]), None)
            row = next((j for j in range(len(ys) - 1) if ys[j] <= cy < ys[j+1]), None)
            if row is not None and col is not None:
                rows[row][col].append((y0, x0, line["text"]))
                used.add(i)
                confidence.append(line["confidence"])
        occupied = sum(bool(cell) for row in rows for cell in row)
        if occupied < max(3, len(rows) * len(rows[0]) * .4):
            continue
        tables.append({"bbox": [x, y, x+w, y+h], "rows": [["\n".join(text for _, _, text in sorted(cell)) for cell in row] for row in rows],
                       "confidence": sum(confidence) / len(confidence)})
        consumed.update(used)
    return tables, consumed
