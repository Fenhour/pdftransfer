"""Map local PP-StructureV3 results into the reading extractor's content model.

Coordinates and reference crops remain tied to the rendered source page. Font
size/color below are bounded visual observations, never claims of font recovery.
"""

from .usage import tracked
from html.parser import HTMLParser
import math
from statistics import median

import numpy as np
import pymupdf as fitz

from .layout import inside
from .model import Block


class _TableParser(HTMLParser):
    """Read only table topology/text, rather than publishing model HTML verbatim."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.row = -1
        self.cells = []
        self.current = None
        self.occupied = set()
        self.column = 0
        self.has_table = False
        self.header_rows = 0
        self.ignored = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"script", "style"}:
            self.ignored += 1
        if self.ignored:
            return
        if tag == "table":
            if self.has_table:
                raise ValueError("嵌套或重复表格")
            self.has_table = True
        elif tag == "tr":
            if self.current is not None:
                raise ValueError("未闭合的单元格")
            self.row += 1
            self.column = 0
        elif tag in {"td", "th"}:
            if not self.has_table or self.row < 0 or self.current is not None:
                raise ValueError("表格缺少行或单元格重复")
            while (self.row, self.column) in self.occupied:
                self.column += 1
            rowspan = int(attrs.get("rowspan", 1))
            colspan = int(attrs.get("colspan", 1))
            if not 1 <= rowspan <= 1000 or not 1 <= colspan <= 1000:
                raise ValueError("单元格跨度无效")
            if self.row + rowspan > 1000 or self.column + colspan > 256:
                raise ValueError("表格尺寸过大")
            if len(self.occupied) + rowspan * colspan > 20000:
                raise ValueError("表格单元格数量过大")
            for r in range(self.row, self.row + rowspan):
                for c in range(self.column, self.column + colspan):
                    if (r, c) in self.occupied:
                        raise ValueError("单元格相互覆盖")
                    self.occupied.add((r, c))
            self.current = {"row": self.row, "col": self.column, "rowspan": rowspan,
                            "colspan": colspan, "text": ""}
            if tag == "th":
                self.header_rows = max(self.header_rows, self.row + 1)
            self.column += colspan
        elif tag == "br" and self.current is not None:
            self.current["text"] += "\n"

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.ignored:
            self.ignored -= 1
            return
        if tag in {"td", "th"} and self.current is not None:
            self.current["text"] = self.current["text"].strip()
            self.cells.append(self.current)
            self.current = None

    def handle_data(self, data):
        if self.current is not None and not self.ignored:
            self.current["text"] += data


def parse_table(source):
    parser = _TableParser()
    parser.feed(source)
    parser.close()
    if not parser.has_table or not parser.cells or parser.current is not None:
        raise ValueError("没有完整的表格单元格")
    height = max(c["row"] + c["rowspan"] for c in parser.cells)
    width = max(c["col"] + c["colspan"] for c in parser.cells)
    if len(parser.occupied) != height * width:
        raise ValueError("表格行列存在缺口")
    rows = [[None] * width for _ in range(height)]
    for cell in parser.cells:
        rows[cell["row"]][cell["col"]] = cell["text"]
    return rows, parser.cells, parser.header_rows


def _box(value):
    try:
        if len(value) == 4 and not isinstance(value[0], (list, tuple)):
            box = [float(v) for v in value]
        else:
            box = [min(float(p[0]) for p in value), min(float(p[1]) for p in value),
                   max(float(p[0]) for p in value), max(float(p[1]) for p in value)]
        if all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]:
            return box
    except (ValueError, TypeError, IndexError):
        pass
    return None


def _ocr_lines(raw):
    raw = raw.get("overall_ocr_res") or {}
    boxes = raw.get("rec_polys") or raw.get("rec_boxes") or raw.get("dt_polys") or []
    scores = raw.get("rec_scores") or []
    lines = []
    for index, text in enumerate(raw.get("rec_texts") or []):
        box = _box(boxes[index]) if index < len(boxes) else None
        if box and str(text).strip():
            lines.append({"bbox": box, "text": str(text),
                          "score": float(scores[index]) if index < len(scores) else None})
    return lines


def _observed_style(rgb, box, lines, sx, sy):
    """Observe line height and foreground color without font enumeration/fitting."""
    heights = [(line["bbox"][3] - line["bbox"][1]) * sy for line in lines]
    height = median(heights) if heights else min((box[3] - box[1]) * sy, 18)
    meta = {"font_size": round(max(5, min(72, height * .8)), 2),
            "style_source": "line_geometry_and_foreground", "style_approximate": True}
    x0, y0, x1, y1 = [int(v) for v in box]
    crop = rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    if crop.size:
        # At most a few thousand samples per block; exclude light page pixels.
        step = max(1, int(math.sqrt(crop.shape[0] * crop.shape[1] / 4096)))
        sample = crop[::step, ::step, :3].reshape(-1, 3)
        ink = sample[np.min(sample, axis=1) < 150]
        if len(ink):
            color = np.median(ink, axis=0).astype(int)
            meta["color"] = "#" + "".join(f"{v:02x}" for v in color)
    return meta


def _merge_native_text(model_text, originals, ocr_lines, mapped):
    """Replace only spatially matched OCR lines; keep other scanned lines.

    Native blocks may cover one part of a semantic paragraph while the rest is
    scanned. A character-count comparison cannot establish that ownership.
    """
    native_lines = []
    for _, block in originals:
        lines = block.meta.get("text_lines") or [{"bbox": block.bbox, "text": block.text}]
        native_lines.extend({"bbox": list(line["bbox"]), "text": line["text"], "source": "native"}
                            for line in lines if line.get("text"))

    def overlaps(first, second):
        area_first = max(0, first[2]-first[0]) * max(0, first[3]-first[1])
        overlap = max(0, min(first[2], second[2])-max(first[0], second[0])) * max(0, min(first[3], second[3])-max(first[1], second[1]))
        # A small native fragment inside a long OCR line does not explain the
        # rest of that line. Keep uncertain OCR evidence rather than drop it.
        return overlap / max(.01, area_first) >= .55

    merged = list(native_lines)
    for line in ocr_lines:
        box = mapped(line["bbox"])
        if not any(overlaps(box, original["bbox"]) for original in native_lines):
            merged.append({"bbox": box, "text": line["text"], "source": "ocr"})
    uncertain = False
    if not ocr_lines and model_text.strip():
        exact = "\n".join(line["text"] for line in native_lines)
        if "".join(model_text.split()) != "".join(exact.split()):
            # No line geometry can explain which model characters belong to
            # the native text. Retain both pieces of evidence for review.
            bounds = [min(l["bbox"][0] for l in native_lines), max(l["bbox"][3] for l in native_lines),
                      max(l["bbox"][2] for l in native_lines), max(l["bbox"][3] for l in native_lines)]
            merged.append({"bbox": bounds, "text": model_text, "source": "ocr"})
            uncertain = True
    merged.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    sources = sorted({line["source"] for line in merged})
    return "\n".join(line["text"] for line in merged), sources, uncertain


@tracked('reading.v3_adapt')
def extract_structure_page(page, work, dpi, warnings, *, original_rotation=0, native=()):
    from .structure import recognize_page

    scale = min(dpi / 72, 3500 / max(page.rect.width, page.rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
    raw = recognize_page(rgb)
    if "res" in raw and "parsing_res_list" not in raw:
        raw = raw["res"]
    sx, sy = page.rect.width / pix.width, page.rect.height / pix.height
    # The transport guarantees unrotated input/output coordinates. Reject a
    # mismatched result rather than silently misplacing blocks and table crops.
    if raw.get("width", pix.width) != pix.width or raw.get("height", pix.height) != pix.height:
        raise ValueError("V3 结果尺寸与原页渲染不一致，无法可靠还原坐标。")
    lines = _ocr_lines(raw)
    blocks = []
    owned = set()
    native_owned = set()
    reference = f"assets/page_{page.number + 1:04d}_reference.png"
    pix.save(work / reference)

    def mapped(box):
        return [round(box[0] * sx, 2), round(box[1] * sy, 2),
                round(box[2] * sx, 2), round(box[3] * sy, 2)]

    def crop(box, number, purpose="image"):
        path = f"assets/page_{page.number + 1:04d}_v3_{purpose}_{number:03d}.png"
        rect = fitz.Rect(mapped(box)) & page.rect
        if rect.is_empty:
            raise ValueError("V3 返回的区域位于页面之外。")
        image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
        image.save(work / path)
        return path, image.width, image.height

    # parsing_res_list is already assembled and ordered by V3; block_order may
    # legitimately be null for tables/figures, so never sort nulls to the end.
    for number, item in enumerate(raw.get("parsing_res_list") or [], 1):
        box = _box(item.get("block_bbox"))
        if not box:
            warnings.append(f"第 {page.number + 1} 页 V3 区域坐标无效，保留原页参考供校对。")
            continue
        box = [max(0, min(box[0], pix.width)), max(0, min(box[1], pix.height)),
               max(0, min(box[2], pix.width)), max(0, min(box[3], pix.height))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = str(item.get("block_label", "text"))
        text = str(item.get("block_content") or "").strip()
        matches = [(index, line) for index, line in enumerate(lines) if inside(line["bbox"], box, .6)]
        meta = {"semantic_label": label, "recognition_backend": "PP-StructureV3",
                "v3_block_id": item.get("block_id"), "v3_block_order": item.get("block_order"),
                "source_reference": reference, "original_page_rotation": original_rotation,
                "source_bbox_pixels": box, "coordinate_scale": [sx, sy]}
        scores = [line["score"] for _, line in matches if line["score"] is not None]
        confidence = min(scores) if scores else None
        is_picture = label in {"image", "chart", "seal", "header_image", "footer_image", "formula", "formula_number", "algorithm"}
        if label == "table":
            preview, width, height = crop(box, number, "table")
            try:
                rows, cells, header_rows = parse_table(text)
            except (ValueError, TypeError) as exc:
                meta.update({"width": width, "height": height, "structure_warning": str(exc),
                             "table_recovery": "unresolved"})
                blocks.append(Block("image", page.number + 1, mapped(box), source="ocr", asset=preview, meta=meta))
                warnings.append(f"第 {page.number + 1} 页表格结构未可靠恢复，保留原表图片：{exc}")
            else:
                meta.update({"grid_cells": cells, "header_rows": header_rows, "preview_asset": preview,
                             "structure_inferred": True, "strategy": "v3_visual_table"})
                blocks.append(Block("table", page.number + 1, mapped(box), source="ocr", rows=rows,
                                    confidence=confidence, meta=meta))
            owned.update(index for index, _ in matches)
        elif is_picture or not text:
            asset, width, height = crop(box, number)
            meta.update({"width": width, "height": height})
            image = Block("image", page.number + 1, mapped(box), source="ocr", asset=asset, meta=meta)
            blocks.append(image)
            for index, line in matches:
                if index in owned:
                    continue
                owned.add(index)
                blocks.append(Block("ocr_text", page.number + 1, mapped(line["bbox"]), line["text"],
                                    source="ocr", confidence=line["score"],
                                    meta={**meta, "image": asset, "figure_label": True}))
        else:
            kind = "text"
            if label in {"doc_title", "paragraph_title", "abstract_title", "reference_title"}:
                kind = "heading"
            elif label == "number":
                kind = "page_number"
            elif label in {"header", "footer", "aside_text"}:
                kind = "header_footer"
            meta.update(_observed_style(rgb, box, [line for _, line in matches], sx, sy))
            # Match native text to OCR line geometry; a semantic block may
            # contain both an editable paragraph and additional scanned lines.
            original = [(index, block) for index, block in enumerate(native)
                        if index not in native_owned and inside(block.bbox, mapped(box), .85)]
            sources = ["ocr"]
            if original:
                text, sources, uncertain = _merge_native_text(text, original, [line for _, line in matches], mapped)
                native_owned.update(index for index, _ in original)
                meta["text_sources"] = sources
                meta["text_source"] = "native_pdf" if sources == ["native"] else "native_pdf_and_ocr"
                if uncertain:
                    meta["needs_review"] = True
                    warnings.append(f"第 {page.number + 1} 页混合文字块缺少 OCR 行坐标，原生文字与模型文字均保留，请对照原页去重校对。")
            blocks.append(Block(kind, page.number + 1, mapped(box), text, source="native" if sources == ["native"] else "ocr",
                                level=1 if label == "doc_title" else 2 if kind == "heading" else 0,
                                confidence=confidence, meta=meta))
            owned.update(index for index, _ in matches)
    for index, block in enumerate(native):
        if index in native_owned:
            continue
        # Text inside an image/table already has a visible owner. Native text
        # missed by layout detection remains editable and explicitly reviewable.
        if any(b.kind in {"image", "table"} and inside(block.bbox, b.bbox, .8) for b in blocks):
            continue
        if any(b.kind not in {"image", "table"} and inside(block.bbox, b.bbox, .85) for b in blocks):
            continue
        block.meta.update({"semantic_label": "unassigned_native_text", "source_reference": reference,
                           "recognition_backend": "PP-StructureV3", "needs_review": True})
        blocks.append(block)
    # Do not discard OCR lines solely because layout detection missed a block.
    for index, line in enumerate(lines):
        if index not in owned:
            blocks.append(Block("ocr_text", page.number + 1, mapped(line["bbox"]), line["text"],
                                source="ocr", confidence=line["score"],
                                meta={"semantic_label": "unassigned_text", "source_reference": reference,
                                      "recognition_backend": "PP-StructureV3", "needs_review": True}))
    if not blocks:
        blocks.append(Block("image", page.number + 1, [0, 0, page.rect.width, page.rect.height],
                            source="ocr", asset=reference,
                            meta={"semantic_label": "unresolved_page", "scan_reference": True,
                                  "width": pix.width, "height": pix.height}))
        warnings.append(f"第 {page.number + 1} 页 V3 未恢复可用内容，保留原页图片。")
    warnings.append(f"第 {page.number + 1} 页使用本地 V3 识别；字体与字号为近似观察，文字、表格和边界仍需校对。")
    return blocks
