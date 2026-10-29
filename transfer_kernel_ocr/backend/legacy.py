"""Read former transfer_kernel_ocr/1 projects without weakening the scene validator.

Old SVG fragments are validated and rasterized locally (never injected in HTML).
The old project remains untouched; imported scenes have independent identifiers.
"""
import base64
from datetime import datetime, timezone
import io
import math
import uuid

import pymupdf as fitz
from PIL import Image

from .legacy_model import validate_document as validate_legacy
from .legacy_render import edited_text, local_content
from .models import validate_document


SCHEMA = "transfer_kernel_ocr/1"


def is_legacy(data):
    return isinstance(data, dict) and data.get("schema") == SCHEMA


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _snapshot(element, definitions, original=False):
    appearance = element.get("appearance", {})
    if appearance.get("kind") == "image" and (original or not (element["type"] == "text" and edited_text(element))):
        return appearance["data"]
    width, height = element["base_width"], element["base_height"]
    if width * height > 40_000_000:
        raise ValueError("旧元素尺寸过大，无法安全迁移")
    scale = min(2.0, math.sqrt(40_000_000 / (width * height)) * .999)
    current = element
    if original and element["type"] == "text":
        current = dict(element, render_mode="original", text=element.get("original_text", element.get("text", "")),
                       style=element.get("original_style", element.get("style", {})))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">'
           f'<defs>{definitions}</defs>{local_content(current)}</svg>')
    with fitz.open(stream=svg.encode("utf-8"), filetype="svg") as source:
        pixmap = source[0].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
        return "data:image/png;base64," + base64.b64encode(pixmap.tobytes("png")).decode()


def convert_legacy(data):
    legacy = validate_legacy(data)
    if len(legacy["pages"]) > 100:
        raise ValueError("新编辑器最多支持 100 页；请先在旧版本拆分项目")
    now = datetime.now(timezone.utc).isoformat()
    document = {
        "schemaVersion": 1, "id": uuid.uuid4().hex, "name": legacy.get("name") or "旧版项目",
        "createdAt": legacy.get("created_at", now), "updatedAt": now, "pages": [],
        "warnings": list(legacy.get("warnings", [])) + [
            "由 transfer_kernel_ocr/1 迁移；旧 SVG/线条保存为原外观位图，原位置、旋转、显隐和文字保留。修改文字后按当前字体重绘，需校对。"],
        "migration": {"schema": SCHEMA, "sourceId": legacy.get("id"), "sourceRevision": legacy.get("revision")},
    }
    for index, source_page in enumerate(legacy["pages"]):
        if len(source_page["elements"]) > 15000:
            raise ValueError("旧页面超过新编辑器的 15000 元素限制")
        if source_page["width"] * source_page["height"] > 40_000_000:
            raise ValueError("旧页面尺寸过大")
        page = {"id": f"page_{index+1}", "index": index, "width": source_page["width"],
                "height": source_page["height"], "rotation": 0, "elements": [],
                "warnings": source_page.get("warnings", []), "sourcePage": source_page.get("source_page")}
        if source_page.get("reference"):
            page["sourceSrc"] = source_page["reference"]
        color = source_page.get("paper_color", "#ffffff")
        # The current canvas has white paper. Preserve non-white legacy paper as
        # a separate locked background instead of silently changing its color.
        if color.lower() not in {"#ffffff", "#fff"}:
            page["elements"].append({"id": uuid.uuid4().hex, "type": "background", "name": "旧版纸色",
                "x": 0, "y": 0, "width": page["width"], "height": page["height"], "zIndex": -1000000,
                "locked": True, "src": _png(Image.new("RGBA", (1, 1), color))})
        for old in source_page["elements"]:
            style = old.get("style", {})
            scale_y = old["height"] / old["base_height"]
            source_type = old["type"]
            element = {"id": old["id"], "type": "region" if source_type == "line" else source_type,
                "name": old.get("name", source_type), "x": old["x"], "y": old["y"],
                "width": old["width"], "height": old["height"], "rotation": old.get("rotation", 0),
                "zIndex": old.get("z", 0), "visible": old.get("visible", True),
                "locked": old.get("locked", False), "opacity": old.get("opacity", 1),
                "text": old.get("text", ""), "fontFamily": style.get("font_family", "sans-serif"),
                "fontSize": style.get("font_size", 12) * scale_y,
                "fontWeight": style.get("font_weight", "normal"), "fontStyle": style.get("font_style", "normal"),
                "align": style.get("align", "left"), "lineHeight": style.get("line_height", 1.2),
                "color": style.get("color", "#222222"), "source": "legacy-migration", "reviewRequired": True,
                "legacy": {"type": source_type, "baseWidth": old["base_width"], "baseHeight": old["base_height"],
                           "source": old.get("source", {})}}
            src = _snapshot(old, source_page.get("defs", ""))
            if source_type == "text":
                element["appearanceSrc"] = src
                element["originalAppearanceSrc"] = _snapshot(old, source_page.get("defs", ""), original=True)
                element["originalText"] = old.get("original_text", old.get("text", ""))
            else:
                element["src"] = src
                element["appearanceSrc"] = src
            page["elements"].append(element)
        document["pages"].append(page)
    return validate_document(document)
