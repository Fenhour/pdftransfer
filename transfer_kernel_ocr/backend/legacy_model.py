"""The JSON document is the single source of truth, in PDF points (72/inch)."""
from __future__ import annotations

import copy
import json
import math
import re
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

SCHEMA = "transfer_kernel_ocr/1"
TYPES = {"text", "image", "vector", "line", "background"}
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
SAFE_TAGS = set("svg g defs path use image clipPath mask rect circle ellipse line polyline polygon text tspan linearGradient radialGradient stop pattern symbol title desc".split())
COLOR = re.compile(r"^(#[0-9a-fA-F]{3,8}|none|transparent)$")
DATA_IMAGE = re.compile(r"^data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=\s]+$")


def uid(prefix="el"):
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def clean_svg(fragment):
    """SVG imported through JSON is data, never executable markup or a URL."""
    root = ET.fromstring(f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}">{fragment}</svg>')
    for node in root.iter():
        if node.tag.split("}")[-1] not in SAFE_TAGS:
            raise ValueError("SVG 含不支持的标签")
        for key, value in node.attrib.items():
            name = key.split("}")[-1].lower()
            if name.startswith("on") or name == "style":
                raise ValueError("SVG 不允许脚本或内联 CSS")
            if name == "href" and not (value.startswith("#") or DATA_IMAGE.fullmatch(value)):
                raise ValueError("SVG 只允许文档内引用和内嵌位图")
            if "url(" in value and not re.fullmatch(r"url\(#[\w.:-]+\)", value):
                raise ValueError("SVG 不允许外部资源")
    return "".join(ET.tostring(node, encoding="unicode") for node in root)


def finite(value, name, low=-1_000_000, high=1_000_000):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"无效的 {name}")


def validate_document(document):
    """Validate a copy so bad imports cannot corrupt the open project."""
    doc = copy.deepcopy(document)
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("pages"), list) or not 1 <= len(doc["pages"]) <= 1000:
        raise ValueError("不支持的项目格式或页数")
    ids = set()
    total = 0
    for page in doc["pages"]:
        for key in ("width", "height"):
            finite(page.get(key), key, .1, 30_000)
        page["defs"] = clean_svg(page.get("defs", ""))
        ref = page.get("reference", "")
        if ref and not DATA_IMAGE.fullmatch(ref):
            raise ValueError("原页参考必须为内嵌位图")
        if not isinstance(page.get("elements"), list):
            raise ValueError("缺少元素集合")
        total += len(page["elements"])
        if total > 100_000:
            raise ValueError("项目元素过多")
        for element in page["elements"]:
            identifier = element.get("id")
            if not isinstance(identifier, str) or identifier in ids or len(identifier) > 100:
                raise ValueError("元素 ID 缺失或重复")
            ids.add(identifier)
            if element.get("type") not in TYPES:
                raise ValueError("未知元素类型")
            for key in ("x", "y", "rotation", "z"):
                finite(element.get(key, 0), key)
            for key in ("width", "height", "base_width", "base_height"):
                finite(element.get(key), key, .01)
            finite(element.get("opacity", 1), "opacity", 0, 1)
            if len(element.get("text", "")) > 200_000:
                raise ValueError("文字过长")
            style = element.get("style", {})
            for key in ("color", "fill", "stroke"):
                if key in style and not COLOR.fullmatch(str(style[key])):
                    raise ValueError("颜色必须是十六进制或 none")
            for key in ("font_size", "line_height", "stroke_width"):
                if key in style:
                    finite(style[key], key, .01, 10000)
            if style.get("align", "left") not in ("left", "center", "right"):
                raise ValueError("未知对齐方式")
            source = element.get("source", {})
            if "text_rotation" in source:
                finite(source["text_rotation"], "text_rotation")
            if "text_origin" in element:
                origin = element["text_origin"]
                if not isinstance(origin, list) or len(origin) != 2:
                    raise ValueError("文字基线无效")
                for value in origin:
                    finite(value, "text_origin")
            appearance = element.get("appearance", {})
            if appearance.get("kind") == "svg":
                appearance["svg"] = clean_svg(appearance.get("svg", ""))
            elif appearance.get("kind") == "image":
                if not DATA_IMAGE.fullmatch(appearance.get("data", "")):
                    raise ValueError("图片必须为内嵌 PNG/JPEG/WebP")
            elif appearance.get("kind") != "generated":
                raise ValueError("未知外观格式")
    return doc

