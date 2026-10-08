"""Conservative typography classification and geometric reading order."""
import re
from collections import Counter

from .model import Block

PAGE_NUMBER = re.compile(r"^(?:[-–—·•]\s*)?(?:第\s*)?(?:\d+|[ivxlcdm]+)(?:\s*页)?(?:\s*(?:/|of|共)\s*\d+\s*页?)?(?:\s*[-–—·•])?$", re.I)
NUMBERED_TITLE = re.compile(r"^(?:第[一二三四五六七八九十百\d]+[章节篇部]|(?:\d+|[A-Z])(?:\.\d+){0,4}\.?\s+|[一二三四五六七八九十]+[、．])")


def inside(box, region, threshold=0.55):
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    intersect = max(0, min(box[2], region[2]) - max(box[0], region[0])) * max(0, min(box[3], region[3]) - max(box[1], region[1]))
    return area > 0 and intersect / area >= threshold


def _gap_split(blocks, axis):
    ordered = sorted(blocks, key=lambda b: b.bbox[axis])
    end = ordered[0].bbox[axis + 2]
    cuts = []
    for i, block in enumerate(ordered[1:], 1):
        gap = block.bbox[axis] - end
        if gap > (12 if axis == 0 else 5):
            cuts.append((gap, i))
        end = max(end, block.bbox[axis + 2])
    if cuts:
        _, index = max(cuts)
        return ordered[:index], ordered[index:]
    return None


def reading_order(blocks: list[Block], mode="columns") -> list[Block]:
    if mode == "rows" or len(blocks) < 2:
        return sorted(blocks, key=lambda b: (round(b.bbox[1], 1), b.bbox[0]))
    # Whitespace XY-cut: a full vertical gutter wins; spanning titles create
    # horizontal bands before columns are considered within each band.
    def cut(items, depth=0):
        if len(items) < 2 or depth >= 40:
            return sorted(items, key=lambda b: (b.bbox[1], b.bbox[0]))
        split = _gap_split(items, 0) or _gap_split(items, 1)
        if split:
            return cut(split[0], depth + 1) + cut(split[1], depth + 1)
        return sorted(items, key=lambda b: (b.bbox[1], b.bbox[0]))
    # Full-page scan images are references, not column separators.
    refs = [b for b in blocks if b.meta.get("scan_reference")]
    marginal = [b for b in blocks if b.meta.get("classification") == "side_margin"]
    rest = [b for b in blocks if not b.meta.get("scan_reference") and b.meta.get("classification") != "side_margin"]
    return refs + cut(rest) + marginal


def classify(blocks: list[Block], page_sizes: dict, outline: list[dict]):
    frequencies = Counter()
    for block in blocks:
        if block.kind == "text" and not block.meta.get("vertical") and not block.meta.get("figure_label") and not block.meta.get("code"):
            frequencies[round(block.meta.get("font_size", 10), 1)] += len(block.text)
    body_size = frequencies.most_common(1)[0][0] if frequencies else 10
    title_sizes = sorted({size for size in frequencies if size > body_size * 1.12}, reverse=True)
    repeated = {}
    for block in blocks:
        if block.kind not in ("text", "ocr_text"):
            continue
        if block.meta.get("figure_label") or block.meta.get("caption_for") or block.meta.get("code"):
            continue
        height = page_sizes[block.page][1]
        margin = block.bbox[3] <= height * 0.09 or block.bbox[1] >= height * 0.90
        if margin:
            key = re.sub(r"\d+", "#", block.text.strip().casefold())
            repeated.setdefault(key, set()).add(block.page)
    def title_key(text):
        text = re.sub(r"^(?:\d+|[A-Z])(?:\.\d+)*\.?\s+", "", text.strip())
        return re.sub(r"\s+", "", text).casefold()
    outline_map = {(item["page"], title_key(item["title"])): item["level"] for item in outline}
    for block in blocks:
        if block.kind not in ("text", "ocr_text"):
            continue
        if block.meta.get("figure_label") or block.meta.get("caption_for") or block.meta.get("code"):
            continue
        height = page_sizes[block.page][1]
        margin = block.bbox[3] <= height * 0.09 or block.bbox[1] >= height * 0.90
        text = block.text.strip()
        width = page_sizes[block.page][0]
        if block.meta.get("vertical") and (block.bbox[2] < width * .10 or block.bbox[0] > width * .90):
            block.kind = "header_footer"
            block.meta["classification"] = "side_margin"
            continue
        if margin and (PAGE_NUMBER.fullmatch(text) or re.fullmatch(r"page\s+\d+(?:\s+of\s+\d+)?", text, re.I)):
            block.kind = "page_number"
            continue
        key = re.sub(r"\d+", "#", text.casefold())
        if margin and len(repeated.get(key, set())) >= max(2, len(page_sizes) * 0.4):
            block.kind = "header_footer"
            continue
        outline_level = outline_map.get((block.page, title_key(text)))
        size = round(block.meta.get("font_size", body_size), 1)
        short = len(text) <= 180 and text.count("\n") <= 2
        heading = short and (size > body_size * 1.12 or (block.meta.get("bold") and NUMBERED_TITLE.match(text)))
        if outline_level or heading:
            block.kind = "heading"
            block.level = min(6, outline_level or (title_sizes.index(size) + 1 if size in title_sizes else 3))
            block.meta["classification"] = "pdf_outline" if outline_level else "font_heuristic"
