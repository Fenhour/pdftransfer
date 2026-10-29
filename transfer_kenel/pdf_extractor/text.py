"""Reassemble visual lines without dropping punctuation or code indentation."""
import re


LIST_ITEM = re.compile(r"^(?:[（(]\s*(?:\d{1,3}|[一二三四五六七八九十]+)\s*[）)]|[①-⑳]|[•●▪‣]|\d{1,3}[.)、])(?:\s|(?=[\u3400-\u9fff]))")


def prose_quotes(lines):
    """Quoted prompts are prose, even when the PDF uses a monospace font."""
    closing = None
    pairs = {'“': '”', '«': '»', '「': '」', '『': '』', '"': '"'}
    result = []
    for original in lines:
        line = dict(original)
        text = line['text']
        if LIST_ITEM.match(text) and not line['code']:
            closing = None
        quoted = closing is not None or text.startswith(('“', '«', '「', '『'))
        # Do not treat quotes in executable code as prose quotation boundaries.
        if not line['code'] or quoted:
            for index, char in enumerate(text):
                if index and text[index - 1] == "\\":
                    continue
                if closing:
                    if char == closing:
                        closing = None
                elif char in pairs:
                    closing = pairs[char]
            if quoted:
                line['code'] = False
        result.append(line)
    return result


def clean_inline(value):
    return re.sub(r"[ \t\u00a0\u2000-\u200a\u202f]+", " ", value).strip()


def reflow(value):
    lines = [clean_inline(line) for line in value.splitlines() if line.strip()]
    text = ""
    for line in lines:
        if not text:
            text = line
        elif re.search(r"[\u3400-\u9fff]$", text) and re.match(r"[\u3400-\u9fff，。；：！？]", line):
            text += line
        elif text.endswith("\u00ad"):
            text = text[:-1] + line
        elif text.endswith("-") and re.match(r"[a-z]", line):
            text += line
        else:
            # Keep real hyphens: joining a wrapped line must not change identifiers.
            text += " " + line
    return text


def visual_lines(raw):
    """Merge fragments sharing a baseline; preserve every PDF span character."""
    groups = []
    for line in raw.get("lines", []):
        spans = [s for s in line.get("spans", []) if s["text"]]
        if not spans:
            continue
        vertical = abs(line.get("dir", (1, 0))[1]) > .5
        baseline = spans[0].get("origin", (0, line["bbox"][3]))[1]
        def same_line(group):
            if vertical or group["vertical"] or abs(group["baseline"] - baseline) >= 1.6:
                return False
            left, right = min(s["bbox"][0] for s in spans), max(s["bbox"][2] for s in spans)
            old_left = min(s["bbox"][0] for s in group["spans"])
            old_right = max(s["bbox"][2] for s in group["spans"])
            gap = max(0, left - old_right, old_left - right)
            return gap <= max(12, min(s["size"] for s in spans) * 2)
        group = next((g for g in groups if same_line(g)), None)
        if group is None:
            group = {"baseline": baseline, "vertical": vertical, "spans": []}
            groups.append(group)
        group["spans"].extend(spans)
    result = []
    for group in groups:
        spans = sorted(group["spans"], key=lambda s: s["bbox"][0]) if not group["vertical"] else group["spans"]
        text = ""
        previous = None
        runs = []
        for span in spans:
            value = span["text"].replace("\n", " ")
            prefix = ""
            if previous and text and not text[-1].isspace() and value and not value[0].isspace():
                gap = span["bbox"][0] - previous["bbox"][2]
                # Spatial word separator; adjacent punctuation stays attached.
                if gap > min(previous["size"], span["size"]) * .16:
                    text += " "
                    prefix = " "
            text += value
            flags = int(span.get("flags", 0))
            runs.append({"text": value, "separator": prefix, "bbox": [round(float(v), 2) for v in span["bbox"]],
                         "origin": [round(float(v), 2) for v in span.get("origin", (span['bbox'][0], baseline))],
                         "font": str(span.get("font", "")), "size": round(float(span["size"]), 2),
                         "color": f'#{int(span.get("color", 0)) & 0xffffff:06x}',
                         "bold": bool(flags & 16), "italic": bool(flags & 2),
                         "monospace": bool(flags & 8)})
            previous = span
        box = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
               max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
        weights = sum(len(s["text"].strip()) for s in spans) or 1
        mono = sum(len(s["text"].strip()) for s in spans if s["flags"] & 8) / weights > .65
        bold = sum(len(s["text"].strip()) for s in spans if s["flags"] & 16) / weights > .65
        size = max(s["size"] for s in spans if s["text"].strip()) if any(s["text"].strip() for s in spans) else 10
        result.append({"text": text.strip() if mono else clean_inline(text), "bbox": box, "size": round(size, 1),
                       "bold": bold, "code": mono, "vertical": group["vertical"], "runs": runs})
    return sorted(result, key=lambda line: (line["bbox"][1], line["bbox"][0]))


def merge_code_blocks(blocks):
    """Join fragmented code, using one shared indentation origin per region."""
    result = []
    for block in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0])):
        previous = result[-1] if result else None
        compatible = (previous and previous.meta.get("code") and block.meta.get("code")
                      and -2 <= block.bbox[1] - previous.bbox[3] <= 28
                      and min(block.bbox[2], previous.bbox[2]) > max(block.bbox[0], previous.bbox[0])
                      and abs(block.meta["font_size"] - previous.meta["font_size"]) < 1)
        if compatible:
            previous.bbox = [min(previous.bbox[0], block.bbox[0]), min(previous.bbox[1], block.bbox[1]),
                             max(previous.bbox[2], block.bbox[2]), max(previous.bbox[3], block.bbox[3])]
            previous.meta["code_lines"].extend(block.meta["code_lines"])
            previous.meta["text_lines"].extend(block.meta["text_lines"])
        else:
            result.append(block)
    for block in result:
        if not block.meta.get("code"):
            continue
        lines = sorted(block.meta["code_lines"], key=lambda line: (line["bbox"][1], line["bbox"][0]))
        unit = max(3, block.meta["font_size"] * .5)
        rebuilt = []
        previous_y = None
        for line in lines:
            if previous_y is not None and line["bbox"][1] - previous_y > block.meta["font_size"] * 1.8:
                rebuilt.append("")
            rebuilt.append(" " * max(0, round((line["bbox"][0] - block.bbox[0]) / unit)) + line["text"])
            previous_y = line["bbox"][1]
        block.text = "\n".join(rebuilt)
    return result
