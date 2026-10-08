"""Reassemble visual lines without dropping punctuation or code indentation."""
import re


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
        group = next((g for g in groups if not vertical and not g["vertical"] and abs(g["baseline"] - baseline) < 1.6), None)
        if group is None:
            group = {"baseline": baseline, "vertical": vertical, "spans": []}
            groups.append(group)
        group["spans"].extend(spans)
    result = []
    for group in groups:
        spans = sorted(group["spans"], key=lambda s: s["bbox"][0]) if not group["vertical"] else group["spans"]
        text = ""
        previous = None
        for span in spans:
            value = span["text"].replace("\n", " ")
            if previous and text and not text[-1].isspace() and value and not value[0].isspace():
                gap = span["bbox"][0] - previous["bbox"][2]
                # Spatial word separator; adjacent punctuation stays attached.
                if gap > min(previous["size"], span["size"]) * .16:
                    text += " "
            text += value
            previous = span
        box = [min(s["bbox"][0] for s in spans), min(s["bbox"][1] for s in spans),
               max(s["bbox"][2] for s in spans), max(s["bbox"][3] for s in spans)]
        weights = sum(len(s["text"].strip()) for s in spans) or 1
        mono = sum(len(s["text"].strip()) for s in spans if s["flags"] & 8) / weights > .65
        bold = sum(len(s["text"].strip()) for s in spans if s["flags"] & 16) / weights > .65
        size = max(s["size"] for s in spans if s["text"].strip()) if any(s["text"].strip() for s in spans) else 10
        result.append({"text": text.strip() if mono else clean_inline(text), "bbox": box, "size": round(size, 1),
                       "bold": bold, "code": mono, "vertical": group["vertical"]})
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
