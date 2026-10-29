"""Physical page selection shared by HTTP imports and the compatibility CLI."""
import re


def selected_pages(spec, total):
    if not spec or not spec.strip():
        return list(range(total))
    selected = set()
    for part in spec.split(","):
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not match:
            raise ValueError("页码格式应为 1-3,5")
        first = int(match[1])
        last = int(match[2] or first)
        if not 1 <= first <= last <= total:
            raise ValueError(f"页码超出范围 1–{total}：{part.strip()}")
        selected.update(range(first - 1, last))
    return sorted(selected)
