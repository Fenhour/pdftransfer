from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal["text", "heading", "table", "image", "page_number", "header_footer", "ocr_text"]


@dataclass
class Block:
    kind: Kind
    page: int
    bbox: list[float]
    text: str = ""
    id: str = ""
    level: int = 0
    source: str = "native"
    confidence: float | None = None
    asset: str = ""
    rows: list[list[str | None]] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    source: str
    total_pages: int
    selected_pages: list[int]
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outline: list[dict] = field(default_factory=list)
    page_sizes: dict[int, list[float]] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class Options:
    pages: str = ""
    ocr: str = "auto"
    password: str = ""
    dpi: int = 180
    table_strategy: str = "auto"
    order: str = "columns"
    # Library callers retain the historical dual export and lightweight OCR.
    # User-facing HTML / MD entries set one format and opt into local V3.
    output_format: str = "both"
    recognition_backend: str = "legacy"


def parse_pages(spec: str, total: int) -> list[int]:
    """Return sorted, unique zero-based indices; reject malformed/out-of-range input."""
    if not spec.strip():
        return list(range(total))
    chosen = set()
    for part in spec.replace("，", ",").split(","):
        ends = part.strip().split("-")
        if len(ends) > 2 or any(not x.strip().isdigit() for x in ends):
            raise ValueError("页码格式应为 1-3,5,8-10。")
        start, end = int(ends[0]), int(ends[-1])
        if not 1 <= start <= end <= total:
            raise ValueError(f"页码范围无效：{part}（文档共 {total} 页）。")
        chosen.update(range(start - 1, end))
    return sorted(chosen)
