"""Validate the shared scene graph at the trust boundary, without discarding metadata."""
from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote
from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,100}$")
COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|transparent|none|black|white|red|blue|green)$")


def validate_asset(value: str, document_id: str) -> str:
    if value.startswith("data:image/"):
        if not re.match(r"^data:image/(png|jpeg|webp|gif|bmp);base64,[a-zA-Z0-9+/=\s]+$", value):
            raise ValueError("只允许 PNG/JPEG/WebP/GIF/BMP 内嵌图片")
        if len(value) > 30_000_000:
            raise ValueError("内嵌图片过大")
        return value
    prefix = f"/files/{document_id}/"
    decoded = unquote(value)
    if not decoded.startswith(prefix) or "\\" in decoded or any(p in (".", "..") for p in decoded[len(prefix):].split("/")):
        raise ValueError("图片路径必须属于当前文档")
    if "?" in decoded or "#" in decoded:
        raise ValueError("无效图片路径")
    return value


def validate_svg(value: str) -> str:
    if len(value) > 2_000_000 or "<!" in value:
        raise ValueError("SVG 太大或含不支持的声明")
    try:
        root = ET.fromstring(value)
    except ET.ParseError as exc:
        raise ValueError("无效 SVG") from exc
    allowed = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "defs", "clipPath", "mask", "linearGradient", "radialGradient", "stop", "title", "desc"}
    for node in root.iter():
        if node.tag.split("}")[-1] not in allowed:
            raise ValueError("SVG 包含不允许的元素")
        for key, val in node.attrib.items():
            key = key.split("}")[-1].lower()
            if key.startswith("on") or key in {"href", "src"} or any(s in val.lower() for s in ("javascript:", "http:", "https:", "data:", "@import", "expression(")):
                raise ValueError("SVG 不允许外部资源或脚本")
            if "url(" in val.lower() and not re.fullmatch(r"url\(#[\w-]+\)", val):
                raise ValueError("SVG 只能引用内部图形")
    return value


class Element(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=100)
    type: str
    name: str = Field(default="元素", max_length=500)
    x: float = Field(ge=-50000, le=50000)
    y: float = Field(ge=-50000, le=50000)
    width: float = Field(gt=0, le=20000)
    height: float = Field(gt=0, le=20000)
    rotation: float = Field(default=0, ge=-360000, le=360000)
    zIndex: float = Field(default=0, ge=-1000000, le=1000000)
    opacity: float = Field(default=1, ge=0, le=1)
    visible: bool = True
    locked: bool = False
    text: str = Field(default="", max_length=100000)
    fontFamily: str = Field(default="Microsoft YaHei", max_length=200)
    fontSize: float = Field(default=14, gt=0, le=1000)
    fontWeight: str | int = "normal"
    fontStyle: str = "normal"
    align: str = "left"
    lineHeight: float = Field(default=1.2, ge=0.5, le=5)
    color: str = "#172c3c"
    fill: str | None = None
    stroke: str | None = None
    strokeWidth: float = Field(default=1, ge=0, le=1000)

    @model_validator(mode="after")
    def check_element(self):
        if not ID_RE.fullmatch(self.id) or self.type not in {"text", "image", "vector", "region", "background", "shape"}:
            raise ValueError("无效元素 ID 或类型")
        if self.align not in {"left", "center", "right", "justify"} or self.fontStyle not in {"normal", "italic", "oblique"}:
            raise ValueError("无效文字样式")
        if str(self.fontWeight) not in {"normal", "bold", "100", "200", "300", "400", "500", "600", "700", "800", "900"}:
            raise ValueError("无效字重")
        for c in (self.color, self.fill, self.stroke):
            if c is not None and not COLOR_RE.fullmatch(c):
                raise ValueError("颜色请使用十六进制值或透明色")
        if any(c in self.fontFamily for c in "<>;{}\\\"'"):
            raise ValueError("字体名称包含无效字符")
        extra = self.model_extra or {}
        if extra.get("svg"):
            validate_svg(extra["svg"])
        if self.type == "shape" and extra.get("shape", "rect") not in {"rect", "ellipse", "line"}:
            raise ValueError("无效图形类型")
        return self


class Page(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    id: str
    index: int = Field(ge=0)
    width: float = Field(gt=0, le=20000)
    height: float = Field(gt=0, le=20000)
    rotation: float = 0
    elements: list[Element] = Field(max_length=15000)
    warnings: list = Field(default_factory=list)


class Document(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    schemaVersion: int = 1
    id: str
    name: str = Field(min_length=1, max_length=500)
    pages: list[Page] = Field(min_length=1, max_length=100)
    warnings: list = Field(default_factory=list)

    @model_validator(mode="after")
    def check_document(self):
        if self.schemaVersion != 1 or not ID_RE.fullmatch(self.id):
            raise ValueError("不支持的文档版本或 ID")
        ids = set()
        for page in self.pages:
            if not ID_RE.fullmatch(page.id) or page.id in ids:
                raise ValueError("页面 ID 无效或重复")
            ids.add(page.id)
            if page.width * page.height > 40_000_000:
                raise ValueError("页面尺寸过大")
            extra = page.model_extra or {}
            if extra.get("sourceSrc"):
                validate_asset(extra["sourceSrc"], self.id)
            tables = extra.get("tables", [])
            if not isinstance(tables, list) or len(tables) > 1000:
                raise ValueError("页面表格列表无效或过大")
            for table in tables:
                if not isinstance(table, dict):
                    raise ValueError("无效表格元数据")
                reference = table.get("referenceSrc")
                if reference is not None:
                    if not isinstance(reference, str):
                        raise ValueError("表格参考必须是图片路径")
                    validate_asset(reference, self.id)
            elem_ids = set()
            for elem in page.elements:
                if elem.id in elem_ids:
                    raise ValueError("页面包含重复元素 ID")
                elem_ids.add(elem.id)
                for field in ("src", "appearanceSrc", "originalAppearanceSrc"):
                    src = (elem.model_extra or {}).get(field)
                    if src:
                        validate_asset(src, self.id)
        return self


def validate_document(data: dict, expected_id: str | None = None) -> dict:
    result = Document.model_validate(data).model_dump(exclude_none=True)
    if expected_id and result["id"] != expected_id:
        raise ValueError("文档 ID 不匹配")
    return result
