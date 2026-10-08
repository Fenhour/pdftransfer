from collections import Counter
from pathlib import Path
import re
import shutil
import threading
import uuid

import pymupdf as fitz

from .layout import classify, inside, reading_order
from .model import Block, Options, Result, parse_pages
from .ocr import LocalOCR, detect_grid_tables
from .text import visual_lines, reflow, merge_code_blocks
from .figures import figure_regions


class Cancelled(Exception):
    pass


def _check(cancel):
    if cancel and cancel.is_set():
        raise Cancelled("已取消提取。")


def _box(rect):
    return [round(float(v), 2) for v in rect]


def _native_text(page, tables):
    result = []
    # Omit embedded image bytes here; images are extracted separately.
    data = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)
    for raw in data["blocks"]:
        if raw["type"] != 0:
            continue
        group = []

        def flush():
            if not group:
                return
            sizes = Counter()
            for line in group:
                sizes[line["size"]] += len(line["text"])
            bounds = fitz.Rect(group[0]["bbox"])
            for line in group[1:]:
                bounds |= fitz.Rect(line["bbox"])
            code = group[0]["code"]
            if code:
                unit = max(3, sizes.most_common(1)[0][0] * .5)
                text = "\n".join(" " * max(0, round((line["bbox"][0] - bounds.x0) / unit)) + line["text"] for line in group)
            else:
                text = reflow("\n".join(line["text"] for line in group))
            result.append(Block("text", page.number + 1, _box(bounds), text,
                                meta={"font_size": sizes.most_common(1)[0][0], "bold": all(line["bold"] for line in group),
                                      "vertical": any(line["vertical"] for line in group), "code": code,
                                      "code_lines": [{"text": line["text"], "bbox": line["bbox"]} for line in group] if code else []}))
            group.clear()

        for line in visual_lines(raw):
            if any(inside(line["bbox"], table.bbox) for table in tables):
                flush()
                continue
            if not line["text"]:
                continue
            # Preserve paragraph boundaries, but split typography changes and margins.
            margin = line["bbox"][3] < page.rect.height * .09 or line["bbox"][1] > page.rect.height * .90
            if group and (abs(group[-1]["size"] - line["size"]) > 1 or group[-1]["code"] != line["code"] or margin):
                flush()
            group.append(line)
            if margin:
                flush()
        flush()
    return merge_code_blocks(result)


def _tables(page, strategy, warnings):
    try:
        found = page.find_tables(strategy=strategy)
        tables = []
        for table in found.tables:
            rows = table.extract()
            if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
                continue
            tables.append(Block("table", page.number + 1, _box(table.bbox), rows=rows,
                                meta={"cells": [_box(cell) if cell else None for cell in table.cells],
                                      "strategy": strategy, "merged_cells": "null cells retained as blanks"}))
        return tables
    except Exception as exc:
        warnings.append(f"第 {page.number + 1} 页表格检测失败，保留为普通文字：{exc}")
        return []


def _render(page, rect, path, dpi):
    # Bound memory for unusually large posters while preserving coordinate mapping.
    scale = min(dpi / 72, 3500 / max(rect.width, rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    pix.save(path)
    return pix.width, pix.height


def extract(source: Path, work: Path, options: Options, progress=None, cancel: threading.Event | None = None) -> Result:
    progress = progress or (lambda done, total, message: None)
    if options.ocr not in ("off", "auto", "all"):
        raise ValueError("OCR 模式无效。")
    if options.order not in ("columns", "rows") or options.table_strategy not in ("lines_strict", "lines", "text"):
        raise ValueError("版面或表格模式无效。")
    if not 72 <= options.dpi <= 400:
        raise ValueError("图片分辨率应在 72–400 DPI 之间。")
    assets = work / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ocr = LocalOCR()
    ocr_failed = False
    with fitz.open(source) as doc:
        if not doc.is_pdf:
            raise ValueError("请选择 PDF 文件。")
        if doc.needs_pass and not doc.authenticate(options.password):
            raise ValueError("PDF 已加密，请提供正确的密码。")
        selected = parse_pages(options.pages, len(doc))
        if not selected:
            raise ValueError("PDF 没有可提取的页面。")
        result = Result(str(source.resolve()), len(doc), [p + 1 for p in selected])
        result.outline = [{"level": level, "title": title, "page": page} for level, title, page in doc.get_toc()]
        sizes = {}
        for count, index in enumerate(selected):
            _check(cancel)
            page = doc[index]
            # Normalize rotation in-memory so all bboxes share the same coordinate system.
            rotation = page.rotation
            if rotation:
                page.set_rotation(0)
            progress(count, len(selected), f"正在分析第 {index + 1} 页 / 共 {len(doc)} 页")
            sizes[index + 1] = (page.rect.width, page.rect.height)
            tables = _tables(page, options.table_strategy, result.warnings)
            native = _native_text(page, tables)
            blocks = native + tables
            text_chars = sum(len(b.text) for b in native) + sum(len(str(c or "")) for t in tables for r in t.rows for c in r)
            scan = text_chars < 30
            image_regions = figure_regions(page, native, tables)
            # A whole-page reference also captures vector-only pages when OCR is needed.
            if scan and options.ocr != "off":
                image_regions = [(page.rect, None)]
            if scan and options.ocr == "off":
                result.warnings.append(f"第 {index + 1} 页可提取文字很少，OCR 已关闭，扫描文字可能缺失。")
            for number, (region, caption) in enumerate(image_regions, 1):
                _check(cancel)
                asset = f"assets/page_{index + 1:04d}_image_{number:03d}.png"
                try:
                    width, height = _render(page, region, work / asset, options.dpi)
                except Exception as exc:
                    result.warnings.append(f"第 {index + 1} 页图片 {number} 导出失败：{exc}")
                    continue
                image_block = Block("image", index + 1, _box(region), asset=asset,
                                    meta={"width": width, "height": height, "scan_reference": scan,
                                          "original_page_rotation": rotation, "rendered_crop": True,
                                          "caption": caption.text if caption else ""})
                blocks.append(image_block)
                if caption:
                    caption.meta["caption_for"] = asset
                if not scan:
                    for block in native:
                        if block is not caption and inside(block.bbox, region, .8):
                            block.meta["image"] = asset
                            block.meta["figure_label"] = True
                has_raster = any(inside(info["bbox"], region, .5) for info in page.get_image_info())
                needs_ocr = options.ocr == "all" or (options.ocr == "auto" and (scan or has_raster))
                if not needs_ocr or ocr_failed:
                    continue
                progress(count, len(selected), f"第 {index + 1} 页：正在本地识别图片 {number}")
                try:
                    lines = ocr.recognize(work / asset)
                    image_block.meta["ocr_status"] = "complete" if lines else "no_text"
                    def mapped(box):
                        x0, y0, x1, y1 = box
                        return _box([region.x0 + x0 * region.width / width, region.y0 + y0 * region.height / height,
                                     region.x0 + x1 * region.width / width, region.y0 + y1 * region.height / height])
                    grids, consumed = detect_grid_tables(work / asset, lines) if scan else ([], set())
                    for grid in grids:
                        blocks.append(Block("table", index + 1, mapped(grid["bbox"]), rows=grid["rows"], source="ocr",
                                            confidence=grid["confidence"], meta={"image": asset, "strategy": "ocr_grid"}))
                    for line_index, line in enumerate(lines):
                        if line_index in consumed:
                            continue
                        bbox = mapped(line["bbox"])
                        # Avoid duplicating an existing text layer or an extracted table.
                        normalized = re.sub(r"\s+", "", line["text"]).casefold()
                        duplicate = any(inside(bbox, b.bbox, .45) and normalized in re.sub(r"\s+", "", b.text).casefold() for b in native)
                        if duplicate or any(inside(bbox, t.bbox) for t in tables):
                            continue
                        blocks.append(Block("ocr_text", index + 1, _box(bbox), line["text"], source="ocr",
                                            confidence=line["confidence"], meta={"image": asset, "figure_label": not scan}))
                    if scan and not lines:
                        result.warnings.append(f"第 {index + 1} 页 OCR 未识别到文字，请查看原页图片。")
                except Exception as exc:
                    ocr_failed = True
                    image_block.meta["ocr_status"] = "failed"
                    result.warnings.append(f"OCR 不可用，已保留图片并继续提取；后续图片未执行 OCR。原因：{exc}")
            if scan and options.ocr != "off":
                result.warnings.append(f"第 {index + 1} 页按扫描页处理。规则网格表格可转为表格；无边框、合并单元格及标题层级需要人工校对。")
            result.blocks.extend(blocks)
        _check(cancel)
        classify(result.blocks, sizes, result.outline)
        ordered = []
        for page_index in selected:
            page_blocks = [b for b in result.blocks if b.page == page_index + 1]
            attached = [b for b in page_blocks if b.meta.get("figure_label") or b.meta.get("caption_for")]
            top = reading_order([b for b in page_blocks if b not in attached], options.order)
            for block in top:
                ordered.append(block)
                if block.kind == "image":
                    ordered.extend(sorted([b for b in attached if b.meta.get("image", b.meta.get("caption_for")) == block.asset], key=lambda b: (b.bbox[1], b.bbox[0])))
        for i, block in enumerate(ordered, 1):
            block.id = f"b{i:06d}"
            if "\ufffd" in block.text:
                result.warnings.append(f"第 {block.page} 页 {block.id} 含无法解码字符，可尝试对原页扫描件使用 OCR。")
        result.blocks = ordered
        result.warnings = list(dict.fromkeys(result.warnings))
        progress(len(selected), len(selected), "正在生成可编辑文档")
        return result


def run(source, output, options=None, progress=None, cancel=None):
    """Publish matching Markdown/HTML trees, with archived source PDFs."""
    from .export import export_all
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"找不到 PDF：{source}")
    output.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", source.stem).strip(" .")[:90] or "document"
    # Inherit the output directory's ACL on Windows (mkdtemp's private ACL
    # can exclude a restricted process token even when it created the folder).
    work = output / f".extract-{uuid.uuid4().hex}"
    work.mkdir()
    reserved = []
    source_copies = []
    try:
        result = extract(source, work, options or Options(), progress, cancel)
        _check(cancel)
        for fmt in ("md", "html"):
            (output / f"output_{fmt}" / "source").mkdir(parents=True, exist_ok=True)
        for number in range(10000):
            name = safe_name if number == 0 else f"{safe_name}_{number + 1}"
            destinations = [output / f"output_{fmt}" / name for fmt in ("md", "html")]
            if any(p.exists() for p in destinations) or any((output / f"output_{fmt}" / "source" / f"{name}.pdf").exists() for fmt in ("md", "html")):
                continue
            try:
                for destination in destinations:
                    destination.mkdir()
                    reserved.append(destination)
                break
            except FileExistsError:
                for path in reserved:
                    path.rmdir()
                reserved.clear()
                continue
        else:
            raise ValueError("输出目录中同名结果过多。")
        export_all(result, work / "publish", work / "assets", name)
        _check(cancel)
        for fmt, destination in zip(("md", "html"), destinations):
            for child in (work / "publish" / fmt).iterdir():
                shutil.move(str(child), destination / child.name)
            archived = output / f"output_{fmt}" / "source" / f"{name}.pdf"
            # Exclusive creation protects unrelated files even in concurrent runs.
            with source.open("rb") as reader, archived.open("xb") as writer:
                source_copies.append(archived)
                shutil.copyfileobj(reader, writer)
        return destinations[1], result
    except BaseException:
        for path in reserved:
            shutil.rmtree(path, ignore_errors=True)
        for path in source_copies:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
