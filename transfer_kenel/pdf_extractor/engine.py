
from .usage import tracked, span, trace_headers
from collections import Counter
from contextlib import ExitStack
import multiprocessing
# Register Windows pipe reducers before callers install temporary import maps.
# Lazy registration inside patch.dict(sys.modules) can leave stale reducers
# after that map is restored, breaking the next otherwise valid spawn.
import multiprocessing.connection
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import uuid

import pymupdf as fitz

from .layout import classify, inside, reading_order
from .model import Block, Options, Result, parse_pages
from .ocr import LocalOCR, detect_grid_tables, duplicates_native
from .text import visual_lines, reflow, merge_code_blocks, prose_quotes, LIST_ITEM
from .figures import figure_regions
from .tables import native_tables


class Cancelled(Exception):
    pass


def _table_worker(source, password, strategy, jobs, results, trace):
    """Own the only table Document in a separate, one-CPU process.

    No PyMuPDF document/page is sent through either bounded queue. Process CPU
    in these spans belongs to the table worker and is excluded from host CPU.
    """
    os.environ['PDFTRANSFER_LOG_ROLE'] = 'table-worker'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        if hasattr(os, 'sched_getaffinity'):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        elif os.name == 'nt':
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            process = kernel.GetCurrentProcess()
            process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
            if kernel.GetProcessAffinityMask(ctypes.c_void_p(process), ctypes.byref(process_mask), ctypes.byref(system_mask)):
                available = process_mask.value
                kernel.SetProcessAffinityMask(ctypes.c_void_p(process), ctypes.c_size_t(available & -available))
    except (OSError, AttributeError, ValueError):
        pass  # A one-process queue remains usable when OS affinity is unavailable.
    try:
        with fitz.open(source) as document:
            if document.needs_pass and not document.authenticate(password):
                raise ValueError('PDF authentication failed')
            while True:
                index = jobs.get()
                if index is None:
                    return
                warnings = []
                with span('reading.table_worker', remote=trace, page=index + 1) as usage:
                    page = document[index]
                    if page.rotation:
                        page.set_rotation(0)
                    tables = _tables(page, strategy, warnings)
                    if warnings and not tables:
                        usage.status = 'error'
                results.put((index, tables, warnings))
    except BaseException:
        # Error details may contain private paths. The host retains the original
        # page and reports a stable warning rather than exposing worker errors.
        results.put((-1, [], ['原生表格独立进程异常，表格未确认恢复。']))


class _TableQueue:
    """Feed pages without blocking text/image work; bound both IPC queues."""
    def __init__(self, source, selected, options):
        context = multiprocessing.get_context('spawn')
        self.jobs = context.Queue(maxsize=2)
        self.results = context.Queue(maxsize=2)
        self.stop = threading.Event()
        self.ready = {}
        headers = trace_headers()
        trace = dict(traceId=headers.get('X-Pdftransfer-Trace'),
                     parentId=headers.get('X-Pdftransfer-Parent'),
                     feature=headers.get('X-Pdftransfer-Feature'))
        self.process = context.Process(target=_table_worker,
            args=(str(source), options.password, options.table_strategy, self.jobs, self.results, trace),
            name='pdftransfer-native-tables', daemon=True)
        self.selected = list(selected)

    def __enter__(self):
        try:
            self.process.start()
        except BaseException:
            for channel in (self.jobs, self.results):
                channel.close()
            self.process.close()
            raise
        self.feeder = threading.Thread(target=self._feed, name='pdftransfer-table-feeder', daemon=True)
        self.feeder.start()
        return self

    def _feed(self):
        for index in [*self.selected, None]:
            while not self.stop.is_set():
                try:
                    self.jobs.put(index, timeout=.1)
                    break
                except queue.Full:
                    continue
            if self.stop.is_set():
                return

    def take(self, index, cancel=None):
        while index not in self.ready:
            _check(cancel)
            try:
                completed, tables, warnings = self.results.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive():
                    return [], ['原生表格独立进程未返回本页，已保留原页供校对。']
                continue
            if completed < 0:
                return [], warnings
            self.ready[completed] = (tables, warnings)
        return self.ready.pop(index)

    def __exit__(self, kind, error, traceback):
        self.stop.set()
        self.feeder.join(timeout=1)
        self.process.join(timeout=1 if kind else 2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        for channel in (self.jobs, self.results):
            channel.cancel_join_thread()
            channel.close()
        self.process.close()
        return False


def _check(cancel):
    if cancel and cancel.is_set():
        raise Cancelled("已取消提取。")


def _box(rect):
    return [round(float(v), 2) for v in rect]


@tracked('reading.native_text')
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
                                      "native_style": True,
                                      "font": group[0].get("runs", [{}])[0].get("font", ""),
                                      "color": group[0].get("runs", [{}])[0].get("color", "#000000"),
                                      "italic": all(run.get("italic", False) for line in group for run in line.get("runs", [])),
                                      "vertical": any(line["vertical"] for line in group), "code": code,
                                      "text_lines": [dict(line) for line in group],
                                      "code_lines": [{"text": line["text"], "bbox": line["bbox"]} for line in group] if code else []}))
            group.clear()

        for line in prose_quotes(visual_lines(raw)):
            if any(inside(line["bbox"], table.bbox) for table in tables):
                flush()
                continue
            if not line["text"]:
                continue
            # Preserve paragraph boundaries, but split typography changes and margins.
            margin = line["bbox"][3] < page.rect.height * .09 or line["bbox"][1] > page.rect.height * .90
            item_start = not line["code"] and LIST_ITEM.match(line["text"])
            # Separate distant labels that share a baseline (e.g. two panels).
            column_label = group and abs(group[-1]["bbox"][1] - line["bbox"][1]) < 1.6 and max(
                line["bbox"][0] - group[-1]["bbox"][2], group[-1]["bbox"][0] - line["bbox"][2]
            ) > max(12, line["size"] * 2)
            if group and (item_start or column_label or abs(group[-1]["size"] - line["size"]) > 1 or group[-1]["code"] != line["code"] or margin):
                flush()
            group.append(line)
            if margin:
                flush()
        flush()
    return merge_code_blocks(result)


def _figure_text(blocks, mode):
    """Keep native line coordinates and use a figure-sized column gutter."""
    expanded = []
    captions = []
    for block in blocks:
        if block.meta.get("caption_for"):
            captions.append(block)
            continue
        lines = block.meta.get("text_lines") if block.meta.get("figure_label") else None
        if not lines:
            expanded.append(block)
            continue
        for line in lines:
            text = line["text"]
            if block.meta.get("code"):
                unit = max(3, block.meta["font_size"] * .5)
                text = " " * max(0, round((line["bbox"][0] - block.bbox[0]) / unit)) + text
            meta = {k: v for k, v in block.meta.items() if k not in ("text_lines", "code_lines")}
            expanded.append(Block(block.kind, block.page, _box(line["bbox"]), text,
                                  source=block.source, confidence=block.confidence, meta=meta))
    # A full-width caption is outside the image: it must not bridge its columns.
    positioned = reading_order(expanded, mode, column_gap=5, min_column_width=30, align_rows=True)
    rows = []
    for block in positioned:
        prefix = rows[-1] if rows else None
        if prefix and prefix.source == block.source == "ocr" and re.fullmatch(r"[+\-–—]{1,3}", prefix.text):
            overlap = min(prefix.bbox[3], block.bbox[3]) - max(prefix.bbox[1], block.bbox[1])
            small_height = min(prefix.bbox[3] - prefix.bbox[1], block.bbox[3] - block.bbox[1])
            gap = block.bbox[0] - prefix.bbox[2]
            if small_height > 0 and overlap >= small_height * .5 and 0 <= gap <= max(18, (block.bbox[3] - block.bbox[1]) * 3):
                scores = [b.confidence for b in (prefix, block) if b.confidence is not None]
                rows[-1] = Block(block.kind, block.page,
                                 [prefix.bbox[0], min(prefix.bbox[1], block.bbox[1]), block.bbox[2], max(prefix.bbox[3], block.bbox[3])],
                                 prefix.text + " " + block.text, source="ocr",
                                 confidence=min(scores) if scores else None, meta=dict(block.meta))
                continue
        rows.append(block)
    return rows + sorted(captions, key=lambda b: (b.bbox[1], b.bbox[0]))


@tracked('reading.tables')
def _tables(page, strategy, warnings):
    try:
        return native_tables(page, strategy, warnings)
    except Exception as exc:
        warnings.append(f"第 {page.number + 1} 页表格检测失败，保留为普通文字：{exc}")
        return []


def _scan_page(page, native, tables):
    """Native watermark text must not hide a large scanned document image."""
    text_chars = sum(len(b.text) for b in native) + sum(len(str(c or "")) for t in tables for row in t.rows for c in row)
    if text_chars < 30:
        return True
    image_area = max((fitz.Rect(info['bbox']).get_area() for info in page.get_image_info()), default=0)
    if image_area < page.rect.get_area() * .2:
        return False
    substantial_body = [b for b in native if len(b.text) >= 30 and b.bbox[1] > page.rect.height*.08
                        and b.bbox[3] < page.rect.height*.9]
    return not substantial_body and not tables


def _structure_scan_page(page, native, tables):
    """Keep ordinary pictures in an editable PDF out of automatic OCR."""
    images = page.get_image_info()
    usable = [b for b in native if b.text.strip() and '\ufffd' not in b.text
              and not any(ord(c) < 32 and c not in '\n\r\t' for c in b.text)]
    if usable and not images:
        return False
    # A short editable heading may legitimately sit in the page margin. Small
    # embedded pictures do not make it a watermark on a scanned document; keep
    # the larger-image guard below for actual scans carrying native headers.
    largest_image = max(((fitz.Rect(info['bbox']) & page.rect).get_area() for info in images), default=0)
    if usable and largest_image < page.rect.get_area() * .2:
        return False
    # Preserve the full-scan/watermark guard. Even a short native title outside
    # a picture is useful evidence; image size alone cannot force whole-page OCR.
    if any(b.bbox[1] > page.rect.height * .08 and b.bbox[3] < page.rect.height * .9
           and not any(inside(b.bbox, info['bbox'], .8) for info in images) for b in usable):
        return False
    return _scan_page(page, usable, tables)


def _table_preview(page, block, assets, number, dpi, warnings):
    name = f"page_{page.number+1:04d}_table_{number:03d}.png"
    rect = (fitz.Rect(block.bbox) + (-2, -2, 2, 2)) & page.rect
    try:
        _render(page, rect, assets/name, dpi)
        block.meta['preview_asset'] = 'assets/' + name
    except Exception as exc:
        warnings.append(f'第 {page.number+1} 页表格原图生成失败：{exc}')


@tracked('reading.render')
def _render(page, rect, path, dpi):
    # Bound memory for unusually large posters while preserving coordinate mapping.
    scale = min(dpi / 72, 3500 / max(rect.width, rect.height))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    pix.save(path)
    return pix.width, pix.height


def _join_native_tables(page, previous, tables, work, options, warnings):
    """Reconcile after concurrent table work, preserving text/table ownership."""
    native = _native_text(page, tables)
    for number, table in enumerate(tables, 1):
        _table_preview(page, table, work / 'assets', number, options.dpi, warnings)
    old_images = [b for b in previous if b.kind == 'image']
    scan = any(b.meta.get('scan_reference') for b in old_images)
    images = []
    if scan:
        images = old_images
    else:
        for number, (region, caption) in enumerate(figure_regions(page, native, tables), 1):
            asset = f'assets/page_{page.number + 1:04d}_image_{number:03d}.png'
            existing = next((b for b in old_images if b.bbox == _box(region)), None)
            if existing and existing.asset == asset:
                image = existing
            else:
                try:
                    width, height = _render(page, region, work / asset, options.dpi)
                except Exception as exc:
                    warnings.append(f'第 {page.number+1} 页图片重组失败：{exc}')
                    continue
                image = Block('image', page.number+1, _box(region), asset=asset,
                              meta=dict(width=width, height=height, scan_reference=False,
                                        rendered_crop=True, caption=caption.text if caption else ''))
            images.append(image)
            if caption:
                caption.meta['caption_for'] = asset
            for block in native:
                if block is not caption and inside(block.bbox, region, .8):
                    block.meta.update(image=asset, figure_label=True)
    ocr = [b for b in previous if b.source == 'ocr' and b.kind != 'image'
           and not any(inside(b.bbox, table.bbox, .8) for table in tables)]
    for block in ocr:
        if block.meta.get('figure_label'):
            image = next((b for b in images if inside(block.bbox, b.bbox, .8)), None)
            if image:
                block.meta['image'] = image.asset
            else:
                block.meta.pop('figure_label', None)
                block.meta.pop('image', None)
    return native + tables + images + ocr


@tracked('reading.extract')
def extract(source: Path, work: Path, options: Options, progress=None, cancel: threading.Event | None = None) -> Result:
    progress = progress or (lambda done, total, message: None)
    if options.ocr not in ("off", "auto", "all"):
        raise ValueError("OCR 模式无效。")
    if options.order not in ("columns", "rows") or options.table_strategy not in ("auto", "lines_strict", "lines", "text"):
        raise ValueError("版面或表格模式无效。")
    if not 72 <= options.dpi <= 400:
        raise ValueError("图片分辨率应在 72–400 DPI 之间。")
    if options.recognition_backend not in ("legacy", "v3"):
        raise ValueError("识别后端应为 legacy 或 v3。")
    assets = work / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ocr = LocalOCR()
    ocr_failed = False
    with fitz.open(source) as doc, ExitStack() as resources:
        if not doc.is_pdf:
            raise ValueError("请选择 PDF 文件。")
        if doc.needs_pass and not doc.authenticate(options.password):
            raise ValueError("PDF 已加密，请提供正确的密码。")
        selected = parse_pages(options.pages, len(doc))
        if not selected:
            raise ValueError("PDF 没有可提取的页面。")
        result = Result(str(source.resolve()), len(doc), [p + 1 for p in selected])
        result.outline = [{"level": level, "title": title, "page": page} for level, title, page in doc.get_toc()]
        _check(cancel)
        table_jobs = resources.enter_context(_TableQueue(source, selected, options))
        semantic_pages = set()
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
            result.page_sizes[index + 1] = [float(page.rect.width), float(page.rect.height)]
            # Native tables run in their own bounded process queue. Text and
            # images are extracted immediately; reconcile after all pages.
            tables = []
            native = _native_text(page, tables)
            blocks = native + tables
            scan = (_structure_scan_page(page, native, tables) if options.ocr == 'auto'
                    else _scan_page(page, native, tables))
            if options.recognition_backend == "v3":
                # Explicitly recognizing every image must keep the selected
                # backend even on otherwise native pages. The semantic adapter
                # reuses trustworthy native characters after the model pass.
                scan = options.ocr == "all" or _structure_scan_page(page, native, tables)
            if scan and options.ocr != "off" and options.recognition_backend == "v3":
                from .structure_reading import extract_structure_page
                progress(count, len(selected), f"第 {index + 1} 页：本地 V3 版面、文字与表格识别")
                blocks = extract_structure_page(page, work, options.dpi, result.warnings,
                                                original_rotation=rotation, native=native)
                _check(cancel)
                result.blocks.extend(blocks)
                semantic_pages.add(index)
                continue
            for table_number, table in enumerate(tables, 1):
                _table_preview(page, table, assets, table_number, options.dpi, result.warnings)
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
                # Reliable native text is retained verbatim. The browser's V3
                # automatic path does not OCR pictures in an editable PDF.
                if not scan and options.ocr == "auto":
                    needs_ocr = False
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
                        table = Block("table", index + 1, mapped(grid["bbox"]), rows=grid["rows"], source="ocr",
                                      confidence=grid["confidence"], meta={"image": asset, "strategy": "ocr_grid"})
                        _table_preview(page, table, assets, len(tables)+len([b for b in blocks if b.kind == 'table' and b.source == 'ocr'])+1, options.dpi, result.warnings)
                        blocks.append(table)
                    for line_index, line in enumerate(lines):
                        if line_index in consumed:
                            continue
                        bbox = mapped(line["bbox"])
                        # Avoid duplicating an existing text layer or an extracted table.
                        duplicate = duplicates_native(bbox, line["text"], native)
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
        progress(len(selected), len(selected), '正文与图片提取完成，正在汇合独立表格队列')
        with span('reading.table_queue_wait'):
            for index in selected:
                tables, warnings = table_jobs.take(index, cancel)
                if index in semantic_pages:
                    continue  # The V3 page already owns its conditional tables.
                result.warnings.extend(warnings)
                page = doc[index]
                if page.rotation:
                    page.set_rotation(0)
                if tables:
                    previous = [b for b in result.blocks if b.page == index+1]
                    merged = _join_native_tables(page, previous, tables, work, options, result.warnings)
                    result.blocks = [b for b in result.blocks if b.page != index+1] + merged
                elif warnings:
                    # Keep an intact visual reference if native table recovery
                    # failed, instead of claiming that no table exists.
                    asset = f'assets/page_{index+1:04d}_table_unresolved.png'
                    try:
                        width, height = _render(page, page.rect, work / asset, options.dpi)
                        result.blocks.append(Block('image', index+1, _box(page.rect), asset=asset,
                            meta=dict(width=width, height=height, table_recovery='unresolved',
                                      structure_warning='原生表格未可靠恢复，请对照原页。')))
                    except Exception:
                        result.warnings.append(f'第 {index+1} 页表格失败参考图无法生成，请查阅源 PDF。')
        _check(cancel)
        classify([b for b in result.blocks if not b.meta.get("semantic_label")], sizes, result.outline)
        ordered = []
        for page_index in selected:
            page_blocks = [b for b in result.blocks if b.page == page_index + 1]
            if page_blocks and all(b.meta.get("semantic_label") for b in page_blocks):
                # V3 has already resolved reading order and figure ownership.
                ordered.extend(page_blocks)
                continue
            attached = [b for b in page_blocks if b.meta.get("figure_label") or b.meta.get("caption_for")]
            top = reading_order([b for b in page_blocks if b not in attached], options.order)
            for block in top:
                ordered.append(block)
                if block.kind == "image":
                    children = [b for b in attached if b.meta.get("image", b.meta.get("caption_for")) == block.asset]
                    ordered.extend(_figure_text(children, options.order))
        for i, block in enumerate(ordered, 1):
            block.id = f"b{i:06d}"
            if "\ufffd" in block.text:
                result.warnings.append(f"第 {block.page} 页 {block.id} 含无法解码字符，可尝试对原页扫描件使用 OCR。")
        result.blocks = ordered
        result.warnings = list(dict.fromkeys(result.warnings))
        progress(len(selected), len(selected), "正在生成可编辑文档")
        return result


@tracked('reading.convert', feature=lambda v: 'reading.' + getattr(v.get('options'), 'output_format', 'both'))
def run(source, output, options=None, progress=None, cancel=None):
    """Publish only requested reading trees, with archived source PDFs."""
    from .export import export_all
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    options = options or Options()
    if options.output_format not in ("html", "md", "both"):
        raise ValueError("输出格式应为 html、md 或 both。")
    formats = ("md", "html") if options.output_format == "both" else (options.output_format,)
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
        result = extract(source, work, options, progress, cancel)
        _check(cancel)
        for fmt in formats:
            (output / f"output_{fmt}" / "source").mkdir(parents=True, exist_ok=True)
        for number in range(10000):
            name = safe_name if number == 0 else f"{safe_name}_{number + 1}"
            destinations = [output / f"output_{fmt}" / name for fmt in formats]
            if any(p.exists() for p in destinations) or any((output / f"output_{fmt}" / "source" / f"{name}.pdf").exists() for fmt in formats):
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
        export_all(result, work / "publish", work / "assets", name, formats=formats)
        _check(cancel)
        for fmt, destination in zip(formats, destinations):
            for child in (work / "publish" / fmt).iterdir():
                shutil.move(str(child), destination / child.name)
            archived = output / f"output_{fmt}" / "source" / f"{name}.pdf"
            # Exclusive creation protects unrelated files even in concurrent runs.
            with source.open("rb") as reader, archived.open("xb") as writer:
                source_copies.append(archived)
                shutil.copyfileobj(reader, writer)
        return destinations[-1], result
    except BaseException:
        for path in reserved:
            shutil.rmtree(path, ignore_errors=True)
        for path in source_copies:
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
