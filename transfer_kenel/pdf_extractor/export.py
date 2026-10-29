"""Readable Markdown and HTML, with mirrored per-document directories."""

from .usage import tracked
import html
import json
import math
import re
import shutil
from pathlib import Path
from urllib.parse import quote

from .text import reflow

LABELS = {"text": "正文", "heading": "标题目录", "table": "表格", "image": "图片", "page_number": "页码", "header_footer": "页眉页脚", "ocr_text": "OCR 文字"}
FOLDERS = {"text": "text", "heading": "heading", "table": "table", "image": "graph", "page_number": "page_number", "header_footer": "header_footer", "ocr_text": "ocr"}


def md_text(value):
    value = html.escape(str(value or ""), quote=False)
    value = value.replace("\\", "\\\\").replace("`", "\\`").replace("*", "\\*").replace("_", "\\_")
    # Ordinary [citations] stay literal. Escape only constructs parsed as links.
    value = re.sub(r"\[([^\]\n]*)\](?=[(\[])", lambda m: "\\[" + m[1] + "\\]", value)
    return re.sub(r"(?m)^(\s*)([#>+\-]|\d+[.)])(?=\s)", r"\1\\\2", value)


def code_fence(value):
    length = max((len(s) for s in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, length + 1)
    return f"{fence}text\n{value}\n{fence}"


def attachments(result, block):
    # Scan OCR remains in the reading flow; its copy here is a reference appendix.
    return [b for b in result.blocks if block.asset and b.page == block.page
            and b.meta.get("image") == block.asset and not b.meta.get("caption_for")
            and b.kind not in ("image", "table")
            and (b.meta.get("figure_label") or block.meta.get("scan_reference")
                 and (b.source == "ocr" or b.kind == "ocr_text"))]


def top_blocks(result):
    return [b for b in result.blocks if not b.meta.get("figure_label") and not b.meta.get("caption_for")]


def image_name(block):
    return Path(block.asset).name


def _bbox(value):
    """Reject unusable geometry without emitting NaN or invalid CSS."""
    try:
        box = [float(v) for v in value]
        if (len(box) == 4 and all(math.isfinite(v) for v in box)
                and box[2] > box[0] and box[3] > box[1]):
            return box
    except (TypeError, ValueError, OverflowError):
        pass
    return []


def _coords(box):
    return json.dumps(box, separators=(",", ":"), allow_nan=False)


def _position_attrs(block):
    return (f'data-page="{html.escape(str(block.page), quote=True)}" '
            f'data-bbox="{_coords(_bbox(block.bbox))}" '
            f'data-source="{html.escape(block.source, quote=True)}"')


def _page_size(result, page):
    try:
        value = result.page_sizes.get(page, result.page_sizes.get(str(page), []))
        size = [float(number) for number in value]
        return size if len(size) == 2 and all(math.isfinite(v) and v > 0 for v in size) else []
    except (AttributeError, TypeError, ValueError, OverflowError):
        return []


def _geometry_style(box, prefix='box', origin=(0, 0)):
    box = _bbox(box)
    if not box:
        return ''
    x, y, right, bottom = box
    return (f'--{prefix}-left:{x-origin[0]:.8f}pt;--{prefix}-top:{y-origin[1]:.8f}pt;'
            f'--{prefix}-width:{right-x:.8f}pt;--{prefix}-height:{bottom-y:.8f}pt')


def _image_lines(block, children):
    """Use PDF page coordinates, never raster pixels, as the common frame."""
    image_box = _bbox(block.bbox)
    lines = []
    for child in children:
        if child.page != block.page:
            continue
        for line in child.meta.get("text_lines") or [{"text": child.text, "bbox": child.bbox}]:
            box = _bbox(line.get("bbox"))
            relative = []
            if box and image_box:
                x, y, right, bottom = image_box
                relative = [(box[0] - x) / (right - x) * 100,
                            (box[1] - y) / (bottom - y) * 100,
                            (box[2] - x) / (right - x) * 100,
                            (box[3] - y) / (bottom - y) * 100]
            visible = bool(relative and relative[2] > 0 and relative[3] > 0
                           and relative[0] < 100 and relative[1] < 100)
            label = "OCR" if child.source == "ocr" or child.kind == "ocr_text" else "原文文字"
            if child.confidence is not None:
                label += f" · {child.confidence:.0%}"
            lines.append({"key": str(len(lines) + 1), "text": str(line.get("text", "")),
                          "bbox": box, "relative": relative, "visible": visible,
                          "label": label, "block": child})
    return lines


def _line_attrs(line):
    child = line["block"]
    return (f'data-line="{line["key"]}" data-page="{html.escape(str(child.page), quote=True)}" '
            f'data-bbox="{_coords(line["bbox"])}" '
            f'data-relative-bbox="{_coords(line["relative"])}" '
            f'data-block-id="{html.escape(child.id, quote=True)}" '
            f'data-source="{html.escape(child.source, quote=True)}"')


def _coordinate_label(line):
    relative = ", ".join(f"{v:.4f}".rstrip("0").rstrip(".") for v in line["relative"])
    return (f'第 {line["block"].page} 页 · PDF bbox {_coords(line["bbox"])} · '
            f'原图 bbox (%) [{relative}]' + ("" if line["visible"] else " · 图内位置不可用"))


def _transcript(lines):
    rows = []
    for line in lines:
        disabled = "" if line["visible"] else " disabled"
        rows.append(f'<div class="transcript-row" {_line_attrs(line)}>'
                    f'<div class="transcript-meta" contenteditable="false">'
                    f'<button type="button" class="locate-text"{disabled}>定位</button>'
                    f'<small>{html.escape(line["label"])}</small>'
                    f'<code>{html.escape(_coordinate_label(line))}</code></div>'
                    f'<span class="transcript-text" contenteditable="true">{html.escape(line["text"])}</span></div>')
    return ('<details class="figure-text" contenteditable="false"><summary contenteditable="false">图中文字 · 点击转录行定位，可直接编辑</summary>'
            '<p class="coordinate-note" contenteditable="false">坐标顺序均为 [左, 上, 右, 下]；PDF 页绝对坐标，原图左上角为百分比原点。</p>'
            '<div class="transcript">' + "".join(rows) + '</div></details>')


def _image_html(block, image_prefix, children, caption=None):
    esc = html.escape
    url = esc(quote(image_prefix + image_name(block), safe="/"), quote=True)
    lines = _image_lines(block, children)
    boxes = []
    for line in lines:
        if not line["visible"]:
            continue
        left, top, right, bottom = line["relative"]
        style = f"left:{left:.8f}%;top:{top:.8f}%;width:{right-left:.8f}%;height:{bottom-top:.8f}%"
        # Text is hydrated from the editable transcript using textContent. There
        # is one stored transcription, and the bitmap never gets overprinted.
        boxes.append(f'<span class="image-text-box" {_line_attrs(line)} style="{style}">'
                     '<svg viewBox="0 0 100 100" preserveAspectRatio="none">'
                     '<text x="0" y="80" font-size="90" textLength="100" lengthAdjust="spacingAndGlyphs"></text>'
                     '</svg></span>')
    body = ('<figure><div class="image-stage" contenteditable="false">'
            f'<img src="{url}" alt="第 {block.page} 页图片" loading="lazy">')
    if boxes:
        body += '<div class="image-text-layer" hidden>' + "".join(boxes) + '</div>'
    body += '</div><div class="image-tools" contenteditable="false">'
    if boxes:
        body += '<button type="button" class="toggle-image-text" aria-pressed="false">图上文字层（选择 / 定位）</button>'
    body += f'<a href="{url}" target="_blank" rel="noopener">查看原图</a></div>'
    if block.meta.get("caption"):
        if caption is not None and _bbox(caption.bbox) and _bbox(block.bbox):
            style = _geometry_style(caption.bbox, 'caption', block.bbox[:2])
            content = _native_markup(caption) if caption.meta.get('native_style') else esc(caption.text)
            body += f'<figcaption class="source-caption" {_position_attrs(caption)} style="{style}">{content}</figcaption>'
        else:
            body += f'<figcaption>{esc(block.meta["caption"])}</figcaption>'
    return body + '</figure>' + (_transcript(lines) if lines else "")


def table_html(block):
    """Render geometric cells instead of padding merged cells with empty TDs."""
    esc = html.escape
    widths = block.meta.get('column_widths', [])
    cells = block.meta.get('grid_cells')
    if not cells:
        cells = [{'row': r, 'col': c, 'rowspan': 1, 'colspan': 1, 'text': str(value or '')}
                 for r, row in enumerate(block.rows) for c, value in enumerate(row)]
    cols = ''
    if widths and sum(widths) > 0:
        cols = '<colgroup>' + ''.join(f'<col style="width:{w/sum(widths)*100:.3f}%">' for w in widths) + '</colgroup>'
    rows = []
    header_rows = block.meta.get('header_rows', 1)
    for r in range(len(block.rows)):
        body = []
        for cell in sorted((c for c in cells if c['row'] == r), key=lambda c: c['col']):
            tag = 'th' if r < header_rows else 'td'
            span = f' rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}"'
            borders = cell.get('borders')
            style = ''
            if borders is not None:
                style = ' style="border:0;' + ''.join(f'border-{side}:1px solid #89959b;' for side, present in borders.items() if present) + '"'
            body.append(f'<{tag}{span}{style}>{esc(str(cell.get("text", ""))).replace(chr(10), "<br>")}</{tag}>')
        rows.append('<tr>' + ''.join(body) + '</tr>')
    return '<div class="table-scroll"><table class="structured-table">' + cols + '<tbody>' + ''.join(rows) + '</tbody></table></div>'


def table_reference(block, prefix):
    warning = block.meta.get('structure_warning', '')
    notice = f'<p class="table-warning">{html.escape(warning)}</p>' if warning else ''
    if block.meta.get('structure_inferred'):
        notice += '<p class="table-note" contenteditable="false">行列由原文位置推断，请对照原表校对。</p>'
    asset = block.meta.get('preview_asset')
    if not asset:
        return notice
    url = html.escape(prefix + Path(asset).name, quote=True)
    return notice + f'<details class="table-reference"><summary contenteditable="false">原表截图 · 核对形状与排版</summary><img src="{url}" alt="第 {block.page} 页原表截图" loading="lazy" contenteditable="false"></details>'


def _native_markup(block):
    """Retain PDF span styles and coordinates without trusting font names as CSS."""
    lines = block.meta.get("text_lines") or []
    rendered = []
    for line in lines:
        spans = []
        for run in line.get("runs") or []:
            rules = []
            size = run.get("size")
            if isinstance(size, (int, float)) and math.isfinite(size) and 1 <= size <= 500:
                rules.append(f'font-size:{size:.2f}pt')
            color = run.get("color", "")
            if re.fullmatch(r'#[0-9a-fA-F]{6}', color):
                rules.append('color:' + color)
            font = str(run.get("font", ""))
            # PDF font identifiers are metadata; availability of the original
            # font on the viewing computer is not assumed.
            if font and re.fullmatch(r'[\w .,+-]{1,160}', font):
                rules.append('font-family:"' + font + '"')
            if run.get("bold"):
                rules.append('font-weight:bold')
            if run.get("italic"):
                rules.append('font-style:italic')
            local = _geometry_style(run.get('bbox'), 'run', block.bbox[:2]) if _bbox(block.bbox) else ''
            attributes = (f'data-bbox="{_coords(_bbox(run.get("bbox")))}" '
                          f'data-pdf-font="{html.escape(font, quote=True)}" '
                          f'style="{html.escape(";".join(rules) + ";" + local, quote=True)}"')
            spans.append(html.escape(str(run.get('separator', ''))) +
                         f'<span class="native-run" {attributes}>{html.escape(str(run.get("text", "")))}</span>')
        rendered.append(''.join(spans) if spans else html.escape(line.get("text", "")))
    return '<br>'.join(rendered) if rendered else html.escape(block.text)


def markdown(block, image_prefix="../graph/", children=()):
    if block.kind == "table":
        # Markdown pipe tables cannot express row/column spans or source borders.
        # Inline HTML retains these in renderers that permit HTML.
        return table_html(block) + table_reference(block, image_prefix)
    if block.kind == "image":
        caption = block.meta.get("caption", "")
        body = f"![第 {block.page} 页图片](<{quote(image_prefix + image_name(block), safe='/')}>)"
        if caption:
            body += f"\n\n*{md_text(caption)}*"
        lines = _image_lines(block, children)
        if lines:
            body += (f'\n\n<details class="figure-text" {_position_attrs(block)}>'
                     '<summary>图中文字坐标附录（可编辑；OCR 请核对原图）</summary>'
                     '<p>坐标顺序 [左, 上, 右, 下]；PDF 页绝对坐标；相对原图左上角的百分比。</p>'
                     '<table><thead><tr><th>来源 / 位置</th><th>转录</th></tr></thead><tbody>')
            for line in lines:
                body += (f'<tr {_line_attrs(line)}><td>{html.escape(line["label"])}<br>'
                         f'{html.escape(_coordinate_label(line))}</td><td contenteditable="true">'
                         f'{html.escape(line["text"]).replace(chr(10), "<br>")}</td></tr>')
            body += '</tbody></table></details>'
        return body
    if block.meta.get("code"):
        return code_fence(block.text)
    if block.kind == "heading":
        content = _native_markup(block) if block.meta.get("native_style") else md_text(reflow(block.text))
        return "#" * min(6, max(2, block.level + 1)) + " " + content
    if block.meta.get("native_style"):
        return '<p class="native-text">' + _native_markup(block) + '</p>'
    return md_text(block.text)


def html_block(block, image_prefix="../graph/", children=(), *, positioned=False, caption=None):
    esc = html.escape
    observed = ""
    if block.meta.get("style_approximate"):
        size = block.meta.get("font_size")
        color = block.meta.get("color", "")
        rules = []
        if isinstance(size, (int, float)) and math.isfinite(size) and 5 <= size <= 72:
            rules.append(f"font-size:{size:.2f}pt")
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            rules.append(f"color:{color}")
        if rules:
            observed = ' style="' + ';'.join(rules) + '"'
    if block.kind == "table":
        body = table_html(block) + table_reference(block, image_prefix)
    elif block.kind == "image":
        body = _image_html(block, image_prefix, children, caption)
    elif block.meta.get("code"):
        content = _native_markup(block) if positioned and block.meta.get('native_style') else esc(block.text)
        body = f'<pre><code>{content}</code></pre>'
    elif block.kind == "heading":
        level = min(6, max(2, block.level + 1))
        content = _native_markup(block) if block.meta.get("native_style") else esc(reflow(block.text))
        body = f'<h{level}{observed}>{content}</h{level}>'
    else:
        content = _native_markup(block) if block.meta.get("native_style") else esc(block.text)
        body = f'<p{observed}>{content}</p>'
    placement = f' class="source-position" style="{_geometry_style(block.bbox)}"' if positioned else ''
    return (f'<article id="{esc(block.id, quote=True)}" data-kind="{esc(block.kind, quote=True)}"{placement} '
            f'{_position_attrs(block)}>{body}</article>')


CSS = """
:root{color-scheme:light;--ink:#22333b;--muted:#73828a;--accent:#147886}*{box-sizing:border-box}
body{margin:0;background:#f4f6f7;color:var(--ink);font:16px/1.75 'Microsoft YaHei',system-ui,sans-serif}
header{background:#fff;border-bottom:1px solid #dce4e7;padding:26px max(24px,calc((100vw - 1280px)/2))}header h1{font-size:25px;margin:0 0 6px}header p{font-size:13px;color:var(--muted);margin:6px 0}
a{color:var(--accent)}button{cursor:pointer;background:var(--accent);border:0;color:white;border-radius:6px;padding:9px 15px}.tools{display:flex;align-items:center;gap:18px;margin-top:12px}
.layout{max-width:1280px;margin:24px auto;display:grid;grid-template-columns:235px minmax(0,1fr);gap:24px;padding:0 22px}aside{position:sticky;top:18px;align-self:start;max-height:92vh;overflow:auto;font-size:13px;padding-right:10px}aside h2{font-size:13px;color:var(--muted);margin:0 0 12px}aside a{display:block;text-decoration:none;color:#42545d;line-height:1.5;padding:6px 0}aside a:hover{color:var(--accent)}aside details{margin-top:20px}
main{min-width:0;background:white;border:1px solid #e0e7e9;border-radius:8px;padding:32px 40px}.page+.page{margin-top:30px;padding-top:24px;border-top:1px solid #e5eaec}.page-label{font-size:11px;color:var(--muted);margin-bottom:16px;letter-spacing:1px}article{margin:12px 0}p{margin:10px 0;overflow-wrap:anywhere;text-align:start}h2{font-size:24px;line-height:1.4;margin:24px 0 14px}h3{font-size:20px;margin:22px 0 12px}h4,h5,h6{font-size:17px;margin:20px 0 10px}
figure{margin:22px 0 8px;text-align:center}img{display:block;max-width:100%;height:auto;margin:0 auto}figcaption{font-size:13px;line-height:1.6;text-align:left;margin-top:10px;color:#53666f}.figure-text{font-size:13px;background:#f5f8f9;border-radius:6px;padding:10px 14px;margin-bottom:20px}.figure-text summary{cursor:pointer;color:var(--muted)}.transcript{margin-top:12px}.transcript-row{display:grid;gap:4px;margin:10px 0;padding:6px;min-width:0}.transcript-text{white-space:pre-wrap;overflow-wrap:anywhere}.transcript-row small{color:var(--muted)}
.image-stage{position:relative;display:inline-block;max-width:100%;vertical-align:top;line-height:0}.image-stage img{margin:0}.image-text-layer{position:absolute;inset:0;overflow:hidden;pointer-events:none;user-select:text}.image-text-layer[hidden]{display:none}.image-text-box{position:absolute;display:block;pointer-events:auto;user-select:text;cursor:text}.image-text-box svg{display:block;width:100%;height:100%;overflow:hidden}.image-text-box text{fill:transparent;user-select:text;cursor:text}.image-text-box text::selection{fill:#16363f;background:#b9e4ef}.image-text-box:hover,.image-text-box.is-active{outline:2px solid #147886;background:rgba(20,120,134,.12)}.image-tools{display:flex;justify-content:center;align-items:center;flex-wrap:wrap;gap:14px;font-size:12px;line-height:1.6;margin-top:8px}.image-tools button{font-size:12px}.image-tools button[aria-pressed=true]{background:#22505b}.transcript-meta{display:flex;align-items:center;gap:8px;flex-wrap:wrap;line-height:1.5}.transcript-meta code{font-size:11px;overflow-wrap:anywhere;color:var(--muted)}.locate-text{padding:2px 8px;font-size:12px}.locate-text:disabled{cursor:default;background:#89959b}.transcript-row.is-active{background:#e1f1f4;outline:1px solid #83bec4}.coordinate-note{font-size:11px;color:var(--muted)}
pre{white-space:pre;overflow:auto;padding:18px 20px;background:#f5f7f8;border:1px solid #e4e9eb;border-radius:6px;tab-size:4;font:13px/1.65 Consolas,'Courier New',monospace}.table-scroll{overflow:auto;margin:18px 0}table{border-collapse:collapse;font-size:14px;min-width:100%}td,th{border:1px solid #d9e1e5;padding:8px 12px;vertical-align:top;text-align:left}th{background:#f2f6f7;font-weight:600}.structured-table{table-layout:fixed}.table-reference{font-size:13px;margin:8px 0 20px;color:var(--muted)}.table-reference summary{cursor:pointer}.table-reference img{margin:12px 0}.table-note{font-size:12px;color:var(--muted)}.table-warning{background:#fff7e2;padding:10px;font-size:13px}
[data-kind=page_number],[data-kind=header_footer]{color:var(--muted);font-size:12px;margin:8px 0}[data-kind=page_number]{text-align:center}.notice{background:#fff7e2;padding:12px 16px;margin-bottom:24px;font-size:13px}.editing-note{color:var(--muted)}[contenteditable=true]:focus{outline:1px solid #83bec4;outline-offset:6px}
.view-note{font-size:12px;color:var(--muted)}.page-canvas{position:relative}.page-surface{position:relative}
body[data-view=positioned] .position-capable .page-canvas{width:100%;aspect-ratio:var(--page-ratio);overflow:hidden;background:#fff;outline:1px solid #e0e7e9}
body[data-view=positioned] .position-capable .page-surface{position:absolute;left:0;top:0;width:var(--page-width);height:var(--page-height);transform:scale(var(--page-scale,1));transform-origin:0 0}
body[data-view=positioned] .source-position{position:absolute;left:var(--box-left);top:var(--box-top);width:var(--box-width);height:var(--box-height);margin:0;padding:0;line-height:1;color:#000}
body[data-view=positioned] .source-position>p,body[data-view=positioned] .source-position>h2,body[data-view=positioned] .source-position>h3,body[data-view=positioned] .source-position>h4,body[data-view=positioned] .source-position>h5,body[data-view=positioned] .source-position>h6{margin:0;padding:0;line-height:1;overflow:visible}
body[data-view=positioned] .source-position .native-run{position:absolute;left:var(--run-left);top:var(--run-top);width:var(--run-width);height:var(--run-height);line-height:var(--run-height);white-space:pre;overflow:visible}
body[data-view=positioned] .source-position p>br,body[data-view=positioned] .source-position h2>br,body[data-view=positioned] .source-position h3>br{display:none}
body[data-view=positioned] .source-position>figure{position:relative;margin:0;width:100%;height:100%;padding:0;text-align:left;line-height:0}
body[data-view=positioned] .source-position .image-stage{position:absolute;inset:0;display:block;width:100%;height:100%;max-width:none;line-height:0}
body[data-view=positioned] .source-position .image-stage>img{display:block;width:100%;height:100%;max-width:none;margin:0;object-fit:fill}
body[data-view=positioned] .source-position .image-tools,body[data-view=positioned] .source-position .figure-text,body[data-view=positioned] .source-position .table-reference,body[data-view=positioned] .source-position .table-note,body[data-view=positioned] .source-position .table-warning{display:none}
body[data-view=positioned] .source-position .source-caption{position:absolute;left:var(--caption-left);top:var(--caption-top);width:var(--caption-width);height:var(--caption-height);margin:0;padding:0;line-height:1;color:#000}
body[data-view=positioned] .source-position .table-scroll{margin:0;width:100%;height:100%;overflow:visible}
body[data-view=positioned] .source-position .structured-table{width:100%;height:100%;min-width:0;font-size:10pt;margin:0;line-height:1.2}
body[data-view=positioned] .source-position td,body[data-view=positioned] .source-position th{padding:0 2pt;font-weight:normal;background:transparent}
@media(max-width:960px){.layout{display:block;padding:0 14px}aside{position:static;max-height:240px;margin-bottom:22px;border-bottom:1px solid #dce4e7;padding-bottom:15px}main{padding:22px}header{padding:22px}h2{font-size:22px}}@media print{aside,.tools,.editing-note,.figure-text,.image-tools,.image-text-layer{display:none}.layout{display:block;padding:0}main{border:0;padding:0}.page+.page{break-before:page}header{border:0}body{background:white}}
"""


SCRIPT = """
const main=document.querySelector('main');
const viewToggle=document.getElementById('toggle-page-view');
function fitSourcePages(){
  document.querySelectorAll('.position-capable .page-canvas').forEach(canvas=>{
    const width=Number(canvas.dataset.pageWidth);
    if(width>0) canvas.style.setProperty('--page-scale',String(canvas.clientWidth/(width*96/72)));
  });
}
if(viewToggle){
  viewToggle.addEventListener('click',()=>{
    const positioned=document.body.dataset.view!=='positioned';
    document.body.dataset.view=positioned?'positioned':'reading';
    viewToggle.setAttribute('aria-pressed',String(positioned));
    viewToggle.textContent=positioned?'切换为阅读视图':'切换为原页位置';
    fitSourcePages();
  });
  if(typeof ResizeObserver!=='undefined'){
    const sourceObserver=new ResizeObserver(fitSourcePages);
    document.querySelectorAll('.position-capable .page-canvas').forEach(canvas=>sourceObserver.observe(canvas));
  }else window.addEventListener('resize',fitSourcePages);
  fitSourcePages();
}
function setLayer(figure,visible){
  const layer=figure.querySelector('.image-text-layer');
  const toggle=figure.querySelector('.toggle-image-text');
  if(layer) layer.hidden=!visible;
  if(toggle) toggle.setAttribute('aria-pressed',String(visible));
}
function syncLine(row){
  const figure=row.closest('article[data-kind="image"]');
  const value=row.querySelector('.transcript-text').textContent;
  figure.querySelectorAll('.image-text-box').forEach(box=>{
    if(box.dataset.line===row.dataset.line) box.querySelector('text').textContent=value;
  });
}
function activate(figure,key,scroll){
  const boxes=[...figure.querySelectorAll('.image-text-box')];
  const target=boxes.find(box=>box.dataset.line===key);
  if(!target) return;
  setLayer(figure,true);
  figure.querySelector('.figure-text').open=true;
  figure.querySelectorAll('.image-text-box,.transcript-row').forEach(item=>{
    item.classList.toggle('is-active',item.dataset.line===key);
  });
  if(scroll) target.scrollIntoView({behavior:'smooth',block:'center',inline:'nearest'});
}
main.querySelectorAll('.transcript-row').forEach(syncLine);
main.addEventListener('click',event=>{
  const figure=event.target.closest('article[data-kind="image"]');
  if(!figure) return;
  if(event.target.closest('.toggle-image-text')){
    setLayer(figure,figure.querySelector('.image-text-layer').hidden);
    return;
  }
  const row=event.target.closest('.transcript-row');
  const box=event.target.closest('.image-text-box');
  if(row) activate(figure,row.dataset.line,!row.classList.contains('is-active') || !!event.target.closest('.locate-text'));
  else if(box) activate(figure,box.dataset.line,false);
});
main.addEventListener('input',event=>{
  const text=event.target.closest('.transcript-text');
  if(text) syncLine(text.closest('.transcript-row'));
});
main.addEventListener('paste',event=>{
  if(!event.target.closest('.transcript-text')) return;
  event.preventDefault();
  document.execCommand('insertText',false,event.clipboardData.getData('text/plain'));
});
document.getElementById('save').addEventListener('click',()=>{
  main.querySelectorAll('.transcript-row').forEach(syncLine);
  const snapshot=document.documentElement.cloneNode(true);
  const blob=new Blob(['<!doctype html>\\n'+snapshot.outerHTML],{type:'text/html;charset=utf-8'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download=document.body.dataset.filename;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(a.href),1000);
});
"""


def _toc(blocks, fmt):
    headings = [b for b in blocks if b.kind == "heading"]
    if fmt == "md":
        return "\n".join("  " * max(0, b.level - 1) + f"- [{md_text(reflow(b.text))}](#{b.id})" for b in headings)
    return "".join(f'<a href="#{html.escape(b.id, quote=True)}" style="padding-left:{max(0, b.level - 1) * 12}px">{html.escape(reflow(b.text))}</a>' for b in headings)


def _render_document(result, blocks, fmt, title, image_prefix, source_name, is_summary=False):
    pages = sorted({b.page for b in blocks})
    source_url = quote(source_name + '.pdf')
    if fmt == "md":
        intro = f"# {md_text(title)}\n\n[源 PDF](<../../source/{source_url}>) · {len(pages)} 页\n\n"
        if is_summary and result.page_sizes:
            intro += '> Markdown 按阅读顺序呈现；保留页尺寸和区域坐标元数据，阅读器不保证原页绝对排版。\n\n'
        toc = _toc(blocks, fmt)
        if toc:
            intro += "## 目录\n\n" + toc + "\n\n---\n\n"
        parts = [intro]
        for page in pages:
            size = _coords(_page_size(result, page))
            parts.append(f'<a id="page-{page}" data-page-size="{size}" data-coordinate-unit="pt"></a>\n\n*第 {page} 页*\n\n')
            reference = next((b.meta.get('source_reference') for b in blocks if b.page == page and b.meta.get('source_reference')), '')
            if reference:
                url = html.escape(image_prefix + Path(reference).name, quote=True)
                parts.append(f'<details><summary>原页参考</summary><img src="{url}" alt="第 {page} 页原图"></details>\n\n')
            for b in [b for b in blocks if b.page == page]:
                anchor = f'<a id="{html.escape(b.id, quote=True)}" {_position_attrs(b)} data-page-size="{size}"></a>\n\n'
                parts.append(anchor + markdown(b, image_prefix, attachments(result, b) if b.kind == "image" else ()) + "\n\n")
            parts.append("---\n\n")
        if result.warnings and is_summary:
            parts.append("## 提取提示\n\n" + "\n".join("- " + md_text(w) for w in result.warnings))
        return "".join(parts)
    content = []
    capable_pages = []
    for page in pages:
        page_blocks = [b for b in blocks if b.page == page]
        size = _page_size(result, page)
        positioned = bool(is_summary and size and page_blocks and all(
            b.source == 'native' and _bbox(b.bbox) and not b.meta.get('table_recovery')
            for b in page_blocks))
        if positioned:
            capable_pages.append(page)
        body = "".join(html_block(b, image_prefix, attachments(result, b) if b.kind == "image" else (),
                                 positioned=positioned,
                                 caption=next((c for c in result.blocks if c.page == page
                                               and c.meta.get('caption_for') == b.asset), None)
                                 if b.kind == 'image' else None) for b in page_blocks)
        reference = next((b.meta.get('source_reference') for b in blocks if b.page == page and b.meta.get('source_reference')), '')
        if reference:
            url = html.escape(image_prefix + Path(reference).name, quote=True)
            body = f'<details class="table-reference"><summary contenteditable="false">原页参考</summary><img src="{url}" alt="第 {page} 页原图" loading="lazy" contenteditable="false"></details>' + body
        if positioned:
            width, height = size
            style = f'--page-width:{width:.8f}pt;--page-height:{height:.8f}pt;--page-ratio:{width:.8f}/{height:.8f}'
            body = (f'<div class="page-canvas" data-page-width="{width}" data-page-size="{_coords(size)}" '
                    f'style="{style}"><div class="page-surface">{body}</div></div>')
        page_class = 'page position-capable' if positioned else 'page'
        content.append(f'<section class="{page_class}" id="page-{page}" data-page-size="{_coords(size)}"><div class="page-label" contenteditable="false">第 {page} 页</div>{body}</section>')
    page_links = "".join(f'<a href="#page-{page}">第 {page} 页</a>' for page in pages)
    toc = _toc(blocks, fmt) or '<span class="editing-note">本分类无章节标题</span>'
    warnings = '<details class="notice"><summary>提取提示</summary>' + "".join(f"<p>{html.escape(w)}</p>" for w in result.warnings) + "</details>" if result.warnings and is_summary else ""
    filename = "summary-edited.html" if is_summary else "edited.html"
    view = 'positioned' if capable_pages else 'reading'
    toggle = '<button id="toggle-page-view" type="button" aria-pressed="true">切换为阅读视图</button>' if capable_pages else ''
    view_note = '<p class="view-note">原生页面按原页坐标显示，可切换阅读视图；字体显示取决于本机可用字体，表格内容请对照原表。</p>' if capable_pages else ''
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body data-filename="{filename}" data-view="{view}">
<header><h1>{html.escape(title)}</h1><p>{len(pages)} 页 · 按原文顺序整理</p><div class="tools"><button id="save">保存修改后的 HTML</button>{toggle}<a href="../../source/{source_url}">查看源 PDF</a></div><p class="editing-note">正文与表格可直接编辑。保存后请放回当前目录，保持图片相对路径。</p>{view_note}</header>
<div class="layout"><aside><h2>内容目录</h2>{toc}<details><summary>按页跳转</summary>{page_links}</details></aside><main contenteditable="true" spellcheck="false">{warnings}{''.join(content)}</main></div><script>{SCRIPT}</script></body></html>'''


@tracked('reading.export')
def export_all(result, directory, asset_dir, source_name, formats=("md", "html")):
    directory, asset_dir = Path(directory), Path(asset_dir)
    primary = top_blocks(result)
    for fmt in formats:
        root = directory / fmt
        root.mkdir(parents=True, exist_ok=True)
        graphs = [b for b in primary if b.kind == "image"]
        previews = [b.meta['preview_asset'] for b in primary if b.meta.get('preview_asset')]
        previews += [b.meta['source_reference'] for b in result.blocks if b.meta.get('source_reference')]
        if graphs or previews:
            (root / 'graph').mkdir()
            for name in {image_name(b) for b in graphs} | {Path(a).name for a in previews}:
                shutil.copy2(asset_dir / name, root / 'graph' / name)
        for kind, folder in FOLDERS.items():
            blocks = [b for b in primary if b.kind == kind]
            if kind == "ocr_text":
                blocks = [b for b in result.blocks if b.kind == "ocr_text"]
            if not blocks:
                continue
            destination = root / folder
            destination.mkdir(exist_ok=True)
            title = f"{Path(result.source).stem} · {LABELS[kind]}"
            image_prefix = "" if kind == "image" else "../graph/"
            (destination / f"{folder}.{fmt}").write_text(_render_document(result, blocks, fmt, title, image_prefix, source_name), encoding="utf-8")
        summary = root / "summary"
        summary.mkdir()
        (summary / f"summary.{fmt}").write_text(_render_document(result, primary, fmt, Path(result.source).stem, "../graph/", source_name, True), encoding="utf-8")
