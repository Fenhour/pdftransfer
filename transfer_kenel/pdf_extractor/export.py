"""Readable Markdown and HTML, with mirrored per-document directories."""
import html
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
    return [b for b in result.blocks if b.meta.get("figure_label") and b.meta.get("image") == block.asset]


def top_blocks(result):
    return [b for b in result.blocks if not b.meta.get("figure_label") and not b.meta.get("caption_for")]


def image_name(block):
    return Path(block.asset).name


def markdown(block, image_prefix="../graph/", children=()):
    if block.kind == "table":
        width = max((len(row) for row in block.rows), default=0)
        rows = [[f"列 {i + 1}" for i in range(width)], ["---"] * width]
        rows += [[md_text(row[i]).replace("|", "\\|").replace("\n", "<br>") if i < len(row) else "" for i in range(width)] for row in block.rows]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)
    if block.kind == "image":
        caption = block.meta.get("caption", "")
        body = f"![第 {block.page} 页图片](<{image_prefix}{image_name(block)}>)"
        if caption:
            body += f"\n\n*{md_text(caption)}*"
        if children:
            body += "\n\n<details>\n<summary>图中文字（可编辑；OCR 请核对原图）</summary>\n\n"
            body += code_fence("\n".join(b.text for b in children)) + "\n\n</details>"
        return body
    if block.meta.get("code"):
        return code_fence(block.text)
    if block.kind == "heading":
        return "#" * min(6, max(2, block.level + 1)) + " " + md_text(reflow(block.text))
    return md_text(block.text)


def html_block(block, image_prefix="../graph/", children=()):
    esc = html.escape
    if block.kind == "table":
        body = '<div class="table-scroll"><table><tbody>' + "".join("<tr>" + "".join("<td>" + esc(str(c or "")).replace("\n", "<br>") + "</td>" for c in row) + "</tr>" for row in block.rows) + "</tbody></table></div>"
    elif block.kind == "image":
        caption = block.meta.get("caption", "")
        body = f'<figure><a href="{esc(image_prefix + image_name(block), quote=True)}" target="_blank" contenteditable="false"><img src="{esc(image_prefix + image_name(block), quote=True)}" alt="第 {block.page} 页图片" loading="lazy"></a>'
        if caption:
            body += f'<figcaption>{esc(caption)}</figcaption>'
        body += "</figure>"
        if children:
            body += '<details class="figure-text"><summary contenteditable="false">图中文字 · 点击展开编辑</summary><div class="transcript">'
            for child in children:
                label = f"OCR · {child.confidence:.0%}" if child.confidence is not None else "原文文字"
                body += f'<div class="transcript-row"><small contenteditable="false">{label}</small><span>{esc(child.text)}</span></div>'
            body += "</div></details>"
    elif block.meta.get("code"):
        body = f'<pre><code>{esc(block.text)}</code></pre>'
    elif block.kind == "heading":
        level = min(6, max(2, block.level + 1))
        body = f'<h{level}>{esc(reflow(block.text))}</h{level}>'
    else:
        body = f'<p>{esc(block.text)}</p>'
    return f'<article id="{block.id}" data-kind="{block.kind}">{body}</article>'


CSS = """
:root{color-scheme:light;--ink:#22333b;--muted:#73828a;--accent:#147886}*{box-sizing:border-box}
body{margin:0;background:#f4f6f7;color:var(--ink);font:16px/1.75 'Microsoft YaHei',system-ui,sans-serif}
header{background:#fff;border-bottom:1px solid #dce4e7;padding:26px max(24px,calc((100vw - 1280px)/2))}header h1{font-size:25px;margin:0 0 6px}header p{font-size:13px;color:var(--muted);margin:6px 0}
a{color:var(--accent)}button{cursor:pointer;background:var(--accent);border:0;color:white;border-radius:6px;padding:9px 15px}.tools{display:flex;align-items:center;gap:18px;margin-top:12px}
.layout{max-width:1280px;margin:24px auto;display:grid;grid-template-columns:235px minmax(0,1fr);gap:24px;padding:0 22px}aside{position:sticky;top:18px;align-self:start;max-height:92vh;overflow:auto;font-size:13px;padding-right:10px}aside h2{font-size:13px;color:var(--muted);margin:0 0 12px}aside a{display:block;text-decoration:none;color:#42545d;line-height:1.5;padding:6px 0}aside a:hover{color:var(--accent)}aside details{margin-top:20px}
main{min-width:0;background:white;border:1px solid #e0e7e9;border-radius:8px;padding:32px 40px}.page+.page{margin-top:30px;padding-top:24px;border-top:1px solid #e5eaec}.page-label{font-size:11px;color:var(--muted);margin-bottom:16px;letter-spacing:1px}article{margin:12px 0}p{margin:10px 0;overflow-wrap:anywhere;text-align:start}h2{font-size:24px;line-height:1.4;margin:24px 0 14px}h3{font-size:20px;margin:22px 0 12px}h4,h5,h6{font-size:17px;margin:20px 0 10px}
figure{margin:22px 0 8px;text-align:center}img{display:block;max-width:100%;height:auto;margin:0 auto}figcaption{font-size:13px;line-height:1.6;text-align:left;margin-top:10px;color:#53666f}.figure-text{font-size:13px;background:#f5f8f9;border-radius:6px;padding:10px 14px;margin-bottom:20px}.figure-text summary{cursor:pointer;color:var(--muted)}.transcript{margin-top:12px}.transcript-row{display:grid;grid-template-columns:88px 1fr;gap:10px;margin:4px 0}.transcript-row span{white-space:pre-wrap}.transcript-row small{color:var(--muted)}
pre{white-space:pre;overflow:auto;padding:18px 20px;background:#f5f7f8;border:1px solid #e4e9eb;border-radius:6px;tab-size:4;font:13px/1.65 Consolas,'Courier New',monospace}.table-scroll{overflow:auto;margin:18px 0}table{border-collapse:collapse;font-size:14px;min-width:100%}td{border:1px solid #d9e1e5;padding:8px 12px;vertical-align:top}tr:first-child{background:#f2f6f7;font-weight:600}
[data-kind=page_number],[data-kind=header_footer]{color:var(--muted);font-size:12px;margin:8px 0}[data-kind=page_number]{text-align:center}.notice{background:#fff7e2;padding:12px 16px;margin-bottom:24px;font-size:13px}.editing-note{color:var(--muted)}[contenteditable=true]:focus{outline:1px solid #83bec4;outline-offset:6px}
@media(max-width:960px){.layout{display:block;padding:0 14px}aside{position:static;max-height:240px;margin-bottom:22px;border-bottom:1px solid #dce4e7;padding-bottom:15px}main{padding:22px}header{padding:22px}h2{font-size:22px}}@media print{aside,.tools,.editing-note,.figure-text{display:none}.layout{display:block;padding:0}main{border:0;padding:0}.page+.page{break-before:page}header{border:0}body{background:white}}
"""


def _toc(blocks, fmt):
    headings = [b for b in blocks if b.kind == "heading"]
    if fmt == "md":
        return "\n".join("  " * max(0, b.level - 1) + f"- [{md_text(reflow(b.text))}](#{b.id})" for b in headings)
    return "".join(f'<a href="#{b.id}" style="padding-left:{max(0, b.level - 1) * 12}px">{html.escape(reflow(b.text))}</a>' for b in headings)


def _render_document(result, blocks, fmt, title, image_prefix, source_name, is_summary=False):
    pages = sorted({b.page for b in blocks})
    source_url = quote(source_name + '.pdf')
    if fmt == "md":
        intro = f"# {md_text(title)}\n\n[源 PDF](<../../source/{source_url}>) · {len(pages)} 页\n\n"
        toc = _toc(blocks, fmt)
        if toc:
            intro += "## 目录\n\n" + toc + "\n\n---\n\n"
        parts = [intro]
        for page in pages:
            parts.append(f'<a id="page-{page}"></a>\n\n*第 {page} 页*\n\n')
            for b in [b for b in blocks if b.page == page]:
                anchor = f'<a id="{b.id}"></a>\n\n' if b.kind == 'heading' else ''
                parts.append(anchor + markdown(b, image_prefix, attachments(result, b) if b.kind == "image" else ()) + "\n\n")
            parts.append("---\n\n")
        if result.warnings and is_summary:
            parts.append("## 提取提示\n\n" + "\n".join("- " + md_text(w) for w in result.warnings))
        return "".join(parts)
    content = []
    for page in pages:
        body = "".join(html_block(b, image_prefix, attachments(result, b) if b.kind == "image" else ()) for b in blocks if b.page == page)
        content.append(f'<section class="page" id="page-{page}"><div class="page-label" contenteditable="false">第 {page} 页</div>{body}</section>')
    page_links = "".join(f'<a href="#page-{page}">第 {page} 页</a>' for page in pages)
    toc = _toc(blocks, fmt) or '<span class="editing-note">本分类无章节标题</span>'
    warnings = '<details class="notice"><summary>提取提示</summary>' + "".join(f"<p>{html.escape(w)}</p>" for w in result.warnings) + "</details>" if result.warnings and is_summary else ""
    script = """document.getElementById('save').addEventListener('click',()=>{const snapshot=document.documentElement.cloneNode(true);const blob=new Blob(['<!doctype html>\\n'+snapshot.outerHTML],{type:'text/html;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=document.body.dataset.filename;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);});"""
    filename = "summary-edited.html" if is_summary else "edited.html"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{CSS}</style></head><body data-filename="{filename}">
<header><h1>{html.escape(title)}</h1><p>{len(pages)} 页 · 按原文顺序整理</p><div class="tools"><button id="save">保存修改后的 HTML</button><a href="../../source/{source_url}">查看源 PDF</a></div><p class="editing-note">正文与表格可直接编辑。保存后请放回当前目录，保持图片相对路径。</p></header>
<div class="layout"><aside><h2>内容目录</h2>{toc}<details><summary>按页跳转</summary>{page_links}</details></aside><main contenteditable="true" spellcheck="false">{warnings}{''.join(content)}</main></div><script>{script}</script></body></html>'''


def export_all(result, directory, asset_dir, source_name):
    directory, asset_dir = Path(directory), Path(asset_dir)
    primary = top_blocks(result)
    for fmt in ("md", "html"):
        root = directory / fmt
        root.mkdir(parents=True, exist_ok=True)
        graphs = [b for b in primary if b.kind == "image"]
        if graphs:
            (root / "graph").mkdir()
            for b in graphs:
                shutil.copy2(asset_dir / image_name(b), root / "graph" / image_name(b))
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
