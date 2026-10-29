"""All export formats consume the same current scene, including visibility and transforms."""
from __future__ import annotations

from .usage import tracked
import base64
import copy
import html
import io
import json
import math
import mimetypes
import os
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import pymupdf as fitz
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .font_library import get_library, ink_metrics, match_document_fonts, valid_runs
from .pdf_safety import character_inventory, prepare_fonts, save_verified
from .progress import progress_span, report
from .rebuild_flow import flow_cell_text, is_cell_flow


def read_asset(src: str, output_dir: Path, document_id: str) -> bytes:
    if src.startswith("data:image/"):
        return base64.b64decode(src.split(",", 1)[1], validate=True)
    prefix = f"/files/{document_id}/"
    if not src.startswith(prefix):
        raise ValueError("资源不属于当前文档")
    path = (output_dir / src[len(prefix):]).resolve()
    if not path.is_relative_to(output_dir.resolve()) or not path.is_file():
        raise ValueError("资源不存在或路径无效")
    return path.read_bytes()


@tracked('export.json')
def portable_document(document: dict, output_dir: Path) -> dict:
    result = copy.deepcopy(document)
    cache = {}
    def embed(src):
        if not src or src.startswith("data:"):
            return src
        if src not in cache:
            mime = mimetypes.guess_type(src)[0] or "image/png"
            cache[src] = f"data:{mime};base64," + base64.b64encode(read_asset(src, output_dir, document["id"])).decode()
        return cache[src]
    for index,page in enumerate(result["pages"]):
        report(100*index/max(1,len(result['pages'])),'打包文档资源',current=index+1,total=len(result['pages']))
        if page.get("sourceSrc"):
            page["sourceSrc"] = embed(page["sourceSrc"])
        for table in page.get('tables',[]):
            if table.get('referenceSrc'):
                table['referenceSrc'] = embed(table['referenceSrc'])
        for el in page["elements"]:
            for key in ("src", "appearanceSrc", "originalAppearanceSrc"):
                if el.get(key):
                    el[key] = embed(el[key])
            if el.get("svg"):
                el["svgSrc"] = "data:image/svg+xml;base64," + base64.b64encode(el["svg"].encode()).decode()
    return result


def styled_svg(el: dict) -> str:
    root = ET.fromstring(el["svg"])
    for node in root.iter():
        if node.tag.split("}")[-1] in {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}:
            for key in ("fill", "stroke"):
                if el.get(key) is not None:
                    node.set(key, el[key])
            if el.get("stroke") is not None:
                node.set("stroke-width", str(el.get("strokeWidth", 1)))
    return ET.tostring(root, encoding="unicode")


@tracked('export.html')
def scene_html(document: dict, output_dir: Path) -> str:
    doc = portable_document(document, output_dir)
    title = html.escape(doc["name"])
    payload = json.dumps(doc, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>''' + title + '''</title>
<style>*{box-sizing:border-box}body{margin:0;background:#e9edee;font-family:"Microsoft YaHei",sans-serif}header{position:sticky;top:0;z-index:999999;background:#162e37;color:#fff;padding:16px 24px;display:flex;gap:20px;align-items:center;font-size:13px}header strong{margin-right:auto}button{background:#fff;color:#143b3b;border:0;padding:8px 16px;border-radius:6px;cursor:pointer}.page{position:relative;isolation:isolate;background:white;margin:28px auto;box-shadow:0 5px 30px #182c3820;overflow:hidden;touch-action:none}.element{position:absolute;transform-origin:center;white-space:pre-wrap;overflow:hidden;overflow-wrap:break-word;cursor:move}.element img,.element svg{width:100%;height:100%;display:block;pointer-events:none}.element:focus{outline:1px solid #167c73}.element[contenteditable=true]{cursor:text}@media print{body{background:white}header{display:none}.page{margin:0;box-shadow:none;break-after:page;page-break-after:always}.page:last-child{break-after:auto}}</style>
<header><strong>''' + title + '''</strong><span>拖动元素 · 双击修改文字 · JSON 可重新导入解析器</span><button id="save">下载 JSON</button><button onclick="print()">打印</button></header><main id="pages"></main>
<script id="scene-data" type="application/json">''' + payload + '''</script>
<script>
const scene=JSON.parse(document.getElementById('scene-data').textContent);
function syncCorners(){scene.pages.forEach(p=>p.elements.forEach(e=>{const a=(e.rotation||0)*Math.PI/180,c=Math.cos(a),s=Math.sin(a),cx=e.x+e.width/2,cy=e.y+e.height/2;e.corners=[[-e.width/2,-e.height/2],[e.width/2,-e.height/2],[e.width/2,e.height/2],[-e.width/2,e.height/2]].map(([x,y])=>[Math.round((cx+x*c-y*s)*100)/100,Math.round((cy+x*s+y*c)*100)/100])}))}
function render(){const main=document.getElementById('pages');main.replaceChildren();scene.pages.forEach(page=>{const p=document.createElement('section');p.className='page';p.style.width=page.width+'px';p.style.height=page.height+'px';main.append(p);[...page.elements].sort((a,b)=>a.zIndex-b.zIndex).forEach(el=>{if(el.visible===false)return;const d=document.createElement('div');d.className='element';d.dataset.elementId=el.id;d.tabIndex=0;Object.assign(d.style,{left:el.x+'px',top:el.y+'px',width:el.width+'px',height:el.height+'px',transform:`rotate(${el.rotation||0}deg)`,opacity:el.opacity??1,zIndex:el.zIndex,fontFamily:el.fontFamily||'Microsoft YaHei',fontSize:(el.fontSize||14)+'px',fontWeight:el.fontWeight||'normal',fontStyle:el.fontStyle||'normal',color:el.color||'#172c3c',textAlign:el.align||'left',lineHeight:el.lineHeight||1.2});
const src=el.appearanceSrc||el.src;if(src){const img=new Image();img.src=src;img.draggable=false;d.append(img)}else if(el.type==='text'){d.textContent=el.text||''}else if(el.svg){const xml=new DOMParser().parseFromString(el.svg,'image/svg+xml');const root=xml.documentElement;root.querySelectorAll('path,rect,circle,ellipse,line,polyline,polygon').forEach(n=>{if(el.fill!=null)n.setAttribute('fill',el.fill);if(el.stroke!=null){n.setAttribute('stroke',el.stroke);n.setAttribute('stroke-width',el.strokeWidth??1)}});d.append(document.importNode(root,true))}else if(el.type==='shape'){const ns='http://www.w3.org/2000/svg',s=document.createElementNS(ns,'svg');s.setAttribute('viewBox',`0 0 ${el.width} ${el.height}`);const n=document.createElementNS(ns,el.shape==='ellipse'?'ellipse':el.shape==='line'?'line':'rect');const attrs=el.shape==='ellipse'?{cx:el.width/2,cy:el.height/2,rx:Math.max(0,(el.width-(el.strokeWidth??1))/2),ry:Math.max(0,(el.height-(el.strokeWidth??1))/2)}:el.shape==='line'?{x1:0,y1:el.height/2,x2:el.width,y2:el.height/2}:{x:(el.strokeWidth??1)/2,y:(el.strokeWidth??1)/2,width:Math.max(0,el.width-(el.strokeWidth??1)),height:Math.max(0,el.height-(el.strokeWidth??1))};Object.entries({...attrs,fill:el.fill||'none',stroke:el.stroke||'#167c73','stroke-width':el.strokeWidth??1}).forEach(([k,v])=>n.setAttribute(k,v));s.append(n);d.append(s)}p.append(d);
if(!el.locked){d.ondblclick=()=>{if(el.type!=='text')return;delete el.appearanceSrc;d.textContent=el.text||'';d.contentEditable='true';d.focus()};d.oninput=()=>{el.text=d.innerText};d.onblur=()=>{d.contentEditable='false'};d.onpointerdown=e=>{if(d.isContentEditable)return;e.preventDefault();const sx=e.clientX,sy=e.clientY,x=el.x,y=el.y;d.setPointerCapture(e.pointerId);d.onpointermove=m=>{el.x=x+m.clientX-sx;el.y=y+m.clientY-sy;d.style.left=el.x+'px';d.style.top=el.y+'px'};d.onpointerup=()=>{d.onpointermove=null;d.onpointerup=null}}}
})})}render();document.getElementById('save').onclick=()=>{syncCorners();scene.updatedAt=new Date().toISOString();const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(scene,null,2)],{type:'application/json'}));a.download=scene.name+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500)};
</script></html>'''


def font_for(el: dict, size: int):
    folder = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    family = el.get("fontFamily", "").lower()
    bold = str(el.get("fontWeight", "normal")) in {"bold", "600", "700", "800", "900"}
    italic = el.get("fontStyle") in {"italic", "oblique"}
    if "song" in family or "宋" in family:
        names = ["simsun.ttc"]
    elif "times" in family or "serif" == family:
        names = ["timesbi.ttf" if bold and italic else "timesbd.ttf" if bold else "timesi.ttf" if italic else "times.ttf"]
    elif "arial" in family or "helvetica" in family:
        names = ["arialbi.ttf" if bold and italic else "arialbd.ttf" if bold else "ariali.ttf" if italic else "arial.ttf"]
    elif "courier" in family or "mono" in family:
        names = ["courbd.ttf" if bold else "cour.ttf"]
    else:
        names = ["msyhbd.ttc" if bold else "msyh.ttc"]
    if any(ord(c) > 0x2fff for c in el.get("text", "")):
        names.insert(0, "msyhbd.ttc" if bold else "msyh.ttc")
    for candidate in [*(folder / n for n in names), folder / "msyh.ttc", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), max(1, size))
    return ImageFont.load_default(size=max(1, size))


def rgba(color, default=(0, 0, 0, 0)):
    if not color or color in {"none", "transparent"}:
        return default
    return ImageColor.getcolor(color, "RGBA")


def element_bitmap(el: dict, doc_id: str, output_dir: Path, scale: float) -> Image.Image:
    w, h = max(1, round(el["width"] * scale)), max(1, round(el["height"] * scale))
    if w*h > 40_000_000:
        raise ValueError("单个元素太大，请缩小后导出")
    canvas = Image.new("RGBA", (w, h))
    src = el.get("appearanceSrc") or el.get("src")
    if src:
        with Image.open(io.BytesIO(read_asset(src, output_dir, doc_id))) as im:
            return im.convert("RGBA").resize((w,h), Image.Resampling.LANCZOS)
    if el["type"] == "text":
        draw = ImageDraw.Draw(canvas)
        font = font_for(el, round(el.get("fontSize",14)*scale))
        lines = []
        for line in el.get("text", "").split("\n"):
            current = ""
            for c in line:
                if current and draw.textlength(current+c, font=font) > w:
                    lines.append(current)
                    current = c
                else:
                    current += c
            lines.append(current)
        step = el.get("fontSize",14) * el.get("lineHeight",1.2)*scale
        for i, line in enumerate(lines):
            length = draw.textlength(line,font=font)
            x = (w-length)/2 if el.get("align") == "center" else w-length if el.get("align") == "right" else 0
            y = i*step + max(0,(step-el.get("fontSize",14)*scale)/2)
            draw.text((x,y),line,font=font,fill=rgba(el.get("color","#172c3c")),anchor="lt")
    elif el.get("svg"):
        with fitz.open(stream=styled_svg(el).encode(),filetype="svg") as svg:
            p = svg[0]
            pix = p.get_pixmap(matrix=fitz.Matrix(w/p.rect.width,h/p.rect.height),alpha=True)
            canvas = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA").resize((w,h))
    elif el["type"] == "shape":
        draw=ImageDraw.Draw(canvas)
        sw=max(0,round(el.get("strokeWidth",1)*scale))
        box=(sw/2,sw/2,max(sw/2,w-sw/2),max(sw/2,h-sw/2))
        if el.get("shape") == "line":
            if sw:
                draw.line((0,h/2,w,h/2), fill=rgba(el.get("stroke","#167c73")),width=sw)
        else:
            fn=draw.ellipse if el.get("shape")=="ellipse" else draw.rectangle
            fn(box,fill=rgba(el.get("fill")),outline=rgba(el.get("stroke")) if sw else None,width=max(1,sw))
    return canvas


def render_page(page: dict, document_id: str, output_dir: Path, scale: float = 2) -> Image.Image:
    scale=min(scale,math.sqrt(30_000_000/(page["width"]*page["height"])))
    result=Image.new("RGBA",(max(1,round(page["width"]*scale)),max(1,round(page["height"]*scale))),"white")
    for el in sorted(page["elements"],key=lambda x:x.get("zIndex",0)):
        if not el.get("visible",True) or el.get("opacity",1)<=0:
            continue
        layer=element_bitmap(el,document_id,output_dir,scale)
        opacity=el.get("opacity",1)
        if opacity!=1:
            layer.putalpha(layer.getchannel("A").point(lambda x:round(x*opacity)))
        cx=(el["x"]+el["width"]/2)*scale
        cy=(el["y"]+el["height"]/2)*scale
        if el.get("rotation",0)%360:
            layer=layer.rotate(-el["rotation"],Image.Resampling.BICUBIC,expand=True)
        result.alpha_composite(layer,(round(cx-layer.width/2),round(cy-layer.height/2)))
    return result.convert("RGB")


@tracked('export.pdf')
def write_pdf(document: dict, output_dir: Path, destination: Path):
    pdf=fitz.open()
    for index,p in enumerate(document["pages"]):
        report(85*index/max(1,len(document['pages'])),'渲染 PDF 页面',current=index+1,total=len(document['pages']))
        rendered=render_page(p,document["id"],output_dir)
        buf=io.BytesIO()
        rendered.save(buf,format="PNG")
        page=pdf.new_page(width=p["width"],height=p["height"])
        page.insert_image(page.rect,stream=buf.getvalue())
    with progress_span(85,95):
        payload=json.dumps(portable_document(document,output_dir),ensure_ascii=False).encode()
    pdf.embfile_add("scene-document.json",payload,filename="scene-document.json",desc="Editable scene graph for PDF OCR Parser")
    pdf.set_metadata({"title":document["name"],"creator":"PDF OCR Parser · editable scene attached","subject":"Rendered page with editable JSON attachment"})
    report(95,'保存视觉 PDF')
    pdf.save(str(destination),deflate=True)
    report(100,'视觉 PDF 已生成')
    pdf.close()


@lru_cache(maxsize=5)
def _pdf_font(name):
    return fitz.Font(name)


def _write_text(page, element):
    """Write visible PDF text at the current box, including center rotation."""
    text=element.get('text','')
    if not text.strip():
        return
    if is_cell_flow(element):
        return _write_cell_flow(page, element)
    if valid_runs(element):
        return _write_font_runs(page,element)
    # Standard PDF fonts have stable Unicode maps (some system fonts map the
    # shared space/NBSP glyph back to NBSP). OCR fonts are estimates anyway.
    bold=str(element.get('fontWeight','normal')) in {'bold','600','700','800','900'}
    italic=element.get('fontStyle') in {'italic','oblique'}
    face='hebi' if bold and italic else 'hebo' if bold else 'heit' if italic else 'helv'
    library=get_library()
    chosen=library.resolve(text,element.get('fontFamily',''),bold,italic,safe_only=element.get('safePdfFonts',False))
    font=(library.export_font_for_key(chosen.key) if element.get('safePdfFonts') else library.font_for_key(chosen.key)) if chosen else _pdf_font(face if all(ord(char)<256 for char in text) else 'cjk')
    size=element.get('fontSize',14)
    lines=text.split('\n')
    step=size*element.get('lineHeight',1.2)
    line_widths=[font.text_length(line,fontsize=size) for line in lines]
    natural_width=max(.1,*line_widths)
    natural_height=max(.1,(font.ascender-font.descender)*size+step*(len(lines)-1))
    # OCR box geometry estimates the observed line; fit the replacement to it.
    fit=min(1.,element['width']/natural_width,element['height']/natural_height)
    sx=sy=fit
    writer=fitz.TextWriter(page.rect)
    for index,line in enumerate(lines):
        x=(natural_width-line_widths[index])/2 if element.get('align')=='center' else natural_width-line_widths[index] if element.get('align')=='right' else 0
        if line:
            writer.append((x,font.ascender*size+index*step),line,font=font,fontsize=size)
    angle=math.radians(element.get('rotation',0))
    c,s=math.cos(angle),math.sin(angle)
    cx,cy=element['x']+element['width']/2,element['y']+element['height']/2
    dx,dy=element['x']-cx,element['y']-cy
    matrix=fitz.Matrix(sx*c,sx*s,-sy*s,sy*c,cx+dx*c-dy*s,cy+dx*s+dy*c)
    color=tuple(value/255 for value in rgba(element.get('color','#172c3c'))[:3])
    # TextWriter's matrix operates in bottom-left PDF coordinates, while the
    # scene uses top-left points. Conjugate the complete current transform.
    pdf_matrix=page.transformation_matrix*matrix*~page.transformation_matrix
    writer.write_text(page,matrix=pdf_matrix,color=color,opacity=element.get('opacity',1),overlay=True)


def _write_cell_flow(page, element):
    """Wrap model HTML cell text without turning a cell box into a font box."""
    font, layout = flow_cell_text(element)
    writer = fitz.TextWriter(page.rect)
    size = layout['fontSize']
    for index, line in enumerate(layout['lines']):
        if not line:
            continue
        width = font.text_length(line, fontsize=size)
        x = ((element['width']-width)/2 if element.get('align') == 'center' else
             element['width']-width if element.get('align') == 'right' else 0.)
        writer.append((x, layout['baseline']+index*layout['lineStep']), line, font=font, fontsize=size)
    angle = math.radians(element.get('rotation', 0))
    c, s = math.cos(angle), math.sin(angle)
    cx, cy = element['x']+element['width']/2, element['y']+element['height']/2
    dx, dy = element['x']-cx, element['y']-cy
    matrix = fitz.Matrix(c, s, -s, c, cx+dx*c-dy*s, cy+dx*s+dy*c)
    pdf_matrix = page.transformation_matrix*matrix*~page.transformation_matrix
    color = tuple(value/255 for value in rgba(element.get('color', '#172c3c'))[:3])
    writer.write_text(page, matrix=pdf_matrix, color=color, opacity=element.get('opacity', 1), overlay=True)


def _write_font_runs(page,element):
    """Embed matched faces and place glyphs with tracking, without stretching."""
    library=get_library()
    writer=fitz.TextWriter(page.rect)
    for run in element['fontRuns']:
        text=run['text']
        visible=text.strip() or text
        font=library.export_font_for_key(run['fontKey']) if element.get('safePdfFonts') else library.font_for_key(run['fontKey'])
        left,top,right,bottom=ink_metrics(run['fontKey'],visible)
        size=run['fontSize']
        natural=max(.1,(right-left)/64*size)
        # Fit large font mismatches by reducing the actual font size. Tracking
        # then recovers the observed line width, never an anisotropic matrix.
        if natural>run['width']*1.10:
            size*=run['width']/natural
            natural=run['width']
        tracking=(run['width']-natural)/max(1,len(visible)-1)
        tracking=max(-size*.08,min(size*.12,tracking))
        if element.get('safePdfFonts') and len(visible)<=2:
            # Broad OCR boxes around short table labels are not evidence of
            # letter spacing. Tracking them creates false spaces on copy.
            tracking=0
        leading=len(text)-len(text.lstrip())
        x=run['x']-left/64*size-font.text_length(text[:leading],fontsize=size)
        baseline=run.get('baseline',run['y']-top/64*size+max(0,run['height']-(bottom-top)/64*size)/2)
        for index,char in enumerate(text):
            writer.append((x,baseline),char,font=font,fontsize=size)
            x+=font.text_length(char,fontsize=size)+(tracking if leading<=index<leading+len(visible)-1 else 0)
    angle=math.radians(element.get('rotation',0))
    c,s=math.cos(angle),math.sin(angle)
    cx,cy=element['x']+element['width']/2,element['y']+element['height']/2
    dx,dy=element['x']-cx,element['y']-cy
    matrix=fitz.Matrix(c,s,-s,c,cx+dx*c-dy*s,cy+dx*s+dy*c)
    pdf_matrix=page.transformation_matrix*matrix*~page.transformation_matrix
    color=tuple(value/255 for value in rgba(element.get('color','#172c3c'))[:3])
    writer.write_text(page,matrix=pdf_matrix,color=color,opacity=element.get('opacity',1),overlay=True)


def _insert_rebuild_image(page, box, image, paper):
    """Keep alpha exact while avoiding black sampling fringes in PDF readers.

    MuPDF writes zero RGB beneath fully transparent RGBA pixels. Some PDF
    readers sample that black before applying the separately encoded mask.
    Repair only those invisible samples in a verified 8-bit RGB image stream;
    all visible RGB, the soft mask, scene assets and source pixels stay intact.
    """
    pixels = np.asarray(image)
    hidden = pixels[:, :, 3] == 0
    if not np.any(hidden):
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return page.insert_image(box, stream=buffer.getvalue(), keep_proportion=False)
    # The insertion digest includes hidden RGB. Give different paper colors
    # distinct image identities before MuPDF normalizes those samples, so an
    # image reused on another page cannot change an earlier page's backdrop.
    backdrop = tuple(rgba(paper or '#ffffff')[:3])
    prepared = pixels.copy()
    del pixels
    prepared[hidden, :3] = backdrop
    buffer = io.BytesIO()
    Image.fromarray(prepared).save(buffer, format='PNG')
    del prepared
    pdf = page.parent
    first_new_xref = pdf.xref_length()
    previous_names = {item[7] for item in page.get_images()}
    xref = page.insert_image(box, stream=buffer.getvalue(), keep_proportion=False)
    del buffer
    marker = '['+' '.join(map(str, backdrop))+']'
    if pdf.xref_get_key(xref, 'PDFTransferAlphaBackdrop') == ('array', marker):
        return xref
    width, height = image.size
    stream = pdf.xref_stream(xref)
    pixmap = fitz.Pixmap(pdf, xref)
    rgb_stream = (pixmap.width == width and pixmap.height == height and pixmap.n == 3 and
                  pixmap.colorspace and pixmap.colorspace.n == 3)
    del pixmap
    if (not rgb_stream or
            pdf.xref_get_key(xref, 'BitsPerComponent') != ('int', '8') or
            len(stream) != width*height*3):
        return xref
    if xref < first_new_xref:
        # An unmarked cache hit may belong to a legacy page. Isolate only the
        # resource added by this call instead of mutating its shared XObject.
        name = next(item[7] for item in page.get_images() if item[7] not in previous_names)
        resource_kind, resource = pdf.xref_get_key(page.xref, 'Resources')
        if resource_kind != 'xref':
            return xref
        isolated = pdf.get_new_xref()
        pdf.update_object(isolated, '<<>>')
        pdf.xref_copy(xref, isolated)
        pdf.xref_set_key(int(resource.split()[0]), 'XObject/'+name, f'{isolated} 0 R')
        xref = isolated
    encoded = np.frombuffer(stream, np.uint8).copy().reshape(height, width, 3)
    encoded[hidden] = backdrop
    pdf.xref_set_key(xref, 'DecodeParms', 'null')
    pdf.update_stream(xref, encoded.tobytes())
    pdf.xref_set_key(xref, 'PDFTransferAlphaBackdrop', marker)
    return xref


@tracked('export.text_pages')
def append_text_pages(pdf, document: dict, output_dir: Path):
    """Append a scene without retaining its JSON or raster buffers in memory."""
    count=0
    for index,source in enumerate(document['pages']):
        report(100*index/max(1,len(document['pages'])),'写入 PDF 页面',current=index+1,total=len(document['pages']))
        page=pdf.new_page(width=source['width'],height=source['height'])
        if source.get('appearanceStrategy')=='blank-page-typed-reconstruction':
            color=tuple(v/255 for v in rgba(source.get('paperColor','#ffffff'))[:3])
            page.draw_rect(page.rect,color=None,fill=color,overlay=False)
            for table in source.get('tables',[]):
                if table.get('borderElements'):
                    # Editable border elements already carry the visible
                    # cell boundaries, including merged-cell geometry.
                    continue
                style=table['observedStyle']
                if style.get('borderWidth',0)<=0 or not table.get('borderVisible',True):
                    continue
                color=tuple(v/255 for v in rgba(style['borderColor'])[:3])
                if any(c.get('rowspan',1)>1 or c.get('colspan',1)>1 for c in table['cells']):
                    # A full grid would cut through a model's merged cells.
                    # Deduplicate coincident perimeter segments instead.
                    edges=set()
                    for cell in table['cells']:
                        x0,y0,x1,y1=cell['bbox']
                        for a,b in [((x0,y0),(x1,y0)),((x1,y0),(x1,y1)),
                                    ((x0,y1),(x1,y1)),((x0,y0),(x0,y1))]:
                            edge=tuple(round(v,4) for point in (a,b) for v in point)
                            if edge not in edges:
                                page.draw_line(a,b,color=color,width=style['borderWidth'])
                                edges.add(edge)
                else:
                    for x in table['columns']:
                        page.draw_line((x,table['rows'][0]),(x,table['rows'][-1]),color=color,width=style['borderWidth'])
                    for y in table['rows']:
                        page.draw_line((table['columns'][0],y),(table['columns'][-1],y),color=color,width=style['borderWidth'])
        scale=min(2.,math.sqrt(30_000_000/(source['width']*source['height'])))
        for element in sorted(source['elements'],key=lambda e:e.get('zIndex',0)):
            if not element.get('visible',True) or element.get('opacity',1)<=0:
                continue
            if element['type']=='text':
                _write_text(page,element)
                count+=bool(element.get('text','').strip())
                continue
            if element.get('rebuildVector') and element.get('shape')=='rect':
                box=fitz.Rect(element['x'],element['y'],element['x']+element['width'],element['y']+element['height'])
                color=tuple(v/255 for v in rgba(element.get('stroke','#000000'))[:3])
                page.draw_rect(box,color=color,width=element.get('strokeWidth',1),fill=None)
                continue
            if element.get('rebuildVector') and element.get('shape')=='line':
                width=element.get('strokeWidth',1)
                if width>0:
                    cx,cy=element['x']+element['width']/2,element['y']+element['height']/2
                    angle=math.radians(element.get('rotation',0))
                    dx,dy=math.cos(angle)*element['width']/2,math.sin(angle)*element['width']/2
                    color=tuple(v/255 for v in rgba(element.get('stroke','#000000'))[:3])
                    page.draw_line((cx-dx,cy-dy),(cx+dx,cy+dy),color=color,width=width)
                continue
            image=element_bitmap(element,document['id'],output_dir,scale)
            if element.get('opacity',1)!=1:
                image.putalpha(image.getchannel('A').point(lambda a:round(a*element['opacity'])))
            if element.get('rotation',0)%360:
                image=image.rotate(-element['rotation'],Image.Resampling.BICUBIC,expand=True)
            cx=element['x']+element['width']/2
            cy=element['y']+element['height']/2
            box=fitz.Rect(cx-image.width/(2*scale),cy-image.height/(2*scale),cx+image.width/(2*scale),cy+image.height/(2*scale))
            if source.get('appearanceStrategy')=='blank-page-typed-reconstruction' and element['type']=='image':
                _insert_rebuild_image(page,box,image,source.get('paperColor','#ffffff'))
            else:
                buffer=io.BytesIO()
                image.save(buffer,format='PNG')
                page.insert_image(box,stream=buffer.getvalue(),keep_proportion=False)
    return count


@tracked('export.text_pdf')
def write_text_pdf(document: dict, output_dir: Path, destination: Path):
    """Visible, searchable text plus preserved images, never a hidden OCR layer."""
    # Current edits win over stored OCR appearances. Only unchanged OCR text
    # with an appearance is eligible for matching.
    work=copy.deepcopy(document)
    for page in work['pages']:
        for element in page['elements']:
            if element.get('fontMatch') and not valid_runs(element):
                element['source']='manual'
    with progress_span(0,60):
        v3_pages=[p for p in work['pages'] if p.get('recognitionBackend') in {'v3','pp-structurev3','native+v3'}]
        legacy_pages=[p for p in work['pages'] if p.get('recognitionBackend') not in {'v3','pp-structurev3','native+v3'}]
        if v3_pages:
            # V3 scenes already carry semantic blocks and physical OCR
            # lines. Keep the same bounded fitter as the rebuild pipeline.
            from .rebuild_fonts import match_rebuild_fonts
            match_rebuild_fonts(dict(work,pages=v3_pages),output_dir)
        if legacy_pages:
            match_document_fonts(dict(work,pages=legacy_pages),output_dir)
    with progress_span(60,70):
        prepare_fonts(work)
    with fitz.open() as pdf:
        with progress_span(70,85):
            count=append_text_pages(pdf,work,output_dir)
        if not count:
            raise ValueError('没有可导出的独立文字；请开启 OCR 或校对文字区域。图文混合区域仍保留为图片。')
        with progress_span(85,90):
            payload=json.dumps(portable_document(document,output_dir),ensure_ascii=False).encode()
        pdf.embfile_add('scene-document.json',payload,filename='scene-document.json',desc='Editable scene with independent text and preserved imagery')
        pdf.set_metadata({'title':document['name'],'creator':'PDF OCR Parser','subject':'Visible PDF text and preserved composite images'})
        with progress_span(90,100):
            save_verified(pdf,destination,[character_inventory(page) for page in work['pages']])
