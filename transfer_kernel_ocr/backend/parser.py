"""Coordinate preserving PDF/image decomposition.

The semantic model and the original appearance deliberately coexist.  Every
visible source pixel belongs to at most one appearance asset.  A separate
estimated background fills holes after edits; it is never an unchanged scan
under editable text.  This preserves unsupported glyphs, clips and blends on
import without pretending to reconstruct hidden artwork from a flat scan.
"""
from __future__ import annotations

from .usage import tracked

import html
import importlib.util
import math
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import pymupdf as fitz

from .progress import report
from .options import selected_pages
from .scan_layout import partition_text
from .rebuild_layout import image_regions, region_pixels, _overlap
from .rebuild_tables import ruled_tables
from .rebuild_frames import text_frames, frame_rule_mask
from .rebuild_text import organize_text
from .style_templates import observe_styles


MAX_PAGES = 100
MAX_PIXELS = 40_000_000
MAX_ELEMENTS = 1800
RENDER_SCALE = 2.0
_OCR_ENGINE = None
_OCR_ERROR = None
_OCR_LOCK = threading.Lock()
_OCR_RUN_LOCK = threading.Lock()


def ocr_status() -> dict:
    available = importlib.util.find_spec("rapidocr_onnxruntime") is not None
    return {
        "available": available and _OCR_ERROR is None,
        "engine": "RapidOCR / ONNX Runtime",
        "languages": ["中文", "English"],
        "local": True,
        "loaded": _OCR_ENGINE is not None,
        "message": _OCR_ERROR or ("本地 OCR 可用，首次解析时加载模型" if available else "未安装 rapidocr-onnxruntime，扫描件将保留为可编辑图像区域"),
    }


@tracked('document.legacy_ocr_initialize_or_reuse')
def _engine():
    global _OCR_ENGINE, _OCR_ERROR
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _OCR_ENGINE = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)
            except Exception as exc:
                _OCR_ERROR = f"本地 OCR 初始化失败：{type(exc).__name__}: {exc}"
                raise RuntimeError(_OCR_ERROR) from exc
    return _OCR_ENGINE


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _color(value, default="#000000") -> str:
    if value is None:
        return default
    if isinstance(value, int):
        return f"#{value & 0xFFFFFF:06x}"
    values = list(value)
    if len(values) == 1:
        values *= 3
    if len(values) == 4:
        c, m, y, k = values
        values = [1 - min(1, c + k), 1 - min(1, m + k), 1 - min(1, y + k)]
    return "#" + "".join(f"{max(0, min(255, round(float(v) * 255))):02x}" for v in values[:3])


def _estimate_ink_color(crop) -> str:
    """Infer foreground color without treating sparse ink as background.

    Comparing with the border also works for pale lettering on dark paper;
    choosing the darkest fraction would incorrectly select the dark paper.
    """
    if crop.size == 0:
        return "#000000"
    border=np.concatenate((crop[0],crop[-1],crop[:,0],crop[:,-1]))
    background=np.median(border,axis=0)
    pixels=crop.reshape(-1,3).astype(np.float32)
    contrast=np.max(np.abs(pixels-background),axis=1)
    different=contrast[contrast>2]
    if len(different):
        # Estimate from solid foreground, leaving anti-aliased edge pixels out.
        threshold=max(2.,float(np.percentile(different,95))*.55)
        ink=np.median(pixels[contrast>=threshold],axis=0)
    else:
        # A blank OCR box has no trustworthy ink sample. Choose a visible
        # editing default instead of the background color.
        ink=np.array([0,0,0] if np.mean(background)>127 else [255,255,255])
    return "#"+"".join(f"{max(0,min(255,round(float(v)))):02x}" for v in ink)


def _url(document_id: str, relative: str) -> str:
    return f"/files/{document_id}/{relative.replace(chr(92), '/')}"


def _base(kind: str, name: str, bbox, seq: int, source="native") -> dict:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    return {"id": _id(kind), "type": kind, "name": name, "x": x0, "y": y0,
            "width": max(.1, x1 - x0), "height": max(.1, y1 - y0),
            "rotation": 0, "zIndex": int(seq), "opacity": 1, "visible": True,
            "locked": False, "source": source, "reviewRequired": False}


def _transform_point(p, matrix):
    q = fitz.Point(float(p[0]), float(p[1])) * matrix
    return [float(q.x), float(q.y)]


def _transform_rect(rect, matrix):
    return list(fitz.Rect(rect) * matrix)


def _bbox_from_polygon(polygon):
    arr = np.asarray(polygon, dtype=float)
    return [float(arr[:, 0].min()), float(arr[:, 1].min()), float(arr[:, 0].max()), float(arr[:, 1].max())]


def _rect_polygon(bbox):
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _enlarge_quad(polygon, padding):
    pts = np.array(polygon, dtype=float)
    u = pts[1] - pts[0]
    v = pts[3] - pts[0]
    u /= max(1e-6, np.linalg.norm(u))
    v /= max(1e-6, np.linalg.norm(v))
    return (pts + np.array([-u-v, u-v, u+v, -u+v]) * padding).tolist()


def _set_oriented_geometry(element, polygon):
    pts = np.asarray(polygon, dtype=float)
    center = np.mean(pts, axis=0)
    width = float(np.linalg.norm(pts[1] - pts[0]))
    height = float(np.linalg.norm(pts[3] - pts[0]))
    element.update(x=float(center[0]-width/2), y=float(center[1]-height/2),
                   width=max(.1,width), height=max(.1,height),
                   rotation=float(math.degrees(math.atan2(pts[1,1]-pts[0,1], pts[1,0]-pts[0,0]))))
    if abs(element["rotation"]) < .05:
        element["rotation"] = 0
    element["polygon"] = [[round(float(v), 4) for v in p] for p in polygon]


def _span_quad(span, bbox, matrix):
    # get_texttrace bboxes have height=size, unlike get_text('rawdict').
    normalized = dict(span, bbox=bbox)
    delta = float(span.get("ascender", .8))-float(span.get("descender", -.2))
    normalized["ascender"] = float(span.get("ascender", .8))/max(.01,delta)
    normalized["descender"] = float(span.get("descender", -.2))/max(.01,delta)
    try:
        quad = fitz.recover_quad(span.get("dir", (1,0)), normalized)
        return [_transform_point(p, matrix) for p in (quad.ul, quad.ur, quad.lr, quad.ll)]
    except (ValueError, KeyError, ZeroDivisionError):
        return _rect_polygon(_transform_rect(bbox, matrix))


def _split_trace(span):
    """Trace order is painting order; split baseline changes, never sort chars."""
    chars = list(span.get("chars", []))
    if not chars:
        return []
    dx, dy = span.get("dir", (1, 0))
    normal = np.array([-dy, dx])
    size = max(1., float(span.get("size", 12)))
    lines, current = [], []
    last = None
    for char in chars:
        origin = np.asarray(char[2], dtype=float)
        if last is not None:
            delta = origin-last
            if abs(float(np.dot(delta, normal))) > size*.35 or float(np.dot(delta, (dx,dy))) < -size:
                if current:
                    lines.append(current)
                current = []
        current.append(char)
        last = origin
    if current:
        lines.append(current)
    return lines


def _trace_text(chars, span):
    text = []
    previous = None
    direction = np.asarray(span.get("dir", (1,0)), dtype=float)
    space = max(1., float(span.get("spacewidth", span.get("size",12)*.25)))
    for char in chars:
        codepoint = int(char[0])
        value = chr(codepoint) if 0 <= codepoint <= 0x10ffff else "�"
        if previous is not None and value != " " and text and not text[-1].endswith(" "):
            prev_box = fitz.Rect(previous[3])
            corners = np.array([[prev_box.x0,prev_box.y0],[prev_box.x1,prev_box.y0],[prev_box.x1,prev_box.y1],[prev_box.x0,prev_box.y1]])
            end = float(np.max(corners @ direction))
            start = float(np.dot(char[2], direction))
            if start-end > space*.55:
                text.append(" ")
        text.append(value)
        previous = char
    return "".join(text)


@tracked('document.native_text')
def _native_text(page, matrix):
    candidates = []
    seen = set()
    for span in page.get_texttrace():
        if span.get("type") == 3 or span.get("opacity",1) <= 0:
            continue  # Invisible OCR layers must not cover visible artwork.
        for chars in _split_trace(span):
            text = _trace_text(chars, span)
            if not text.strip():
                continue
            bbox = fitz.Rect(chars[0][3])
            for char in chars[1:]:
                bbox |= fitz.Rect(char[3])
            key = (text, tuple(round(v,2) for v in bbox), span.get("font"))
            if key in seen:  # Common PDF fill+stroke text pair.
                continue
            seen.add(key)
            polygon = _enlarge_quad(_span_quad(span, bbox, matrix), .65)
            element = _base("text", text[:30], _bbox_from_polygon(polygon), span.get("seqno",0))
            _set_oriented_geometry(element, polygon)
            font = str(span.get("font") or "sans-serif")
            flags = int(span.get("flags",0))
            element.update(text=text, fontFamily=font, fontSize=float(span.get("size",12)),
                           fontWeight="700" if flags & 16 or "bold" in font.lower() else "400",
                           fontStyle="italic" if flags & 2 or "italic" in font.lower() or "oblique" in font.lower() else "normal",
                           color=_color(span.get("color")), align="left", lineHeight=1., confidence=1.,
                           nativeFont=font, nativeOpacity=float(span.get("opacity",1)),
                           writingMode=span.get("wmode",0), paintOrder=span.get("seqno",0),
                           direction=list(span.get("dir",(1,0))),
                           glyphs=[{"unicode":int(c[0]), "glyphId":int(c[1]), "origin":_transform_point(c[2],matrix),
                                    "bbox":_transform_rect(c[3],matrix)} for c in chars])
            if "�" in text or any(ord(c)<32 and c not in "\n\t" for c in text):
                element["reviewRequired"] = True
            masks = [_enlarge_quad(_span_quad(span,c[3],matrix),.5) for c in chars if int(c[0]) != 32]
            candidates.append({"element":element,"bbox":_bbox_from_polygon(polygon),"polygons":masks,"quad":polygon,"textMask":True})
    return candidates


def _svg_for_drawing(drawing, matrix):
    bbox = _transform_rect(drawing.get("rect",(0,0,0,0)),matrix)
    padding = max(.6, float(drawing.get("width") or 0)/2 + .4)
    x0,y0,x1,y1 = bbox
    x0,y0,x1,y1 = x0-padding,y0-padding,x1+padding,y1+padding
    if x1 <= x0 or y1 <= y0:
        return None
    def point(p):
        q = _transform_point(p,matrix)
        return f"{q[0]-x0:.4f},{q[1]-y0:.4f}"
    segments = []
    last = None
    for item in drawing.get("items",[]):
        code = item[0]
        if code == "l":
            if last != item[1]:
                segments.append("M"+point(item[1]))
            segments.append("L"+point(item[2]))
            last=item[2]
        elif code == "c":
            if last != item[1]:
                segments.append("M"+point(item[1]))
            segments.append("C"+" ".join(point(p) for p in item[2:5]))
            last=item[4]
        elif code == "re":
            rect = fitz.Rect(item[1])
            points = [rect.tl,rect.tr,rect.br,rect.bl]
            if len(item)>2 and item[2]<0:
                points.reverse()
            segments.append("M"+point(points[0])+" "+" ".join("L"+point(p) for p in points[1:])+"Z")
            last=None
        elif code == "qu":
            q=item[1]
            points=[q.ul,q.ur,q.lr,q.ll]
            segments.append("M"+point(points[0])+" "+" ".join("L"+point(p) for p in points[1:])+"Z")
            last=None
    if drawing.get("closePath"):
        segments.append("Z")
    if not segments:
        return None
    fill = _color(drawing.get("fill"),"none")
    stroke = _color(drawing.get("color"),"none")
    width = float(drawing.get("width") or 0)
    linecap = drawing.get("lineCap",(0,0,0))
    if isinstance(linecap,(tuple,list)):
        linecap=max(linecap)
    cap = ["butt","round","square"][min(2,max(0,int(linecap or 0)))]
    join = ["miter","round","bevel"][min(2,max(0,int(drawing.get("lineJoin") or 0)))]
    dash = ""
    dash_string = drawing.get("dashes") or ""
    if "[" in dash_string and "]" in dash_string:
        inside = dash_string.split("[",1)[1].split("]",1)[0].strip()
        after = dash_string.split("]",1)[1].strip()
        if inside:
            dash=f' stroke-dasharray="{html.escape(inside)}" stroke-dashoffset="{html.escape(after or "0")}"'
    path = " ".join(segments)
    fill_opacity=drawing.get("fill_opacity")
    stroke_opacity=drawing.get("stroke_opacity")
    fill_opacity=1. if fill_opacity is None else max(0.,min(1.,float(fill_opacity)))
    stroke_opacity=1. if stroke_opacity is None else max(0.,min(1.,float(stroke_opacity)))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{x1-x0:.4f}" height="{y1-y0:.4f}" viewBox="0 0 {x1-x0:.4f} {y1-y0:.4f}">'
           f'<path d="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
           f'fill-opacity="{fill_opacity}" stroke-opacity="{stroke_opacity}" '
           f'fill-rule="{"evenodd" if drawing.get("even_odd") else "nonzero"}" stroke-linecap="{cap}" stroke-linejoin="{join}"{dash}/></svg>')
    return [x0,y0,x1,y1],svg


@tracked('document.native_graphics')
def _native_graphics(page, matrix, output_dir, document_id, page_index):
    candidates=[]
    drawings=page.get_drawings()
    for index,drawing in enumerate(drawings[:MAX_ELEMENTS]):
        result=_svg_for_drawing(drawing,matrix)
        if result is None:
            continue
        bbox,svg=result
        element=_base("vector",f"矢量路径 {index+1}",bbox,drawing.get("seqno",index))
        relative=f"vectors/page-{page_index+1:03d}-{element['id']}.svg"
        (output_dir/relative).write_text(svg,encoding="utf-8")
        element.update(svg=svg,svgSrc=_url(document_id,relative),fill=_color(drawing.get("fill"),"none"),
                       stroke=_color(drawing.get("color"),"none"),strokeWidth=float(drawing.get("width") or 0),
                       paintOrder=drawing.get("seqno",index))
        candidates.append({"element":element,"bbox":bbox,"vectorMask":True})
    bboxlog=page.get_bboxlog()
    image_orders=[(i,_transform_rect(b,matrix)) for i,(typ,b,*_) in enumerate(bboxlog) if typ=="fill-image"]
    used=set()
    for index,info in enumerate(page.get_image_info(xrefs=True)):
        bbox=_transform_rect(info["bbox"],matrix)
        # A full-page scan is decomposed by OCR and connected regions instead.
        area=max(0,bbox[2]-bbox[0])*max(0,bbox[3]-bbox[1])
        if area>page.rect.width*page.rect.height*.82:
            continue
        options=[(sum(abs(a-b) for a,b in zip(rect,bbox)),seq) for seq,rect in image_orders if seq not in used]
        seq=min(options)[1] if options else len(bboxlog)+index
        used.add(seq)
        element=_base("image",f"嵌入图像 {index+1}",bbox,seq)
        element.update(paintOrder=seq,nativeTransform=list(info.get("transform",[])),
                       nativeImageSize=[info.get("width"),info.get("height")],xref=info.get("xref",0))
        candidate={"element":element,"bbox":bbox}
        transform=info.get("transform")
        if transform:
            image_matrix=fitz.Matrix(transform)*matrix
            polygon=[_transform_point(p,image_matrix) for p in ((0,0),(1,0),(1,1),(0,1))]
            u=np.asarray(polygon[1])-polygon[0]
            v=np.asarray(polygon[3])-polygon[0]
            orthogonal=abs(float(np.dot(u,v)))<max(.01,float(np.linalg.norm(u)*np.linalg.norm(v))*.001)
            if orthogonal and u[0]*v[1]-u[1]*v[0]>0:
                _set_oriented_geometry(element,polygon)
                candidate.update(quad=polygon,polygons=[polygon])
            else:
                element.update(nativePolygon=polygon,reviewRequired=True)
        candidates.append(candidate)
    return candidates


@tracked('document.legacy_ocr')
def _ocr_candidates(rgb, scale, existing, start_z, warnings):
    try:
        engine=_engine()
        with _OCR_RUN_LOCK:
            results,_=engine(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    except Exception as exc:
        warnings.append(f"OCR 未完成：{exc}。可见内容仍保留在图像区域中。")
        return []
    candidates=[]
    native_boxes=[c["bbox"] for c in existing if c["element"]["type"]=="text"]
    for index,result in enumerate(results or []):
        points,text,confidence=result[:3]
        if not str(text).strip():
            continue
        polygon=(np.asarray(points,dtype=float)/scale).tolist()
        bbox=_bbox_from_polygon(polygon)
        area=max(1,(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        overlaps=False
        for box in native_boxes:
            overlap=max(0,min(bbox[2],box[2])-max(bbox[0],box[0]))*max(0,min(bbox[3],box[3])-max(bbox[1],box[1]))
            if overlap/area>.45:
                overlaps=True
                break
        if overlaps:
            continue
        polygon=_enlarge_quad(polygon,.6)
        element=_base("text",str(text)[:30],_bbox_from_polygon(polygon),start_z+index,"ocr")
        _set_oriented_geometry(element,polygon)
        font_size=max(4,element["height"]*.78)
        x0,y0,x1,y1=_pixel_bbox(_bbox_from_polygon(polygon),scale,rgb.shape)
        ink_color=_estimate_ink_color(rgb[y0:y1,x0:x1])
        element.update(text=str(text),fontFamily="sans-serif",fontSize=font_size,fontWeight="400",fontStyle="normal",
                       color=ink_color,align="left",lineHeight=1.,
                       confidence=float(confidence),reviewRequired=float(confidence)<.9,
                       fontEstimated=True,readingOrder=index)
        candidates.append({"element":element,"bbox":_bbox_from_polygon(polygon),"polygons":[polygon],"quad":polygon,"textMask":True})
    return candidates


def _pixel_bbox(bbox, scale, shape):
    x0,y0,x1,y1=bbox
    h,w=shape[:2]
    return max(0,min(w,int(math.floor(x0*scale)))),max(0,min(h,int(math.floor(y0*scale)))),max(0,min(w,int(math.ceil(x1*scale)))),max(0,min(h,int(math.ceil(y1*scale))))


def _candidate_mask(candidate,rgb,scale):
    x0,y0,x1,y1=_pixel_bbox(candidate["bbox"],scale,rgb.shape)
    if x1<=x0 or y1<=y0:
        return None
    mask=np.zeros((y1-y0,x1-x0),np.uint8)
    if candidate.get("polygons"):
        for polygon in candidate["polygons"]:
            pts=np.round(np.asarray(polygon)*scale-[x0,y0]).astype(np.int32)
            cv2.fillConvexPoly(mask,pts,255)
    elif candidate.get("vectorMask"):
        try:
            svg=candidate["element"]["svg"]
            with fitz.open(stream=svg.encode(),filetype="svg") as doc:
                pix=doc[0].get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=True)
                alpha=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,4)[:,:,3]
            # The clipping offset matters for paths extending beyond the page.
            bx0,by0,_,_=candidate["bbox"]
            offset_x=max(0,int(round(x0-bx0*scale)))
            offset_y=max(0,int(round(y0-by0*scale)))
            hh=min(mask.shape[0],alpha.shape[0]-offset_y)
            ww=min(mask.shape[1],alpha.shape[1]-offset_x)
            if hh>0 and ww>0:
                mask[:hh,:ww]=np.where(alpha[offset_y:offset_y+hh,offset_x:offset_x+ww]>0,255,0)
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
        except Exception:
            mask[:]=255
    else:
        mask[:]=255
    if candidate.get("textMask"):
        crop=rgb[y0:y1,x0:x1]
        border=np.concatenate((crop[0],crop[-1],crop[:,0],crop[:,-1]))
        bg=np.median(border,axis=0)
        # Uniform backgrounds stay with the background, not the text. Texture
        # may remain attached to a glyph; flat scans cannot reveal occluded art.
        contrast=np.max(np.abs(crop.astype(np.float32)-bg),axis=2)
        foreground=np.where(contrast>3,255,0).astype(np.uint8)
        foreground=cv2.dilate(foreground,np.ones((3,3),np.uint8))
        mask=cv2.bitwise_and(mask,foreground)
    return (x0,y0,x1,y1),mask


def _save_png(array,output_dir,document_id,relative):
    Image.fromarray(array).save(output_dir/relative,optimize=False)
    return _url(document_id,relative)


def _local_paper(pixels):
    """Dominant local color, without using neighboring ink as replacement paper."""
    samples=pixels.reshape(-1,3)[::max(1,pixels.size//60000)].astype(np.uint8)
    quantized=(samples//8).astype(np.int32)
    keys=quantized[:,0]*1024+quantized[:,1]*32+quantized[:,2]
    values,counts=np.unique(keys,return_counts=True)
    return np.median(samples[keys==values[counts.argmax()]],axis=0).astype(np.uint8)


def _appearance(candidate,rgb,owner,index,scale,output_dir,document_id,page_index):
    bbox,mask_data=candidate["maskData"]
    x0,y0,x1,y1=bbox
    owned=owner[y0:y1,x0:x1]==index
    if not np.any(owned):
        return None
    crop=np.dstack((rgb[y0:y1,x0:x1],np.where(owned,255,0).astype(np.uint8)))
    element=candidate["element"]
    # Pixel aligned, axis-aligned elements composite exactly without resampling.
    if not candidate.get("quad") or not element.get("rotation"):
        element.update(x=x0/scale,y=y0/scale,width=(x1-x0)/scale,height=(y1-y0)/scale)
    elif abs(element["rotation"]/90-round(element["rotation"]/90)) < .0001:
        quarter_turns=int(round(element["rotation"]/90))
        crop=np.ascontiguousarray(np.rot90(crop,k=quarter_turns))
        cx,cy=(x0+x1)/(2*scale),(y0+y1)/(2*scale)
        local_width,local_height=crop.shape[1]/scale,crop.shape[0]/scale
        element.update(x=cx-local_width/2,y=cy-local_height/2,width=local_width,height=local_height)
    else:
        polygon=np.asarray(candidate["quad"],dtype=np.float32)*scale-[x0,y0]
        w=max(1,int(round(element["width"]*scale)))
        h=max(1,int(round(element["height"]*scale)))
        transform=cv2.getAffineTransform(polygon[[0,1,3]].astype(np.float32),np.array([[0,0],[w,0],[0,h]],np.float32))
        crop=cv2.warpAffine(crop,transform,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0,0))
        element["appearanceResampled"]=True
    folder="text" if element["type"]=="text" else "vectors" if element["type"]=="vector" else "images"
    relative=f"{folder}/page-{page_index+1:03d}-{element['id']}.png"
    src=_save_png(crop,output_dir,document_id,relative)
    element.update(appearanceSrc=src,originalAppearanceSrc=src,appearanceMode="original-pixels")
    if element["type"] in ("image","region","background"):
        element["src"]=src
    return element


def _compose_scan_page(rgb,width,height,scale,output_dir,document_id,index,source_rotation,source_name,ocr_enabled,strict=False):
    """Keep all artwork in one image; extract only independent body text."""
    warnings=[]
    candidates=_ocr_candidates(rgb,scale,[],0,warnings) if ocr_enabled else []
    if strict and warnings:
        raise ValueError('无法生成文本 PDF：'+warnings[0])
    if len(candidates)>MAX_ELEMENTS:
        candidates=candidates[:MAX_ELEMENTS]
        warnings.append('本页文字候选较多，超出上限的内容保留为原图。')
    candidates,retained=partition_text(rgb,candidates,scale)
    source_src=_save_png(rgb,output_dir,document_id,f'pages/page-{index+1:03d}.png')
    owner=np.full(rgb.shape[:2],-1,dtype=np.int32)
    for n,candidate in enumerate(candidates):
        candidate['maskData']=_candidate_mask(candidate,rgb,scale)
        if candidate['maskData']:
            (x0,y0,x1,y1),mask=candidate['maskData']
            owner[y0:y1,x0:x1][mask>0]=n
    preserved=rgb.copy()
    for n,candidate in enumerate(candidates):
        if not candidate['maskData']:
            continue
        (x0,y0,x1,y1),_=candidate['maskData']
        # Only accepted, locally uniform text regions are cleared. Everything
        # else, including unrecognized characters, remains in the page image.
        pad=max(2,round(scale*2))
        left,top,right,bottom=max(0,x0-pad),max(0,y0-pad),min(rgb.shape[1],x1+pad),min(rgb.shape[0],y1+pad)
        crop=rgb[top:bottom,left:right]
        unclaimed=crop[owner[top:bottom,left:right]<0]
        paper=_local_paper(unclaimed if len(unclaimed) else crop)
        preserved[y0:y1,x0:x1][owner[y0:y1,x0:x1]==n]=paper
    image=_base('image','保留的图文与页面底图',[0,0,width,height],-10000,'raster')
    image.update(src=_save_png(preserved,output_dir,document_id,f'images/page-{index+1:03d}-preserved.png'),
                 locked=True,scanComposite=True,retainedText=retained,reviewRequired=bool(retained))
    elements=[image]
    for n,candidate in enumerate(candidates):
        if candidate.get('maskData'):
            element=_appearance(candidate,rgb,owner,n,scale,output_dir,document_id,index)
            if element:
                element['textRole']='independent'
                elements.append(element)
    warnings.append('仅转换独立文字；图案及图文混合区域整体保留为图片，不再自动拆分图案对象。')
    if retained:
        warnings.append(f'{len(retained)} 行文字涉及图案、复杂背景或不确定的识别边界，已随图片保留；转录与原坐标保存在图像元数据中。')
    if ocr_enabled and not candidates:
        warnings.append('本页未找到可独立转换的文字，完整页面保留为图片。')
    warnings.append('OCR 转录、区域判断和替代字体需对照原图校对；未识别区域不会被清除。')
    return {'id':_id('page'),'index':index,'width':width,'height':height,'rotation':0,
            'sourceRotation':source_rotation,'sourceSrc':source_src,'sourceName':source_name,
            'elements':elements,'warnings':warnings,'renderScale':scale,
            'appearanceStrategy':'conservative-text-and-preserved-image','backgroundReconstruction':'local-paper'}


@tracked('document.legacy_rebuild_compose')
def _compose_rebuild_page(rgb,width,height,scale,native,output_dir,document_id,index,source_rotation,source_name,ocr_enabled):
    """Recognize objects then rebuild from raw crops, never a cleared scan."""
    warnings=[]
    candidates=list(native)
    if ocr_enabled:
        candidates.extend(_ocr_candidates(rgb,scale,candidates,len(candidates),warnings))
        if warnings:
            raise ValueError('无法重建文档：'+warnings[0])
    if len(candidates)>MAX_ELEMENTS:
        raise ValueError('本页文字候选超过安全上限，请降低页面复杂度或分批处理')
    structures=ruled_tables(rgb,candidates,scale,include_frames=True)
    # Frame geometry is useful even if OCR misses a word or an entire line.
    # It owns only its strokes, not every pixel or text candidate inside it.
    prose_frames=text_frames(rgb,candidates,scale,excluded_boxes=[t['bbox'] for t in structures])
    for frame in prose_frames:
        frame['id']='prose-'+frame['id']
    rule_mask=frame_rule_mask(rgb.shape,prose_frames,scale,margin=1) if prose_frames else None
    table_indices={i for structure in structures for i in structure['candidateIndices']}
    independent,regions,paper=image_regions(rgb,[c for i,c in enumerate(candidates) if i not in table_indices],scale,
                                             excluded_boxes=[t['bbox'] for t in structures],ignored_mask=rule_mask)
    if any(_overlap(region['bbox'],structure['bbox']) for region in regions for structure in structures):
        # A figure enclosing a grid is not proven to be an independent table.
        structures=[]
        table_indices=set()
        independent,regions,paper=image_regions(rgb,candidates,scale,ignored_mask=rule_mask)
        warnings.append('表格候选与图案区域交叠，整区保留为图片，未声称恢复表格结构。')
    structures.extend(prose_frames)
    for structure in structures:
        for i in structure['candidateIndices']:
            candidates[i]['element']['tableId' if structure['contentType']=='table' else 'frameId']=structure['id']
    accepted=independent+[candidates[i] for i in sorted(table_indices)]
    source=_save_png(rgb,output_dir,document_id,f'pages/page-{index+1:03d}.png')
    elements=[]
    tables=[t for t in structures if t['contentType']=='table']
    for table in tables:
        # Keep original pixels alongside inferred cells, without placing a
        # second raster table behind the editable reconstruction.
        b=table['bbox']
        x0,y0=max(0,int(np.floor(b[0]*scale))),max(0,int(np.floor(b[1]*scale)))
        x1,y1=min(rgb.shape[1],int(np.ceil(b[2]*scale))),min(rgb.shape[0],int(np.ceil(b[3]*scale)))
        table['referenceBbox']=[x0/scale,y0/scale,x1/scale,y1/scale]
        table['referenceSrc']=_save_png(rgb[y0:y1,x0:x1],output_dir,document_id,
                                        f'images/page-{index+1:03d}-{table["id"]}-reference.png')
    for frame in (t for t in structures if t['contentType']=='decoration'):
        bbox=[frame['columns'][0],frame['rows'][0],frame['columns'][-1],frame['rows'][-1]]
        e=_base('shape','简单矩形文本框',bbox,-2000,'layout')
        e['id']=frame['id']
        e.update(shape='rect',stroke=frame['observedStyle']['borderColor'],
                 strokeWidth=frame['observedStyle']['borderWidth'],fill='none',
                 templateId='decoration.rect.v1',contentType='decoration',rebuildVector=True,reviewRequired=True)
        if frame.get('frameEvidence'):
            e['frameEvidence']=frame['frameEvidence']
        elements.append(e)
    for n,region in enumerate(regions):
        pixels=region_pixels(rgb,region,scale,excluded_mask=rule_mask)
        if rule_mask is not None:
            x0,y0,x1,y1=region['pixelBox']
            if np.any(rule_mask[y0:y1,x0:x1]):
                region['boundary']['clip']='alpha-text-and-rule-exclusion' if region.get('excludedText') else 'alpha-rule-exclusion'
        region.pop('pixelBox')
        e=_base('image','完整图文区域',region['bbox'],-1000+n,'raster')
        e.update(region,src=_save_png(pixels,output_dir,document_id,f'images/page-{index+1:03d}-region-{n+1}.png'),
                 templateId='image.original.v1',scanComposite=True)
        elements.append(e)
    owner=np.full(rgb.shape[:2],-1,dtype=np.int32)
    for n,candidate in enumerate(accepted):
        candidate['maskData']=_candidate_mask(candidate,rgb,scale)
        if candidate['maskData']:
            (x0,y0,x1,y1),mask=candidate['maskData']
            owner[y0:y1,x0:x1][mask>0]=n
    for n,candidate in enumerate(accepted):
        if not candidate.get('maskData'):
            continue
        e=_appearance(candidate,rgb,owner,n,scale,output_dir,document_id,index)
        if e:
            e['textRole']='independent'
            elements.append(e)
    for structure in structures:
        structure.pop('candidateIndices')
    page=dict(id=_id('page'),index=index,width=width,height=height,rotation=0,
              sourceRotation=source_rotation,sourceSrc=source,sourceName=source_name,elements=elements,
              tables=tables,warnings=warnings,renderScale=scale,paperColor=paper,
              appearanceStrategy='blank-page-typed-reconstruction',backgroundReconstruction='none',
              contentModelVersion=1)
    observe_styles(page)
    organize_text(page)
    # Keep the recognition/ownership evidence separate from PDF editor guesses
    # about text boxes. No font files or full-page raster are duplicated here.
    owners={e['id']:e['id'] for e in elements if e['type']=='text'}
    for e in elements:
        for line in e.get('retainedText',[]):
            if line.get('textId'):
                owners[line['textId']]=e['id']
    page['recognition']=[dict(id=c['element']['id'],text=c['element']['text'],bbox=c['bbox'],
                              polygon=c['quad'],source=c['element'].get('source'),
                              confidence=c['element'].get('confidence'),ownerId=owners.get(c['element']['id']))
                         for c in candidates]
    warnings.append('新建文档按文字、简单网格表格和完整图片重建；图文边界为保守几何推断，复杂表格及未知区域需对照原图校对。')
    return page


@tracked('document.legacy_compose')
def _compose_page(rgb,width,height,scale,candidates,output_dir,document_id,index,source_rotation,source_name,ocr_enabled,scan_like):
    if scan_like and not any(c['element']['type']=='text' for c in candidates):
        return _compose_scan_page(rgb,width,height,scale,output_dir,document_id,index,source_rotation,source_name,ocr_enabled)
    warnings=[]
    if len(candidates)>MAX_ELEMENTS:
        candidates=candidates[:MAX_ELEMENTS]
        warnings.append("本页元素较多，超出独立提取上限的内容完整保留在残余图层中。")
    if ocr_enabled and scan_like:
        start=max([c["element"]["zIndex"] for c in candidates]+[0])+100
        candidates.extend(_ocr_candidates(rgb,scale,candidates,start,warnings))
    for candidate in candidates:
        mask_data=_candidate_mask(candidate,rgb,scale)
        candidate["maskData"]=mask_data
    candidates=[c for c in candidates if c.get("maskData")]
    candidates.sort(key=lambda c:c["element"]["zIndex"])
    owner=np.full(rgb.shape[:2],-1,dtype=np.int32)
    for n,candidate in enumerate(candidates):
        (x0,y0,x1,y1),mask=candidate["maskData"]
        owner[y0:y1,x0:x1][mask>0]=n
    original=f"pages/page-{index+1:03d}.png"
    source_src=_save_png(rgb,output_dir,document_id,original)
    # A clean estimated substrate has no source text pixels under editable text.
    border=np.concatenate((rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]))
    paper=np.median(border,axis=0).astype(np.uint8)
    background=np.empty_like(rgb)
    background[:]=paper
    unowned=owner<0
    background[unowned]=rgb[unowned]
    # Fill only small masks via inpainting. Large pictures use estimated paper.
    text_holes=np.zeros(owner.shape,dtype=np.uint8)
    for n,candidate in enumerate(candidates):
        if candidate["element"]["type"]=="text":
            (x0,y0,x1,y1),_=candidate["maskData"]
            text_holes[y0:y1,x0:x1][owner[y0:y1,x0:x1]==n]=255
    if np.any(text_holes):
        # Work on patches to avoid a costly full-page inpaint and to keep source
        # pixels from other movable objects out of the estimated substrate.
        holes=cv2.dilate(text_holes,np.ones((5,5),np.uint8))
        contours,_=cv2.findContours(holes,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            pad=5
            x0,y0,x1,y1=max(0,x-pad),max(0,y-pad),min(rgb.shape[1],x+w+pad),min(rgb.shape[0],y+h+pad)
            patch_mask=text_holes[y0:y1,x0:x1]
            if patch_mask.size<2_000_000 and min(patch_mask.shape)>2:
                # The surrounding source pixels can contain a colored native
                # panel or photograph. They inform the estimate, but only the
                # repaired pixels are copied into the substrate; the original
                # text never survives underneath an editable text element.
                patch=cv2.inpaint(rgb[y0:y1,x0:x1],patch_mask,3,cv2.INPAINT_TELEA)
                changed=patch_mask>0
                background[y0:y1,x0:x1][changed]=patch[changed]
    # Separate residual from substrate. Initial pixel ownership is disjoint.
    substrate=np.empty_like(rgb)
    substrate[:]=paper
    substrate[~unowned]=background[~unowned]
    substrate_src=_save_png(substrate,output_dir,document_id,f"backgrounds/page-{index+1:03d}-substrate.png")
    residual=np.dstack((rgb,np.where(unowned,255,0).astype(np.uint8)))
    residual_src=_save_png(residual,output_dir,document_id,f"backgrounds/page-{index+1:03d}-residual.png")
    base=_base("background","修复底色（移动后填补）",[0,0,width,height],-10002,"residual")
    base.update(src=substrate_src,locked=True,reviewRequired=True,backgroundEstimated=True)
    rest=_base("background","残余内容（未分类可见像素）",[0,0,width,height],-10001,"residual")
    rest.update(src=residual_src,locked=True,reviewRequired=True)
    elements=[base,rest]
    for n,candidate in enumerate(candidates):
        element=_appearance(candidate,rgb,owner,n,scale,output_dir,document_id,index)
        if element:
            elements.append(element)
    # Reading order remains auxiliary. Never reorder geometry or paint order.
    text_elements=[e for e in elements if e["type"]=="text"]
    for order,element in enumerate(sorted(text_elements,key=lambda e:(round(e["y"]/max(1,e.get("fontSize",12))),e["x"]))):
        element["readingOrder"]=order
    if any(e["source"]=="ocr" for e in elements):
        warnings.append("OCR 文字与字体估计可能有误；初始外观使用原始像素，修改文字或样式后会使用可编辑字体重新排版。")
    if any(e.get("appearanceResampled") for e in elements):
        warnings.append("非直角旋转元素保留角度和相对位置；局部外观经过重采样，边缘可能存在轻微像素差异。")
    warnings.append("移动或删除对象后，被原对象遮挡的内容只能估计修复；复杂背景、交叠图案、透明混合和特殊字体需人工检查。")
    if scan_like:
        warnings.append("图案与未识别内容保留在原图层，不执行扫描图案对象拆分。")
    return {"id":_id("page"),"index":index,"width":width,"height":height,"rotation":0,
            "sourceRotation":source_rotation,"sourceSrc":source_src,"sourceName":source_name,
            "elements":elements,"warnings":warnings,"renderScale":scale,
            "appearanceStrategy":"disjoint-pixel-ownership", "backgroundReconstruction":"estimated"}


def _image_pages(path):
    try:
        image=Image.open(path)
        for frame in range(getattr(image,"n_frames",1)):
            image.seek(frame)
            if image.width*image.height>MAX_PIXELS:
                raise ValueError(f"图片 {path.name} 第 {frame+1} 页超过 4000 万像素限制。")
            current=ImageOps.exif_transpose(image.copy())
            if current.mode in ("RGBA","LA") or "transparency" in current.info:
                rgba=current.convert("RGBA")
                flat=Image.new("RGBA",rgba.size,"white")
                flat.alpha_composite(rgba)
                current=flat.convert("RGB")
            else:
                current=current.convert("RGB")
            # Scanned image points use a deterministic 144 dpi convention.
            # Store native pixel size as metadata; do not trust inconsistent EXIF dpi.
            yield np.asarray(current).copy(),current.width/2,current.height/2,2.
        image.close()
    except (UnidentifiedImageError,OSError) as exc:
        raise ValueError(f"无法读取图片 {path.name}：{exc}") from exc


@tracked('document.parse')
def parse_document(paths: list[Path], output_dir: Path, document_id: str, name: str, ocr: bool=True, pages: str="", mode: str="editable", recognition_backend: str | None=None) -> dict:
    """Parse PDF and raster files into a local, editable multi-page model."""
    if recognition_backend == 'v3':
        from .structure_parser import parse_document as parse_structure
        return parse_structure(paths, output_dir, document_id, name, ocr=ocr, pages=pages, mode=mode)
    if recognition_backend not in {None, 'legacy'}:
        raise ValueError('不支持的识别后端')
    if mode not in {'editable','text_pdf','rebuild'}:
        raise ValueError('不支持的转换模式')
    if mode in {'text_pdf','rebuild'} and not ocr:
        raise ValueError('扫描 PDF 转文本 PDF 需要开启 OCR')
    output_dir=Path(output_dir)
    for folder in ("images","vectors","text","backgrounds","pages","summary","exports"):
        (output_dir/folder).mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc).isoformat()
    document={"schemaVersion":1,"id":document_id,"name":name,"createdAt":now,"updatedAt":now,
              "pages":[],"warnings":[],"stats":{},
              "parser":{"name":"local-layout-parser","version":"1.0","coordinateSystem":"points-top-left",
                        "originalAppearance":"exclusive pixel ownership","ocrRequested":bool(ocr),"mode":mode,
                        "scanArtworkExtraction":False}}
    if not paths:
        raise ValueError("请至少上传一个 PDF 或图片文件。")
    supported={".pdf",".png",".jpg",".jpeg",".tif",".tiff",".webp",".bmp"}
    for file_index,file_path in enumerate(paths):
        path=Path(file_path)
        suffix=path.suffix.lower()
        if suffix not in supported:
            raise ValueError(f"不支持的文件类型：{path.name}")
        if suffix==".pdf":
            try:
                pdf=fitz.open(path)
            except Exception as exc:
                raise ValueError(f"无法打开 PDF {path.name}：{exc}") from exc
            with pdf:
                if pdf.needs_pass:
                    raise ValueError(f"PDF {path.name} 已加密，请先解除密码保护后上传。")
                selection = selected_pages(pages, len(pdf))
                if len(document["pages"])+len(selection)>MAX_PAGES:
                    raise ValueError("每个文档最多支持 100 页。")
                for page_ordinal,page_number in enumerate(selection):
                    report(100*(file_index+page_ordinal/max(1,len(selection)))/len(paths),'解析页面与 OCR',f'文件 {file_index+1} / {len(paths)}；物理第 {page_number+1} 页',current=page_ordinal+1,total=len(selection))
                    pdf_page = pdf[page_number]
                    index=len(document["pages"])
                    width,height=float(pdf_page.rect.width),float(pdf_page.rect.height)
                    if width<=0 or height<=0:
                        raise ValueError(f"PDF 第 {index+1} 页尺寸无效。")
                    scale=min(RENDER_SCALE,math.sqrt(MAX_PIXELS/(width*height))*.999)
                    if scale<.2:
                        raise ValueError(f"PDF 第 {index+1} 页尺寸过大，无法安全渲染。")
                    pix=pdf_page.get_pixmap(matrix=fitz.Matrix(scale,scale),colorspace=fitz.csRGB,alpha=False)
                    rgb=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3).copy()
                    matrix=pdf_page.rotation_matrix
                    extraction_warnings=[]
                    try:
                        candidates=_native_text(pdf_page,matrix)
                    except Exception as exc:
                        candidates=[]
                        extraction_warnings.append(f"原生文字结构提取失败，已保留页面像素并尝试图像识别：{type(exc).__name__}")
                    if mode!='rebuild':
                        try:
                            candidates+=_native_graphics(pdf_page,matrix,output_dir,document_id,index)
                        except Exception as exc:
                            extraction_warnings.append(f"部分矢量或图像结构无法提取，已保留在残余图层：{type(exc).__name__}")
                    native_chars=sum(len(c["element"].get("text","")) for c in candidates)
                    try:
                        image_area=sum(max(0,fitz.Rect(i["bbox"]).width)*max(0,fitz.Rect(i["bbox"]).height) for i in pdf_page.get_image_info())
                    except Exception:
                        image_area=width*height
                    scan_like=native_chars<8 or image_area>width*height*.12 or bool(extraction_warnings)
                    if mode=='rebuild':
                        page=_compose_rebuild_page(rgb,width,height,scale,candidates,output_dir,document_id,index,pdf_page.rotation,path.name,ocr and scan_like)
                    elif mode=='text_pdf':
                        page=_compose_scan_page(rgb,width,height,scale,output_dir,document_id,index,pdf_page.rotation,path.name,True,strict=True)
                    else:
                        page=_compose_page(rgb,width,height,scale,candidates,output_dir,document_id,index,pdf_page.rotation,path.name,ocr,scan_like)
                    page["sourcePage"] = page_number + 1
                    page["warnings"].extend(extraction_warnings)
                    if scale<RENDER_SCALE:
                        page["warnings"].append("超大页面已降低原始外观渲染分辨率以控制内存占用。")
                    document["pages"].append(page)
        else:
            with Image.open(path) as image:
                selection = set(selected_pages(pages, getattr(image, "n_frames", 1)))
            frame_ordinal=0
            for frame_number, (rgb,width,height,scale) in enumerate(_image_pages(path)):
                if frame_number not in selection:
                    continue
                report(100*(file_index+frame_ordinal/max(1,len(selection)))/len(paths),'识别图片页面',current=frame_ordinal+1,total=len(selection))
                frame_ordinal+=1
                if len(document["pages"])>=MAX_PAGES:
                    raise ValueError("每个文档最多支持 100 页。")
                if mode=='rebuild':
                    page=_compose_rebuild_page(rgb,width,height,scale,[],output_dir,document_id,len(document['pages']),0,path.name,ocr)
                else:
                    page=_compose_scan_page(rgb,width,height,scale,output_dir,document_id,len(document['pages']),0,path.name,ocr,strict=mode=='text_pdf')
                page["sourcePage"] = frame_number + 1
                page["nativePixelSize"]=[rgb.shape[1],rgb.shape[0]]
                document["pages"].append(page)
    if not document["pages"]:
        raise ValueError("文件中没有可解析的页面。")
    counts=Counter(e["type"] for p in document["pages"] for e in p["elements"])
    document["stats"]={"pageCount":len(document["pages"]),"elementCount":sum(counts.values()),"byType":dict(counts),
                       "reviewCount":sum(bool(e.get("reviewRequired")) for p in document["pages"] for e in p["elements"])}
    document["warnings"]=["原始外观用于保真，可编辑文本用于修改。扫描图片中被遮挡的背景、精确字体和隐藏图层无法可靠反推。"]
    report(100,'页面解析完成')
    return document
