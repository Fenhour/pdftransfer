from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import uuid
import zipfile

import pymupdf as fitz
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from PIL import Image

from .demo import create_demo
from .exporter import portable_document, read_asset, scene_html, write_pdf, write_text_pdf
from .reading import convert_reading
from .text_pdf import convert_text_pdf
from .reconstruction import convert_reconstruction
from .font_library import get_library
from .models import validate_document
from .legacy import convert_legacy, is_legacy
from .parser import parse_document as _parse_document
from .recognition import selected_backend, status as ocr_status
from .options import selected_pages
from .progress import bind_progress, progress_span, report, store as progress_store
from .usage import span, tracked

ROOT=Path(__file__).resolve().parent.parent
DATA_ROOT=Path(os.environ.get("OCR_DATA_ROOT",str(ROOT))).resolve()
INPUT=DATA_ROOT/"input"
OUTPUT=DATA_ROOT/"output"
INPUT.mkdir(parents=True,exist_ok=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
ALLOWED={".pdf",".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp",".json"}
config=json.loads((ROOT/"config.json").read_text(encoding="utf-8-sig"))
RELEASE=json.loads((ROOT.parent/"VERSION.json").read_text(encoding="utf-8"))
MAX_UPLOAD_MB=config.get("max_upload_mb",100)
if type(MAX_UPLOAD_MB) is not int or not 16 <= MAX_UPLOAD_MB <= 512:
    raise ValueError("config.json max_upload_mb must be an integer between 16 and 512")
MAX_UPLOAD=MAX_UPLOAD_MB*1024*1024
locks: dict[str, threading.RLock]={}
locks_guard=threading.Lock()
app=FastAPI(title="PDF OCR 解析器",version="1.0.0",docs_url="/api/docs")
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["localhost","127.0.0.1","[::1]","testserver"])


def parse_document(*args, **kwargs):
    kwargs.setdefault('recognition_backend', selected_backend())
    return _parse_document(*args, **kwargs)


@app.middleware("http")
async def local_request_guard(request: Request, call_next):
    origin=request.headers.get("origin")
    if request.method in {"POST","PUT","PATCH","DELETE"} and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail":"仅接受本地应用发起的修改请求"},status_code=403)
    length=request.headers.get("content-length")
    if length and int(length)>MAX_UPLOAD+1024*1024:
        return JSONResponse({"detail":f"上传总大小不能超过 {MAX_UPLOAD_MB} MB"},status_code=413)
    progress_id=request.headers.get('x-progress-id')
    if progress_id:
        try:
            progress_store.begin(progress_id)
        except ValueError as exc:
            return JSONResponse({'detail':str(exc)},status_code=422)
    feature = None
    if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}:
        feature = {'/api/convert/reading': 'reading', '/api/documents': 'editor.import',
                   '/api/convert/rebuild': 'rebuild', '/api/convert/text-pdf': 'text_pdf',
                   '/api/detection-preview': 'detection.preview'}.get(request.url.path)
        if feature is None and request.url.path.startswith('/api/documents/'):
            feature = 'editor.export' if '/export/' in request.url.path else 'editor.save'
    # Health/progress polling and static resources are intentionally excluded.
    from contextlib import nullcontext
    with bind_progress(progress_id), (span('http.request', feature=feature) if feature else nullcontext()) as usage:
        try:
            response=await call_next(request)
        except BaseException:
            if progress_id:
                progress_store.finish(progress_id,False)
            raise
        if progress_id:
            progress_store.finish(progress_id,response.status_code<400)
        if usage is not None:
            usage.metadata['httpStatus'] = response.status_code
            usage.status = 'error' if response.status_code >= 400 else 'ok'
            response.headers['X-Pdftransfer-Trace'] = usage.state['traceId']
    response.headers["X-Content-Type-Options"]="nosniff"
    response.headers["Referrer-Policy"]="no-referrer"
    return response


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail":str(exc)[:1500]},status_code=422)


@app.post('/api/progress')
def create_progress():
    return {'id':progress_store.create()}


@app.get('/api/progress/{progress_id}')
def get_progress(progress_id: str):
    snapshot=progress_store.get(progress_id)
    if snapshot is None:
        raise HTTPException(404,'进度任务不存在或已过期')
    return JSONResponse(snapshot,headers={'Cache-Control':'no-store'})


def doc_dir(document_id: str, must_exist=True) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}",document_id):
        raise HTTPException(404,"文档不存在")
    path=OUTPUT/document_id
    if must_exist and not (path/"summary"/"document.json").is_file():
        raise HTTPException(404,"文档不存在")
    return path


def lock_for(document_id):
    with locks_guard:
        return locks.setdefault(document_id,threading.RLock())


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    tmp.write_text(text,encoding="utf-8")
    os.replace(tmp,path)


@tracked('document.persist')
def persist(data: dict, expected_id: str | None=None):
    report(0,'保存当前文档','正在校验编辑内容与资源')
    doc=validate_document(data,expected_id)
    # Source polygons/glyphs stay immutable; corners always describe the edited box.
    for page in doc["pages"]:
        for element in page["elements"]:
            angle=math.radians(element.get("rotation",0))
            cosine,sine=math.cos(angle),math.sin(angle)
            cx,cy=element["x"]+element["width"]/2,element["y"]+element["height"]/2
            element["corners"]=[[round(cx+dx*cosine-dy*sine,6),round(cy+dx*sine+dy*cosine,6)]
                                for dx,dy in [(-element["width"]/2,-element["height"]/2),(element["width"]/2,-element["height"]/2),(element["width"]/2,element["height"]/2),(-element["width"]/2,element["height"]/2)]]
    out=doc_dir(doc["id"],False)
    counts=Counter(e["type"] for p in doc["pages"] for e in p["elements"])
    doc["updatedAt"]=datetime.now(timezone.utc).isoformat()
    doc["stats"]={"pageCount":len(doc["pages"]),"elementCount":sum(counts.values()),"types":dict(counts),"reviewCount":sum(bool(e.get("reviewRequired")) for p in doc["pages"] for e in p["elements"])}
    with lock_for(doc["id"]):
        for folder in ["text","vectors","images","backgrounds","pages","summary","exports"]:
            (out/folder).mkdir(parents=True,exist_ok=True)
        # Both summary formats are generated from precisely this validated snapshot.
        merged_html=scene_html(doc,out)
        atomic_write(out/"summary"/"index.html",merged_html)
        for kind,folder in [("text","text"),("vector","vectors"),("shape","vectors"),("image","images"),("region","images"),("background","backgrounds")]:
            grouped=[{"pageId":p["id"],"elements":[e for e in p["elements"] if e["type"]==kind]} for p in doc["pages"]]
            atomic_write(out/folder/f"{kind}.json",json.dumps(grouped,ensure_ascii=False,indent=2))
        atomic_write(out/"summary"/"document.json",json.dumps(doc,ensure_ascii=False,indent=2))
    report(100,'文档已保存')
    return doc


def import_scene(data: dict, document_id: str, output_dir: Path, name: str):
    if is_legacy(data):
        data = convert_legacy(data)
    data=validate_document(data)
    old_id=data["id"]
    data["id"]=document_id
    data["name"]=name
    output_dir.mkdir(parents=True,exist_ok=True)
    cache={}
    for page in data["pages"]:
        assets = [(page,["sourceSrc"])]
        assets.extend((e,["src","appearanceSrc","originalAppearanceSrc"]) for e in page["elements"])
        assets.extend((table,["referenceSrc"]) for table in page.get("tables",[]))
        for obj,keys in assets:
            for key in keys:
                src=obj.get(key)
                if not src:
                    continue
                if src not in cache:
                    raw=read_asset(src,OUTPUT/old_id,old_id)
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.width*image.height>40_000_000:
                            raise ValueError("图片尺寸超过限制")
                        image.load()
                        buffer=io.BytesIO()
                        image.convert("RGBA").save(buffer,format="PNG")
                        raw=buffer.getvalue()
                    suffix=".png"
                    digest=hashlib.sha256(raw).hexdigest()[:24]
                    target=output_dir/"images"/(digest+suffix)
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    cache[src]=f"/files/{document_id}/images/{target.name}"
                obj[key]=cache[src]
    return data


@app.get("/api/health")
def health():
    return {"status":"ok","ocr":ocr_status(),"version":"1.0.0","release":RELEASE,"storage":{"input":str(INPUT),"output":str(OUTPUT)}}


@app.get("/api/documents")
def documents():
    result=[]
    for path in OUTPUT.glob("*/summary/document.json"):
        try:
            data=json.loads(path.read_text(encoding="utf-8"))
            result.append({"id":data["id"],"name":data["name"],"pageCount":len(data["pages"]),"elementCount":sum(len(p["elements"]) for p in data["pages"]),"updatedAt":data.get("updatedAt"),"thumbnail":data["pages"][0].get("sourceSrc")})
        except (OSError,ValueError,KeyError):
            continue
    return sorted(result,key=lambda x:x.get("updatedAt") or "",reverse=True)


@app.get('/api/fonts')
def fonts():
    return {'fonts':get_library().public_fonts()}


@app.get("/api/documents/{document_id}")
def get_document(document_id: str):
    return json.loads((doc_dir(document_id)/"summary"/"document.json").read_text(encoding="utf-8"))


@app.get("/api/documents/{document_id}/preview")
def preview_document(document_id: str):
    return FileResponse(doc_dir(document_id)/"summary"/"index.html",media_type="text/html")


@tracked('document.store_uploads')
def store_uploads(files, allowed=ALLOWED):
    if not files or len(files)>30:
        raise ValueError("每次可选择 1–30 个文件")
    document_id=uuid.uuid4().hex
    input_dir=INPUT/document_id
    out=doc_dir(document_id,False)
    input_dir.mkdir(parents=True)
    total=0
    paths=[]
    for i,file in enumerate(files):
        report(2+6*i/len(files),'接收输入文件',current=i+1,total=len(files))
        original=Path((file.filename or "document").replace("\\","/")).name
        ext=Path(original).suffix.lower()
        if ext not in allowed:
            raise ValueError(f"不支持 {ext or '无扩展名'} 文件")
        safe=re.sub(r'[^\w.\-\u4e00-\u9fff]',"_",original)[:140]
        target=input_dir/f"{i+1:02d}_{safe}"
        with target.open("wb") as handle:
            while chunk:=file.file.read(1024*1024):
                total+=len(chunk)
                if total>MAX_UPLOAD:
                    raise HTTPException(413,f"上传总大小不能超过 {MAX_UPLOAD_MB} MB")
                handle.write(chunk)
        if target.stat().st_size==0:
            raise ValueError("文件为空")
        paths.append(target)
    return document_id,out,paths


@app.post('/api/detection-preview')
def detection_preview(files: list[UploadFile]=File(...),detector: str=Form('layout'),pages: str=Form('')):
    from .detection_preview import DETECTORS, run_detection
    if detector not in DETECTORS:
        raise ValueError('不支持的检测模型')
    document_id,out,paths=store_uploads(files,allowed={'.pdf'})
    scene=run_detection(paths,out,detector,pages)
    with progress_span(92,98):
        doc=import_scene(scene,document_id,out,scene['name'])
        return persist(doc)


@app.post('/api/convert/reading')
def reading(files: list[UploadFile]=File(...),format: str=Form('html'),ocr: bool=Form(True),pages: str=Form('')):
    if format not in {'html','md'}:
        raise ValueError('阅读转换仅支持 HTML 或 Markdown')
    _,out,paths=store_uploads(files,{'.pdf'})
    out.mkdir(parents=True,exist_ok=True)
    try:
        with progress_span(8,96):
            destination,warnings=convert_reading(paths,out,format=format,ocr=ocr,pages=pages,recognition_backend=selected_backend())
    except (ValueError,HTTPException):
        raise
    except Exception as exc:
        logging.exception('Reading conversion failed')
        raise HTTPException(422,'转换失败：请检查 PDF 是否损坏或加密。原件已保留。') from exc
    # Warnings are kept with the conversion rather than treating fallback as
    # accurate OCR. The selected format's summary includes the same notices.
    atomic_write(out/'conversion.json',json.dumps({'format':format,'warnings':warnings},ensure_ascii=False,indent=2))
    return FileResponse(destination,filename=f'PDF-{format}.zip',media_type='application/zip')


@app.post("/api/documents")
def upload(files: list[UploadFile]=File(...),ocr: bool=Form(True),pages: str=Form(""),mode: str=Form('editable')):
    if mode not in {'editable','text_pdf'}:
        raise ValueError('不支持的转换模式')
    if mode=='text_pdf' and not ocr:
        raise ValueError('扫描 PDF 转文本 PDF 需要开启 OCR')
    document_id,out,paths=store_uploads(files)
    name=Path(files[0].filename or "未命名文档").stem + (f" 等 {len(files)} 个文件" if len(files)>1 else "")
    try:
        embedded=None
        if mode=='editable' and len(paths)==1 and paths[0].suffix.lower()==".json":
            embedded=json.loads(paths[0].read_text(encoding="utf-8-sig"))
        elif mode=='editable' and len(paths)==1 and paths[0].suffix.lower()==".pdf":
            with fitz.open(paths[0]) as pdf:
                if not pdf.needs_pass and "scene-document.json" in pdf.embfile_names():
                    info=pdf.embfile_info("scene-document.json")
                    if info.get("size",0)>MAX_UPLOAD:
                        raise ValueError("PDF 中的编辑数据超过限制")
                    embedded=json.loads(pdf.embfile_get("scene-document.json"))
        if embedded and mode=='editable':
            if pages:
                embedded["pages"] = [embedded["pages"][i] for i in selected_pages(pages, len(embedded["pages"]))]
            doc=import_scene(embedded,document_id,out,name)
        else:
            if any(p.suffix.lower()==".json" for p in paths):
                raise ValueError("JSON 请单独导入")
            with progress_span(8,90):
                doc=parse_document(paths,out,document_id,name,ocr=ocr,pages=pages,mode=mode)
        with progress_span(90,98):
            return persist(doc)
    except (ValueError,HTTPException):
        raise
    except Exception as exc:
        logging.exception("Document parsing failed")
        raise HTTPException(422,"解析失败：文件可能损坏或包含不支持的编码。原件已保留。"+str(exc)[:200]) from exc


@app.post('/api/convert/text-pdf')
def convert_book(files: list[UploadFile]=File(...),ocr: bool=Form(True),pages: str=Form('')):
    if not ocr:
        raise ValueError('扫描 PDF 转文本 PDF 需要开启 OCR')
    document_id,out,paths=store_uploads(files,{'.pdf'})
    out.mkdir(parents=True,exist_ok=True)
    try:
        with progress_span(8,96):
            destination,report=convert_text_pdf(paths,out,document_id,pages=pages)
    except (ValueError,HTTPException):
        raise
    except Exception as exc:
        logging.exception('Text PDF book conversion failed')
        raise HTTPException(422,'转换失败：请检查 PDF 或 OCR 环境。原件已保留。') from exc
    atomic_write(out/'conversion.json',json.dumps(report,ensure_ascii=False,indent=2))
    return FileResponse(destination,filename='OCR-text.pdf',media_type='application/pdf')


@app.post('/api/convert/rebuild')
def rebuild(files: list[UploadFile]=File(...),format: str=Form('pdf'),ocr: bool=Form(True),pages: str=Form('')):
    if format not in {'pdf','html','md'}:
        raise ValueError('结构化重建仅支持 PDF、HTML 或 Markdown')
    if not ocr:
        raise ValueError('PDF / 图片重建需要开启图片内容识别')
    document_id,out,paths=store_uploads(files,{'.pdf','.png','.jpg','.jpeg','.tif','.tiff','.webp','.bmp'})
    try:
        with progress_span(8,96):
            destination,report=convert_reconstruction(paths,out,document_id,format=format,pages=pages,recognition_backend=selected_backend())
    except (ValueError,HTTPException):
        raise
    except Exception as exc:
        logging.exception('Typed reconstruction failed')
        raise HTTPException(422,'重建失败：请检查 PDF / 图片、OCR 或字体环境。原件已保留。') from exc
    return FileResponse(destination,filename='OCR-rebuilt.pdf' if format=='pdf' else f'OCR-rebuilt-{format}.zip',
                        media_type='application/pdf' if format=='pdf' else 'application/zip',
                        headers={'X-Reconstruction-Status':report['status']})


@app.post("/api/demo")
def demo():
    document_id=uuid.uuid4().hex
    folder=INPUT/document_id
    folder.mkdir(parents=True)
    path=folder/"结构示例.pdf"
    create_demo(path)
    with progress_span(0,90):
        doc=parse_document([path],doc_dir(document_id,False),document_id,"页面结构示例 · 原生与扫描",ocr=True)
    with progress_span(90,98):
        return persist(doc)


@app.put("/api/documents/{document_id}")
def save(document_id: str, data: dict):
    doc_dir(document_id)
    return persist(data,document_id)


@app.post("/api/documents/{document_id}/export/{format}")
def export(document_id: str, format: str, data: dict):
    out=doc_dir(document_id)
    if format not in {"json","html","pdf","pdf-text","zip"}:
        raise HTTPException(404,"不支持的导出格式")
    with lock_for(document_id):
        with progress_span(0,12):
            doc=persist(data,document_id)
        # Unique files avoid simultaneous downloads racing with later exports.
        extension='pdf' if format=='pdf-text' else format
        destination=out/"exports"/f"{uuid.uuid4().hex[:10]}.{extension}"
        with progress_span(12,96):
            if format=="json":
                destination.write_text(json.dumps(portable_document(doc,out),ensure_ascii=False,indent=2),encoding="utf-8")
            elif format=="html":
                destination.write_text(scene_html(doc,out),encoding="utf-8")
            elif format=='pdf-text':
                try:
                    write_text_pdf(doc,out,destination)
                except ValueError:
                    raise
                except Exception as exc:
                    logging.exception('Editable text PDF export failed')
                    raise HTTPException(422,'文本 PDF 导出失败：请检查字体或输出目录后重试。当前文档已保存。') from exc
                (out/'summary'/'document-text.pdf').write_bytes(destination.read_bytes())
            elif format=="pdf":
                write_pdf(doc,out,destination)
                (out/"summary"/"document.pdf").write_bytes(destination.read_bytes())
            else:
                pdf_path=out/"summary"/"document.pdf"
                with progress_span(0,75):
                    write_pdf(doc,out,pdf_path)
                with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
                    with progress_span(75,85):
                        archive.writestr("summary/document.json",json.dumps(portable_document(doc,out),ensure_ascii=False,indent=2))
                    assets=[path for path in out.rglob('*') if path.is_file() and path.relative_to(out).parts[0]!='exports' and path.relative_to(out).as_posix()!='summary/document.json' and not path.name.endswith('.tmp')]
                    for asset_index,path in enumerate(assets):
                        report(85+15*asset_index/max(1,len(assets)),'打包项目归档',current=asset_index+1,total=len(assets))
                        relative=path.relative_to(out)
                        if path.is_file() and relative.parts[0]!="exports" and relative.as_posix()!="summary/document.json" and not path.name.endswith(".tmp"):
                            archive.write(path,str(relative))
        report(98,'结果已生成','正在准备下载响应')
        filename=re.sub(r'[<>:"/\\|?*]',"_",doc["name"])+('-text' if format=='pdf-text' else '')+"."+extension
        return FileResponse(destination,filename=filename,media_type={"json":"application/json","html":"text/html","pdf":"application/pdf","pdf-text":"application/pdf","zip":"application/zip"}[format])


@app.get("/files/{document_id}/{asset_path:path}")
def asset(document_id: str, asset_path: str):
    root=doc_dir(document_id)
    target=(root/asset_path).resolve()
    if not target.is_relative_to(root.resolve()) or not target.is_file() or target.suffix.lower() not in {".png",".jpg",".jpeg",".webp",".bmp",".gif",".svg"}:
        raise HTTPException(404,"资源不存在")
    return FileResponse(target)


app.mount("/",StaticFiles(directory=ROOT/"frontend",html=True),name="frontend")
