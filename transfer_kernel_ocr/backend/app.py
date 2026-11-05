"""Local-only HTTP interface for the three document conversions."""
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .adaptive_reconstruction import convert_adaptive
from .conversion_jobs import ConversionJobs
from .progress import bind_progress, progress_span, report, store as progress_store
from .reading import convert_reading
from .recognition import status as ocr_status

ROOT=Path(__file__).resolve().parent.parent
DATA_ROOT=Path(os.environ.get('OCR_DATA_ROOT',str(ROOT))).resolve()
INPUT, OUTPUT=DATA_ROOT/'input', DATA_ROOT/'output'
INPUT.mkdir(parents=True,exist_ok=True)
OUTPUT.mkdir(parents=True,exist_ok=True)
config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
RELEASE=json.loads((ROOT.parent/'VERSION.json').read_text(encoding='utf8'))
MAX_UPLOAD_MB=config.get('max_upload_mb',256)
if type(MAX_UPLOAD_MB) is not int or not 16<=MAX_UPLOAD_MB<=512:
    raise ValueError('max_upload_mb must be an integer between 16 and 512')
MAX_UPLOAD=MAX_UPLOAD_MB*1024*1024
ALLOWED={'.pdf','.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}

def convert_pdf(paths, output_dir, document_id, *, format='pdf', pages='', on_page=None):
    if format!='pdf':
        raise ValueError('扫描转换仅支持可编辑 PDF')
    return convert_adaptive(paths,output_dir,document_id,format='pdf',pages=pages,on_page=on_page)

_jobs=None
_jobs_lock=threading.Lock()
def job_manager():
    global _jobs
    with _jobs_lock:
        if _jobs is None:
            _jobs=ConversionJobs(OUTPUT,convert_pdf)
    return _jobs

@asynccontextmanager
async def lifespan(app):
    yield
    if _jobs is not None:
        _jobs.close()

app=FastAPI(title='pdftransfer',version=RELEASE['release'],docs_url='/api/docs',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['localhost','127.0.0.1','[::1]','testserver'])

@app.middleware('http')
async def local_request_guard(request: Request, call_next):
    origin=request.headers.get('origin')
    if request.method in {'POST','PUT','PATCH','DELETE'} and origin:
        from urllib.parse import urlparse
        if urlparse(origin).netloc!=request.headers.get('host'):
            return JSONResponse({'detail':'仅接受本地应用发起的修改请求'},status_code=403)
    length=request.headers.get('content-length')
    try:
        if length and int(length)>MAX_UPLOAD+1024*1024:
            return JSONResponse({'detail':f'上传总大小不能超过 {MAX_UPLOAD_MB} MB'},status_code=413)
    except ValueError:
        return JSONResponse({'detail':'无效的请求长度'},status_code=400)
    progress_id=request.headers.get('x-progress-id')
    if progress_id:
        try:
            progress_store.begin(progress_id)
        except ValueError as error:
            return JSONResponse({'detail':str(error)},status_code=422)
    try:
        with bind_progress(progress_id):
            response=await call_next(request)
    except BaseException:
        if progress_id: progress_store.finish(progress_id,False)
        raise
    if progress_id: progress_store.finish(progress_id,response.status_code<400)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='no-referrer'
    return response

@app.exception_handler(ValueError)
async def value_error(request, error):
    return JSONResponse({'detail':str(error)[:1500]},status_code=422)

@app.get('/api/health')
def health():
    return dict(status='ok',release=RELEASE,ocr=ocr_status())

@app.post('/api/progress')
def create_progress():
    return dict(id=progress_store.create())

@app.get('/api/progress/{progress_id}')
def get_progress(progress_id: str):
    snapshot=progress_store.get(progress_id)
    if snapshot is None: raise HTTPException(404,'进度任务不存在或已过期')
    return JSONResponse(snapshot,headers={'Cache-Control':'no-store'})

def validate_pages(pages):
    if pages and not re.fullmatch(r'[1-9]\d*(?:-[1-9]\d*)?(?:,[1-9]\d*(?:-[1-9]\d*)?)*',pages):
        raise ValueError('物理页码请填写 1-3,5,8-10 这样的格式')
    for value in pages.split(',') if pages else []:
        parts=value.split('-')
        if len(parts)==2 and int(parts[1])<int(parts[0]):
            raise ValueError('结束页不能小于起始页')

def store_uploads(files, allowed=ALLOWED):
    if not files or len(files)>30: raise ValueError('每次可选择 1–30 个文件')
    identifier=uuid.uuid4().hex
    incoming=INPUT/identifier
    incoming.mkdir()
    output=OUTPUT/identifier
    output.mkdir()
    total=0
    paths=[]
    try:
        for index,file in enumerate(files):
            original=Path((file.filename or 'document').replace('\\','/')).name
            suffix=Path(original).suffix.lower()
            if suffix not in allowed: raise ValueError('文件格式不支持当前转换')
            name=re.sub(r'[^\w.\-]','_',Path(original).stem)[:100] or 'document'
            target=incoming/f'{index+1:02d}_{name}{suffix}'
            with target.open('wb') as handle:
                while chunk:=file.file.read(1024*1024):
                    total+=len(chunk)
                    if total>MAX_UPLOAD: raise HTTPException(413,f'上传总大小不能超过 {MAX_UPLOAD_MB} MB')
                    handle.write(chunk)
            if not target.stat().st_size: raise ValueError('文件为空')
            paths.append(target)
            report(2+6*(index+1)/len(files),'接收输入文件',current=index+1,total=len(files))
    except BaseException:
        shutil.rmtree(incoming)
        shutil.rmtree(output)
        raise
    return identifier,output,paths

@app.post('/api/convert/reading')
def reading(files: list[UploadFile]=File(...),format: str=Form('html'),pages: str=Form('')):
    if format not in {'html','md'}: raise ValueError('仅支持 HTML 或 Markdown')
    validate_pages(pages)
    identifier,out,paths=store_uploads(files,{'.pdf'})
    with progress_span(8,98):
        destination,_=convert_reading(paths,out,format=format,pages=pages)
    return FileResponse(destination,filename=f'PDF-{format}.zip',media_type='application/zip')

@app.post('/api/convert/rebuild')
def rebuild(files: list[UploadFile]=File(...),format: str=Form('pdf'),pages: str=Form('')):
    if format!='pdf': raise ValueError('扫描转换仅支持可编辑 PDF')
    validate_pages(pages)
    identifier,out,paths=store_uploads(files)
    with progress_span(8,98):
        destination,report=convert_pdf(paths,out,identifier,pages=pages)
    return FileResponse(destination,filename='OCR-editable.pdf',media_type='application/pdf',
                        headers={'X-Reconstruction-Status':report['status']})

@app.post('/api/conversion-jobs',status_code=202)
def enqueue_rebuild(files: list[UploadFile]=File(...),format: str=Form('pdf'),pages: str=Form('')):
    if format!='pdf': raise ValueError('扫描转换仅支持可编辑 PDF')
    validate_pages(pages)
    identifier,out,paths=store_uploads(files)
    return job_manager().submit(identifier,[p.name for p in paths],dict(format='pdf',pages=pages),paths)

@app.get('/api/conversion-jobs')
def list_jobs():
    return job_manager().list()

@app.get('/api/conversion-jobs/{identifier}/files/{name:path}')
def conversion_job_file(identifier: str,name: str):
    if not re.fullmatch(r'[a-f0-9]{32}',identifier): raise HTTPException(404,'任务不存在')
    item=next((i for i in job_manager().list() if i['id']==identifier),None)
    if item is None: raise HTTPException(404,'任务不存在')
    root=(OUTPUT/identifier).resolve()
    target=(root/name).resolve()
    if not target.is_relative_to(root) or not target.is_file(): raise HTTPException(404,'文件不存在')
    result=item['status']=='completed' and name==item.get('result')
    preview=name in {p['preview'] for p in item['pages']}
    asset=target.is_relative_to(root/'assets') and target.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}
    if not (result or preview or asset): raise HTTPException(404,'文件尚未发布')
    headers={'Cache-Control':'no-store'}
    if target.suffix=='.html':
        headers['Content-Security-Policy']="sandbox allow-same-origin; default-src 'none'; img-src 'self'; style-src 'unsafe-inline'"
    return FileResponse(target,filename='OCR-editable.pdf' if result else None,headers=headers)

app.mount('/',StaticFiles(directory=ROOT/'frontend',html=True),name='frontend')
