"""Exercise the released HTTP surface, real downloads, and queue isolation."""
import copy
import importlib
import io
import json
import time
import zipfile

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from test_adaptive_reconstruction import evidence

@pytest.fixture
def local_app(tmp_path,monkeypatch):
    module=importlib.import_module('backend.app')
    incoming,out=tmp_path/'input',tmp_path/'output'
    incoming.mkdir();out.mkdir()
    monkeypatch.setattr(module,'INPUT',incoming)
    monkeypatch.setattr(module,'OUTPUT',out)
    monkeypatch.setattr(module,'_jobs',None)
    monkeypatch.setattr(module,'ocr_status',lambda:dict(available=True,local=True))
    with TestClient(module.app) as client:
        yield client,module,out

def native_pdf():
    with fitz.open() as pdf:
        page=pdf.new_page()
        page.insert_text((40,60),'Public native PDF example',fontsize=12)
        return pdf.tobytes()

@pytest.mark.parametrize('format',['html','md'])
def test_native_reading_downloads_selected_format_without_ocr(local_app,monkeypatch,format):
    client,module,_=local_app
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',
                        lambda *_:pytest.fail('Native extraction must not run scan OCR'))
    response=client.post('/api/convert/reading',files={'files':('native.pdf',native_pdf(),'application/pdf')},
                         data={'format':format,'ocr':'true'})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        texts=[archive.read(name).decode('utf8') for name in archive.namelist() if name.endswith('.'+format)]
        assert any('Public native PDF example' in text for text in texts)
        assert not any(name.endswith('.md' if format=='html' else '.html') for name in archive.namelist())

def test_scan_queue_publishes_every_page_and_downloadable_editable_pdf(local_app,monkeypatch):
    client,module,out=local_app
    rgb,raw,_=evidence()
    image=io.BytesIO()
    Image.fromarray(rgb).save(image,format='TIFF',save_all=True,append_images=[Image.fromarray(rgb)])
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',lambda *args,**kw:copy.deepcopy(raw))
    response=client.post('/api/conversion-jobs',files={'files':('scan.tiff',image.getvalue(),'image/tiff')})
    assert response.status_code==202,response.text
    identifier=response.json()['id']
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        item=next(item for item in client.get('/api/conversion-jobs').json() if item['id']==identifier)
        if item['status'] not in {'queued','running'}: break
        time.sleep(.02)
    assert item['status']=='completed',item.get('error')
    assert len(item['pages'])==2
    base=f'/api/conversion-jobs/{identifier}/files/'
    preview=client.get(base+item['pages'][0]['preview'])
    assert preview.status_code==200 and 'sandbox' in preview.headers['content-security-policy']
    result=client.get(base+item['result'])
    assert result.status_code==200
    with fitz.open(stream=result.content,filetype='pdf') as pdf:
        assert len(pdf)==2 and all('Adaptive title' in page.get_text() for page in pdf)
    assert client.get(base+'queue-job.json').status_code==404
    assert client.get(base+'../not-published.json').status_code==404

@pytest.mark.parametrize('path',['/api/documents','/api/demo','/api/detection-preview',
                                  '/api/text-box-debug','/api/convert/text-pdf'])
def test_other_feature_endpoints_are_absent(local_app,path):
    client,_,_=local_app
    assert client.post(path).status_code in {404,405}

@pytest.mark.parametrize('path',['/api/conversion-jobs','/api/convert/rebuild'])
def test_scan_output_accepts_only_pdf(local_app,path):
    client,_,_=local_app
    response=client.post(path,files={'files':('scan.pdf',native_pdf(),'application/pdf')},data={'format':'html'})
    assert response.status_code==422

def test_local_origin_and_upload_validation(local_app):
    client,_,out=local_app
    assert client.post('/api/progress',headers={'origin':'https://example.com'}).status_code==403
    response=client.post('/api/convert/reading',files={'files':('data.json',b'{}','application/json')})
    assert response.status_code==422 and list(out.iterdir())==[]
    response=client.post('/api/convert/reading',files={'files':('empty.pdf',b'','application/pdf')})
    assert response.status_code==422 and list(out.iterdir())==[]
    response=client.post('/api/conversion-jobs',files={'files':('scan.pdf',native_pdf(),'application/pdf')},data={'pages':'3-1'})
    assert response.status_code==422 and list(out.iterdir())==[]

def test_frontend_and_api_publish_only_three_conversions(local_app):
    client,_,_=local_app
    page=client.get('/').text
    assert page.count('name="mode"')==3
    assert all(label in page for label in ('可编辑 PDF → HTML','可编辑 PDF → Markdown','扫描 PDF → 可编辑 PDF'))
    paths=client.get('/openapi.json').json()['paths']
    assert not any('debug' in path or 'documents' in path or 'detection' in path for path in paths)
