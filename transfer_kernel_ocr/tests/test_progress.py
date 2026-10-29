"""Real synchronous requests expose progress while worker threads are busy."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pymupdf as fitz
import pytest

from backend.progress import ProgressStore, bind_progress, progress_span, report, store
from test_app import image_document, local_app


def create(client):
    response=client.post('/api/progress')
    assert response.status_code==200,response.text
    return response.json()['id']


def test_nested_progress_is_monotonic_and_separate_between_operations():
    first,second=store.create(),store.create()
    store.begin(first)
    store.begin(second)
    with bind_progress(first):
        with progress_span(20,80,page=(2,4)):
            report(50,'匹配文字字体',current=3,total=9)
            assert '第 2 / 4 页' in store.get(first)['message']
            with bind_progress(second):
                report(10,'其他任务')
                assert store.get(second)['message']==''
            report(50,'匹配文字字体')
            assert '第 2 / 4 页' in store.get(first)['message']
        report(5,'稍后的阶段')
        with bind_progress(second):
            report(10,'其他任务')
    assert store.get(first)['percent']==50
    assert store.get(second)['percent']==10
    assert store.get(first)['status']=='running'
    store.finish(first,False)
    store.finish(second,True)
    assert store.get(first)['percent']==50
    assert store.get(second)['percent']==100


def test_progress_is_readable_while_real_pdf_export_is_running(local_app,image_document,monkeypatch):
    client,_,_=local_app
    from backend import app as app_module
    original=app_module.write_pdf
    entered,release=threading.Event(),threading.Event()

    def held_export(document,output,destination):
        report(25,'渲染 PDF 页面',current=1,total=2)
        entered.set()
        assert release.wait(10),'Test did not release the export'
        original(document,output,destination)

    monkeypatch.setattr(app_module,'write_pdf',held_export)
    key,other=create(client),create(client)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(client.post,f"/api/documents/{image_document['id']}/export/pdf",
                           json=image_document,headers={'X-Progress-ID':key})
        try:
            assert entered.wait(5)
            response=client.get(f'/api/progress/{key}')
            assert response.headers['cache-control']=='no-store'
            current=response.json()
            assert current['status']=='running'
            assert 0<current['percent']<100
            assert current['stage']=='渲染 PDF 页面'
            assert current['current']==1 and current['total']==2
            assert client.get(f'/api/progress/{other}').json()['status']=='waiting'
        finally:
            release.set()
        result=future.result(timeout=10)
    assert result.status_code==200,result.text
    completed=client.get(f'/api/progress/{key}').json()
    assert completed['status']=='completed' and completed['percent']==100
    assert client.post(f"/api/documents/{image_document['id']}/export/pdf",json=image_document,
                       headers={'X-Progress-ID':key}).status_code==422


def test_reading_conversion_reports_real_page_callbacks(local_app,monkeypatch):
    client,_,_=local_app
    from backend import reading
    updates=[]
    original=reading.report

    def capture(*args,**kwargs):
        updates.append((args,kwargs))
        original(*args,**kwargs)

    monkeypatch.setattr(reading,'report',capture)
    with fitz.open() as source:
        for number in range(2):
            source.new_page().insert_text((30,50),f'PAGE {number+1}')
        payload=source.tobytes()
    key=create(client)
    response=client.post('/api/convert/reading',files=[('files',('pages.pdf',payload,'application/pdf'))],
                         data={'ocr':'false','format':'html'},headers={'X-Progress-ID':key})
    assert response.status_code==200,response.text
    assert response.content.startswith(b'PK')
    assert any(args[1]=='转换阅读文档' and kwargs.get('total')==2 for args,kwargs in updates)
    assert any(args[1]=='打包阅读结果' for args,kwargs in updates)
    final=client.get(f'/api/progress/{key}').json()
    assert final['status']=='completed' and final['percent']==100


def test_failed_conversion_does_not_claim_complete_or_consume_inputs(local_app):
    client,input_root,_=local_app
    key=create(client)
    response=client.post('/api/convert/rebuild',files=[('files',('bad.pdf',b'bad','application/pdf'))],
                         headers={'X-Progress-ID':key})
    assert response.status_code==422,response.text
    state=client.get(f'/api/progress/{key}').json()
    assert state['status']=='failed' and state['percent']<100
    before=list(input_root.iterdir())
    response=client.post('/api/documents',files=[('files',('bad.pdf',b'bad','application/pdf'))],
                         headers={'X-Progress-ID':'missing'})
    assert response.status_code==422
    assert list(input_root.iterdir())==before
    assert client.get('/api/progress/missing').status_code==404


def test_progress_memory_is_bounded_and_finished_records_can_be_replaced():
    bounded=ProgressStore(limit=1)
    first=bounded.create()
    with pytest.raises(ValueError):
        bounded.create()
    bounded.begin(first)
    with pytest.raises(ValueError):
        bounded.create()
    bounded.finish(first,True)
    second=bounded.create()
    assert second!=first and bounded.get(first) is None
