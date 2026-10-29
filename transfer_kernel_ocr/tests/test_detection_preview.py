"""Detection import boundary: real scene migration, no production OCR calls."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf as fitz
import pytest

from test_app import local_app, png_bytes
from backend import detection_preview as runner


def sample_pdf():
    with fitz.open() as pdf:
        pdf.new_page(width=200,height=300)
        return pdf.tobytes()


def test_unknown_detector_rejected_before_storing_upload(local_app):
    client,inputs,_=local_app
    response=client.post('/api/detection-preview',files={'files':('one.pdf',sample_pdf(),'application/pdf')},
                         data={'detector':'anything'})
    assert response.status_code==422
    assert not list(inputs.iterdir())


def test_selected_model_process_and_scene_import(local_app,monkeypatch):
    client,_,outputs=local_app
    calls=[]
    def fake_process(command,**kwargs):
        calls.append(command)
        destination=Path(command[command.index('--output')+1]);destination.mkdir(parents=True)
        import base64
        src='data:image/png;base64,'+base64.b64encode(png_bytes()).decode()
        scene=dict(schemaVersion=1,id='layout-preview',name='类别框测试',pages=[dict(id='page-1',index=0,
            width=200,height=300,sourceSrc=src,elements=[
                dict(id='bg',type='background',x=0,y=0,width=200,height=300,src=src,locked=True),
                dict(id='box',type='shape',x=20,y=30,width=80,height=60,fill='none',stroke='#ff0000',
                     detectorOutput=dict(label='table',cls_id=8,score=.9)),
                dict(id='label',type='text',x=20,y=20,width=100,height=10,text='表格 / table')])])
        (destination/'scene-document.json').write_text(json.dumps(scene),encoding='utf-8')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',fake_process)
    response=client.post('/api/detection-preview',files={'files':('one.pdf',sample_pdf(),'application/pdf')},
                         data={'detector':'layout','pages':'1'})
    assert response.status_code==200,response.text
    doc=response.json();elements=doc['pages'][0]['elements']
    assert elements[1]['detectorOutput']['label']=='table'
    assert elements[2]['text']=='表格 / table'
    assert elements[0]['src'].startswith(f"/files/{doc['id']}/")
    assert (outputs/doc['id']/'summary/document.json').is_file()
    assert calls[0][calls[0].index('--detector')+1]=='layout'
    assert 'native_worker.py' not in ' '.join(calls[0])


@pytest.mark.parametrize('pages',['2','0','1-0'])
def test_invalid_pages_do_not_start_model(local_app,monkeypatch,pages):
    client,_,_=local_app
    def forbidden(*a,**k):raise AssertionError('Must validate pages before spawning')
    monkeypatch.setattr(runner.subprocess,'run',forbidden)
    response=client.post('/api/detection-preview',files={'files':('one.pdf',sample_pdf(),'application/pdf')},
                         data={'detector':'region','pages':pages})
    assert response.status_code==422


def test_timeout_releases_detector_slot(tmp_path,monkeypatch):
    source=tmp_path/'one.pdf';source.write_bytes(sample_pdf())
    def timeout(*a,**k):raise runner.subprocess.TimeoutExpired('detector',1800)
    monkeypatch.setattr(runner.subprocess,'run',timeout)
    with pytest.raises(ValueError,match='30分钟'):
        runner.run_detection([source],tmp_path/'out','det')
    assert runner._RUNNING.acquire(blocking=False)
    runner._RUNNING.release()
