"""Actual V3 HTTP dispatch with synthetic, manually specified model evidence."""
import io
import copy
import json
import zipfile

from PIL import Image
import pymupdf as fitz
import pytest

from test_app import local_app
from test_structure_parser import truth
from test_adaptive_reconstruction import evidence


def pdf_bytes(rgb):
    image = io.BytesIO()
    Image.fromarray(rgb).save(image, format='PNG')
    with fitz.open() as pdf:
        page = pdf.new_page(width=rgb.shape[1]/2,height=rgb.shape[0]/2)
        page.insert_image(page.rect,stream=image.getvalue())
        return pdf.tobytes()


def use_v3(monkeypatch):
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION','v3')
    def forbidden(*args,**kwargs):
        raise AssertionError('V3 request must not invoke RapidOCR legacy recognition')
    monkeypatch.setattr('backend.parser._engine',forbidden)


def test_new_canvas_upload_calls_v3_and_owns_picture_transcript(local_app,monkeypatch):
    use_v3(monkeypatch)
    client,inputs,outputs = local_app
    rgb,raw = truth()
    seen=[]
    def recognize(pixels):
        seen.append(pixels.shape)
        return raw
    monkeypatch.setattr('backend.structure_parser.recognize_page',recognize)
    original=pdf_bytes(rgb)
    response=client.post('/api/documents',files=[('files',('truth.pdf',original,'application/pdf'))],data={'ocr':'true'})
    assert response.status_code==200,response.text
    scene=response.json()
    assert seen==[(240,400,3)]
    assert scene['parser']['name']=='pp-structure-v3'
    page=scene['pages'][0]
    assert page['recognitionBackend']=='v3'
    assert len(page['tables'])==1
    assert any(e.get('tableBorder') for e in page['elements'])
    assert not any(e['type']=='text' and e['text']=='Map Label' for e in page['elements'])
    assert next(inputs.glob('*/01_truth.pdf')).read_bytes()==original
    assert list(outputs.glob('*/summary/document.json'))
    saved=client.put(f'/api/documents/{scene["id"]}',json=scene)
    assert saved.status_code==200,saved.text
    assert saved.json()['pages'][0]['tables'][0]['cells'][0]['colspan']==3


@pytest.mark.parametrize('format',['html','md'])
def test_fourth_api_v3_merged_table_and_single_requested_format(local_app,monkeypatch,format):
    use_v3(monkeypatch)
    client,_,_=local_app
    rgb,raw,_=evidence()
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',lambda pixels:copy.deepcopy(raw))
    response=client.post('/api/convert/rebuild',files=[('files',('truth.pdf',pdf_bytes(rgb),'application/pdf'))],
                         data={'format':format,'ocr':'true'})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        project=json.loads(archive.read('project.json'))
        body=archive.read(f'summary/summary.{format}').decode()
        names=archive.namelist()
        assert not any(name.endswith('.md' if format=='html' else '.html') for name in names)
        assert 'colspan="2"' in body
        assert 'Values' in body and 'Alpha' in body and '42' in body
        tables=[b for b in project['blocks'] if b['kind']=='table']
        assert len(tables)==1
        assert project['schema']=='feature4-rebuild/1'
        assert tables[0]['content']['cells'][0]['colspan']==2
        assert tables[0]['source']['reference_asset'] in names
        assert {p['block_id'] for p in project['layout_result']['placements']}=={b['id'] for b in project['blocks']}


@pytest.mark.parametrize('route',['/api/documents','/api/convert/rebuild'])
def test_v3_service_failure_is_422_without_legacy_fallback(local_app,monkeypatch,route):
    use_v3(monkeypatch)
    client,inputs,_=local_app
    rgb,_=truth()
    original=pdf_bytes(rgb)
    def unavailable(pixels):
        raise ValueError('本地 V3 服务不可用：synthetic outage')
    monkeypatch.setattr('backend.structure_parser.recognize_page',unavailable)
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',unavailable)
    response=client.post(route,files=[('files',('truth.pdf',original,'application/pdf'))],data={'format':'html','ocr':'true'})
    assert response.status_code==422,response.text
    assert '本地 V3 服务不可用' in response.text
    assert next(inputs.glob('*/01_truth.pdf')).read_bytes()==original


def test_canvas_native_pdf_works_without_v3_service(local_app,monkeypatch):
    use_v3(monkeypatch)
    client,_,_=local_app
    def forbidden(pixels):
        raise AssertionError('Native page should skip model transport')
    monkeypatch.setattr('backend.structure_parser.recognize_page',forbidden)
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=200)
        page.insert_text((20,40),'Original native document',fontsize=12)
        payload=pdf.tobytes()
    response=client.post('/api/documents',files=[('files',('native.pdf',payload,'application/pdf'))],data={'ocr':'true'})
    assert response.status_code==200,response.text
    page=response.json()['pages'][0]
    assert page['recognitionBackend']=='native'
    assert any(e['type']=='text' and 'native document' in e['text'] for e in page['elements'])


def test_canvas_short_native_ocr_off_stays_editable(local_app,monkeypatch):
    use_v3(monkeypatch)
    client,_,_=local_app
    def forbidden(pixels):
        raise AssertionError('OCR off native page should never start models')
    monkeypatch.setattr('backend.structure_parser.recognize_page',forbidden)
    with fitz.open() as pdf:
        page=pdf.new_page(width=200,height=100)
        page.insert_text((20,40),'Hi',fontsize=12)
        payload=pdf.tobytes()
    response=client.post('/api/documents',files=[('files',('short.pdf',payload,'application/pdf'))],data={'ocr':'false'})
    assert response.status_code==200,response.text
    scene=response.json()
    assert scene['parser']['ocrRequested'] is False
    page=scene['pages'][0]
    assert page['recognitionBackend']=='native'
    assert [e['text'] for e in page['elements'] if e['type']=='text']==['Hi']


@pytest.mark.parametrize('kind',['pdf','png'])
def test_canvas_scan_ocr_off_preserves_one_intact_image(local_app,monkeypatch,kind):
    use_v3(monkeypatch)
    client,_,outputs=local_app
    rgb,_=truth()
    def forbidden(*args,**kwargs):
        raise AssertionError('OCR disabled must preserve source without a recognizer or old partition')
    monkeypatch.setattr('backend.structure_parser.recognize_page',forbidden)
    monkeypatch.setattr('backend.parser.partition_text',forbidden)
    if kind=='pdf':
        payload=pdf_bytes(rgb)
    else:
        image=io.BytesIO()
        Image.fromarray(rgb).save(image,format='PNG')
        payload=image.getvalue()
    response=client.post('/api/documents',files=[('files',(f'scan.{kind}',payload,f'application/{kind}'))],data={'ocr':'false'})
    assert response.status_code==200,response.text
    scene=response.json()
    page=scene['pages'][0]
    assert page['recognitionBackend']=='disabled'
    assert len(page['elements'])==1
    image=page['elements'][0]
    assert image['type']=='image' and image['contentType']=='unrecognized'
    assert image['width']==200 and image['height']==120
    assert not page['recognition'] and not page['tables']
    asset=outputs/scene['id']/image['src'].split(f'/files/{scene["id"]}/',1)[1]
    import numpy as np
    np.testing.assert_array_equal(np.asarray(Image.open(asset)),rgb)
