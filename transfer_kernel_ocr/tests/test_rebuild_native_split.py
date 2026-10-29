"""Manually specified native / raster truth for feature 4's shared split."""
import copy
import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz
import pytest

from backend.structure_parser import _native_table_result, _offset_v3_result, parse_document
from test_app import local_app


def picture(size=(240,160),color='black'):
    image=Image.new('RGB',size,'white')
    ImageDraw.Draw(image).rectangle((10,20,80,40),fill=color)
    stream=io.BytesIO()
    image.save(stream,format='PNG')
    return stream.getvalue()


def prediction(text='Image text',box=(10,20,80,40)):
    return dict(parsing_res_list=[dict(block_label='text',block_bbox=list(box),block_content=text)],
                overall_ocr_res=dict(rec_texts=[text],rec_boxes=[list(box)],rec_scores=[.99]),
                table_res_list=[],_service=dict(engine='manual-test',seconds=.1,cacheHit=False,
                    modelProfile=[dict(model='test-model',phase='predict',wallSeconds=.08,processCpuSeconds=.04)]))


def adaptive_prediction(rgb,text='Image text'):
    """Manual full-page evidence for the new feature-four contract only."""
    height,width=rgb.shape[:2]
    # Cover the drawn/native fixture ink completely. A stub that leaves ink
    # outside its box creates a genuine ambiguous residual-source overlap.
    box=[0.,0.,width*.9,height*.45]
    return dict(width=width,height=height,
                layout_det_res=dict(boxes=[dict(id='layout-0',label='text',score=.99,coordinate=box)]),
                region_det_res=dict(boxes=[dict(id='region-0',label='region',coordinate=[0,0,width,height])]),
                overall_ocr_res=dict(rec_texts=[text],rec_boxes=[box],rec_scores=[.99],block_ids=['layout-0']),
                table_res_list=[],diagnostics=[],_service=dict(cacheHit=False,modelProfile=[]))


def parse(path,tmp_path):
    return parse_document([path],tmp_path/'scene','manual-truth','Manual truth',mode='rebuild')['pages'][0]


def test_native_shared_text_styles_and_table_skip_v3(monkeypatch,tmp_path):
    path=tmp_path/'native.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=320,height=240)
        page.insert_text((20,25),'Native heading',fontname='hebo',fontsize=15,color=(1,0,0))
        page.insert_text((20,50),'Italic body',fontname='heit',fontsize=11,color=(0,0,1))
        for x in (20,150,280):
            page.draw_line((x,80),(x,200))
        for y in (80,140,200):
            page.draw_line((20,y),(280,y))
        for value,x,y in [('A1',30,110),('B1',160,110),('A2',30,170),('B2',160,170)]:
            page.insert_text((x,y),value,fontsize=11)
        pdf.save(path)
    original=path.read_bytes()
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda *_:pytest.fail('Native content must skip V3'))
    page=parse(path,tmp_path)
    assert path.read_bytes()==original
    assert page['recognitionBackend']=='native'
    assert page['modelDispatch']['modelCalls']==0
    assert page['modelDispatch']['modelProfiles']==[]
    heading=next(e for e in page['elements'] if e.get('text')=='Native heading')
    assert heading['fontSize']==15 and heading['color']=='#ff0000' and heading['fontWeight']=='700'
    body=next(e for e in page['elements'] if e.get('text')=='Italic body')
    assert body['fontStyle']=='italic' and body['color']=='#0000ff'
    assert len(page['tables'])==1
    table=page['tables'][0]
    assert table['structureStatus']=='native-extracted'
    assert table['geometrySource']=='native'
    assert table['referenceSrc']
    assert [c['htmlText'] for c in table['cells']]==['A1','B1','A2','B2']
    assert sorted(e['text'] for e in page['elements'] if e['type']=='text')==[
        'A1','A2','B1','B2','Italic body','Native heading']


def test_mixed_pdf_only_raster_crop_and_offsets(monkeypatch,tmp_path):
    path=tmp_path/'mixed.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=220)
        page.insert_text((20,30),'Native title',fontsize=12)
        page.insert_image(fitz.Rect(100,80,220,160),stream=picture())
        pdf.save(path)
    calls=[]
    def recognize(rgb):
        calls.append(rgb.shape)
        return prediction()
    monkeypatch.setattr('backend.structure_parser.recognize_page',recognize)
    page=parse(path,tmp_path)
    assert calls==[(160,240,3)]
    assert page['recognitionBackend']=='native+v3'
    texts=[e for e in page['elements'] if e['type']=='text']
    assert sorted(e['text'] for e in texts)==['Image text','Native title']
    raster=next(e for e in texts if e['text']=='Image text')
    assert [raster['x'],raster['y'],raster['width'],raster['height']]==pytest.approx([105,90,35,10])
    dispatch=page['modelDispatch']
    assert dispatch['modelCalls']==1 and dispatch['regions'][0]['bbox']==[100,80,220,160]
    assert dispatch['modelProfiles'][0]['regionIndex']==0
    assert dispatch['modelProfiles'][0]['model']=='test-model'
    assert all(r.get('ownerId') for r in page['recognition'])


def test_native_table_over_bitmap_has_one_owner_even_if_v3_repeats_table(monkeypatch,tmp_path):
    path=tmp_path/'native-table-over-image.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=320,height=240)
        page.insert_image(page.rect,stream=picture((640,480)))
        page.insert_text((20,30),'Native title',fontsize=12)
        for x in (20,150,280):
            page.draw_line((x,80),(x,200))
        for y in (80,140,200):
            page.draw_line((20,y),(280,y))
        for value,x,y in [('A1',30,110),('B1',160,110),('A2',30,170),('B2',160,170)]:
            page.insert_text((x,y),value,fontsize=11)
        pdf.save(path)
    markup='<table><tr><td>A1</td><td>B1</td></tr><tr><td>A2</td><td>B2</td></tr></table>'
    result=dict(parsing_res_list=[dict(block_label='table',block_bbox=[40,160,560,400],block_content=markup)],
                table_res_list=[dict(pred_html=markup,cell_box_list=[[40,160,300,280],[300,160,560,280],
                                                                  [40,280,300,400],[300,280,560,400]])])
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda _:copy.deepcopy(result))
    page=parse(path,tmp_path)
    assert len(page['tables'])==1
    assert page['tables'][0]['geometrySource']=='native'
    assert sorted(e['text'] for e in page['elements'] if e['type']=='text')==['A1','A2','B1','B2','Native title']
    assert all(r.get('ownerId') for r in page['recognition'])


def test_multiple_regions_have_unique_ownership_and_model_profiles(monkeypatch,tmp_path):
    path=tmp_path/'two-images.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=400,height=300)
        page.insert_text((20,30),'Native',fontsize=12)
        page.insert_image(fitz.Rect(20,80,140,160),stream=picture())
        page.insert_image(fitz.Rect(240,180,360,260),stream=picture(color='blue'))
        pdf.save(path)
    calls=[]
    def recognize(rgb):
        calls.append(rgb.shape)
        return prediction(f'Raster {len(calls)}')
    monkeypatch.setattr('backend.structure_parser.recognize_page',recognize)
    page=parse(path,tmp_path)
    assert len(calls)==2 and page['modelDispatch']['modelCalls']==2
    assert [p['regionIndex'] for p in page['modelDispatch']['modelProfiles']]==[0,1]
    text=[e for e in page['elements'] if e['type']=='text']
    assert sorted(e['text'] for e in text)==['Native','Raster 1','Raster 2']
    assert len({e['id'] for e in page['elements']})==len(page['elements'])
    assert all(r.get('ownerId') for r in page['recognition'])


def test_native_text_over_image_not_duplicated_as_ocr(monkeypatch,tmp_path):
    path=tmp_path/'overlay.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=220)
        page.insert_image(fitz.Rect(100,80,220,160),stream=picture())
        page.insert_text((110,110),'NATIVE',fontsize=12)
        pdf.save(path)
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda _:prediction('NATIVE',(18,36,112,68)))
    page=parse(path,tmp_path)
    texts=[e for e in page['elements'] if e['type']=='text']
    assert [e['text'] for e in texts]==['NATIVE']
    assert texts[0]['source']=='native'
    assert any(r.get('supersededBy') for r in page['recognition'])
    assert all(r.get('ownerId') for r in page['recognition'])


def test_overlapping_images_use_one_merged_crop(monkeypatch,tmp_path):
    path=tmp_path/'overlapping.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=220)
        page.insert_text((20,30),'Native title',fontsize=12)
        page.insert_image(fitz.Rect(100,80,220,160),stream=picture())
        page.insert_image(fitz.Rect(180,100,240,180),stream=picture((120,160),color='red'))
        pdf.save(path)
    calls=[]
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda rgb:(calls.append(rgb.shape) or prediction()))
    page=parse(path,tmp_path)
    assert calls==[(200,280,3)]
    assert page['modelDispatch']['modelCalls']==1
    assert page['modelDispatch']['regions'][0]['bbox']==[100,80,240,180]


def test_image_table_cell_coordinates_map_to_full_page(monkeypatch,tmp_path):
    path=tmp_path/'image-table.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=220)
        page.insert_text((20,30),'Native title',fontsize=12)
        page.insert_image(fitz.Rect(100,80,220,160),stream=picture())
        pdf.save(path)
    def recognize(_):
        markup='<table><tr><td>Top cell</td></tr><tr><td>Bottom cell</td></tr></table>'
        return dict(parsing_res_list=[dict(block_label='table',block_bbox=[10,20,210,120],block_content=markup)],
                    table_res_list=[dict(pred_html=markup,cell_box_list=[[10,20,210,70],[10,70,210,120]])])
    monkeypatch.setattr('backend.structure_parser.recognize_page',recognize)
    page=parse(path,tmp_path)
    table=page['tables'][0]
    assert table['bbox']==[105,90,205,140]
    assert table['rows']==[90,115,140] and table['columns']==[105,205]
    assert [c['bbox'] for c in table['cells']]==[[105,90,205,115],[105,115,205,140]]
    assert sorted(e['text'] for e in page['elements'] if e['type']=='text')==['Bottom cell','Native title','Top cell']


def test_v3_reconstruction_uses_block_evidence_for_every_source_page(monkeypatch,tmp_path):
    from backend.structure_parser import _reading_engine
    from backend.reconstruction import convert_reconstruction
    path=tmp_path/'two-native-pages.pdf'
    with fitz.open() as pdf:
        for text in ('Page one','Page two'):
            pdf.new_page(width=300,height=150).insert_text((20,40),text,fontsize=12)
        pdf.save(path)
    engine=_reading_engine()
    calls=[]
    def recognize(rgb):
        calls.append(rgb.shape)
        return adaptive_prediction(rgb,f'Page {len(calls)}')
    monkeypatch.setattr(engine,'_TableQueue',lambda *_:pytest.fail('V3 block rebuild must not use the old native scene queue'))
    monkeypatch.setattr('backend.reconstruction.parse_document',lambda *_:pytest.fail('V3 must not use the fixed scene parser'))
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',recognize)
    destination,report=convert_reconstruction([path],tmp_path/'result','native-book',format='md',recognition_backend='v3')
    assert calls==[(300,600,3),(300,600,3)]
    assert report['sourcePageCount']==2 and destination.is_file()
    assert report['recognitionBackend']=='v3'
    with zipfile.ZipFile(destination) as archive:
        project=json.loads(archive.read('project.json'))
        assert [p['physical_page_index'] for p in project['source_pages']]==[0,1]
        texts=[b for b in project['blocks'] if b['kind']=='text']
        assert [b['content']['text'] for b in texts]==['Page 1','Page 2']
        assert [b['source']['page_id'] for b in texts]==['source-1','source-2']
        assert len(set(b['id'] for b in project['blocks']))==len(project['blocks'])


def test_scan_without_native_text_uses_whole_page(monkeypatch,tmp_path):
    path=tmp_path/'scan.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=200,height=120)
        page.insert_image(page.rect,stream=picture((400,240)))
        pdf.save(path)
    calls=[]
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda rgb:(calls.append(rgb.shape) or prediction()))
    page=parse(path,tmp_path)
    assert calls==[(240,400,3)]
    assert page['recognitionBackend']=='v3'
    assert page['modelDispatch']['strategy']=='full-page-no-native-text'
    assert page['modelDispatch']['modelCalls']==1


def test_crop_offsets_nested_polygons_and_table_cells_without_mutating_input():
    raw=prediction()
    raw['overall_ocr_res']['rec_polys']=[[[1,2],[3,2],[3,4],[1,4]]]
    raw['table_res_list']=[dict(cell_box_list=[[1,2,3,4],[1,2,3,2,3,4,1,4]],
                               table_ocr_pred=dict(rec_boxes=[[1,2,3,4]]))]
    before=copy.deepcopy(raw)
    mapped=_offset_v3_result(raw,100,50)
    assert raw==before
    assert mapped['parsing_res_list'][0]['block_bbox']==[110,70,180,90]
    assert mapped['overall_ocr_res']['rec_polys'][0]==[[101,52],[103,52],[103,54],[101,54]]
    assert mapped['table_res_list'][0]['cell_box_list'][0]==[101,52,103,54]
    assert mapped['table_res_list'][0]['cell_box_list'][1]==[101,52,103,52,103,54,101,54]
    assert mapped['table_res_list'][0]['table_ocr_pred']['rec_boxes'][0]==[101,52,103,54]
    assert mapped['_service']==raw['_service']


def test_incomplete_native_cell_geometry_is_not_fabricated(tmp_path):
    from types import SimpleNamespace
    table=SimpleNamespace(bbox=[10,20,110,100],rows=[['A','B'],['C','D']],meta=dict(
        row_heights=[40,40],column_widths=[50,50],grid_cells=[
            dict(row=0,col=0,rowspan=1,colspan=1,bbox=[10,20,60,60],text='A'),
            dict(row=1,col=0,rowspan=1,colspan=1,bbox=[10,60,60,100],text='C')]))
    with fitz.open() as pdf:
        source=pdf.new_page(width=200,height=150)
        assert _native_table_result(table,source,.5,.5) is None
        table.meta['grid_cells'].extend([
            dict(row=0,col=1,rowspan=1,colspan=1,bbox=[60,20,110,60],text='B'),
            dict(row=1,col=1,rowspan=1,colspan=1,bbox=[60,60,110,100],text='D')])
        result=_native_table_result(table,source,.5,.5)
        assert result and result['_source']=='native'
        table.meta['grid_cells'][0]['bbox']=[190,20,200,60]
        assert _native_table_result(table,source,.5,.5) is None


def test_rotated_native_page_keeps_visible_orientation(monkeypatch,tmp_path):
    path=tmp_path/'rotated.pdf'
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=200)
        page.insert_text((20,40),'Native rotated',fontsize=12)
        page.set_rotation(90)
        pdf.save(path)
    monkeypatch.setattr('backend.structure_parser.recognize_page',lambda *_:pytest.fail('Native must skip V3'))
    page=parse(path,tmp_path)
    assert (page['width'],page['height'])==(200,300)
    assert page['sourceRotation']==90
    native=next(e for e in page['elements'] if e.get('text')=='Native rotated')
    assert native['rotation']==90
    assert 0 <= native['x'] <= page['width']
    assert all(r.get('ownerId') for r in page['recognition'])


@pytest.mark.parametrize('kind',['png','tiff'])
def test_feature_four_image_api_and_frame_selection(local_app,monkeypatch,kind):
    client,inputs,_=local_app
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION','v3')
    calls=[]
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',
                        lambda rgb:(calls.append(rgb.shape) or adaptive_prediction(rgb)))
    stream=io.BytesIO()
    first=Image.open(io.BytesIO(picture()))
    if kind=='tiff':
        second=Image.new('RGB',(360,240),'white')
        first.save(stream,format='TIFF',save_all=True,append_images=[second])
    else:
        first.save(stream,format='PNG')
    payload=stream.getvalue()
    response=client.post('/api/convert/rebuild',files=[('files',(f'manual.{kind}',payload,f'image/{kind}'))],
                         data=dict(format='md',pages='2' if kind=='tiff' else '1'))
    assert response.status_code==200,response.text
    assert calls==[(240,360,3)] if kind=='tiff' else calls==[(160,240,3)]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        report=json.loads(archive.read('conversion-report.json'))
        project=json.loads(archive.read('project.json'))
        assert report['sourcePageCount']==1
        assert project['source_pages'][0]['physical_page_index']==(1 if kind=='tiff' else 0)
        assert project['schema']=='feature4-rebuild/1'
        assert project['blocks'][0]['source']['layout_detection_id']=='layout-0'
        assert 'Image text' in archive.read('summary/summary.md').decode()
    assert next(inputs.glob(f'*/01_manual.{kind}')).read_bytes()==payload


def test_feature_four_small_image_uses_source_frame_without_old_margin_rejection(local_app,monkeypatch):
    client,_,_=local_app
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION','v3')
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',adaptive_prediction)
    stream=io.BytesIO()
    Image.new('RGB',(120,100),'white').save(stream,format='PNG')
    response=client.post('/api/convert/rebuild',files=[('files',('tiny.png',stream.getvalue(),'image/png'))],
                         data=dict(format='md'))
    assert response.status_code==200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        project=json.loads(archive.read('project.json'))
        assert project['layout_result']['target_structure']=='source-preserving-regions'
        assert len(project['layout_result']['pages'])==1
        assert project['layout_result']['pages'][0]['width']==60


def test_feature_four_excessive_growth_reports_frame_instead_of_paging_forever(local_app,monkeypatch):
    client,_,_=local_app
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION','v3')
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',
                        lambda rgb:adaptive_prediction(rgb,'Much longer content '*100))
    stream=io.BytesIO()
    Image.new('RGB',(120,100),'white').save(stream,format='PNG')
    response=client.post('/api/convert/rebuild',files=[('files',('tiny.png',stream.getvalue(),'image/png'))],
                         data=dict(format='md'))
    assert response.status_code==422
    assert '可读下限' in response.text


def test_feature_four_ui_accepts_images_without_changing_canvas_mode():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/'frontend/app.js').read_text(encoding='utf-8')
    feature=source.split('text_pdf:{',1)[1].split('\n',1)[0]
    assert '.png' in feature and '.tiff' in feature and '自适应重排与分页' in feature
