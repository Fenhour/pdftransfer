"""Manually constructed geometry/color truth, plus actual measured PDF checks."""
import copy
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz
import pytest

from backend.adaptive_reconstruction import blocks_from_page, derive_relations
from backend.source_geometry import infer_alignment
from backend.adaptive_layout import layout_document, measure_block
from backend.adaptive_export import export_project, _validate


def fixture(panel=True,texture=False,two=False,missing=False):
    image=Image.new('RGB',(600,800),'white');draw=ImageDraw.Draw(image)
    rect=[180,40,420,100 if not two else 180]
    if panel:draw.rectangle(rect,fill='black' if not two else 'white',outline='black',width=2)
    coords=[[220,55,380,78]] if not two else [[205,60,390,85],[205,110,390,135]]
    for b in coords:
        draw.text((b[0]+3,b[1]+4),'Heading' if not two else 'Body content',fill='white' if not two else 'black')
    if texture:
        rng=np.random.default_rng(44);arr=np.asarray(image).copy()
        arr[80:94,185:415]=rng.integers(0,255,(14,230,3),dtype=np.uint8)
        image=Image.fromarray(arr)
    boxes=[dict(id=f'layout-{i}',label='text' if two else 'image',coordinate=b if two else rect,score=.99)
           for i,b in enumerate(coords)]
    raw=dict(width=600,height=800,layout_det_res=dict(boxes=boxes),
        overall_ocr_res=dict(rec_texts=['Body content' if two else 'Heading']*len(coords),rec_boxes=coords,
             rec_scores=[.99]*len(coords),block_ids=[f'layout-{i}' for i in range(len(coords))],
             line_ids=[f'line-{i}' for i in range(len(coords))]),diagnostics=[],table_res_list=[])
    assignments=[dict(line_id=f'line-{i}',block_id=f'layout-{i}',state='recognized',box=b,
        polygon=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]) for i,b in enumerate(coords)]
    if missing:assignments[0]['state']='recognition-rejected'
    raw['ocr_pipeline']=dict(line_assignments=assignments)
    page=dict(id='source-1',width_px=600,height_px=800,width_pt=300.,height_pt=400.)
    return np.asarray(image),raw,page


def project(blocks,page):
    p=dict(blocks=blocks,source_pages=[page])
    p['relations']=derive_relations(p)
    layout_document(p)
    return p


def test_black_badge_becomes_one_centered_white_text_container(tmp_path):
    rgb,raw,page=fixture()
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==1 and blocks[0]['kind']=='composite'
    parent=blocks[0];child=parent['content']['children'][0]
    assert child['id']=='source-1-layout-0'
    assert child['source']['label']=='image'
    assert int(child['style']['color'][1:3],16)>220
    assert child['style']['align']=='center'
    p=project(blocks,page)
    placed=p['layout_result']['placements'][0]
    source_box=parent['source']['bbox_pt']
    assert placed['bbox'][0]==source_box[0]
    assert placed['bbox'][2]==source_box[2]
    path,d=export_project(p,tmp_path)
    assert d['compositeCount']==1 and d['editableCharacterCount']==len('Heading')
    with fitz.open(path) as pdf:
        assert pdf[0].get_text().strip()=='Heading'
        assert not pdf[0].get_images()
        span=pdf[0].get_text('dict')['blocks'][0]['lines'][0]['spans'][0]
        assert span['color']==int(child['style']['color'][1:],16)
        assert abs((span['bbox'][0]+span['bbox'][2])/2-150)<1
        assert pdf[0].get_drawings()


def test_compact_badge_with_expanded_detector_polygon_keeps_background(tmp_path):
    rgb,raw,page=fixture()
    raw['ocr_pipeline']['line_assignments'][0].update(box=[203,49,397,91],
        polygon=[[203,49],[397,49],[397,91],[203,91]])
    raw['overall_ocr_res']['rec_boxes']=[[203,49,397,91]]
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==1 and blocks[0]['kind']=='composite'
    assert blocks[0]['content']['background']['fill']=='#000000'


def test_two_paragraphs_keep_ids_and_resize_together(tmp_path):
    rgb,raw,page=fixture(two=True)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==1 and blocks[0]['kind']=='composite'
    parent=blocks[0]
    assert [c['id'] for c in parent['content']['children']]==['source-1-layout-0','source-1-layout-1']
    for child in parent['content']['children']:child['content']['text']='Long content '*20
    narrow=measure_block(parent,100);wide=measure_block(parent,250)
    assert narrow['height']>wide['height']
    assert len(wide['children'])==2
    assert all(v['y']+v['measurement']['height']<=wide['height'] for v in wide['children'])
    assert len(derive_relations(dict(blocks=blocks,source_pages=[page]))['source_order'])==1


def test_textured_image_has_metadata_transcript_and_no_extra_pdf_text(tmp_path):
    rgb,raw,page=fixture(texture=True)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==1 and blocks[0]['kind']=='image'
    assert blocks[0]['content']['transcript'][0]['text']=='Heading'
    assert blocks[0]['content']['transcript_rendering']=='metadata-only'
    path,d=export_project(project(blocks,page),tmp_path)
    assert d['editableCharacterCount']==0
    with fitz.open(path) as pdf:
        assert pdf[0].get_text()=='' and len(pdf[0].get_images())==1
        embedded=json.loads(pdf.embfile_get('rebuild-project.json'))
        assert embedded['blocks'][0]['content']['transcript'][0]['text']=='Heading'


def test_incomplete_container_ocr_preserves_entire_original(tmp_path):
    rgb,raw,page=fixture(two=True,missing=True)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==1 and blocks[0]['kind']=='image'
    assert blocks[0]['source']['label']=='container'
    assert blocks[0]['quality']['state']=='image_fallback'
    assert blocks[0]['source']['bbox_px']==[180,40,421,182]
    assert len(blocks[0]['source']['child_sources'])==2


@pytest.mark.parametrize('format',['html','md'])
def test_composite_zip_retains_children_background_and_references(tmp_path,format):
    rgb,raw,page=fixture(two=True)
    p=project(blocks_from_page(raw,rgb,page,tmp_path),page)
    path,_=export_project(p,tmp_path,format=format)
    with zipfile.ZipFile(path) as z:
        body=z.read(f'summary/summary.{format}').decode()
        assert body.count('Body content')==2 and 'composite-container' in body
        assert 'background:#ffffff' in body
        assert all(c['source']['reference_asset'] in z.namelist() for c in p['blocks'][0]['content']['children'])


@pytest.mark.parametrize('defect',['duplicate','missing','overflow','overlap','text'])
def test_composite_validation_rejects_loss_and_bad_geometry(tmp_path,defect):
    rgb,raw,page=fixture(two=True)
    p=project(blocks_from_page(raw,rgb,page,tmp_path),page)
    children=p['layout_result']['placements'][0]['measurement']['children']
    if defect=='duplicate':children.append(copy.deepcopy(children[0]))
    if defect=='missing':children.pop()
    if defect=='overflow':children[0]['x']=10000
    if defect=='overlap':children[1]['y']=children[0]['y']
    if defect=='text':children[0]['measurement']['lines'][0]['text']='Changed'
    with pytest.raises(ValueError):_validate(p)


def test_composite_stays_whole_when_paginated(tmp_path):
    rgb,raw,page=fixture(two=True)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    body=copy.deepcopy(blocks[0]['content']['children'][0]);body['id']='prefix'
    body['content']['text']='line\n'*15
    # Old saved flow projects retain legal whole-composite pagination. The
    # artificial prefix overlaps the panel in source space, so this fixture
    # deliberately selects the compatibility flow rather than source layout.
    p=dict(blocks=[body]+blocks,source_pages=[page])
    p['relations']=derive_relations(p)
    p['relations']['layout_policy']='single-column-reflow'
    layout_document(p)
    placements=[v for v in p['layout_result']['placements'] if v['block_id']==blocks[0]['id']]
    assert len(placements)==1 and placements[0]['content_range']==dict(whole=True)
    _validate(p)


def test_offcenter_heading_not_forced_center_and_source_unchanged(tmp_path):
    rgb,raw,page=fixture(two=True)
    raw['layout_det_res']['boxes']=raw['layout_det_res']['boxes'][:1]
    raw['layout_det_res']['boxes'][0].update(label='paragraph_title',coordinate=[30,300,180,330])
    raw['overall_ocr_res'].update(rec_boxes=[[30,300,180,325]],block_ids=['layout-0'],
                                 rec_texts=['Heading'],rec_scores=[.99],line_ids=['line-0'])
    rgb=np.full_like(rgb,255)
    original=copy.deepcopy(raw)
    block=blocks_from_page(raw,rgb,page,tmp_path)[0]
    assert block['style']['align']!='center'
    assert block['local_layout']['anchor']['horizontal']=='flow-left'
    assert raw==original


def test_heading_next_never_attaches_across_source_pages():
    a=dict(id='a',kind='text',source=dict(page_id='p1',label='doc_title',bbox_pt=[1,1,9,5],bbox_px=[1,1,9,5]),local_layout={})
    b=copy.deepcopy(a);b.update(id='b');b['source'].update(page_id='p2',label='text')
    pages=[dict(id=p,width_px=100,height_px=100) for p in ('p1','p2')]
    r=derive_relations(dict(blocks=[a,b],source_pages=pages))
    assert r['source_order']==['a','b'] and r['attachments']==[]


def test_column_heading_uses_supported_column_not_page_center():
    def item(identifier,label,box):
        return dict(id=identifier,kind='text',source=dict(label=label,bbox_px=box,
            ocr_lines=[dict(bbox_px=box)]),style={},local_layout={})
    heading=item('h','paragraph_title',[80,30,180,55])
    peers=[heading,item('p1','text',[30,80,230,150]),item('p2','text',[30,180,230,250])]
    page=dict(width_px=600,height_px=800,detections=dict(regions=[]))
    infer_alignment(heading,peers,page)
    assert heading['style']['align']=='center'
    assert heading['style']['align_reference']=='column'
    assert heading['local_layout']['anchor']['reference_bbox_px']==[30,80,230,150]


def test_large_unknown_region_does_not_force_unrelated_blocks_into_composite(tmp_path):
    rgb,raw,page=fixture(panel=False,two=True)
    raw['region_det_res']=dict(boxes=[dict(id='region-0',coordinate=[0,0,600,800])])
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(blocks)==2 and all(b['kind']=='text' for b in blocks)


def test_wide_body_with_common_left_edge_is_not_internally_centered():
    block=dict(id='body', kind='text', source=dict(label='text', bbox_px=[90,195,936,291],
        ocr_lines=[dict(bbox_px=b) for b in
                   [[92,197,914,219],[92,220,932,242],[92,244,880,266],[93,268,700,290]]]),
        style={},local_layout={})
    infer_alignment(block,[block],dict(width_px=1014,height_px=1308,detections={}))
    assert block['style']['align']=='left'
    assert block['local_layout']['anchor']['horizontal']=='flow-left'
