"""Manual evidence for feature four's block identity and two-stage boundary."""
import copy
import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import pymupdf as fitz
import pytest

from backend.adaptive_reconstruction import blocks_from_page, derive_relations, convert_adaptive
from test_app import local_app


def evidence():
    image = Image.new('RGB',(600,800),'white')
    draw = ImageDraw.Draw(image)
    draw.text((40,30),'Adaptive title',fill='black')
    draw.rectangle((40,400,280,560),outline='blue',width=3)
    boxes = [dict(id='layout-0',label='paragraph_title',score=.95,coordinate=[40,20,550,70]),
             dict(id='layout-1',label='text',score=.95,coordinate=[40,90,550,190]),
             dict(id='layout-2',label='table',score=.95,coordinate=[40,220,550,350]),
             dict(id='layout-3',label='image',score=.95,coordinate=[40,400,280,560])]
    raw = dict(width=600,height=800,layout_det_res=dict(boxes=boxes),
               region_det_res=dict(boxes=[dict(id='region-0',label='region',coordinate=[20,10,580,760])]),
               overall_ocr_res=dict(rec_texts=['Adaptive title','Original body with symbols [A] and {B}.'],
                                   rec_boxes=[[45,25,340,60],[45,100,540,130]],rec_scores=[.99,.99],
                                   block_ids=['layout-0','layout-1']),
               table_res_list=[dict(block_id='layout-2',
                   pred_html='<table><tr><th colspan="2">Values</th></tr><tr><td>Alpha</td><td>42</td></tr></table>',
                   cell_box_list=[[40,220,550,280],[40,280,295,350],[295,280,550,350]])],
               diagnostics=[],_service=dict(cacheHit=False,modelProfile=[]))
    page = dict(id='source-1',width_px=600,height_px=800,width_pt=300.,height_pt=400.)
    return np.asarray(image),raw,page


def test_three_branches_share_identity_source_and_content_contract(tmp_path):
    rgb,raw,page=evidence()
    original=copy.deepcopy(raw)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert raw==original
    assert [b['kind'] for b in blocks]==['text','text','table','image']
    assert len({b['id'] for b in blocks})==4
    assert all({'id','kind','source','content','style','local_layout','quality'}<=b.keys() for b in blocks)
    assert blocks[1]['content']['text']=='Original body with symbols [A] and {B}.'
    assert blocks[2]['content']['cells'][0]['colspan']==2
    assert blocks[2]['content']['cells'][0]['text']=='Values'
    assert blocks[2]['source']['bbox_px']==[40,220,550,350]
    assert blocks[2]['source']['bbox_pt']==[20,110,275,175]
    assert all((tmp_path/b['source']['reference_asset']).is_file() for b in blocks)
    project=dict(blocks=blocks,source_pages=[page])
    relationships=derive_relations(project)
    assert relationships['source_order']==[b['id'] for b in blocks]
    assert all(v=='source-1-region-0' for v in relationships['region_membership'].values())


def test_table_repair_evidence_survives_without_modifying_model_result(tmp_path):
    rgb,raw,page=evidence()
    table=raw['table_res_list'][0]
    table.update(reconstruction_evidence=dict(method='ruled-row-bands-and-repeated-ocr-gutters',
        coordinate_space='table-local-pixels',line_assignments=[dict(ocr_indexes=[0])]),
        crop_origin=[40,220],raw_model_pred_html='<table></table>',raw_model_cell_box_list=[[0,0,510,130]])
    original=copy.deepcopy(raw)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    repaired=next(b for b in blocks if b['kind']=='table')
    assert repaired['source']['table_reconstruction']['evidence']==table['reconstruction_evidence']
    assert repaired['source']['table_reconstruction']['crop_origin']==[40,220]
    assert repaired['source']['table_reconstruction']['raw_model_cell_box_list']==[[0,0,510,130]]
    assert 'table-geometry-reconciled-from-row-bands' in repaired['quality']['issues']
    repaired['source']['table_reconstruction']['evidence']['line_assignments'].clear()
    assert raw==original


def test_page_line_identity_labels_and_batch_evidence_survive_block_adaptation(tmp_path):
    rgb,raw,page=evidence()
    raw['overall_ocr_res'].update(line_ids=['line-title','line-body'],labels=['paragraph_title','text'])
    raw['ocr_pipeline']=dict(policy='full-page-det-owned-line-batches-v1',coordinate_space='page-pixels',
        inputDetCount=3,acceptedLineCount=2,skippedVisualLineCount=1,
        recognitionBatches=[dict(line_ids=['line-body','line-title'],batch_size=2,padded_pixels=20000)],
        line_assignments=[dict(line_id='line-title',block_id='layout-0',label='paragraph_title',state='recognized'),
                          dict(line_id='line-body',block_id='layout-1',label='text',state='recognized'),
                          dict(line_id='line-art',block_id=None,state='visual-conflict',excluded_ids=['layout-3'])])
    original=copy.deepcopy(raw)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    by_id={b['source']['layout_detection_id']:b for b in blocks}
    for index,identifier in enumerate(['layout-0','layout-1']):
        lines=by_id[identifier]['source']['ocr_lines']
        assert len(lines)==1
        assert lines[0]['line_id']==raw['overall_ocr_res']['line_ids'][index]
        assert lines[0]['label']==raw['overall_ocr_res']['labels'][index]
        assert lines[0]['bbox_px']==raw['overall_ocr_res']['rec_boxes'][index]
        assert lines[0]['evidence_index']==index
        assert lines[0]['text']==raw['overall_ocr_res']['rec_texts'][index]
    assert page['recognition']['ocr_pipeline']==original['ocr_pipeline']
    assert page['recognition']['service']==original['_service']
    assert json.loads(json.dumps(page['recognition']))['ocr_pipeline']==original['ocr_pipeline']
    assert raw==original,'Adapting block evidence must not rewrite shared OCR results'


@pytest.mark.parametrize('reason',[
    'ocr-line-crosses-layout-blocks',
    'ocr-line-crosses-visual-block',
    'ocr-line-unmatched-block',
])
def test_ambiguous_ownership_preserves_involved_text_and_table_as_images(tmp_path,reason):
    rgb,raw,page=evidence()
    raw['overall_ocr_res']['rec_texts'].append('Recognized table evidence')
    raw['overall_ocr_res']['rec_boxes'].append([45,230,540,260])
    raw['overall_ocr_res']['rec_scores'].append(.98)
    raw['overall_ocr_res']['block_ids'].append('layout-2')
    raw['overall_ocr_res'].update(line_ids=['line-title','line-body','line-table'],
                                labels=['paragraph_title','text','table'])
    # Unmatched lines have candidate IDs but no selected owner. The other
    # diagnostics can identify both the selected owner and another candidate.
    diagnostic=dict(reason=reason,line_id='line-ambiguous',
                    candidate_ids=['layout-1','layout-2'],box=[45,180,540,240])
    if reason!='ocr-line-unmatched-block':
        diagnostic['block_id']='layout-1'
    if reason=='ocr-line-crosses-visual-block':
        diagnostic['excluded_ids']=['layout-3']
    raw['diagnostics'].append(diagnostic)
    original=copy.deepcopy(raw)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    by_id={b['source']['layout_detection_id']:b for b in blocks}
    assert len(blocks)==4,'Fallback must replace each logical block, not append another copy'
    for identifier,label,line_id in [('layout-1','text','line-body'),('layout-2','table','line-table')]:
        block=by_id[identifier]
        assert block['kind']=='image'
        assert block['source']['label']==label
        assert block['quality']['state']=='image_fallback'
        assert block['quality']['issues']==['ambiguous-ocr-line-ownership']
        assert block['source']['ocr_lines'][0]['line_id']==line_id,'Evidence remains available for review'
        assert not {'text','paragraphs','cells','pred_html'} & block['content'].keys()
        assert block['content']['asset']==block['source']['reference_asset']
        x0,y0,x1,y1=map(int,block['source']['bbox_px'])
        with Image.open(tmp_path/block['content']['asset']) as crop:
            assert crop.size==(x1-x0,y1-y0)
            assert np.array_equal(np.asarray(crop.convert('RGB')),rgb[y0:y1,x0:x1])
            assert np.asarray(crop.getchannel('A')).min()==255
    editable=[b for b in blocks if b['kind']=='text']
    assert [b['id'] for b in editable]==['source-1-layout-0']
    assert editable[0]['content']['text']=='Adaptive title'
    assert not any(b['kind']=='table' for b in blocks),'Ambiguous HTML must not produce a second visible table'
    assert page['recognition']['diagnostics']==original['diagnostics']
    assert raw==original


@pytest.mark.parametrize('reason',['ocr-line-recognition-rejected','empty-ocr-line-crop'])
@pytest.mark.parametrize('identifier',['layout-1','layout-2'])
def test_partial_recognition_failure_preserves_the_whole_source_block(tmp_path,reason,identifier):
    rgb,raw,page=evidence()
    raw['overall_ocr_res']['rec_texts'].append('Successful table line')
    raw['overall_ocr_res']['rec_boxes'].append([45,230,540,260])
    raw['overall_ocr_res']['rec_scores'].append(.98)
    raw['overall_ocr_res']['block_ids'].append('layout-2')
    raw['diagnostics'].append(dict(reason=reason,line_id='line-rejected',block_id=identifier))
    # Successful OCR/table HTML alone must not hide a second failed line.
    rgb=rgb.copy()
    detection=next(b for b in raw['layout_det_res']['boxes'] if b['id']==identifier)
    x0,y0,x1,y1=map(int,detection['coordinate'])
    rgb[y1-15:y1-5,x0+5:x1-5]=[20,50,180]
    original=copy.deepcopy(raw)
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    by_id={b['source']['layout_detection_id']:b for b in blocks}
    block=by_id[identifier]
    assert block['kind']=='image'
    assert block['quality']['state']=='image_fallback'
    assert 'ambiguous-ocr-line-ownership' in block['quality']['issues']
    assert block['source']['ocr_lines'],'Successful lines remain as evidence, rather than visible partial text'
    assert not {'text','paragraphs','cells','pred_html'} & block['content'].keys()
    with Image.open(tmp_path/block['content']['asset']) as crop:
        assert np.array_equal(np.asarray(crop.convert('RGB')),rgb[y0:y1,x0:x1])
    other='layout-2' if identifier=='layout-1' else 'layout-1'
    assert by_id[other]['kind']==('table' if other=='layout-2' else 'text')
    assert len(blocks)==4
    assert raw==original


def test_fully_owned_text_never_reappears_as_fallback_image(tmp_path):
    rgb,raw,page=evidence()
    raw['layout_det_res']['boxes'].append(dict(id='nested',label='text',coordinate=[60,420,200,450]))
    raw['diagnostics'].append(dict(block_id='nested',reason='fully-owned-by-another-block'))
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert not any(b['id']=='source-1-nested' for b in blocks)
    assert page['suppressed_detection_ids']==['nested']


def test_invalid_table_keeps_original_label_and_reference(tmp_path):
    rgb,raw,page=evidence()
    raw['table_res_list'][0]['pred_html']='<table><tr><td rowspan="0">bad</td></tr></table>'
    table=next(b for b in blocks_from_page(raw,rgb,page,tmp_path) if b['source']['label']=='table')
    assert table['kind']=='image'
    assert table['quality']['state']=='image_fallback'
    assert table['content']['asset']==table['source']['reference_asset']


def test_overlap_uses_owned_pixels_and_keeps_full_reference(tmp_path):
    rgb,raw,page=evidence()
    raw['block_ownership']={'layout-3':dict(excluded_block_ids=['another'],
        excluded_rects=[[40,400,100,460]],owned_pixel_count=34800,total_pixel_count=38400)}
    image=next(b for b in blocks_from_page(raw,rgb,page,tmp_path) if b['source']['label']=='image')
    with Image.open(tmp_path/image['content']['asset']) as crop:
        assert crop.getpixel((0,0))[3]==0
        assert crop.getpixel((100,100))[3]==255
    with Image.open(tmp_path/image['source']['reference_asset']) as reference:
        assert reference.getpixel((0,0))[3]==255


def test_uncovered_foreground_retained_without_ocr_or_duplicate_pixels(tmp_path):
    rgb,raw,page=evidence()
    rgb=rgb.copy()
    rgb[650:690,50:180]=[230,20,20]
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    unknown=[b for b in blocks if b['source']['label']=='unknown']
    assert len(unknown)==1
    assert unknown[0]['kind']=='image'
    assert unknown[0]['quality']['state']=='image_fallback'
    assert 'unclassified-foreground-preserved-as-image' in unknown[0]['quality']['issues']
    assert unknown[0]['source']['bbox_px'][1]>=640
    assert unknown[0]['source']['ocr_lines']==[]


def test_table_border_slivers_stay_in_reference_but_nearby_text_is_retained(tmp_path):
    rgb,raw,page=evidence()
    rgb=rgb.copy()
    rgb[220:350,555:558]=0  # measured table edge extends 5px beyond layout box
    rgb[355:358,40:520]=0  # disconnected edge fragments, not an L-shaped graphic
    rgb[290:305,580:600]=0  # a nearby text fragment must remain
    blocks=blocks_from_page(raw,rgb,page,tmp_path)
    assert len(page['coverage_diagnostics'])==2
    assert all(d['block_id']=='layout-2' for d in page['coverage_diagnostics'])
    unknown=[b for b in blocks if b['source']['label']=='unknown']
    assert unknown
    assert any(b['source']['bbox_px'][2]>=600 for b in unknown)
    table=next(b for b in blocks if b['kind']=='table')
    assert table['content']['column_widths']==[127.5,127.5]
    rgb,raw,page=evidence()
    rgb=rgb.copy()
    rgb[250:266,555:558]=0  # short letter by the table edge is not decoration
    blocks=blocks_from_page(raw,rgb,page,tmp_path/'short')
    assert any(b['source']['label']=='unknown' for b in blocks)


def test_occluded_table_does_not_publish_successful_topology(tmp_path):
    rgb,raw,page=evidence()
    raw['block_ownership']={'layout-2':dict(excluded_block_ids=['figure'],
        excluded_rects=[[40,220,80,260]],owned_pixel_count=64700,total_pixel_count=66300)}
    table=next(b for b in blocks_from_page(raw,rgb,page,tmp_path) if b['source']['label']=='table')
    assert table['kind']=='image'
    assert 'overlapping-table-preserved-as-image' in table['quality']['issues']
    assert table['content']['asset']!=table['source']['reference_asset']


def test_source_columns_and_spanning_title_have_computed_order():
    def block(identifier,box):
        return dict(id=identifier,kind='text',source=dict(page_id='p',label='text',bbox_pt=box,bbox_px=box))
    blocks=[block('right',[320,100,550,180]),block('title',[40,20,550,60]),
            block('left-second',[40,200,270,280]),block('left-first',[40,100,270,180])]
    page=dict(id='p',width_px=600,height_px=800,detections=dict(regions=[]))
    relations=derive_relations(dict(blocks=blocks,source_pages=[page]))
    assert relations['source_order']==['title','left-first','left-second','right']
    assert any(g['type']=='columns' for g in relations['layout_groups'])


@pytest.mark.parametrize('format',['pdf','html','md'])
def test_feature_four_real_export_uses_new_project_contract(tmp_path,monkeypatch,format):
    rgb,raw,_=evidence()
    source=tmp_path/'input.png'
    Image.fromarray(rgb).save(source)
    before=source.read_bytes()
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',lambda pixels:copy.deepcopy(raw))
    output,report=convert_adaptive([source],tmp_path/'output','test',format=format)
    assert source.read_bytes()==before
    project=json.loads((tmp_path/'output/rebuild-project.json').read_text(encoding='utf-8'))
    assert project['schema']=='feature4-rebuild/1'
    assert len(project['blocks'])==4
    assert project['layout_result']['placements']
    assert report['recognitionBackend']=='v3' and report['tableCount']==1
    if format=='pdf':
        with fitz.open(output) as pdf:
            text=''.join(page.get_text() for page in pdf)
            assert 'Adaptive title' in text and 'Alpha' in text and '42' in text
            assert 'rebuild-project.json' in pdf.embfile_names()
    else:
        with zipfile.ZipFile(output) as archive:
            body=archive.read(f'summary/summary.{format}').decode('utf-8')
            assert 'colspan="2"' in body and 'Alpha' in body
            assert 'project.json' in archive.namelist()


def test_fourth_http_route_dispatches_new_v3_operation(local_app,monkeypatch):
    client,_,_=local_app
    rgb,raw,_=evidence()
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION','v3')
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',lambda pixels:copy.deepcopy(raw))
    def forbidden(*args,**kwargs):
        raise AssertionError('Fixed scene parser must not implement new feature four')
    monkeypatch.setattr('backend.structure_parser.recognize_page',forbidden)
    image=io.BytesIO()
    Image.fromarray(rgb).save(image,format='PNG')
    response=client.post('/api/convert/rebuild',files=[('files',('input.png',image.getvalue(),'image/png'))],
                         data={'ocr':'true','format':'html'})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert json.loads(archive.read('project.json'))['schema']=='feature4-rebuild/1'


def test_image_pixel_limit_fails_before_recognition(tmp_path,monkeypatch):
    source=tmp_path/'large.png'
    Image.new('RGB',(200,200),'white').save(source)
    monkeypatch.setattr('backend.adaptive_reconstruction.MAX_PIXELS',10000)
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',
        lambda *_:pytest.fail('Oversized image must fail before recognition'))
    with pytest.raises(ValueError,match='像素限制'):
        convert_adaptive([source],tmp_path/'out','large')


def test_large_pdf_raster_is_capped_while_source_points_remain(tmp_path,monkeypatch):
    source=tmp_path/'large.pdf'
    with fitz.open() as pdf:
        pdf.new_page(width=200,height=200)
        pdf.save(source)
    calls=[]
    def recognize(rgb):
        h,w=rgb.shape[:2]
        calls.append((w,h))
        return dict(width=w,height=h,layout_det_res=dict(boxes=[]))
    monkeypatch.setattr('backend.adaptive_reconstruction.MAX_PIXELS',10000)
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',recognize)
    convert_adaptive([source],tmp_path/'out','large',format='md')
    assert calls and calls[0][0]*calls[0][1]<=10000
    p=json.loads((tmp_path/'out/rebuild-project.json').read_text(encoding='utf-8'))
    assert p['source_pages'][0]['width_pt']==200
    assert p['blocks'][0]['source']['bbox_pt']==[0,0,200,200]
