"""Manually specified synthetic truth for the V3-to-scene adapter.

These tests validate adaptation and ownership, not neural recognition accuracy.
No private PDF, downloaded image, model network call or local font is needed.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import pymupdf as fitz
import pytest

from backend.models import validate_document
from backend.structure_parser import _table_structure, compose_page, parse_document
from backend.structure_parser import _select_native


def folders(tmp_path):
    for folder in ('pages', 'images', 'text', 'vectors', 'backgrounds'):
        (tmp_path/folder).mkdir(exist_ok=True)
    return tmp_path


def test_fragmented_native_glyphs_replace_one_enclosing_ocr_line():
    records = [dict(id='ocr', text='ABCD', bbox=[10,10,50,20])]
    records += [dict(id=f'n{i}', text=c, bbox=[10+i*10,10,20+i*10,20],
                     nativeElement={'text':c}) for i,c in enumerate('ABCD')]
    selected = _select_native(records)
    assert [r['text'] for r in selected] == list('ABCD')
    assert records[0]['supersededBy'] == 'n0'


def test_overlapping_native_boxes_cannot_inflate_line_coverage():
    records = [dict(id='ocr', text='Native and scanned', bbox=[0,10,100,20])]
    records += [dict(id=f'n{i}', text='Native', bbox=[0,10,20,20],
                     nativeElement={'text':'Native'}) for i in range(6)]
    assert records[0] in _select_native(records)
    assert 'supersededBy' not in records[0]


def test_native_coverage_crossing_layout_blocks_does_not_trigger_fallback(tmp_path):
    rgb = np.full((50,100,3),255,np.uint8)
    raw = dict(parsing_res_list=[
        dict(block_label='text',block_bbox=[0,0,100,40],block_content='ABCD'),
        dict(block_label='text',block_bbox=[10,10,20,20],block_content='A')],
        overall_ocr_res=dict(rec_texts=['ABCD'],rec_boxes=[[10,10,50,20]]))
    native=[]
    from backend.parser import _base
    for i,c in enumerate('ABCD'):
        b=[10+i*10,10,20+i*10,20]
        e=_base('text',c,b,i,'native')
        e.update(text=c,fontFamily='sans-serif',fontSize=10,fontWeight='400',
                 fontStyle='normal',color='#000000')
        native.append(dict(element=e,bbox=b))
    page=compose_page(raw,rgb,100,50,folders(tmp_path),'truth',native=native)
    assert sorted(e['text'] for e in page['elements'] if e['type']=='text') == list('ABCD')
    assert all(r.get('ownerId') for r in page['recognition'])


def truth():
    # Independent title; picture's internal label; adjacent body; a merged
    # header, empty lower cell, and two lower populated cells in one table.
    rgb = np.full((240, 400, 3), 255, np.uint8)
    rgb[15:28, 20:100] = 0
    rgb[45:95, 20:120] = [190, 220, 230]
    rgb[60:70, 40:80] = 0
    rgb[60:72, 140:250] = 0
    for y in (120,170,219):
        rgb[y,20:320] = 0
    for x in (20,319):
        rgb[120:220,x] = 0
    for x in (120,220):
        rgb[170:220,x] = 0
    raw = dict(
        parsing_res_list=[
            dict(block_label='paragraph_title', block_bbox=[20, 12, 150, 32], block_content='Title', block_order=0),
            dict(block_label='image', block_bbox=[20, 45, 120, 95], block_content='', block_order=1),
            dict(block_label='text', block_bbox=[140, 45, 280, 95], block_content='Body', block_order=2),
            dict(block_label='table', block_bbox=[20, 120, 320, 220], block_content='', block_order=3)],
        overall_ocr_res=dict(rec_texts=['Title', 'Map Label', 'Body', 'Header', 'Left', 'Right'],
                             rec_boxes=[[20,15,100,28], [40,60,80,70], [140,60,250,72],
                                        [45,130,120,145], [40,180,90,195], [240,180,290,195]],
                             rec_scores=[.99]*6),
        table_res_list=[dict(pred_html='<table><tr><th colspan="3">Header</th></tr>'
                            '<tr><td>Left</td><td></td><td>Right</td></tr></table>',
                            cell_box_list=[[20,120,320,170], [20,170,120,220],
                                           [120,170,220,220], [220,170,320,220]])])
    return rgb, raw


def test_coordinates_semantics_and_picture_ownership(tmp_path):
    rgb, raw = truth()
    raw['_service'] = dict(engine='test-build', seconds=.25, cacheHit=False)
    # Unequal pixel/point scales exercise conversion independently in x/y.
    page = compose_page(raw, rgb, 200, 80, folders(tmp_path), 'truth')
    assert page['modelDispatch'] == raw['_service']
    picture = next(e for e in page['elements'] if e.get('layoutLabel') == 'image')
    assert picture['bbox'] == pytest.approx([10, 15, 60, 95/3])
    assert [r['text'] for r in picture['retainedText']] == ['Map Label']
    visible = [e for e in page['elements'] if e['type'] == 'text']
    assert 'Map Label' not in [e['text'] for e in visible]
    title = next(e for e in visible if e['text'] == 'Title')
    assert title['contentType'] == 'heading'  # no size heuristic may undo it
    assert title['x'] == 10
    assert title['y'] == 5
    assert all(r.get('ownerId') for r in page['recognition'])
    scene = dict(schemaVersion=1, id='truth', name='Synthetic manual truth', pages=[page])
    assert validate_document(scene)['pages'][0]['recognitionBackend'] == 'v3'


def test_merged_and_empty_cells_preserved_once(tmp_path):
    rgb, raw = truth()
    page = compose_page(raw, rgb, 400, 240, folders(tmp_path), 'truth')
    table = page['tables'][0]
    assert len(table['cells']) == 4
    assert table['columns'] == [20,120,220,320]
    assert table['rows'] == [120,170,220]
    header = table['cells'][0]
    assert (header['rowspan'], header['colspan']) == (1,3)
    assert header['bbox'] == [20,120,320,170]
    empty = next(c for c in table['cells'] if c['row'] == 1 and c['column'] == 1)
    assert empty['htmlText'] == '' and empty['textIds'] == []
    assert table['referenceSrc'].endswith('-reference.png')
    assert table['geometryStatus'] == 'model-cell-boxes-aligned'
    text = [e['text'] for e in page['elements'] if e['type'] == 'text']
    assert text == ['Title', 'Body', 'Header', 'Left', 'Right']
    borders = [e for e in page['elements'] if e.get('tableBorder')]
    assert borders and borders[0]['width'] == 300
    assert all(e['shape'] == 'line' for e in borders)
    assert table['cells'][0]['borders'] == dict(top=True,bottom=True,left=True,right=True)


def test_html_mismatch_uses_single_content_without_fake_appearance(tmp_path):
    rgb, raw = truth()
    raw['table_res_list'][0]['pred_html'] = raw['table_res_list'][0]['pred_html'].replace('Header', 'Correct Header')
    page = compose_page(raw, rgb, 400, 240, folders(tmp_path), 'truth')
    text = [e for e in page['elements'] if e['type'] == 'text']
    header = next(e for e in text if e['text'] == 'Correct Header')
    assert header['textOrigin'] == 'table-html'
    assert 'appearanceSrc' not in header
    assert header['fontSize'] < 20  # never use the 50px cell height as font size
    assert not any(e['text'] == 'Header' for e in text)
    assert header['tableCell']['colspan'] == 3
    assert all(r.get('ownerId') for r in page['recognition'])


def test_residual_alpha_has_no_text_or_table_pixels(tmp_path):
    rgb, raw = truth()
    rgb[230:233, 350:355] = 0  # truly unclassified decoration
    page = compose_page(raw, rgb, 400, 240, folders(tmp_path), 'truth')
    residual = next(e for e in page['elements'] if e.get('layoutLabel') == 'unclassified')
    assert [residual['x'],residual['y'],residual['width'],residual['height']] == [350,230,5,3]
    path = tmp_path/residual['src'].split('/truth/',1)[1]
    pixels = np.asarray(Image.open(path))
    assert pixels.shape == (3,5,4)
    assert np.all(pixels[:,:,3] == 255)
    assert not any(e.get('scanComposite') and e['width'] == 400 and e['height'] == 240 for e in page['elements'])


def test_table_source_edge_halo_does_not_erase_neighboring_unknown_content(tmp_path):
    rgb,raw=truth()
    rgb[122:218,323]=0  # a source border a few pixels outside predicted box
    rgb[225:229,330:340]=0  # neighboring unknown content must still survive
    page=compose_page(raw,rgb,400,240,folders(tmp_path),'truth')
    residual=next(e for e in page['elements'] if e.get('layoutLabel')=='unclassified')
    assert residual['x']==330 and residual['y']==225
    assert residual['width']==10 and residual['height']==4


def test_layout_failure_is_explicit_and_input_unchanged(tmp_path):
    rgb = np.full((60,90,3),255,np.uint8)
    original = rgb.copy()
    with pytest.raises(ValueError, match='有效版面区域'):
        compose_page({}, rgb, 90, 60, folders(tmp_path), 'truth')
    np.testing.assert_array_equal(rgb, original)


def test_wireless_table_has_no_invented_visible_grid(tmp_path):
    rgb, raw = truth()
    rgb[120:220] = 255
    page = compose_page(raw, rgb, 400,240,folders(tmp_path),'truth')
    assert page['tables'][0]['observedStyle']['borderWidth'] == 0
    assert not any(e.get('tableBorder') for e in page['elements'])


def test_horizontal_only_rules_do_not_gain_vertical_borders(tmp_path):
    rgb, raw = truth()
    rgb[120:220] = 255
    for y in (120,170,219):
        rgb[y,20:320] = 0
    page = compose_page(raw,rgb,400,240,folders(tmp_path),'truth')
    table = page['tables'][0]
    assert table['observedStyle']['borderVisible']
    assert all(not c['borders']['left'] and not c['borders']['right'] for c in table['cells'])
    assert all(e['rotation'] == 0 for e in page['elements'] if e.get('tableBorder'))


def test_invalid_cell_geometry_does_not_override_html_topology():
    block = dict(id='table', bbox=[10,20,110,80], text='')
    table = _table_structure(dict(pred_html='<table><tr><td rowspan="2">A</td><td>B</td></tr>'
                                          '<tr><td>C</td></tr></table>',
                                   cell_box_list=[[float('nan'),0,4,4], [[2,3]]]),
                             block, .5, .25, 200, 100)
    assert table['geometryStatus'] == 'estimated-from-html-topology'
    assert table['rows'] == [5,12.5,20]
    assert table['columns'] == [5,30,55]
    assert [(c['row'],c['column'],c['rowspan'],c['colspan']) for c in table['cells']] == [(0,0,2,1),(0,1,1,1),(1,1,1,1)]
    assert _table_structure(dict(pred_html='<table><tr><td rowspan="-1">bad</td></tr></table>'), block, 1,1,200,100) is None
    assert _table_structure(dict(pred_html='<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>'), block,1,1,200,100) is None


def test_overlapping_cell_boxes_mark_geometry_estimated():
    block=dict(id='table',bbox=[0,0,100,100],text='')
    result=dict(pred_html='<table><tr><td>A</td><td>B</td></tr></table>',cell_box_list=[[0,0,90,100],[10,0,100,100]])
    table=_table_structure(result,block,1,1,100,100)
    assert table['geometryStatus']=='estimated-from-html-topology'
    assert table['columns']==[0,50,100]


def test_incomplete_html_table_is_retained_as_picture(tmp_path):
    rgb,raw=truth()
    raw['table_res_list'][0]['pred_html']='<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>'
    page=compose_page(raw,rgb,400,240,folders(tmp_path),'truth')
    assert not page['tables']
    picture=next(e for e in page['elements'] if e.get('layoutLabel')=='table')
    assert picture['type']=='image'
    assert any('未得到有效 HTML 拓扑' in warning for warning in page['warnings'])


def test_two_tables_without_geometry_match_explicit_html_instead_of_reversing(tmp_path):
    rgb=np.full((200,400,3),255,np.uint8)
    first='<table><tr><td>First</td><td>1</td></tr></table>'
    second='<table><tr><td>Second</td><td>2</td></tr></table>'
    raw=dict(parsing_res_list=[dict(block_label='table',block_bbox=[10,10,190,90],block_content=first,block_order=0),
                                dict(block_label='table',block_bbox=[210,110,390,190],block_content=second,block_order=1)],
             table_res_list=[dict(pred_html=second,cell_box_list=[]),dict(pred_html=first,cell_box_list=[])])
    page=compose_page(raw,rgb,400,200,folders(tmp_path),'truth')
    assert len(page['tables'])==2
    assert [table['cells'][0]['htmlText'] for table in page['tables']]==['First','Second']
    assert all(t['resultAssociation']=='matching-html-topology-and-content' for t in page['tables'])
    assert all(t['geometryStatus']=='estimated-from-html-topology' for t in page['tables'])


def test_geometryless_table_results_cannot_claim_unidentified_blocks(tmp_path):
    rgb=np.full((200,400,3),255,np.uint8)
    raw=dict(parsing_res_list=[dict(block_label='table',block_bbox=[10,10,190,90],block_content='',block_order=0),
                                dict(block_label='table',block_bbox=[210,110,390,190],block_content='',block_order=1)],
             table_res_list=[dict(pred_html='<table><tr><td>First</td></tr></table>',cell_box_list=[]),
                             dict(pred_html='<table><tr><td>Second</td></tr></table>',cell_box_list=[])])
    page=compose_page(raw,rgb,400,200,folders(tmp_path),'truth')
    assert not page['tables']
    assert len([e for e in page['elements'] if e.get('layoutLabel')=='table'])==2
    assert not any(e['type']=='text' for e in page['elements'])


def test_unrelated_table_geometry_does_not_assign_wrong_html(tmp_path):
    rgb=np.full((200,400,3),255,np.uint8)
    own='<table><tr><td>Own</td></tr></table>'
    raw=dict(parsing_res_list=[dict(block_label='table',block_bbox=[10,10,190,90],block_content=own)],
             table_res_list=[dict(pred_html='<table><tr><td>Wrong</td></tr></table>',cell_box_list=[[220,110,390,190]])])
    page=compose_page(raw,rgb,400,200,folders(tmp_path),'truth')
    assert page['tables'][0]['resultAssociation']=='layout-block-html-only'
    assert page['tables'][0]['cells'][0]['htmlText']=='Own'
    assert page['tables'][0]['geometryStatus']=='estimated-from-html-topology'


def test_native_pdf_skips_models(monkeypatch, tmp_path):
    path = tmp_path/'native.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=300,height=200)
        page.insert_text((20,40),'Native editable document',fontsize=12)
        pdf.save(path)
    def forbidden(rgb):
        raise AssertionError('Native page must not start models')
    monkeypatch.setattr('backend.structure_parser.recognize_page', forbidden)
    scene = parse_document([path],tmp_path/'scene','native','Native')
    assert scene['parser']['name'] == 'pp-structure-v3'
    assert scene['pages'][0]['recognitionBackend'] == 'native'
    assert any(e['type']=='text' and 'Native editable' in e['text'] for e in scene['pages'][0]['elements'])


def test_scanned_pdf_uses_models_once_and_preserves_file(monkeypatch,tmp_path):
    rgb, raw = truth()
    image = tmp_path/'source.png'
    Image.fromarray(rgb).save(image)
    source = tmp_path/'scan.pdf'
    with fitz.open() as pdf:
        page = pdf.new_page(width=200,height=120)
        page.insert_image(page.rect,filename=str(image))
        pdf.save(source)
    before = source.read_bytes()
    seen = []
    def recognize(pixels):
        seen.append(pixels.shape)
        return raw
    monkeypatch.setattr('backend.structure_parser.recognize_page',recognize)
    scene = parse_document([source],tmp_path/'scene','scan','Scan',mode='rebuild')
    assert seen == [(240,400,3)]
    assert len(scene['pages'][0]['tables']) == 1
    assert source.read_bytes() == before
