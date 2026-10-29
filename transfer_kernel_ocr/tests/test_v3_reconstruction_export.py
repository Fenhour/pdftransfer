"""V3 model-table topology and cell text must survive the editable exports."""
import copy
import json
from html.parser import HTMLParser
import re
import zipfile
from PIL import Image

import pymupdf as fitz
import pytest

from backend.exporter import append_text_pages, write_text_pdf, portable_document
from backend.pdf_safety import character_inventory, prepare_fonts, save_verified
from backend.rebuild_export import _table_node, _text_node
from backend.rebuild_flow import flow_cell_text
from backend.reconstruction import convert_reconstruction
from test_adaptive_reconstruction import evidence


def _table():
    return dict(id='table',bbox=[20.,20.,220.,100.],columns=[20.,120.,220.],rows=[20.,60.,100.],
                observedStyle=dict(borderWidth=.5,borderColor='#000000'),cells=[
                    dict(row=0,column=0,rowspan=1,colspan=2,bbox=[20.,20.,220.,60.],textIds=['merged']),
                    dict(row=1,column=0,rowspan=1,colspan=1,bbox=[20.,60.,120.,100.],textIds=['a']),
                    dict(row=1,column=1,rowspan=1,colspan=1,bbox=[120.,60.,220.,100.],textIds=['b'])])


def _scene(elements=(),table=None):
    return dict(id='v3-export',name='V3 synthetic',pages=[dict(id='page',index=0,width=250.,height=150.,
        recognitionBackend='v3',appearanceStrategy='blank-page-typed-reconstruction',
        tables=[table] if table else [],elements=list(elements),warnings=[])])


def _text(identifier='merged',text='Merged cell title',box=(25.,25.,210.,30.)):
    x,y,width,height=box
    return dict(id=identifier,type='text',source='ocr',tableId='table',textOrigin='table-html',text=text,
                x=x,y=y,width=width,height=height,fontFamily='Arial',fontSize=12.,fontWeight='normal',
                fontStyle='normal',color='#000000',lineHeight=1.1)


def test_merged_cell_generates_real_html_span_and_no_phantom_cells():
    table=_table()
    by_id={e['id']:e for e in [_text(),_text('a','Left'),_text('b','Right')]}
    fixed=_table_node(table,by_id)
    reading=_table_node(table,by_id,fixed=False)
    for fragment in (fixed,reading):
        assert 'rowspan="1" colspan="2"' in fragment
        assert fragment.count('<td ')==3
        class TextOnly(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts=[]
            def handle_data(self,value):
                self.parts.append(value)
        parser=TextOnly()
        parser.feed(fragment)
        assert ''.join(parser.parts).count('Merged cell title')==1
        assert 'height:40.000pt' in fragment


def test_portable_scene_embeds_original_table_reference(tmp_path):
    Image.new('RGB',(10,10),'white').save(tmp_path/'reference.png')
    table=_table()
    table['referenceSrc']='/files/v3-export/reference.png'
    scene=_scene(table=table)
    exported=portable_document(scene,tmp_path)
    assert exported['pages'][0]['tables'][0]['referenceSrc'].startswith('data:image/png;base64,')
    assert table['referenceSrc'].startswith('/files/'), 'Export does not alter saved scene references'


def test_pdf_cell_boundaries_do_not_draw_through_merged_header(tmp_path):
    with fitz.open() as pdf:
        append_text_pages(pdf,_scene(table=_table()),tmp_path)
        segments=[item for drawing in pdf[0].get_drawings() for item in drawing['items'] if item[0]=='l']
        vertical=[(a,b) for _,a,b in segments if abs(a.x-120)<.01 and abs(b.x-120)<.01]
        assert vertical and all(min(a.y,b.y)>=60 for a,b in vertical)


def test_wireless_table_does_not_gain_visible_grid_rules(tmp_path):
    table=_table()
    table['observedStyle']['borderWidth']=0
    assert 'border:none' in _table_node(table,{})
    with fitz.open() as pdf:
        append_text_pages(pdf,_scene(table=table),tmp_path)
        assert len(pdf[0].get_drawings())==1  # page paper only


def test_partial_table_edges_and_rotated_rule_are_preserved(tmp_path):
    table=_table()
    table['borderElements']=True
    for cell in table['cells']:
        cell['borders']=dict(top=True,bottom=True,left=False,right=False)
    fragment=_table_node(table,{})
    assert 'border-left:none;' in fragment and 'border-top:0.500pt solid' in fragment
    element=dict(id='rule',type='shape',shape='line',rebuildVector=True,tableBorder=True,
                 x=90.,y=79.75,width=80.,height=.5,rotation=90,stroke='#000000',strokeWidth=.5)
    with fitz.open() as pdf:
        append_text_pages(pdf,_scene([element],table),tmp_path)
        segments=[item for drawing in pdf[0].get_drawings() for item in drawing['items'] if item[0]=='l']
        assert len(segments)==1
        _,start,end=segments[0]
        assert start.x==pytest.approx(130) and end.x==pytest.approx(130)
        assert start.y==pytest.approx(40) and end.y==pytest.approx(120)


def test_editable_table_border_elements_are_not_drawn_twice(tmp_path):
    table=_table()
    table['borderElements']=True
    elements=[]
    for index,cell in enumerate(table['cells']):
        x0,y0,x1,y1=cell['bbox']
        elements.append(dict(id=f'border-{index}',type='shape',shape='rect',rebuildVector=True,
                             tableBorder=True,tableId='table',x=x0,y=y0,width=x1-x0,height=y1-y0,
                             stroke='#000000',strokeWidth=.5))
    with fitz.open() as pdf:
        append_text_pages(pdf,_scene(elements,table),tmp_path)
        assert len(pdf[0].get_drawings())==4  # paper plus the three editable rectangles


def test_cell_html_fallback_wraps_and_fits_without_giant_font_or_lost_text(tmp_path):
    text='A long merged cell has several words that must remain editable and fit inside its border.'
    element=_text(text=text,box=(25.,25.,92.,43.))
    original=copy.deepcopy(element)
    font,layout=flow_cell_text(element)
    assert element==original
    assert len(layout['lines'])>1
    assert 1<=layout['fontSize']<=12
    assert len(layout['lines'])*layout['lineStep']<=element['height']+.001
    assert all(font.text_length(line,fontsize=layout['fontSize'])<=element['width']+.01 for line in layout['lines'])
    assert re.sub(r'\s','', ''.join(layout['lines']))==re.sub(r'\s','',text)
    html=_text_node(element)
    assert html.count('class="run"')==len(layout['lines'])
    scene=_scene([element])
    prepare_fonts(scene)
    with fitz.open() as pdf:
        append_text_pages(pdf,scene,tmp_path)
        save_verified(pdf,tmp_path/'cell.pdf',[character_inventory(scene['pages'][0])])
    with fitz.open(tmp_path/'cell.pdf') as pdf:
        assert re.sub(r'\s','',pdf[0].get_text())==re.sub(r'\s','',text)
        for span in pdf[0].get_texttrace():
            x0,y0,x1,y1=span['bbox']
            assert x0>=element['x']-.1 and x1<=element['x']+element['width']+.1
            assert y0>=element['y']-.1 and y1<=element['y']+element['height']+.1


def test_v3_canvas_pdf_export_uses_bounded_matcher_and_current_cell_edit(tmp_path,monkeypatch):
    element=_text(text='Edited content')
    element['fontMatch']=dict(text='Old content')
    scene=_scene([element])
    seen=[]
    def bounded(document,output_dir):
        seen.append(document['pages'][0]['recognitionBackend'])
    def exhaustive(*args,**kwargs):
        pytest.fail('V3 export must not re-run exhaustive legacy font matching')
    monkeypatch.setattr('backend.rebuild_fonts.match_rebuild_fonts',bounded)
    monkeypatch.setattr('backend.exporter.match_document_fonts',exhaustive)
    write_text_pdf(scene,tmp_path,tmp_path/'edited.pdf')
    assert seen==['v3']
    with fitz.open(tmp_path/'edited.pdf') as pdf:
        assert 'Edited content' in pdf[0].get_text()
    assert element['source']=='ocr', 'Export must not mutate saved canvas data'


@pytest.mark.parametrize('format',['pdf','html','md'])
def test_reconstruction_forwards_v3_and_generates_only_requested_format(tmp_path,monkeypatch,format):
    rgb,raw,_=evidence()
    source=tmp_path/'source.pdf'
    with fitz.open() as pdf:
        pdf.new_page(width=rgb.shape[1]/2,height=rgb.shape[0]/2)
        pdf.save(source)
    calls=[]
    def recognize(pixels):
        calls.append(pixels.shape)
        return copy.deepcopy(raw)
    def forbidden(*args,**kwargs):
        pytest.fail('V3 reconstruction must use typed blocks instead of the fixed scene parser')
    monkeypatch.setattr('backend.adaptive_reconstruction.recognize_page',recognize)
    monkeypatch.setattr('backend.reconstruction.parse_document',forbidden)
    root=tmp_path/'out'
    destination,report=convert_reconstruction([source],root,'v3-export',format=format,recognition_backend='v3')
    assert calls==[rgb.shape]
    assert destination.is_file()
    assert report['recognitionBackend']=='v3'
    assert report['schema']=='feature4-conversion/1'
    assert report['sourcePageCount']==1
    assert report['strategy']=='typed-blocks-two-level-adaptive'
    project=json.loads((root/'rebuild-project.json').read_text(encoding='utf-8'))
    assert project['schema']=='feature4-rebuild/1'
    assert len(project['blocks'])==4
    assert project['layout_result']['placements']
    if format!='pdf':
        with zipfile.ZipFile(destination) as archive:
            assert f'summary/summary.{format}' in archive.namelist()
            assert json.loads(archive.read('project.json'))['schema']=='feature4-rebuild/1'
            other='md' if format=='html' else 'html'
            assert not any(name.endswith('.'+other) for name in archive.namelist())
    else:
        with fitz.open(destination) as pdf:
            assert 'rebuild-project.json' in pdf.embfile_names()
            assert 'Adaptive title' in ''.join(page.get_text() for page in pdf)
    assert not (root/'rebuild-page-work').exists()
