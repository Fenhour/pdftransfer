"""Real-font export checks for the measured feature-four reconstruction IR."""
import copy
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pymupdf as fitz
import pytest

from backend.adaptive_export import _validate, export_project
from backend.font_library import get_library


@pytest.fixture(scope='module')
def font_data():
    lib = get_library()
    face = lib.resolve('Header Left Right Body <&> 第二页', 'Arial', False, False, safe_only=True)
    assert face is not None
    font = lib.export_font_for_key(face.key)
    return face.key, font


def text_block(identifier, text):
    return dict(id=identifier,kind='text',source=dict(page_id='p1',label='text',bbox_pt=[10,10,150,60]),
                content=dict(text=text,paragraphs=[dict(id=identifier+'-p',text=text)]),
                style=dict(font_size_pt=12,family='Arial',bold=False,italic=False,color='#000000',origin='role-template'),
                local_layout={},quality=dict(state='needs_review',issues=[]))


def text_measure(text, font_data):
    key,font = font_data
    return dict(width=200.,height=(font.ascender-font.descender)*12,
                font_key=key,font_size=12.,color='#000000',line_step=16.,
                lines=[dict(text=text,start=0,end=len(text),x=0.,baseline=font.ascender*12,
                            height=(font.ascender-font.descender)*12)])


def placement(identifier, measurement, page=0, x=20., y=20., start=0, end=None):
    return dict(fragment_id=f'{identifier}-fragment-{page}',block_id=identifier,page_index=page,
                bbox=[x,y,x+measurement['width'],y+measurement['height']],
                measurement=measurement,content_range=dict(start=start,end=end if end is not None else len(measurement['lines'][0]['text'])),
                repeat_header=False)


def project(blocks, placements, pages=1):
    return dict(schema='feature4-rebuild/1',blocks=blocks,
                source_pages=[dict(id='p1',width_pt=300,height_pt=300)],
                relations=dict(source_order=[b['id'] for b in blocks]),
                layout_result=dict(pages=[dict(index=i,width=300.,height=300.) for i in range(pages)],
                                   placements=placements,resolved_order=[b['id'] for b in blocks],order_changes=[]))


def table_block():
    cells=[dict(id='header',row=0,column=0,rowspan=1,colspan=2,text='Header',header=True),
           dict(id='left',row=1,column=0,rowspan=1,colspan=1,text='Left',header=False),
           dict(id='right',row=1,column=1,rowspan=1,colspan=1,text='Right',header=False)]
    return dict(id='table',kind='table',source=dict(page_id='p1',label='table',reference_asset='assets/table.png'),
                content=dict(row_count=2,column_count=2,cells=cells),style=dict(border_width_pt=.5),
                local_layout={},quality=dict(state='needs_review',issues=[]))


def table_measure(font_data):
    key,font=font_data
    result=dict(width=200.,height=60.,column_widths=[100.,100.],row_heights=[30.,30.],cells=[])
    for cell in table_block()['content']['cells']:
        m=dict(cell,x=cell['column']*100.,y=cell['row']*30.,width=cell['colspan']*100.,height=30.,
               font_key=key,font_size=12.,color='#000000',line_step=16.,
               lines=[dict(text=cell['text'],start=0,end=len(cell['text']),x=4.,baseline=4.+font.ascender*12,
                           height=(font.ascender-font.descender)*12)])
        result['cells'].append(m)
    return result


def test_pdf_visible_text_has_safe_embedded_fonts_and_project_attachment(tmp_path,font_data):
    text='Body <&> 第二页'
    b=text_block('body',text)
    p=project([b],[placement('body',text_measure(text,font_data))])
    destination,diagnostics=export_project(p,tmp_path)
    assert diagnostics['pdfSafety']['reopened']['hiddenText'] is False
    with fitz.open(destination) as pdf:
        assert 'Body <&>' in pdf[0].get_text()
        assert '第二页' in pdf[0].get_text()
        assert 'rebuild-project.json' in pdf.embfile_names()
        saved=json.loads(pdf.embfile_get('rebuild-project.json'))
        assert saved['blocks'][0]['id']=='body'
        assert all(span['type']==0 for span in pdf[0].get_texttrace())


def test_pdf_merged_table_uses_real_text_and_keeps_original_reference(tmp_path,font_data):
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(40,20),'white').save(tmp_path/'assets/table.png')
    b=table_block()
    m=table_measure(font_data)
    p=project([b],[dict(fragment_id='table-f1',block_id='table',page_index=0,bbox=[20.,20.,220.,80.],
                       content_range=dict(row_start=0,row_end=2),measurement=m,repeat_header=False)])
    destination,diagnostics=export_project(p,tmp_path)
    assert diagnostics['tableCount']==1
    with fitz.open(destination) as pdf:
        text=pdf[0].get_text()
        assert all(text.count(t)==1 for t in ['Header','Left','Right'])
        assert 'table-reference.png' in pdf.embfile_names()
        assert pdf[0].get_drawings()
        assert not pdf[0].get_images(), 'Original table is attached, never painted behind true text'


@pytest.mark.parametrize('format',['html','md'])
def test_zip_contains_real_table_and_requested_format_only(tmp_path,font_data,format):
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(40,20),'white').save(tmp_path/'assets/table.png')
    table=table_block()
    p=project([table],[dict(fragment_id='table-f1',block_id='table',page_index=0,bbox=[20.,20.,220.,80.],
                           content_range=dict(row_start=0,row_end=2),measurement=table_measure(font_data),repeat_header=False)])
    destination,_=export_project(p,tmp_path,format=format)
    with zipfile.ZipFile(destination) as archive:
        names=archive.namelist()
        assert 'project.json' in names and 'model/project.json' in names
        assert 'assets/table.png' in names
        body=archive.read(f'summary/summary.{format}').decode()
        assert 'colspan="2"' in body
        assert '<table' in body and 'Header' in body
        assert not any(n.endswith('.md' if format=='html' else '.html') for n in names)


def test_text_fragments_use_exact_content_ranges_on_different_pages(tmp_path,font_data):
    text='FirstSecond'
    b=text_block('body',text)
    first=text_measure('First',font_data)
    second=text_measure('Second',font_data)
    second['lines'][0].update(start=5,end=11)
    p=project([b],[placement('body',first,page=0,end=5),placement('body',second,page=1,start=5,end=11)],pages=2)
    destination,_=export_project(p,tmp_path)
    with fitz.open(destination) as pdf:
        assert len(pdf)==2 and 'First' in pdf[0].get_text() and 'Second' in pdf[1].get_text()


def test_image_only_document_preserves_real_image_without_hidden_ocr(tmp_path):
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(80,40),'blue').save(tmp_path/'assets/image.png')
    b=dict(id='image',kind='image',source=dict(page_id='p1',label='image'),
           content=dict(asset='assets/image.png',pixel_size=[80,40],aspect_ratio=2),style={},local_layout={},quality={})
    m=dict(width=200.,height=100.,asset='assets/image.png')
    p=project([b],[dict(fragment_id='image-f1',block_id='image',page_index=0,bbox=[20,20,220,120],
                       content_range={},measurement=m)])
    destination,diagnostics=export_project(p,tmp_path)
    assert diagnostics['textBlockCount']==0
    with fitz.open(destination) as pdf:
        assert len(pdf[0].get_images())==1 and not pdf[0].get_text().strip()


def test_html_escapes_ocr_content_and_preserves_measured_positions(tmp_path,font_data):
    text='<script>alert(1)</script> & body'
    b=text_block('body',text)
    m=text_measure(text,font_data)
    p=project([b],[placement('body',m)])
    destination,_=export_project(p,tmp_path,format='html')
    with zipfile.ZipFile(destination) as archive:
        body=archive.read('summary/summary.html').decode()
        assert '<script>' not in body and '&lt;script&gt;' in body and '&amp;' in body
        assert 'left:20.000pt;top:20.000pt' in body


@pytest.mark.parametrize('asset',['../outside.png','C:/private.png','assets\\image.png','/absolute.png'])
def test_asset_references_cannot_escape_project(tmp_path,asset):
    b=dict(id='image',kind='image',source=dict(label='image'),content=dict(asset=asset))
    m=dict(width=20.,height=20.,asset=asset)
    p=project([b],[dict(fragment_id='f',block_id='image',page_index=0,bbox=[20,20,40,40],content_range={},measurement=m)])
    with pytest.raises(ValueError,match='资源'):
        export_project(p,tmp_path)


def test_mismatched_measurement_and_missing_contents_are_rejected(tmp_path,font_data):
    b=text_block('body','Body')
    p=project([b],[placement('body',text_measure('Body',font_data))])
    p['layout_result']['placements'][0]['bbox'][2]=299
    with pytest.raises(ValueError,match='测量'):
        export_project(p,tmp_path)
    p=project([b],[placement('body',text_measure('Wrong',font_data),end=4)])
    with pytest.raises(ValueError,match='源内容'):
        export_project(p,tmp_path)


def test_repeated_table_header_is_generated_content_not_duplicate_body(tmp_path,font_data):
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(40,20),'white').save(tmp_path/'assets/table.png')
    table=table_block()
    table['content']['row_count']=3
    for old in list(table['content']['cells'][1:]):
        table['content']['cells'].append(dict(old,id=old['id']+'2',row=2,text=old['text']+'2'))
    first=table_measure(font_data)
    second=copy.deepcopy(first)
    second['cells'][0]['repeat_header']=True
    for cell in second['cells'][1:]:
        cell['id']+='2'
        cell['row']=2
        cell['text']+='2'
        cell['lines'][0]['text']+='2'
        cell['lines'][0]['end']+=1
    placements=[]
    for i,m in enumerate([first,second]):
        placements.append(dict(fragment_id=f'table-f{i}',block_id='table',page_index=i,bbox=[20.,20.,220.,80.],
                               content_range=dict(row_start=0 if i==0 else 2,row_end=2 if i==0 else 3),
                               measurement=m,repeat_header=i==1))
    p=project([table],placements,pages=2)
    destination,_=export_project(p,tmp_path)
    with fitz.open(destination) as pdf:
        assert len(pdf)==2
        assert all('Header' in page.get_text() for page in pdf)
        assert 'Left2' not in pdf[0].get_text() and 'Left2' in pdf[1].get_text()


def test_overlapping_or_missing_text_ranges_are_not_exported(tmp_path,font_data):
    b=text_block('body','BodyBody')
    m=text_measure('Body',font_data)
    p=project([b],[placement('body',m,page=0,end=4),placement('body',m,page=1,start=0,end=4)],pages=2)
    with pytest.raises(ValueError,match='遗漏或重复'):
        export_project(p,tmp_path)


def test_measured_text_must_fit_without_horizontal_squashing(tmp_path,font_data):
    b=text_block('body','Body')
    m=text_measure('Body',font_data)
    m['width']=3
    p=project([b],[placement('body',m,end=4)])
    with pytest.raises(ValueError,match='溢出内容块'):
        export_project(p,tmp_path,format='html')


@pytest.mark.parametrize('format',['pdf','html','md'])
def test_real_layout_measurements_export_without_a_second_font_fit(tmp_path,format):
    from backend.adaptive_layout import layout_document
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(120,40),'blue').save(tmp_path/'assets/image.png')
    Image.new('RGB',(40,20),'white').save(tmp_path/'assets/table.png')
    body=text_block('body','Measured text grows into several lines. '*12)
    table=table_block()
    table['style'].update(font_size_pt=11,family='Arial',line_height=1.25,color='#000000')
    image=dict(id='image',kind='image',source=dict(page_id='p1',label='image',bbox_pt=[0,0,180,60]),
               content=dict(asset='assets/image.png',pixel_size=[120,40],aspect_ratio=3),
               style={},local_layout={},quality={})
    p=project([body,table,image],[])
    layout_document(p,page_size=(300,180),margin=24)
    assert len(p['layout_result']['pages'])>1
    destination,diagnostics=export_project(p,tmp_path,format=format)
    assert destination.is_file()
    assert diagnostics['editableCharacterCount']>0
    if format=='pdf':
        with fitz.open(destination) as pdf:
            assert len(pdf)==len(p['layout_result']['pages'])
            text=''.join(page.get_text() for page in pdf)
            assert text.count('Measured')==12


@pytest.mark.parametrize('format',['pdf','html','md'])
def test_embedded_and_disk_json_are_the_same_final_snapshot(tmp_path,font_data,format):
    body=text_block('body','Body')
    p=project([body],[placement('body',text_measure('Body',font_data))])
    p['report']=dict(schema='feature4-conversion/1',layoutElapsedSeconds=1.25)
    (tmp_path/'assets').mkdir()
    Image.new('RGB',(20,20),'white').save(tmp_path/'assets/source.png')
    p['source_pages'][0]['reference_asset']='assets/source.png'
    destination,diagnostics=export_project(p,tmp_path,format=format)
    project_bytes=(tmp_path/'rebuild-project.json').read_bytes()
    report_bytes=(tmp_path/'conversion-report.json').read_bytes()
    assert json.loads(project_bytes)==p
    assert json.loads(report_bytes)==diagnostics==p['report']['export']
    if format=='pdf':
        with fitz.open(destination) as pdf:
            assert pdf.embfile_get('rebuild-project.json')==project_bytes
            assert pdf.embfile_get('conversion-report.json')==report_bytes
        assert diagnostics['pdfSafety']['finalPackaging']['characterInventory'] is True
    else:
        with zipfile.ZipFile(destination) as archive:
            assert archive.read('project.json')==project_bytes
            assert archive.read('model/project.json')==project_bytes
            assert archive.read('conversion-report.json')==report_bytes
            assert 'assets/source.png' in archive.namelist()


def test_a_body_cell_cannot_bypass_completeness_as_a_repeated_header(tmp_path,font_data):
    table=table_block()
    first=table_measure(font_data)
    second=copy.deepcopy(first)
    second['cells']=second['cells'][1:2]
    second['cells'][0].update(repeat_header=True,header=True)
    placements=[dict(fragment_id='table-f0',block_id='table',page_index=0,bbox=[20.,20.,220.,80.],
                     content_range=dict(row_start=0,row_end=2),measurement=first,repeat_header=False),
                dict(fragment_id='table-f1',block_id='table',page_index=1,bbox=[20.,20.,220.,80.],
                     content_range=dict(row_start=1,row_end=2),measurement=second,repeat_header=True)]
    with pytest.raises(ValueError,match='源表头'):
        export_project(project([table],placements,pages=2),tmp_path)


def test_whitespace_and_line_offsets_cannot_change_silently(tmp_path,font_data):
    block=text_block('body','A B')
    measured=text_measure('A B',font_data)
    measured['lines'][0]['text']='AB'
    with pytest.raises(ValueError,match='源内容'):
        export_project(project([block],[placement('body',measured,end=3)]),tmp_path)
    measured=text_measure('A B',font_data)
    measured['lines'][0]['start']=1
    with pytest.raises(ValueError,match='源内容范围'):
        export_project(project([block],[placement('body',measured,end=3)]),tmp_path)


def test_the_same_header_cannot_be_repeated_twice_inside_one_fragment(tmp_path,font_data):
    table=table_block()
    first=table_measure(font_data)
    second=copy.deepcopy(first)
    second['cells'][0]['repeat_header']=True
    second['cells'].append(copy.deepcopy(second['cells'][0]))
    placements=[dict(fragment_id='table-f0',block_id='table',page_index=0,bbox=[20.,20.,220.,80.],
                     content_range=dict(row_start=0,row_end=1),measurement=dict(first,cells=first['cells'][:1]),repeat_header=False),
                dict(fragment_id='table-f1',block_id='table',page_index=1,bbox=[20.,20.,220.,80.],
                     content_range=dict(row_start=1,row_end=2),measurement=second,repeat_header=True)]
    with pytest.raises(ValueError,match='单元格 ID 重复'):
        export_project(project([table],placements,pages=2),tmp_path)


def test_pdf_project_resource_paths_resolve_from_attachments(tmp_path,font_data):
    (tmp_path/'assets').mkdir()
    references=['assets/source-1.png','assets/table.png','assets/image.png','assets/image-crop.png']
    for index,name in enumerate(references):
        Image.new('RGB',(80,40),(index*50,80,150)).save(tmp_path/name)
    table=table_block()
    image=dict(id='image',kind='image',source=dict(page_id='p1',label='image',reference_asset=references[3]),
               content=dict(asset=references[2],pixel_size=[80,40],aspect_ratio=2),style={},local_layout={},quality={})
    placements=[dict(fragment_id='table-f0',block_id='table',page_index=0,bbox=[20.,20.,220.,80.],
                     content_range=dict(row_start=0,row_end=2),measurement=table_measure(font_data),repeat_header=False),
                dict(fragment_id='image-f0',block_id='image',page_index=0,bbox=[20.,100.,220.,200.],
                     content_range={},measurement=dict(width=200.,height=100.,asset=references[2]))]
    p=project([table,image],placements)
    p['source_pages'][0]['reference_asset']=references[0]
    destination,_=export_project(p,tmp_path)
    with fitz.open(destination) as pdf:
        saved=json.loads(pdf.embfile_get('rebuild-project.json'))
        paths={page['reference_asset'] for page in saved['source_pages']}
        for block in saved['blocks']:
            if block['source'].get('reference_asset'):
                paths.add(block['source']['reference_asset'])
            if block['kind']=='image':
                paths.add(block['content']['asset'])
        assert paths==set(references)
        for name in paths:
            assert pdf.embfile_get(name)==(tmp_path/name).read_bytes()


def scaled_composite_project(font_data):
    """A 90% display snapshot, independent of the layout scaling helper."""
    child = text_block('child', 'Body')
    block = dict(id='container', kind='composite', source=dict(page_id='p1', label='container'),
                 content=dict(children=[child], background=dict(fill='#ffffff'),
                              decorations=dict(border_width_pt=1., border_color='#123456')))
    cm = text_measure('Body', font_data)
    cm.update(width=171., height=cm['height']*.9, font_size=10.8, line_step=14.4, uniform_scale=.9)
    cm['lines'][0]['baseline'] *= .9
    cm['lines'][0]['height'] *= .9
    measured = dict(width=180., height=36., uniform_scale=.9,
                    children=[dict(block_id='child', x=4.5, y=4.5, measurement=cm)],
                    background=dict(fill='#ffffff'),
                    decorations=dict(border_width_pt=.9, border_color='#123456'))
    return project([block], [dict(fragment_id='container-f0', block_id='container', page_index=0,
                                 bbox=[20.,20.,200.,56.], measurement=measured,
                                 content_range=dict(whole=True))])


@pytest.mark.parametrize('format', ['pdf', 'html'])
def test_uniformly_scaled_composite_keeps_editable_text_and_scaled_border(tmp_path, font_data, format):
    p = scaled_composite_project(font_data)
    destination, diagnostics = export_project(p, tmp_path, format=format)
    assert diagnostics['editableCharacterCount'] == 4
    assert p['blocks'][0]['content']['decorations']['border_width_pt'] == 1.
    if format == 'pdf':
        with fitz.open(destination) as pdf:
            assert pdf[0].get_text().strip() == 'Body'
            assert not pdf[0].get_images()
            span = pdf[0].get_text('dict')['blocks'][0]['lines'][0]['spans'][0]
            assert span['size'] == pytest.approx(10.8)
            assert pdf[0].get_drawings()[0]['width'] == pytest.approx(.9)
    else:
        with zipfile.ZipFile(destination) as z:
            body = z.read('summary/summary.html').decode()
        assert 'font-size:10.800pt' in body
        assert 'box-shadow:inset 0 0 0 0.900pt #123456' in body


@pytest.mark.parametrize('defect', ['scale', 'border', 'background'])
def test_scaled_composite_requires_matching_scale_and_source_decoration(font_data, defect):
    p = scaled_composite_project(font_data)
    measured = p['layout_result']['placements'][0]['measurement']
    if defect == 'scale':
        measured['uniform_scale'] = .8
    elif defect == 'border':
        measured['decorations']['border_color'] = '#654321'
    else:
        measured['background']['fill'] = '#000000'
    with pytest.raises(ValueError, match='源内容不一致'):
        _validate(p)


def test_scaled_composite_rejects_a_child_with_an_independent_scale(font_data):
    p = scaled_composite_project(font_data)
    measured = p['layout_result']['placements'][0]['measurement']
    measured['children'][0]['measurement']['uniform_scale'] = 1.
    with pytest.raises(ValueError, match='父子必须采用相同'):
        _validate(p)


def test_cumulative_measurement_scaling_exports_the_same_source_text(tmp_path, font_data):
    from backend.regional_layout import scale_measurement
    p = scaled_composite_project(font_data)
    original = copy.deepcopy(p['layout_result']['placements'][0]['measurement'])
    measured = scale_measurement(original, .95)
    assert original['uniform_scale'] == .9
    assert measured['uniform_scale'] == pytest.approx(.855)
    fragment = p['layout_result']['placements'][0]
    fragment.update(measurement=measured, bbox=[20.,20.,20.+measured['width'],20.+measured['height']])
    destination, _ = export_project(p, tmp_path)
    with fitz.open(destination) as pdf:
        assert pdf[0].get_text().strip() == 'Body'
        assert pdf[0].get_drawings()[0]['width'] == pytest.approx(.855)


def regional_project(font_data):
    body = text_block('body', 'Body')
    p = project([body], [placement('body', text_measure('Body', font_data))])
    p['layout_result'].update(target_structure='source-preserving-regions',
                             frames=[dict(id='r1', page_index=0, source_page_id='p1',
                                          bbox=[10.,10.,250.,70.], block_ids=['body'])])
    p['layout_result']['pages'][0]['source_page_id'] = 'p1'
    p['layout_result']['placements'][0]['frame_id'] = 'r1'
    return p


def test_regional_export_preserves_page_and_frame_identity(tmp_path, font_data):
    p = regional_project(font_data)
    destination, _ = export_project(p, tmp_path)
    with fitz.open(destination) as pdf:
        saved = json.loads(pdf.embfile_get('rebuild-project.json'))
    assert saved['layout_result']['placements'][0]['bbox'] == p['layout_result']['placements'][0]['bbox']
    assert saved['layout_result']['pages'][0]['source_page_id'] == 'p1'


@pytest.mark.parametrize('defect,reason', [
    ('duplicate-frame', 'ID'), ('missing-frame', '所属区域'),
    ('overflow-frame', '区域边界'), ('cross-page-frame', '所属区域'),
    ('source-page', '一一对应'), ('missing-source-page', '一一对应'),
    ('block-source-page', '其他源页'), ('frame-source-page', '源页归属'),
    ('frame-page-overflow', '区域边界越出'), ('wrong-member', '区域成员'),
    ('duplicate-owner', '一个可见归属'),
])
def test_regional_export_rejects_cross_boundary_or_ambiguous_ownership(font_data, defect, reason):
    p = regional_project(font_data)
    layout = p['layout_result']
    frame = layout['frames'][0]
    if defect == 'duplicate-frame':
        layout['frames'].append(copy.deepcopy(frame))
    elif defect == 'missing-frame':
        layout['placements'][0]['frame_id'] = 'missing'
    elif defect == 'overflow-frame':
        frame['bbox'][2] = 210.
    elif defect == 'cross-page-frame':
        layout['pages'].append(dict(index=1, source_page_id='p2', width=300., height=300.))
        p['source_pages'].append(dict(id='p2', width_pt=300., height_pt=300.))
        frame.update(page_index=1, source_page_id='p2')
    elif defect == 'source-page':
        layout['pages'][0]['source_page_id'] = 'wrong'
    elif defect == 'missing-source-page':
        p['source_pages'].append(dict(id='p2', width_pt=300., height_pt=300.))
    elif defect == 'block-source-page':
        p['blocks'][0]['source']['page_id'] = 'p2'
    elif defect == 'frame-source-page':
        frame['source_page_id'] = 'p2'
    elif defect == 'frame-page-overflow':
        frame['bbox'][2] = 310.
    elif defect == 'duplicate-owner':
        layout['frames'].append(dict(frame, id='r2'))
    else:
        frame['block_ids'] = ['other']
    with pytest.raises(ValueError, match=reason):
        _validate(p)


def test_regional_export_still_rejects_overlapping_blocks(font_data):
    p = regional_project(font_data)
    other = copy.deepcopy(p['blocks'][0])
    other['id'] = 'other'
    p['blocks'].append(other)
    duplicate = copy.deepcopy(p['layout_result']['placements'][0])
    duplicate.update(fragment_id='other-f0', block_id='other')
    p['layout_result']['placements'].append(duplicate)
    p['layout_result']['frames'][0]['block_ids'].append('other')
    with pytest.raises(ValueError, match='发生重叠'):
        _validate(p)


def owned_overlap_project(scale=1.):
    boxes = ([20.,20.,100.,70.], [70.,40.,150.,90.])
    blocks, placements = [], []
    transform = [scale, 0., 0., scale, 5., 10.]
    for identifier, box in zip(('fine', 'mask'), boxes):
        source = dict(page_id='p1', label='image', layout_detection_id=identifier+'-det',
                      bbox_pt=list(box), bbox_px=list(box), reference_asset=f'assets/{identifier}-reference.png',
                      crop_to_page=[[1,0,box[0]],[0,1,box[1]],[0,0,1]])
        if identifier == 'mask':
            source['ownership'] = dict(excluded_block_ids=['fine-det'],
                                       excluded_rects=[[70.,40.,100.,70.]])
        block = dict(id=identifier, kind='image', source=source,
                     content=dict(asset=f'assets/{identifier}-owned.png', pixel_size=[80,50], aspect_ratio=1.6))
        blocks.append(block)
        target = [box[0]*scale+5., box[1]*scale+10., box[2]*scale+5., box[3]*scale+10.]
        placements.append(dict(fragment_id=identifier+'-f0', block_id=identifier, page_index=0,
                               bbox=target, measurement=dict(width=80.*scale, height=50.*scale,
                                                            asset=block['content']['asset']),
                               content_range=dict(whole=True), frame_id='r1', source_transform=list(transform)))
    p = project(blocks, placements)
    p['layout_result'].update(target_structure='source-preserving-regions',
                             frames=[dict(id='r1', page_index=0, source_page_id='p1',
                                          bbox=[10.,10.,200.,120.], block_ids=['fine','mask'])])
    p['layout_result']['pages'][0]['source_page_id'] = 'p1'
    return p


def write_owned_images(tmp_path):
    (tmp_path/'assets').mkdir()
    for identifier in ('fine', 'mask'):
        original = Image.new('RGBA', (80,50), 'blue' if identifier == 'fine' else 'red')
        original.save(tmp_path/f'assets/{identifier}-reference.png')
        if identifier == 'mask':
            alpha = Image.new('L', (80,50), 255)
            alpha.paste(0, (0,0,30,30))
            original.putalpha(alpha)
        original.save(tmp_path/f'assets/{identifier}-owned.png')


@pytest.mark.parametrize('scale', [1., .9])
def test_owned_alpha_mask_overlap_is_preserved_only_under_one_source_transform(tmp_path, scale):
    p = owned_overlap_project(scale)
    write_owned_images(tmp_path)
    destination, diagnostics = export_project(p, tmp_path)
    assert diagnostics['imageCount'] == 2
    with fitz.open(destination) as pdf:
        assert len(pdf[0].get_images()) == 2
        assert not pdf[0].get_text().strip()


def test_owned_overlap_uses_integer_crop_footprint_instead_of_fractional_detector_box(tmp_path):
    p = owned_overlap_project(.9)
    for block in p['blocks']:
        for key in ('bbox_pt', 'bbox_px'):
            source = block['source'][key]
            block['source'][key] = [source[0]+.2, source[1]+.2, source[2]-.2, source[3]-.2]
    write_owned_images(tmp_path)
    destination, _ = export_project(p, tmp_path)
    assert destination.is_file()


@pytest.mark.parametrize('defect', ['ownership', 'exclusion', 'transform', 'movement',
                                  'missing-transform', 'frame', 'shear'])
def test_owned_alpha_overlap_cannot_authorize_new_geometry(defect):
    p = owned_overlap_project()
    first, second = p['layout_result']['placements']
    if defect == 'ownership':
        p['blocks'][1]['source'].pop('ownership')
    elif defect == 'exclusion':
        p['blocks'][1]['source']['ownership']['excluded_rects'] = [[70.,40.,80.,50.]]
    elif defect == 'transform':
        second['source_transform'][4] += 2.
        second['bbox'][0] += 2.
        second['bbox'][2] += 2.
    elif defect == 'movement':
        second['bbox'][0] += 2.
        second['bbox'][2] += 2.
    elif defect == 'missing-transform':
        first.pop('source_transform')
    elif defect == 'frame':
        second['frame_id'] = 'r2'
        p['layout_result']['frames'][0]['block_ids'] = ['fine']
        p['layout_result']['frames'].append(dict(id='r2', page_index=0, source_page_id='p1',
                                               bbox=[10.,10.,200.,120.], block_ids=['mask']))
    else:
        first['source_transform'][1] = .1
    with pytest.raises(ValueError, match='发生重叠'):
        _validate(p)


@pytest.mark.parametrize('defect', ['opaque', 'single-pixel', 'wrong-size', 'rendered-asset'])
def test_owned_overlap_requires_actual_transparent_pixels(tmp_path, defect):
    p = owned_overlap_project()
    write_owned_images(tmp_path)
    name = tmp_path/'assets/mask-owned.png'
    if defect == 'opaque':
        Image.new('RGB', (80,50), 'red').save(name)
    elif defect == 'single-pixel':
        with Image.open(name) as source:
            changed = source.convert('RGBA')
        changed.putpixel((29,29), (255,0,0,1))
        changed.save(name)
    elif defect == 'wrong-size':
        Image.new('RGBA', (81,50), (255,0,0,0)).save(name)
    else:
        p['layout_result']['placements'][1]['measurement']['asset'] = 'assets/mask-reference.png'
    with pytest.raises(ValueError, match='透明像素'):
        export_project(p, tmp_path)
