"""Reconstruction ownership and honest image-boundary regressions."""
import io
import json
import re
import zipfile
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables._g_l_y_f import Glyph

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
import pytest

from backend.rebuild_layout import image_regions
from backend.rebuild_tables import ruled_tables
from backend.pdf_safety import character_inventory, prepare_fonts, save_verified, validate_pdf
from backend.exporter import append_text_pages
from backend.font_library import get_library
from test_app import local_app


def candidate(text, box):
    x0, y0, x1, y1 = box
    quad = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    return dict(element=dict(id=text, type='text', text=text, height=y1-y0,
                             polygon=quad, confidence=.99), quad=quad, bbox=box)


def fixture():
    image = Image.new('RGB', (600,450), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    draw.text((30,25), 'Independent body', font=font, fill='black')
    draw.rectangle((80,130,520,400), outline='black', width=3)
    # Disconnected diagram pieces inside one enclosing frame must stay intact.
    draw.ellipse((110,180,190,260), outline='blue', width=4)
    draw.rectangle((350,280,460,350), outline='red', width=3)
    draw.text((245,200), 'Figure label', font=font, fill='black')
    return np.asarray(image), [candidate('Independent body',[27,23,225,52]),
                              candidate('Figure label',[242,198,385,228])]


def test_boundary_is_full_artwork_not_ocr_rectangle_and_source_is_immutable():
    rgb, candidates = fixture()
    before = rgb.copy()
    text, regions, paper = image_regions(rgb, candidates, 1)
    assert [c['element']['text'] for c in text] == ['Independent body']
    assert len(regions) == 1
    region = regions[0]
    assert region['bbox'][0] <= 80 and region['bbox'][2] >= 521
    assert region['bbox'][1] <= 130 and region['bbox'][3] >= 401
    assert region['retainedText'][0]['text'] == 'Figure label'
    assert region['boundary']['exact'] is False
    assert paper == '#ffffff'
    assert np.array_equal(rgb,before)


def test_boundary_has_one_visible_owner_even_with_text_inside_frame():
    rgb, candidates = fixture()
    # A confident OCR line inside a large frame must not be duplicated by a
    # full rectangular crop, even when the frame has disconnected artwork.
    candidates.append(candidate('Extra line',[220,320,330,345]))
    text, regions, _ = image_regions(rgb,candidates,1)
    retained = [line['text'] for r in regions for line in r['retainedText']]
    independent = [c['element']['text'] for c in text]
    assert sorted(retained+independent) == sorted(c['element']['text'] for c in candidates)
    assert len(set(retained+independent)) == len(candidates)


def test_plain_page_does_not_keep_a_full_scan_behind_real_text():
    image = Image.new('RGB',(400,120),'white')
    ImageDraw.Draw(image).text((30,30),'BODY',font=ImageFont.load_default(size=24),fill='black')
    text, regions, _ = image_regions(np.asarray(image),[candidate('BODY',[27,27,110,62])],1)
    assert len(text) == 1 and not regions


def test_no_ocr_still_retains_art_and_reports_no_editable_text():
    rgb, _ = fixture()
    text, regions, _ = image_regions(rgb,[],1)
    assert not text and regions
    assert all(r['reviewRequired'] for r in regions)


def test_nearby_aligned_artwork_without_frame_is_not_guessed_as_fragments():
    image=Image.new('RGB',(600,240),'white')
    draw=ImageDraw.Draw(image)
    draw.ellipse((40,80,140,180),outline='blue',width=4)
    draw.rectangle((155,80,275,180),outline='red',width=4)
    _,regions,_=image_regions(np.asarray(image),[],1)
    assert len(regions)==1
    assert regions[0]['bbox'][0]<=40 and regions[0]['bbox'][2]>=276
    assert regions[0]['boundary']['exact'] is False


def test_independent_paragraph_separates_otherwise_aligned_figures():
    image=Image.new('RGB',(300,450),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((20,30,180,130),outline='blue',width=3)
    draw.rectangle((20,300,180,410),outline='red',width=3)
    draw.text((25,205),'BODY',font=ImageFont.load_default(size=20),fill='black')
    text,regions,_=image_regions(np.asarray(image),[candidate('BODY',[22,202,90,235])],1)
    assert len(text)==1 and len(regions)==2


def grid_fixture():
    image=Image.new('RGB',(380,300),'white')
    draw=ImageDraw.Draw(image)
    for x in [20,180,340]:
        draw.line((x,80,x,260),fill='black',width=2)
    for y in [80,170,260]:
        draw.line((20,y,340,y),fill='black',width=2)
    candidates=[]
    for text,x,y in [('A1',45,105),('B1',205,105),('A2',45,195),('B2',205,195)]:
        draw.text((x,y),text,font=ImageFont.load_default(size=20),fill='black')
        candidates.append(candidate(text,[x-2,y-2,x+35,y+27]))
    return np.asarray(image),candidates


def test_simple_grid_has_real_cells_and_does_not_use_a_picture_as_table():
    rgb,candidates=grid_fixture()
    tables=ruled_tables(rgb,candidates,1)
    assert len(tables)==1
    table=tables[0]
    assert len(table['columns'])==len(table['rows'])==3
    assert len(table['cells'])==4
    assert [c['textIds'] for c in table['cells']]==[['A1'],['B1'],['A2'],['B2']]
    assert table['reviewRequired']


def test_grid_with_unexplained_artwork_is_not_falsely_declared_a_table():
    rgb,candidates=grid_fixture()
    image=Image.fromarray(rgb)
    ImageDraw.Draw(image).ellipse((95,115,145,150),outline='red',width=3)
    assert ruled_tables(np.asarray(image),candidates,1)==[]


def _raster_pdf(rgb):
    buffer=io.BytesIO()
    Image.fromarray(rgb).save(buffer,format='PNG')
    with fitz.open() as pdf:
        pdf.new_page(width=rgb.shape[1]/2,height=rgb.shape[0]/2).insert_image(
            fitz.Rect(0,0,rgb.shape[1]/2,rgb.shape[0]/2),stream=buffer.getvalue())
        return pdf.tobytes()


def _mock_ocr(monkeypatch,candidates):
    monkeypatch.setattr('backend.parser._engine',lambda:lambda _:
                        ([(c['quad'],c['element']['text'],.99) for c in candidates],None))


def test_rebuild_pdf_uses_blank_page_and_verified_visible_fonts(local_app,monkeypatch):
    client,inputs,outputs=local_app
    rgb,candidates=fixture()
    _mock_ocr(monkeypatch,candidates)
    original=_raster_pdf(rgb)
    response=client.post('/api/convert/rebuild',files=[('files',('scan.pdf',original,'application/pdf'))])
    assert response.status_code==200,response.text
    assert response.headers['x-reconstruction-status']=='partial'
    with fitz.open(stream=response.content,filetype='pdf') as pdf:
        assert 'Independent body' in pdf[0].get_text()
        assert 'Figure label' not in pdf[0].get_text()
        assert len(pdf[0].get_image_info())==1
        box=pdf[0].get_image_info()[0]['bbox']
        assert box[1]>40 and box[3]>195, 'Not a full-page scan or an OCR-line crop'
        assert all(s['type']!=3 for s in pdf[0].get_texttrace())
        report=json.loads(pdf.embfile_get('conversion-report.json'))
        assert report['strategy']=='blank-page-typed-reconstruction'
    saved=json.loads(next(outputs.glob('*/conversion.json')).read_text(encoding='utf-8'))
    assert saved['pdfSafety']['afterSubset']['referencedGlyphOutlines']
    assert not list(outputs.glob('*/rebuild-page-work'))
    assert next(inputs.glob('*/01_scan.pdf')).read_bytes()==original
    assert client.get('/api/documents').json()==[]


@pytest.mark.parametrize('format',['html','md'])
def test_rebuild_web_formats_keep_intact_crop_coordinates_and_transcript(local_app,monkeypatch,format):
    client,_,outputs=local_app
    rgb,candidates=fixture()
    _mock_ocr(monkeypatch,candidates)
    response=client.post('/api/convert/rebuild',files=[('files',('scan.pdf',_raster_pdf(rgb),'application/pdf'))],
                         data={'format':format})
    assert response.status_code==200,response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        summary=archive.read(f'summary/summary.{format}').decode()
        assert 'Independent body' in summary and 'Figure label' in summary
        assert 'data-bbox=' in summary and 'data-relative-bbox=' in summary
        assets=[n for n in archive.namelist() if n.startswith('assets/')]
        assert len(assets)==1
        crop=Image.open(io.BytesIO(archive.read(assets[0])))
        model=json.loads(archive.read('model/page-0001.json'))
        image=next(e for e in model['elements'] if e['type']=='image')
        x0,y0,x1,y1=[round(v*2) for v in image['bbox']]
        assert np.array_equal(np.asarray(crop),rgb[y0:y1,x0:x1])
        assert image['boundary']['exact'] is False
        assert not any('appearanceSrc' in e for e in model['elements'])
        assert not any(n.endswith('.ttf') for n in archive.namelist())
    assert not list(outputs.glob('*/rebuild-page-work'))


@pytest.mark.parametrize('format',['pdf','html','md'])
def test_rebuilt_table_has_cells_in_web_and_vector_rules_in_pdf(local_app,monkeypatch,format):
    client,_,_=local_app
    rgb,candidates=grid_fixture()
    _mock_ocr(monkeypatch,candidates)
    response=client.post('/api/convert/rebuild',files=[('files',('table.pdf',_raster_pdf(rgb),'application/pdf'))],
                         data={'format':format})
    assert response.status_code==200,response.text
    if format=='pdf':
        with fitz.open(stream=response.content,filetype='pdf') as pdf:
            assert all(t in pdf[0].get_text() for t in ['A1','B1','A2','B2']),repr(pdf[0].get_text())
            assert not pdf[0].get_images()
            assert len(pdf[0].get_drawings())>=6
            report=json.loads(pdf.embfile_get('conversion-report.json'))
            reference=report['pages'][0]['adapter']['tableReferences'][0]
            assert len(reference['cells'])==4
            crop=Image.open(io.BytesIO(pdf.embfile_get(reference['attachment'])))
            x0,y0,x1,y1=[round(v*2) for v in reference['bbox']]
            assert np.array_equal(np.asarray(crop),rgb[y0:y1,x0:x1])
    else:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            summary=archive.read(f'summary/summary.{format}').decode()
            assert '<table' in summary and summary.count('<td ')==4
            assert all(t in summary for t in ['A1','B1','A2','B2'])
            assert f'table/table.{format}' in archive.namelist()
            assert '表格原图参考' in summary
            model=json.loads(archive.read('model/page-0001.json'))
            table=model['tables'][0]
            asset=table['referenceSrc'].removeprefix('../')
            crop=Image.open(io.BytesIO(archive.read(asset)))
            x0,y0,x1,y1=[round(v*2) for v in table['referenceBbox']]
            assert np.array_equal(np.asarray(crop),rgb[y0:y1,x0:x1])


def test_rebuild_rejects_invalid_requests_and_ocr_failure_without_success_file(local_app,monkeypatch):
    client,_,outputs=local_app
    rgb,_=fixture()
    files=[('files',('scan.pdf',_raster_pdf(rgb),'application/pdf'))]
    for data in [{'format':'docx'},{'ocr':'false'},{'pages':'2'}]:
        assert client.post('/api/convert/rebuild',files=files,data=data).status_code==422
    def fail():
        raise RuntimeError('test unavailable')
    monkeypatch.setattr('backend.parser._engine',fail)
    response=client.post('/api/convert/rebuild',files=files)
    assert response.status_code==422 and 'OCR' in response.text
    assert not list(outputs.glob('*/rebuilt-text.pdf'))
    assert not list(outputs.glob('*/rebuild-page-work'))


def test_pdf_alias_glyphs_keep_hyphen_and_soft_hyphen_distinct(tmp_path):
    text='Questions 1-10; soft\u00adhyphen'
    page=dict(width=300,height=100,elements=[dict(id='test',type='text',source='manual',text=text,
              fontFamily='Times New Roman',fontSize=12,x=10,y=20,width=280,height=20)])
    scene=dict(id='aliases',pages=[page])
    prepare_fonts(scene)
    with fitz.open() as pdf:
        append_text_pages(pdf,scene,tmp_path)
        destination=tmp_path/'aliases.pdf'
        checks=save_verified(pdf,destination,[character_inventory(page)])
    assert checks['beforeSubset']['characterInventory'] and checks['afterSubset']['characterInventory']


def test_pdf_gate_rejects_incorrect_expected_unicode_even_if_file_renders(tmp_path):
    scene=dict(id='gate',pages=[dict(width=300,height=100,elements=[dict(id='gate-text',type='text',
               text='Visible body',x=20,y=20,width=200,height=20,fontSize=12,fontFamily='Arial')])])
    prepare_fonts(scene)
    with fitz.open() as pdf:
        append_text_pages(pdf,scene,tmp_path)
        with pytest.raises(ValueError,match='映射'):
            validate_pdf(pdf.tobytes(),[{'not-the-original':1}])


def test_missing_family_fallback_preserves_serif_sans_and_monospace():
    library=get_library()
    for family,generic in [('Unknown Serif Family','serif'),('Unknown Sans Family','sans-serif'),('Unknown Monospace','monospace')]:
        face=library.resolve('BODY Text',family,safe_only=True)
        assert face and face.generic==generic


def test_different_pages_with_latin_chinese_and_styles_keep_valid_mappings(tmp_path):
    pages=[]
    for index,text in enumerate(['FIRST-42','中文 Test 99']):
        pages.append(dict(width=350,height=150,elements=[dict(id=f'p{index}',type='text',text=text,
                     x=20,y=30,width=280,height=25,fontSize=16,fontFamily='Times New Roman',
                     fontWeight='bold' if index else 'normal')]))
    scene=dict(id='multi',pages=pages)
    prepare_fonts(scene)
    with fitz.open() as pdf:
        append_text_pages(pdf,scene,tmp_path)
        target=tmp_path/'multi.pdf'
        save_verified(pdf,target,[character_inventory(p) for p in pages])
    with fitz.open(target) as result:
        assert len(result)==2
        assert 'FIRST-42' in result[0].get_text()
        assert '中文' in result[1].get_text()


def test_gate_detects_empty_outline_even_when_unicode_remains_readable(tmp_path):
    page=dict(width=300,height=100,elements=[dict(id='glyph',type='text',text='X',x=20,y=20,
              width=30,height=20,fontSize=14,fontFamily='Arial')])
    scene=dict(id='glyphs',pages=[page])
    prepare_fonts(scene)
    with fitz.open() as pdf:
        append_text_pages(pdf,scene,tmp_path)
        span=pdf[0].get_texttrace()[0]
        gid=span['chars'][0][1]
        xref=pdf[0].get_fonts(full=True)[0][0]
        with TTFont(io.BytesIO(pdf.extract_font(xref)[3])) as font:
            empty=Glyph()
            empty.numberOfContours=0
            font['glyf'][font.getGlyphOrder()[gid]]=empty
            stream=io.BytesIO()
            font.save(stream)
        # get_fonts() returns the Type0 resource, not the FontFile2 stream.
        descendants=pdf.xref_get_key(xref,'DescendantFonts')[1]
        cid=int(re.search(r'(\d+)\s+0\s+R',descendants).group(1))
        descriptor=int(pdf.xref_get_key(cid,'FontDescriptor')[1].split()[0])
        binary=int(pdf.xref_get_key(descriptor,'FontFile2')[1].split()[0])
        pdf.update_stream(binary,stream.getvalue())
        with pytest.raises(ValueError,match='字形'):
            validate_pdf(pdf.tobytes(),[character_inventory(page)])


def test_plain_frame_is_rebuilt_separately_from_its_editable_text(local_app,monkeypatch):
    client,_,_=local_app
    image=Image.new('RGB',(400,200),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,380,180),outline='black',width=2)
    draw.text((60,75),'Ordinary prose',font=ImageFont.load_default(size=20),fill='black')
    _mock_ocr(monkeypatch,[candidate('Ordinary prose',[57,72,230,102])])
    response=client.post('/api/convert/rebuild',files=[('files',('boxed.pdf',_raster_pdf(np.asarray(image)),'application/pdf'))])
    assert response.status_code==200,response.text
    with fitz.open(stream=response.content,filetype='pdf') as pdf:
        assert 'Ordinary prose' in pdf[0].get_text()
        assert not pdf[0].get_images()
        assert pdf[0].get_drawings()


def test_safe_export_identity_preserves_real_family_for_editor_font_matching():
    library=get_library()
    face=library.resolve('BODY','Times New Roman',safe_only=True)
    font=library.export_font_for_key(face.key)
    with TTFont(io.BytesIO(font.buffer)) as embedded:
        assert embedded['name'].getBestFamilyName()==face.family
        assert embedded['name'].getDebugName(6).startswith('PT_')
