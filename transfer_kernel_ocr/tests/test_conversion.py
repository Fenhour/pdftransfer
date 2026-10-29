"""Conversion contracts: actual searchable PDF text and intact composite imagery."""
import io
import json
import zipfile

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.exporter import read_asset, write_text_pdf
from backend.scan_layout import partition_text
from test_app import local_app


def scan_fixture():
    image=Image.new('RGB',(600,500),'white')
    draw=ImageDraw.Draw(image)
    font=ImageFont.load_default(size=24)
    draw.text((40,40),'BODY TEXT 123',font=font,fill='black')
    draw.rectangle((80,180,520,420),outline='black',width=4)
    draw.ellipse((120,230,200,310),outline='blue',width=4)
    draw.text((240,245),'DIAGRAM LABEL',font=font,fill='black')
    draw.line((200,270,450,330),fill='red',width=3)
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    return image,buffer.getvalue()


def recognized_scan(_):
    return [([[37,38],[225,38],[225,70],[37,70]],'BODY TEXT 123',.99),
            ([[237,243],[435,243],[435,275],[237,275]],'DIAGRAM LABEL',.99)],None


def raster_pdf(png):
    with fitz.open() as pdf:
        page=pdf.new_page(width=300,height=250)
        page.insert_image(page.rect,stream=png)
        return pdf.tobytes()


def test_scan_to_visible_text_pdf_preserves_mixed_figure(local_app,monkeypatch):
    client,input_root,output_root=local_app
    image,png=scan_fixture()
    monkeypatch.setattr('backend.parser._engine',lambda:recognized_scan)
    original=raster_pdf(png)
    with fitz.open(stream=original,filetype='pdf') as source:
        assert source[0].get_text()==''
    response=client.post('/api/documents',files=[('files',('scan.pdf',original,'application/pdf'))],
                         data={'mode':'text_pdf','ocr':'true'})
    assert response.status_code==200,response.text
    doc=response.json()
    elements=doc['pages'][0]['elements']
    assert [e['text'] for e in elements if e['type']=='text']==['BODY TEXT 123']
    assert not any(e['type']=='region' for e in elements), 'Artwork must not be fragmented into guessed objects'
    preserved=next(e for e in elements if e.get('scanComposite'))
    assert preserved['retainedText'][0]['text']=='DIAGRAM LABEL'
    assert preserved['retainedText'][0]['bbox'][1]>100
    root=output_root/doc['id']
    remaining=Image.open(io.BytesIO(read_asset(preserved['src'],root,doc['id']))).convert('RGB')
    assert np.array_equal(np.asarray(remaining)[175:425,75:525],np.asarray(image)[175:425,75:525])
    assert np.asarray(remaining)[38:72,36:230].min()>240, 'Original body text pixels must be removed'
    exported=client.post(f"/api/documents/{doc['id']}/export/pdf-text",json=doc)
    assert exported.status_code==200,exported.text
    assert '.pdf' in exported.headers['content-disposition']
    with fitz.open(stream=exported.content,filetype='pdf') as result:
        page=result[0]
        assert 'BODY TEXT 123' in page.get_text()
        assert 'DIAGRAM LABEL' not in page.get_text(), 'Mixed graphic label must remain in its image'
        assert all(span['type']!=3 for span in page.get_texttrace()), 'PDF text must be visible, not a hidden overlay'
        assert page.get_images()
        assert 'scene-document.json' in result.embfile_names()
        raster=page.get_pixmap(matrix=fitz.Matrix(2,2))
        actual=np.frombuffer(raster.samples,np.uint8).reshape(raster.height,raster.width,raster.n)[:,:,:3]
        assert np.max(np.abs(actual[180:420,80:520].astype(int)-np.asarray(image)[180:420,80:520].astype(int)))<3
    assert (input_root/doc['id']/'01_scan.pdf').read_bytes()==original


def test_text_pdf_keeps_unicode_geometry_visibility_and_current_edits(tmp_path):
    elements=[{'id':'body','type':'text','text':'中文 Text 42','x':40,'y':35,'width':170,'height':25,
               'fontSize':16,'fontFamily':'Arial','zIndex':1,'rotation':0,'source':'ocr'},
              {'id':'hidden','type':'text','text':'DO NOT EXPORT','x':0,'y':0,'width':100,'height':20,'visible':False,'zIndex':2}]
    doc={'id':'sample','name':'文字与图片','pages':[{'width':300,'height':200,'elements':elements}]}
    output=tmp_path/'text.pdf'
    write_text_pdf(doc,tmp_path,output)
    with fitz.open(output) as pdf:
        assert '中文 Text 42' in pdf[0].get_text()
        assert 'DO NOT EXPORT' not in pdf[0].get_text()
        words=pdf[0].get_text('words')
        assert all(34<=word[1]<61 and word[0]>=39 for word in words)
    elements[0].update(text='Moved 99',x=90,y=70,rotation=90)
    write_text_pdf(doc,tmp_path,tmp_path/'moved.pdf')
    with fitz.open(tmp_path/'moved.pdf') as pdf:
        assert 'Moved 99' in pdf[0].get_text()
        assert '42' not in pdf[0].get_text()
        span=pdf[0].get_text('dict')['blocks'][0]['lines'][0]
        assert abs(span['dir'][0])<.01 and span['dir'][1]>.99


def test_text_pdf_ocr_failure_and_disabled_ocr_are_explicit(local_app,monkeypatch):
    client,_,_=local_app
    _,png=scan_fixture()
    files=[('files',('scan.png',png,'image/png'))]
    disabled=client.post('/api/documents',files=files,data={'mode':'text_pdf','ocr':'false'})
    assert disabled.status_code==422
    def fail():
        raise RuntimeError('test OCR unavailable')
    monkeypatch.setattr('backend.parser._engine',fail)
    failed=client.post('/api/documents',files=files,data={'mode':'text_pdf','ocr':'true'})
    assert failed.status_code==422
    assert 'OCR' in failed.text
    assert client.get('/api/documents').json()==[]


def test_border_around_plain_prose_does_not_turn_it_into_a_figure():
    image=Image.new('RGB',(400,220),'white')
    draw=ImageDraw.Draw(image)
    draw.rectangle((20,20,380,200),outline='black',width=3)
    draw.text((60,80),'Ordinary boxed prose',font=ImageFont.load_default(size=20),fill='black')
    candidate={'element':{'height':28,'text':'Ordinary boxed prose','confidence':.99,
                          'polygon':[[55,75],[300,75],[300,108],[55,108]]},
               'quad':[[55,75],[300,75],[300,108],[55,108]],'bbox':[55,75,300,108]}
    accepted,retained=partition_text(np.asarray(image),[candidate],1)
    assert len(accepted)==1 and not retained


def test_decorative_frame_keeps_many_prose_lines_editable_but_art_stays_in_pixels():
    image=Image.new('RGB',(500,380),'white')
    draw=ImageDraw.Draw(image)
    draw.line([(20,20),(480,20),(480,330),(410,350),(350,330),(280,350),(210,330),(140,350),(20,330),(20,20)],fill='black',width=3)
    candidates=[]
    for index,y in enumerate(range(45,280,35)):
        text=f'Ordinary note line {index}'
        draw.text((60,y),text,font=ImageFont.load_default(size=20),fill='black')
        quad=[[57,y-2],[310,y-2],[310,y+24],[57,y+24]]
        candidates.append({'element':{'height':26,'text':text,'confidence':.99,'polygon':quad},'quad':quad,'bbox':[57,y-2,310,y+24]})
    accepted,retained=partition_text(np.asarray(image),candidates,1)
    assert len(accepted)==len(candidates) and not retained
    draw.ellipse((45,35,390,310),fill='gray',outline='black',width=4)
    for index,y in enumerate(range(45,280,35)):
        draw.text((60,y),f'Ordinary note line {index}',font=ImageFont.load_default(size=20),fill='black')
    accepted,retained=partition_text(np.asarray(image),candidates,1)
    assert retained, 'Text over real interior artwork must not be treated as a plain note frame'


def test_adjacent_paragraph_lines_are_not_background_and_leave_no_ghost(local_app,monkeypatch):
    image=Image.new('RGB',(500,180),'white')
    draw=ImageDraw.Draw(image)
    candidates=[]
    for index,y in enumerate([30,55,80,105]):
        text=f'Ordinary paragraph line {index}'
        draw.text((30,y),text,font=ImageFont.load_default(size=22),fill='black')
        polygon=[[27,y-2],[380,y-2],[380,y+24],[27,y+24]]
        candidates.append({'element':{'height':26,'text':text,'confidence':.99,'polygon':polygon},
                           'quad':polygon,'bbox':[27,y-2,380,y+24]})
    accepted,retained=partition_text(np.asarray(image),candidates,1)
    assert len(accepted)==4 and not retained
    monkeypatch.setattr('backend.parser._engine',lambda:lambda _:
                        ([(c['quad'],c['element']['text'],.99) for c in candidates],None))
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    client,_,output=local_app
    response=client.post('/api/documents',files=[('files',('prose.png',buffer.getvalue(),'image/png'))],
                         data={'mode':'text_pdf','ocr':'true'})
    assert response.status_code==200,response.text
    doc=response.json()
    assert sum(e['type']=='text' for e in doc['pages'][0]['elements'])==4
    preserved=doc['pages'][0]['elements'][0]
    cleared=np.asarray(Image.open(io.BytesIO(read_asset(preserved['src'],output/doc['id'],doc['id']))).convert('RGB'))
    assert cleared[26:132,25:385].min()>240, 'Paper fill must not preserve gray copies of the original glyphs'


def test_partial_ocr_glyph_at_line_boundary_is_retained_in_original_image():
    image=Image.new('RGB',(400,100),'white')
    draw=ImageDraw.Draw(image)
    draw.text((25,30),'B First line',font=ImageFont.load_default(size=30),fill='black')
    quad=[[36,28],[250,28],[250,64],[36,64]]
    candidate={'element':{'height':36,'text':'3 First line','confidence':.99,'polygon':quad},
               'quad':quad,'bbox':[36,28,250,64]}
    accepted,retained=partition_text(np.asarray(image),[candidate],1)
    assert not accepted and retained[0]['reason']=='partial-text-boundary'


def test_overlapping_drop_cap_and_full_line_remain_in_image():
    image=Image.new('RGB',(400,100),'white')
    draw=ImageDraw.Draw(image)
    draw.text((20,30),'B First paragraph',font=ImageFont.load_default(size=25),fill='black')
    candidates=[]
    for text,box in [('B',[18,28,37,58]),('3 First paragraph',[27,27,310,60])]:
        x0,y0,x1,y1=box
        quad=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        candidates.append({'element':{'height':y1-y0,'text':text,'confidence':.99,'polygon':quad},
                           'quad':quad,'bbox':box})
    accepted,retained=partition_text(np.asarray(image),candidates,1)
    assert not accepted and len(retained)==2
    assert all(line['reason']=='overlapping-text' for line in retained)


@pytest.mark.parametrize('format',['html','md'])
def test_browser_reading_conversion_is_complete_and_respects_pages(local_app,format):
    client,input_root,_=local_app
    with fitz.open() as pdf:
        for label in ['FIRST PAGE','SECOND PAGE']:
            pdf.new_page().insert_text((50,80),label)
        original=pdf.tobytes()
    response=client.post('/api/convert/reading',files=[('files',('paper.pdf',original,'application/pdf'))],
                         data={'format':format,'ocr':'false','pages':'2'})
    assert response.status_code==200,response.text
    assert response.headers['content-type']=='application/zip'
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        summary=next(n for n in archive.namelist() if n.endswith(f'/summary/summary.{format}'))
        content=archive.read(summary).decode()
        assert 'SECOND PAGE' in content and 'FIRST PAGE' not in content
        assert archive.read(next(n for n in archive.namelist() if '/source/' in n))==original
        assert not any(f'output_{"md" if format=="html" else "html"}/' in n for n in archive.namelist())
    assert next(input_root.glob('*/01_paper.pdf')).read_bytes()==original


def test_reading_conversion_rejects_nonpdf_format_and_page_errors(local_app):
    client,_,_=local_app
    _,png=scan_fixture()
    assert client.post('/api/convert/reading',files=[('files',('scan.png',png,'image/png'))]).status_code==422
    original=raster_pdf(png)
    files=[('files',('scan.pdf',original,'application/pdf'))]
    for params in [{'format':'docx'},{'format':'html','pages':'9','ocr':'false'}]:
        response=client.post('/api/convert/reading',files=files,data=params)
        assert response.status_code==422,response.text
    response=client.post('/api/convert/reading',files=[('files',('broken.pdf',b'broken','application/pdf'))])
    assert response.status_code==422,response.text


def test_direct_text_pdf_uses_real_parser_and_preserves_source(local_app,monkeypatch):
    client,input_root,output_root=local_app
    _,png=scan_fixture()
    monkeypatch.setattr('backend.parser._engine',lambda:recognized_scan)
    with fitz.open() as source:
        for _ in range(2):
            source.new_page(width=300,height=250).insert_image(fitz.Rect(0,0,300,250),stream=png)
        original=source.tobytes()
    response=client.post('/api/convert/text-pdf',files=[('files',('book.pdf',original,'application/pdf'))],
                         data={'pages':'2'})
    assert response.status_code==200,response.text
    assert response.headers['content-type']=='application/pdf'
    with fitz.open(stream=response.content,filetype='pdf') as result:
        assert len(result)==1
        assert 'BODY TEXT 123' in result[0].get_text()
        assert 'DIAGRAM LABEL' not in result[0].get_text()
        assert result[0].get_images()
        assert all(span['type']!=3 for span in result[0].get_texttrace())
        assert 'scene-document.json' not in result.embfile_names()
        report=json.loads(result.embfile_get('conversion-report.json'))
        assert report['pages'][0]['sourcePage']==2
        assert report['independentTextLines']==report['retainedMixedLines']==1
    assert next(input_root.glob('*/01_book.pdf')).read_bytes()==original
    assert not list(output_root.glob('*/page-work'))
    assert client.get('/api/documents').json()==[]


def test_direct_conversion_exceeds_canvas_page_limit_and_releases_scratch(tmp_path,monkeypatch):
    from backend.text_pdf import convert_text_pdf
    path=tmp_path/'book.pdf'
    with fitz.open() as source:
        for _ in range(101):
            source.new_page(width=200,height=100)
        source.save(path)
    out=tmp_path/'result'
    out.mkdir()
    calls=[]
    def single_page(paths,folder,document_id,name,*,ocr,pages,mode):
        assert paths==[path] and ocr and mode=='text_pdf'
        assert not (folder/'previous-page.bin').exists(), 'Previous page assets must be released'
        (folder/'previous-page.bin').write_bytes(b'page assets')
        calls.append(pages)
        return {'id':document_id,'pages':[{'width':200,'height':100,'elements':[
            {'type':'text','text':f'Physical page {pages}','x':10,'y':10,'width':180,'height':20,'fontSize':12}]}]}
    monkeypatch.setattr('backend.text_pdf.parse_document',single_page)
    destination,report=convert_text_pdf([path],out,'book')
    assert calls==[str(i) for i in range(1,102)]
    assert report['pageCount']==report['independentTextLines']==101
    assert not (out/'page-work').exists()
    with fitz.open(destination) as result:
        assert len(result)==101 and 'Physical page 101' in result[-1].get_text()


def test_direct_conversion_validation_and_no_text_failure_are_explicit(local_app,monkeypatch):
    client,_,output=local_app
    _,png=scan_fixture()
    files=[('files',('scan.pdf',raster_pdf(png),'application/pdf'))]
    for data in [{'ocr':'false'},{'pages':'3'}]:
        response=client.post('/api/convert/text-pdf',files=files,data=data)
        assert response.status_code==422,response.text
    assert client.post('/api/convert/text-pdf',files=[('files',('bad.pdf',b'bad','application/pdf'))]).status_code==422
    assert client.post('/api/convert/text-pdf',files=[('files',('scan.png',png,'image/png'))]).status_code==422
    monkeypatch.setattr('backend.parser._engine',lambda:lambda _:([],None))
    response=client.post('/api/convert/text-pdf',files=files)
    assert response.status_code==422 and '独立' in response.text
    assert not list(output.glob('*/converted-text.pdf'))
    assert not list(output.glob('*/page-work'))
