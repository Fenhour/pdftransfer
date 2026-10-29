"""Canvas downloads must survive optional font subsetting failures on Windows."""
import json
import base64
import io

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

from backend.pdf_safety import character_inventory, save_verified, validate_pdf
from backend.font_library import get_library
from test_app import editable_text, image_document, local_app


def test_text_pdf_download_handles_a_single_pixel_ocr_stroke(local_app, image_document):
    client, _, _ = local_app
    document = image_document
    image = Image.new('RGBA', (64, 25), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for box in [(0, 0, 7, 24), (28, 0, 28, 24), (50, 0, 63, 24)]:
        draw.rectangle(box, fill='black')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    # No OCR spaces forces the same beam-alignment path as the failed request.
    element = editable_text('alphaIomega')
    element.update(source='ocr', rotation=0, x=30, y=30, width=128, height=50,
                   appearanceSrc='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
    page = document['pages'][0]
    page.update(width=400, height=180, elements=[element])
    response = client.post(f"/api/documents/{document['id']}/export/pdf-text", json=document)
    assert response.status_code == 200, response.text
    validate_pdf(response.content, [character_inventory(page)])
    with fitz.open(stream=response.content, filetype='pdf') as result:
        assert result[0].get_text().replace(' ', '').strip() == element['text']
        assert all(span['type'] != 3 for span in result[0].get_texttrace())
        attached = json.loads(result.embfile_get('scene-document.json'))
        assert attached['pages'][0]['elements'][0]['text'] == element['text']


@pytest.mark.parametrize('failure', ['locked-file', 'damaged-subset', None])
def test_text_pdf_download_preserves_verified_text_when_subsetting_fails(
        local_app, image_document, monkeypatch, failure):
    client, _, output = local_app
    document = image_document
    page = document['pages'][0]
    page.update(width=500, height=200, elements=[editable_text('Current editable text 2026')])
    original_subset = fitz.Document.subset_fonts
    calls = []

    def subset(pdf, *, fallback):
        calls.append(fallback)
        if failure == 'locked-file':
            raise PermissionError(32, 'oldfont.ttf is in use')
        if failure == 'damaged-subset':
            # A successful call can still destroy content; never download it.
            pdf.delete_page(0)
            return
        original_subset(pdf, fallback=fallback)

    monkeypatch.setattr(fitz.Document, 'subset_fonts', subset)
    response = client.post(f"/api/documents/{document['id']}/export/pdf-text", json=document)
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/pdf'
    assert '-text.pdf' in response.headers['content-disposition']
    assert calls == [False], 'The Windows temporary-file fallback must not run'
    validate_pdf(response.content, [character_inventory(page)])
    with fitz.open(stream=response.content, filetype='pdf') as result:
        assert len(result) == 1
        assert 'Current editable text 2026' in result[0].get_text()
        assert all(span['type'] != 3 for span in result[0].get_texttrace())
        attached = json.loads(result.embfile_get('scene-document.json'))
        assert attached['pages'][0]['elements'][0]['text'] == 'Current editable text 2026'
        assert 'safePdfFonts' not in attached['pages'][0]['elements'][0]
    saved = client.get(f"/api/documents/{document['id']}").json()
    assert saved['pages'][0]['elements'][0]['text'] == 'Current editable text 2026'
    assert 'safePdfFonts' not in saved['pages'][0]['elements'][0]
    summary = output / document['id'] / 'summary' / 'document-text.pdf'
    assert summary.read_bytes() == response.content


def test_text_pdf_export_error_is_actionable_json_and_keeps_saved_edits(
        local_app, image_document, monkeypatch):
    client, _, _ = local_app
    document = image_document
    document['pages'][0]['elements'] = [editable_text('Saved before export')]

    def fail(*args):
        raise RuntimeError('font engine error')

    monkeypatch.setattr('backend.app.write_text_pdf', fail)
    response = client.post(f"/api/documents/{document['id']}/export/pdf-text", json=document)
    assert response.status_code == 422
    assert '文本 PDF 导出失败' in response.json()['detail']
    assert '当前文档已保存' in response.json()['detail']
    saved = client.get(f"/api/documents/{document['id']}").json()
    assert saved['pages'][0]['elements'][0]['text'] == 'Saved before export'


@pytest.mark.parametrize('hidden_style', [{'visible': False}, {'opacity': 0}])
def test_hidden_text_does_not_block_pdf_with_an_unavailable_font(
        local_app, image_document, monkeypatch, hidden_style):
    client, _, _ = local_app
    document = image_document
    shown = editable_text('Visible content')
    hidden = dict(editable_text('UNAVAILABLE GLYPH'), id='hidden', **hidden_style)
    document['pages'][0].update(width=500, height=200, elements=[shown, hidden])
    library = get_library()
    resolve = library.resolve

    def visible_font_only(text, *args, **kwargs):
        if text == hidden['text']:
            raise ValueError('No font covers this hidden text')
        return resolve(text, *args, **kwargs)

    monkeypatch.setattr(library, 'resolve', visible_font_only)
    response = client.post(f"/api/documents/{document['id']}/export/pdf-text", json=document)
    assert response.status_code == 200, response.text
    with fitz.open(stream=response.content, filetype='pdf') as result:
        assert 'Visible content' in result[0].get_text()
        assert hidden['text'] not in result[0].get_text()


def test_native_subset_checks_glyphs_with_a_truncated_font_display_name(tmp_path):
    library = get_library()
    face = library.resolve('Verified glyphs 2026', 'Times New Roman', True, True, safe_only=True)
    if face is None or face.family != 'Times New Roman':
        pytest.skip('Requires installed Times New Roman bold italic')
    with fitz.open() as pdf:
        page = pdf.new_page(width=400, height=150)
        writer = fitz.TextWriter(page.rect)
        writer.append((30, 50), 'Verified glyphs 2026', font=library.export_font_for_key(face.key), fontsize=18)
        writer.write_text(page)
        expected = character_inventory({'elements': [{'type': 'text', 'text': 'Verified glyphs 2026'}]})
        checks = save_verified(pdf, tmp_path / 'subset.pdf', [expected])
    assert checks['afterSubset']['referencedGlyphOutlines']
    assert checks['reopened']['characterInventory']
    assert 'subsetFallback' not in checks
