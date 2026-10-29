"""Fourth-entry HTTP checks for generic layouts, preservation and safety gates.

Only OCR inference is supplied as deterministic evidence. Parsing, ownership,
font fitting, format adapters, PDF writing and validation use the real code.
"""
import io
import json
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
import pytest

from test_app import local_app
from test_rebuild import _mock_ocr, _raster_pdf, candidate, fixture as graphic_fixture
from test_rebuild_frames import _page as framed_prose


def _upload(client, rgb, format='pdf', *, payload=None):
    return client.post('/api/convert/rebuild',
                       files=[('files', ('generic-layout.pdf', _raster_pdf(rgb) if payload is None else payload, 'application/pdf'))],
                       data={'format': format})


def _pdf_layout(pdf):
    report = json.loads(pdf.embfile_get('conversion-report.json'))
    attachment = report['pages'][0]['adapter']['layoutAttachment']
    assert attachment == 'page-0001-layout.json'
    return json.loads(pdf.embfile_get(attachment))


def _two_columns():
    image = Image.new('RGB', (1000, 260), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    candidates = []

    def add(value, x, y):
        draw.text((x, y), value, font=font, fill='black')
        box = list(draw.textbbox((x, y), value, font=font))
        candidates.append(candidate(value, box))
        return box

    box = add('Continuous', 40, 35)
    add('15 minutes', box[2]+10, 35)
    add('Ordinary text continues.', 40, 73)
    add('Another ordinary line.', 40, 111)
    for index, value in enumerate(['Separate right column.', 'Different adjacent content.', 'A final right hand line.']):
        add(value, 600, 35+index*38)
    return np.asarray(image), candidates


@pytest.mark.parametrize('format', ['pdf', 'html', 'md'])
def test_http_rebuild_retains_columns_and_positioned_line_fragments(local_app, monkeypatch, format):
    client, inputs, outputs = local_app
    rgb, candidates = _two_columns()
    original = _raster_pdf(rgb)
    _mock_ocr(monkeypatch, candidates)
    response = _upload(client, rgb, format, payload=original)
    assert response.status_code == 200, response.text
    if format == 'pdf':
        with fitz.open(stream=response.content, filetype='pdf') as pdf:
            model = _pdf_layout(pdf)
            visible = pdf[0].get_text()
            assert not pdf[0].get_images(), 'Ordinary columns must remain visible PDF text'
            assert not any(t['type'] == 3 for t in pdf[0].get_texttrace())
    else:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            model = json.loads(archive.read('model/page-0001.json'))
            visible = archive.read(f'summary/summary.{format}').decode('utf-8')
            assert not any(name.startswith('assets/') for name in archive.namelist())
    # These assertions use both the actual readable artifact and its sidecar;
    # no behavior of a particular external PDF editor is assumed.
    assert all(word in visible for word in ['Continuous', '15', 'minutes', 'Ordinary', 'Separate', 'Different'])
    text = {e['text']: e for e in model['elements'] if e['type'] == 'text'}
    left, continuation, right = (text[value] for value in ['Continuous', '15 minutes', 'Separate right column.'])
    assert left['lineId'] == continuation['lineId']
    assert left['regionId'] != right['regionId']
    assert left['lineId'] != right['lineId']
    assert left['x']+left['width'] < continuation['x'] < right['x']
    assert [left['text'], continuation['text']] == ['Continuous', '15 minutes']
    assert model['textLayout']['semanticsInferred'] is False
    owners = {r['id']: r['ownerId'] for r in model['recognition']}
    assert len(owners) == len(candidates)
    assert all(owners[e['id']] == e['id'] for e in text.values())
    if format == 'pdf':
        saved = json.loads(next(outputs.glob('*/conversion.json')).read_text(encoding='utf-8'))
        assert all(saved['pdfSafety'][stage]['characterInventory'] and
                   saved['pdfSafety'][stage]['referencedGlyphOutlines']
                   for stage in ('beforeSubset', 'reopened'))
    assert next(inputs.glob('*/01_generic-layout.pdf')).read_bytes() == original
    assert not list(outputs.glob('*/rebuild-page-work'))


def test_http_pdf_keeps_graphic_labels_inside_one_intact_image_with_owner_evidence(local_app, monkeypatch):
    client, _, _ = local_app
    rgb, candidates = graphic_fixture()
    _mock_ocr(monkeypatch, candidates)
    response = _upload(client, rgb)
    assert response.status_code == 200, response.text
    with fitz.open(stream=response.content, filetype='pdf') as pdf:
        model = _pdf_layout(pdf)
        images = [e for e in model['elements'] if e['type'] == 'image']
        assert len(images) == len(pdf[0].get_image_info()) == 1
        image = images[0]
        assert image['bbox'][0] <= 80/2 and image['bbox'][2] >= 520/2
        assert image['bbox'][1] <= 130/2 and image['bbox'][3] >= 400/2
        assert 'Independent body' in pdf[0].get_text()
        assert 'Figure label' not in pdf[0].get_text()
        evidence = {r['text']: r for r in model['recognition']}
        assert evidence['Figure label']['ownerId'] == image['id']
        assert evidence['Independent body']['ownerId'] != image['id']
        assert [line['text'] for line in image['retainedText']] == ['Figure label']
        assert image['boundary']['exact'] is False


def test_http_pdf_preserves_uncertain_and_unrecognized_framed_text_without_double_frame(local_app, monkeypatch):
    client, _, _ = local_app
    rgb, candidates = framed_prose(missed=True, touch=True)
    candidates[0]['element']['confidence'] = .42
    # This fixture intentionally retains the real low confidence, unlike the
    # standard helper whose candidates are all accepted at .99.
    monkeypatch.setattr('backend.parser._engine', lambda: lambda _:
                        ([(c['quad'], c['element']['text'], c['element']['confidence']) for c in candidates], None))
    response = _upload(client, rgb)
    assert response.status_code == 200, response.text
    with fitz.open(stream=response.content, filetype='pdf') as pdf:
        model = _pdf_layout(pdf)
        frames = [e for e in model['elements'] if e.get('rebuildVector')]
        assert len(frames) == 1
        assert frames[0]['frameEvidence']['ocrConfidenceRequired'] is False
        strokes = [d for d in pdf[0].get_drawings() if d['color'] is not None and d['fill'] is None]
        assert len(strokes) == 1, 'The frame must be one editable vector outline'
        image_elements = {e['id']: e for e in model['elements'] if e['type'] == 'image'}
        evidence = {r['text']: r for r in model['recognition']}
        uncertain = 'A framed note with ordinary prose.'
        assert evidence[uncertain]['confidence'] == .42
        assert evidence[uncertain]['ownerId'] in image_elements
        assert uncertain not in pdf[0].get_text(), 'Uncertain transcription must not masquerade as recovered text'
        assert 'The original pixels remain available.' in pdf[0].get_text()
        missing = 'An uncertain line does not erase the frame.'
        assert missing not in evidence, 'The unrecognized line really is absent from supplied OCR'

        # Reconstruct only visible retained source pixels. The missed line must
        # survive in raster objects. Unambiguous outline pixels are excluded;
        # a local source intersection with uncertain text stays conservative.
        visible_source = np.zeros(rgb.shape[:2], bool)
        excluded_frame_pixels = 0
        pdf_images = {image[0]: image[1] for image in pdf[0].get_images(full=True)}
        for info in pdf[0].get_image_info(xrefs=True):
            x0, y0, x1, y1 = [round(v*2) for v in info['bbox']]
            crop = np.asarray(Image.open(io.BytesIO(pdf.extract_image(info['xref'])['image'])).convert('RGB'))
            mask_xref = pdf_images[info['xref']]
            assert mask_xref, 'Frame exclusions must remain actual PDF soft masks'
            alpha = np.asarray(Image.open(io.BytesIO(pdf.extract_image(mask_xref)['image'])).convert('L'))
            source = rgb[y0:y1, x0:x1]
            assert crop.shape == source.shape == (*alpha.shape, 3)
            opaque = alpha > 0
            assert np.array_equal(crop[opaque], source[opaque])
            visible_source[y0:y1, x0:x1] |= opaque
            rule = np.all(source == [37, 67, 106], axis=2)
            if np.any(rule):
                yy, xx = np.indices(rule.shape)
                contact = (xx+x0 < 45) & (yy+y0 >= 55) & (yy+y0 <= 125)
                assert np.all(alpha[rule & ~contact] == 0), 'Unambiguous border pixels must not remain duplicated'
                excluded_frame_pixels += int(np.count_nonzero(rule & (alpha == 0)))
        assert excluded_frame_pixels > 0
        font = ImageFont.load_default(size=20)
        missed_box = ImageDraw.Draw(Image.fromarray(rgb)).textbbox((55, 195), missing, font=font)
        x0, y0, x1, y1 = missed_box
        dark = np.any(rgb[y0:y1, x0:x1] < 100, axis=2)
        assert np.count_nonzero(dark) > 100
        assert np.all(visible_source[y0:y1, x0:x1][dark]), 'Every missed glyph pixel must remain visible'
        touch_box = ImageDraw.Draw(Image.fromarray(rgb)).textbbox((32, 75), uncertain, font=font)
        x0, y0, x1, y1 = touch_box
        dark = np.all(rgb[y0:y1, x0:x1] == 0, axis=2)
        assert np.all(visible_source[y0:y1, x0:x1][dark]), 'Low-confidence glyphs touching the frame must remain whole'


def test_http_rebuild_still_blocks_font_and_unicode_gate_failures(local_app, monkeypatch):
    client, _, outputs = local_app
    rgb, candidates = _two_columns()
    _mock_ocr(monkeypatch, candidates)
    with monkeypatch.context() as patch:
        def reject_fonts(_):
            raise ValueError('font preparation rejected the supplied glyphs')
        patch.setattr('backend.reconstruction.prepare_fonts', reject_fonts)
        response = _upload(client, rgb)
        assert response.status_code == 422
        assert 'font preparation rejected' in response.text
    with monkeypatch.context() as patch:
        # The actual PDF validator must reject a mismatch, rather than accept
        # readable/renderable output merely because grouping succeeded.
        patch.setattr('backend.reconstruction.character_inventory', lambda _: {'not-the-visible-content': 1})
        response = _upload(client, rgb)
        assert response.status_code == 422
        assert '映射' in response.text
    assert not list(outputs.glob('*/rebuilt-text.pdf'))
    assert not list(outputs.glob('*/rebuild-page-work'))
