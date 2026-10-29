"""PDF soft masks must not introduce visible rectangles around raster crops."""
import copy
import io
import shutil
import subprocess

import numpy as np
from PIL import Image
import pymupdf as fitz
import pytest

from backend.exporter import append_text_pages


def _page(paper='#ffffff', *, rebuild=True, src='crop.png'):
    page = dict(id='page', index=0, width=250, height=150,
                elements=[dict(id='image', type='image', x=20.25, y=30.25, width=180, height=50,
                               src='/files/edges/'+src)])
    if rebuild:
        page.update(appearanceStrategy='blank-page-typed-reconstruction', paperColor=paper)
    return page


def _crop(tmp_path, *, artwork=True, mode='RGBA'):
    image = Image.new('RGBA', (360, 100), 'white')
    alpha = image.getchannel('A')
    alpha.paste(0, (1, 0, 359, 7))
    image.putalpha(alpha)
    if artwork:
        image.putpixel((40, 40), (31, 63, 95, 255))
        image.putpixel((41, 40), (47, 79, 111, 128))
    image = image.convert(mode)
    image.save(tmp_path/'crop.png')
    return image


def _samples(pdf, page=0):
    xref, mask, *_ = pdf[page].get_images(full=True)[0]
    rgb = np.asarray(Image.open(io.BytesIO(pdf.extract_image(xref)['image'])).convert('RGB'))
    alpha = (np.asarray(Image.open(io.BytesIO(pdf.extract_image(mask)['image'])).convert('L'))
             if mask else np.full(rgb.shape[:2], 255, np.uint8))
    return xref, rgb, alpha


@pytest.mark.parametrize('mode', ['RGBA', 'LA'])
def test_pdf_rebuild_repairs_only_hidden_rgb_and_preserves_source_assets(tmp_path, mode):
    _crop(tmp_path, mode=mode)
    before = (tmp_path/'crop.png').read_bytes()
    legacy = dict(id='edges', pages=[_page(rebuild=False)])
    rebuilt = dict(id='edges', pages=[_page('#d8c6b4')])
    original = copy.deepcopy(rebuilt)
    with fitz.open() as old, fitz.open() as new:
        append_text_pages(old, legacy, tmp_path)
        append_text_pages(new, rebuilt, tmp_path)
        _, previous, old_alpha = _samples(old)
        _, actual, alpha = _samples(new)
        assert np.array_equal(alpha, old_alpha)
        assert np.array_equal(actual[alpha > 0], previous[alpha > 0])
        assert np.all(actual[alpha == 0] == [216, 198, 180])
        new.save(tmp_path/'verified.pdf')
    with fitz.open(tmp_path/'verified.pdf') as reopened:
        _, actual, alpha = _samples(reopened)
        assert np.all(actual[alpha == 0] == [216, 198, 180])
    assert (tmp_path/'crop.png').read_bytes() == before
    assert rebuilt == original


def test_shared_image_keeps_earlier_page_paper_and_legacy_stream(tmp_path):
    _crop(tmp_path)
    with fitz.open() as pdf:
        for page in [_page(rebuild=False), _page('#fff'), _page('#e0d0c0'), _page('#fff')]:
            append_text_pages(pdf, dict(id='edges', pages=[page]), tmp_path)
        samples = [_samples(pdf, page) for page in range(4)]
        assert samples[0][0] != samples[1][0], 'A cached legacy XObject must be isolated before changing hidden RGB'
        assert samples[1][0] != samples[2][0], 'Different paper colors must not share a repaired XObject'
        for (_, rgb, alpha), expected in zip(samples, [(0, 0, 0), (255, 255, 255), (224, 208, 192), (255, 255, 255)]):
            assert np.all(rgb[alpha == 0] == expected)
        assert np.array_equal(samples[1][1][samples[1][2] > 0], samples[0][1][samples[0][2] > 0])
        pdf.save(tmp_path/'mixed-papers.pdf')
    with fitz.open(tmp_path/'mixed-papers.pdf') as reopened:
        for page, expected in [(0, (0, 0, 0)), (1, (255, 255, 255)), (2, (224, 208, 192)), (3, (255, 255, 255))]:
            _, rgb, alpha = _samples(reopened, page)
            assert np.all(rgb[alpha == 0] == expected)


def test_opaque_images_keep_the_original_encoded_samples(tmp_path):
    _crop(tmp_path, mode='RGB')
    with fitz.open() as old, fitz.open() as new:
        append_text_pages(old, dict(id='edges', pages=[_page(rebuild=False)]), tmp_path)
        append_text_pages(new, dict(id='edges', pages=[_page()]), tmp_path)
        assert np.array_equal(_samples(old)[1], _samples(new)[1])
        assert np.array_equal(_samples(old)[2], _samples(new)[2])


def test_poppler_renders_no_rectangle_from_completely_transparent_samples(tmp_path):
    poppler = shutil.which('pdftoppm')
    if not poppler:
        pytest.skip('Poppler is unavailable; encoded RGB/alpha regressions still run')
    _crop(tmp_path, artwork=False)
    arrays = []
    for mode in ['legacy', 'rebuild']:
        path = tmp_path/(mode+'.pdf')
        with fitz.open() as pdf:
            append_text_pages(pdf, dict(id='edges', pages=[_page(rebuild=mode == 'rebuild')]), tmp_path)
            pdf.save(path)
        prefix = tmp_path/mode
        result = subprocess.run([poppler, '-scale-to', '700', '-png', '-singlefile', str(path), str(prefix)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        arrays.append(np.asarray(Image.open(prefix.with_suffix('.png')).convert('RGB')))
    assert np.min(arrays[0]) < 245, 'The negative control must reproduce the gray image edge'
    assert np.min(arrays[1]) == 255, 'An invisible exclusion must render as uninterrupted white paper'
