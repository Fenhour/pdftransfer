"""Synthetic boundary truth: whitespace is not evidence of image ownership.

These tests use known local pixels and supplied OCR boxes. They exercise region
geometry without reading private PDFs, running OCR, or guessing semantic labels.
"""
from collections import Counter
import io
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from backend.rebuild_layout import _merge_regions, image_regions, region_pixels


def _candidate(text, box):
    x0, y0, x1, y1 = box
    quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        'element': {
            'id': text, 'type': 'text', 'text': text, 'height': y1-y0,
            'polygon': quad, 'confidence': .99,
        },
        'quad': quad,
        'bbox': list(box),
    }


def _visible_owners(text, regions):
    return Counter(
        [candidate['element']['text'] for candidate in text]
        + [line['text'] for region in regions for line in region['retainedText']]
    )


def _scaled_image(size, scale):
    return Image.new('RGB', tuple(round(length*scale) for length in size), 'white')


def _draw_rectangle(draw, box, scale, **kwargs):
    draw.rectangle(tuple(round(value*scale) for value in box), **kwargs)


def _corner_fixture(scale):
    image = _scaled_image((560, 350), scale)
    draw = ImageDraw.Draw(image)
    # Two disconnected graphics have intersecting bounding envelopes. Neither
    # component encloses the lower-left prose. Merging their envelopes cannot
    # revoke independent ownership already established before that merge.
    draw.line([(65*scale, 35*scale), (480*scale, 35*scale), (400*scale, 130*scale)],
              fill='navy', width=4*scale)
    _draw_rectangle(draw, (450, 105, 480, 285), scale, fill='navy')
    draw.text((80*scale, 210*scale), 'Independent corner',
              font=ImageFont.load_default(size=18*scale), fill='black')
    return np.asarray(image), _candidate('Independent corner', [77, 207, 310, 235])


def _assert_corner_alpha(rgb, rgba, pixel_box, candidate, scale):
    x0, y0, x1, y1 = pixel_box
    source_crop = rgb[y0:y1, x0:x1]
    assert rgba.shape == (*source_crop.shape[:2], 4)
    assert np.array_equal(rgba[:, :, :3], source_crop)
    # The fixture's known axis-aligned polygon is transparent in its entirety.
    bx0, by0, bx1, by1 = [round(value*scale) for value in candidate['bbox']]
    excluded = np.zeros(source_crop.shape[:2], dtype=bool)
    excluded[by0-y0:by1-y0+1, bx0-x0:bx1-x0+1] = True
    assert np.all(rgba[:, :, 3][excluded] == 0)
    assert np.all(rgba[:, :, 3][~excluded] == 255)
    # Both graphics remain fully opaque; only prose is excluded.
    artwork = np.all(source_crop == [0, 0, 128], axis=2)
    assert np.count_nonzero(artwork) > 0
    assert np.all(rgba[:, :, 3][artwork] == 255)


@pytest.mark.parametrize('scale', [1, 2])
def test_dilation_margin_does_not_capture_next_heading_and_source_is_immutable(scale):
    image = _scaled_image((360, 260), scale)
    draw = ImageDraw.Draw(image)
    # Four original white rows separate artwork from the following heading.
    # Dilation can bridge the gap for analysis, but cannot establish ownership.
    _draw_rectangle(draw, (60, 35, 240, 150), scale, fill='navy')
    draw.text((60*scale, 155*scale), 'Next heading',
              font=ImageFont.load_default(size=18*scale), fill='black')
    candidates = [_candidate('Next heading', [58, 155, 215, 177])]
    rgb = np.asarray(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, candidates, scale)

    assert [line['element']['text'] for line in text] == ['Next heading']
    assert len(regions) == 1
    assert regions[0]['bbox'][3] <= candidates[0]['bbox'][1]
    assert regions[0]['retainedText'] == []
    assert _visible_owners(text, regions) == Counter({'Next heading': 1})
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale, size', [(1, (600, 400)), (1.5, (2100, 1350))])
def test_faint_one_pixel_artwork_edge_survives_analysis_crop(scale, size):
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = [round(value*scale) for value in (100, 80, 260, 210)]
    # This one-source-pixel antialias fringe is lighter than the foreground
    # threshold. A crop based only on detected ink would silently trim it.
    draw.rectangle((x0-1, y0-1, x1+1, y1+1), fill=(245, 245, 245))
    draw.rectangle((x0, y0, x1, y1), fill='navy')
    rgb = np.asarray(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [], scale)

    assert text == []
    assert len(regions) == 1
    left, top, right, bottom = regions[0]['pixelBox']
    assert left <= x0-1 and top <= y0-1
    assert right >= x1+2 and bottom >= y1+2
    crop = region_pixels(rgb, regions[0], scale)
    assert np.array_equal(crop, rgb[top:bottom, left:right])
    gray_count = np.count_nonzero(np.all(rgb == [245, 245, 245], axis=2))
    assert gray_count > 0
    assert np.count_nonzero(np.all(crop == [245, 245, 245], axis=2)) == gray_count
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('boxes', [
    [[20, 20, 100, 80], [20, 500, 100, 560]],
    [[20, 20, 80, 100], [500, 20, 560, 100]],
])
def test_distant_aligned_unframed_components_are_not_unconditionally_merged(boxes):
    # Forty-two line heights of whitespace are not evidence of one figure.
    merged = _merge_regions(boxes, typical=10, barriers=[])
    assert sorted(merged) == sorted(boxes)


@pytest.mark.parametrize('boxes, expected', [
    ([[20, 20, 100, 80], [20, 88, 100, 148]], [20, 20, 100, 148]),
    ([[20, 20, 80, 100], [88, 20, 148, 100]], [20, 20, 148, 100]),
])
def test_nearby_aligned_parts_can_still_form_a_reviewable_whole(boxes, expected):
    assert _merge_regions(boxes, typical=10, barriers=[]) == [expected]


@pytest.mark.parametrize('scale', [1, 2])
def test_distant_unframed_artwork_stays_separate_in_image_regions(scale):
    image = _scaled_image((320, 600), scale)
    draw = ImageDraw.Draw(image)
    _draw_rectangle(draw, (40, 35, 180, 115), scale, fill='navy')
    _draw_rectangle(draw, (40, 480, 180, 560), scale, fill='darkred')
    rgb = np.asarray(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [], scale)

    assert text == []
    assert len(regions) == 2
    assert regions[0]['bbox'][3] < regions[1]['bbox'][1]
    for region in regions:
        assert region['pixelBox'] == [round(value*scale) for value in region['bbox']]
        assert region['reviewRequired'] is True
        assert region['boundary']['exact'] is False
    assert np.array_equal(rgb, before)


def test_independent_text_in_corridor_blocks_nearby_region_merge():
    boxes = [[20, 20, 180, 80], [20, 100, 180, 160]]
    paragraph = [25, 84, 170, 96]
    assert _merge_regions(boxes, typical=10, barriers=[paragraph]) == boxes


@pytest.mark.parametrize('scale', [1, 2])
def test_independent_middle_paragraph_has_exactly_one_visible_owner(scale):
    image = _scaled_image((360, 450), scale)
    draw = ImageDraw.Draw(image)
    _draw_rectangle(draw, (40, 35, 240, 130), scale, fill='navy')
    _draw_rectangle(draw, (40, 310, 240, 410), scale, fill='darkred')
    draw.text((45*scale, 210*scale), 'Independent body',
              font=ImageFont.load_default(size=18*scale), fill='black')
    candidates = [_candidate('Independent body', [42, 207, 235, 235])]

    text, regions, _ = image_regions(np.asarray(image), candidates, scale)

    assert [line['element']['text'] for line in text] == ['Independent body']
    assert len(regions) == 2
    assert _visible_owners(text, regions) == Counter({'Independent body': 1})
    assert all(not region['retainedText'] for region in regions)
    assert regions[0]['bbox'][3] < candidates[0]['bbox'][1]
    assert regions[1]['bbox'][1] > candidates[0]['bbox'][3]


@pytest.mark.parametrize('scale', [1, 2])
def test_independent_text_in_merged_envelope_empty_corner_stays_independent(scale):
    rgb, candidate = _corner_fixture(scale)
    candidates = [candidate]
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, candidates, scale)

    assert [line['element']['text'] for line in text] == ['Independent corner']
    assert len(regions) == 1
    assert regions[0]['bbox'][1] < candidates[0]['bbox'][1] < regions[0]['bbox'][3]
    assert regions[0]['retainedText'] == []
    assert _visible_owners(text, regions) == Counter({'Independent corner': 1})
    region = regions[0]
    assert region['boundary']['clip'] == 'alpha-text-exclusion'
    assert region['excludedText'] == [{
        'textId': candidate['element']['id'],
        'bbox': candidate['bbox'],
        'polygon': candidate['quad'],
    }]
    rgba = region_pixels(rgb, region, scale)
    _assert_corner_alpha(rgb, rgba, region['pixelBox'], candidate, scale)
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
def test_single_open_figure_envelope_does_not_claim_its_inner_label_is_body(scale):
    image = _scaled_image((560, 350), scale)
    draw = ImageDraw.Draw(image)
    # A single open shape can still have internal labels. Empty pixels alone
    # do not establish semantic separation, so this ambiguous line stays art.
    _draw_rectangle(draw, (65, 35, 480, 130), scale, fill='navy')
    _draw_rectangle(draw, (430, 100, 480, 285), scale, fill='navy')
    draw.text((80*scale, 210*scale), 'Possible figure label',
              font=ImageFont.load_default(size=18*scale), fill='black')
    candidate = _candidate('Possible figure label', [77, 207, 310, 235])
    rgb = np.asarray(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [candidate], scale)

    assert text == []
    assert len(regions) == 1
    assert [line['text'] for line in regions[0]['retainedText']] == ['Possible figure label']
    assert regions[0]['excludedText'] == []
    assert _visible_owners(text, regions) == Counter({'Possible figure label': 1})
    x0, y0, x1, y1 = regions[0]['pixelBox']
    assert np.array_equal(region_pixels(rgb, regions[0], scale), rgb[y0:y1, x0:x1])
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
def test_parser_and_html_preserve_real_alpha_asset_and_one_independent_corner(tmp_path, scale):
    import pymupdf as fitz

    from backend.exporter import append_text_pages, read_asset
    from backend.parser import _compose_rebuild_page
    from backend.rebuild_export import export_page, finalize_html

    rgb, candidate = _corner_fixture(scale)
    before = rgb.copy()
    candidate['element'].update(
        fontFamily='Arial', fontSize=18, color='#000000', fontWeight='normal',
        fontStyle='normal', lineHeight=1, rotation=0, zIndex=10, visible=True,
    )
    scratch = tmp_path/'scratch'
    for folder in ('images', 'text', 'pages'):
        (scratch/folder).mkdir(parents=True)
    document_id = 'region-boundary-fixture'
    page = _compose_rebuild_page(
        rgb, 560, 350, scale, [candidate], scratch, document_id, 0, 0,
        'synthetic.png', ocr_enabled=False,
    )
    scene = {'id': document_id, 'pages': [page]}
    images = [element for element in page['elements'] if element['type'] == 'image']
    text = [element for element in page['elements'] if element['type'] == 'text']
    assert len(images) == len(text) == 1
    assert text[0]['text'] == 'Independent corner'
    assert images[0]['retainedText'] == []
    assert images[0]['excludedText'][0]['textId'] == text[0]['id']
    pixel_box = [round(value*scale) for value in images[0]['bbox']]
    with Image.open(io.BytesIO(read_asset(images[0]['src'], scratch, document_id))) as asset:
        assert asset.mode == 'RGBA'
        _assert_corner_alpha(rgb, np.asarray(asset), pixel_box, candidate, scale)

    tree = tmp_path/'html'
    export_page(scene, scratch, tree, 1, 'html')
    finalize_html(tree)
    html = (tree/'summary'/'summary.html').read_text(encoding='utf-8')
    assert html.count('class="object text"') == 1
    assert html.count('<figure ') == 1
    assert html.count('>Independent corner</p>') == 1
    assert '<details class="transcript"' not in html
    exported = json.loads((tree/'model'/'page-0001.json').read_text(encoding='utf-8'))
    exported_image = next(element for element in exported['elements'] if element['type'] == 'image')
    asset_path = (tree/'summary'/exported_image['src']).resolve()
    with Image.open(asset_path) as asset:
        assert asset.mode == 'RGBA'
        _assert_corner_alpha(rgb, np.asarray(asset), pixel_box, candidate, scale)
    assert not any('appearanceSrc' in element for element in exported['elements'])

    # Exercise the PDF consumer with the image alone: a clipped source glyph
    # must leave white paper, and the PDF must retain a real image soft mask.
    image_only = {'id': document_id, 'pages': [dict(page, elements=images)]}
    with fitz.open() as pdf:
        assert append_text_pages(pdf, image_only, scratch) == 0
        payload = pdf.tobytes()
    with fitz.open(stream=payload, filetype='pdf') as pdf:
        assert not pdf[0].get_text().strip()
        pdf_images = pdf[0].get_images(full=True)
        assert len(pdf_images) == 1 and pdf_images[0][1] > 0
        pixmap = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        rendered = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n,
        )
        bx0, by0, bx1, by1 = candidate['bbox']
        original_text = rgb[round(by0*scale):round(by1*scale), round(bx0*scale):round(bx1*scale)]
        assert np.any(original_text < 100), 'The original crop must actually contain dark text'
        cleared = rendered[(by0+2)*2:(by1-2)*2, (bx0+2)*2:(bx1-2)*2]
        assert cleared.size and np.all(cleared == 255)
        assert np.max(np.abs(rendered[240*2, 460*2].astype(int)-[0, 0, 128])) <= 1
    assert np.array_equal(rgb, before)


def test_real_intersection_and_enclosing_frame_are_not_split_by_whitespace_rules():
    overlapping = [[10, 10, 100, 100], [70, 80, 140, 150]]
    assert _merge_regions(overlapping, typical=10, barriers=[[80, 85, 90, 95]]) == [
        [10, 10, 140, 150]
    ]
    assert _merge_regions([[10, 10, 300, 300], [220, 220, 250, 250]],
                          typical=10, barriers=[]) == [[10, 10, 300, 300]]


def test_text_over_actual_artwork_is_retained_once_with_original_pixels():
    image = Image.new('RGB', (360, 260), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 320, 220), fill='navy')
    draw.text((95, 110), 'Inside figure', font=ImageFont.load_default(size=20), fill='white')
    candidates = [_candidate('Inside figure', [92, 107, 245, 135])]
    rgb = np.asarray(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, candidates, 1)

    assert text == []
    assert len(regions) == 1
    assert [line['text'] for line in regions[0]['retainedText']] == ['Inside figure']
    assert _visible_owners(text, regions) == Counter({'Inside figure': 1})
    assert regions[0]['contentType'] == 'mixed'
    assert regions[0]['excludedText'] == []
    x0, y0, x1, y1 = regions[0]['pixelBox']
    assert np.array_equal(region_pixels(rgb, regions[0], 1), rgb[y0:y1, x0:x1])
    assert np.array_equal(rgb, before)


def test_label_enclosed_by_a_figure_frame_is_retained_even_on_white_paper():
    image = Image.new('RGB', (420, 300), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((35, 35, 385, 265), outline='black', width=3)
    draw.ellipse((60, 65, 130, 135), outline='navy', width=4)
    draw.text((160, 180), 'Figure label', font=ImageFont.load_default(size=18), fill='black')
    candidates = [_candidate('Figure label', [157, 177, 300, 205])]

    text, regions, _ = image_regions(np.asarray(image), candidates, 1)

    assert text == []
    assert len(regions) == 1
    assert [line['text'] for line in regions[0]['retainedText']] == ['Figure label']
    assert _visible_owners(text, regions) == Counter({'Figure label': 1})
