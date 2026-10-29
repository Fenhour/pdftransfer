"""Rules are raster content, not evidence that they own neighboring prose."""
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from backend.rebuild_layout import foreground, image_regions, region_pixels
from backend.scan_layout import horizontal_rules


def _candidate(text, bbox):
    x0, y0, x1, y1 = bbox
    polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return dict(element=dict(id='line-1', type='text', text=text, confidence=.99,
                             height=y1-y0, polygon=polygon), bbox=bbox, quad=polygon)


def _text(draw, position, text, scale):
    font = ImageFont.load_default(size=20*scale)
    point = tuple(round(v*scale) for v in position)
    draw.text(point, text, font=font, fill='black')
    box = draw.textbbox(point, text, font=font)
    return _candidate(text, [(box[0]-scale)/scale, (box[1]-scale)/scale,
                             (box[2]+scale)/scale, (box[3]+scale)/scale])


def _rule(draw, start, end, y, scale, dotted):
    if dotted:
        for x in range(start, end, 5):
            draw.rectangle((x*scale, y*scale, (x+1)*scale, (y+1)*scale), fill='navy')
    else:
        draw.line((start*scale, y*scale, end*scale, y*scale), fill='navy', width=2*scale)


def _raster_visibility(rgb, regions, scale):
    visible = np.zeros(rgb.shape[:2], bool)
    for region in regions:
        x0, y0, x1, y1 = region['pixelBox']
        pixels = region_pixels(rgb, region, scale)
        assert np.array_equal(pixels[:, :, :3], rgb[y0:y1, x0:x1])
        visible[y0:y1, x0:x1] |= pixels[:, :, 3] > 0 if pixels.shape[2] == 4 else True
    return visible


def _owners(text, regions):
    return Counter([c['element']['text'] for c in text] +
                   [line['text'] for region in regions for line in region['retainedText']])


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('side,dotted', [('left', False), ('right', False), ('left', True), ('right', True)])
def test_standalone_leader_preserves_source_pixels_without_owning_adjacent_prose(scale, side, dotted):
    image = Image.new('RGB', (560*scale, 140*scale), 'white')
    draw = ImageDraw.Draw(image)
    candidate = _text(draw, (190, 50), 'Ordinary body line', scale)
    box = candidate['bbox']
    y = round((box[1]+box[3])/2)
    if side == 'left':
        _rule(draw, 30, round(box[0])-2, y, scale, dotted)
    else:
        _rule(draw, round(box[2])+2, 535, y, scale, dotted)
    rgb = np.array(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [candidate], scale)

    assert [c['element']['text'] for c in text] == ['Ordinary body line']
    assert _owners(text, regions) == Counter({'Ordinary body line': 1})
    assert not any(r['retainedText'] for r in regions)
    visible = _raster_visibility(rgb, regions, scale)
    leader_ink = np.all(rgb == [0, 0, 128], axis=2)
    text_ink = np.all(rgb == [0, 0, 0], axis=2)
    assert leader_ink.any() and np.all(visible[leader_ink])
    assert not np.any(visible[text_ink])
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('dotted', [False, True])
def test_short_number_on_leader_beside_substantial_diagram_remains_in_complete_image(scale, dotted):
    image = Image.new('RGB', (460*scale, 300*scale), 'white')
    draw = ImageDraw.Draw(image)
    draw.line([(230*scale, 60*scale), (400*scale, 60*scale),
               (400*scale, 245*scale), (250*scale, 210*scale)],
              fill='navy', width=5*scale)
    draw.ellipse((290*scale, 105*scale, 360*scale, 175*scale), outline='navy', width=4*scale)
    candidate = _text(draw, (80, 140), '42', scale)
    _rule(draw, round(candidate['bbox'][2]), 300, 155, scale, dotted)
    rgb = np.array(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [candidate], scale)

    assert not text
    assert len(regions) == 1
    assert [line['text'] for line in regions[0]['retainedText']] == ['42']
    assert _owners(text, regions) == Counter({'42': 1})
    visible = _raster_visibility(rgb, regions, scale)
    assert np.all(visible[np.any(rgb != 255, axis=2)])
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
def test_long_prose_in_envelope_empty_corner_is_not_a_diagram_label_despite_leader(scale):
    image = Image.new('RGB', (600*scale, 350*scale), 'white')
    draw = ImageDraw.Draw(image)
    draw.line([(40*scale, 35*scale), (540*scale, 35*scale), (440*scale, 120*scale)],
              fill='navy', width=4*scale)
    draw.rectangle((520*scale, 100*scale, 550*scale, 290*scale), fill='navy')
    candidate = _text(draw, (70, 205), 'A separate ordinary paragraph', scale)
    _rule(draw, round(candidate['bbox'][2])+2, 540, 220, scale, dotted=True)
    rgb = np.array(image)
    before = rgb.copy()

    text, regions, _ = image_regions(rgb, [candidate], scale)

    assert [c['element']['text'] for c in text] == ['A separate ordinary paragraph']
    assert len(regions) == 1
    assert not regions[0]['retainedText']
    assert _owners(text, regions) == Counter({'A separate ordinary paragraph': 1})
    visible = _raster_visibility(rgb, regions, scale)
    assert np.all(visible[np.all(rgb == [0, 0, 128], axis=2)])
    assert not np.any(visible[np.all(rgb == [0, 0, 0], axis=2)])
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
def test_real_letter_dots_are_not_classified_as_a_dotted_leader_or_lost(scale):
    image = Image.new('RGB', (330*scale, 120*scale), 'white')
    draw = ImageDraw.Draw(image)
    candidate = _text(draw, (35, 35), 'initial minimum iii', scale)
    rgb = np.array(image)
    before = rgb.copy()
    ink, _ = foreground(rgb)
    rules = horizontal_rules(ink, candidate['element']['height']*scale)

    assert not rules.any()
    text, regions, _ = image_regions(rgb, [candidate], scale)
    assert [c['element']['text'] for c in text] == ['initial minimum iii']
    assert not regions
    assert _owners(text, regions) == Counter({'initial minimum iii': 1})
    assert np.array_equal(rgb, before)
