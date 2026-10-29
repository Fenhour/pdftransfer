"""Frame geometry is independent from OCR confidence and text ownership."""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.rebuild_frames import frame_rule_mask, text_frames
from backend.rebuild_layout import image_regions, region_pixels


def _candidate(text, box, confidence=.99, rotation=0):
    x0, y0, x1, y1 = box
    quad = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return dict(element=dict(id=f'text-{y0}-{x0}', text=text, height=y1-y0,
                             polygon=quad, confidence=confidence, rotation=rotation),
                bbox=list(box), quad=quad)


def _page(*, scale=1, missed=False, watermark=False, touch=False):
    image = Image.new('RGB', (600, 420), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    draw.rectangle((30, 30, 570, 390), outline='#25436a', width=3)
    candidates = []
    for index, line in enumerate(['A framed note with ordinary prose.',
                                  'Recognition confidence varies by line.',
                                  'An uncertain line does not erase the frame.',
                                  'The original pixels remain available.']):
        position = (32 if touch and index == 0 else 55, 75+index*60)
        draw.text(position, line, font=font, fill='black')
        bbox = list(draw.textbbox(position, line, font=font))
        if not missed or index != 2:
            candidates.append(_candidate(line, bbox, confidence=.42 if index == 1 else .99))
    if watermark:
        # A recognized watermark is still a text candidate, not artwork.
        position = (310, 330)
        draw.text(position, 'ARCHIVE', font=font, fill='#b0b0b0')
        candidates.append(_candidate('ARCHIVE', list(draw.textbbox(position, 'ARCHIVE', font=font)), .4, 15))
    rgb = np.asarray(image)
    if scale != 1:
        rgb = np.asarray(image.resize((600*scale, 420*scale), Image.Resampling.NEAREST))
    return rgb, candidates


@pytest.mark.parametrize('scale', [1, 2])
def test_complete_frame_does_not_require_every_ocr_line_to_be_confident(scale):
    rgb, candidates = _page(scale=scale)
    before = rgb.copy()
    frames = text_frames(rgb, candidates, scale)
    assert len(frames) == 1
    frame = frames[0]
    assert frame['contentType'] == 'decoration'
    assert frame['templateId'] == 'decoration.rect.v1'
    assert frame['reviewRequired']
    assert frame['candidateIndices'] == list(range(4))
    assert frame['frameEvidence']['ocrConfidenceRequired'] is False
    assert len(frame['ruleSegments']) == 4
    assert frame['observedStyle']['borderColor'] == '#25436a'
    assert frame['observedStyle']['borderWidth'] == pytest.approx(3, abs=.5)
    assert frame['columns'] == pytest.approx([31, 569], abs=.6)
    assert frame['rows'] == pytest.approx([31, 389], abs=.6)
    assert np.array_equal(rgb, before)


@pytest.mark.parametrize('scale', [1, 2])
def test_missing_ocr_line_stays_as_unexplained_source_content(scale):
    rgb, candidates = _page(scale=scale, missed=True)
    frames = text_frames(rgb, candidates, scale)
    assert len(frames) == 1
    assert frames[0]['frameEvidence']['unexplainedComponents'] > 0
    mask = frame_rule_mask(rgb.shape, frames, scale)
    # The helper only masks the outline; it must not blank the missed line.
    assert not mask[180*scale:230*scale, 55*scale:500*scale].any()
    assert mask[30*scale:33*scale, 40*scale:550*scale].all()


def test_recognized_watermark_does_not_disqualify_otherwise_plain_frame():
    rgb, candidates = _page(watermark=True)
    frames = text_frames(rgb, candidates, 1)
    assert len(frames) == 1
    assert 4 in frames[0]['candidateIndices']


def test_known_text_crossing_frame_is_coverage_without_forced_containment():
    rgb, candidates = _page()
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=22)
    position = (300, 382)
    draw.text(position, 'Reference watermark', font=font, fill='#408f8f')
    candidates.append(_candidate('Reference watermark', list(draw.textbbox(position, 'Reference watermark', font=font))))
    frames = text_frames(np.asarray(image), candidates, 1)
    assert len(frames) == 1
    assert 4 not in frames[0]['candidateIndices']


def test_dotted_leaders_do_not_turn_a_prose_frame_into_a_diagram():
    rgb, candidates = _page()
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    # These marks were not recognized by OCR; keep them as original pixels.
    for y in (118, 178, 238, 298):
        for x in range(250, 540, 7):
            draw.ellipse((x, y, x+1, y+1), fill='black')
    frames = text_frames(np.asarray(image), candidates, 1)
    assert len(frames) == 1
    assert not frame_rule_mask(image.size[::-1], frames, 1)[110:305, 240:550].any()


def test_text_touching_outline_is_not_required_to_fit_a_table_cell_exactly():
    rgb, candidates = _page(touch=True)
    assert len(text_frames(rgb, candidates, 1)) == 1


def test_incomplete_frame_is_not_invented_as_a_complete_rectangle():
    rgb, candidates = _page()
    rgb = rgb.copy()
    rgb[30:34, 240:270] = 255
    assert text_frames(rgb, candidates, 1) == []


@pytest.mark.parametrize('art', ['map', 'photo', 'table', 'internal_rule'])
def test_artwork_and_grid_inside_outer_frame_are_preserved_as_composites(art):
    rgb, candidates = _page()
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    # Keep text aligned like prose: alignment alone must not release a figure.
    if art == 'map':
        draw.ellipse((430, 310, 520, 370), outline='black', width=3)
        draw.line((400, 350, 470, 315, 540, 375), fill='black', width=3)
    elif art == 'photo':
        rng = np.random.default_rng(42)
        image.paste(Image.fromarray(rng.integers(0, 255, (65, 120, 3), dtype=np.uint8)), (400, 310))
    elif art == 'table':
        for x in (30, 210, 390, 570):
            draw.line((x, 30, x, 390), fill='black', width=2)
        for y in (30, 150, 270, 390):
            draw.line((30, y, 570, y), fill='black', width=2)
    else:
        draw.line((100, 340, 500, 340), fill='black', width=2)
    assert text_frames(np.asarray(image), candidates, 1) == []


def test_fragmentary_short_labels_are_not_enough_to_call_a_map_prose():
    image = Image.new('RGB', (400, 220), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 380, 200), outline='black', width=2)
    draw.text((70, 70), 'A', fill='black')
    draw.text((290, 140), 'B', fill='black')
    candidates = [_candidate('A', [68, 68, 85, 85]), _candidate('B', [288, 138, 305, 155])]
    assert text_frames(np.asarray(image), candidates, 1) == []


def test_already_accepted_table_blocks_frame_detection():
    rgb, candidates = _page()
    assert text_frames(rgb, candidates, 1, excluded_boxes=[[30, 30, 570, 390]]) == []


def test_rule_mask_has_no_interior_filled_rectangles_and_supports_analysis_scale():
    rgb, candidates = _page(scale=2)
    frames = text_frames(rgb, candidates, 2)
    assert frames
    full = frame_rule_mask(rgb.shape[:2], frames, 2)
    small = frame_rule_mask((420, 600), frames, 1)
    assert full.shape == (840, 1200)
    assert small.shape == (420, 600)
    assert full[60:65, 100:1000].any()
    assert small[30:34, 50:500].any()
    assert not full[80:760, 80:1120].any()
    assert not small[40:380, 40:560].any()


def test_large_page_analysis_is_bounded_and_preserves_frame_geometry():
    rgb, candidates = _page()
    large = np.asarray(Image.fromarray(rgb).resize((2400, 1680), Image.Resampling.BICUBIC))
    frames = text_frames(large, candidates, 4)
    assert len(frames) == 1
    assert frames[0]['columns'] == pytest.approx([31, 569], abs=1)
    assert frames[0]['rows'] == pytest.approx([31, 389], abs=1)
    # At the original scale, a small explicit margin includes resampled edges.
    mask = frame_rule_mask(large.shape, frames, 4, margin=1)
    assert mask[120:132, 160:2200].any()
    assert not mask[150:1500, 150:2250].any()


def test_no_text_evidence_does_not_guess_a_bare_rectangular_illustration():
    rgb, _ = _page()
    assert text_frames(rgb, [], 1) == []


@pytest.mark.parametrize('scale', [1, 2])
@pytest.mark.parametrize('missing', [False, True])
def test_same_color_unknown_or_uncertain_glyph_touching_frame_remains_visible(scale, missing):
    rgb, candidates = _page(touch=True, scale=scale)
    rgb = rgb.copy()
    # Make the frame and lettering identical in color. Pixel color alone
    # cannot tell their intersection apart, so that contact must stay raster.
    rgb[np.all(rgb == [37, 67, 106], axis=2)] = 0
    glyphs = Image.new('L', (600, 420))
    ImageDraw.Draw(glyphs).text((32, 75), candidates[0]['element']['text'],
                               font=ImageFont.load_default(size=20), fill=255)
    truth = np.asarray(glyphs.resize((600*scale, 420*scale), Image.Resampling.NEAREST)) > 0
    if missing:
        candidates = candidates[1:]
    else:
        candidates[0]['element']['confidence'] = .42
    before = rgb.copy()
    frames = text_frames(rgb, candidates, scale)
    assert len(frames) == 1
    mask = frame_rule_mask(rgb.shape, frames, scale, margin=1)
    # Search-band padding must never erase blank paper or glyph protrusions.
    assert not mask[np.all(rgb == 255, axis=2)].any()
    assert not mask[truth].any()
    _, regions, _ = image_regions(rgb, candidates, scale, ignored_mask=mask)
    visible = np.zeros(rgb.shape[:2], bool)
    for region in regions:
        pixels = region_pixels(rgb, region, scale, excluded_mask=mask)
        x0, y0, x1, y1 = region['pixelBox']
        visible[y0:y1, x0:x1] |= pixels[:, :, 3] > 0 if pixels.shape[2] == 4 else True
    assert np.all(visible[truth]), 'Every original glyph pixel, including the frame intersection, must survive'
    assert mask[40*scale:65*scale, 30*scale:33*scale].all(), 'Unambiguous frame support is still separated'
    assert np.array_equal(rgb, before)
    # Source-resolution evidence allocates only four edge strips, not four
    # full-page masks; it also remains plain serializable geometry.
    evidence_pixels = sum((p['pixelBox'][2]-p['pixelBox'][0])*(p['pixelBox'][3]-p['pixelBox'][1])
                          for p in frames[0]['ruleSupport']['patches'])
    assert evidence_pixels < rgb.shape[0]*rgb.shape[1]*.15
