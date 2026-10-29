"""Content-preserving measurement and constrained feature-four pagination."""
from collections import Counter
from copy import deepcopy

import pytest

from backend.adaptive_layout import layout_document, measure_block


def text(identifier='body', value='A readable paragraph.', width=180, **style):
    return dict(id=identifier, kind='text', source=dict(page_id='p1', label='text', bbox_pt=[0, 0, width, 20]),
                content=dict(text=value, paragraphs=[dict(id=identifier+'-p', text=value)]),
                style=dict(font_size_pt=11, family='sans-serif', line_height=1.25, **style),
                local_layout={}, quality=dict(state='needs_review', issues=[]))


def image(identifier='image', width=100, height=60):
    return dict(id=identifier, kind='image', source=dict(page_id='p1', label='image', bbox_pt=[0, 0, width, height]),
                content=dict(asset=identifier+'.png', pixel_size=[width, height], aspect_ratio=width/height),
                style={}, local_layout={}, quality=dict(state='ready', issues=[]))


def table(identifier='table', rows=12, rowspan=False):
    cells = []
    for r in range(rows):
        for c in range(2):
            if rowspan and r == 2 and c == 0:
                continue
            cells.append(dict(id=f'{identifier}-r{r}c{c}', row=r, column=c,
                              rowspan=2 if rowspan and r == 1 and c == 0 else 1,
                              colspan=1, text=f'Row {r} column {c} content', header=r == 0,
                              source_bbox=[c*80, r*20, (c+1)*80, (r+1)*20]))
    return dict(id=identifier, kind='table', source=dict(page_id='p1', label='table', bbox_pt=[0, 0, 160, rows*20]),
                content=dict(row_count=rows, column_count=2, cells=cells),
                style=dict(font_size_pt=10, family='sans-serif', line_height=1.2),
                local_layout={}, quality=dict(state='needs_review', issues=[]))


def project(blocks, order=None, **relations):
    return dict(blocks=blocks, source_pages=[dict(id='p1', width_pt=240, height_pt=220)],
                relations=dict(source_order=order or [b['id'] for b in blocks], **relations))


def assert_geometry(result):
    for page in result['pages']:
        boxes = [p['bbox'] for p in result['placements'] if p['page_index'] == page['index']]
        for b in boxes:
            assert 0 <= b[0] < b[2] <= page['width']+.001
            assert 0 <= b[1] < b[3] <= page['height']+.001
        for a, b in zip(boxes, boxes[1:]):
            assert a[3] <= b[1]+.001


def test_local_rewrap_retains_source_and_font_size_with_new_signature():
    block = text(value='Different widths change wrapping without changing the original contents. '*4)
    source = deepcopy(block['source'])
    narrow = measure_block(block, 100)
    wide = measure_block(block, 220)
    assert narrow['height'] > wide['height']
    assert narrow['signature'] != wide['signature']
    assert narrow['font_size'] == wide['font_size'] == 11
    assert block['source'] == source
    assert block['local_layout']['measurement']['signature'] == wide['signature']
    assert block['local_layout']['source_height_is_limit'] is False


def test_old_flow_keeps_its_text_pagination_without_source_height_floor():
    block=text(value='Short paragraph.')
    block['source']['bbox_pt']=[0,0,180,180]
    result=layout_document(project([block]))
    assert result['target_structure']=='single-column-reflow'
    assert len(result['placements'])==1
    assert result['placements'][0]['measurement']['height'] < 30
    assert block['local_layout']['preserve_source_envelope'] is False


def test_text_source_ranges_include_spaces_newlines_and_tabs_exactly_once():
    value = 'First line with spaces.\n\n  Indented\ttext and a verylongunbrokenidentifier.\n尾部中文'
    measured = measure_block(text(value=value), 110)
    assert measured['lines'][0]['start'] == 0
    assert measured['lines'][-1]['end'] == len(value)
    assert all(a['end'] == b['start'] for a, b in zip(measured['lines'], measured['lines'][1:]))
    assert ''.join(value[l['start']:l['end']] for l in measured['lines']) == value


def test_image_preserves_ratio_and_extreme_image_fits_page_without_stretch():
    block = image(width=60, height=600)
    measured = measure_block(block, 40)
    assert measured['height'] == pytest.approx(400)
    result = layout_document(project([block]))
    box = result['placements'][0]['bbox']
    assert (box[2]-box[0])/(box[3]-box[1]) == pytest.approx(.1)
    assert result['issues'][0]['reason'] == 'image-scaled-proportionally-to-page'
    assert_geometry(result)


def test_small_image_is_not_enlarged_to_minimum_text_width():
    block = image(width=18, height=12)
    result = layout_document(project([block]))
    placement = result['placements'][0]
    assert placement['measurement']['width'] == 18
    assert placement['measurement']['height'] == 12
    assert_geometry(result)


def test_long_text_paginates_with_continuous_nonduplicated_ranges():
    value = 'A long text fragment with punctuation, words and spacing. '*100
    block = text(value=value)
    result = layout_document(project([block]))
    assert len(result['pages']) > 2
    ranges = [p['content_range'] for p in result['placements']]
    assert ranges[0]['start'] == 0
    assert ranges[-1]['end'] == len(value)
    assert all(a['end'] == b['start'] for a, b in zip(ranges, ranges[1:]))
    assert ''.join(value[r['start']:r['end']] for r in ranges) == value
    assert {p['block_id'] for p in result['placements']} == {block['id']}
    assert len({p['fragment_id'] for p in result['placements']}) == len(result['placements'])
    assert_geometry(result)


def test_small_blocks_close_old_source_gaps_and_ignore_source_page_breaks():
    a, b = text('a', 'First.'), text('b', 'Second.')
    b['source'].update(page_id='p2', bbox_pt=[0, 900, 180, 930])
    result = layout_document(project([a, b]))
    assert len(result['pages']) == 1
    first, second = result['placements']
    assert second['bbox'][1]-first['bbox'][3] == pytest.approx(6)
    assert_geometry(result)


def test_table_columns_and_rowspan_are_measured_and_never_cut():
    block = table(rows=18, rowspan=True)
    measured = measure_block(block, 190)
    assert 2 not in measured['breaks']
    assert 3 in measured['breaks']
    assert sum(measured['column_widths']) == pytest.approx(190)
    merged = next(c for c in measured['cells'] if c['rowspan'] == 2)
    assert merged['height'] == pytest.approx(sum(measured['row_heights'][1:3]))
    result = layout_document(project([block]))
    assert len(result['pages']) > 1
    actual = Counter(c['id'] for p in result['placements'] for c in p['measurement']['cells']
                     if not c['repeat_header'])
    assert actual == Counter(c['id'] for c in block['content']['cells'])
    for p in result['placements'][1:]:
        assert p['repeat_header'] is True
        assert p['content_range']['row_start'] != 2
        assert any(c['repeat_header'] for c in p['measurement']['cells'])
    assert_geometry(result)


def test_only_explicit_independent_siblings_can_fill_remaining_space():
    # Printable frame is 192 x 172. The first image leaves 66 pt, B needs
    # 100 pt, and C needs 40 pt. Only [B,C] may exchange.
    blocks = [image('a', 192, 100), image('b', 192, 100), image('c', 192, 40)]
    baseline = layout_document(project(deepcopy(blocks)))
    assert baseline['resolved_order'] == ['a', 'b', 'c']
    adjusted = layout_document(project(deepcopy(blocks), reorder_groups=[['b', 'c']]))
    assert adjusted['resolved_order'] == ['a', 'c', 'b']
    assert adjusted['order_changes'][0]['reason'] == 'independent-sibling-fits-remaining-frame'
    assert_geometry(adjusted)
    hard = layout_document(project(deepcopy(blocks), reorder_groups=[['b', 'c']], hard_before=[['b', 'c']]))
    assert hard['resolved_order'] == ['a', 'b', 'c']


def test_reorder_cannot_jump_across_unrelated_or_dependent_content():
    blocks = [image('a', 192, 100), image('b', 192, 100), image('middle', 192, 90), image('c', 192, 40)]
    result = layout_document(project(blocks, reorder_groups=[['b', 'c']]))
    assert result['resolved_order'] == ['a', 'b', 'middle', 'c']


def test_hard_dependency_repair_is_reported_as_order_change():
    result = layout_document(project([text('a'), text('b')], hard_before=[['b', 'a']]))
    assert result['resolved_order'] == ['b', 'a']
    assert result['order_changes'] == [
        dict(block_id='b', source_index=1, target_index=0, reason='hard-dependency-overrides-source-order'),
        dict(block_id='a', source_index=0, target_index=1, reason='hard-dependency-overrides-source-order'),
    ]
    assert_geometry(result)


def test_cycle_and_impossible_single_line_fail_instead_of_paging_forever():
    with pytest.raises(ValueError, match='循环'):
        layout_document(project([text('a'), text('b')], hard_before=[['a', 'b'], ['b', 'a']]))
    with pytest.raises(ValueError, match='一行'):
        layout_document(project([text()]), page_size=(100, 55), margin=24)


def test_tall_indivisible_rowspan_fails_without_cutting_cells():
    block = table(rows=2)
    block['content']['cells'] = [dict(id='merged', row=0, column=0, rowspan=2, colspan=2,
                                     text='Very long merged cell. '*100, header=False)]
    with pytest.raises(ValueError, match='不可拆表格行组'):
        layout_document(project([block]))


def test_measurement_is_recomputed_when_content_or_style_changes():
    block = text(value='Short.')
    before = measure_block(block, 160)
    block['content']['text'] += ' More words.'*30
    after = measure_block(block, 160)
    assert after['signature'] != before['signature']
    assert after['height'] > before['height']
    block['style']['font_size_pt'] = 13
    style = measure_block(block, 160)
    assert style['signature'] != after['signature']
    assert style['font_size'] == 13
