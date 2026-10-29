"""Proven cell geometry preserves source line anchors without guessing text."""
from copy import deepcopy

import pytest

from backend.adaptive_layout import _table_measurement
from backend.adaptive_export import _validate_lines, _validate_source_lines
from backend.source_table import enrich_table_content, measure_source_table


def table():
    return dict(id='table', kind='table', source=dict(page_id='p', label='table',
        bbox_px=[20, 40, 420, 240], bbox_pt=[10, 20, 210, 120],
        ocr_lines=[dict(text='Heading', bbox_px=[60, 50, 160, 70], line_id='a'),
                   dict(text='Body text', bbox_px=[35, 90, 185, 110], line_id='b'),
                   dict(text='Right', bbox_px=[300, 70, 360, 90], line_id='c')]),
        content=dict(row_count=1, column_count=2, cells=[
            dict(id='left', row=0, column=0, rowspan=1, colspan=1,
                 text='Heading Body text', source_bbox=[10,20,110,120]),
            dict(id='right', row=0, column=1, rowspan=1, colspan=1,
                 text='Right', source_bbox=[110,20,210,120])]),
        style=dict(font_size_pt=9, family='sans-serif', line_height=1.25))


def enriched():
    block = table()
    enrich_table_content(block['content'], block['source'], dict(geometryStatus='model-cell-boxes-aligned'))
    return block


def test_equivalent_cell_keeps_title_line_and_unique_source_anchors():
    block = enriched()
    left, right = block['content']['cells']
    assert left['text'] == 'Heading\nBody text'
    assert left['original_html_text'] == 'Heading Body text'
    assert [f['text'] for f in left['source_layout']['fragments']] == ['Heading', 'Body text']
    assert [f['text'] for f in right['source_layout']['fragments']] == ['Right']
    assert block['content']['source_layout_diagnostics'] == []


def test_mismatched_ocr_preserves_model_text_once_with_review_reason():
    block = table()
    block['source']['ocr_lines'][0]['text'] = 'Changed'
    enrich_table_content(block['content'], block['source'])
    left = block['content']['cells'][0]
    assert left['text'] == 'Heading Body text'
    assert 'source_layout' not in left
    assert {'reason':'cell-ocr-html-content-mismatch', 'cell_id':'left'} in block['content']['source_layout_diagnostics']


def test_multiline_ocr_record_is_retained_as_model_text_without_anchors():
    block = table()
    block['source']['ocr_lines'][0]['text'] = 'Heading\n'
    enrich_table_content(block['content'], block['source'])
    assert block['content']['cells'][0]['text'] == 'Heading Body text'
    assert 'source_layout' not in block['content']['cells'][0]
    assert dict(reason='cell-source-lines-unproven', cell_id='left') in block['content']['source_layout_diagnostics']


def test_cross_cell_ocr_is_not_duplicated_or_used_as_anchor():
    block = table()
    block['source']['ocr_lines'][0]['bbox_px'] = [180, 50, 260, 70]
    enrich_table_content(block['content'], block['source'])
    assert all('source_layout' not in cell for cell in block['content']['cells'])
    assert any(v['reason']=='ambiguous-cell-ocr-ownership' for v in block['content']['source_layout_diagnostics'])


def test_estimated_cell_grid_does_not_claim_observed_text_positions():
    block = table()
    enrich_table_content(block['content'], block['source'], dict(geometryStatus='estimated-from-html-topology'))
    assert block['content']['source_geometry']['reliable'] is False
    assert measure_source_table(block, 200, _table_measurement) is None
    assert all('source_layout' not in c for c in block['content']['cells'])


def test_source_row_space_and_offsets_remain_when_ink_is_smaller():
    block = enriched()
    block['content']['cells'][0]['style'] = dict(align='left')
    measured = measure_source_table(block, 200, _table_measurement)
    assert measured['height'] >= 100
    assert measured['row_heights'] == pytest.approx([100])
    left, right = measured['cells']
    assert left['x'] == 0
    assert right['x'] == pytest.approx(100)
    assert left['lines'][0]['x'] == pytest.approx(20, abs=.1)
    assert left['lines'][1]['x'] == pytest.approx(7.5, abs=.1)
    assert right['lines'][0]['x'] == pytest.approx(40, abs=.1)
    assert left['lines'][0]['baseline'] < left['lines'][1]['baseline']
    for cell in measured['cells']:
        _validate_lines(cell)
        source = next(c for c in block['content']['cells'] if c['id']==cell['id'])
        _validate_source_lines(cell['lines'],source['text'])


def test_repeated_centered_cell_lines_keep_observed_centers():
    block = table()
    # Center alignment is evidenced by different source line lengths sharing
    # the same center, rather than the semantic category or cell bounding box.
    block['source']['ocr_lines'][0]['bbox_px'] = [80,50,140,70]
    block['source']['ocr_lines'][1]['bbox_px'] = [40,90,180,110]
    enrich_table_content(block['content'], block['source'])
    assert block['content']['cells'][0]['source_alignment']['align'] == 'center'
    measured = measure_source_table(block, 200, _table_measurement)
    cell = measured['cells'][0]
    from backend.font_library import get_library
    font = get_library().export_font_for_key(cell['font_key'])
    for line in cell['lines']:
        center = line['x']+font.text_length(line['text'],fontsize=line['font_size'])/2
        assert center == pytest.approx(45, abs=.1)


def test_enrichment_and_measurement_leave_source_evidence_unchanged():
    block = table()
    original = deepcopy(block['source'])
    enrich_table_content(block['content'], block['source'])
    text_before = [c['text'] for c in block['content']['cells']]
    enrich_table_content(block['content'], block['source'])
    measure_source_table(block, 200, _table_measurement)
    assert block['source'] == original
    assert [c['text'] for c in block['content']['cells']] == text_before


def test_empty_cell_does_not_gain_unmatched_ocr_or_visible_duplicate():
    block = table()
    block['content']['cells'][1]['text'] = ''
    enrich_table_content(block['content'], block['source'])
    cell = block['content']['cells'][1]
    assert cell['text'] == ''
    assert 'source_layout' not in cell
    assert dict(reason='ocr-in-model-empty-cell', cell_id='right') in block['content']['source_layout_diagnostics']


def test_last_resort_releases_only_unused_cell_envelope():
    block = enriched()
    fixed = measure_source_table(block, 200, _table_measurement)
    compact = measure_source_table(block, 200, _table_measurement, preserve_envelope=False)
    assert compact['height'] < fixed['height']
    assert [[l['text'] for l in c['lines']] for c in compact['cells']] == [[l['text'] for l in c['lines']] for c in fixed['cells']]
    assert compact['breaks'] == fixed['breaks']


def test_changed_width_scales_source_row_envelope_proportionally():
    block = table()
    block['source']['ocr_lines'] = []
    for cell in block['content']['cells']:
        cell['text'] = ''
    enrich_table_content(block['content'], block['source'])
    measured = measure_source_table(block, 100, _table_measurement)
    assert measured['width'] == 100
    assert measured['height'] == pytest.approx(50)
    assert measured['column_widths'] == pytest.approx([50,50])


def test_rowspan_topology_and_legal_breaks_survive_source_measurement():
    block = enriched()
    # Add a second row: the right cell spans both, while the left cells retain
    # their original source split and independent physical lines.
    block['source'].update(bbox_px=[20,40,420,440], bbox_pt=[10,20,210,220])
    block['content']['row_count'] = 2
    left, right = block['content']['cells']
    right.update(rowspan=2, source_bbox=[110,20,210,220])
    block['content']['cells'].append(dict(id='bottom', row=1, column=0, rowspan=1,
        colspan=1, text='', source_bbox=[10,120,110,220]))
    enrich_table_content(block['content'], block['source'])
    measured = measure_source_table(block, 200, _table_measurement)
    assert measured['breaks'] == [2]
    merged = next(c for c in measured['cells'] if c['id']=='right')
    assert merged['rowspan'] == 2
    assert merged['height'] == pytest.approx(sum(measured['row_heights']))
