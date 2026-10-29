"""Source physical rows survive the first reconstruction layer."""
from copy import deepcopy

import pytest

from backend.adaptive_export import _validate_lines, _validate_source_lines
from backend.adaptive_layout import _text_measurement
from backend.font_library import get_library
from backend.source_text import build_text_content, measure_source_text


STYLE = dict(font_size_pt=10, family='sans-serif', align='left', line_height=1.3)


def record(text, box):
    return dict(text=text, bbox_px=box)


def measure(records, box=None, **style):
    content = build_text_content(records, 'body')
    box = box or [0, 0, 400, 200]
    source = dict(bbox_px=box, bbox_pt=[v/2 for v in box])
    value = measure_source_text(content, (box[2]-box[0])/2, dict(STYLE, **style), source, _text_measurement)
    _validate_lines(value)
    _validate_source_lines(value['lines'], content['text'])
    return content, value


def test_toc_baseline_clustering_keeps_entry_before_one_pixel_higher_page_number():
    records = []
    for i in range(12):
        records.extend([record(str(4+i*8), [300, 20+i*30-1, 325, 40+i*30-1]),
                        record('Introduction' if i == 0 else f'Test {i}', [20, 20+i*30, 180, 40+i*30])])
    content, value = measure(records, [0, 0, 400, 400])
    assert len(content['source_layout']['rows']) == 12
    assert len(value['lines']) == 24
    assert content['text'].startswith('Introduction\n4\nTest 1\n12')
    assert value['height'] == 200
    assert value['lines'][0]['x'] == 10
    assert value['lines'][1]['x'] == 150
    assert value['lines'][0]['baseline'] == pytest.approx(value['lines'][1]['baseline']+.5)


def test_body_equal_source_left_edges_do_not_invent_indent_and_real_indent_survives():
    records = [record('First physical line.', [92, 20, 350, 40]),
               record('Second physical line.', [92, 50, 370, 70]),
               record('Third physical line.', [93, 80, 330, 100])]
    _, value = measure(records, [90, 18, 400, 150])
    assert [v['x'] for v in value['lines']] == [1, 1, 1.5]
    records[0]['bbox_px'][0] = 114
    _, indented = measure(records, [90, 18, 400, 150])
    assert [v['x'] for v in indented['lines']] == [12, 1, 1.5]


def test_source_height_and_unused_trailing_space_are_retained_until_explicit_compaction():
    records = [record('Short.', [20, 20, 100, 40])]
    content, preserved = measure(records)
    compact = measure_source_text(content, 200, STYLE,
                                  dict(bbox_px=[0, 0, 400, 200], bbox_pt=[0, 0, 200, 100]),
                                  _text_measurement, preserve_envelope=False)
    assert preserved['height'] == 100
    assert compact['height'] < preserved['height']
    assert compact['lines'] == preserved['lines']
    assert preserved['source_envelope_preserved'] is True


def test_row_fit_stops_before_next_fragment_and_does_not_overlap_following_row():
    records = [record('Long left heading needs to fit.', [10, 10, 230, 30]),
               record('42', [240, 10, 270, 30]),
               record('Next row.', [10, 25, 200, 45])]
    _, value = measure(records, [0, 0, 300, 100])
    library = get_library()
    font = library.export_font_for_key(value['font_key'])
    first = value['lines'][0]
    page = next(v for v in value['lines'] if v['text'] == '42')
    last = next(v for v in value['lines'] if v['text'] == 'Next row.')
    assert first['x']+font.text_length(first['text'], fontsize=first['font_size']) <= page['x']+1e-6
    assert first['font_size'] <= STYLE['font_size_pt']
    assert last['baseline']-font.ascender*last['font_size'] >= max(
        v['baseline']-font.descender*v['font_size'] for v in value['lines'] if v['start'] < last['start'])-1e-6


def test_body_rows_keep_common_readable_size_and_only_overlong_fragment_wraps():
    records = [record('This line is far too long for the observed narrow source row.', [10, 10, 120, 30]),
               record('Short.', [10, 45, 100, 65])]
    content, value = measure(records, [0, 0, 150, 120])
    assert len(value['lines']) > 2
    assert {v['font_size'] for v in value['lines']} == {8.5}
    assert value['font_size'] == 8.5
    assert value['lines'][-1]['text'] == 'Short.'
    assert any(v['reason'] == 'source-fragment-wrap' for v in value['internal_adjustments'])
    assert value['lines'][-1]['end'] == len(content['text'])


def test_wrapped_fragment_keeps_its_resolved_font_metrics_in_mixed_script_block():
    records = [record('中文文字', [10, 10, 130, 30]),
               record('A long English fragment needs several physical lines here.', [10, 45, 130, 65])]
    content, value = measure(records, [0, 0, 150, 160])
    assert len(value['lines']) > 2
    assert value['lines'][-1]['end'] == len(content['text'])
    assert all(v['font_size'] >= 8.5 for v in value['lines'])


def test_center_alignment_uses_each_source_fragment_center():
    _, value = measure([record('Badge', [120, 20, 280, 40])], align='center')
    line = value['lines'][0]
    font = get_library().export_font_for_key(value['font_key'])
    assert line['x']+font.text_length(line['text'], fontsize=line['font_size'])/2 == pytest.approx(100)


def test_edited_text_or_invalid_geometry_disables_stale_source_layout():
    content = build_text_content([record('Original.', [10, 10, 150, 30])], 'body')
    source = dict(bbox_px=[0, 0, 400, 200], bbox_pt=[0, 0, 200, 100])
    changed = deepcopy(content)
    changed['text'] = 'Edited.'
    assert measure_source_text(changed, 200, STYLE, source, _text_measurement) is None
    source['bbox_px'][2] = float('nan')
    assert measure_source_text(content, 200, STYLE, source, _text_measurement) is None
    invalid = build_text_content([record('Keep these words.', [0, 0, 0, 10])], 'body')
    assert invalid['text'] == 'Keep these words.'
    assert 'source_layout' not in invalid
