"""General geometric text grouping without document-specific vocabulary."""
import copy
import json

from backend.rebuild_text import organize_text


def text(id, value, x, y, width, height=12, **extra):
    return dict(id=id, type='text', text=value, x=x, y=y, width=width,
                height=height, fontSize=height, rotation=0, source='ocr', **extra)


def page(*elements, **extra):
    return dict(id='page-1', width=600, height=800, elements=list(elements), **extra)


def test_nearby_baseline_fragments_share_line_without_changing_text_or_coordinates():
    data = page(text('a', 'A continuous line', 30, 30, 110),
                text('b', '15', 144, 33, 12, 9),
                text('c', 'units', 160, 30, 35))
    original = copy.deepcopy(data['elements'])
    result = organize_text(data)
    assert len(result['lines']) == 1
    assert result['lines'][0]['fragmentIds'] == ['a', 'b', 'c']
    assert [e['fragmentIndex'] for e in data['elements']] == [0, 1, 2]
    assert all(e['fragmentCount'] == 3 for e in data['elements'])
    for before, after in zip(original, data['elements']):
        assert all(after[k] == v for k, v in before.items())
    assert result['lines'][0]['fragments'][1]['gapBefore'] == 4


def test_large_same_row_gap_preserves_independent_fragments_without_padded_strings():
    data = page(text('a', 'Left', 30, 30, 40), text('b', 'Right', 300, 30, 40))
    result = organize_text(data)
    assert len(result['lines']) == 2
    assert len(result['blocks']) == 2
    assert [e['text'] for e in data['elements']] == ['Left', 'Right']


def test_repeated_narrow_gutter_prevents_two_columns_from_becoming_one_line():
    elements = []
    for row in range(4):
        elements += [text(f'l{row}', 'abcdefghij', 30, 30+row*17, 100),
                     text(f'r{row}', 'klmnopqrst', 144, 30+row*17, 100)]
    result = organize_text(page(*elements))
    assert len(result['lines']) == 8
    assert len(result['regions']) == 2
    assert len(result['blocks']) == 2
    assert result['columns'][0]['count'] == 2
    assert result['columns'][0]['gutters'][0]['supportRows'] == 4
    assert elements[0]['regionId'] != elements[1]['regionId']


def test_full_width_heading_can_span_two_body_columns():
    heading = text('heading', 'A title', 30, 5, 220, 20, contentType='heading')
    elements = [heading]
    for row in range(3):
        elements += [text(f'l{row}', 'abcdefghij', 30, 45+row*17, 100),
                     text(f'r{row}', 'klmnopqrst', 150, 45+row*17, 100)]
    result = organize_text(page(*elements))
    assert len(result['regions']) == 3
    assert next(r for r in result['regions'] if r['id'] == heading['regionId'])['columnIndex'] == 'spanning'
    assert len(result['blocks']) == 3
    assert result['semanticsInferred'] is False


def test_continuous_body_forms_block_and_large_gap_starts_another():
    elements = [text(f'line{i}', 'Generic body content', 30, y, 240)
                for i, y in enumerate([30, 47, 64, 110, 127])]
    result = organize_text(page(*elements))
    assert [len(b['lineIds']) for b in result['blocks']] == [3, 2]


def test_heading_and_body_separate_without_using_fitted_fontsize_alone():
    elements = [text('title', 'Title', 30, 20, 170, 20, contentType='heading'),
                text('a', 'Ordinary first line', 30, 45, 220),
                text('b', 'Ordinary second line', 30, 62, 210)]
    elements[-1]['fontSize'] = 40
    result = organize_text(page(*elements))
    assert [len(b['lineIds']) for b in result['blocks']] == [1, 2]
    assert elements[-1]['fontSize'] == 40, 'Layout does not repair styles by guessing'


def test_image_between_close_body_lines_is_a_block_barrier_and_is_not_mutated():
    image = dict(id='image', type='image', x=80, y=43, width=100, height=4,
                 retainedText=[dict(text='label')], src='source.png')
    before = copy.deepcopy(image)
    result = organize_text(page(text('a', 'Above', 30, 30, 220), image,
                                text('b', 'Below', 30, 48, 220)))
    assert len(result['blocks']) == 2
    assert image == before


def test_image_in_horizontal_gap_is_a_line_barrier():
    image = dict(id='image', type='image', x=111, y=28, width=8, height=18)
    result = organize_text(page(text('a', 'left', 30, 30, 80), image,
                                text('b', 'right', 120, 30, 80)))
    assert len(result['lines']) == 2


def test_independent_rules_block_horizontal_and_vertical_grouping():
    vertical = dict(id='v', type='shape', shape='line', x=111, y=28, width=1, height=18)
    result = organize_text(page(text('a', 'left', 30, 30, 80), vertical,
                                text('b', 'right', 120, 30, 80)))
    assert len(result['lines']) == 2
    horizontal = dict(id='h', type='shape', shape='line', x=20, y=44, width=250, height=1)
    result = organize_text(page(text('a', 'Above', 30, 30, 220), horizontal,
                                text('b', 'Below', 30, 48, 220)))
    assert len(result['blocks']) == 2


def test_frame_container_is_separate_from_nearby_outside_body():
    frame = dict(id='frame-1', type='shape', shape='rect', contentType='decoration',
                 x=20, y=30, width=250, height=70, fill='none', stroke='#000000')
    elements = [text('before', 'Outside', 30, 15, 200), frame,
                text('inside1', 'Inside one', 30, 35, 200), text('inside2', 'Inside two', 30, 52, 200),
                text('after', 'Outside again', 30, 105, 200)]
    result = organize_text(page(*elements))
    assert elements[2]['regionId'] == elements[3]['regionId']
    assert elements[0]['regionId'] != elements[2]['regionId']
    region = next(r for r in result['regions'] if r['id'] == elements[2]['regionId'])
    assert region['containerType'] == 'frame'
    assert region['containerId'] == 'frame-1'


def test_nested_frame_prefers_smallest_container_and_explicit_owner_wins():
    outer = dict(id='outer', type='shape', shape='rect', contentType='decoration', x=0, y=0, width=400, height=400)
    inner = dict(id='inner', type='shape', shape='rect', contentType='decoration', x=20, y=20, width=150, height=90)
    elements = [outer, inner, text('auto', 'Inside', 30, 30, 80),
                text('explicit', 'Keep owner', 30, 50, 80, frameId='outer')]
    result = organize_text(page(*elements))
    regions = {r['id']: r for r in result['regions']}
    assert regions[elements[2]['regionId']]['containerId'] == 'inner'
    assert regions[elements[3]['regionId']]['containerId'] == 'outer'


def test_table_cells_are_distinct_even_when_text_baselines_match():
    elements = [text('a', 'First', 30, 30, 45, tableId='table-1'),
                text('b', 'Second', 90, 30, 45, tableId='table-1')]
    table = dict(id='table-1', cells=[dict(bbox=[20, 20, 80, 60], textIds=['a']),
                                    dict(bbox=[80, 20, 140, 60], textIds=['b'])])
    result = organize_text(page(*elements, tables=[table]))
    assert len(result['lines']) == len(result['regions']) == 2
    assert [r['cellIndex'] for r in result['regions']] == [0, 1]
    assert elements[0]['regionId'] != elements[1]['regionId']


def test_unresolved_table_owner_remains_separate_from_body():
    elements = [text('a', 'Body', 30, 30, 80),
                text('b', 'Known table owner', 30, 47, 80, tableId='table-pending')]
    result = organize_text(page(*elements))
    assert len(result['regions']) == 2
    assert any(r['containerId'] == 'table-pending' for r in result['regions'])


def test_rotated_text_is_preserved_in_independent_lines():
    a = text('a', 'Rotated', 30, 30, 80)
    b = text('b', 'Also rotated', 112, 30, 80)
    a['rotation'] = b['rotation'] = 90
    result = organize_text(page(a, b))
    assert len(result['lines']) == len(result['blocks']) == 2
    assert all(l['rotated'] for l in result['lines'])
    assert result['stats']['rotatedFragments'] == 2


def test_empty_invalid_hidden_and_blank_text_are_safe_and_json_serializable():
    assert organize_text(page())['lines'] == []
    data = page(text('blank', ' ', 30, 30, 10), text('hidden', 'Hidden', 30, 50, 30, visible=False),
                dict(id='bad', type='text', text='Invalid', x=0, y=0, width=0, height=12))
    result = organize_text(data)
    assert result['stats']['skippedElements'] == 3
    assert result['lines'] == []
    json.dumps(result, allow_nan=False)


def test_source_provenance_and_deterministic_ids_are_retained():
    data = page(text('a', 'Original', 30, 30, 50, sourceLineId='ocr-line-7'))
    first = copy.deepcopy(organize_text(data))
    second = organize_text(data)
    first.pop('stats')
    second = {k: v for k, v in second.items() if k != 'stats'}
    assert first == second
    assert first['lines'][0]['fragments'][0]['sourceLineId'] == 'ocr-line-7'


def test_large_page_uses_bounded_neighbor_comparisons():
    elements = [text(f'line-{i}', 'Representative ordinary text', 30, 10+i*16, 300) for i in range(1600)]
    result = organize_text(page(*elements))
    assert result['stats']['lineCount'] == 1600
    assert result['stats']['blockCount'] == 1
    assert result['stats']['rowComparisons'] <= len(elements)*24
    assert result['stats']['blockComparisons'] <= len(elements)*64
    assert result['stats']['elapsedSeconds'] >= 0
