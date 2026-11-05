from copy import deepcopy

import pytest

from backend.table_image_binding import bind_table_images


def scene():
    blocks = [dict(id='t', kind='table', source=dict(page_id='p', bbox_pt=[0, 0, 100, 20]),
                   content=dict(cells=[dict(id='c', row=1, column=0, source_bbox=[0, 10, 100, 20])])),
              dict(id='f', kind='image', source=dict(page_id='p', bbox_pt=[40, 12, 60, 18]), content={})]
    placements = [dict(block_id='t', page_index=0, frame_id='frame', bbox=[10, 20, 110, 80],
                       measurement=dict(cells=[dict(id='c', x=0, y=30, width=100, height=30)])),
                  dict(block_id='f', page_index=0, frame_id='old', bbox=[50, 32, 70, 38],
                       measurement=dict(width=20, height=6, available_width=20, uniform_scale=1.))]
    return blocks, placements


def test_formula_follows_cell_after_rows_expand_without_raster_stretch():
    blocks, placements = scene()
    bind_table_images(blocks, placements)
    image = placements[1]
    assert image['bbox'] == pytest.approx([50, 62, 70, 68])
    assert image['measurement']['width']/image['measurement']['height'] == pytest.approx(20/6)
    assert image['table_cell_binding']['cell_id'] == 'c'
    assert blocks[1]['source']['table_cell_binding']['row'] == 1
    assert image['source_transform'] == pytest.approx([1, 0, 0, 1, 10, 50])
    first = deepcopy(image['bbox'])
    bind_table_images(blocks, placements)
    assert image['bbox'] == first  # Repeat export/layout must not accumulate offsets.


def test_page_scale_and_column_change_are_applied_once():
    blocks, placements = scene()
    placements[0]['measurement']['cells'][0].update(x=5, y=15, width=50, height=15)
    placements[1]['measurement'].update(width=10, height=3, uniform_scale=.5)
    bind_table_images(blocks, placements)
    assert placements[1]['bbox'] == pytest.approx([35, 41, 45, 44])
    assert placements[1]['measurement']['uniform_scale'] == .5


def test_cross_cell_or_other_page_images_are_not_guessed_into_a_cell():
    blocks, placements = scene()
    blocks[1]['source']['bbox_pt'] = [40, 5, 60, 15]  # Only half is inside cell.
    before = deepcopy(placements[1])
    bind_table_images(blocks, placements)
    assert placements[1] == before
    blocks[1]['source'].update(page_id='another', bbox_pt=[40, 12, 60, 18])
    bind_table_images(blocks, placements)
    assert placements[1] == before


def test_multiple_formula_rows_bind_to_contiguous_cell_range():
    blocks, placements = scene()
    blocks[0]['content']['cells'].insert(0,dict(id='c0', row=0,column=0,source_bbox=[0,0,100,10]))
    placements[0]['measurement']['cells'].insert(0,dict(id='c0', x=0,y=0,width=100,height=30))
    blocks[1]['source']['bbox_pt'] = [40,5,60,15]
    placements[1]['measurement'].update(width=20,height=10)
    bind_table_images(blocks,placements)
    binding = placements[1]['table_cell_binding']
    assert binding['cell_ids'] == ['c0','c']
    assert binding['target_cell_bbox'] == [10,20,110,80]
    assert placements[1]['bbox'] == pytest.approx([50,45,70,55])
