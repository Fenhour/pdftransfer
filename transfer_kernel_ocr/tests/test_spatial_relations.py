"""Small manually specified page geometries for source-preserving stage 07."""
from copy import deepcopy

import pytest

from backend.spatial_relations import derive_spatial_relations


def block(identifier, box, label='text', kind='text', **extra):
    return dict(id=identifier, kind=kind,
        source=dict(page_id='p', label=label, bbox_pt=box, bbox_px=box),
        content={}, style={}, local_layout={}, **extra)


def project(blocks, regions=(), **extra):
    return dict(blocks=blocks, source_pages=[dict(id='p', width_pt=600., height_pt=800.,
        width_px=600, height_px=800, detections=dict(regions=list(regions)))], **extra)


def region(identifier, box):
    return dict(id=identifier, coordinate=box, label='Region')


def test_spanning_title_does_not_make_two_columns_read_row_by_row():
    p = project([block('right2', [330, 140, 560, 180]), block('title', [30, 20, 560, 60], 'doc_title'),
        block('left2', [30, 140, 260, 180]), block('right1', [330, 80, 560, 120]),
        block('left1', [30, 80, 260, 120])])
    before = deepcopy(p)
    r = derive_spatial_relations(p)
    assert r['source_order'] == ['title', 'left1', 'left2', 'right1', 'right2']
    assert {'columns', 'bands'} <= {g['type'] for g in r['layout_groups']}
    assert p == before


def test_separated_regions_have_unique_membership_and_original_frame_coordinates():
    p = project([block('a', [30, 30, 250, 100]), block('b', [340, 30, 550, 100])],
        [region('left', [20, 20, 270, 400]), region('right', [320, 20, 580, 400])])
    r = derive_spatial_relations(p)
    assert r['frame_membership'] == {'a': 'p-left', 'b': 'p-right'}
    assert r['region_membership'] == {'a': 'p-left', 'b': 'p-right'}
    assert [f['bbox_pt'] for f in r['frames']] == [[20, 20, 270, 400], [320, 20, 580, 400]]
    assert all(f['isolated'] for f in r['frames'])
    assert r['spatial_edges'] == [], 'Independent regions must not acquire a shared layout gap'


def test_duplicate_and_nested_region_boxes_do_not_give_children_two_visible_owners():
    p = project([block('a', [40, 40, 200, 100])],
        [region('outer', [20, 20, 500, 500]), region('same', [20, 20, 500, 500]),
         region('inner', [30, 30, 230, 200])])
    r = derive_spatial_relations(p)
    assert len(r['frames']) == 1
    assert r['frame_membership'] == {'a': 'p-outer'}
    assert r['frames'][0]['evidence']['duplicate_detection_ids'] == ['p-same']
    assert {'duplicate-region-detection', 'nested-region-uses-outer-frame'} <= {i['reason'] for i in r['issues']}


def test_crossing_outside_text_prevents_region_isolation_instead_of_swallowing_it():
    p = project([block('inside', [30, 40, 200, 100]), block('crossing', [240, 150, 340, 180])],
        [region('r', [20, 20, 270, 300])])
    r = derive_spatial_relations(p)
    assert all(f['kind'] == 'page' for f in r['frames'])
    assert r['region_membership'] == {'inside': None, 'crossing': None}
    assert any(i['reason'] == 'region-crosses-unowned-content' and i['block_ids'] == ['crossing']
               for i in r['issues'])


def test_partial_image_membership_stays_a_hint():
    # 95% ordinary text could belong; 95% of an image is insufficient.
    p = project([block('image', [20, 20, 120, 120], 'image', 'image')],
        [region('r', [20, 20, 115, 120])])
    r = derive_spatial_relations(p)
    assert r['frame_membership']['image'] == 'p-page-frame'
    assert r['region_membership']['image'] is None
    assert r['region_hints']['image'][0]['coverage'] == pytest.approx(.95)


def test_overlapping_region_predictions_are_not_hard_isolation_boundaries():
    p = project([block('a', [30, 40, 100, 80]), block('b', [380, 40, 450, 80])],
        [region('r1', [20, 20, 300, 300]), region('r2', [250, 20, 500, 300])])
    r = derive_spatial_relations(p)
    assert len(r['frames']) == 1 and r['frames'][0]['kind'] == 'page'
    assert all(value is None for value in r['region_membership'].values())


def test_shared_edges_stored_once_and_hidden_neighbours_removed():
    p = project([block('a', [20, 20, 80, 80]), block('b', [100, 20, 160, 80]),
                 block('c', [180, 20, 240, 80])])
    r = derive_spatial_relations(p)
    assert {(e['a'], e['b'], e['relation']) for e in r['spatial_edges']} == {
        ('a', 'b', 'left_of'), ('b', 'c', 'left_of')}
    assert all(e['gap_pt'] == 20 for e in r['spatial_edges'])
    assert len({frozenset((e['a'], e['b'])) for e in r['spatial_edges']}) == len(r['spatial_edges'])
    assert r['hard_before'] == [], 'Horizontal proximity alone must not become reading dependencies'


def test_spanning_module_can_have_multiple_neighbours_in_one_direction():
    p = project([block('a', [20, 20, 80, 180]), block('b', [100, 20, 160, 80]),
                 block('c', [100, 120, 160, 180])])
    r = derive_spatial_relations(p)
    assert {e['b'] for e in r['spatial_edges'] if e['a'] == 'a' and e['relation'] == 'left_of'} == {'b', 'c'}


def test_caption_must_match_same_scope_projection_and_no_intervening_content():
    p = project([block('figure', [30, 30, 220, 150], 'image', 'image'),
        block('caption', [30, 160, 220, 180], 'figure_title'),
        block('wrong-column', [340, 181, 550, 300], 'image', 'image')])
    r = derive_spatial_relations(p)
    assert ['figure', 'caption'] in r['hard_before']
    assert [(a['anchor'], a['dependent']) for a in r['attachments']] == [('figure', 'caption')]
    assert not any('wrong-column' in e for e in r['hard_before'])


def test_equally_plausible_caption_candidates_only_add_soft_evidence():
    p = project([block('figure1', [30, 30, 220, 150], 'image', 'image'),
        block('caption', [30, 160, 220, 180], 'figure_title'),
        block('figure2', [30, 190, 220, 300], 'image', 'image')])
    r = derive_spatial_relations(p)
    assert r['hard_before'] == [] and r['attachments'] == []
    assert r['soft_before'][0]['evidence']['ambiguous'] is True


def test_only_top_level_modules_participate_in_page_reading_order():
    parent = block('panel', [20, 20, 200, 200], 'container', 'composite')
    parent['content']['children'] = [block('child', [30, 30, 190, 80])]
    r = derive_spatial_relations(project([parent]))
    assert r['source_order'] == ['panel']
    assert r['contains'] == [dict(parent='panel', child='child')]
    assert r['frame_membership'] == {'panel': 'p-page-frame'}


def test_duplicate_visible_ids_and_container_cycles_are_rejected():
    parent = block('panel', [20, 20, 200, 200], 'container', 'composite')
    child = block('child', [30, 30, 190, 80])
    parent['content']['children'] = [child]
    with pytest.raises(ValueError, match='一个可见归属'):
        derive_spatial_relations(project([parent, child]))
    child['content']['children'] = [parent]
    with pytest.raises(ValueError, match='循环'):
        derive_spatial_relations(project([parent]))


@pytest.mark.parametrize('edges', [[['a', 'missing']], [['a', 'b'], ['b', 'a']]])
def test_explicit_dependencies_are_validated_before_layout(edges):
    with pytest.raises(ValueError, match='不存在|循环'):
        derive_spatial_relations(project([block('a', [20, 20, 100, 60]), block('b', [20, 80, 100, 120])],
                                         relations=dict(hard_before=edges)))


def test_valid_explicit_dependency_takes_priority_over_inferred_attachment():
    p = project([block('heading', [20, 20, 200, 40], 'paragraph_title'),
                 block('body', [20, 50, 200, 100])], relations=dict(hard_before=[['body', 'heading']]))
    r = derive_spatial_relations(p)
    assert r['source_order'] == ['heading', 'body'], 'Keep the source evidence separate from a user override'
    assert r['hard_before'] == [['body', 'heading']]
    assert r['attachments'] == []
    assert any(i['reason'] == 'inferred-reading-edge-conflicts-with-explicit-constraint' for i in r['issues'])
