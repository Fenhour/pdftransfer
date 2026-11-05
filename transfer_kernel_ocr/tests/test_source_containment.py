"""Original geometry ownership, including multiple containers and old projects."""
from copy import deepcopy

import pytest

from backend.source_containment import build_containment, conflict_is_internal, source_contains_pair
from backend.adaptive_reconstruction import blocks_from_page
from backend.four_layer_regions import solve
from backend.adaptive_export import _validate
from backend.adaptive_layout import layout_document
from test_adaptive_reconstruction import evidence
from test_regional_layout import text, project


def test_geometry_tree_retains_all_containers_and_ignores_categories():
    boxes = [dict(id=i, coordinate=b, label='anything') for i,b in [
        ('a',[0,0,100,100]), ('b',[10,10,50,50]), ('c',[20,20,30,30]),
        ('other',[15,0,60,60]), ('equal',[20,20,30,30]), ('cross',[90,90,110,110])]]
    original=deepcopy(boxes)
    nodes=build_containment(boxes)['nodes']
    assert boxes==original
    assert nodes['a']['contained_count']==4
    assert nodes['a']['child_ids']==['b','other']
    assert nodes['c']['parent_id']=='b'
    assert set(nodes['c']['container_ids'])=={'a','b','other'}
    assert 'equal' not in nodes['c']['contained_ids']
    assert nodes['cross']['parent_id'] is None
    assert build_containment(list(reversed(boxes)))['nodes']==nodes


def test_sparse_page_prunes_candidates():
    result=build_containment([dict(id=str(i),coordinate=[i*10,0,i*10+5,5]) for i in range(200)])
    assert result['candidate_checks']==0


@pytest.mark.parametrize('scale',[(.5,.5),(1.,1.),(.25,.75)])
def test_one_point_tolerance_is_physical_and_includes_boundary(scale):
    # Parent arrives after child in the sweep. All four sides are tested;
    # width grows despite permitted top/left/right excursions.
    def boxes(over):
        rects=[('p',[10,10,100,100]),('c',[10-over,20,100+over,50]),
               ('lower',[20,70,60,100+over])]
        return [dict(id=i,coordinate=[b[0]/scale[0],b[1]/scale[1],
                                      b[2]/scale[0],b[3]/scale[1]]) for i,b in rects]
    result=build_containment(boxes(1.),pixel_to_point=scale)
    assert result['nodes']['p']['contained_ids']==['c','lower']
    assert result['tolerance_pt']==1
    assert result['nodes']==build_containment(list(reversed(boxes(1.))),pixel_to_point=scale)['nodes']
    assert result['nodes']['c']['parent_id']=='p'
    assert not build_containment(boxes(1.001),pixel_to_point=scale)['nodes']['p']['contained_ids']


def test_tolerant_near_equal_frames_do_not_form_cycles_or_false_parents():
    result=build_containment([dict(id=i,coordinate=b) for i,b in [
        ('a',[0,0,100,100]),('b',[.1,.1,100.1,100.1]),('c',[0,0,100.1,100.1])]])
    assert all(n['parent_id'] is None for n in result['nodes'].values())


def test_empty_geometric_parent_does_not_turn_into_image(tmp_path):
    rgb,raw,page=evidence()
    # No semantic reference rules: an arbitrary text parent is also a container.
    raw['layout_det_res']['boxes'].append(dict(id='outer',label='text',coordinate=[30,80,560,200]))
    blocks=blocks_from_page(raw,rgb,page,tmp_path,four_layer=True)
    assert not any(b['id']=='source-1-outer' for b in blocks)
    parent=page['non_rendering_containers'][0]
    assert parent['id']=='source-1-outer' and parent['participates_in_layout'] is False
    child=next(b for b in blocks if b['id']=='source-1-layout-1')
    assert child['kind']=='text'
    assert child['source']['containment']['parent_id']==parent['id']


def test_nested_visual_exclusion_does_not_force_table_to_image(tmp_path):
    rgb,raw,page=evidence()
    raw['layout_det_res']['boxes'].append(dict(id='formula',label='formula',coordinate=[100,240,130,260]))
    raw['block_ownership']={'layout-2':dict(excluded_block_ids=['formula'],excluded_rects=[[100,240,130,260]])}
    raw['diagnostics']=[dict(reason='ocr-line-crosses-visual-block',block_id='layout-2',
                             candidate_ids=['layout-2'],excluded_ids=['formula'])]
    blocks=blocks_from_page(raw,rgb,page,tmp_path,four_layer=True)
    table=next(b for b in blocks if b['id']=='source-1-layout-2')
    assert table['kind']=='table'
    assert table['source']['containment']['contained_ids']==['source-1-formula']
    assert 'pred_html' in table['content'] and table['source']['reference_asset']


def test_unrelated_conflict_and_recognition_failure_still_rejected():
    nodes=build_containment([dict(id='p',coordinate=[0,0,100,100]),
                             dict(id='c',coordinate=[10,10,20,20])])['nodes']
    diagnostic=dict(reason='ocr-line-crosses-visual-block',block_id='p',excluded_ids=['c'])
    assert conflict_is_internal(diagnostic,'p',nodes)
    diagnostic['excluded_ids'].append('unrelated')
    assert not conflict_is_internal(diagnostic,'p',nodes)
    diagnostic.update(reason='ocr-line-recognition-rejected',excluded_ids=['c'])
    assert not conflict_is_internal(diagnostic,'p',nodes)


def test_original_nested_pair_stays_exempt_after_child_changes_size():
    a=text('parent',[0,0,200,200]); b=text('child',[20,20,40,40])
    pos,_,_,_=solve([a,b],{'parent':dict(width=200,height=200),
                         'child':dict(width=90,height=60)},[0,0,300,300])
    assert pos=={'parent':[0,0],'child':[20,20]}
    b['source']['page_id']='another-page'
    assert not source_contains_pair(a,b)


def test_export_accepts_nested_original_blocks_without_shared_transform():
    p=project([text('parent',[10,10,200,200]),text('child',[20,20,80,50])])
    layout_document(p)
    _validate(p)
    assert len(p['layout_result']['placements'])==2


def test_partial_overlap_remains_an_error():
    a=text('a',[10,10,100,100]);b=text('b',[50,50,150,150])
    with pytest.raises(ValueError,match='重叠关系不明确'):
        solve([a,b],{'a':dict(width=90,height=90),'b':dict(width=100,height=100)},[0,0,300,300])
