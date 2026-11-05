from copy import deepcopy

import pytest

from backend.text_linkage import solve_text_rectangles, overlaps
from backend.four_layer_text import measure_four_layer, deduplicate_records
from backend.source_text import build_text_content
from test_text_adaptation_v2 import fixed_font


def rectangle(id,x,y,w,h,source=None,row=None):
    return dict(id=id,x=x,y=y,width=w,height=h,source_bbox=source or [x,y,x+w,y+h],row=row)


def assert_geometry(items,positions,bounds):
    for i in items:
        r=positions[i['id']]
        assert r[2]-r[0]==pytest.approx(i['width'])
        assert r[3]-r[1]==pytest.approx(i['height'])
        assert bounds[0]-1e-7<=r[0] and r[2]<=bounds[2]+1e-7
        assert bounds[1]-1e-7<=r[1] and r[3]<=bounds[3]+1e-7
    assert not any(overlaps(a,b) for n,a in enumerate(positions.values()) for b in list(positions.values())[n+1:])


def test_chain_includes_previously_clear_left_obstacle_and_exact_gaps():
    items=[rectangle('F',0,0,20,10,[0,0,20,10],0),
           rectangle('A',30,0,40,10,[30,0,50,10],0),
           rectangle('B',60,0,40,10,[60,0,80,10],0),
           rectangle('C',90,0,30,10,[90,0,110,10],0)]
    original=deepcopy(items)
    bounds=[0,0,160,20]
    pos,detail=solve_text_rectangles(items,bounds)
    assert detail['status']=='optimal'
    assert_geometry(items,pos,bounds)
    assert pos['A'][0]-pos['F'][2]==pytest.approx(10)
    assert pos['B'][0]-pos['A'][2]==pytest.approx(10)
    assert pos['C'][0]-pos['B'][2]==pytest.approx(10)
    assert items==original


def test_multi_axis_linkage_and_source_containment_are_solved_jointly():
    items=[rectangle('A',10,10,35,20,[10,10,30,20],0),
           rectangle('B',35,10,35,20,[35,10,55,20],0),
           rectangle('D',10,25,60,20,[10,25,70,35],1),
           rectangle('E',20,0,30,15,[20,0,50,8],2)]
    bounds=[0,0,90,65]
    pos,detail=solve_text_rectangles(items,bounds)
    assert detail['status']=='optimal'
    assert_geometry(items,pos,bounds)
    assert pos['E'][3]<=pos['A'][1]+1e-7
    assert pos['A'][3]<=pos['D'][1]+1e-7
    # Sibling containment is a collision, not parent ownership.
    siblings=[rectangle('outer',0,0,20,20),rectangle('inner',5,5,5,5)]
    pos,detail=solve_text_rectangles(siblings,[0,0,30,30])
    assert detail['status']=='optimal'
    assert_geometry(siblings,pos,[0,0,30,30])


def test_bounded_failure_does_not_commit_partial_pushes():
    items=[rectangle('a',0,0,60,10,[0,0,40,10],0),rectangle('b',50,0,60,10,[50,0,90,10],0)]
    pos,detail=solve_text_rectangles(items,[0,0,100,20])
    assert detail['status']=='infeasible'
    assert pos['a']==[0,0,60,10] and pos['b']==[50,0,110,10]
    assert detail['failure_reasons']==['boundary-capacity']
    assert not detail['committed']


def test_budget_exhaustion_is_not_an_infeasibility_proof():
    items=[rectangle('a',0,0,20,20),rectangle('b',5,5,10,10)]
    _,detail=solve_text_rectangles(items,[0,0,40,40],max_seconds=0)
    assert detail['status']=='undetermined' and detail['frontier']>0


def test_vertical_collision_keeps_all_horizontal_anchors_and_clear_title():
    items=[dict(rectangle('a',70,40,50,15,[70,40,120,50],0),row_order=0),
           dict(rectangle('b',90,50,50,15,[90,50,140,60],1),row_order=1),
           rectangle('title',130,5,60,12)]
    original=deepcopy(items)
    pos,detail=solve_text_rectangles(items,[0,0,300,200])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,300,200])
    assert all(pos[i['id']][0]==i['x'] for i in items)
    assert pos['title']==[130,5,190,17]
    assert abs(pos['a'][1]-40)==pytest.approx(2.5,abs=1e-5)
    assert abs(pos['b'][1]-50)==pytest.approx(2.5,abs=1e-5)
    assert detail['moved_ids_by_axis']['x']==[] and items==original


def test_horizontal_collision_never_changes_baselines_or_clear_sibling():
    items=[rectangle('a',50,80,30,10,[50,80,70,90],0),
           rectangle('b',75,80,30,10,[75,80,95,90],0),
           rectangle('clear',150,140,20,10)]
    pos,detail=solve_text_rectangles(items,[0,0,250,200])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,250,200])
    assert all(pos[i['id']][1]==i['y'] for i in items)
    assert pos['clear']==[150,140,170,150]
    assert pos['b'][0]-pos['a'][2]==pytest.approx(5)
    assert detail['moved_ids_by_axis']['y']==[]


def test_clear_fragments_are_not_repacked_to_reduce_source_gap_or_area():
    items=[rectangle('a',40,30,10,10,[40,30,70,40],0),
           rectangle('b',90,30,10,10,[90,30,110,40],0),
           rectangle('title',120,3,40,10)]
    pos,detail=solve_text_rectangles(items,[0,0,300,100])
    assert detail['status']=='unchanged'
    assert all(pos[i['id']]==[i['x'],i['y'],i['x']+i['width'],i['y']+i['height']] for i in items)


def test_vertical_capacity_failure_does_not_escape_horizontally():
    items=[rectangle('a',30,0,20,20,[30,0,50,10],0),
           rectangle('b',35,10,20,20,[35,10,55,20],1)]
    pos,detail=solve_text_rectangles(items,[0,0,200,30])
    assert detail['status']=='infeasible'
    assert pos['a']==[30,0,50,20] and pos['b']==[35,10,55,30]
    assert {x['axis'] for x in detail['collision_axes']}=={'y'}


def test_vertical_propagation_includes_next_obstacle_without_x_motion():
    items=[rectangle('a',30,0,30,20,[30,0,60,10],0),
           rectangle('b',40,10,30,12,[40,10,70,20],1),
           rectangle('c',35,25,30,12,[35,25,65,35],2)]
    pos,detail=solve_text_rectangles(items,[0,0,150,60])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,150,60])
    assert all(pos[i['id']][0]==i['x'] for i in items)
    assert len(detail['collision_axes'])>=2


def test_right_envelope_growth_does_not_move_clear_text_left(fixed_font):
    records=[dict(text='a'*20,line_id='title',bbox_px=[30,20,100,30])]
    content=build_text_content(records,'test',four_layer=True)
    m=measure_four_layer(content,100,dict(font_size_pt=10),
        dict(bbox_px=[0,0,100,80],bbox_pt=[0,0,100,80]),text_bounds_pt=[0,0,180,80])
    assert m['lines'][0]['x']==30
    assert m['width']==130
    assert m['block_expansion']['left_pt']==0


def test_crop_normalisation_compensates_page_placement_origin(fixed_font):
    from backend.four_layer_regions import solve
    record=dict(text='title',line_id='title',bbox_px=[19,19,44,29])
    source=dict(bbox_px=[20,20,100,60],bbox_pt=[20,20,100,60])
    m=measure_four_layer(build_text_content([record],'test',four_layer=True),80,
        dict(font_size_pt=10),source,text_bounds_pt=[-20,-20,140,80])
    b=dict(id='title',kind='text',source=source)
    pos,_,scale,_=solve([b],{'title':m},[0,0,160,100])
    assert scale==1
    assert pos['title'][0]+m['lines'][0]['x']==19
    assert pos['title'][1]+m['lines'][0]['baseline']-8==19


def test_isolated_caption_keeps_original_centre_with_wider_font(fixed_font):
    record=dict(text='a'*20,line_id='title',bbox_px=[100,20,150,30])
    source=dict(bbox_px=[100,20,180,60],bbox_pt=[100,20,180,60])
    m=measure_four_layer(build_text_content([record],'test',four_layer=True),80,
        dict(font_size_pt=10,align='center'),source,text_bounds_pt=[-100,-20,160,80])
    line=m['lines'][0]
    page_x=100+m['block_expansion']['left_pt']+line['x']
    assert page_x+line['advance_width']/2==pytest.approx(125)
    assert 20+m['block_expansion']['top_pt']+line['baseline']-8==20


def test_code_row_order_and_indentation_survive_joint_compaction():
    items=[dict(rectangle('a',0,0,30,14,[0,0,30,10],0),row_order=0,paragraph_id='p'),
           dict(rectangle('b',40,10,30,14,[40,10,70,20],1),row_order=1,paragraph_id='p'),
           dict(rectangle('c',20,20,30,14,[20,20,50,30],2),row_order=2,paragraph_id='p')]
    pos,detail=solve_text_rectangles(items,[0,0,100,60])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,100,60])
    assert pos['a'][1]<=pos['b'][1] and pos['b'][3]<=pos['c'][1]+1e-7
    assert pos['a']==[0,0,30,14]  # Clear row stays at its own anchor.
    assert pos['b'][0]-pos['a'][0]==pytest.approx(40)
    assert pos['c'][0]-pos['a'][0]==pytest.approx(20)


def test_code_indent_is_shared_across_physical_paragraph_breaks():
    items=[dict(rectangle('for',48,0,60,14,row=0),row_order=0,paragraph_id='p0',indentation_group='code'),
           dict(rectangle('body',72,12,60,14,row=1),row_order=1,paragraph_id='p0',indentation_group='code'),
           dict(rectangle('if',48,30,60,14,row=2),row_order=2,paragraph_id='p1',indentation_group='code'),
           dict(rectangle('tail',72,42,60,14,row=3),row_order=3,paragraph_id='p1',indentation_group='code')]
    pos,detail=solve_text_rectangles(items,[0,0,160,90])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,160,90])
    assert pos['if'][0]==pytest.approx(pos['for'][0])
    assert pos['tail'][0]-pos['if'][0]==pytest.approx(24)


def test_non_aligned_line_preserves_start_when_other_rows_shrink(fixed_font):
    records=[dict(text=t,line_id=str(n),bbox_px=[x,n*18,x+(35 if n<4 else 5),n*18+10])
             for n,(x,t) in enumerate([(0,'aaa bbb')]*3+[(0,'aaaa aaaa'),(65,'x')])]
    content=build_text_content(records,'test',four_layer=True)
    result=measure_four_layer(content,150,dict(font_size_pt=10,word_gap_pt=5,alignment_tolerance_ratio=1.),
        dict(bbox_px=[0,0,150,120],bbox_pt=[0,0,150,120]))
    state=result['paragraph_adjustments'][0]
    assert state['deferred_row_ids']==[3]
    event=state['row_adjustments'][4]
    assert not event['align_left'] and not event['align_right']
    assert event['actual_endpoints'][0]==pytest.approx(event['previous_endpoints'][0])
    assert result['lines'][4]['spacing_pt']['latin_word_gap_pt']==5


def test_explicit_sequence_gap_overrides_source_gap_without_silent_repair():
    items=[rectangle('a',0,0,20,10,row=0),rectangle('b',15,0,20,10,row=0)]
    pos,detail=solve_text_rectangles(items,[0,0,55,15],[dict(a='a',b='b',axis='x',gap=10,kind='exact')])
    assert detail['status']=='optimal'
    assert pos['b'][0]-pos['a'][2]==pytest.approx(10)
    _,detail=solve_text_rectangles(items,[0,0,55,15],[dict(a='a',b='b',axis='x',gap=-5,kind='exact')])
    assert detail['status']=='infeasible'


def measure(records,style=None):
    content=build_text_content(records,'test',four_layer=True)
    source=dict(bbox_px=[0,0,400,200],bbox_pt=[0,0,400,200])
    return measure_four_layer(content,400,dict(font_size_pt=10,**(style or {})),source)


@pytest.mark.parametrize('texts,lefts,expected_status',[
    (['aaaaaa','aaaaa'],[0,0],'no-expandable-gap'),
    (['aaaaaa','aaaaa'],[0,20],'no-expandable-gap'),
    (['aaa bbb','aaa bb'],[0,20],'aligned'),
])
def test_after_alignment_locks_actual_offsets_even_when_target_unreached(
        fixed_font,texts,lefts,expected_status):
    records=[dict(text=t,line_id=str(n),bbox_px=[x,n*18,100,n*18+10])
             for n,(x,t) in enumerate(zip(lefts,texts))]
    result=measure(records)
    events=result['paragraph_adjustments'][0]['row_adjustments']
    assert all(e['status']==expected_status for e in events)
    if expected_status=='no-expandable-gap':
        assert any(abs(e['actual_endpoints'][1]-e['target_endpoints'][1])>1 for e in events)
    final=result['text_linkage']['after_alignment']
    assert final['status'] in {'unchanged','optimal','feasible'}
    expected=events[1]['actual_endpoints'][0]-events[0]['actual_endpoints'][0]
    locks=[c for c in final['constraints'] if c['origin']=='current-row-indentation']
    assert len(locks)==1 and locks[0]['offset_pt']==pytest.approx(expected)
    assert final['indentation_lock_basis']=='latest-post-alignment-row-positions'
    assert result['lines'][1]['x']-result['lines'][0]['x']==pytest.approx(expected)


def test_latest_indentation_survives_vertical_collision_repair():
    items=[dict(rectangle('a',10,0,50,14,[10,0,60,8],0),paragraph_id='p',row_order=0),
           dict(rectangle('b',27,9,45,14,[10,9,60,17],1),paragraph_id='p',row_order=1)]
    pos,detail=solve_text_rectangles(items,[0,0,150,60])
    assert detail['status']=='optimal'
    assert_geometry(items,pos,[0,0,150,60])
    assert pos['b'][0]-pos['a'][0]==pytest.approx(17)
    assert any(c['origin']=='current-row-indentation' and c['offset_pt']==17
               for c in detail['constraints'])


def test_long_unaligned_right_edge_cannot_trigger_paragraph_shrink(fixed_font):
    records=[dict(text=t,line_id=str(n),bbox_px=[0,n*12,35 if n<3 else 300,n*12+10])
             for n,t in enumerate(['aaa bbb']*3+['a'*50])]
    result=measure(records)
    state=result['paragraph_adjustments'][0]
    assert not state['row_adjustments'][3]['align_right']
    assert state['font_scale']==1 and not state['deferred_row_ids']
    assert result['lines'][3]['font_size']==10
    assert result['lines'][3]['advance_width']==250


def test_return_comment_short_rows_do_not_shrink_or_expand_to_dominant_width(fixed_font):
    records=[dict(text=t,line_id=str(n),bbox_px=[x,n*12,x+300,n*12+10])
             for n,(x,t) in enumerate([(24,'Return: null if timeout otherwise return value'),(26,'of callback'),(23,'*/'),(1,'}')])]
    result=measure(records)
    assert all(l['font_size']==10 for l in result['lines'])
    assert result['paragraph_adjustments'][0]['font_scale']==1
    assert result['paragraph_adjustments'][0]['deferred_row_ids']==[]


def test_size_normalisation_requires_residual_collision_after_bounded_translation(fixed_font):
    records=[dict(text='aaa',line_id='a',bbox_px=[0,0,15,20],measurement_style=dict(font_size_pt=20)),
             dict(text='bbb',line_id='b',bbox_px=[20,0,35,20]),
             dict(text='ccc',line_id='peer',bbox_px=[0,22,15,42])]
    roomy=measure(records)
    assert not any(a['type']=='residual-collision-size-normalization' for a in roomy['internal_adjustments'])
    content=build_text_content(records,'test',four_layer=True)
    tight=measure_four_layer(content,40,dict(font_size_pt=10),dict(bbox_px=[0,0,40,45],bbox_pt=[0,0,40,45]))
    assert any(a['type']=='residual-collision-size-normalization' for a in tight['internal_adjustments'])


def test_duplicate_recomputes_a_end_and_b_start_and_end_without_altering_evidence():
    records=[dict(text='abcdefXY',line_id='a',bbox_px=[0,0,100,10],rebuilt_bbox_pt=[0,0,100,10],prefix_advance_pt=[n*12 for n in range(9)]),
             dict(text='XYghijkl',line_id='b',bbox_px=[80,0,180,10],rebuilt_bbox_pt=[80,0,180,10],prefix_advance_pt=[n*10 for n in range(9)])]
    original=deepcopy(records)
    result,events=deduplicate_records(records)
    assert events[0]['matched']==2
    assert result[0]['rebuilt_bbox_pt'][2]==96
    assert result[1]['rebuilt_bbox_pt']==[100,0,160,10]
    assert records==original
