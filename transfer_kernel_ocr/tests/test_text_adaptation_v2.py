from copy import deepcopy
from types import SimpleNamespace
import json
import runpy
from pathlib import Path

import pytest

from backend.four_layer_text import measure_four_layer, deduplicate_records
from backend.source_text import build_text_content
from backend.four_layer_regions import solve


class Font:
    ascender=0.8
    descender=-0.2

    def text_length(self,text,fontsize):
        return len(text)*fontsize*.5


@pytest.fixture
def fixed_font(monkeypatch):
    font=Font()
    library=SimpleNamespace(resolve=lambda *a,**k:SimpleNamespace(key='fixed'),export_font_for_key=lambda key:font)
    monkeypatch.setattr('backend.font_library.get_library',lambda:library)
    return font


def measured(texts,spacing=5,tolerance=.6,boxes=None):
    records=[dict(text=t,bbox_px=(boxes[i] if boxes else
        [0,i*18,len(t.replace(' ',''))*5+t.count(' ')*spacing,i*18+10]),line_id=str(i))
        for i,t in enumerate(texts)]
    content=build_text_content(records,'b',four_layer=True)
    source=dict(bbox_px=[0,0,100,len(texts)*20],bbox_pt=[0,0,100,len(texts)*20])
    return measure_four_layer(content,100,dict(font_size_pt=10,word_gap_pt=spacing,alignment_tolerance_ratio=tolerance),source)


def test_tightened_overflow_is_retained_before_shared_font_scale(fixed_font):
    # Both edges are within tolerance: widths 35,35,35,45. The final
    # row tightens to 40, and must scale from this retained gap state.
    result=measured(['aaa bbb']*3+['aaaa aaaa'],tolerance=1.)
    state=result['paragraph_adjustments'][0]
    assert state['deferred_row_ids']==[3]
    assert state['before_shrink_widths_pt'][3]==40
    assert state['font_scale']==pytest.approx(.875,abs=1e-7)
    assert all(line['font_size']==pytest.approx(8.75,abs=1e-6) for line in result['lines'])
    deferred=state['row_adjustments'][3]
    assert deferred['tightened_spacing_pt'][0]['latin_word_gap_pt']==0
    assert all(e['second_pass_complete'] for e in state['row_adjustments'])
    assert all(line['advance_width']==pytest.approx(35,abs=1e-6) for line in result['lines'])


def test_gap_adjustment_without_any_overflow_uses_different_row_values(fixed_font):
    result=measured(['aaa bbb','aaa bb','aaa bbb'])
    state=result['paragraph_adjustments'][0]
    assert not state['paragraph_font_scaled']
    assert result['lines'][1]['spacing_pt']['latin_word_gap_pt']==pytest.approx(10)
    assert result['lines'][0]['spacing_pt']['latin_word_gap_pt']==pytest.approx(5)
    assert all(line['advance_width']==pytest.approx(35) for line in result['lines'])


def test_two_rows_use_extremes_and_keep_short_right_endpoint(fixed_font):
    result=measured(['aaa bbb','x x'])
    state=result['paragraph_adjustments'][0]
    assert state['edge_method']=='two-row-extrema'
    assert not state['row_adjustments'][1]['align_right']
    assert result['lines'][1]['advance_width']==15


def test_no_expandable_gap_is_not_misclassified_as_too_many_characters(fixed_font):
    result=measured(['aaaaaa','aaaaa','aaaaaa'])
    state=result['paragraph_adjustments'][0]
    assert state['deferred_row_ids']==[]
    assert state['row_adjustments'][1]['first_pass_status']=='no-expandable-gap'


def test_right_only_without_slots_keeps_actual_start_and_reports_unmet_edge(fixed_font):
    result=measured(['aaaaaa','aaaaaa','aaaaaa','tail'],tolerance=.15,
        boxes=[[0,0,80,10],[5,18,80,28],[5,36,80,46],[5,54,25,64]])
    state=result['paragraph_adjustments'][0]['row_adjustments'][0]
    assert not state['align_left'] and state['align_right']
    assert state['status']=='no-expandable-gap'
    assert state['actual_endpoints'][0]==pytest.approx(0)
    assert state['actual_endpoints'][1] < state['target_endpoints'][1]
    assert not state['second_pass_complete']


def test_original_edges_align_rows_despite_different_rebuilt_widths(fixed_font):
    result=measured(['aaa bbb','a b','aaa bbb','tail'],tolerance=.15,
        boxes=[[0,0,80,10],[0,18,80,28],[0,36,80,46],[0,54,25,64]])
    state=result['paragraph_adjustments'][0]
    assert state['edge_basis']=='original-ocr-boxes'
    assert state['dominant_right_pt']==80
    assert [r['align_right'] for r in state['row_adjustments']]==[True,True,True,False]
    assert all(r['actual_endpoints'][1]==pytest.approx(80) for r in state['row_adjustments'][:3])


def test_two_rows_use_original_extremes_and_preserve_indent(fixed_font):
    result=measured(['a b','aaa bbb'],tolerance=.15,
        boxes=[[10,0,80,10],[0,18,80,28]])
    state=result['paragraph_adjustments'][0]
    assert (state['dominant_left_pt'],state['dominant_right_pt'])==(0,80)
    assert not state['row_adjustments'][0]['align_left']
    assert all(r['align_right'] for r in state['row_adjustments'])
    assert state['row_adjustments'][0]['actual_endpoints'][0]==pytest.approx(10)
    assert all(r['actual_endpoints'][1]==pytest.approx(80) for r in state['row_adjustments'])


def duplicate_records(source_b=80,target_b=80):
    return [dict(text='abcdefXY',line_id='a',bbox_px=[0,0,100,10],rebuilt_bbox_pt=[0,0,100,10],prefix_advance_pt=list(range(9))),
            dict(text='XYghijkl',line_id='b',bbox_px=[source_b,0,source_b+100,10],
                 rebuilt_bbox_pt=[target_b,0,target_b+100,10],prefix_advance_pt=[n*10 for n in range(9)])]


def test_source_overlap_controls_alpha_with_rebuilt_character_area():
    records=duplicate_records(source_b=100,target_b=80)
    assert deduplicate_records(records)==(records,[])
    records=duplicate_records(target_b=90)
    # Source intersection 200 is used even when rebuilt intersection is 100.
    cleaned,events=deduplicate_records(records)
    assert events[0]['alpha']==2
    assert events[0]['overlap_area_pt2']==200
    assert events[0]['rebuilt_overlap_area_pt2']==100
    assert cleaned[1]['text']=='ghijkl'
    records=duplicate_records()
    original=deepcopy(records)
    cleaned,events=deduplicate_records(records)
    assert cleaned[1]['text']=='ghijkl'
    assert cleaned[1]['rebuilt_start_offset_pt']==20
    assert events[0]['overlap_area_pt2']==200
    assert records==original


def test_region_uniform_transform_scales_positions_and_gap():
    items=[dict(id='a',kind='text',source=dict(bbox_pt=[0,0,50,20])),
           dict(id='b',kind='text',source=dict(bbox_pt=[60,0,110,20]))]
    sizes=dict(a=dict(width=50,height=20),b=dict(width=50,height=20))
    pos,_,scale,detail=solve(items,sizes,[0,0,100,30],uniform_region=True)
    assert scale==pytest.approx(100/110)
    assert pos['b'][0]-pos['a'][0]-50*scale==pytest.approx(10*scale)
    assert detail['gap_policy']=='whole-region-positions-and-distances-scaled'


def test_alpha_source_area_is_converted_to_points_independently_of_rebuilt_boxes():
    records=duplicate_records(target_b=90)
    for record in records:
        record['bbox_px']=[v*2 for v in record['bbox_px']]
        record['source_bbox_pt']=[v/2 for v in record['bbox_px']]
    _,events=deduplicate_records(records)
    assert events[0]['overlap_area_pt2']==200
    assert events[0]['rebuilt_overlap_area_pt2']==100
    assert events[0]['alpha']==2


def test_figure_caption_keeps_anchors_spacing_and_size_despite_matching_edges(fixed_font):
    texts=['caption first line', 'caption second line', 'short end']
    records=[dict(text=t,bbox_px=[0,i*18,60,i*18+10],line_id=str(i)) for i,t in enumerate(texts)]
    content=build_text_content(records,'caption',four_layer=True)
    source=dict(label='figure_title',bbox_px=[0,0,100,60],bbox_pt=[0,0,100,60])
    m=measure_four_layer(content,100,dict(font_size_pt=10),source)
    state=m['paragraph_adjustments'][0]
    assert state['edge_policy']=='preserve-caption-anchors'
    assert not state['paragraph_font_scaled']
    assert all(not e['align_left'] and not e['align_right'] for e in state['row_adjustments'])
    assert [l['font_size'] for l in m['lines']]==[10,10,10]
    assert all(e['initial_spacing_pt']==e['final_spacing_pt'] for e in state['row_adjustments'])


def test_duplicate_prefix_can_match_inside_the_alpha_tail():
    records=duplicate_records(source_b=70,target_b=70)
    records[0]['text']='abcdeXYZ'
    records[1]['text']='XYQghijk'
    cleaned,events=deduplicate_records(records)
    assert events[0]['alpha']==3
    assert events[0]['matched']==2
    assert cleaned[1]['text']=='Qghijk'
    assert records[1]['text']=='XYQghijk'


def test_new_policy_saves_pending_without_replacing_text(tmp_path,monkeypatch):
    from backend.regional_layout import RegionLayoutError
    from backend.layout_pending import LayoutPendingError
    original=dict(layout_options=dict(text_adaptation_version='20261103'),blocks=[dict(id='keep',kind='text')])
    saved=deepcopy(original)
    error=RegionLayoutError('源块重叠关系不明确',dict(id='f'),dict(id='p'))
    def fail(project):raise error
    monkeypatch.setattr('backend.adaptive_layout.layout_document',fail)
    from backend.region_recovery import layout_with_region_recovery
    module={'layout_with_region_recovery':layout_with_region_recovery}
    with pytest.raises(LayoutPendingError):module['layout_with_region_recovery'](original,tmp_path)
    assert original==saved
    snapshot=json.loads((tmp_path/'layout-pending.json').read_text(encoding='utf8'))
    assert snapshot['project']['blocks'][0]['kind']=='text'
    assert snapshot['status']=='waiting-for-layout-resolution'
