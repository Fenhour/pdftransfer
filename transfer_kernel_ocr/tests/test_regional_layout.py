"""Geometry truth for source-preserving 08; no OCR/model accuracy claims."""
from copy import deepcopy

import pytest

from backend.adaptive_layout import layout_document, measure_block
from backend.adaptive_export import _validate
from backend.regional_layout import scale_measurement
from backend.spatial_relations import derive_spatial_relations


def text(identifier, box, value='Word', size=10):
    return dict(id=identifier,kind='text',source=dict(page_id='p',label='text',bbox_pt=box,bbox_px=box),
                content=dict(text=value),style=dict(font_size_pt=size,line_height=1.2),
                local_layout={},quality=dict(state='needs_review',issues=[]))


def project(blocks, regions=()):
    p=dict(blocks=blocks,source_pages=[dict(id='p',width_px=400,height_px=400,width_pt=400,height_pt=400,
                  detections=dict(regions=[dict(id=f'r{i}',coordinate=b) for i,b in enumerate(regions)]))])
    p['relations']=derive_spatial_relations(p)
    return p


def placement(p, identifier):
    return next(v for v in p['layout_result']['placements'] if v['block_id']==identifier)


def test_original_anchors_indentation_and_whitespace_remain_when_content_fits():
    a=text('a',[32,30,160,70]);b=text('b',[49,110,170,150]);c=text('c',[240,30,350,70])
    p=project([a,b,c]);source=deepcopy([v['source'] for v in p['blocks']])
    layout_document(p)
    assert [(placement(p,i)['bbox'][:2]) for i in ('a','b','c')]==[[32,30],[49,110],[240,30]]
    assert all(v['uniform_scale']==1 for v in p['layout_result']['region_adjustments'])
    assert [v['source'] for v in p['blocks']]==source
    _validate(p)


def test_growth_uses_existing_whitespace_before_moving_a_neighbor():
    a=text('a',[20,20,120,35],value='Line\nLine');b=text('b',[20,80,120,105])
    p=project([a,b]);layout_document(p)
    assert placement(p,'b')['bbox'][1]==80
    assert p['layout_result']['region_adjustments'][0]['moved_block_ids']==[]
    _validate(p)


def test_region_uniform_shrink_keeps_other_region_exactly_unchanged():
    boxes=[[10,10,180,110],[210,10,390,110]]
    blocks=[text('a',[20,20,150,40]),text('b',[20,80,150,100]),text('other',[220,20,360,40])]
    p=project(blocks,boxes);layout_document(p);before=deepcopy(placement(p,'other'))
    blocks[1]['content']['text']='Line\nLine\nLine'
    layout_document(p)
    adjusted=placement(p,'b');factor=adjusted['measurement']['uniform_scale']
    assert .85<=factor<1
    assert placement(p,'a')['measurement']['uniform_scale']==factor
    assert placement(p,'other')==before
    assert p['layout_result']['resolved_order']==p['relations']['source_order']
    assert p['blocks'][1]['local_layout']['measurement']['font_size']==10
    assert adjusted['measurement']['font_size']==pytest.approx(10*factor)
    _validate(p)


def test_collision_moves_only_same_column_within_region_and_preserves_x_offset():
    blocks=[text('a',[20,20,150,35],value='Line\nLine\nLine'),
            text('b',[33,40,150,60]),text('other',[220,20,360,40])]
    p=project(blocks,[[10,10,180,180],[210,10,390,180]])
    layout_document(p)
    assert placement(p,'b')['bbox'][0]==33
    assert placement(p,'b')['bbox'][1]>=placement(p,'a')['bbox'][3]
    assert placement(p,'other')['bbox'][:2]==[220,20]
    assert placement(p,'b')['measurement']['uniform_scale']==1
    _validate(p)


def test_excessive_growth_reports_region_and_does_not_silently_spill():
    p=project([text('a',[20,20,150,80],value='Line\n'*30)],[[10,10,180,100]])
    with pytest.raises(ValueError,match='可读下限.*未挤入其他大区域'):
        layout_document(p)
    assert 'layout_result' not in p


def test_font_floor_is_checked_even_when_geometric_scale_is_allowed():
    p=project([text('a',[20,60,150,80],value='Line\nLine\nLine',size=7)],[[10,10,180,80]])
    with pytest.raises(ValueError,match='可读下限'):
        layout_document(p)


def test_source_pages_stay_separate_with_individual_page_sizes():
    a=text('a',[20,20,150,60]);b=text('b',[40,40,180,80]);b['source']['page_id']='second'
    p=project([a]);second=dict(id='second',width_px=500,height_px=300,width_pt=500,height_pt=300,detections=dict(regions=[]))
    p['source_pages'].append(second);p['blocks'].append(b);p['relations']=derive_spatial_relations(p)
    layout_document(p)
    assert [(v['width'],v['height']) for v in p['layout_result']['pages']]==[(400,400),(500,300)]
    assert placement(p,'a')['page_index']==0 and placement(p,'b')['page_index']==1
    assert placement(p,'b')['bbox'][:2]==[40,40]
    _validate(p)


def test_source_composite_keeps_children_offsets_and_original_height():
    children=[text('a',[40,40,160,60]),text('b',[55,90,170,110])]
    block=dict(id='panel',kind='composite',source=dict(page_id='p',label='container',bbox_pt=[20,20,190,150]),
               content=dict(children=children,layout_mode='source-relative',background=dict(fill='#ffffff'),
                            decorations=dict(border_color='#000000',border_width_pt=1)),style={},local_layout={})
    m=measure_block(block,170)
    assert [(v['x'],v['y']) for v in m['children']]==[(20,20),(35,70)]
    assert m['height']==130
    scaled=scale_measurement(m,.9)
    assert scaled['children'][1]['x']==31.5
    assert scaled['children'][0]['measurement']['font_size']==9
    assert scaled['decorations']['border_width_pt']==.9
    assert m['children'][0]['measurement']['font_size']==10
    assert scaled['children'][0]['measurement']['lines'][-1]['end']==4


def test_table_scale_keeps_cell_identity_spans_and_content_ranges():
    block=dict(id='table',kind='table',source=dict(page_id='p',label='table',bbox_pt=[20,20,160,80]),
               content=dict(row_count=2,column_count=2,cells=[
                   dict(id='merged',row=0,column=0,rowspan=2,colspan=1,text='Merged'),
                   dict(id='b',row=0,column=1,rowspan=1,colspan=1,text='One'),
                   dict(id='c',row=1,column=1,rowspan=1,colspan=1,text='Two')]),
               style=dict(font_size_pt=10),local_layout={})
    original=measure_block(block,140);scaled=scale_measurement(original,.9)
    assert scaled['breaks']==original['breaks']
    assert scaled['row_offsets']==pytest.approx([v*.9 for v in original['row_offsets']])
    for a,b in zip(original['cells'],scaled['cells']):
        assert {k:b[k] for k in ('id','row','column','rowspan','colspan','text')}=={k:a[k] for k in ('id','row','column','rowspan','colspan','text')}
        assert b['font_size']==9
        assert b['height']==pytest.approx(a['height']*.9)


def test_container_rewrap_uses_original_child_relation_and_records_movement():
    from backend.regional_layout import measure_source_composite
    children=[text('a',[90,10,101,20]),text('b',[90,25,99,35])]
    block=dict(source=dict(bbox_pt=[0,0,100,100]),content=dict(children=children,padding_pt=0))
    def measure(child,width):
        return dict(width=width,height=30 if child['id']=='a' and width<11 else 10,min_width=1)
    m=measure_source_composite(block,100,measure)
    assert m['children'][1]['y']==42
    assert m['local_adjustments']==['b']
    assert children[0]['source']['bbox_pt']==[90,10,101,20]


def test_new_collisions_are_not_allowed_as_source_layering():
    p=project([text('a',[20,20,150,80]),text('b',[40,30,170,90])])
    p['blocks'][0]['content']['text']='Line\n'*4
    with pytest.raises(ValueError,match='重叠关系不明确'):
        layout_document(p)


def test_smaller_contents_keep_source_envelope_without_overflow():
    p=project([text('a',[20,20,150,100],value='Short')],[[10,10,180,150]])
    layout_document(p)
    m=placement(p,'a')['measurement']
    assert m['height']==80
    assert p['blocks'][0]['local_layout']['preserve_source_envelope'] is True
    assert 'envelope_compaction' not in p['layout_result']['region_adjustments'][0]
    _validate(p)


def test_readable_uniform_region_fit_keeps_unused_block_space():
    p=project([text('a',[20,20,150,40],value='Line\n'*4),
               text('b',[20,45,150,90])],[[10,10,180,105]])
    layout_document(p)
    adjustment=p['layout_result']['region_adjustments'][0]
    assert .85 <= adjustment['region_scale'] < 1
    assert 'envelope_compaction' not in adjustment
    assert p['blocks'][1]['local_layout']['measurement']['height']==45
    _validate(p)


def test_only_unreadable_region_fit_releases_unused_trailing_text_envelope():
    blocks=[text('a',[20,20,150,40],value='Line\n'*4),
            text('b',[20,45,150,90]),text('other',[220,20,360,80])]
    p=project(blocks,[[10,10,180,95],[210,10,390,110]])
    # Other Region has no reason to adapt and must remain exactly unchanged.
    isolated=project([deepcopy(blocks[2])],[[210,10,390,110]])
    layout_document(isolated)
    expected=deepcopy(placement(isolated,'other'))
    layout_document(p)
    adjustment=p['layout_result']['region_adjustments'][0]
    compact=adjustment['envelope_compaction']
    assert compact['first_region_scale'] < .85
    assert compact['block_ids']==['b']
    assert compact['reason']=='region-overflow-uniform-fit-below-readable-floor'
    assert p['blocks'][1]['local_layout']['measurement']['height'] < 45
    assert placement(p,'other')['bbox']==expected['bbox']
    assert placement(p,'other')['measurement']==expected['measurement']
    _validate(p)
