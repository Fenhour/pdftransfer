"""Source geometry first, with bounded changes confined to one source frame."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from .stage_timing import timed_step


LAYOUT_VERSION = '08-four-layer-v5-bounded'
POLICY = 'source-preserving-regions-v1'
_EPS = .05


class RegionLayoutError(ValueError):
    """A bounded geometry failure with the source frame retained for recovery."""

    def __init__(self, message, frame, page):
        super().__init__(message)
        self.frame = deepcopy(frame)
        self.page = deepcopy(page)


def recoverable_geometry_error(exc):
    return any(token in str(exc) for token in (
        '越出', '越界', '溢出', '超过可读下限', '区域内碰撞无法有界解决',
        '重叠关系不明确', '需继续布局处理'))


def _signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def _rect(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('保原版面需要完整源坐标')
    result = list(map(float, value))
    if (not all(math.isfinite(v) for v in result) or
            result[2] <= result[0] or result[3] <= result[1]):
        raise ValueError('保原版面的源坐标无效')
    return result


def _overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def image_crop_bbox(block):
    """Actual integer raster crop, whose origin can differ from the detector."""
    transform = block.get('source', {}).get('crop_to_page')
    pixels = block.get('content', {}).get('pixel_size')
    if (not isinstance(transform, list) or len(transform) != 3 or
            transform[0][:2] != [1,0] or transform[1][:2] != [0,1] or
            transform[2] != [0,0,1] or not isinstance(pixels, list) or len(pixels) != 2):
        return None
    x, y = transform[0][2], transform[1][2]
    return _rect([x,y,x+pixels[0],y+pixels[1]])


def image_source_bbox(block):
    """Physical raster footprint; preserve aspect and common page-pixel mapping."""
    source = block['source']
    original = _rect(source['bbox_pt'])
    crop = image_crop_bbox(block) if block['kind']=='image' else None
    pixels = source.get('bbox_px')
    if crop is None or not pixels:
        return original
    pixels = _rect(pixels)
    sx, sy = (original[2]-original[0])/(pixels[2]-pixels[0]), (original[3]-original[1])/(pixels[3]-pixels[1])
    # The page renderer rounds its pixel dimensions; one uniform pixel scale
    # keeps adjacent alpha masks aligned instead of stretching individual crops.
    scale = min(sx,sy)
    ox, oy = original[0]-pixels[0]*sx, original[1]-pixels[1]*sy
    return [ox+crop[0]*scale,oy+crop[1]*scale,ox+crop[2]*scale,oy+crop[3]*scale]


def owned_image_pair(first, second):
    """A raster mask explicitly removes the other block's shared pixels.

    This is evidence for preserving an existing overlap, never permission to
    introduce a new one. The caller must also verify a common source transform.
    """
    if first['kind'] != 'image' or second['kind'] != 'image':
        return False
    if first['source'].get('page_id') != second['source'].get('page_id'):
        return False
    for masked, other in ((first, second), (second, first)):
        ownership = masked['source'].get('ownership', {})
        identifier = other['source'].get('layout_detection_id')
        if not identifier or identifier not in ownership.get('excluded_block_ids', []):
            continue
        # Full reference crops are not the alpha-masked display assets.
        asset = masked.get('content', {}).get('asset')
        if not asset or asset == masked['source'].get('reference_asset'):
            continue
        a = image_crop_bbox(masked); b = image_crop_bbox(other)
        if not a or not b:
            continue
        intersection = [max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])]
        if intersection[2] <= intersection[0] or intersection[3] <= intersection[1]:
            continue
        excluded_ids = ownership.get('excluded_block_ids', [])
        excluded_boxes = ownership.get('excluded_rects', [])
        if len(excluded_ids) != len(excluded_boxes):
            continue
        for detection_id, excluded in zip(excluded_ids, excluded_boxes):
            if detection_id != identifier:
                continue
            r = _rect(excluded)
            if (r[0] <= intersection[0] and r[1] <= intersection[1] and
                    r[2] >= intersection[2] and r[3] >= intersection[3]):
                return True
    return False


@timed_step('08','apply_scale')
def scale_measurement(measured, factor):
    """Scale physical lengths only; never change source ranges or topology."""
    factor = float(factor)
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError('区域缩放必须为 (0,1] 的等比因子')
    result = deepcopy(measured)
    for key in ('x', 'y', 'width', 'height', 'font_size', 'line_step', 'min_width', 'padding',
                'border_width', 'available_width', 'source_envelope_height', 'occupied_height'):
        if key in result:
            result[key] *= factor
    for key in ('column_widths', 'row_heights', 'row_offsets'):
        if key in result:
            result[key] = [v*factor for v in result[key]]
    for line in result.get('lines', []):
        for key in ('x', 'baseline', 'height', 'font_size','advance_width'):
            if key in line:
                line[key] *= factor
        for run in line.get('text_runs',[]):
            run['x'] *= factor
        for key in line.get('spacing_pt',{}):
            line['spacing_pt'][key] *= factor
    for adjustment in result.get('internal_adjustments', []):
        for key in ('shift_pt', 'font_size'):
            if key in adjustment:
                adjustment[key] *= factor
    for cell in result.get('cells', []):
        scaled = scale_measurement(cell, factor)
        cell.clear()
        cell.update(scaled)
    for child in result.get('children', []):
        child['x'] *= factor
        child['y'] *= factor
        child['measurement'] = scale_measurement(child['measurement'], factor)
    if 'border_width_pt' in result.get('decorations', {}):
        result['decorations']['border_width_pt'] *= factor
    result['uniform_scale'] = measured.get('uniform_scale', 1.)*factor
    result['source_measurement_signature'] = measured.get('source_measurement_signature', measured.get('signature'))
    result['signature'] = _signature(result)
    result['scaling_result_signature'] = result['signature']
    return result


def _font_sizes(measured):
    if 'font_size' in measured:
        yield float(measured['font_size'])
    for line in measured.get('lines', []):
        if 'font_size' in line:
            yield float(line['font_size'])
    for cell in measured.get('cells', []):
        yield from _font_sizes(cell)
    for child in measured.get('children', []):
        yield from _font_sizes(child['measurement'])


def _arrange(items, measurements, frame, spatial_edges=()):
    """Keep source anchors; consume whitespace before moving colliding siblings.

    A source pair separated vertically may move down; a horizontal pair may
    move right. An unresolved overlap is reported rather than guessing layering.
    Reading order is deliberately not an input to this physical adjustment.
    """
    left, top, _, _ = frame
    from .source_containment import source_contains_pair
    boxes = {b['id']: image_source_bbox(b) for b in items}
    positions = {i: [b[0]-left+measurements[i].get('block_expansion',{}).get('left_pt',0.),
                     b[1]-top+measurements[i].get('block_expansion',{}).get('top_pt',0.)] for i, b in boxes.items()}
    gaps = {}
    for edge in spatial_edges:
        a, b = edge.get('a'), edge.get('b')
        if a in boxes and b in boxes:
            gaps[a, b, edge.get('relation')] = max(0., float(edge.get('gap_pt', 0)))
    identifiers = sorted(boxes, key=lambda i: (boxes[i][1], boxes[i][0], i))
    by_id = {b['id']:b for b in items}

    def target(i):
        x, y = positions[i]
        m = measurements[i]
        return [x, y, x+m['width'], y+m['height']]

    moved = set()
    for _ in range(len(items)+1):
        changed = False
        for index, a in enumerate(identifiers):
            for b in identifiers[index+1:]:
                if source_contains_pair(by_id[a], by_id[b]):
                    continue
                ta, tb = target(a), target(b)
                if _overlap(ta, tb) <= _EPS:
                    continue
                sa, sb = boxes[a], boxes[b]
                if owned_image_pair(by_id[a], by_id[b]) and all(
                        abs(v-w) <= _EPS for i, source in ((a,sa),(b,sb))
                        for v,w in zip(target(i), [source[0]-left,source[1]-top,source[2]-left,source[3]-top])):
                    continue
                # Small detector-edge overlaps have a stable source direction.
                if sa[3] <= sb[1]+1. or (sa[1] < sb[1] and
                        min(sa[3], sb[3])-sb[1] <= .1*min(sa[3]-sa[1], sb[3]-sb[1])):
                    gap = min(2., gaps.get((a, b, 'above'), max(0., sb[1]-sa[3])))
                    positions[b][1] = ta[3]+gap
                    moved.add(b)
                elif sa[2] <= sb[0]+1.:
                    gap = min(2., gaps.get((a, b, 'left_of'), max(0., sb[0]-sa[2])))
                    positions[b][0] = ta[2]+gap
                    moved.add(b)
                elif sb[2] <= sa[0]+1.:
                    gap = min(2., gaps.get((b, a, 'left_of'), max(0., sa[0]-sb[2])))
                    positions[a][0] = tb[2]+gap
                    moved.add(a)
                else:
                    raise ValueError(f'源块 {a}/{b} 重叠关系不明确，需校对；未猜测层叠或跨区域移动')
                changed = True
        if not changed:
            return positions, sorted(moved)
    raise ValueError('区域内碰撞无法有界解决，需校对；未向其他区域传播')


def measure_source_composite(block, width, measure):
    """Keep child offsets/widths; only propagate actual local growth."""
    source = _rect(block['source']['bbox_pt'])
    ratio = width/(source[2]-source[0])
    children = block['content'].get('children', [])
    if not children or any(c['kind'] != 'text' for c in children):
        raise ValueError('组合块目前仅接受文字子块')
    # Horizontal source dimensions adapt to the requested block width. Vertical
    # whitespace and offsets use the same ratio, independently of font sizing.
    local = []
    measurements = {}
    for child in children:
        box = _rect(child['source']['bbox_pt'])
        child_copy = dict(child, source=dict(child['source'], bbox_pt=[
            (box[0]-source[0])*ratio, (box[1]-source[1])*ratio,
            (box[2]-source[0])*ratio, (box[3]-source[1])*ratio]))
        local.append(child_copy)
        measurements[child['id']] = measure(child, (box[2]-box[0])*ratio)
    positions, moved = _arrange(local, measurements, [0., 0., width, (source[3]-source[1])*ratio])
    padding = max(0., float(block['content'].get('padding_pt', 4)))
    # Membership tolerances do not authorize negative target child coordinates.
    dx = max(0., -min(p[0] for p in positions.values()))
    dy = max(0., -min(p[1] for p in positions.values()))
    values = [dict(block_id=c['id'], x=positions[c['id']][0]+dx,
                   y=positions[c['id']][1]+dy, measurement=measurements[c['id']]) for c in children]
    right = max(v['x']+v['measurement']['width'] for v in values)
    if right > width+.05:
        # A slightly protruding source child gets its available width measured
        # again, retaining its text and the other children's source offsets.
        for v, child in zip(values, children):
            if v['x']+v['measurement']['width'] > width:
                v['measurement'] = measure(child, width-v['x'])
        measurements = {v['block_id']:v['measurement'] for v in values}
        shifted = [dict(c, source=dict(c['source'], bbox_pt=[c['source']['bbox_pt'][0]+dx,
                       c['source']['bbox_pt'][1]+dy,c['source']['bbox_pt'][2]+dx,
                       c['source']['bbox_pt'][3]+dy]))
                   for c, v in zip(local, values)]
        pos, more_moved = _arrange(shifted, measurements, [0.,0.,width,(source[3]-source[1])*ratio])
        moved = sorted(set(moved) | set(more_moved))
        for v in values:
            v['x'], v['y'] = pos[v['block_id']]
    height = max((source[3]-source[1])*ratio,
                 max(v['y']+v['measurement']['height']+padding for v in values))
    return dict(width=width, height=height, children=values, breaks=[],
                background=deepcopy(block['content'].get('background', {})),
                decorations=deepcopy(block['content'].get('decorations', {})),
                min_width=max(v['measurement'].get('min_width', 0)+v['x'] for v in values),
                local_adjustments=moved, internal_layout='source-relative')


def _restore_clear_text_anchors(items, measurements, positions, factor, frame, moved, attachments=()):
    """Keep unrelated text at its source anchor if the final scene permits it.

    The existing block solver still resolves colliding blocks. An untouched
    heading/caption need not inherit another block's Region scale/translation.
    Each restoration is checked against the complete final sibling scene.
    """
    left,top,right,bottom=frame
    from .source_containment import source_contains_pair
    by_id={b['id']:b for b in items}
    current={i:list(p) for i,p in positions.items()}
    scales={b['id']:factor for b in items}
    restored=[]
    def rect(identifier):
        x,y=current[identifier];m=measurements[identifier];s=scales[identifier]
        return [x,y,x+m['width']*s,y+m['height']*s]
    for b in items:
        identifier=b['id']
        if b['kind']!='text' or identifier in moved:
            continue
        if any(link.get('keep')=='required' and identifier in {link.get('anchor'),link.get('dependent')} and
               ({link.get('anchor'),link.get('dependent')}-{identifier}) & set(moved) for link in attachments):
            continue
        source=image_source_bbox(b);m=measurements[identifier];expansion=m.get('block_expansion',{})
        anchor=[source[0]-left+expansion.get('left_pt',0.),source[1]-top+expansion.get('top_pt',0.)]
        candidate=[*anchor,anchor[0]+m['width'],anchor[1]+m['height']]
        if (candidate[0]<-_EPS or candidate[1]<-_EPS or
                candidate[2]>right-left+_EPS or candidate[3]>bottom-top+_EPS):
            continue
        if any(_overlap(candidate,rect(other))>_EPS for other in current if other!=identifier
               and not source_contains_pair(b,by_id[other])):
            continue
        if (abs(current[identifier][0]-anchor[0])>1e-8 or abs(current[identifier][1]-anchor[1])>1e-8 or
                abs(scales[identifier]-1.)>1e-8):
            current[identifier]=anchor;scales[identifier]=1.;restored.append(identifier)
    return current,scales,restored


def layout_regions(project, page_size=None, margin=24):
    from .adaptive_layout import measure_block, _topological_order, ENGINE_VERSION

    blocks = project.get('blocks', [])
    by_id = {b['id']:b for b in blocks}
    if len(by_id) != len(blocks):
        raise ValueError('重建块 ID 不唯一')
    relations = project['relations']
    order = list(relations.get('source_order', []))
    if len(order) != len(set(order)) or set(order) != set(by_id):
        raise ValueError('参考阅读顺序必须完整且唯一覆盖所有块')
    resolved, _ = _topological_order(order, relations.get('hard_before', []))
    policy = project.get('layout_options', {})
    minimum_scale = float(policy.get('minimum_region_scale', .85))
    minimum_font = float(policy.get('minimum_font_pt', 7.))
    if (not math.isfinite(minimum_scale) or not 0 < minimum_scale <= 1 or
            not math.isfinite(minimum_font) or minimum_font <= 0):
        raise ValueError('区域缩放下限无效')
    pages, placements, frames, adjustments, issues = [], [], [], [], []
    assigned = set()
    for page in project['source_pages']:
        sw = float(page.get('width_pt', page.get('width_px', 0)))
        sh = float(page.get('height_pt', page.get('height_px', 0)))
        _rect([0.,0.,sw,sh])
        pw, ph = map(float, page_size or (sw,sh))
        _rect([0.,0.,pw,ph])
        page_factor = min(1., pw/sw, ph/sh)
        ox, oy = (pw-sw*page_factor)/2, (ph-sh*page_factor)/2
        page_index = len(pages)
        pages.append(dict(index=page_index,width=pw,height=ph,source_page_id=page['id']))
        source_frames = [f for f in relations.get('frames', []) if f['page_id']==page['id']]
        isolated = any(f.get('kind')=='region' and f.get('isolated') for f in source_frames)
        selected = []
        for frame in source_frames:
            if isolated and frame.get('kind')=='page':
                # An unassigned item must not move an already isolated Region.
                for identifier in frame['block_ids']:
                    selected.append(dict(id=frame['id']+'-'+identifier,page_id=page['id'],kind='source-block',
                                         isolated=True,bbox_pt=by_id[identifier]['source']['bbox_pt'],block_ids=[identifier]))
            else:
                selected.append(frame)
        for frame in selected:
            identifiers = frame['block_ids']
            if not identifiers:
                continue
            if any(i in assigned or i not in by_id or by_id[i]['source']['page_id']!=page['id'] for i in identifiers):
                raise ValueError('区域归属重复或引用了错误的模块')
            assigned.update(identifiers)
            measured, positions = {}, {}
            try:
                box = _rect(frame['bbox_pt'])
                if min(box[0],box[1]) < 0 or box[2]>sw+.05 or box[3]>sh+.05:
                    raise ValueError('隔离区域越出源页面')
                members = [by_id[i] for i in identifiers]
                def measure_member(b, preserve=True):
                    sb=image_source_bbox(b)
                    extra=0.
                    text_bounds=None
                    if policy.get('four_layer') and b['kind']=='text':
                        limit=box[2]; left_limit=box[0];top_limit=box[1];bottom_limit=box[3]
                        for other in members:
                            ob=image_source_bbox(other)
                            if other is not b and ob[0]>=sb[2] and min(sb[3],ob[3])>max(sb[1],ob[1]):
                                limit=min(limit,ob[0])
                            if other is not b and ob[2]<=sb[0] and min(sb[3],ob[3])>max(sb[1],ob[1]):
                                left_limit=max(left_limit,ob[2])
                            if other is not b and min(sb[2],ob[2])>max(sb[0],ob[0]):
                                if ob[3]<=sb[1]:top_limit=max(top_limit,ob[3])
                                if ob[1]>=sb[3]:bottom_limit=min(bottom_limit,ob[1])
                        extra=max(0.,limit-sb[2])
                        text_bounds=[max(0.,left_limit)-sb[0],max(0.,top_limit)-sb[1],
                                     min(sw,limit)-sb[0],min(sh,bottom_limit)-sb[1]]
                    return measure_block(b,sb[2]-sb[0],preserve_envelope=preserve,whitespace_right_pt=extra,
                                         text_bounds_pt=text_bounds)
                measured = {b['id']:measure_member(b) for b in members}
                for identifier,value in measured.items():
                    detail=value.get('text_linkage',{})
                    if detail.get('status') in {'infeasible','undetermined'}:
                        # Internal layout problems remain editable and visible;
                        # no whole-region image fallback or silent success.
                        issues.append(dict(frame_id=frame['id'],block_id=identifier,source_page_id=page['id'],
                            physical_page_number=page.get('physical_page_index',0)+1,
                            reason='unresolved-text-linkage',action='export-current-geometry-and-continue',
                            text_linkage=deepcopy(detail)))
                fw, fh = box[2]-box[0], box[3]-box[1]
                solver_diagnostics = {}
                def fit(values):
                    if policy.get('four_layer'):
                        from .four_layer_regions import solve
                        pos,changed,scale,detail=solve(members,values,box,
                            empty_bands_enabled=bool(policy.get('empty_bands',False)),
                            attachments=relations.get('attachments',[]),
                            minimum_scale=minimum_scale,minimum_font=minimum_font,page_factor=page_factor,
                            max_rounds=policy.get('max_layout_rounds',16),
                            uniform_region=policy.get('text_adaptation_version') in {'20261103','20261104'})
                        solver_diagnostics.clear(); solver_diagnostics.update(detail)
                        return pos,changed,[0.,0.,fw,fh],scale,True
                    pos, changed = _arrange(members, values, box, relations.get('spatial_edges', []))
                    bounds = [min(0., *(p[0] for p in pos.values())),
                              min(0., *(p[1] for p in pos.values())),
                              max(fw, *(pos[i][0]+values[i]['width'] for i in identifiers)),
                              max(fh, *(pos[i][1]+values[i]['height'] for i in identifiers))]
                    scale = min(1., fw/(bounds[2]-bounds[0]), fh/(bounds[3]-bounds[1]))
                    sizes = [v for m in values.values() for v in _font_sizes(m)]
                    total = scale*page_factor
                    readable = (total >= minimum_scale-1e-8 and
                                all(s*total >= min(minimum_font,s)-1e-8 for s in sizes))
                    return pos, changed, bounds, scale, readable

                compacted = []
                try:
                    positions, moved, bounds, factor, readable = fit(measured)
                except ValueError as first_error:
                    if not policy.get('four_layer') or not recoverable_geometry_error(first_error):
                        raise
                    # Only trailing free text envelope is expendable. No row
                    # gap, raster pixels, table geometry or panel is trimmed.
                    candidate=dict(measured)
                    for b in members:
                        if b['kind']=='text':
                            compact=measure_member(b,False)
                            if compact['height']<measured[b['id']]['height']-_EPS:
                                candidate[b['id']]=compact; compacted.append(b['id'])
                    if not compacted:
                        raise
                    positions,moved,bounds,factor,readable=fit(candidate)
                    measured=candidate
                first_factor = factor
                # Only an actual local overflow followed by an unreadable uniform
                # fit permits releasing unused text envelope. Internal row gaps,
                # table rows, image pixels and panel borders remain intact.
                if factor < 1.-1e-8 and not readable:
                    candidate = dict(measured)
                    candidate_blocks = {}
                    for b in members:
                        if b['kind'] != 'text':
                            continue
                        copy = deepcopy(b)
                        compact = measure_block(copy, measured[b['id']]['width'], preserve_envelope=False)
                        if compact['height'] < measured[b['id']]['height']-_EPS:
                            candidate[b['id']] = compact
                            candidate_blocks[b['id']] = copy
                    if candidate_blocks:
                        pos, changed, cbounds, scale, accepted = fit(candidate)
                        if accepted and scale > factor+1e-8:
                            positions, moved, bounds, factor, readable = pos, changed, cbounds, scale, True
                            measured = candidate
                            compacted = sorted(candidate_blocks)
                            for b in members:
                                if b['id'] in candidate_blocks:
                                    b['local_layout'] = candidate_blocks[b['id']]['local_layout']
                x0,y0,x1,y1 = bounds
                total_factor = factor*page_factor
                if not readable:
                    raise ValueError(f'区域 {frame["id"]} 需要缩放至 {total_factor:.3f}，超过可读下限；需校对样式或区域，未挤入其他大区域')
                target = [ox+box[0]*page_factor,oy+box[1]*page_factor,
                          ox+box[2]*page_factor,oy+box[3]*page_factor]
                frames.append(dict(id=frame['id'],source_page_id=page['id'],page_index=page_index,
                                   bbox=target,source_bbox_pt=box,block_ids=list(identifiers),kind=frame['kind']))
                member_scales={identifier:factor for identifier in identifiers}
                preserved=[]
                if policy.get('four_layer'):
                    positions,member_scales,preserved=_restore_clear_text_anchors(
                        members,measured,positions,factor,box,moved,relations.get('attachments',[]))
                adjustment = dict(frame_id=frame['id'],uniform_scale=total_factor,region_scale=factor,
                                  page_scale=page_factor,moved_block_ids=moved,
                                  source_bbox_pt=box,target_bbox=target,local_content_bounds=[x0,y0,x1,y1])
                if preserved:
                    adjustment.update(source_anchor_restored_ids=preserved,member_region_scales=member_scales)
                if solver_diagnostics:
                    adjustment['solver']=deepcopy(solver_diagnostics)
                if compacted:
                    adjustment.update(envelope_compaction=dict(
                        reason='region-overflow-uniform-fit-below-readable-floor',
                        first_region_scale=first_factor, block_ids=compacted,
                        scope='unused-trailing-text-envelope-only'))
                adjustments.append(adjustment)
                if factor < 1.-1e-8 or moved:
                    issues.append(dict(frame_id=frame['id'],reason='region-local-adaptation',uniform_scale=total_factor))
                for identifier in identifiers:
                    member_factor=member_scales[identifier]*page_factor
                    m = scale_measurement(measured[identifier], member_factor)
                    position_factor=page_factor if policy.get('four_layer') else total_factor
                    x = target[0]+(positions[identifier][0]-x0)*position_factor
                    y = target[1]+(positions[identifier][1]-y0)*position_factor
                    kind = by_id[identifier]['kind']
                    content_range = (dict(start=0,end=len(by_id[identifier]['content'].get('text','')))
                                     if kind=='text' else dict(row_start=0,row_end=by_id[identifier]['content']['row_count'])
                                     if kind=='table' else dict(whole=True))
                    placements.append(dict(fragment_id=identifier+'-fragment-1',block_id=identifier,page_index=page_index,
                                           bbox=[x,y,x+m['width'],y+m['height']],measurement=m,
                                           content_range=content_range,repeat_header=False,frame_id=frame['id'],
                                           source_transform=[member_factor,0.,0.,member_factor,
                                                             x-image_source_bbox(by_id[identifier])[0]*member_factor,
                                                             y-image_source_bbox(by_id[identifier])[1]*member_factor]))
            except ValueError as exc:
                if not recoverable_geometry_error(exc):
                    raise
                failure = RegionLayoutError(str(exc), frame, page)
                failure.measurements = deepcopy(measured)
                failure.positions = deepcopy(positions)
                failure.solver_diagnostics = deepcopy(getattr(exc,'solver_diagnostics',{}))
                if policy.get('text_adaptation_version') in {'20261103','20261104'}:
                    from .layout_pending import save_layout_issue,diagnostic_root
                    failure.positions=deepcopy(getattr(exc,'positions',positions))
                    diagnostic=save_layout_issue(project, diagnostic_root.get(), failure)
                    # Keep completed regions intact. Freeze this region's last
                    # measured geometry; never rasterise or block later pages.
                    target=[ox+box[0]*page_factor,oy+box[1]*page_factor,
                            ox+box[2]*page_factor,oy+box[3]*page_factor]
                    frames.append(dict(id=frame['id'],source_page_id=page['id'],page_index=page_index,
                        bbox=target,source_bbox_pt=box,block_ids=list(identifiers),kind=frame['kind'],
                        unresolved_geometry=True))
                    issue=dict(frame_id=frame['id'],source_page_id=page['id'],
                        physical_page_number=page.get('physical_page_index',0)+1,
                        reason=str(exc),action='export-current-geometry-and-continue',
                        block_ids=list(identifiers),solver=failure.solver_diagnostics,
                        diagnostic_asset='layout-diagnostics/'+diagnostic.name if diagnostic else None)
                    issues.append(issue)
                    positions=getattr(exc,'positions',positions)
                    for identifier in identifiers:
                        b=by_id[identifier]
                        sb=image_source_bbox(b)
                        m=measured.get(identifier) or measure_block(b,sb[2]-sb[0])
                        m=scale_measurement(m,page_factor)
                        pos=positions.get(identifier,[sb[0]-box[0],sb[1]-box[1]])
                        x,y=target[0]+pos[0]*page_factor,target[1]+pos[1]*page_factor
                        kind=b['kind']
                        content_range=(dict(start=0,end=len(b['content'].get('text',''))) if kind=='text'
                            else dict(row_start=0,row_end=b['content']['row_count']) if kind=='table' else dict(whole=True))
                        placements.append(dict(fragment_id=identifier+'-fragment-1',block_id=identifier,
                            page_index=page_index,bbox=[x,y,x+m['width'],y+m['height']],measurement=m,
                            content_range=content_range,repeat_header=False,frame_id=frame['id'],
                            unresolved_geometry=True))
                    continue
                raise failure from exc
    if assigned != set(by_id):
        raise ValueError('区域未完整且唯一覆盖所有顶层模块')
    from .table_image_binding import bind_table_images
    bind_table_images(blocks, placements)
    # A diagnostic canvas exposes overflow without scaling or cropping it.
    for spec in pages:
        visible=[p for p in placements if p['page_index']==spec['index']]
        if not any(p.get('unresolved_geometry') for p in visible):
            continue
        left=min(0.,*(p['bbox'][0] for p in visible))
        top=min(0.,*(p['bbox'][1] for p in visible))
        right=max(spec['width'],*(p['bbox'][2] for p in visible))
        bottom=max(spec['height'],*(p['bbox'][3] for p in visible))
        if [left,top,right,bottom] == [0.,0.,spec['width'],spec['height']]:
            continue
        spec['diagnostic_canvas']=dict(original_size=[spec['width'],spec['height']],offset=[-left,-top])
        spec.update(width=right-left,height=bottom-top)
        for p in visible+[f for f in frames if f['page_index']==spec['index']]:
            p['original_bbox']=list(p['bbox'])
            p['bbox']=[v-(left if n%2==0 else top) for n,v in enumerate(p['bbox'])]
            if 'source_transform' in p:
                p['source_transform'][4]-=left;p['source_transform'][5]-=top
    rank = {i:n for n,i in enumerate(resolved)}
    placements.sort(key=lambda p:rank[p['block_id']])
    changes = [dict(block_id=i,source_index=order.index(i),target_index=n,
                    reason='hard-dependency-overrides-source-order') for n,i in enumerate(resolved) if order[n]!=i]
    result = dict(engine=ENGINE_VERSION,algorithm_version=LAYOUT_VERSION,target_structure='source-preserving-regions',
                  unit='pt',pages=pages,placements=placements,frames=frames,resolved_order=resolved,
                  order_changes=changes,region_adjustments=adjustments,issues=issues)
    project['layout_result'] = result
    return result
