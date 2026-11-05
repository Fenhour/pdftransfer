"""05b/06 source anchors, duplicate boundaries and paragraph adjustments.

No font classifier is treated as typography truth. Every accepted spacing or
size change is stored separately from immutable OCR evidence.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import unicodedata
from statistics import median

from .stage_timing import stage_span


VERSION = 'four-layer-text-v10-unreachable-edge-anchor'


def _max_sliding_prefix_overlap(a_text, b_text, alpha):
    """Match B's longest prefix anywhere in A's alpha-character tail.

    C retains reading order after reversing A's backwards lookup. For each
    decreasing window size, scan C from right to left. B's start stays fixed;
    neither noncontiguous matches nor matches outside C are eligible.
    """
    c = a_text[-int(alpha):] if alpha > 0 else ''
    limit = min(len(c), len(b_text), max(0, int(alpha)))
    for size in range(limit, 0, -1):
        prefix = b_text[:size]
        for start in range(len(c)-size, -1, -1):
            if c[start:start+size] == prefix:
                return size
    return 0


def deduplicate_records(records, gamma=.8):
    if not math.isfinite(gamma) or not .5 <= gamma <= 1.2:
        raise ValueError('重复OCR检查gamma须在0.5到1.2之间')
    values = deepcopy(records)
    changes = []
    ordered = sorted(range(len(values)), key=lambda i: (values[i]['bbox_px'][0], values[i]['bbox_px'][1], i))
    for n, ia in enumerate(ordered):
        for ib in ordered[n+1:]:
            a, b = values[ia], values[ib]
            sa, sb = a['bbox_px'], b['bbox_px']
            if min(sa[2],sb[2]) <= max(sa[0],sb[0]) or min(sa[3],sb[3]) <= max(sa[1],sb[1]):
                continue
            aa, bb = a.get('rebuilt_bbox_pt'), b.get('rebuilt_bbox_pt')
            if aa is None or bb is None:
                continue
            if aa[0] > bb[0]:
                continue
            ox = max(0., min(aa[2],bb[2])-max(aa[0],bb[0]))
            oy = max(0., min(aa[3],bb[3])-max(aa[1],bb[1]))
            h = min(aa[3]-aa[1],bb[3]-bb[1])
            if not ox or oy < .5*h or abs(aa[3]-bb[3]) > .35*max(aa[3]-aa[1],bb[3]-bb[1]):
                continue
            ta, tb = str(a.get('text','')), str(b.get('text',''))
            count = len(ta)+len(tb)
            if not count:
                continue
            area = (aa[2]-aa[0])*(aa[3]-aa[1])+(bb[2]-bb[0])*(bb[3]-bb[1])
            character_area = gamma*area/count
            source_a = a.get('source_bbox_pt', sa)
            source_b = b.get('source_bbox_pt', sb)
            source_overlap = (max(0., min(source_a[2],source_b[2])-max(source_a[0],source_b[0])) *
                              max(0., min(source_a[3],source_b[3])-max(source_a[1],source_b[1])))
            alpha_raw = source_overlap/character_area
            alpha = int(math.floor(alpha_raw+.5))
            # alpha is only the estimated candidate-window upper bound.  Find
            # the largest B-prefix window sliding right-to-left inside C; this is
            # what turns the author's `u13,` / `3,Li` example into `3,`.
            matched = _max_sliding_prefix_overlap(ta, tb, alpha)
            event = dict(type='rebuilt-overlap-duplicate-check',a=a.get('line_id',str(ia)),
                         b=b.get('line_id',str(ib)),gamma=gamma,alpha=alpha,matched=matched,
                         overlap_area_pt2=source_overlap,overlap_area_basis='original-ocr-boxes',
                         rebuilt_overlap_area_pt2=ox*oy,original_text_b=tb,
                         candidate_window=min(alpha,len(ta),len(tb)),matching_method='max-b-prefix-in-a-tail-right-to-left',
                         candidate_text_a=ta[-alpha:] if alpha else '',candidate_text_b=tb[:alpha],
                         alpha_raw=alpha_raw,total_rebuilt_area_pt2=area,
                         extracted_character_count=count,estimated_character_area_pt2=character_area,
                         source_overlap=True,rebuilt_bbox_a=list(aa),rebuilt_bbox_b=list(bb))
            if matched:
                b['text'] = tb[matched:]
                # Prefix advances are measured with the same font/spacing as B.
                starts = b.get('prefix_advance_pt',[])
                if len(starts) <= len(tb):
                    raise ValueError('重复字符缺少实测前缀位置，拒绝猜测B起点')
                delta = starts[matched]
                b['rebuilt_start_offset_pt'] = b.get('rebuilt_start_offset_pt',0.)+delta
                # Recompute BOTH effective envelopes. A's endpoint is measured,
                # not inherited from a detector/previous envelope. B's new start
                # is a content advance; it is not an avoidance translation.
                a_starts = a.get('prefix_advance_pt',[])
                if len(a_starts) <= len(ta):
                    raise ValueError('重复字符缺少A实测终点')
                a['rebuilt_bbox_pt'] = [aa[0],aa[1],aa[0]+a_starts[len(ta)],aa[3]]
                b['rebuilt_bbox_pt'] = [bb[0]+delta,bb[1],bb[0]+starts[len(tb)],bb[3]]
                a['effective_geometry_updated']=b['effective_geometry_updated']=True
                b['prefix_advance_pt'] = [v-delta for v in starts[matched:]]
                b['effective_source_range'] = [b.get('effective_source_range',[0,len(tb)])[0]+matched,
                                                b.get('effective_source_range',[0,len(tb)])[1]]
                event.update(removed_prefix=tb[:matched],owner='left',remaining_text_b=b['text'],
                             updated_bbox_a=list(a['rebuilt_bbox_pt']),updated_bbox_b=list(b['rebuilt_bbox_pt']),
                             geometry_action='remeasure-both-effective-content-envelopes')
            changes.append(event)
    return values, changes


def measure_record(record, style, sx, sy):
    """Target evidence for the gated duplicate pass, before any deletion."""
    from .font_library import get_library
    result = deepcopy(record)
    style = dict(style,**record.get('measurement_style',{}))
    library = get_library()
    face = library.resolve(record['text'],style.get('family','sans-serif'),
                           bool(style.get('bold')),bool(style.get('italic')),safe_only=True)
    if face is None:
        return result
    font = library.export_font_for_key(face.key)
    size = float(style['font_size_pt'])
    gaps = dict(latin_word_gap_pt=font.text_length(' ',fontsize=size),
                cjk_tracking_pt=0.,cross_script_gap_pt=0.)
    text = record['text']
    _,advance = text_runs(text,font,size,gaps)
    b = record['bbox_px']
    result['source_bbox_pt'] = [b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy]
    result['rebuilt_bbox_pt'] = [b[0]*sx,b[1]*sy,b[0]*sx+advance,
                                b[1]*sy+(font.ascender-font.descender)*size]
    result['prefix_advance_pt'] = [text_runs(text[:n],font,size,gaps)[1] for n in range(len(text)+1)]
    result['measurement_style'] = dict(style)
    return result


def _script(c):
    if '\u3400' <= c <= '\u9fff' or '\uf900' <= c <= '\ufaff':
        return 'cjk'
    if c.isascii() and c.isalnum():
        return 'latin'
    return 'other'


def text_runs(text, font, size, spacing):
    """Absolute per-class added gaps; words and combining clusters stay intact."""
    tokens = re.findall(r' +|[\u3400-\u9fff\uf900-\ufaff]|[^ \u3400-\u9fff\uf900-\ufaff]+',text)
    merged = []
    for token in tokens:
        if merged and all(unicodedata.combining(c) or c in {'\ufe0f','\u200d'} for c in token):
            merged[-1] += token
        else:
            merged.append(token)
    runs, cursor, previous = [], 0., None
    for token in merged:
        if token.isspace():
            # Repeated/leading spaces may encode indentation or tabular stops.
            # They are structural rather than paragraph word-gap slack.
            cursor += (spacing['latin_word_gap_pt'] if len(token)==1 and previous is not None
                       else font.text_length(token,fontsize=size))
            previous = None
            continue
        script = _script(token[0])
        if previous == script == 'cjk':
            cursor += spacing['cjk_tracking_pt']
        elif {previous,script} == {'cjk','latin'}:
            cursor += spacing['cross_script_gap_pt']
        runs.append(dict(text=token,x=cursor))
        cursor += font.text_length(token,fontsize=size)
        previous = script
    return runs, cursor


def _spacing_bounds(initial, style):
    # Zero is the geometric no-extra-gap bound. A product can configure a
    # stricter bound; the old 65% template is deliberately not inherited.
    lower = dict.fromkeys(initial, 0.)
    lower.update(style.get('minimum_spacing_pt', {}))
    if any(not math.isfinite(v) or v < 0 or v > initial[k] for k,v in lower.items()):
        raise ValueError('最小间距必须是有限非负值且不超过初始间距')
    return lower


def _mix(a, b, t):
    return {k:a[k]+t*(b[k]-a[k]) for k in a}


def _row_layout(row, size_factor, gap_sets):
    cursor, lines = 0., []
    for index, item in enumerate(row['items']):
        if index:
            cursor += row['structural_gaps'][index-1]
        size = item['size']*size_factor
        runs, advance = text_runs(item['text'],item['font'],size,gap_sets[index])
        lines.append(dict(text=item['text'],start=item['fragment']['start'],end=item['fragment']['end'],
            effective_start=item['fragment']['start']+item['fragment'].get('removed_prefix_count',0),
            x=cursor,font_size=size,height=(item['font'].ascender-item['font'].descender)*size,
            font_key=item['font_key'],text_runs=runs,advance_width=advance,
            row_id=row['id'],line_id=item['fragment'].get('line_id'),spacing_pt=dict(gap_sets[index]),
            source_top=item['top'],effective_source_range=item['fragment'].get('effective_source_range')))
        cursor += advance
    return lines,cursor


def _fit_gaps(row, factor, target, current, lower):
    """Return actual per-line gaps, retaining the lower-bound state on overflow."""
    _, natural = _row_layout(row,factor,current)
    _, minimum = _row_layout(row,factor,lower)
    if minimum > target+1e-7:
        return deepcopy(lower),'content-too-wide',minimum
    if abs(natural-target) <= 1e-7:
        return deepcopy(current),'aligned',natural
    if natural > target:
        lo,hi = 0.,1.
        for _ in range(42):
            mid=(lo+hi)/2
            trial=[_mix(a,b,mid) for a,b in zip(lower,current)]
            if _row_layout(row,factor,trial)[1] <= target: lo=mid
            else: hi=mid
        gaps=[_mix(a,b,lo) for a,b in zip(lower,current)]
        return gaps,'aligned',_row_layout(row,factor,gaps)[1]
    # Expansion changes only an adjustable word/CJK/cross-script slot.
    unit=[{k:v+1 for k,v in g.items()} for g in current]
    slots=_row_layout(row,factor,unit)[1]-natural
    if slots <= 1e-9:
        return deepcopy(current),'no-expandable-gap',natural
    extra=(target-natural)/slots
    gaps=[{k:v+extra for k,v in g.items()} for g in current]
    return gaps,'aligned',_row_layout(row,factor,gaps)[1]


def _prepare_text_linkage(groups, bounds, adjustments):
    """Translate first; then handle residual same-row collisions, then align."""
    from .text_linkage import solve_text_rectangles, overlaps
    items=[i for rows in groups.values() for row in rows.values() for i in row['items']]
    def boxes():
        return {i['uid']:[i['x'],i['top'],i['x']+text_runs(i['text'],i['font'],i['size'],i['initial'])[1],
                          i['top']+(i['font'].ascender-i['font'].descender)*i['size']] for i in items}
    before=boxes()
    rectangles=[dict(id=i['uid'],row=i['row_key'],x=before[i['uid']][0],y=before[i['uid']][1],
                     row_order=i['fragment']['row'],paragraph_id=i['paragraph_id'],
                     indentation_group=i.get('indentation_group',i['paragraph_id']),
                     width=max(1e-6,before[i['uid']][2]-before[i['uid']][0]),
                     height=before[i['uid']][3]-before[i['uid']][1],source_bbox=i['source_bbox'],
                     original_source_bbox=i['alignment_source_bbox']) for i in items]
    positions,detail=solve_text_rectangles(rectangles,bounds)
    if detail['status'] in {'unchanged','optimal','feasible'}:
        for i in items:
            i['x'],i['top']=positions[i['uid']][:2]
    adjustments.append(dict(scope='block-text',type='bounded-text-linkage',**detail))
    current=boxes()
    colliding={a['uid'] for a in items for b in items if a is not b and overlaps(current[a['uid']],current[b['uid']])}
    # Only residual collisions permit anomalously large reconstructed fonts to
    # be normalised. Ordinary alignment alone never triggers this correction.
    for i in items:
        if i['uid'] not in colliding: continue
        peers=[p['size'] for p in items if p['uid'] not in colliding and p['paragraph_id']==i['paragraph_id']
               and p['font_key']==i['font_key']]
        if peers and i['size']>median(peers):
            old=i['size'];i['size']=median(peers)
            adjustments.append(dict(scope='line',type='residual-collision-size-normalization',
                line_id=i['fragment'].get('line_id'),before_font_size_pt=old,after_font_size_pt=i['size']))
    for rows in groups.values():
        for row in rows.values():
            row['items'].sort(key=lambda i:(i['x'],i['uid']))
            for a,b in zip(row['items'],row['items'][1:]):
                current=boxes()
                if not overlaps(current[a['uid']],current[b['uid']]): continue
                pair=dict(id=row['id'],items=[a,b],structural_gaps=[0.])
                capacity=max(current[a['uid']][2],current[b['uid']][2])-a['x']
                gaps,status,_=_fit_gaps(pair,1.,capacity,[a['initial'],b['initial']],[a['lower'],b['lower']])
                a['initial'],b['initial']=gaps
                if status=='aligned':
                    b['x']=a['x']+text_runs(a['text'],a['font'],a['size'],a['initial'])[1]
                adjustments.append(dict(scope='row',type='residual-collision-word-spacing',
                    line_ids=[a['fragment'].get('line_id'),b['fragment'].get('line_id')],status=status))
                if status=='content-too-wide':
                    # Retain their content order as a single row. No repeated
                    # pairwise push loop, and no forced paragraph font shrink.
                    adjustments.append(dict(scope='row',type='residual-pair-as-one-row',
                        line_ids=[a['fragment'].get('line_id'),b['fragment'].get('line_id')]))
            row['structural_gaps']=[max(0.,b['x']-(a['x']+text_runs(a['text'],a['font'],a['size'],a['initial'])[1]))
                                   for a,b in zip(row['items'],row['items'][1:])]
    return detail


def measure_four_layer(content, width, style, source, *, preserve_envelope=True, whitespace_right_pt=0.,
                       text_bounds_pt=None):
    from .font_library import get_library
    from .source_text import _valid_fragments, _box
    fragments = _valid_fragments(content)
    pixels, points = _box(source.get('bbox_px')), _box(source.get('bbox_pt'))
    if not fragments or not pixels or not points:
        raise ValueError('新版文字缺少有效源行几何，需重新建立源片段；禁止静默自动换行')
    library=get_library()
    requested=float(style.get('font_size_pt',11))
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('文字字号无效')
    if not math.isfinite(whitespace_right_pt) or whitespace_right_pt < 0:
        raise ValueError('可用空白无效')
    sx=width/(pixels[2]-pixels[0])
    sy=(points[3]-points[1])/(pixels[3]-pixels[1])*width/(points[2]-points[0])
    groups={}
    # Physical whitespace may split code into several paragraphs. Those
    # paragraphs still share a single indentation coordinate system.
    code_block = source.get('label') == 'algorithm' or content.get('role') == 'algorithm'
    preserve_caption = source.get('label') == 'figure_title' or content.get('role') == 'figure_title'
    for fragment,b,text in fragments:
        pid=fragment.get('paragraph_id','paragraph-0')
        rows=groups.setdefault(pid,{})
        row=rows.setdefault(fragment['row'],dict(id=fragment['row'],items=[]))
        evidence=fragment.get('measurement_style',{})
        face=library.resolve(text,evidence.get('family',style.get('family','sans-serif')),
            bool(evidence.get('bold',style.get('bold'))),bool(evidence.get('italic',style.get('italic'))),safe_only=True)
        if face is None:
            raise ValueError('新版文字没有可用字体，禁止静默自动换行')
        font=library.export_font_for_key(face.key)
        size=float(evidence.get('font_size_pt',requested))
        if not math.isfinite(size) or size <= 0:
            raise ValueError('文字样式组字号无效')
        gaps=dict(latin_word_gap_pt=float(style.get('word_gap_pt',font.text_length(' ',fontsize=size))),
                  cjk_tracking_pt=float(style.get('cjk_tracking_pt',0.)),
                  cross_script_gap_pt=float(style.get('cross_script_gap_pt',0.)))
        if any(not math.isfinite(v) or v < 0 for v in gaps.values()):
            raise ValueError('文字间距无效')
        source_box=[(b[0]-pixels[0])*sx,(b[1]-pixels[1])*sy,(b[2]-pixels[0])*sx,(b[3]-pixels[1])*sy]
        # Immutable OCR edges express alignment intent. Linkage and duplicate
        # removal may change source_box below, but must not change that intent.
        alignment_source_box=list(source_box)
        if fragment.get('effective_geometry_updated') and fragment.get('rebuilt_bbox_pt'):
            effective=fragment['rebuilt_bbox_pt'];scale=width/(points[2]-points[0])
            source_box=[(effective[0]-points[0])*scale,(effective[1]-points[1])*scale,
                        (effective[2]-points[0])*scale,(effective[3]-points[1])*scale]
        row['items'].append(dict(fragment=fragment,text=text,font=font,font_key=face.key,size=size,
            top=(b[1]-pixels[1])*sy,x=(b[0]-pixels[0])*sx+fragment.get('rebuilt_start_offset_pt',0.),
            source_right=(b[2]-pixels[0])*sx,initial=gaps,lower=_spacing_bounds(gaps,style),
            uid=str(len([i for rs in groups.values() for r in rs.values() for i in r['items']])),
            row_key=(pid,fragment['row']),paragraph_id=pid,
            indentation_group='code-block' if code_block else pid,
            source_bbox=source_box,alignment_source_bbox=alignment_source_box))
    lines,adjustments,states=[],[],[]
    source_height=(points[3]-points[1])*width/(points[2]-points[0])
    bounds=text_bounds_pt or [0.,0.,width+whitespace_right_pt,source_height]
    with stage_span('05','bounded_text_linkage'):
        linkage=_prepare_text_linkage(groups,bounds,adjustments)
    previous_bottom=0.
    for pid,raw_rows in groups.items():
        rows=[]
        for _,row in sorted(raw_rows.items()):
            row['items'].sort(key=lambda v:v['x'])
            row['left']=row['items'][0]['x']
            row['initial']=[dict(i['initial']) for i in row['items']]
            row['lower']=[dict(i['lower']) for i in row['items']]
            row['right']=row['left']+_row_layout(row,1.,row['initial'])[1]
            row['height']=max((i['font'].ascender-i['font'].descender)*i['size'] for i in row['items'])
            originals=[i['alignment_source_bbox'] for i in row['items']]
            row['source_left']=min(b[0] for b in originals)
            row['source_right']=max(b[2] for b in originals)
            row['source_height']=max(b[3] for b in originals)-min(b[1] for b in originals)
            rows.append(row)
        left=(min(r['source_left'] for r in rows) if len(rows)==2 else median(r['source_left'] for r in rows))
        right=(max(r['source_right'] for r in rows) if len(rows)==2 else median(r['source_right'] for r in rows))
        if right <= left:
            raise ValueError('段落主导边缘区间无效')
        tolerance=float(style.get('alignment_tolerance_ratio',.15))
        if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
            raise ValueError('对齐容差比例无效')
        deferred=[]
        for row in rows:
            tol=tolerance*row['source_height']
            # A single row has no peer alignment evidence; keep its existing
            # anchors rather than turning its own OCR box into forced fitting.
            row['align_left']=not preserve_caption and len(rows)>1 and abs(row['source_left']-left)<=tol
            row['align_right']=not preserve_caption and len(rows)>1 and abs(row['source_right']-right)<=tol
            row['target_left']=left if row['align_left'] else row['left']
            row['target_right']=right if row['align_right'] else row['right']
            row['capacity']=row['target_right']-row['target_left']
            if row['capacity'] <= 0:
                row['gaps']=deepcopy(row['initial']);row['first_status']='conflicting-endpoints'
            elif row['align_left'] or row['align_right']:
                row['gaps'],row['first_status'],_= _fit_gaps(row,1.,row['capacity'],row['initial'],row['lower'])
                if row['first_status']=='content-too-wide' and not (row['align_left'] and row['align_right']):
                    # A one-sided tolerance correction is not evidence of
                    # excessive content. Preserve this row's previous anchors.
                    row['gaps']=deepcopy(row['initial'])
                    row['first_status']='preserve-anchor-conflict'
                    row['target_left'],row['target_right']=row['left'],row['right']
            else:
                row['gaps']=deepcopy(row['initial']);row['first_status']='preserve-anchor'
            if (row['align_left'] and row['align_right'] and
                    row['first_status']=='content-too-wide'):
                deferred.append(row['id'])
            row['tightened_gaps']=deepcopy(row['gaps'])
        factor=1.
        shrink_widths={r['id']:_row_layout(r,1.,r['tightened_gaps'])[1] for r in rows}
        if deferred:
            # Keep *each* first-pass gap result, especially an overflowing row's
            # legal lower bound. Only font sizes change during this search.
            def fits(s):
                return all(_row_layout(r,s,r['tightened_gaps'])[1] <= r['capacity']+1e-8
                           for r in rows if r['align_left'] and r['align_right'])
            floor=float(style.get('minimum_font_scale',.01))
            if not math.isfinite(floor) or not 0 < floor <= 1:
                raise ValueError('段落最小字号比例无效')
            lo,hi=floor,1.
            if not fits(lo):
                raise ValueError('段落固定间距或配置字号下限无法容纳，需继续布局处理')
            for _ in range(48):
                mid=(lo+hi)/2
                if fits(mid):lo=mid
                else:hi=mid
            factor=lo
        row_events=[]
        for row in rows:
            gaps=deepcopy(row['tightened_gaps'])
            status=row['first_status']
            if (row['align_left'] or row['align_right']) and status!='preserve-anchor-conflict':
                gaps,status,_=_fit_gaps(row,factor,row['capacity'],gaps,
                    [{k:min(v,g[k]) for k,v in low.items()} for low,g in zip(row['lower'],gaps)])
            result,advance=_row_layout(row,factor,gaps)
            x=(row['left'] if status in {'preserve-anchor-conflict','no-expandable-gap'} else
               row['target_left'] if row['align_left'] else (row['target_right']-advance if row['align_right'] else row['left']))
            if len(rows)==1 and style.get('align') in {'center','right'}:
                # A centred caption preserves its SOURCE centre when its new
                # font has another width; retaining only its left edge drifts
                # its visual position even though the solver did not move it.
                x=((row['source_left']+row['source_right']-advance)/2 if style['align']=='center'
                   else row['source_right']-advance)
            top=min(i['top'] for i in row['items'])
            shift=0.
            for line,item in zip(result,row['items']):
                line['x']+=x
                line['baseline']=top+shift+item['font'].ascender*line['font_size']
                line['paragraph_id']=pid
                line.pop('source_top',None)
                lines.append(line)
            previous_bottom=max(previous_bottom,top+shift+max(l['height'] for l in result))
            row_events.append(dict(row_id=row['id'],line_ids=[v.get('line_id') for v in result],
                align_left=row['align_left'],align_right=row['align_right'],tolerance_pt=tolerance*row['source_height'],
                source_endpoints=[row['source_left'],row['source_right']],source_height_pt=row['source_height'],
                previous_endpoints=[row['left'],row['right']],target_endpoints=[row['target_left'],row['target_right']],
                actual_endpoints=[x,x+advance],first_pass_status=row['first_status'],status=status,
                initial_spacing_pt=row['initial'],tightened_spacing_pt=row['tightened_gaps'],final_spacing_pt=gaps,
                before_shrink_width_pt=shrink_widths[row['id']],after_width_pt=advance,
                second_pass_complete=status in {'aligned','preserve-anchor','preserve-anchor-conflict'}))
        spacing_changed=any(e['initial_spacing_pt'] != e['final_spacing_pt'] for e in row_events)
        events=[]
        if spacing_changed:events.append(dict(scope='line',type='line_spacing_adjustment',status='applied'))
        if factor<1.-1e-8:events.append(dict(scope='paragraph',type='font_size_scaling',status='applied',
            reason='content-too-wide-after-legal-line-spacing',font_scale=factor))
        states.append(dict(paragraph_id=pid,paragraph_adjusted=bool(events),paragraph_spacing_compressed=False,
            paragraph_spacing_adjusted=spacing_changed,paragraph_font_scaled=factor<1.-1e-8,font_scale=factor,
            dominant_left_pt=left,dominant_right_pt=right,edge_method='two-row-extrema' if len(rows)==2 else 'median',
            edge_basis='original-ocr-boxes',
            edge_policy='preserve-caption-anchors' if preserve_caption else 'source-edge-tolerance',
            deferred_row_ids=deferred,longest_row_id=max(shrink_widths,key=shrink_widths.get),
            before_shrink_widths_pt=shrink_widths,row_adjustments=row_events,adjustments=events,
            affected_line_ids=[i['fragment'].get('line_id') for r in rows for i in r['items']],
            source_overlap_checks=content.get('source_overlap_checks',[]),
            whitespace_attempt=dict(available_right_pt=whitespace_right_pt,alignment='paragraph-dominant-edges')))
        adjustments.extend(events)
    with stage_span('06','text_envelope'):
        lines.sort(key=lambda v:v['start'])
        # Freeze the latest row offsets, whether alignment reached its target
        # or not. The common solver rebuilds indentation locks from these x
        # values. Do not also demand ideal edges that may not have been reached.
        from .text_linkage import solve_text_rectangles
        rectangles=[];row_lines={}
        for state in states:
            for event in state['row_adjustments']:
                key=state['paragraph_id']+':'+str(event['row_id'])
                members=[l for l in lines if l['paragraph_id']==state['paragraph_id'] and l['row_id']==event['row_id']]
                row_lines[key]=members
                x0=min(l['x'] for l in members);x1=max(l['x']+l['advance_width'] for l in members)
                tops=[l['baseline']-library.export_font_for_key(l['font_key']).ascender*l['font_size'] for l in members]
                y0=min(tops);y1=max(t+l['height'] for t,l in zip(tops,members))
                source_boxes=[i['source_bbox'] for i in groups[state['paragraph_id']][event['row_id']]['items']]
                source_box=[min(b[0] for b in source_boxes),min(b[1] for b in source_boxes),
                            max(b[2] for b in source_boxes),max(b[3] for b in source_boxes)]
                original_boxes=[i['alignment_source_bbox'] for i in groups[state['paragraph_id']][event['row_id']]['items']]
                original_box=[min(b[0] for b in original_boxes),min(b[1] for b in original_boxes),
                              max(b[2] for b in original_boxes),max(b[3] for b in original_boxes)]
                rectangles.append(dict(id=key,row=key,row_order=event['row_id'],paragraph_id=state['paragraph_id'],
                                       indentation_group='code-block' if code_block else state['paragraph_id'],
                                       x=x0,y=y0,width=max(1e-6,x1-x0),height=y1-y0,source_bbox=source_box,
                                       original_source_bbox=original_box))
        with stage_span('06','text_linkage_after_alignment'):
            final_positions,final_detail=solve_text_rectangles(rectangles,bounds)
        final_detail['indentation_lock_basis']='latest-post-alignment-row-positions'
        if final_detail['status'] in {'optimal','feasible'}:
            for rect in rectangles:
                dx=final_positions[rect['id']][0]-rect['x'];dy=final_positions[rect['id']][1]-rect['y']
                for line in row_lines[rect['id']]:line['x']+=dx;line['baseline']+=dy
        adjustments.append(dict(scope='block-text',type='bounded-text-linkage-after-alignment',**final_detail))
        linkage=dict(before_alignment=linkage,after_alignment=final_detail,status=final_detail['status'])
        top=min(l['baseline']-library.export_font_for_key(l['font_key']).ascender*l['font_size'] for l in lines)
        dy=-min(0.,top)
        if dy>1e-8:
            for line in lines:line['baseline']+=dy
            adjustments.append(dict(scope='envelope',type='whole-envelope-vertical-translation',shift_pt=dy))
        previous_bottom=max(l['baseline']-library.export_font_for_key(l['font_key']).ascender*l['font_size']+l['height'] for l in lines)
        x0=min(v['x'] for v in lines);x1=max(v['x']+v['advance_width'] for v in lines)
        source_height=(points[3]-points[1])*width/(points[2]-points[0])
        # Normalise a protruding OCR crop into nonnegative local coordinates.
        # The placement origin compensates this shift, so glyphs retain their
        # page anchors. Right overflow grows the envelope, not a leftward move.
        dx=max(0.,-x0)
        for line in lines:line['x']+=dx
        if abs(dx)>1e-8:adjustments.append(dict(scope='envelope',type='whole-envelope-translation',shift_pt=dx))
        occupied_right=max(v['x']+v['advance_width'] for v in lines)
        for state in states:
            for event in state['row_adjustments']:
                members=[l for l in lines if l['paragraph_id']==state['paragraph_id'] and l['row_id']==event['row_id']]
                event['after_linkage_endpoints']=[min(l['x'] for l in members),max(l['x']+l['advance_width'] for l in members)]
        return dict(width=max(width+dx,occupied_right),height=max(source_height+dy,previous_bottom) if preserve_envelope else previous_bottom,
            font_key=lines[0]['font_key'],font_size=min(v['font_size'] for v in lines),line_step=requested*1.25,
            color=style.get('color','#000000'),lines=lines,breaks=[v['end'] for v in lines],min_width=1.,
            internal_layout=VERSION,source_envelope_preserved=preserve_envelope,source_envelope_height=source_height,
            text_linkage=linkage,
            occupied_height=previous_bottom,internal_adjustments=adjustments,paragraph_adjustments=states,
            paragraph_adjusted=any(v['paragraph_adjusted'] for v in states),paragraph_spacing_compressed=False,
            paragraph_font_scaled=any(v['paragraph_font_scaled'] for v in states),
            block_expansion=dict(left_pt=-dx,top_pt=-dy,right_pt=max(0.,occupied_right-dx-width)),
            source_text_policy='source-lines-dominant-edges-per-line-spacing-retained-before-font-scale')
