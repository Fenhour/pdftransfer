"""Bounded 07/08 solver. O1 only releases verified empty projection bands."""
from copy import deepcopy
import math

from .stage_timing import stage_span, timed_step

EPS = .05


def empty_bands(rects, axis, extent):
    """Complement of the union of occupied projections, in physical points."""
    intervals = sorted((max(0., r[axis]), min(extent, r[axis+2])) for r in rects)
    cursor, result = 0., []
    for lo, hi in intervals:
        if lo > cursor+EPS:
            result.append([cursor, lo])
        cursor = max(cursor, hi)
    if cursor < extent-EPS:
        result.append([cursor, extent])
    return result


@timed_step('08','region_solver')
def solve(items, measurements, frame, *, empty_bands_enabled=False, attachments=(),
          minimum_scale=.85, minimum_font=7., page_factor=1., max_rounds=16,
          uniform_region=False, _return_overflow=False):
    if uniform_region:
        # Keep the November 1 block-collision method. Only its *completed*
        # arrangement is transformed as a whole for Region containment.
        pos,moved,_,detail=solve(items,measurements,frame,empty_bands_enabled=empty_bands_enabled,
            attachments=attachments,minimum_scale=minimum_scale,minimum_font=minimum_font,
            page_factor=page_factor,max_rounds=max_rounds,_return_overflow=True)
        left=min(0.,*(p[0] for p in pos.values()))
        top=min(0.,*(p[1] for p in pos.values()))
        right=max(p[0]+measurements[i]['width'] for i,p in pos.items())
        bottom=max(p[1]+measurements[i]['height'] for i,p in pos.items())
        scale=min(1.,(frame[2]-frame[0])/(right-left),(frame[3]-frame[1])/(bottom-top))
        from .regional_layout import _font_sizes
        sizes=[s for m in measurements.values() for s in _font_sizes(m)]
        if scale*page_factor < minimum_scale-1e-8 or any(
                s*scale*page_factor < min(minimum_font,s)-1e-8 for s in sizes):
            error=ValueError('区域整体缩放超过可读下限，需继续布局处理')
            error.positions=deepcopy(pos)
            error.solver_diagnostics=detail
            raise error
        pos={i:[(p[0]-left)*scale,(p[1]-top)*scale] for i,p in pos.items()}
        detail.update(region_scale=scale,gap_policy='whole-region-positions-and-distances-scaled',
                      source_content_bounds=[left,top,right,bottom])
        return pos,moved,scale,detail
    from .regional_layout import image_source_bbox, owned_image_pair, _overlap, _font_sizes
    from .source_containment import source_contains_pair
    if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 64:
        raise ValueError('07/08迭代预算须为1到64的整数')
    left, top, right, bottom = frame
    extent = [right-left, bottom-top]
    source = {b['id']: image_source_bbox(b) for b in items}
    by_id = {b['id']: b for b in items}
    ids = sorted(source, key=lambda i: (source[i][1], source[i][0], i))
    # Build once for every scale/iteration. Original containment survives child
    # movement, measurement growth and different target transforms.
    nested = {frozenset((a,b)) for n,a in enumerate(ids) for b in ids[n+1:]
              if source_contains_pair(by_id[a], by_id[b])}
    attempts = []
    def duplicate_boundary(a,b):
        left_ids={r.get('line_id') for r in by_id[a]['source'].get('ocr_lines',[])}
        right_ids={r.get('line_id') for r in by_id[b]['source'].get('ocr_lines',[])}
        return any(c.get('matched',0)>0 and c.get('a') in left_ids and c.get('b') in right_ids
                   for i in (a,b) for p in measurements[i].get('paragraph_adjustments',[])
                   for c in p.get('source_overlap_checks',[]))

    def trial(scale):
        positions = {i: [source[i][0]-left+measurements[i].get('block_expansion',{}).get('left_pt',0.),
                         source[i][1]-top+measurements[i].get('block_expansion',{}).get('top_pt',0.)] for i in ids}
        moves, bands_used, seen = set(), [], set()

        def box(i):
            x,y = positions[i]; m = measurements[i]
            return [x,y,x+m['width']*scale,y+m['height']*scale]

        def move_closure(seed, axis, delta):
            # Explicit physical locks and every block intersecting a swept
            # footprint move together. Reading edges alone are not locks.
            linked = {seed}
            changed = True
            while changed:
                changed = False
                for link in attachments:
                    if link.get('keep') != 'required':
                        continue
                    pair = {link.get('anchor'),link.get('dependent')}
                    if pair & linked and pair <= set(ids) and not pair <= linked:
                        linked.update(pair); changed = True
                for i in list(linked):
                    swept = box(i)
                    swept[axis+2] += delta
                    for j in ids:
                        if (j not in linked and positions[j][axis] >= positions[i][axis] and
                                frozenset((i,j)) not in nested and
                                source[j][axis] >= source[i][axis] and _overlap(swept,box(j)) > EPS):
                            linked.add(j); changed = True
            for i in linked:
                positions[i][axis] += delta
            moves.update(linked)

        for iteration in range(1,max_rounds+1):
            state = tuple(round(v,5) for i in ids for v in positions[i])
            if state in seen:
                return None, dict(scale=scale,rounds=iteration,status='cycle-detected')
            seen.add(state)
            with stage_span('07','block_collisions',attempt=len(attempts)+1,iteration=iteration):
                collision = False
                for n,a in enumerate(ids):
                    for b in ids[n+1:]:
                        if frozenset((a,b)) in nested:
                            continue
                        ta,tb = box(a),box(b)
                        if _overlap(ta,tb) <= EPS:
                            continue
                        sa,sb = source[a],source[b]
                        if owned_image_pair(by_id[a],by_id[b]) and scale == 1 and all(
                            abs(v-w)<=EPS for i,s in ((a,sa),(b,sb))
                            for v,w in zip(box(i),[s[0]-left,s[1]-top,s[2]-left,s[3]-top])):
                            continue
                        collision = True
                        if sa[3] <= sb[1]+1.:
                            move_closure(b,1,ta[3]+max(0.,sb[1]-sa[3])-tb[1])
                        elif sb[3] <= sa[1]+1.:
                            move_closure(a,1,tb[3]+max(0.,sa[1]-sb[3])-ta[1])
                        elif sa[2] <= sb[0]+1. or (sa[0]<sb[0] and duplicate_boundary(a,b)):
                            move_closure(b,0,ta[2]+max(0.,sb[0]-sa[2])-tb[0])
                        elif sb[2] <= sa[0]+1. or (sb[0]<sa[0] and duplicate_boundary(b,a)):
                            move_closure(a,0,tb[2]+max(0.,sa[0]-sb[2])-ta[0])
                        else:
                            error=ValueError(f'源块 {a}/{b} 重叠关系不明确，需校对')
                            error.positions=deepcopy(positions)
                            error.solver_diagnostics=dict(status='ambiguous-source-overlap',block_ids=[a,b],rounds=iteration)
                            raise error
            # 08 may translate a whole overflow chain into pure empty space.
            # Rechecking 07 on the next round is mandatory after a band move.
            with stage_span('08','region_boundary',attempt=len(attempts)+1,iteration=iteration):
                band_move = False
                if empty_bands_enabled:
                    with stage_span('O1','empty_bands',attempt=len(attempts)+1,iteration=iteration):
                        for axis in (0,1):
                            rectangles = [box(i) for i in ids]
                            overflow = max(r[axis+2] for r in rectangles)-extent[axis]
                            if overflow <= EPS:
                                continue
                            for lo,hi in reversed(empty_bands(rectangles,axis,extent[axis])):
                                moving = {i for i in ids if box(i)[axis] >= hi-EPS}
                                if not moving:
                                    continue
                                # Do not split an explicit physical lock across a band.
                                if any(l.get('keep')=='required' and
                                       len({l.get('anchor'),l.get('dependent')} & moving)==1
                                       for l in attachments):
                                    continue
                                delta = min(overflow, hi-lo)
                                for i in moving:
                                    positions[i][axis] -= delta
                                moves.update(moving)
                                bands_used.append(dict(axis='x' if axis==0 else 'y',band_pt=[lo,hi],
                                                       shift_pt=-delta,block_ids=sorted(moving)))
                                band_move = True
                                break
            if collision or band_move:
                continue
            with stage_span('08','containment_check',attempt=len(attempts)+1,iteration=iteration):
                rects = [box(i) for i in ids]
                inside = all(r[0]>=-EPS and r[1]>=-EPS and r[2]<=extent[0]+EPS and
                             r[3]<=extent[1]+EPS for r in rects)
            detail = dict(scale=scale,rounds=iteration,status='converged' if inside else 'overflow',
                          empty_band_moves=bands_used)
            return (positions,sorted(moves)) if inside or _return_overflow else None, detail
        return None, dict(scale=scale,rounds=max_rounds,status='iteration-budget-exhausted')

    sizes = [s for m in measurements.values() for s in _font_sizes(m)]
    floor = max(minimum_scale/page_factor,*(
        [min(minimum_font,s)/(s*page_factor) for s in sizes] or [0.]))
    if floor>1.+1e-8:
        raise ValueError('区域页面缩放超过可读下限，需校对')
    floor=min(1.,floor)
    full, detail = trial(1.)
    attempts.append(detail)
    if full:
        return full[0],full[1],1.,dict(status='converged',attempts=attempts,
                                     empty_bands=empty_bands_enabled,max_rounds=max_rounds)
    best, detail = trial(floor)
    attempts.append(detail)
    if best is None:
        error=ValueError('区域内碰撞无法有界解决或溢出超过可读下限，需校对；未挤入其他大区域')
        error.solver_diagnostics=dict(status='needs-review',attempts=attempts,
                                      empty_bands=empty_bands_enabled,max_rounds=max_rounds)
        raise error
    lo,hi = floor,1.
    for _ in range(18):
        mid=(lo+hi)/2
        result,detail=trial(mid)
        attempts.append(detail)
        if result is None:
            hi=mid
        else:
            lo,best=mid,result
    return best[0],best[1],lo,dict(status='converged',attempts=attempts,
                                  empty_bands=empty_bands_enabled,max_rounds=max_rounds,
                                  gap_policy='source-gaps-never-multiplied-by-block-scale')
