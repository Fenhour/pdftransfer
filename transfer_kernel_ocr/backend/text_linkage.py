"""Text linkage constrained to source collision axes and initial anchors.

The block/Region solver is separate. Each pair keeps its source separation
axis; a blocked vertical solution never silently becomes horizontal packing.
"""
from __future__ import annotations

import math
import time
from statistics import median

VERSION = 'text-linkage-axis-anchored-v2'
EPS = 1e-7


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0])+EPS and min(a[3], b[3]) > max(a[1], b[1])+EPS


def _axis(items, edges, axis, bounds):
    """Find a bounded feasible position nearest the initial coordinate anchors.

    Minimise the maximum displacement first. Equality components retain their
    offsets; subsequent coordinate sweeps bring free components back towards
    their median original anchor. No minimum-span/area packing is performed.
    """
    start, end = (bounds[0], bounds[2]) if axis == 'x' else (bounds[1], bounds[3])
    size = 'width' if axis == 'x' else 'height'
    ids = [i['id'] for i in items]
    anchor = {i['id']: i[axis] for i in items}
    upper = {i['id']: end-i[size] for i in items}
    if any(upper[k] < start-EPS for k in ids):
        return None, 'boundary-capacity'
    if (all(start-EPS <= anchor[k] <= upper[k]+EPS for k in ids) and
            all(anchor[b] >= anchor[a]+d-EPS for a,b,d in edges)):
        return anchor, 0.

    # Collapse exact equalities into a translated component with fixed offsets.
    edge_set = set(edges)
    equality = {k: [] for k in ids}
    for a,b,d in edges:
        if (b,a,-d) in edge_set:
            equality[a].append((b,d))
    group, offset, members = {}, {}, []
    for k in ids:
        if k in group:
            continue
        index = len(members); members.append([])
        group[k] = index; offset[k] = 0.; pending = [k]
        while pending:
            a = pending.pop(); members[index].append(a)
            for b,d in equality[a]:
                value = offset[a]+d
                if b in group:
                    if group[b] != index or abs(offset[b]-value) > EPS:
                        return None, 'contradictory-cycle'
                else:
                    group[b] = index; offset[b] = value; pending.append(b)
    reduced = []
    for a,b,d in edges:
        ga,gb = group[a],group[b]; delta = d+offset[a]-offset[b]
        if ga == gb:
            if delta > EPS:
                return None, 'contradictory-cycle'
        else:
            reduced.append((ga,gb,delta))
    reduced = list(dict.fromkeys(reduced))
    lo = [max(start-offset[k] for k in ks) for ks in members]
    hi = [min(upper[k]-offset[k] for k in ks) for ks in members]
    if any(a > b+EPS for a,b in zip(lo,hi)):
        return None, 'boundary-capacity'

    def feasible(radius=None):
        lower = list(lo); limit = list(hi)
        if radius is not None:
            for g,ks in enumerate(members):
                lower[g] = max(lower[g], max(anchor[k]-offset[k]-radius for k in ks))
                limit[g] = min(limit[g], min(anchor[k]-offset[k]+radius for k in ks))
        if any(a > b+EPS for a,b in zip(lower,limit)):
            return None, 'boundary-capacity', lower, limit
        value = list(lower)
        for _ in range(len(members)):
            changed = False
            for a,b,d in reduced:
                if value[b] < value[a]+d-EPS:
                    value[b] = value[a]+d; changed = True
            if any(v > h+EPS for v,h in zip(value,limit)):
                return None, 'boundary-capacity', lower, limit
            if not changed:
                return value, None, lower, limit
        return None, 'contradictory-cycle', lower, limit

    initial, reason, _, _ = feasible()
    if initial is None:
        return None, reason
    low = 0.
    high = max(abs(initial[group[k]]+offset[k]-anchor[k]) for k in ids)+EPS
    for _ in range(40):
        mid = (low+high)/2
        if feasible(mid)[0] is None:
            low = mid
        else:
            high = mid
    value, reason, lower, limit = feasible(high+EPS)
    incoming = [[] for _ in members]; outgoing = [[] for _ in members]
    for a,b,d in reduced:
        incoming[b].append((a,d)); outgoing[a].append((b,d))
    preferred = [median(anchor[k]-offset[k] for k in ks) for ks in members]
    # Recover untouched positions whenever the active constraints permit them.
    for _ in range(max(2, len(members)*2)):
        change = 0.
        for g in range(len(members)):
            a = max([lower[g]]+[value[k]+d for k,d in incoming[g]])
            b = min([limit[g]]+[value[k]-d for k,d in outgoing[g]])
            new = min(max(preferred[g],a),b)
            change = max(change,abs(new-value[g])); value[g] = new
        if change <= EPS:
            break
    result = {k:value[group[k]]+offset[k] for k in ids}
    return result, max(abs(result[k]-anchor[k]) for k in ids)


def _direction(a, b):
    """Freeze an axis/order using source geometry, never the fitted result."""
    sa,sb = a.get('original_source_bbox',a['source_bbox']),b.get('original_source_bbox',b['source_bbox'])
    same_row = a.get('row') is not None and a.get('row') == b.get('row')
    if same_row:
        axis = 'x'; origin = 'source-same-row'
    elif (a.get('row_order') is not None and b.get('row_order') is not None and
          a['row_order'] != b['row_order']):
        axis = 'y'; origin = 'source-row-order'
    else:
        separated_x = sa[2] <= sb[0]+EPS or sb[2] <= sa[0]+EPS
        separated_y = sa[3] <= sb[1]+EPS or sb[3] <= sa[1]+EPS
        if separated_x != separated_y:
            axis = 'x' if separated_x else 'y'; origin = 'source-separated-axis'
        else:
            dx = abs((sa[0]+sa[2])-(sb[0]+sb[2]))/max(EPS,sa[2]-sa[0]+sb[2]-sb[0])
            dy = abs((sa[1]+sa[3])-(sb[1]+sb[3]))/max(EPS,sa[3]-sa[1]+sb[3]-sb[1])
            axis = 'x' if dx > dy else 'y'; origin = 'source-centre-axis-inference'
    n = 0 if axis == 'x' else 1
    if axis == 'y' and origin == 'source-row-order':
        first,last = (a,b) if a['row_order'] < b['row_order'] else (b,a)
    else:
        first,last = sorted((a,b),key=lambda i:(i['source_bbox'][n],i['source_bbox'][n+2],str(i['id'])))
    return axis,first,last,origin


def solve_text_rectangles(items, bounds, constraints=(), *, max_nodes=512, max_seconds=.5):
    """Resolve collisions only along their source axis, retaining other anchors.

    Positive same-row source gaps are exact when that row needs horizontal
    collision repair. A clear row is never repacked merely to enforce a gap.
    New obstacles join through a complete sibling collision check. Failure or
    budget exhaustion retains the full input geometry for later adaptation.
    """
    started = time.perf_counter()
    if len(bounds)!=4 or not all(math.isfinite(v) for v in bounds) or bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
        raise ValueError('文本框联动边界无效')
    if not isinstance(max_nodes,int) or max_nodes < 1 or not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError('文本框联动预算无效')
    by_id = {i['id']:i for i in items}
    if len(by_id)!=len(items):
        raise ValueError('文本框联动ID重复')
    for i in items:
        if (len(i['source_bbox']) != 4 or not all(math.isfinite(v) for v in i['source_bbox']) or
                any(not math.isfinite(i[k]) for k in ('x','y','width','height')) or min(i['width'],i['height'])<=0):
            raise ValueError('文本框联动尺寸无效')
    original = {i['id']:[i['x'],i['y'],i['x']+i['width'],i['y']+i['height']] for i in items}
    edges = dict(x=[],y=[]); evidence = []; collisions = []; rows = {}
    active_rows = set(); added = set(); nodes = 0; failure = None

    def add(a,b,axis,gap,kind,origin):
        if a not in by_id or b not in by_id or axis not in edges or kind not in {'exact','minimum'} or not math.isfinite(gap):
            raise ValueError('文本框联动约束无效')
        delta = by_id[a]['width' if axis=='x' else 'height']+gap
        edges[axis].append((a,b,delta))
        if kind == 'exact':
            edges[axis].append((b,a,-delta))
        evidence.append(dict(a=a,b=b,axis=axis,gap=gap,kind=kind,origin=origin))

    for c in constraints:
        add(c['a'],c['b'],c['axis'],float(c['gap']),c.get('kind','exact'),'explicit')
    explicit_pairs = {(c['a'],c['b'],c['axis']) for c in constraints}
    for i in items:
        if i.get('row') is not None:
            rows.setdefault(i['row'],[]).append(i)
    heads = {}
    for row in rows.values():
        row.sort(key=lambda i:(i['source_bbox'][0],str(i['id'])))
        for a,b in zip(row,row[1:]):
            dy = b['y']-a['y']
            edges['y'].extend([(a['id'],b['id'],dy),(b['id'],a['id'],-dy)])
        first = row[0]; group = first.get('indentation_group',first.get('paragraph_id'))
        if group is not None:
            heads.setdefault(group,[]).append(first)
    for group in heads.values():
        for a,b in zip(group,group[1:]):
            dx = b['x']-a['x']
            # Every invocation takes a fresh snapshot; the second pass receives
            # post-alignment positions, never stale pre-alignment indentation.
            add(a['id'],b['id'],'x',dx-a['width'],'exact','current-row-indentation')
            evidence[-1]['offset_pt']=dx
    ordered_heads = sorted((r[0] for r in rows.values() if r[0].get('row_order') is not None),
                           key=lambda i:(i['row_order'],i['source_bbox'][1],str(i['id'])))
    for a,b in zip(ordered_heads,ordered_heads[1:]):
        if a['row_order'] != b['row_order']:
            # Keep line order without demanding extra inter-line whitespace
            # where the rectangles do not actually collide horizontally.
            edges['y'].append((a['id'],b['id'],0.))

    def activate(a,b):
        axis,first,last,origin = _direction(a,b)
        key = (axis,first['id'],last['id'])
        if key in added:
            return False
        added.add(key)
        # All fragments in the affected source row keep their declared gaps.
        row_key = first.get('row')
        if axis == 'x' and row_key is not None and row_key == last.get('row') and row_key not in active_rows:
            active_rows.add(row_key)
            for left,right in zip(rows[row_key],rows[row_key][1:]):
                if (left['id'],right['id'],'x') in explicit_pairs:
                    continue
                raw = right['source_bbox'][0]-left['source_bbox'][2]
                add(left['id'],right['id'],'x',max(0.,raw),'exact',
                    'source-gap' if raw>=0 else 'source-overlap-inferred-contact')
                evidence[-1]['raw_source_gap'] = raw
        add(first['id'],last['id'],axis,0.,'minimum',origin)
        collisions.append(dict(a=first['id'],b=last['id'],axis=axis,origin=origin))
        return True

    def pairs(pos):
        return [(a,b) for n,a in enumerate(items) for b in items[n+1:] if overlaps(pos[a['id']],pos[b['id']])]

    def inside(pos):
        return all(r[0]>=bounds[0]-EPS and r[1]>=bounds[1]-EPS and r[2]<=bounds[2]+EPS and r[3]<=bounds[3]+EPS for r in pos.values())

    def valid_edges(pos):
        return all(pos[b][n] >= pos[a][n]+d-EPS for axis,n in (('x',0),('y',1)) for a,b,d in edges[axis])

    def detail(status,pos):
        moved = {axis:[i['id'] for i in items if abs(pos[i['id']][n]-i[axis])>EPS]
                 for axis,n in (('x',0),('y',1))}
        return dict(version=VERSION,status=status,nodes=nodes,frontier=int(status=='undetermined'),
            constraints=evidence,collision_axes=collisions,
            moved_ids_by_axis=moved,initial_bboxes=original,
            source_bboxes={i['id']:list(i.get('original_source_bbox',i['source_bbox'])) for i in items},
            effective_source_bboxes={i['id']:list(i['source_bbox']) for i in items},
            failure_reasons=[failure] if failure else [],committed=status in {'optimal','feasible'},
            objective='minimum-maximum-axis-displacement-then-return-to-initial-anchors',
            proof_scope='fixed-source-collision-axes-and-declared-constraints',
            direction_policy='vertical-only-y-horizontal-only-x')

    initial_pairs = pairs(original)
    if not initial_pairs and inside(original) and valid_edges(original):
        return original,detail('unchanged',original)
    for a,b in initial_pairs:
        activate(a,b)
    while nodes < max_nodes and time.perf_counter()-started < max_seconds:
        nodes += 1
        x,reason_x = _axis(items,edges['x'],'x',bounds)
        y,reason_y = _axis(items,edges['y'],'y',bounds)
        if x is None or y is None:
            failure = reason_x if x is None else reason_y
            return original,detail('infeasible',original)
        pos = {i['id']:[x[i['id']],y[i['id']],x[i['id']]+i['width'],y[i['id']]+i['height']] for i in items}
        remaining = pairs(pos)
        if not remaining and inside(pos) and valid_edges(pos):
            return pos,detail('optimal',pos)
        changed = False
        for a,b in remaining:
            changed = activate(a,b) or changed
        if not changed:
            failure = 'fixed-axis-constraints-unsatisfied'
            return original,detail('infeasible',original)
    failure = 'propagation-budget-exhausted'
    return original,detail('undetermined',original)
