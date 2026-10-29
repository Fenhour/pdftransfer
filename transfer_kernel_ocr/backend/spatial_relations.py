"""07: immutable source geometry, local spatial evidence and reading order.

Regions are optional isolation frames, not backgrounds or semantic containers.
Spatial neighbours describe geometry; only unambiguous local attachments add
hard reading dependencies. Confidence values are rule scores, not calibrated
model probabilities.
"""
from __future__ import annotations

from copy import deepcopy
import math


VERSION = '07-spatial-v2'
POLICY = 'source-preserving-regions-v1'
HEADINGS = {'doc_title', 'paragraph_title', 'section_title', 'title', 'reference'}
_EPS = 1e-6


def _area(b):
    return max(0., b[2]-b[0])*max(0., b[3]-b[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))


def _box(value):
    try:
        b = list(map(float, value))
        return b if len(b) == 4 and all(math.isfinite(x) for x in b) and _area(b) > _EPS else None
    except (TypeError, ValueError):
        return None


def _union(items):
    boxes = [b['source']['bbox_pt'] for b in items]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _xy_order(items, groups, depth=0):
    """Keep column-wise XY cuts, separating spanning bands before their rows."""
    if len(items) < 2 or depth >= 40:
        return sorted(items, key=lambda b: (*b['source']['bbox_pt'][1::-1], b['id']))
    for axis, threshold in ((0, 8.), (1, 3.)):
        ordered = sorted(items, key=lambda b: (b['source']['bbox_pt'][axis], b['id']))
        edge = ordered[0]['source']['bbox_pt'][axis+2]
        cuts = []
        for i, item in enumerate(ordered[1:], 1):
            bounds = item['source']['bbox_pt']
            distance = bounds[axis]-edge
            if distance > threshold:
                cuts.append((distance, i, edge, bounds[axis]))
            edge = max(edge, bounds[axis+2])
        if not cuts:
            continue
        cut = max(cuts)[1]
        if axis == 1:
            total = _union(items)
            spanning = [c for c in cuts if any(
                b['source']['bbox_pt'][2]-b['source']['bbox_pt'][0] >= .8*(total[2]-total[0])
                and (abs(b['source']['bbox_pt'][3]-c[2]) < .1 or
                     abs(b['source']['bbox_pt'][1]-c[3]) < .1) for b in items)]
            if spanning:
                cut = min(spanning, key=lambda c: c[1])[1]
        first, second = ordered[:cut], ordered[cut:]
        groups.append(dict(type='columns' if axis == 0 else 'bands',
                           children=[[b['id'] for b in first], [b['id'] for b in second]]))
        return _xy_order(first, groups, depth+1)+_xy_order(second, groups, depth+1)
    return sorted(items, key=lambda b: (b['source']['bbox_pt'][1], b['source']['bbox_pt'][0], b['id']))


def _tree(blocks):
    seen, contains = set(), []

    def visit(block, ancestors):
        identifier = block['id']
        if identifier in ancestors:
            raise ValueError('容器父子关系存在循环')
        if identifier in seen:
            raise ValueError('模块 ID 重复，子项只能有一个可见归属')
        seen.add(identifier)
        for child in block.get('content', {}).get('children', []):
            contains.append(dict(parent=identifier, child=child['id']))
            visit(child, ancestors | {identifier})

    for block in blocks:
        visit(block, set())
    return contains


def _page_frames(page, items, issues):
    pid = page['id']
    pw = float(page.get('width_pt', page.get('width_px', 0)))
    ph = float(page.get('height_pt', page.get('height_px', 0)))
    if not all(math.isfinite(v) and v > 0 for v in (pw, ph)):
        raise ValueError('源页面尺寸无效，不能建立区域坐标')
    sx, sy = pw/float(page.get('width_px', pw)), ph/float(page.get('height_px', ph))
    regions = []
    hints = {b['id']: [] for b in items}
    for index, raw in enumerate(page.get('detections', {}).get('regions', [])):
        pixels = _box(raw.get('coordinate'))
        if pixels is None:
            issues.append(dict(page_id=pid, reason='invalid-region-box', detection_index=index))
            continue
        b = [pixels[0]*sx, pixels[1]*sy, pixels[2]*sx, pixels[3]*sy]
        if b[0] < 0 or b[1] < 0 or b[2] > pw or b[3] > ph:
            issues.append(dict(page_id=pid, reason='region-outside-source-page', bbox_pt=b))
            continue
        identifier = f'{pid}-{raw.get("id", f"region-{index}")}'
        duplicate = next((r for r in regions if _intersection(b, r['bbox_pt'])/
                          (_area(b)+_area(r['bbox_pt'])-_intersection(b, r['bbox_pt'])) >= .95), None)
        if duplicate:
            duplicate['evidence']['duplicate_detection_ids'].append(identifier)
            issues.append(dict(page_id=pid, region_id=identifier, canonical=duplicate['id'],
                               reason='duplicate-region-detection'))
            continue
        regions.append(dict(id=identifier, page_id=pid, bbox_pt=b, block_ids=[], kind='region',
                            isolated=True, evidence=dict(method='unique-source-coverage-and-separation',
                                duplicate_detection_ids=[], rule_confidence=.9)))
    # Prefer the outer boundary for nested predictions. This provides one
    # isolation scale without inventing a hierarchy of model Region boxes.
    outer = []
    for region in regions:
        if any(other is not region and _area(other['bbox_pt']) > _area(region['bbox_pt']) and
               _intersection(region['bbox_pt'], other['bbox_pt'])/_area(region['bbox_pt']) >= .995
               for other in regions):
            issues.append(dict(page_id=pid, region_id=region['id'], reason='nested-region-uses-outer-frame'))
        else:
            outer.append(region)
    invalid = set()
    for i, a in enumerate(outer):
        for b in outer[i+1:]:
            if a['id'] == b['id'] or _intersection(a['bbox_pt'], b['bbox_pt']) > _EPS:
                invalid.update((a['id'], b['id']))
    for region in outer:
        if region['id'] in invalid:
            issues.append(dict(page_id=pid, region_id=region['id'], reason='overlapping-region-frames'))
    available = [r for r in outer if r['id'] not in invalid]
    owners = {}
    for item in items:
        b = item['source']['bbox_pt']
        for r in regions:
            ratio = _intersection(b, r['bbox_pt'])/_area(b)
            if ratio >= .65:
                hints[item['id']].append(dict(region_id=r['id'], coverage=ratio))
        threshold = .995 if item['kind'] == 'image' else .9
        candidates = [r for r in available if _intersection(b, r['bbox_pt'])/_area(b) >= threshold]
        if len(candidates) == 1:
            owners[item['id']] = candidates[0]['id']
        elif hints[item['id']]:
            issues.append(dict(page_id=pid, block_id=item['id'], reason='region-membership-is-hint-only',
                               candidates=deepcopy(hints[item['id']]), required_coverage=threshold))
    frames = []
    for r in available:
        members = [b for b in items if owners.get(b['id']) == r['id']]
        outsiders = [b for b in items if owners.get(b['id']) != r['id'] and
                     _intersection(b['source']['bbox_pt'], r['bbox_pt']) > .05*_area(b['source']['bbox_pt'])]
        if outsiders:
            issues.append(dict(page_id=pid, region_id=r['id'], reason='region-crosses-unowned-content',
                               block_ids=[b['id'] for b in outsiders]))
            for member in members:
                owners.pop(member['id'], None)
        elif members:
            r['block_ids'] = [b['id'] for b in members]
            r['evidence']['minimum_coverage'] = min(_intersection(b['source']['bbox_pt'], r['bbox_pt'])/
                                                   _area(b['source']['bbox_pt']) for b in members)
            frames.append(r)
    remaining = [b['id'] for b in items if b['id'] not in owners]
    if remaining:
        frames.append(dict(id=f'{pid}-page-frame', page_id=pid, kind='page', bbox_pt=[0., 0., pw, ph],
                           isolated=False, block_ids=remaining, evidence=dict(method='source-page-fallback')))
    membership = {identifier: frame['id'] for frame in frames for identifier in frame['block_ids']}
    region_membership = {b['id']: owners.get(b['id']) for b in items}
    return frames, membership, region_membership, hints


def _projection(a, b, axis):
    length = max(0., min(a[axis+2], b[axis+2])-max(a[axis], b[axis]))
    return length/max(_EPS, min(a[axis+2]-a[axis], b[axis+2]-b[axis]))


def _spatial_edges(items, frame):
    candidates = []
    bounds = {b['id']: b['source']['bbox_pt'] for b in items}
    for i, first in enumerate(items):
        for second in items[i+1:]:
            a, b = bounds[first['id']], bounds[second['id']]
            for axis in (0, 1):
                transverse = 1-axis
                projection = _projection(a, b, transverse)
                if projection < .2:
                    continue
                before, after = (first, second) if a[axis] <= b[axis] else (second, first)
                left, right = bounds[before['id']], bounds[after['id']]
                gap = right[axis]-left[axis+2]
                if gap < -_EPS:
                    continue
                lo, hi = max(left[transverse], right[transverse]), min(left[transverse+2], right[transverse+2])
                blocked = any(c['id'] not in {before['id'], after['id']} and
                    bounds[c['id']][axis] < right[axis]-_EPS and bounds[c['id']][axis+2] > left[axis+2]+_EPS and
                    max(0., min(hi, bounds[c['id']][transverse+2])-max(lo, bounds[c['id']][transverse])) >= .8*(hi-lo)
                    for c in items)
                if not blocked:
                    size = frame['bbox_pt'][axis+2]-frame['bbox_pt'][axis]
                    candidates.append(dict(a=before['id'], b=after['id'], relation='left_of' if axis == 0 else 'above',
                        gap_pt=gap, normalized_gap=gap/size, projection_overlap=projection, frame_id=frame['id'],
                        confidence=.85, evidence='projected-visible-neighbours-without-intervening-block'))
                break
    return candidates


def _stable_order(ids, edges):
    predecessors = {identifier: set() for identifier in ids}
    for a, b in edges:
        if a not in predecessors or b not in predecessors:
            raise ValueError('阅读顺序约束引用了不存在的顶层模块')
        predecessors[b].add(a)
    pending, done, result = list(ids), set(), []
    while pending:
        identifier = next((v for v in pending if predecessors[v] <= done), None)
        if identifier is None:
            raise ValueError('阅读顺序约束存在循环')
        pending.remove(identifier)
        done.add(identifier)
        result.append(identifier)
    return result


def _attachments(ordered, edges):
    by_id = {b['id']: b for b in ordered}
    rank = {b['id']: i for i, b in enumerate(ordered)}
    attachments, hard, soft = [], [], []
    for block in ordered:
        label = block['source'].get('label', '')
        if label not in HEADINGS | {'figure_title', 'table_title'}:
            continue
        candidates = []
        for edge in edges:
            if edge['relation'] != 'above' or edge['projection_overlap'] < .5 or block['id'] not in (edge['a'], edge['b']):
                continue
            other = by_id[edge['b'] if edge['a'] == block['id'] else edge['a']]
            box = block['source']['bbox_pt']
            if edge['gap_pt'] > max(24., 2*(box[3]-box[1])):
                continue
            if label in HEADINGS:
                if edge['a'] != block['id'] or other['source'].get('label') in HEADINGS:
                    continue
                kind = 'heading-next'
            else:
                if (other['kind'] not in {'image', 'table'} or
                        other['source'].get('label') not in {'image', 'chart', 'table'}):
                    continue
                kind = 'caption'
            candidates.append((edge['gap_pt'], other['id'], kind, edge))
        candidates.sort(key=lambda c: (c[0], c[1]))
        if not candidates:
            continue
        best = candidates[0]
        ambiguous = len(candidates) > 1 and candidates[1][0]-best[0] <= max(3., best[0]*.2)
        a, b = sorted((block['id'], best[1]), key=rank.get)
        strong = not ambiguous and best[3]['projection_overlap'] >= .8
        record = dict(before=a, after=b, type=best[2], confidence=.9 if strong else .6,
                      evidence=dict(gap_pt=best[0], projection_overlap=best[3]['projection_overlap'],
                                    ambiguous=ambiguous, frame_id=best[3]['frame_id']))
        if strong:
            hard.append([a, b])
            attachments.append(dict(anchor=block['id'] if best[2] == 'heading-next' else best[1],
                                    dependent=best[1] if best[2] == 'heading-next' else block['id'],
                                    type=best[2], keep='prefer', confidence=.9, evidence=record['evidence']))
        else:
            soft.append(record)
    return attachments, hard, soft


def derive_spatial_relations(project):
    blocks = project.get('blocks', [])
    contains = _tree(blocks)
    groups, issues, frames, frame_membership, region_membership, hints, ordered, spatial = [], [], [], {}, {}, {}, [], []
    pages = project.get('source_pages', [])
    if len({p['id'] for p in pages}) != len(pages):
        raise ValueError('源页面 ID 重复')
    if any(b['source']['page_id'] not in {p['id'] for p in pages} for b in blocks):
        raise ValueError('模块引用了不存在的源页面')
    for block in blocks:
        if _box(block['source'].get('bbox_pt')) is None:
            raise ValueError('模块源坐标无效')
    for page in pages:
        items = [b for b in blocks if b['source']['page_id'] == page['id']]
        fs, membership, regions, page_hints = _page_frames(page, items, issues)
        frames.extend(fs)
        frame_membership.update(membership)
        region_membership.update(regions)
        hints.update(page_hints)
        ordered.extend(_xy_order(items, groups))
        for frame in fs:
            members = [b for b in items if membership[b['id']] == frame['id']]
            spatial.extend(_spatial_edges(members, frame))
    attachments, hard, soft = _attachments(ordered, spatial)
    explicit = []
    for edge in project.get('relations', {}).get('hard_before', []):
        if isinstance(edge, dict):
            explicit.append([edge.get('from', edge.get('before')), edge.get('to', edge.get('after'))])
        else:
            explicit.append(list(edge))
    # The reference order is the geometry result. Explicit dependencies are
    # checked for cycles but remain separate so layout can report corrections.
    if explicit:
        _stable_order([b['id'] for b in ordered], explicit)
    inferred = hard
    hard = [list(pair) for pair in dict.fromkeys(tuple(e) for e in explicit)]
    for edge in inferred:
        if edge in hard:
            continue
        try:
            _stable_order([b['id'] for b in ordered], hard+[edge])
        except ValueError:
            # A fallible local attachment cannot overrule an explicit valid
            # dependency. Keep its evidence, but remove the inferred lock.
            issues.append(dict(reason='inferred-reading-edge-conflicts-with-explicit-constraint',
                               before=edge[0], after=edge[1]))
            soft.append(dict(before=edge[0], after=edge[1], confidence=.6,
                             evidence=dict(reason='explicit-dependency-takes-priority')))
            attachments = [a for a in attachments if {a['anchor'], a['dependent']} != set(edge)]
        else:
            hard.append(edge)
    return dict(algorithm_version=VERSION, layout_policy=POLICY,
        source_order=[b['id'] for b in ordered], hard_before=hard, soft_before=soft,
        contains=contains, frames=frames, frame_membership=frame_membership,
        region_membership=region_membership, region_hints=hints, spatial_edges=spatial,
        layout_groups=groups, alignment_references={b['id']: deepcopy(b.get('local_layout', {}).get('anchor'))
            for b in blocks if b.get('local_layout', {}).get('anchor')},
        attachments=attachments, reorder_groups=[], reorder_policy='source-order-preserved', issues=issues)


derive_relations = derive_spatial_relations
