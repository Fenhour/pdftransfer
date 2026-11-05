"""Immutable, category-independent containment of original layout detections.

An x sweep discards horizontally separated candidates. All containment edges
are retained (overlapping outer boxes may both contain a child); the smallest
outer box is the deterministic direct parent. Equal boxes are not containment.
"""
from copy import deepcopy
import math

VERSION = 'source-detection-containment-v2-1pt'
TOLERANCE_PT = 1.


def strictly_contains(outer, inner, tolerance=(0., 0.)):
    ow,oh=outer[2]-outer[0],outer[3]-outer[1]
    iw,ih=inner[2]-inner[0],inner[3]-inner[1]
    # Strict area order prevents cycles; meaningful size growth avoids turning
    # almost-equal duplicate detections into parent/child pairs.
    return (ow*oh > iw*ih and (ow > iw+tolerance[0] or oh > ih+tolerance[1]) and
            all(outer[i] <= inner[i]+tolerance[i] for i in (0, 1)) and
            all(outer[i] >= inner[i]-tolerance[i-2] for i in (2, 3)))


def build_containment(boxes, *, pixel_to_point=(.5, .5)):
    """Return local detection-ID metadata without mutating model evidence."""
    if (len(pixel_to_point)!=2 or any(not math.isfinite(v) or v<=0 for v in pixel_to_point)):
        raise ValueError('包含关系像素到pt比例必须为两个正有限数')
    tolerance=tuple(TOLERANCE_PT/v for v in pixel_to_point)
    valid = {}
    for item in boxes:
        b = item.get('coordinate', [])
        if (len(b) == 4 and all(math.isfinite(v) for v in b) and
                b[2] > b[0] and b[3] > b[1]):
            valid[item['id']] = list(b)
    areas = {i: (b[2]-b[0])*(b[3]-b[1]) for i, b in valid.items()}
    ordered = sorted(valid, key=lambda i: (valid[i][0], -areas[i], i))
    parents = {i: [] for i in valid}
    inside = {i: [] for i in valid}
    active, comparisons = [], 0
    for current in ordered:
        b = valid[current]
        active = [i for i in active if valid[i][2]+tolerance[0] >= b[0]]
        for previous in active:
            # With tolerance a parent's left edge may follow its child's.
            parent,child=(previous,current) if areas[previous]>areas[current] else (current,previous)
            outer,inner=valid[parent],valid[child]
            if areas[parent] <= areas[child] or outer[2]+tolerance[0] < inner[2]:
                continue
            comparisons += 1
            if strictly_contains(outer, inner, tolerance):
                parents[child].append(parent)
                inside[parent].append(child)
        active.append(current)
    direct = {i: min(ps, key=lambda p: (areas[p], p)) if ps else None
              for i, ps in parents.items()}
    children = {i: [] for i in valid}
    for child, parent in direct.items():
        if parent is not None:
            children[parent].append(child)
    nodes = {i: dict(parent_id=direct[i], child_ids=sorted(children[i]),
                    child_count=len(children[i]), container_ids=sorted(parents[i]),
                    contained_ids=sorted(inside[i]), contained_count=len(inside[i]))
             for i in valid}
    return dict(version=VERSION, coordinate_space='original-layout-pixels',
                tolerance_pt=TOLERANCE_PT, tolerance_px=list(tolerance),
                nodes=nodes, source_rects=valid, candidate_checks=comparisons)


def block_metadata(node, page_id):
    """Block references are page-qualified; detection references stay local."""
    result = deepcopy(node)
    result['parent_id'] = f'{page_id}-{node["parent_id"]}' if node['parent_id'] is not None else None
    for key in ('child_ids', 'container_ids', 'contained_ids'):
        result[key] = [f'{page_id}-{i}' for i in node[key]]
    result['version'] = VERSION
    return result


def source_contains_pair(first, second):
    """Never infer a relationship from target/expanded/shifted rectangles.

    The geometry fallback supports old saved projects and synthetic fixtures.
    It uses source point boxes only; categories and target transforms are ignored.
    """
    a, b = first.get('source', {}), second.get('source', {})
    if a.get('page_id') != b.get('page_id'):
        return False
    ca, cb = a.get('containment', {}), b.get('containment', {})
    if ca.get('version') == cb.get('version') == VERSION:
        return (second['id'] in ca.get('contained_ids', ()) or
                first['id'] in cb.get('contained_ids', ()))
    ra, rb = a.get('bbox_pt'), b.get('bbox_pt')
    return bool(ra and rb and (strictly_contains(ra, rb, (TOLERANCE_PT,)*2) or
                               strictly_contains(rb, ra, (TOLERANCE_PT,)*2)))


def conflict_is_internal(diagnostic, identifier, nodes):
    """Only geometrical parent/child ambiguity is waived; rec failures remain."""
    if diagnostic.get('reason') not in {'ocr-line-crosses-layout-blocks', 'ocr-line-crosses-visual-block'}:
        return False
    involved = {str(i) for i in [diagnostic.get('block_id'),
                *diagnostic.get('candidate_ids', []), *diagnostic.get('excluded_ids', [])] if i}
    involved.discard(identifier)
    node = nodes.get(identifier, {})
    relatives = set(node.get('contained_ids', ())) | set(node.get('container_ids', ()))
    return bool(involved) and involved <= relatives
