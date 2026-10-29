"""Lightweight geometry metadata for text already accepted by reconstruction.

This module does not run OCR, rewrite text, split images, or infer document
semantics. A line contains independently positioned fragments, never a string
padded with guessed spaces. Groups remain reviewable geometric hypotheses.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import math
from statistics import median
from time import perf_counter


_ROW_NEIGHBORS = 24
_BLOCK_NEIGHBORS = 64


def _box(element):
    try:
        x, y, w, h = (float(element[k]) for k in ('x', 'y', 'width', 'height'))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        return None
    return [x, y, x+w, y+h]


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def _contains(a, b, tolerance=0.):
    return a[0]-tolerance <= b[0] and a[1]-tolerance <= b[1] and a[2]+tolerance >= b[2] and a[3]+tolerance >= b[3]


class _RectIndex:
    """A page-local grid: queries inspect nearby objects, not every object."""
    def __init__(self, objects, cell):
        self.objects = objects
        self.cell = max(1., cell)
        self.buckets = defaultdict(list)
        self.large = []
        for index, (_, box) in enumerate(objects):
            cells = self._cells(box)
            if len(cells) > 256:
                self.large.append(index)
            else:
                for key in cells:
                    self.buckets[key].append(index)

    def _cells(self, box):
        x0, y0, x1, y1 = [math.floor(v/self.cell) for v in box]
        # A huge image/frame should not allocate thousands of grid buckets.
        if (x1-x0+1)*(y1-y0+1) > 256:
            return range(257)
        return [(x, y) for x in range(x0, x1+1) for y in range(y0, y1+1)]

    def query(self, box):
        cells = self._cells(box)
        if len(cells) > 256:
            return self.objects
        indices = set(self.large)
        for key in cells:
            indices.update(self.buckets.get(key, ()))
        return [self.objects[i] for i in sorted(indices)]


def _char_width(element, box):
    # Only a spacing scale: no characters or blanks are reconstructed here.
    units = sum(1.8 if ord(c) > 0x2fff else 1. for c in element.get('text', '') if not c.isspace())
    h = box[3]-box[1]
    return min(h, max(h*.2, (box[2]-box[0])/max(1., units)))


def _rows(nodes, stats):
    rows = []
    for node in sorted(nodes, key=lambda n: (n['box'][3], n['box'][0], n['id'])):
        best = None
        for row in rows[-_ROW_NEIGHBORS:]:
            stats['rowComparisons'] += 1
            h = max(row['height'], node['height'])
            delta = abs(row['baseline']-node['box'][3])
            overlap = max(0., min(row['box'][3], node['box'][3])-max(row['box'][1], node['box'][1]))
            if delta <= h*.4 and (overlap >= min(row['height'], node['height'])*.45 or delta <= h*.2):
                if best is None or delta < best[0]:
                    best = (delta, row)
        if best is None:
            rows.append(dict(nodes=[node], box=list(node['box']), height=node['height'], baseline=node['box'][3]))
        else:
            row = best[1]
            row['nodes'].append(node)
            row['box'] = _union([row['box'], node['box']])
            # Keep the initial row anchor. Recomputing a median for every
            # appended word makes one very fragmented row quadratic, and a
            # drifting anchor can chain neighboring baselines together.
    return rows


def _gutters(rows, typical):
    """Repeated horizontal gaps are evidence for a vertical column barrier.

    A full-width title can span these barriers. A single large gap still
    separates line fragments, but is not promoted to a page-wide column.
    """
    events = []
    row_boxes = {}
    eligible = 0
    for index, row in enumerate(rows):
        ordered = sorted(row['nodes'], key=lambda n: n['box'][0])
        if len(ordered) < 2:
            continue
        eligible += 1
        row_boxes[index] = row['box']
        for a, b in zip(ordered, ordered[1:]):
            gap = b['box'][0]-a['box'][2]
            required = max(typical*.9, min(a['charWidth'], b['charWidth'])*1.2)
            if gap >= required:
                pad = typical*.12
                events.extend([(a['box'][2]+pad, 1, index), (b['box'][0]-pad, -1, index)])
    minimum = max(2, math.ceil(eligible*.3))
    active = defaultdict(int)
    spans = []
    previous = None
    for x, direction, row_index in sorted(events):
        if previous is not None and x > previous and len(active) >= minimum:
            if spans and abs(spans[-1]['right']-previous) < 1e-6:
                spans[-1]['right'] = x
                spans[-1]['rows'].update(active)
            else:
                spans.append(dict(left=previous, right=x, rows=set(active)))
        active[row_index] += direction
        if not active[row_index]:
            del active[row_index]
        previous = x
    return [dict(x=(s['left']+s['right'])/2,
                 bbox=[s['left'], min(row_boxes[r][1] for r in s['rows']),
                       s['right'], max(row_boxes[r][3] for r in s['rows'])],
                 supportRows=len(s['rows']), evidence='repeated-whitespace')
            for s in spans if s['right']-s['left'] >= typical*.35]


def _barrier(index, box):
    return any(_overlap(box, b) > 0 for _, b in index.query(box))


def _line_groups(rows, gutters, barriers, typical, stats):
    lines = []
    gutter_xs = [g['x'] for g in gutters]
    for row in sorted(rows, key=lambda r: (r['baseline'], r['box'][0])):
        current = []
        for node in sorted(row['nodes'], key=lambda n: (n['box'][0], n['id'])):
            join = False
            if current:
                previous = current[-1]
                a, b = previous['box'], node['box']
                gap = b[0]-a[2]
                scale = max(previous['height'], node['height'])
                allowed = min(scale*2., max(scale*1.1, min(previous['charWidth'], node['charWidth'])*3.))
                local_gutters = gutters[bisect_left(gutter_xs, a[2]):bisect_right(gutter_xs, b[0])]
                crossed_gutter = any(row['box'][3] >= g['bbox'][1] and row['box'][1] <= g['bbox'][3]
                                    for g in local_gutters)
                corridor = [a[2], min(a[1], b[1]), b[0], max(a[3], b[3])]
                stats['fragmentComparisons'] += 1
                join = (-min(a[2]-a[0], b[2]-b[0])*.2 <= gap <= allowed and
                        not crossed_gutter and not _barrier(barriers, corridor))
            if current and not join:
                lines.append(_line(current, typical))
                current = []
            current.append(node)
        if current:
            lines.append(_line(current, typical))
    return lines


def _line(nodes, typical):
    box = _union([n['box'] for n in nodes])
    height = median(n['height'] for n in nodes)
    candidates = [n for n in nodes if n['element'].get('contentType') == 'heading']
    # Fitted fontSize may itself be the defect under investigation. Geometric
    # height, not one anomalous font estimate, must support a heading boundary.
    heading = bool(candidates) and (height >= typical*1.1 or
                                   all(n['element'].get('source') == 'native' for n in candidates))
    return dict(nodes=nodes, bbox=box, baseline=median(n['box'][3] for n in nodes),
                height=height, role='heading-candidate' if heading else 'body')


def _block_match(previous, line, barriers, typical, stats):
    stats['blockComparisons'] += 1
    a, b = previous['bbox'], line['bbox']
    h = max(previous['height'], line['height'])
    small = min(previous['height'], line['height'])
    gap = b[1]-a[3]
    if line['baseline']-previous['baseline'] < small*.5 or not -small*.3 <= gap <= h*.9:
        return None
    if h/small > 1.35 or previous['role'] != line['role']:
        return None
    overlap = max(0., min(a[2], b[2])-max(a[0], b[0]))
    width = min(a[2]-a[0], b[2]-b[0])
    aligned = abs(a[0]-b[0]) <= h*1.75 or overlap >= width*.65
    if not aligned:
        return None
    corridor = [min(a[0], b[0]), a[3], max(a[2], b[2]), b[1]]
    if _barrier(barriers, corridor):
        return None
    return max(0., gap)/max(1., typical)+abs(a[0]-b[0])/max(1., h)*.1


def organize_text(page):
    """Annotate accepted text in place and return ``page['textLayout']``.

    ``regionId``, ``lineId`` and ``blockId`` are geometric grouping hints;
    ``fragmentIndex`` / ``fragmentCount`` describe the line's source elements.
    No existing ID, text, coordinate, style or image is changed. Explicit
    table-cell ownership and rectangular frame containment are kept separate.
    Rotated text is conservatively left as individual lines and blocks.
    """
    started = perf_counter()
    elements = page.get('elements', [])
    stats = dict(textElements=0, skippedElements=0, rotatedFragments=0,
                 rowComparisons=0, fragmentComparisons=0, blockComparisons=0)
    nodes = []
    for element in elements:
        if element.get('type') != 'text':
            continue
        stats['textElements'] += 1
        box = _box(element)
        if not box or element.get('visible', True) is False or not element.get('text', '').strip():
            stats['skippedElements'] += 1
            continue
        rotation = float(element.get('rotation', 0)) % 360
        rotated = min(rotation, 360-rotation) > 5
        nodes.append(dict(element=element, id=element['id'], box=box,
                          height=box[3]-box[1], charWidth=_char_width(element, box), rotated=rotated))
        stats['rotatedFragments'] += int(rotated)
    typical = median(n['height'] for n in nodes) if nodes else 12.
    frames = [(e, b) for e in elements if e.get('type') == 'shape' and e.get('shape', 'rect') == 'rect'
              and e.get('visible', True)
              and (e.get('contentType') == 'decoration' or e.get('templateId') == 'decoration.rect.v1')
              and (b := _box(e))]
    frame_index = _RectIndex(frames, typical*8)
    barriers = []
    for element in elements:
        box = _box(element)
        if not box or not element.get('visible', True):
            continue
        rule = (element.get('type') == 'shape' and
                (element.get('shape') == 'line' or element.get('contentType') == 'decoration' and
                 min(box[2]-box[0], box[3]-box[1]) <= typical*.25))
        if element.get('type') in {'image', 'region'} or rule:
            barriers.append((element, box))
    image_index = _RectIndex(barriers, typical*8)
    cells_by_text = {}
    tables = {t['id']: t for t in page.get('tables', [])}
    for table in tables.values():
        for cell_index, cell in enumerate(table.get('cells', [])):
            for text_id in cell.get('textIds', []):
                cells_by_text[text_id] = (table['id'], cell_index)
    scopes = defaultdict(list)
    scope_sources = {}
    for node in nodes:
        element, box = node['element'], node['box']
        owner = cells_by_text.get(node['id'])
        if owner is None and element.get('tableId'):
            table = tables.get(element['tableId'], {})
            matches = [i for i, c in enumerate(table.get('cells', [])) if _contains(c['bbox'], box, .5)]
            owner = (element['tableId'], matches[0] if len(matches) == 1 else -1)
        if owner:
            key = ('table', *owner)
            scope_sources[key] = dict(containerType='table-cell' if owner[1] >= 0 else 'table',
                                      containerId=owner[0], cellIndex=owner[1], evidence='existing-table-ownership')
        else:
            containing = [(e, b) for e, b in frame_index.query(box) if _contains(b, box, .5)]
            frame_id = element.get('frameId')
            if not frame_id and containing:
                frame_id = min(containing, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))[0]['id']
            key = ('frame', frame_id) if frame_id else ('page',)
            scope_sources[key] = (dict(containerType='frame', containerId=frame_id,
                                       evidence='existing-frame-ownership' if element.get('frameId') else 'frame-containment')
                                  if frame_id else dict(containerType='page', containerId=page.get('id'), evidence='accepted-independent-text'))
        scopes[key].append(node)
    layout = dict(schema='text-layout/1', method='bounded-local-geometry-v1',
                  semanticsInferred=False, coordinates='top-left-pdf-points',
                  regions=[], columns=[], lines=[], blocks=[],
                  limitations=['Geometric groups require review; no semantic relationships are inferred.',
                               'Single OCR elements are not resegmented or corrected.',
                               'External PDF editors may regroup positioned text.'])
    for key, scoped in sorted(scopes.items(), key=lambda item: min((n['box'][1], n['box'][0]) for n in item[1])):
        normal = [n for n in scoped if not n['rotated']]
        local_height = median(n['height'] for n in normal) if normal else typical
        rows = _rows(normal, stats)
        gutters = _gutters(rows, local_height)
        scope_number = len(layout['columns'])+1
        column_id = f'text-columns-{scope_number}'
        layout['columns'].append(dict(id=column_id, **scope_sources[key], gutters=gutters,
                                      count=len(gutters)+1, groupingEvidence='repeated-whitespace' if gutters else 'single-geometric-scope'))
        lines = _line_groups(rows, gutters, image_index, local_height, stats)
        lines += [dict(_line([n], local_height), rotated=True) for n in scoped if n['rotated']]
        grouped = defaultdict(list)
        gutter_xs = [g['x'] for g in gutters]
        for line in lines:
            b = line['bbox']
            crossed = bisect_right(gutter_xs, b[0]) < bisect_left(gutter_xs, b[2])
            column = 'spanning' if crossed else bisect_right(gutter_xs, (b[0]+b[2])/2)
            grouped[column].append(line)
        for column, region_lines in sorted(grouped.items(), key=lambda item: min((l['bbox'][1], l['bbox'][0]) for l in item[1])):
            region_id = f'text-region-{len(layout["regions"])+1}'
            region = dict(id=region_id, **scope_sources[key], columnGroupId=column_id, columnIndex=column,
                          bbox=_union([l['bbox'] for l in region_lines]), lineIds=[], blockIds=[])
            layout['regions'].append(region)
            active = []
            for line in sorted(region_lines, key=lambda l: (l['baseline'], l['bbox'][0])):
                line_id = f'text-line-{len(layout["lines"])+1}'
                active = [b for b in active if line['bbox'][1]-b['_last']['bbox'][3] <= max(local_height, line['height'], b['_last']['height'])*1.1]
                # Dense independent labels at one baseline cannot all be
                # preceding paragraph tails. Bound even this ambiguous case;
                # uncertainty creates an extra block, never a merged object.
                active = active[-_BLOCK_NEIGHBORS*2:]
                nearest = sorted(active, key=lambda b: abs(b['_last']['bbox'][0]-line['bbox'][0]))[:_BLOCK_NEIGHBORS]
                matches = [] if line.get('rotated') else [(score, b) for b in nearest
                    if not b['_last'].get('rotated') and
                    (score := _block_match(b['_last'], line, image_index, local_height, stats)) is not None]
                if matches:
                    block = min(matches, key=lambda m: m[0])[1]
                else:
                    block = dict(id=f'text-block-{len(layout["blocks"])+1}', regionId=region_id,
                                 role=line['role'], lineIds=[], bbox=list(line['bbox']), _last=line)
                    layout['blocks'].append(block)
                    region['blockIds'].append(block['id'])
                    active.append(block)
                block['lineIds'].append(line_id)
                block['bbox'] = _union([block['bbox'], line['bbox']])
                block['_last'] = line
                fragment_ids = [n['id'] for n in line['nodes']]
                fragments = []
                for index, node in enumerate(line['nodes']):
                    element = node['element']
                    element.update(regionId=region_id, lineId=line_id, blockId=block['id'],
                                   fragmentIndex=index, fragmentCount=len(fragment_ids))
                    fragment = dict(elementId=node['id'], bbox=list(node['box']),
                                    source=element.get('source', 'unknown'), gapBefore=None if not index else
                                    node['box'][0]-line['nodes'][index-1]['box'][2])
                    for field in ('sourceLineId', 'ocrLineId', 'sourceCandidateId'):
                        if field in element:
                            fragment[field] = element[field]
                    fragments.append(fragment)
                layout['lines'].append(dict(id=line_id, regionId=region_id, blockId=block['id'],
                    bbox=line['bbox'], baseline=line['baseline'], height=line['height'], role=line['role'],
                    fragmentIds=fragment_ids, fragments=fragments, rotated=line.get('rotated', False),
                    evidence='source-element-geometry', exact=False))
                region['lineIds'].append(line_id)
    for block in layout['blocks']:
        del block['_last']
    stats.update(lineCount=len(layout['lines']), blockCount=len(layout['blocks']),
                 regionCount=len(layout['regions']), columnGroupCount=len(layout['columns']),
                 elapsedSeconds=perf_counter()-started)
    layout['stats'] = stats
    page['textLayout'] = layout
    return layout
