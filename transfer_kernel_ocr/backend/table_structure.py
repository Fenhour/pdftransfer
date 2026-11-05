"""Validate model table topology without rendering untrusted HTML."""
from collections import defaultdict
from html.parser import HTMLParser
import math
import re
import numpy as np

def _area(box):
    return max(0., box[2]-box[0])*max(0., box[3]-box[1])

def _intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))

def _box(value, width, height):
    """Validate model rectangle/polygon and clamp to this rendered page."""
    try:
        points = np.asarray(value, dtype=float)
        if points.shape == (4,):
            b = points.tolist()
        elif points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3:
            b = [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]
        else:
            return None
        if not all(math.isfinite(float(v)) for v in b):
            return None
        b = [max(0., min(width, float(b[0]))), max(0., min(height, float(b[1]))),
             max(0., min(width, float(b[2]))), max(0., min(height, float(b[3])))]
        return b if _area(b) > .01 else None
    except (TypeError, ValueError):
        return None

def _to_points(box, sx, sy):
    return [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]

def _plain(value):
    return re.sub(r'\s+', '', str(value))

class _Cells(HTMLParser):
    """Parse topology without rendering model-generated HTML or scripts."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells = []
        self.row = -1
        self.current = None
        self.occupied = set()
        self.table_depth = 0
        self.invalid = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table_depth += 1
            if self.table_depth > 1:
                self.invalid = True
        if tag == 'tr':
            self.row += 1
        if tag in {'td', 'th'}:
            if self.current is not None or self.row < 0:
                self.invalid = True
                return
            try:
                attrs = dict(attrs)
                rowspan, colspan = int(attrs.get('rowspan', 1)), int(attrs.get('colspan', 1))
                if not 1 <= rowspan <= 100 or not 1 <= colspan <= 100:
                    raise ValueError
            except (TypeError, ValueError):
                self.invalid = True
                return
            column = 0
            while (self.row, column) in self.occupied:
                column += 1
            slots = {(r, c) for r in range(self.row, self.row+rowspan) for c in range(column, column+colspan)}
            if slots & self.occupied or len(self.cells) >= 5000 or column+colspan > 100 or self.row+rowspan > 300:
                self.invalid = True
                return
            self.occupied |= slots
            self.current = dict(row=self.row, column=column, rowspan=rowspan, colspan=colspan,
                                parts=[], header=tag == 'th')
        elif tag == 'br' and self.current is not None:
            self.current['parts'].append('\n')

    def handle_endtag(self, tag):
        if tag in {'td', 'th'} and self.current is not None:
            cell = self.current
            cell['htmlText'] = ''.join(cell.pop('parts')).strip()
            self.cells.append(cell)
            self.current = None
        if tag == 'table':
            self.table_depth -= 1

    def handle_data(self, data):
        if self.current is not None:
            self.current['parts'].append(data)

def _html_signature(value):
    parser = _Cells()
    parser.feed(str(value))
    if parser.invalid or parser.current is not None or not parser.cells:
        return None
    return [(c['row'],c['column'],c['rowspan'],c['colspan'],_plain(c['htmlText'])) for c in parser.cells]

def _table_structure(result, block, sx, sy, w, h):
    parser = _Cells()
    parser.feed(str(result.get('pred_html') or block['text']))
    if parser.invalid or parser.current is not None or not parser.cells:
        return None
    cells = parser.cells
    nr = max(c['row']+c['rowspan'] for c in cells)
    nc = max(c['column']+c['colspan'] for c in cells)
    if len(parser.occupied) != nr*nc:
        # Missing <td> is unknown topology.  An explicit empty <td></td> is a
        # real cell; absent slots must not be fabricated as a recovered grid.
        return None
    pixels = [_box(b, w, h) for b in result.get('cell_box_list') or []]
    reliable = len(pixels) == len(cells) and all(pixels)
    b = block['bbox']
    if reliable:
        tolerance = max(4., min(b[2]-b[0], b[3]-b[1])*.05)
        expanded = [b[0]-tolerance,b[1]-tolerance,b[2]+tolerance,b[3]+tolerance]
        reliable = all(_intersection(v,expanded)/max(.01,_area(v)) > .95 for v in pixels)
        # Cell detector coordinates may have a pixel of overlap. Substantial
        # overlapping rectangles cannot all be independent HTML cells.
        active = []
        for detected in sorted(pixels,key=lambda v:v[1]):
            active = [other for other in active if other[3] > detected[1]]
            if any(_intersection(detected,other)/max(.01,min(_area(detected),_area(other))) > .15 for other in active):
                reliable = False
                break
            active.append(detected)
    xobs, yobs = defaultdict(list), defaultdict(list)
    if reliable:
        for cell, detected in zip(cells, pixels):
            cell['detectedBbox'] = _to_points(detected, sx, sy)
            xobs[cell['column']].append(detected[0])
            xobs[cell['column']+cell['colspan']].append(detected[2])
            yobs[cell['row']].append(detected[1])
            yobs[cell['row']+cell['rowspan']].append(detected[3])
    def edges(observations, count, low, high):
        if not observations:
            return np.linspace(low, high, count+1).tolist()
        observations.setdefault(0, [low])
        observations.setdefault(count, [high])
        indexes = sorted(observations)
        values = [float(np.median(observations[i])) for i in indexes]
        values = np.interp(range(count+1), indexes, values).tolist()
        return values if all(z-a > .05 for a, z in zip(values, values[1:])) else np.linspace(low, high, count+1).tolist()
    columns = [v*sx for v in edges(xobs, nc, b[0], b[2])]
    rows = [v*sy for v in edges(yobs, nr, b[1], b[3])]
    for index, cell in enumerate(cells):
        r, c = cell['row'], cell['column']
        cell.update(id=f'{block["id"]}-cell-{index}', textIds=[],
                    bbox=[columns[c], rows[r], columns[c+cell['colspan']], rows[r+cell['rowspan']]])
    native = result.get('_source') == 'native'
    return dict(id=block['id'], bbox=_to_points(b, sx, sy), rows=rows, columns=columns,
                cells=cells, contentType='table', layoutLabel='table',
                predHtml=str(result.get('pred_html') or block['text']),
                structureStatus='native-extracted' if native else 'model-predicted-needs-review', reviewRequired=not native,
                geometryStatus=('native-cell-boxes-aligned' if native else 'model-cell-boxes-aligned')
                               if reliable else 'estimated-from-html-topology',
                geometrySource='native' if native else 'v3',
                observedStyle=dict(borderColor='#000000', borderWidth=.5),
                templateId='table.native.v1' if native else 'table.v3.v1', borderElements=True)
