"""Native table geometry and conservative recovery of captioned sparse-rule tables."""
import re
from statistics import median

import pymupdf as fitz

from .layout import inside
from .model import Block
from .text import clean_inline, visual_lines

CAPTION = re.compile(r"^(?:Table|表)\s*(?:[A-Z][.]?)?\s*\d+", re.I)


def _edges(values, tolerance=1.5):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([float(value)])
        else:
            groups[-1].append(float(value))
    return [sum(group) / len(group) for group in groups]


def _rules(page):
    horizontal, vertical = [], []
    for drawing in page.get_drawings():
        # Filled chart bars must not become table rules.
        if drawing.get('type') not in ('s', 'fs'):
            # TeX often paints thin filled rectangles instead of stroked rules.
            for item in drawing['items']:
                if item[0] == 're':
                    r = item[1]
                    if r.height <= 2 and r.width >= 80:
                        horizontal.append([r.x0, (r.y0+r.y1)/2, r.x1])
                    elif r.width <= 2 and r.height >= 15:
                        vertical.append([(r.x0+r.x1)/2, r.y0, r.y1])
            continue
        for item in drawing['items']:
            segments = []
            if item[0] == 'l':
                segments = [(item[1], item[2])]
            elif item[0] == 're':
                r = item[1]
                segments = [(r.tl, r.tr), (r.bl, r.br), (r.tl, r.bl), (r.tr, r.br)]
            for a, b in segments:
                if abs(a.y - b.y) < 1 and abs(a.x - b.x) > 12:
                    horizontal.append([min(a.x, b.x), (a.y + b.y) / 2, max(a.x, b.x)])
                elif abs(a.x - b.x) < 1 and abs(a.y - b.y) > 8:
                    vertical.append([(a.x + b.x) / 2, min(a.y, b.y), max(a.y, b.y)])
    return horizontal, vertical


def _border(start, end, fixed, rules):
    spans = sorted((max(start, a), min(end, b)) for a, y, b in rules
                   if abs(y - fixed) < 2 and b > start and a < end)
    covered, prior = 0, start
    for a, b in spans:
        covered += max(0, b - max(a, prior))
        prior = max(prior, b)
    return covered >= (end - start) * .55


def _style_cells(cells, horizontal, vertical):
    v = [[y0, x, y1] for x, y0, y1 in vertical]
    for cell in cells:
        x0, y0, x1, y1 = cell['bbox']
        cell['borders'] = {
            'top': _border(x0, x1, y0, horizontal),
            'bottom': _border(x0, x1, y1, horizontal),
            'left': _border(y0, y1, x0, v),
            'right': _border(y0, y1, x1, v),
        }


def _meta(xs, ys, cells, method):
    return {'grid_cells': cells, 'column_widths': [b-a for a, b in zip(xs, xs[1:])],
            'row_heights': [b-a for a, b in zip(ys, ys[1:])], 'strategy': method,
            'geometry_source': 'native', 'header_rows': 1}


def _native_layout(table, horizontal, vertical):
    values = table.extract()
    boxes = [cell for row in table.rows for cell in row.cells if cell is not None]
    xs = _edges([table.bbox[0], table.bbox[2]] + [cell[i] for cell in boxes for i in (0, 2)])
    ys = _edges([table.bbox[1], table.bbox[3]] + [cell[i] for cell in boxes for i in (1, 3)])
    rows = [[None for _ in range(len(xs)-1)] for _ in range(len(ys)-1)]
    cells, seen = [], set()
    for r, row in enumerate(table.rows):
        for c, box in enumerate(row.cells):
            if box is None:
                continue
            col = min(range(len(xs)), key=lambda i: abs(xs[i]-box[0]))
            end_col = min(range(len(xs)), key=lambda i: abs(xs[i]-box[2]))
            start_row = min(range(len(ys)), key=lambda i: abs(ys[i]-box[1]))
            end_row = min(range(len(ys)), key=lambda i: abs(ys[i]-box[3]))
            key = (start_row, col, end_row, end_col)
            if key in seen or end_col <= col or end_row <= start_row:
                continue
            seen.add(key)
            text = str(values[r][c] or '')
            rows[start_row][col] = text
            cells.append({'row': start_row, 'col': col, 'rowspan': end_row-start_row,
                          'colspan': end_col-col, 'bbox': list(box), 'text': text})
    # Covered coordinates remain None; genuine blank cells have their own geometry.
    _style_cells(cells, horizontal, vertical)
    meta = _meta(xs, ys, cells, 'lines_strict')
    meta['external_header'] = bool(table.header.external)
    meta['header_names'] = list(table.header.names)
    meta['header_rows'] = 0 if table.header.external else 1
    return Block('table', 0, list(table.bbox), rows=rows, meta=meta)


def _word_rows(words):
    groups = []
    for word in sorted(words, key=lambda w: ((w[1]+w[3])/2, w[0])):
        center = (word[1]+word[3])/2
        h = word[3]-word[1]
        if groups and abs(center-median((w[1]+w[3])/2 for w in groups[-1])) < max(h, median(w[3]-w[1] for w in groups[-1])) * .6:
            groups[-1].append(word)
        else:
            groups.append([word])
    return [sorted(row, key=lambda w: w[0]) for row in groups]


def _cell_text(words):
    return '\n'.join(clean_inline(' '.join(w[4] for w in row)) for row in _word_rows(words))


def _sparse_regions(page, horizontal):
    captions = []
    for raw in page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)['blocks']:
        if raw['type'] == 0:
            captions.extend(line for line in visual_lines(raw) if CAPTION.match(line['text']))
    groups = []
    for line in sorted(horizontal, key=lambda line: (line[0], line[2], line[1])):
        if line[2]-line[0] < 80:
            continue
        group = next((g for g in groups if abs(g[0][0]-line[0]) < 3 and abs(g[0][2]-line[2]) < 3), None)
        if group is None:
            groups.append([line])
        else:
            group.append(line)
    for group in groups:
        group.sort(key=lambda line: line[1])
        clusters = []
        for line in group:
            if not clusters or line[1] - clusters[-1][-1][1] > 110:
                clusters.append([line])
            else:
                clusters[-1].append(line)
        for cluster in clusters:
            ys = _edges([line[1] for line in cluster], 3)
            if len(ys) < 3 or ys[-1]-ys[0] < 18:
                continue
            x0, x1 = median(line[0] for line in cluster), median(line[2] for line in cluster)
            # Without a nearby table label, chart axes are not sufficient evidence.
            caption = next((c for c in captions if c['bbox'][0] < x1 and c['bbox'][2] > x0
                            and (0 <= ys[0]-c['bbox'][3] < 100 or 0 <= c['bbox'][1]-ys[-1] < 65)), None)
            if caption:
                yield [x0, ys[0], x1, ys[-1]], ys


def _recover_sparse(page, box, rules_y, horizontal, vertical, detected):
    # Extend a candidate to a nearly matching detector boundary; do not drop
    # the last row just because a bottom rule was missed in a drawing cluster.
    enclosing = next((t for t in detected if inside(box, t.bbox, .95)
                      and inside(t.bbox, box, .75)), None)
    if enclosing:
        box = [min(box[0], enclosing.bbox[0]), min(box[1], enclosing.bbox[1]),
               max(box[2], enclosing.bbox[2]), max(box[3], enclosing.bbox[3])]
        rules_y = _edges(rules_y + [box[1], box[3]])
    words = page.get_text('words', clip=fitz.Rect(box))
    if len(words) < 6:
        return None
    h = median(w[3]-w[1] for w in words)
    physical_x = _edges([x for x, y0, y1 in vertical if y0 <= box[1]+3 and y1 >= box[3]-3 and box[0]-2 <= x <= box[2]+2])
    known = next((t for t in detected if t.row_count >= 3 and t.col_count >= 3
                  and inside(t.bbox, box, .8) and inside(box, t.bbox, .8)), None)
    if known:
        xs = _edges([box[0], box[2]] + [c[i] for row in known.rows for c in row.cells if c is not None for i in (0, 2)])
    elif len(physical_x) >= 3:
        xs = _edges([box[0], box[2]] + physical_x)
    else:
        # Persistent gutters across all text rows, not each space between words.
        intervals = sorted((w[0], w[2]) for w in words)
        end, cuts = intervals[0][1], []
        for left, right in intervals[1:]:
            if left-end > max(5, h*.75):
                cuts.append((left+end)/2)
            end = max(end, right)
        xs = [box[0]] + cuts + [box[2]]
    if not 2 <= len(xs)-1 <= 25:
        return None
    # Reject inference when a native word straddles a proposed column boundary.
    if not known and any(w[0]+1 < x < w[2]-1 for w in words for x in xs[1:-1]):
        return None
    ys = list(rules_y)
    for top, bottom in zip(rules_y, rules_y[1:]):
        groups = _word_rows([w for w in words if top <= (w[1]+w[3])/2 < bottom])
        if known or bottom-top <= h*2.5 or len(groups) < 2:
            continue
        # A row starts where the label column is populated. Wrapped lines in
        # another cell stay in the same row. This is conservative, not universal.
        anchors = [row for row in groups if any(w[0] < xs[1]-1 for w in row)]
        if len(anchors) < 2:
            continue
        for prior, following in zip(anchors, anchors[1:]):
            ys.append((max(w[3] for w in prior)+min(w[1] for w in following))/2)
    ys = _edges(ys, .8)
    # Thin double rules can bound an empty band; it is not a data row.
    ys = [y for i,y in enumerate(ys) if i == 0 or i == len(ys)-1 or
          any(ys[i-1] <= (w[1]+w[3])/2 < y for w in words)]
    rows, cells = [], []
    for r, (y0, y1) in enumerate(zip(ys, ys[1:])):
        row = []
        for c, (x0, x1) in enumerate(zip(xs, xs[1:])):
            content = [w for w in words if x0 <= (w[0]+w[2])/2 < x1 and y0 <= (w[1]+w[3])/2 < y1]
            text = _cell_text(content)
            row.append(text)
            cells.append({'row': r, 'col': c, 'rowspan': 1, 'colspan': 1,
                          'bbox': [x0, y0, x1, y1], 'text': text})
        rows.append(row)
    populated = sum(sum(bool(cell) for cell in row) >= 2 for row in rows)
    if len(rows) < 2 or populated < 2:
        return None
    _style_cells(cells, horizontal, vertical)
    meta = _meta(xs, ys, cells, 'caption_rules_gutters')
    meta['structure_inferred'] = True
    return Block('table', page.number+1, box, rows=rows, meta=meta)


def native_tables(page, strategy, warnings):
    horizontal, vertical = _rules(page)
    detected = page.find_tables(strategy='lines_strict' if strategy == 'auto' else strategy)
    recovered = []
    if strategy == 'auto':
        for box, ys in _sparse_regions(page, horizontal):
            candidate = _recover_sparse(page, box, ys, horizontal, vertical, detected.tables)
            if candidate:
                recovered.append(candidate)
    for table in detected.tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        # A sparse-table recovery replaces fragments of that same table only.
        if any(inside(table.bbox, b.bbox, .75) or inside(b.bbox, table.bbox, .75) for b in recovered):
            continue
        block = _native_layout(table, horizontal, vertical)
        block.page = page.number+1
        block.meta['strategy'] = 'lines_strict' if strategy == 'auto' else strategy
        if len(block.rows) <= 3 and max((str(v or '').count('\n') for row in block.rows for v in row), default=0) >= 3:
            block.meta['structure_warning'] = '当前边框检测可能把多行/多列合成一个单元格，请对照原表。'
            warnings.append(f'第 {page.number+1} 页有表格结构尚未可靠恢复；已提供原表截图供校对。')
        recovered.append(block)
    return sorted(recovered, key=lambda b: (b.bbox[1], b.bbox[0]))
