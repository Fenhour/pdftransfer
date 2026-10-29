"""Repair ruled row bands only when model geometry and OCR give strong evidence.

No language-specific labels or expected column counts are assumed. Every OCR
line must be assigned exactly once. Unsupported geometry remains a reference
image rather than publishing a plausible but incomplete table.
"""
from collections import Counter
from html import escape
from html.parser import HTMLParser
import re

import numpy as np


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.cells = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'td','th'}:
            self.cells += 1

    def handle_data(self, data):
        self.text.append(data)


def _normalized(text):
    return re.sub(r'\s+', '', str(text))


def _rows(boxes, tolerance):
    rows = []
    for box in sorted(boxes, key=lambda b: (b[1], b[3])):
        match = next((row for row in rows if abs(row[0]-box[1]) <= tolerance
                      and abs(row[1]-box[3]) <= tolerance), None)
        if match is None:
            rows.append([box[1], box[3], [box]])
        else:
            match[2].append(box)
    return rows


def _groups(lines):
    # OCR physical lines in a column overlap horizontally across their rows.
    # Transitive interval components avoid treating large inter-column gaps
    # as ordinary spaces in one wide model cell.
    groups = []
    for line in sorted(lines, key=lambda v:v['box'][0]):
        if groups and line['box'][0] < max(v['box'][2] for v in groups[-1]):
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _rule_support(gray, y, radius):
    low, high = max(2, round(y-radius)), min(len(gray), round(y+radius)+1)
    return any(max(np.mean(gray[q]-gray[q-2] > 25), np.mean(gray[q-2]-gray[q] > 25)) >= .65
               for q in range(low, high))


def repair_row_bands(table, local_ocr, crop):
    """Return a measured alternative or None; preserve the model's raw output."""
    boxes = np.asarray(table.get('cell_box_list', []), dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) < 3 or not np.isfinite(boxes).all():
        return None
    texts = list(local_ocr.get('rec_texts', []))
    rec_boxes = np.asarray(local_ocr.get('rec_boxes', []), dtype=float)
    if rec_boxes.shape != (len(texts),4) or not len(texts) or not np.isfinite(rec_boxes).all():
        return None
    height,width = crop.shape[:2]
    if (np.any(boxes[:,2:] <= boxes[:,:2]) or
            np.any(rec_boxes[:,2:] <= rec_boxes[:,:2])):
        return None
    typical = float(np.median(rec_boxes[:,3]-rec_boxes[:,1]))
    tolerance = max(3., typical*.35)
    rows = _rows(boxes, tolerance)
    if len(rows) < 3 or any(row[1] <= row[0] for row in rows):
        return None
    if any(abs(a[1]-b[0]) > tolerance*2 for a,b in zip(rows,rows[1:])):
        return None  # no guessing spanning/missing rows
    boundaries = [max(0., rows[0][0])] + [(a[1]+b[0])/2 for a,b in zip(rows,rows[1:])] + [min(float(height),rows[-1][1])]
    gray = crop.astype(np.float32).mean(axis=2)
    if not all(_rule_support(gray,y,tolerance+3) for y in boundaries[1:-1]):
        return None
    row_lines = [[] for _ in rows]
    for index,(text,box) in enumerate(zip(texts,rec_boxes)):
        center = (box[1]+box[3])/2
        owners = [i for i,(a,b) in enumerate(zip(boundaries,boundaries[1:])) if a <= center < b]
        if len(owners) != 1:
            return None
        r=owners[0]
        if box[1] < boundaries[r]-tolerance or box[3] > boundaries[r+1]+tolerance:
            return None
        row_lines[r].append(dict(index=index,text=text,box=box))
    if any(not values for values in row_lines):
        return None
    groups = [_groups(values) for values in row_lines]
    columns = max(map(len,groups))
    split_rows = [values for values in groups if len(values)==columns]
    if not 2 <= columns <= 8 or len(split_rows) < 2:
        return None
    if any(len(values) not in (1,columns) for values in groups):
        return None
    # Require the same wide empty gutters in multiple separated row bands.
    gutters = []
    for c in range(columns-1):
        left = max(v['box'][2] for row in split_rows for v in row[c])
        right = min(v['box'][0] for row in split_rows for v in row[c+1])
        if right-left < typical*1.5:
            return None
        gutters.append((left+right)/2)
    xedges = [max(0.,float(boxes[:,0].min()))] + gutters + [min(float(width),float(boxes[:,2].max()))]
    if any(b-a < typical for a,b in zip(xedges,xedges[1:])):
        return None
    cells, html, assignment = [], ['<table>'], []
    for r,values in enumerate(groups):
        html.append('<tr>')
        # A single content group spans the row only if the model also observed
        # a full-width cell. Otherwise empty side columns remain unknown.
        if len(values)==1 and not any(b[2]-b[0] >= width*.85 for b in rows[r][2]):
            return None
        for c,group in enumerate(values):
            span=columns if len(values)==1 else 1
            group=sorted(group,key=lambda v:(v['box'][1],v['box'][0]))
            a,z=(0,columns) if span==columns else (c,c+1)
            if any(v['box'][0] < xedges[a]-tolerance or v['box'][2] > xedges[z]+tolerance for v in group):
                return None
            cells.append([xedges[a],boundaries[r],xedges[z],boundaries[r+1]])
            html.append(f'<td colspan="{span}">'+'<br>'.join(escape(v['text']) for v in group)+'</td>')
            assignment.append(dict(row=r,column=a,colspan=span,ocr_indexes=[v['index'] for v in group]))
        html.append('</tr>')
    html.append('</table>')
    used=[v for cell in assignment for v in cell['ocr_indexes']]
    if sorted(used) != list(range(len(texts))):
        return None
    return dict(table, pred_html=''.join(html),cell_box_list=cells,
                raw_model_pred_html=table.get('pred_html'),raw_model_cell_box_list=table.get('cell_box_list'),
                reconstruction_evidence=dict(method='ruled-row-bands-and-repeated-ocr-gutters',
                    coordinate_space='table-local-pixels',
                    row_boundaries=boundaries,column_boundaries=xedges,line_assignments=assignment))


def validated_table(table, local_ocr, crop):
    """Keep reliable model tables; repair supported bands; reject lost text."""
    parser=_HTMLText()
    parser.feed(str(table.get('pred_html','')))
    boxes=np.asarray(table.get('cell_box_list',[]),dtype=float)
    bad_geometry=(boxes.ndim != 2 or boxes.shape[1:] != (4,) or not len(boxes)
                  or parser.cells != len(boxes) or not np.isfinite(boxes).all())
    if not bad_geometry:
        bad_geometry=bool(np.any(boxes[:,2:] <= boxes[:,:2]))
    if not bad_geometry:
        for i,a in enumerate(boxes):
            for b in boxes[i+1:]:
                intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
                smallest=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
                if smallest <= 0 or intersection/smallest > .15:
                    bad_geometry=True
                    break
    # Counter containment catches duplicated matching as well as missing
    # symbols without requiring identical whitespace or line breaks.
    expected=Counter(_normalized(''.join(local_ocr.get('rec_texts',[]))))
    actual=Counter(_normalized(''.join(parser.text)))
    mismatch=expected != actual
    if bad_geometry or mismatch:
        alternative=repair_row_bands(table,local_ocr,crop)
        if alternative is not None:
            return alternative
        return None
    return table
