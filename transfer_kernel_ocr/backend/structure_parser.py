"""Adapt local PP-StructureV3 results to the existing editable scene contract.

Model boxes determine semantic ownership.  Geometry below only converts units,
matches already detected words, and preserves unclaimed pixels; it does not
reclassify a picture as body text or infer a second table topology.
"""
from __future__ import annotations

from .usage import span, tracked

from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import copy
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image
import pymupdf as fitz

from . import parser as legacy
from .options import selected_pages
from .progress import report
from .rebuild_text import organize_text
from .style_templates import observe_styles


_TEXT_LABELS = {'text', 'title', 'doc_title', 'paragraph_title', 'section_title',
                'figure_title', 'table_title', 'chart_title', 'caption', 'reference',
                'reference_content', 'header', 'footer', 'page_number', 'footnote',
                'list', 'list_item', 'abstract', 'algorithm', 'content', 'aside_text',
                'vision_footnote', 'number', 'formula_number'}
_HEADINGS = {'title', 'doc_title', 'paragraph_title', 'section_title'}
_rebuild_table_queue = ContextVar('rebuild_table_queue', default=None)


def recognize_page(rgb):
    # The reading extractor owns the single persistent, local model transport.
    # Lazy import keeps native PDF imports free of Docker/model startup costs.
    from .recognition import shared_structure
    return shared_structure().recognize_page(rgb)


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


def _blocks(raw, w, h):
    blocks = []
    for n, value in enumerate(raw.get('parsing_res_list') or []):
        b = _box(value.get('block_bbox', value.get('bbox')), w, h)
        if b:
            blocks.append(dict(id=f'v3-block-{n}', bbox=b,
                               label=str(value.get('block_label', 'unknown')).lower(),
                               text=str(value.get('block_content') or ''),
                               order=value.get('block_order') if isinstance(value.get('block_order'), (int, float)) else n,
                               sourceId=value.get('block_id', n), source=value.get('source', 'v3')))
    if not blocks:
        for n, value in enumerate((raw.get('layout_det_res') or {}).get('boxes') or []):
            b = _box(value.get('coordinate', value.get('bbox')), w, h)
            if b:
                blocks.append(dict(id=f'v3-block-{n}', bbox=b, label=str(value.get('label', 'unknown')).lower(),
                                   text='', order=n, sourceId=n))
    if not blocks:
        raise ValueError('PP-StructureV3 未返回有效版面区域；保留源文件，请检查模型结果后重试。')
    return blocks


def _ocr_records(result, w, h, sx, sy, *, prefix='v3-ocr'):
    texts = result.get('rec_texts') or []
    boxes = result.get('rec_boxes') or result.get('rec_polys') or result.get('dt_polys') or []
    scores = result.get('rec_scores') or []
    records = []
    for n, (text, coords) in enumerate(zip(texts, boxes)):
        b = _box(coords, w, h)
        if not b or not str(text).strip():
            continue
        try:
            score = float(scores[n]) if n < len(scores) else 0.
            score = max(0., min(1., score)) if math.isfinite(score) else 0.
        except (TypeError, ValueError):
            score = 0.
        records.append(dict(id=f'{prefix}-{n}', text=str(text), pixelBox=b,
                            bbox=_to_points(b, sx, sy), confidence=score, source='ocr'))
    return records


def _owner(record, blocks):
    area = max(.01, _area(record['pixelBox']))
    choices = []
    for block in blocks:
        fraction = _intersection(record['pixelBox'], block['bbox'])/area
        if fraction >= .5:
            # Maximal word coverage; a tighter region resolves equal coverage.
            choices.append((fraction, -_area(block['bbox']), block['id']))
    return max(choices)[2] if choices else None


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


def _save_crop(rgb, pixel_box, output_dir, document_id, relative):
    x0, y0, x1, y1 = [int(v) for v in (math.floor(pixel_box[0]), math.floor(pixel_box[1]),
                                      math.ceil(pixel_box[2]), math.ceil(pixel_box[3]))]
    return legacy._save_png(rgb[y0:y1, x0:x1], output_dir, document_id, relative)


def _border_observation(table, rgb, sx, sy):
    """Observe rules at model cell edges; never invent a wired-table style."""
    h, w = rgb.shape[:2]
    paper = legacy._local_paper(rgb)
    dark = np.max(np.abs(rgb.astype(np.int16)-paper.astype(np.int16)), axis=2) > 45
    visible = False
    thicknesses = []
    for cell in table['cells']:
        b = cell['bbox']
        x0, y0, x1, y1 = b[0]/sx, b[1]/sy, b[2]/sx, b[3]/sy
        left, right = max(0,round(x0)+2), min(w,round(x1)-2)
        top, bottom = max(0,round(y0)+2), min(h,round(y1)-2)
        borders = {}
        for side,y in (('top',y0),('bottom',y1)):
            row = max(0,min(h-1,round(y)))
            strip = dark[max(0,row-6):min(h,row+7), left:right]
            support = np.mean(strip,axis=1) >= .7 if strip.size else []
            borders[side] = bool(np.any(support))
            if borders[side]:
                thicknesses.append(int(np.count_nonzero(support)))
        for side,x in (('left',x0),('right',x1)):
            col = max(0,min(w-1,round(x)))
            strip = dark[top:bottom,max(0,col-6):min(w,col+7)]
            support = np.mean(strip,axis=0) >= .7 if strip.size else []
            borders[side] = bool(np.any(support))
            if borders[side]:
                thicknesses.append(int(np.count_nonzero(support)))
        cell['borders'] = borders
        visible |= any(borders.values())
    pixels = float(np.median(thicknesses)) if thicknesses else 0.
    table['observedStyle'].update(borderWidth=pixels*math.sqrt(sx*sy), borderColor='#000000',
                                  borderVisible=visible, sourceBorderWidthPixels=pixels,
                                  borderEvidence='sampled-model-cell-edges')
    return visible


def _border_elements(table):
    """Sampled cell edges, including partial rules and merged-cell perimeters."""
    seen = set()
    elements = []
    stroke = table['observedStyle']['borderWidth']
    for cell in table['cells']:
        x0,y0,x1,y1 = cell['bbox']
        edges = dict(top=(x0,y0,x1,y0), bottom=(x0,y1,x1,y1),
                     left=(x0,y0,x0,y1), right=(x1,y0,x1,y1))
        for side, shown in cell.get('borders',{}).items():
            if not shown:
                continue
            a,b,c,d = edges[side]
            key = tuple(round(v,3) for v in (a,b,c,d))
            if key in seen:
                continue
            seen.add(key)
            cx,cy = (a+c)/2,(b+d)/2
            length = abs(c-a)+abs(d-b)
            native = table.get('geometrySource') == 'native'
            e = legacy._base('shape','表格可见边线',[cx-length/2,cy-stroke/2,cx+length/2,cy+stroke/2],-1500,'native' if native else 'v3')
            e.update(shape='line', rotation=90 if a == c else 0, stroke='#000000', strokeWidth=stroke,fill='none',
                     rebuildVector=True,tableBorder=True,tableId=table['id'],cellId=cell['id'],
                     contentType='table-border',reviewRequired=not native)
            elements.append(e)
    return elements


def _mask_box(mask, box, value=255, pad=0):
    h, w = mask.shape[:2]
    x0, y0 = max(0, math.floor(box[0])-pad), max(0, math.floor(box[1])-pad)
    x1, y1 = min(w, math.ceil(box[2])+pad), min(h, math.ceil(box[3])+pad)
    mask[y0:y1, x0:x1] = value


def _line(record, rgb, output_dir, document_id, index, order, block=None):
    b = record['bbox']
    e = dict(record['nativeElement']) if record.get('nativeElement') else legacy._base('text', record['text'][:30], b, order, record.get('source', 'ocr'))
    e['id'] = record['id']
    crop = rgb[max(0, math.floor(record['pixelBox'][1])):math.ceil(record['pixelBox'][3]),
               max(0, math.floor(record['pixelBox'][0])):math.ceil(record['pixelBox'][2])]
    if not record.get('nativeElement'):
        e.update(text=record['text'], fontFamily='sans-serif', fontSize=max(4., min(100., (b[3]-b[1])*.78)),
                 fontWeight='400', fontStyle='normal', color=legacy._estimate_ink_color(crop),
                 align='left', lineHeight=1., confidence=record.get('confidence', 0.), fontEstimated=True)
    e.update(reviewRequired=record.get('source') != 'native', textRole='independent',
             readingOrder=order, textOrigin='native' if record.get('nativeElement') else 'ocr-line')
    if block:
        e.update(layoutBlockId=block['id'], layoutLabel=block['label'], modelReadingOrder=block['order'])
    src = _save_crop(rgb, record['pixelBox'], output_dir, document_id, f'text/page-{index+1:03d}-{e["id"]}.png')
    e.update(appearanceSrc=src, originalAppearanceSrc=src, appearanceMode='original-pixels')
    record['ownerId'] = e['id']
    return e


def _select_native(records):
    """Use visible native text for overlapping OCR, preserving both evidence."""
    native = [r for r in records if r.get('nativeElement') and not r.get('supersededBy')]
    selected = []
    for r in records:
        if r.get('supersededBy'):
            continue
        if r.get('nativeElement'):
            selected.append(r)
            continue
        # PDF producers may paint each glyph separately. Compare the union of
        # native coverage with the OCR line, rather than requiring one glyph
        # to cover most of an entire line. Count overlapping native boxes once.
        b = r['bbox']
        overlaps = [(n, [max(b[0], n['bbox'][0]), max(b[1], n['bbox'][1]),
                         min(b[2], n['bbox'][2]), min(b[3], n['bbox'][3])])
                    for n in native if _intersection(b, n['bbox']) > 0]
        xs = sorted({v for _, box in overlaps for v in (box[0], box[2])})
        covered = 0.
        for left, right in zip(xs, xs[1:]):
            intervals = sorted((box[1], box[3]) for _, box in overlaps
                               if box[0] < right and box[2] > left)
            top = bottom = 0.
            height = 0.
            for lo, hi in intervals:
                if lo > bottom:
                    height += bottom-top
                    top, bottom = lo, hi
                else:
                    bottom = max(bottom, hi)
            covered += (right-left)*(height+bottom-top)
        match = overlaps[0][0] if overlaps and covered/max(.01, _area(b)) > .6 else None
        if match:
            r['supersededBy'] = match['id']
        else:
            selected.append(r)
    return selected


@tracked('document.v3_compose')
def compose_page(raw, rgb, width, height, output_dir, document_id, index=0,
                 source_rotation=0, source_name='', native=()):
    """Public pure adaptation boundary; tests use manually specified model truth."""
    h, w = rgb.shape[:2]
    sx, sy = width/w, height/h
    blocks = _blocks(raw, w, h)
    records = _ocr_records(raw.get('overall_ocr_res') or {}, w, h, sx, sy)
    for n, candidate in enumerate(native):
        e = candidate['element']
        if e.get('type') != 'text':
            continue
        b = candidate['bbox']
        records.append(dict(id=f'v3-native-{n}', text=e['text'], bbox=list(b),
                            pixelBox=[b[0]/sx, b[1]/sy, b[2]/sx, b[3]/sy],
                            confidence=1., source='native', nativeElement=e))
    # Ownership can assign a glyph and its enclosing OCR line to different
    # semantic blocks. Resolve source duplication before that partition.
    _select_native(records)
    by_block = defaultdict(list)
    for r in records:
        r['blockId'] = _owner(r, blocks)
        by_block[r['blockId']].append(r)
    warnings = []
    claimed = np.zeros((h, w), np.uint8)
    elements, tables = [], []
    raw_tables = list(raw.get('table_res_list') or [])
    used_tables = set()
    for block in sorted(blocks, key=lambda b: (b['order'], b['bbox'][1], b['bbox'][0])):
        owned = by_block[block['id']]
        chosen = _select_native(owned)
        label, b = block['label'], block['bbox']
        if label == 'table':
            candidates = []
            block_signature = _html_signature(block['text'])
            for n, result in enumerate(raw_tables):
                if n in used_tables:
                    continue
                cb = [_box(v, w, h) for v in result.get('cell_box_list') or []]
                cb = [v for v in cb if v]
                if cb:
                    bounds = [min(v[0] for v in cb), min(v[1] for v in cb),
                              max(v[2] for v in cb), max(v[3] for v in cb)]
                    coverage = _intersection(b,bounds)/max(.01,_area(bounds))
                    block_coverage = _intersection(b,bounds)/max(.01,_area(b))
                    if coverage >= .65 and block_coverage >= .15:
                        candidates.append((coverage, -n, n, result, 'spatial'))
                elif block_signature and block_signature == _html_signature(result.get('pred_html') or ''):
                    # Without global cell coordinates, only explicit matching
                    # HTML evidence can associate a result with a layout block.
                    candidates.append((.5, -n, n, result, 'matching-html-topology-and-content'))
            result = {}
            association = 'layout-block-html-only'
            if candidates:
                _, _, n, result, association = max(candidates, key=lambda v: v[:2])
                used_tables.add(n)
            table = _table_structure(result, block, sx, sy, w, h)
            if (table and result.get('_source') == 'native' and
                    table['geometryStatus'] != 'native-cell-boxes-aligned'):
                table = None  # Native cells must never be replaced by an estimated grid.
            if table:
                table['resultAssociation'] = association
                _border_observation(table, rgb, sx, sy)
                if table['geometryStatus'] not in {'model-cell-boxes-aligned','native-cell-boxes-aligned'}:
                    warnings.append('表格 HTML 拓扑保留，但单元格坐标无效或不完整，布局位置为估计，需对照原图校对。')
                table['referenceBbox'] = _to_points(b, sx, sy)
                table['referenceSrc'] = _save_crop(rgb, b, output_dir, document_id,
                                                  f'images/page-{index+1:03d}-{table["id"]}-reference.png')
                local = _ocr_records(result.get('table_ocr_pred') or {}, w, h, sx, sy, prefix=f'{block["id"]}-ocr')
                if local:
                    local = _select_native(local+[r for r in chosen if r.get('nativeElement')])
                    known_ids = {r['id'] for r in records}
                    records.extend(r for r in local if r['id'] not in known_ids)
                else:
                    local = chosen
                per_cell = defaultdict(list)
                for r in local:
                    matches = [(_intersection(r['bbox'], c['bbox'])/max(.01, _area(r['bbox'])), c['id']) for c in table['cells']]
                    score, cell_id = max(matches, default=(0., ''))
                    if score >= .35:
                        per_cell[cell_id].append(r)
                    else:
                        r['ownerId'] = table['id']
                        table.setdefault('retainedText', []).append(dict(text=r['text'], bbox=r['bbox'], reason='unmatched-cell'))
                typical = float(np.median([max(4., (r['bbox'][3]-r['bbox'][1])*.78) for r in local])) if local else 12.
                for cell in table['cells']:
                    lines = sorted(per_cell[cell['id']], key=lambda r: (r['bbox'][1], r['bbox'][0]))
                    content = cell['htmlText']
                    # HTML is the structure model's final matched content. A
                    # mismatch is rendered once, never layered over OCR words.
                    if content and _plain(content) != _plain(''.join(r['text'] for r in lines)):
                        cb = cell['bbox']
                        inset = min(2., (cb[2]-cb[0])*.05, (cb[3]-cb[1])*.05)
                        e = legacy._base('text', content[:30], [cb[0]+inset, cb[1]+inset, cb[2]-inset, cb[3]-inset], len(elements), 'ocr')
                        e.update(text=content, fontFamily='sans-serif', fontSize=max(4., min(100., typical)),
                                 fontWeight='700' if cell.get('header') else '400', fontStyle='normal',
                                 color='#000000', align='left', lineHeight=1.1, textOrigin='table-html',
                                 fontEstimated=True, reviewRequired=True, textRole='independent')
                        output = [e]
                        for r in lines:
                            r['ownerId'] = e['id']
                    else:
                        output = [_line(r, rgb, output_dir, document_id, index, len(elements)+n, block) for n, r in enumerate(lines)]
                    for e in output:
                        e.update(tableId=table['id'], cellId=cell['id'], layoutBlockId=block['id'], layoutLabel='table',
                                 tableCell={k:cell[k] for k in ('row', 'column', 'rowspan', 'colspan')})
                        cell['textIds'].append(e['id'])
                        elements.append(e)
                elements.extend(_border_elements(table))
                for r in owned:
                    r.setdefault('ownerId', table['id'])
                tables.append(table)
                # Exclude a small source-pixel halo from the residual: model
                # boxes are not exact stroke outlines.  It must not reappear
                # as a second table border behind the reconstructed rules.
                border_pixels = table['observedStyle']['sourceBorderWidthPixels']
                halo = max(2,min(6,math.ceil(border_pixels+3)))
                _mask_box(claimed, b, pad=halo)
                continue
            warnings.append('表格区域未得到有效 HTML 拓扑，保留为原图参考与转录，未冒充结构化表格。')
        if label in _TEXT_LABELS:
            for n, r in enumerate(chosen):
                elements.append(_line(r, rgb, output_dir, document_id, index, len(elements)+n, block))
                margin = max(2,min(8,math.ceil((r['pixelBox'][3]-r['pixelBox'][1])*.12)))
                _mask_box(claimed, r['pixelBox'], pad=margin)
            if not owned and block['text'].strip():
                bb = _to_points(b, sx, sy)
                text = block['text']
                e = legacy._base('text', text[:30], bb, len(elements), 'ocr')
                e.update(text=text, fontFamily='sans-serif', fontSize=max(4., min(18., (bb[3]-bb[1])*.78)),
                         fontWeight='700' if label in _HEADINGS else '400', fontStyle='normal', color='#000000',
                         align='left', lineHeight=1., reviewRequired=True, fontEstimated=True,
                         layoutBlockId=block['id'], layoutLabel=label, modelReadingOrder=block['order'],
                         textOrigin='layout-block', textRole='independent')
                elements.append(e)
                _mask_box(claimed, b, pad=2)
            continue
        # Formula/chart/unknown regions retain their pixels when their optional
        # recognition branch is disabled; OCR here is a transcript only.
        e = legacy._base('image', f'{label} 原图区域', _to_points(b, sx, sy), -1000+len(elements),
                         'native' if block.get('source') == 'native' else 'v3')
        e['id'] = block['id']
        e.update(src=_save_crop(rgb, b, output_dir, document_id, f'images/page-{index+1:03d}-{block["id"]}.png'),
                 bbox=_to_points(b, sx, sy), layoutLabel=label, contentType=label, templateId='image.original.v1',
                 scanComposite=True, reviewRequired=True,
                 boundary=dict(method='native-pdf-geometry' if block.get('source') == 'native' else 'pp-doclayout-plus-l',
                               coordinateSystem='points-top-left'),
                 retainedText=[dict(textId=r['id'], text=r['text'], bbox=r['bbox'], confidence=r['confidence']) for r in chosen])
        elements.append(e)
        for r in owned:
            r['ownerId'] = e['id']
        _mask_box(claimed, b)
    # Unmatched OCR is genuine model evidence; keep it editable rather than
    # dropping a line because the layout detector missed its enclosing block.
    for r in _select_native(by_block[None]):
        elements.append(_line(r, rgb, output_dir, document_id, index, len(elements)))
        _mask_box(claimed, r['pixelBox'], pad=2)
    # Resolve overlap between model picture boxes and independent visible text.
    text_mask = np.zeros((h, w), np.uint8)
    visible_text_ids = {e['id'] for e in elements if e['type'] == 'text'}
    for r in records:
        if r.get('ownerId') in visible_text_ids:
            _mask_box(text_mask, r['pixelBox'], pad=2)
    for e in elements:
        if e['type'] == 'text' and e.get('textOrigin') in {'layout-block', 'table-html'}:
            _mask_box(text_mask, [e['x']/sx,e['y']/sy,(e['x']+e['width'])/sx,(e['y']+e['height'])/sy], pad=2)
    native_table_masks = [table for table in tables if table.get('geometrySource') == 'native']
    for table in native_table_masks:
        _mask_box(text_mask,[table['bbox'][0]/sx,table['bbox'][1]/sy,table['bbox'][2]/sx,table['bbox'][3]/sy])
    for e in elements:
        if e['type'] != 'image' or e.get('layoutLabel') == 'unclassified':
            continue
        block = next(b for b in blocks if b['id'] == e['id'])
        b = block['bbox']
        x0, y0, x1, y1 = math.floor(b[0]), math.floor(b[1]), math.ceil(b[2]), math.ceil(b[3])
        holes = text_mask[y0:y1, x0:x1]
        if np.any(holes):
            rgba = np.dstack((rgb[y0:y1, x0:x1], 255-holes))
            e['src'] = legacy._save_png(rgba, output_dir, document_id, f'images/page-{index+1:03d}-{e["id"]}.png')
            e['boundary']['clip'] = ('independent-text-and-table-exclusion' if native_table_masks
                                     else 'independent-text-exclusion')
    # Unknown strokes stay visible once, with a transparent alpha rather than
    # an unchanged scan behind the reconstructed words/table cells.
    paper = legacy._local_paper(rgb)
    ink = np.max(np.abs(rgb.astype(np.int16)-paper.astype(np.int16)), axis=2) > 12
    residual = ink & (claimed == 0)
    ys, xs = np.nonzero(residual)
    if len(xs):
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
        alpha = np.where(residual[y0:y1, x0:x1], 255, 0).astype(np.uint8)
        rgba = np.dstack((rgb[y0:y1, x0:x1], alpha))
        e = legacy._base('image', '未分类原始内容（需校对）', _to_points([x0,y0,x1,y1], sx, sy), -3000, 'residual')
        e.update(src=legacy._save_png(rgba, output_dir, document_id, f'images/page-{index+1:03d}-unclassified.png'),
                 layoutLabel='unclassified', contentType='unclassified', templateId='image.original.v1',
                 reviewRequired=True, retainedText=[], boundary=dict(method='unclaimed-pixels-only', excludes='recognized-text-and-tables'))
        elements.append(e)
        warnings.append('未分类笔画以透明原图区域保留；可对照原页核查漏检，不视为识别成功。')
    for r in records:
        if r.get('supersededBy'):
            r['ownerId'] = next((n.get('ownerId') for n in records if n['id'] == r['supersededBy']), None)
    page = dict(id=legacy._id('page'), index=index, width=width, height=height, rotation=0,
                sourceRotation=source_rotation, sourceName=source_name,
                sourceSrc=legacy._save_png(rgb, output_dir, document_id, f'pages/page-{index+1:03d}.png'),
                elements=elements, tables=tables, warnings=warnings, renderScale=w/width,
                paperColor='#'+''.join(f'{int(v):02x}' for v in paper),
                appearanceStrategy='blank-page-typed-reconstruction', backgroundReconstruction='none',
                recognitionBackend='v3', contentModelVersion=2,
                modelDispatch=dict(raw.get('_service') or {}),
                layoutBlocks=[dict(id=b['id'], label=b['label'], bbox=_to_points(b['bbox'], sx, sy), order=b['order'],
                                   source=b.get('source','v3')) for b in blocks],
                recognition=[{k:v for k,v in r.items() if k not in {'nativeElement', 'pixelBox'}} for r in records])
    observe_styles(page)
    for e in elements:
        if e['type'] == 'text':
            heading = e.get('layoutLabel') in _HEADINGS
            e.update(contentType='heading' if heading else 'text', templateId='text.heading.v1' if heading else 'text.body.v1',
                     semanticEvidence='native-pdf-reading' if e.get('source') == 'native' else
                     'pp-doclayout-plus-l' if e.get('layoutBlockId') else 'ocr-unmatched')
    organize_text(page)
    if (raw.get('_service') or {}).get('modelCalls') == 0:
        warnings.append('原生 PDF 文字及表格直接提取，未调用 V3；原图与坐标已保留，复杂表格和导出字体替代仍需校对。')
    else:
        warnings.append('V3 版面、OCR 与表格结构为模型预测；原图与坐标已保留，图片文字字体为近似样式，需校对。')
    return page


def _native_fast_path(page, candidates):
    text = ''.join(c['element'].get('text', '') for c in candidates)
    # A one-word native title is still editable text.  Visible native glyph
    # evidence and image coverage matter; an arbitrary character threshold
    # would needlessly send short originals through OCR.
    if not text.strip() or '�' in text or any(ord(c)<32 and c not in '\n\r\t' for c in text):
        return False
    try:
        image_area = sum(_area(list(info['bbox'])) for info in page.get_image_info())
        return image_area <= page.rect.width*page.rect.height*.12
    except Exception:
        return False


def _unrecognized_page(rgb, width, height, output_dir, document_id, index,
                       source_rotation, source_name):
    """OCR disabled means preserve one intact source, never guess semantics."""
    src = legacy._save_png(rgb, output_dir, document_id, f'pages/page-{index+1:03d}.png')
    e = legacy._base('image', '未识别原页（OCR 已关闭）', [0,0,width,height], 0, 'raster')
    e.update(src=src, scanComposite=True, reviewRequired=True, contentType='unrecognized',
             layoutLabel='unrecognized', retainedText=[])
    return dict(id=legacy._id('page'), index=index, width=width, height=height, rotation=0,
                sourceRotation=source_rotation, sourceName=source_name, sourceSrc=src,
                elements=[e], tables=[], recognition=[], layoutBlocks=[],
                warnings=['OCR 已关闭：本页完整保留为图像，没有推断内容类型或生成可编辑文字。'],
                renderScale=rgb.shape[1]/width, recognitionBackend='disabled',
                appearanceStrategy='unrecognized-source-image', backgroundReconstruction='none')


def _reading_engine():
    """The same native extraction implementation used by features 1 and 2."""
    root = str(Path(__file__).resolve().parents[2]/'transfer_kenel')
    if root not in sys.path:
        sys.path.insert(0, root)
    from pdf_extractor import engine
    return engine


@contextmanager
def native_table_session(path, selection):
    """Reuse one bounded table worker across page-bounded reconstruction."""
    engine = _reading_engine()
    from pdf_extractor.model import Options
    with engine._TableQueue(Path(path),selection,Options(ocr='off')) as queue:
        binding = _rebuild_table_queue.set((Path(path).resolve(),queue))
        try:
            yield
        finally:
            _rebuild_table_queue.reset(binding)


def _offset_v3_result(raw, dx, dy):
    """Map a crop's pixel coordinates back to the full rendered page.

    Only coordinate-bearing schema fields are transformed. Text, structure
    token sequences, probabilities and the service profile remain unchanged.
    """
    rectangles = {'block_bbox', 'bbox', 'coordinate', 'table_bbox'}
    collections = {'rec_boxes', 'rec_polys', 'dt_polys', 'cell_box_list', 'cell_bbox'}

    def coordinate(value):
        if not isinstance(value, (list, tuple)):
            return value
        if len(value) in {4, 8} and all(isinstance(v, (int, float)) for v in value):
            return [v+(dx if n % 2 == 0 else dy) for n, v in enumerate(value)]
        if value and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in value):
            return [[v[0]+dx, v[1]+dy] for v in value]
        return copy.deepcopy(value)

    def transform(value):
        if isinstance(value, dict):
            return {key: (coordinate(item) if key in rectangles else
                          [coordinate(v) for v in item] if key in collections and isinstance(item, (list, tuple)) else
                          transform(item) if key != '_service' else copy.deepcopy(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [transform(v) for v in value]
        return value

    return transform(raw)


def _dispatch_regions(regions, *, strategy):
    profiles = []
    for index, region in enumerate(regions):
        for profile in region.get('modelProfile') or region.get('modelProfiles') or []:
            profiles.append(dict(profile, regionIndex=index))
    return dict(strategy=strategy, modelCalls=len(regions), regions=regions, modelProfiles=profiles,
                cacheHits=sum(bool(r.get('cacheHit')) for r in regions),
                inferenceRegions=sum(not bool(r.get('cacheHit')) for r in regions),
                cacheHit=bool(regions) and all(r.get('cacheHit') for r in regions))


def _image_regions(source, rgb):
    """Use actual embedded image bounds; merge overlap before inference."""
    h, w = rgb.shape[:2]
    sx, sy = w/source.rect.width, h/source.rect.height
    regions = []
    for image in source.get_image_info():
        rect = fitz.Rect(image['bbox'])*source.rotation_matrix
        box = _box([rect.x0*sx, rect.y0*sy, rect.x1*sx, rect.y1*sy], w, h)
        if not box:
            continue
        # Integer crop bounds are the exact origin used for all later offsets.
        box = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
        changed = True
        while changed:
            changed = False
            for other in list(regions):
                if _intersection(box, other) > 0:
                    regions.remove(other)
                    box = [min(box[0],other[0]),min(box[1],other[1]),max(box[2],other[2]),max(box[3],other[3])]
                    changed = True
        regions.append(box)
    return sorted(regions, key=lambda b: (b[1], b[0]))


def _native_table_result(table, source, sx, sy):
    """Adapt native geometry, rejecting missing / inconsistent cell slots."""
    cells = table.meta.get('grid_cells') or []
    if not cells or source.rotation:
        return None
    try:
        ordered = sorted(cells, key=lambda c: (c['row'], c['col']))
        rows = defaultdict(list)
        boxes = []
        occupied = set()
        for cell in ordered:
            row, column = int(cell['row']), int(cell['col'])
            rowspan, colspan = int(cell['rowspan']), int(cell['colspan'])
            if row < 0 or column < 0 or min(rowspan,colspan) < 1:
                return None
            slots = {(r,c) for r in range(row,row+rowspan) for c in range(column,column+colspan)}
            if slots & occupied or row+rowspan > 300 or column+colspan > 100:
                return None
            occupied |= slots
            box = _box([cell['bbox'][0]/sx,cell['bbox'][1]/sy,cell['bbox'][2]/sx,cell['bbox'][3]/sy],
                       source.rect.width/sx, source.rect.height/sy)
            if not box:
                return None
            boxes.append(box)
            rows[row].append(f'<td rowspan="{rowspan}" colspan="{colspan}">{html.escape(str(cell.get("text") or ""))}</td>')
        nr = max(r for r,c in occupied)+1
        nc = max(c for r,c in occupied)+1
        expected_rows = len(table.meta.get('row_heights') or table.rows)
        expected_columns = len(table.meta.get('column_widths') or []) or max(map(len,table.rows),default=0)
        if (len(occupied) != nr*nc or expected_rows and nr != expected_rows or
                expected_columns and nc != expected_columns):
            return None
        bounds = [table.bbox[0]/sx,table.bbox[1]/sy,table.bbox[2]/sx,table.bbox[3]/sy]
        for n,box in enumerate(boxes):
            if _intersection(box,bounds)/max(.01,_area(box)) < .95:
                return None
            if any(_intersection(box,other)/max(.01,min(_area(box),_area(other))) > .15
                   for other in boxes[:n]):
                return None
        markup = '<table>'+''.join('<tr>'+''.join(rows.get(r,[]))+'</tr>' for r in range(nr))+'</table>'
        return dict(pred_html=markup,cell_box_list=boxes,_source='native')
    except (KeyError, ValueError, TypeError, OverflowError):
        return None


@tracked('rebuild.native_image_split')
def _rebuild_pdf_page(source, rgb, output_dir, document_id, index, source_name, native,
                      reading_blocks, native_tables, native_warnings):
    """Native PDF text/table evidence plus V3 only for its bitmap regions."""
    height, width = rgb.shape[:2]
    sx, sy = source.rect.width/width, source.rect.height/height
    usable = [c for c in native if c['element'].get('text','').strip()
              and '�' not in c['element']['text']
              and not any(ord(v)<32 and v not in '\n\r\t' for v in c['element']['text'])]
    if not usable:
        raw = recognize_page(rgb)
        dispatch = dict(raw.get('_service') or {})
        raw = dict(raw, _service=_dispatch_regions([
            dict(dispatch,bbox=[0.,0.,float(source.rect.width),float(source.rect.height)],pixelBbox=[0,0,width,height])],
            strategy='full-page-no-native-text'))
        return compose_page(raw,rgb,float(source.rect.width),float(source.rect.height),output_dir,
                            document_id,index,source.rotation,source_name,())

    def pixels(box):
        rect = fitz.Rect(box)*source.rotation_matrix
        return [rect.x0/sx,rect.y0/sy,rect.x1/sx,rect.y1/sy]

    raw = dict(parsing_res_list=[],overall_ocr_res=dict(rec_texts=[],rec_boxes=[],rec_scores=[]),table_res_list=[])
    warnings = list(native_warnings)
    for block in reading_blocks:
        if block.kind not in {'text','heading','page_number','header_footer'}:
            continue
        label = {'heading':'paragraph_title','page_number':'page_number','header_footer':'header'}.get(block.kind,'text')
        raw['parsing_res_list'].append(dict(block_label=label,block_bbox=pixels(block.bbox),
                                             block_content='',source='native'))
    for table in native_tables:
        result = _native_table_result(table,source,sx,sy)
        raw['parsing_res_list'].append(dict(block_label='table' if result else 'native_table_unresolved',
                                             block_bbox=pixels(table.bbox),block_content=result['pred_html'] if result else '',
                                             source='native'))
        if result:
            raw['table_res_list'].append(result)
        else:
            warnings.append('原生表格缺少完整可靠单元格几何或含旋转，已保留原表图片和文字转录，未冒充恢复成功。')
    # The reading model can omit isolated painting runs; retain their native
    # ownership rather than fabricate OCR text or discard visible characters.
    if not raw['parsing_res_list']:
        raw['parsing_res_list']=[dict(block_label='text',block_bbox=[0,0,width,height],block_content='',source='native')]
    dispatches = []
    native_table_boxes=[b['block_bbox'] for b in raw['parsing_res_list'] if b['block_label']=='table']
    for region_index, region in enumerate(_image_regions(source,rgb)):
        x0,y0,x1,y1 = region
        with span('rebuild.image_region',page=source.number+1):
            prediction = recognize_page(rgb[y0:y1,x0:x1])
        dispatches.append(dict(prediction.get('_service') or {},bbox=_to_points(region,sx,sy),pixelBbox=region))
        mapped = _offset_v3_result(prediction,x0,y0)
        blocks = mapped.get('parsing_res_list') or []
        if not blocks:
            blocks = [dict(block_label=b.get('label','unknown'),block_bbox=b.get('coordinate',b.get('bbox')),block_content='')
                      for b in (mapped.get('layout_det_res') or {}).get('boxes') or []]
        if not blocks:
            # Preserve failed layout as source pixels; no fake successful text.
            blocks = [dict(block_label='unrecognized_image',block_bbox=region,block_content='')]
            warnings.append('图片区域未返回有效版面，已保留原图，未将其包装为已识别内容。')
        for block in blocks:
            b = _box(block.get('block_bbox'),width,height)
            if (str(block.get('block_label','')).lower() == 'table' and b and
                    any(_intersection(b,other)/max(.01,_area(b)) > .7 and
                        _intersection(b,other)/max(.01,_area(other)) > .7 for other in native_table_boxes)):
                continue  # Reliable native table owns this region, including model OCR evidence.
            raw['parsing_res_list'].append(block)
        ocr = mapped.get('overall_ocr_res') or {}
        texts = ocr.get('rec_texts') or []
        boxes = ocr.get('rec_boxes') or ocr.get('rec_polys') or ocr.get('dt_polys') or []
        scores = ocr.get('rec_scores') or []
        for n,(text,box) in enumerate(zip(texts,boxes)):
            raw['overall_ocr_res']['rec_texts'].append(text)
            raw['overall_ocr_res']['rec_boxes'].append(box)
            raw['overall_ocr_res']['rec_scores'].append(scores[n] if n<len(scores) else 0.)
        raw['table_res_list'].extend(mapped.get('table_res_list') or [])
    # Every synthetic / model block gets one global order and identifier.
    # Never reuse crop-local block order values as if they were full-page order.
    raw['parsing_res_list']=[b for b in raw['parsing_res_list'] if _box(b.get('block_bbox'),width,height)]
    raw['parsing_res_list'].sort(key=lambda b:(b['block_bbox'][1],b['block_bbox'][0]))
    for order,block in enumerate(raw['parsing_res_list']):
        block.update(block_order=order,block_id=order)
    raw['_service'] = _dispatch_regions(dispatches,strategy='native-image-regions')
    page = compose_page(raw,rgb,float(source.rect.width),float(source.rect.height),output_dir,
                        document_id,index,source.rotation,source_name,usable)
    page['recognitionBackend'] = 'native+v3' if dispatches else 'native'
    page['warnings'].extend(warnings)
    return page


@tracked('document.v3_parse')
def parse_document(paths: list[Path], output_dir: Path, document_id: str, name: str,
                   ocr: bool=True, pages: str='', mode: str='editable') -> dict:
    """Keep UI/file contract while sharing the local V3 recognition service."""
    if mode not in {'editable', 'text_pdf', 'rebuild'}:
        raise ValueError('不支持的转换模式')
    if not ocr and mode in {'text_pdf','rebuild'}:
        raise ValueError('扫描 PDF 结构化重建需要开启 OCR。')
    if not paths:
        raise ValueError('请至少上传一个 PDF 或图片文件。')
    output_dir = Path(output_dir)
    for folder in ('images', 'vectors', 'text', 'backgrounds', 'pages', 'summary', 'exports'):
        (output_dir/folder).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    document = dict(schemaVersion=1, id=document_id, name=name, createdAt=now, updatedAt=now,
                    pages=[], warnings=[], stats={},
                    parser=dict(name='pp-structure-v3', version='1', coordinateSystem='points-top-left',
                                ocrRequested=bool(ocr), mode=mode, scanArtworkExtraction=bool(ocr),
                                nativeFastPath=True, local=True))
    supported = {'.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.bmp'}
    for file_index, value in enumerate(paths):
        path = Path(value)
        if path.suffix.lower() not in supported:
            raise ValueError(f'不支持的文件类型：{path.name}')
        if path.suffix.lower() == '.pdf':
            with fitz.open(path) as pdf, ExitStack() as resources:
                if pdf.needs_pass:
                    raise ValueError(f'PDF {path.name} 已加密，请先解除密码保护后上传。')
                selection = selected_pages(pages, len(pdf))
                reading_engine = _reading_engine() if mode == 'rebuild' else None
                table_queue = None
                if reading_engine:
                    active = _rebuild_table_queue.get()
                    if active and active[0] == path.resolve():
                        table_queue = active[1]
                    else:
                        from pdf_extractor.model import Options
                        table_queue = resources.enter_context(reading_engine._TableQueue(path, selection, Options(ocr='off')))
                if len(document['pages'])+len(selection) > legacy.MAX_PAGES:
                    raise ValueError('每个画布文档最多支持 100 页。')
                for ordinal, number in enumerate(selection):
                    report(100*(file_index+ordinal/max(1,len(selection)))/len(paths), 'V3 版面与结构识别',
                           f'{path.name}；物理第 {number+1} 页', current=ordinal+1, total=len(selection))
                    source = pdf[number]
                    width, height = float(source.rect.width), float(source.rect.height)
                    scale = min(legacy.RENDER_SCALE, math.sqrt(legacy.MAX_PIXELS/(width*height))*.999)
                    if scale < .2:
                        raise ValueError('PDF 页面尺寸过大，无法安全渲染。')
                    pix = source.get_pixmap(matrix=fitz.Matrix(scale,scale), colorspace=fitz.csRGB, alpha=False)
                    rgb = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3).copy()
                    try:
                        native = legacy._native_text(source, source.rotation_matrix)
                    except Exception:
                        native = []
                    index = len(document['pages'])
                    if mode == 'rebuild':
                        # Render/native text work overlaps the independent
                        # native table worker; source documents never cross IPC.
                        reading_blocks = reading_engine._native_text(source, [])
                        native_tables, native_warnings = table_queue.take(number)
                        if native_tables:
                            reading_blocks = reading_engine._native_text(source,native_tables)
                        page = _rebuild_pdf_page(source,rgb,output_dir,document_id,index,path.name,native,
                                                 reading_blocks,native_tables,native_warnings)
                    elif _native_fast_path(source, native):
                        native += legacy._native_graphics(source, source.rotation_matrix, output_dir, document_id, index)
                        page = legacy._compose_page(rgb, width, height, scale, native, output_dir, document_id,
                                                    index, source.rotation, path.name, False, False)
                        page.update(recognitionBackend='native', tables=[])
                    elif ocr:
                        raw = recognize_page(rgb)
                        page = compose_page(raw, rgb, width, height, output_dir, document_id, index,
                                            source.rotation, path.name, native)
                    else:
                        page = _unrecognized_page(rgb,width,height,output_dir,document_id,index,
                                                  source.rotation,path.name)
                    page['sourcePage'] = number+1
                    document['pages'].append(page)
        else:
            with Image.open(path) as image:
                selection = set(selected_pages(pages, getattr(image, 'n_frames', 1)))
            for number, (rgb,width,height,scale) in enumerate(legacy._image_pages(path)):
                if number not in selection:
                    continue
                if len(document['pages']) >= legacy.MAX_PAGES:
                    raise ValueError('每个画布文档最多支持 100 页。')
                if ocr:
                    raw = recognize_page(rgb)
                    if mode == 'rebuild':
                        raw = dict(raw,_service=_dispatch_regions([dict(raw.get('_service') or {},
                            bbox=[0.,0.,float(width),float(height)],pixelBbox=[0,0,rgb.shape[1],rgb.shape[0]])],
                            strategy='full-image'))
                    page = compose_page(raw, rgb, width, height, output_dir, document_id,
                                        len(document['pages']), 0, path.name)
                else:
                    page = _unrecognized_page(rgb,width,height,output_dir,document_id,
                                              len(document['pages']),0,path.name)
                page.update(sourcePage=number+1, nativePixelSize=[rgb.shape[1],rgb.shape[0]])
                document['pages'].append(page)
    if not document['pages']:
        raise ValueError('文件中没有可解析的页面。')
    counts = Counter(e['type'] for p in document['pages'] for e in p['elements'])
    document['stats'] = dict(pageCount=len(document['pages']), elementCount=sum(counts.values()), byType=dict(counts),
                             reviewCount=sum(bool(e.get('reviewRequired')) for p in document['pages'] for e in p['elements']))
    document['warnings'] = ['扫描页使用本地 V3 模型；可用原生文字页跳过模型。模型预测、近似字体与表格结构仍需校对。']
    report(100, '页面解析完成')
    return document
