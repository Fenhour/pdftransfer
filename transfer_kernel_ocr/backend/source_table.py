"""Conservative table geometry: retain proven OCR anchors and source row space.

The HTML model remains the text/topology authority. OCR may supply physical
line structure only when it belongs to exactly one cell and its non-whitespace
content agrees with that cell. Unproven positions fall back to ordinary cell
flow and remain explicitly reviewable.
"""
from __future__ import annotations

from copy import deepcopy
import math
import statistics


_VERSION = 'source-table-v1'


def _rect(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        result = list(map(float, value))
    except (TypeError, ValueError):
        return None
    return result if (all(math.isfinite(v) for v in result) and
                      result[2] > result[0] and result[3] > result[1]) else None


def _nonspace(value):
    return ''.join(c for c in str(value) if not c.isspace())


def _intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def _alignment(physical):
    """Use repeated line edges only; never guess from a cell's empty space."""
    layout = physical['source_layout']
    rows = layout.get('rows', [])
    if len(rows) < 2 or any(len(r['fragment_indices']) != 1 for r in rows):
        return None
    boxes = [r['bbox_px'] for r in rows]
    tolerance = max(2., statistics.median(b[3]-b[1] for b in boxes)*.12)
    spread = lambda values: max(values)-min(values)
    left = spread([b[0] for b in boxes])
    right = spread([b[2] for b in boxes])
    center = spread([(b[0]+b[2])/2 for b in boxes])
    if left <= tolerance:
        align, error = 'left', left
    elif center <= tolerance and min(left, right) > tolerance*2:
        align, error = 'center', center
    elif right <= tolerance and left > tolerance*2:
        align, error = 'right', right
    else:
        return None
    return dict(align=align, confidence=.9, method='repeated-source-cell-line-edges',
                spread_px=error, tolerance_px=tolerance)


def _axes(content):
    """Recover shared boundaries, never infer missing grid lines from text."""
    nr, nc = int(content['row_count']), int(content['column_count'])
    observations = [[[] for _ in range(nc+1)], [[] for _ in range(nr+1)]]
    for cell in content.get('cells', []):
        box = _rect(cell.get('source_bbox'))
        if not box:
            return None
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        if not (0 <= r < r+rs <= nr and 0 <= c < c+cs <= nc):
            return None
        for bucket, index, value in ((0, c, box[0]), (0, c+cs, box[2]),
                                     (1, r, box[1]), (1, r+rs, box[3])):
            observations[bucket][index].append(value)
    # Missing boundaries in a fully spanning row/column have no observed
    # position. A uniform guessed grid would overstate recovered structure.
    if any(not values for axis in observations for values in axis):
        return None
    if any(max(v)-min(v) > 1. for axis in observations for v in axis):
        return None
    columns, rows = [[statistics.median(v) for v in axis] for axis in observations]
    if any(b-a <= .05 for axis in (columns, rows) for a, b in zip(axis, axis[1:])):
        return None
    return columns, rows


def enrich_table_content(content, source, parsed=None):
    """Attach source-only evidence; return content without mutating source.

    A successful cell preserves its character inventory, but physical newlines
    replace HTML's flattened whitespace. The original HTML cell string remains
    available as ``original_html_text``. Disagreements never select one OCR
    interpretation silently and never add an extra visible copy.
    """
    from .source_text import build_text_content

    issues = []
    axes = _axes(content)
    pixel_box, point_box = _rect(source.get('bbox_px')), _rect(source.get('bbox_pt'))
    status = (parsed or {}).get('geometryStatus', content.get('source_geometry_status'))
    if status:
        content['source_geometry_status'] = status
    trusted = status != 'estimated-from-html-topology' and axes is not None and pixel_box and point_box
    if axes and point_box:
        columns, rows = axes
        # A source-cell grid may differ from its detector envelope by a small
        # halo. Grossly different grids are not a safe physical reference.
        tolerance = max(2., .03*min(point_box[2]-point_box[0], point_box[3]-point_box[1]))
        trusted = bool(trusted and all(abs(a-b) <= tolerance for a, b in
                       zip((columns[0], rows[0], columns[-1], rows[-1]), point_box)))
        content['source_geometry'] = dict(version=_VERSION, columns_pt=columns, rows_pt=rows,
            reliable=bool(trusted), evidence=status or 'coherent-retained-source-cell-grid')
    if not trusted:
        issues.append(dict(reason='unproven-source-cell-geometry'))
        content['source_layout_diagnostics'] = issues
        return content

    sx = (point_box[2]-point_box[0])/(pixel_box[2]-pixel_box[0])
    sy = (point_box[3]-point_box[1])/(pixel_box[3]-pixel_box[1])
    cell_boxes = {c['id']:_rect(c['source_bbox']) for c in content['cells']}
    assignments = {c['id']:[] for c in content['cells']}
    ambiguous = set()
    for index, record in enumerate(source.get('ocr_lines', [])):
        box = _rect(record.get('bbox_px'))
        if not box or not str(record.get('text', '')).strip():
            issues.append(dict(reason='invalid-cell-ocr-record', source_index=index))
            continue
        points = [point_box[0]+(box[0]-pixel_box[0])*sx, point_box[1]+(box[1]-pixel_box[1])*sy,
                  point_box[0]+(box[2]-pixel_box[0])*sx, point_box[1]+(box[3]-pixel_box[1])*sy]
        area = (points[2]-points[0])*(points[3]-points[1])
        ranked = sorted(((_intersection(points, cb)/area, identifier) for identifier, cb in cell_boxes.items()), reverse=True)
        best, owner = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.
        # Small detector spill is normal. A competing cell with substantial
        # coverage is ambiguous even when one overlap is marginally larger.
        if best >= .8 and second <= .2 and best-second >= .6:
            assignments[owner].append(deepcopy(record))
        else:
            candidates = [identifier for ratio, identifier in ranked if ratio > .2]
            ambiguous.update(candidates)
            issues.append(dict(reason='ambiguous-cell-ocr-ownership', source_index=index,
                               candidate_cell_ids=candidates))

    for cell in content['cells']:
        # Re-enriching replayed data must compare with the original model
        # string, never increasingly transform or accumulate physical lines.
        original = str(cell.get('original_html_text', cell.get('text', '')))
        lines = assignments[cell['id']]
        if cell['id'] in ambiguous:
            cell.pop('source_layout', None)
            cell.pop('source_alignment', None)
            cell['text'] = original
            issues.append(dict(reason='cell-ocr-ownership-requires-review', cell_id=cell['id']))
            continue
        physical = build_text_content(lines, cell['id']) if lines else None
        if physical and physical.get('source_layout') and _nonspace(original) == _nonspace(physical['text']):
            cell['original_html_text'] = original
            cell['text'] = physical['text']
            cell['source_layout'] = physical['source_layout']
            cell['source_layout_evidence'] = 'unique-cell-ocr-and-html-nonspace-equivalence'
            alignment = _alignment(physical)
            if alignment:
                cell['source_alignment'] = alignment
        elif _nonspace(original):
            cell.pop('source_layout', None)
            cell.pop('source_alignment', None)
            cell['text'] = original
            reason = ('cell-source-lines-unproven' if physical and not physical.get('source_layout') else
                      'cell-ocr-html-content-mismatch' if lines else 'cell-has-no-ocr-anchors')
            issues.append(dict(reason=reason,
                               cell_id=cell['id']))
        elif lines:
            issues.append(dict(reason='ocr-in-model-empty-cell', cell_id=cell['id']))
    content['source_layout_diagnostics'] = issues
    return content


def measure_source_table(block, width, fallback, *, preserve_envelope=True):
    """Measure proven source cells while retaining their original row space.

    ``fallback`` is the existing topology-validating table measurer. Its grid
    validation, legal rowspan breaks and font checks are retained; source-aware
    cells replace only its internally flattened text. Returns None for old IR.
    """
    from .adaptive_layout import _text_measurement
    from .source_text import measure_source_text

    content, source = block['content'], block.get('source', {})
    geometry = content.get('source_geometry', {})
    if geometry.get('version') != _VERSION or not geometry.get('reliable'):
        return None
    point_box, pixel_box = _rect(source.get('bbox_pt')), _rect(source.get('bbox_px'))
    if not point_box or not pixel_box:
        return None
    result = fallback(block, width)
    nr = int(content['row_count'])
    rows = geometry['rows_pt']
    natural_height = (point_box[3]-point_box[1])*float(width)/(point_box[2]-point_box[0])
    # Normalize tiny cell-grid halos to the observed table envelope. Source
    # row proportions remain unchanged, including otherwise empty whitespace.
    row_heights = [natural_height*(b-a)/(rows[-1]-rows[0]) for a, b in zip(rows, rows[1:])] if preserve_envelope else [.1]*nr
    columns = result['column_widths']
    xoffsets = [0.]
    for value in columns:
        xoffsets.append(xoffsets[-1]+value)
    values = []
    pixels_per_pt_x = (pixel_box[2]-pixel_box[0])/(point_box[2]-point_box[0])
    pixels_per_pt_y = (pixel_box[3]-pixel_box[1])/(point_box[3]-point_box[1])
    prior = {c['id']:c for c in result['cells']}
    for cell in content['cells']:
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        cell_width = sum(columns[c:c+cs])
        style = dict(block.get('style', {}), **cell.get('style', {}))
        alignment = cell.get('source_alignment', {})
        if alignment and cell.get('style', {}).get('align', 'unknown') == 'unknown':
            style['align'] = alignment['align']
        if cell.get('header'):
            style['bold'] = True
        box = cell['source_bbox']
        px = [pixel_box[0]+(box[0]-point_box[0])*pixels_per_pt_x,
              pixel_box[1]+(box[1]-point_box[1])*pixels_per_pt_y,
              pixel_box[0]+(box[2]-point_box[0])*pixels_per_pt_x,
              pixel_box[1]+(box[3]-point_box[1])*pixels_per_pt_y]
        value = measure_source_text(cell, cell_width, style, dict(bbox_px=px, bbox_pt=box),
                                    _text_measurement, preserve_envelope=preserve_envelope)
        if value is None:
            old = prior[cell['id']]
            # The fallback's height includes row siblings, so derive only this
            # cell's ink requirement when its own anchors are unproven.
            value = deepcopy(old)
            font_size = value['font_size']
            from .font_library import get_library
            font = get_library().export_font_for_key(value['font_key'])
            value['height'] = max((line['baseline']-font.descender*line.get('font_size',font_size)
                                   for line in value['lines']), default=.1)+result['padding']
        needed = float(value['height'])
        if rs == 1:
            row_heights[r] = max(row_heights[r], needed)
        values.append((cell, value, needed))
    for cell, value, needed in values:
        r, rs = int(cell['row']), int(cell.get('rowspan', 1))
        deficit = max(0., needed-sum(row_heights[r:r+rs]))
        for index in range(r, r+rs):
            row_heights[index] += deficit/rs
    yoffsets = [0.]
    for value in row_heights:
        yoffsets.append(yoffsets[-1]+value)
    cells = []
    for cell, value, needed in values:
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        measured = deepcopy(prior[cell['id']])
        measured.update(x=xoffsets[c], y=yoffsets[r], width=sum(columns[c:c+cs]),
                        height=sum(row_heights[r:r+rs]), lines=deepcopy(value['lines']),
                        font_key=value['font_key'], font_size=value['font_size'],
                        line_step=value['line_step'], color=value['color'])
        if 'internal_layout' in value:
            measured['internal_layout'] = value['internal_layout']
        cells.append(measured)
    result.update(height=sum(row_heights), cells=cells, row_heights=row_heights,
                  row_offsets=yoffsets, internal_layout=_VERSION,
                  source_envelope_preserved=bool(preserve_envelope),
                  source_layout_diagnostics=deepcopy(content.get('source_layout_diagnostics', [])))
    return result
