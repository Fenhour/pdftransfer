"""Measured two-stage layout with source-preserving Region isolation.

New relationship snapshots retain source anchors and bound local growth.
Older snapshots without that policy retain their constrained flow behaviour.
Measurement and PDF output share the same safe TrueType font objects.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import statistics

from .font_library import get_library


ENGINE_VERSION = 'feature4-adaptive-4-internal-structure'
_EPS = 1e-5
_TITLES = {'doc_title', 'paragraph_title', 'section_title', 'title', 'reference'}


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite number')
    return value


def _font(text, style):
    library = get_library()
    face = library.resolve(text, style.get('family', 'sans-serif'),
                           bool(style.get('bold')), bool(style.get('italic')), safe_only=True)
    if face is None:
        raise ValueError('没有覆盖重建文字的安全字体，未替换或丢弃字符')
    return face.key, library.export_font_for_key(face.key)


def _text_measurement(text, width, style):
    """Wrap measured characters; display strings and source ranges stay separate."""
    size = _positive(style.get('font_size_pt', 11), 'font_size_pt')
    key, font = _font(text, style)
    step = size * max(float(style.get('line_height', 1.25)), font.ascender-font.descender)
    widths = list(font.char_lengths(text, fontsize=size))
    tab_width = font.text_length('    ', fontsize=size)
    widths = [tab_width if c == '\t' else w for c, w in zip(text, widths)]
    widest = max((w for c, w in zip(text, widths) if c not in '\r\n'), default=0)
    if widest > width + _EPS:
        raise ValueError('块宽度小于一个可读字形，不能横向压缩文字')
    lines = []
    start = 0
    # Explicit newlines belong to the preceding source range, including blank
    # lines.  Soft wrapping consumes every original character exactly once.
    while start < len(text):
        newline = text.find('\n', start)
        stop = newline if newline >= 0 else len(text)
        if stop == start:
            end = start + 1
        else:
            end, advance = start, 0.
            while end < stop and advance + widths[end] <= width + _EPS:
                advance += widths[end]
                end += 1
            if end == start:
                raise ValueError('文字无法在可用宽度内排版')
            if end < stop:
                boundary = max(text.rfind(' ', start, end), text.rfind('\t', start, end))
                if boundary > start:
                    end = boundary + 1
            elif newline >= 0:
                end += 1
        value = text[start:end].rstrip('\r\n')
        display = value.replace('\t', '    ')
        advance = font.text_length(display, fontsize=size)
        align = style.get('align', 'left')
        x = max(0., (width-advance)/2 if align == 'center' else width-advance if align == 'right' else 0.)
        lines.append(dict(text=display, start=start, end=end, x=x,
                          baseline=font.ascender*size+len(lines)*step, height=step))
        start = end
    if not lines:
        lines = [dict(text='', start=0, end=0, x=0., baseline=font.ascender*size, height=step)]
    return dict(width=width, height=len(lines)*step, font_key=key, font_size=size,
                line_step=step, color=style.get('color', '#000000'), lines=lines,
                min_width=widest, breaks=[line['end'] for line in lines])


def _column_weights(content, count):
    supplied = content.get('column_widths')
    if isinstance(supplied, list) and len(supplied) == count and all(float(v) > 0 for v in supplied):
        return list(map(float, supplied))
    observations = [[] for _ in range(count)]
    for cell in content.get('cells', []):
        box = cell.get('source_bbox')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        width = (float(box[2])-float(box[0])) / max(1, int(cell.get('colspan', 1)))
        if width > 0:
            for index in range(int(cell['column']), int(cell['column'])+int(cell.get('colspan', 1))):
                if 0 <= index < count:
                    observations[index].append(width)
    return [statistics.median(v) if v else 1. for v in observations]


def _allocate_columns(minimums, weights, width):
    if sum(minimums) > width + _EPS:
        raise ValueError('表格最小可读列宽超过页面宽度，需横向页或校对表格')
    result = [0.] * len(weights)
    remaining, active = width, set(range(len(weights)))
    while active:
        total = sum(weights[i] for i in active)
        proposed = {i: remaining*weights[i]/total for i in active}
        fixed = [i for i in active if proposed[i] < minimums[i]-_EPS]
        if not fixed:
            for i in active:
                result[i] = proposed[i]
            break
        for i in fixed:
            result[i] = minimums[i]
            remaining -= result[i]
            active.remove(i)
    return result


def _table_measurement(block, width):
    content, style = block['content'], block.get('style', {})
    nr, nc = int(content['row_count']), int(content['column_count'])
    if not 0 < nr <= 10000 or not 0 < nc <= 100:
        raise ValueError('无效的表格行列数')
    cells = content.get('cells', [])
    occupied = set()
    padding = max(0., float(style.get('cell_padding_pt', 3)))
    minimums = [2*padding+.1] * nc
    spans = []
    for cell in cells:
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        if r < 0 or c < 0 or rs < 1 or cs < 1 or r+rs > nr or c+cs > nc:
            raise ValueError('无效的表格单元格跨度')
        slots = {(y, x) for y in range(r, r+rs) for x in range(c, c+cs)}
        if occupied & slots:
            raise ValueError('表格单元格重叠')
        occupied |= slots
        text = str(cell.get('text', ''))
        cell_style = dict(style, **cell.get('style', {}))
        if cell.get('header'):
            cell_style['bold'] = True
        key, font = _font(text, cell_style)
        size = _positive(cell_style.get('font_size_pt', 10), 'font_size_pt')
        minimum = max(font.char_lengths(text.replace('\n', ''), fontsize=size), default=.1)+2*padding
        if cs == 1:
            minimums[c] = max(minimums[c], minimum)
        else:
            spans.append((c, cs, minimum))
    if len(occupied) != nr*nc:
        raise ValueError('表格拓扑不完整，不能编造缺失单元格')
    for c, cs, minimum in spans:
        deficit = max(0., minimum-sum(minimums[c:c+cs]))
        for index in range(c, c+cs):
            minimums[index] += deficit/cs
    columns = _allocate_columns(minimums, _column_weights(content, nc), width)
    offsets = [0.]
    for value in columns:
        offsets.append(offsets[-1]+value)
    row_heights = [2*padding+.1] * nr
    measured = []
    for cell in cells:
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        cell_style = dict(style, **cell.get('style', {}))
        if cell.get('header'):
            cell_style['bold'] = True
        cell_style.setdefault('font_size_pt', 10)
        value = _text_measurement(str(cell.get('text', '')), sum(columns[c:c+cs])-2*padding, cell_style)
        needed = value['height']+2*padding
        if rs == 1:
            row_heights[r] = max(row_heights[r], needed)
        measured.append((cell, value, needed))
    for cell, value, needed in measured:
        r, rs = int(cell['row']), int(cell.get('rowspan', 1))
        deficit = max(0., needed-sum(row_heights[r:r+rs]))
        if deficit:
            for index in range(r, r+rs):
                row_heights[index] += deficit/rs
    row_offsets = [0.]
    for value in row_heights:
        row_offsets.append(row_offsets[-1]+value)
    result_cells = []
    for cell, value, needed in measured:
        r, c = int(cell['row']), int(cell['column'])
        rs, cs = int(cell.get('rowspan', 1)), int(cell.get('colspan', 1))
        lines = deepcopy(value['lines'])
        for line in lines:
            line['x'] += padding
            line['baseline'] += padding
        result_cells.append(dict(id=cell['id'], row=r, column=c, rowspan=rs, colspan=cs,
                                 x=offsets[c], y=row_offsets[r], width=sum(columns[c:c+cs]),
                                 height=sum(row_heights[r:r+rs]), text=str(cell.get('text', '')),
                                 header=bool(cell.get('header')), repeat_header=False,
                                 font_key=value['font_key'], font_size=value['font_size'],
                                 line_step=value['line_step'], color=value['color'], lines=lines))
    legal = [k for k in range(1, nr+1)
             if not any(int(c['row']) < k < int(c['row'])+int(c.get('rowspan', 1)) for c in cells)]
    header_count = 0
    while header_count < nr:
        covering = [c for c in cells if int(c['row']) <= header_count < int(c['row'])+int(c.get('rowspan', 1))]
        if not covering or not all(c.get('header') for c in covering):
            break
        header_count += 1
    if header_count == nr or header_count not in legal:
        header_count = 0
    return dict(width=width, height=sum(row_heights), cells=result_cells,
                column_widths=columns, row_heights=row_heights, row_offsets=row_offsets,
                header_count=header_count, breaks=legal, padding=padding,
                border_width=float(style.get('border_width_pt', .5)),
                border_color=style.get('border_color', '#000000'),
                min_width=sum(minimums))


def measure_block(block, width, *, preserve_envelope=True, use_source_layout=True):
    """Stage one: internally lay out a block at an explicit width in points.

    The JSON snapshot is tied to content, style, engine and available width.
    It can be called again after a column/page-width change. No source field is
    overwritten and no text is shrunk to fit the original source height.
    """
    width = _positive(width, 'available_width')
    kind, content = block['kind'], block['content']
    if kind == 'text':
        from .source_text import measure_source_text
        value = measure_source_text(content, width, block.get('style', {}),
                                    block.get('source', {}), _text_measurement,
                                    preserve_envelope=preserve_envelope) if use_source_layout else None
        if value is None:
            value = _text_measurement(str(content.get('text', '')), width, block.get('style', {}))
    elif kind == 'table':
        from .source_table import measure_source_table
        value = measure_source_table(block, width, _table_measurement,
                                     preserve_envelope=preserve_envelope) if use_source_layout else None
        if value is None:
            value = _table_measurement(block, width)
    elif kind == 'image':
        ratio = content.get('aspect_ratio')
        if ratio is None:
            pixels = content['pixel_size']
            ratio = float(pixels[0])/float(pixels[1])
        ratio = _positive(ratio, 'aspect_ratio')
        value = dict(width=width, height=width/ratio, asset=content['asset'], breaks=[])
    elif kind == 'composite' and content.get('layout_mode') == 'source-relative':
        from .regional_layout import measure_source_composite
        value = measure_source_composite(block, width, lambda child, w: measure_block(
            child, w, preserve_envelope=preserve_envelope, use_source_layout=use_source_layout))
    elif kind == 'composite':
        padding = max(0.,float(content.get('padding_pt',4)))
        gap = max(0.,float(content.get('gap_pt',5)))
        inner = _positive(width-2*padding,'composite_inner_width')
        children=[]
        y=padding
        minimum=0.
        for child in content.get('children',[]):
            if child['kind']!='text':
                raise ValueError('组合块目前仅接受文字子块')
            measured=measure_block(child,inner,preserve_envelope=preserve_envelope,use_source_layout=use_source_layout)
            children.append(dict(block_id=child['id'],x=padding,y=y,measurement=measured))
            minimum=max(minimum,measured.get('min_width',0.))
            y+=measured['height']+gap
        if not children:
            raise ValueError('组合块不能为空')
        value=dict(width=width,height=y-gap+padding,children=children,breaks=[],
                   background=deepcopy(content.get('background',{})),
                   decorations=deepcopy(content.get('decorations',{})),min_width=minimum+2*padding)
    else:
        raise ValueError(f'Unsupported reconstruction block kind: {kind}')
    # The editable block footprint includes its source whitespace. Compact
    # measurement is an explicit Region overflow fallback, never the default.
    box = block.get('source', {}).get('bbox_pt')
    if preserve_envelope and kind in {'text','table'} and isinstance(box, (list,tuple)) and len(box)==4 and box[2]>box[0] and box[3]>box[1]:
        value['height'] = max(value['height'], (box[3]-box[1])*width/(box[2]-box[0]))
    payload = dict(engine=ENGINE_VERSION, kind=kind, width=width,
                   content=content, style=block.get('style', {}),
                   source_geometry=block.get('source', {}).get('bbox_pt'),
                   preserve_envelope=preserve_envelope, use_source_layout=use_source_layout)
    value['signature'] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                  allow_nan=False).encode()).hexdigest()
    value['engine'] = ENGINE_VERSION
    value['kind'] = kind
    value['unit'] = 'pt'
    value['available_width'] = width
    block['local_layout'] = dict(block.get('local_layout',{}),method='shape-preserving-measurement', unit='pt',
                                 measurement=deepcopy(value), source_height_is_limit=False,
                                 preserve_source_envelope=preserve_envelope)
    return value


def _topological_order(ids, edges):
    pending, done, result = list(ids), set(), []
    predecessors = {value: set() for value in ids}
    for edge in edges:
        if isinstance(edge, dict):
            a, b = edge.get('from', edge.get('before')), edge.get('to', edge.get('after'))
        else:
            a, b = edge
        if a not in predecessors or b not in predecessors:
            raise ValueError('阅读顺序约束引用了不存在的块')
        predecessors[b].add(a)
    while pending:
        found = next((value for value in pending if predecessors[value] <= done), None)
        if found is None:
            raise ValueError('阅读顺序存在循环，不能在排版时任意破坏依赖')
        result.append(found)
        pending.remove(found)
        done.add(found)
    return result, predecessors


def _text_fragment(measurement, first, last):
    result = deepcopy(measurement)
    step = measurement['line_step']
    result['lines'] = deepcopy(measurement['lines'][first:last])
    for line in result['lines']:
        line['baseline'] -= first*step
    result['height'] = (last-first)*step
    result['breaks'] = [line['end'] for line in result['lines']]
    return result


def _table_fragment(measurement, start, end, repeat):
    result = deepcopy(measurement)
    head = measurement['header_count'] if repeat else 0
    header_height = measurement['row_offsets'][head]
    base = measurement['row_offsets'][start]
    selected = []
    for cell in measurement['cells']:
        if start <= cell['row'] and cell['row']+cell['rowspan'] <= end:
            copied = deepcopy(cell)
            copied['y'] = cell['y']-base+header_height
            selected.append(copied)
        elif repeat and cell['row']+cell['rowspan'] <= head:
            copied = deepcopy(cell)
            copied['repeat_header'] = True
            selected.append(copied)
    result['cells'] = selected
    result['height'] = header_height+measurement['row_offsets'][end]-base
    result['row_heights'] = measurement['row_heights'][:head]+measurement['row_heights'][start:end]
    result['row_range'] = [start, end]
    result['breaks'] = []
    return result


def layout_document(project, page_size=None, margin=24):
    """Dispatch new projects to source-preserving layout; read old flow IR too."""
    if project.get('relations', {}).get('layout_policy') == 'source-preserving-regions-v1':
        from .regional_layout import layout_regions
        return layout_regions(project, page_size=page_size, margin=margin)
    return _layout_flow_document(project, page_size=page_size, margin=margin)


def _layout_flow_document(project, page_size=None, margin=24):
    """Stage two: constrained single-column flow with bounded local exchange.

    Source columns/bands determine source_order in the relationship stage;
    this first implementation intentionally targets one readable column. Only
    explicitly declared independent siblings may exchange their whole-block
    order. Long objects split at semantic/line boundaries, retaining their ID.
    """
    blocks = project.get('blocks', [])
    by_id = {block['id']: block for block in blocks}
    if len(by_id) != len(blocks):
        raise ValueError('重建块 ID 不唯一')
    relations = project.get('relations', {})
    ids = list(relations.get('source_order', by_id))
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('参考阅读顺序必须完整且唯一覆盖所有块')
    pending, predecessors = _topological_order(ids, relations.get('hard_before', []))
    if page_size is None:
        source = (project.get('source_pages') or [{}])[0]
        page_size = (source.get('width_pt', 595.276), source.get('height_pt', 841.89))
    pw, ph = map(float, page_size)
    margin = _positive(margin, 'margin')
    if not all(math.isfinite(v) for v in (pw, ph)) or min(pw, ph) <= margin*2+1:
        raise ValueError('目标页面过小或尺寸无效')
    available, capacity = pw-2*margin, ph-2*margin
    groups = [set(g.get('block_ids', g.get('members', [])) if isinstance(g, dict) else g)
              for g in relations.get('reorder_groups', [])]
    pages, placements, done, resolved, changes, issues = [], [], set(), [], [], []
    if pending != ids:
        reference_index = {identifier: index for index, identifier in enumerate(ids)}
        for index, identifier in enumerate(pending):
            if index != reference_index[identifier]:
                changes.append(dict(block_id=identifier, source_index=reference_index[identifier],
                                    target_index=index, reason='hard-dependency-overrides-source-order'))
    page_index, y = -1, margin
    measurements = {}

    def new_page():
        nonlocal page_index, y
        page_index += 1
        pages.append(dict(index=page_index, width=pw, height=ph))
        y = margin

    def measure(identifier):
        if identifier not in measurements:
            block = by_id[identifier]
            box = block.get('source', {}).get('bbox_pt') or [0, 0, available, 0]
            source_width = max(1., float(box[2])-float(box[0]))
            preferred = min(available, source_width if block['kind'] in {'image','composite'}
                            else max(available*.5, source_width))
            try:
                initial = measure_block(block, preferred, preserve_envelope=False, use_source_layout=False)
            except ValueError:
                if preferred >= available-_EPS:
                    raise
                initial = measure_block(block, available, preserve_envelope=False, use_source_layout=False)
                preferred = available
            # Preserve the original width when practical; widen before causing
            # extra pages. The selected result remains a real remeasurement.
            if block['kind'] != 'image' and preferred < available-_EPS and (
                    initial['height'] > capacity*.7 or block['kind'] == 'table'):
                wide = measure_block(block, available, preserve_envelope=False, use_source_layout=False)
                if wide['height'] < initial['height']*.85:
                    initial = wide
                else:
                    block['local_layout']['measurement'] = deepcopy(initial)
            if block['kind'] == 'image' and initial['height'] > capacity:
                ratio = initial['width']/initial['height']
                initial = measure_block(block, min(available, capacity*ratio), preserve_envelope=False, use_source_layout=False)
                issues.append(dict(block_id=identifier, reason='image-scaled-proportionally-to-page'))
            measurements[identifier] = initial
        return measurements[identifier]

    def add(identifier, value, content_range, repeat=False):
        nonlocal y
        if value['height'] > capacity+_EPS or value['width'] > available+_EPS:
            raise ValueError('不可拆内容超过目标页面，需专页或校对，未裁掉内容')
        count = sum(p['block_id'] == identifier for p in placements)
        anchor=by_id[identifier].get('local_layout',{}).get('anchor',{})
        left=margin
        if anchor.get('horizontal')=='center':
            # Source columns inform reading order; the target is one column.
            # Page/region/column anchors therefore map to the target flow frame.
            left=margin+(available-value['width'])/2
        placements.append(dict(fragment_id=f'{identifier}-fragment-{count+1}', block_id=identifier,
                               page_index=page_index, bbox=[left, y, left+value['width'], y+value['height']],
                               content_range=content_range, measurement=value, repeat_header=repeat))
        y += value['height']

    def gap(block):
        label = block.get('source', {}).get('label', '')
        return 9. if label in _TITLES else 6.

    def remaining():
        return ph-margin-y

    new_page()
    while pending:
        eligible = [value for value in pending if predecessors[value] <= done]
        if not eligible:
            raise ValueError('阅读顺序约束不能满足')
        identifier = eligible[0]
        base = measure(identifier)
        # Bounded lookahead: only exchange an explicitly independent group,
        # and only when the head cannot fit whole while a sibling can use a
        # substantial amount of the remaining frame. No arbitrary bin packing.
        if base['height'] > remaining()+_EPS and y > margin+_EPS:
            for candidate in eligible[1:3]:
                crossed = pending[:pending.index(candidate)+1]
                if not any(set(crossed) <= group for group in groups):
                    continue
                option = measure(candidate)
                if option['height'] <= remaining()+_EPS and option['height'] >= remaining()*.35:
                    changes.append(dict(block_id=candidate, before=identifier, crossed=crossed[:-1],
                                        reason='independent-sibling-fits-remaining-frame'))
                    identifier, base = candidate, option
                    break
        block = by_id[identifier]
        # Keep a short title/attached label with at least two lines of its
        # successor when that preference can be met on a fresh page.
        next_id = next((value for value in pending if value != identifier), None)
        label = block.get('source', {}).get('label', '')
        attached = any(a.get('anchor') == identifier and a.get('dependent') == next_id
                       for a in relations.get('attachments', []) if isinstance(a, dict))
        if next_id and (label in _TITLES or attached) and y > margin+_EPS:
            successor = measure(next_id)
            follow = min(successor['height'], successor.get('line_step', successor['height'])*2)
            needed = base['height']+gap(block)+follow
            if remaining()+_EPS < needed <= capacity:
                new_page()
        if block['kind'] == 'text':
            first, total = 0, len(base['lines'])
            while first < total:
                fit = int((remaining()+_EPS)//base['line_step'])
                count = min(total-first, fit)
                if count <= 0 and y <= margin+_EPS:
                    raise ValueError('页面高度不足以容纳一行可读文字')
                if count <= 0 or (count == 1 and total-first > 1 and y > margin+_EPS):
                    new_page()
                    continue
                # Two-line orphan/widow preference. Very short paragraphs and
                # genuinely tiny frames relax it, rather than looping forever.
                if total-first-count == 1 and count >= 3:
                    count -= 1
                if count <= 0:
                    raise ValueError('页面高度不足以容纳一行可读文字')
                end = first+count
                value = _text_fragment(base, first, end)
                add(identifier, value, dict(start=value['lines'][0]['start'], end=value['lines'][-1]['end'],
                                            line_start=first, line_end=end))
                first = end
                if first < total:
                    new_page()
        elif block['kind'] == 'table':
            start, nr = 0, len(base['row_heights'])
            while start < nr:
                repeat = bool(start and base['header_count'])
                header_height = base['row_offsets'][base['header_count']] if repeat else 0.
                allowed = [end for end in base['breaks'] if end > start and
                           base['row_offsets'][end]-base['row_offsets'][start]+header_height <= remaining()+_EPS]
                if start == 0 and nr > base['header_count']:
                    allowed = [end for end in allowed if end > base['header_count']]
                if not allowed:
                    if y > margin+_EPS:
                        new_page()
                        continue
                    raise ValueError('不可拆表格行组高于整页，需专页或校对，未破坏合并单元格')
                end = max(allowed)
                add(identifier, _table_fragment(base, start, end, repeat), dict(row_start=start, row_end=end), repeat)
                start = end
                if start < nr:
                    new_page()
        else:
            if base['height'] > remaining()+_EPS:
                new_page()
            add(identifier, deepcopy(base), dict(whole=True))
        done.add(identifier)
        pending.remove(identifier)
        resolved.append(identifier)
        y += gap(block)
    if not blocks:
        pages.clear()
    result = dict(engine=ENGINE_VERSION, target_structure='single-column-reflow', unit='pt',
                  pages=pages, placements=placements, resolved_order=resolved,
                  order_changes=changes, issues=issues)
    project['layout_result'] = result
    return result
