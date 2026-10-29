"""Export the measured feature-four IR without fitting it a second time.

All displayed text uses the exact safe font and point coordinates selected by
the local layout pass.  Original crops are review resources, never a second
visible copy of successfully reconstructed text or tables.
"""
from __future__ import annotations

from collections import Counter
import html
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

import pymupdf as fitz

from .font_library import get_library
from .pdf_safety import save_verified, validate_pdf
from .usage import tracked


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def _asset(root, name):
    """Resolve only project-relative resources, including on Windows hosts."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('重建资源必须使用项目内的相对路径')
    relative = Path(name)
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(base):
        raise ValueError('重建资源路径超出当前项目')
    if not path.is_file():
        raise ValueError('重建资源不存在，未生成缺图结果')
    return path


def _color(value):
    value = str(value or '#000000')
    if not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
        raise ValueError('重建文字颜色无效')
    return tuple(int(value[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _nonspace(value):
    return ''.join(c for c in str(value) if not c.isspace())


def _validate_source_lines(lines, source, start=0, end=None):
    """Permit only the local measurer's explicit whitespace transformations."""
    end = len(source) if end is None else end
    cursor = start
    for line in lines:
        first, last = line.get('start'), line.get('end')
        if (not isinstance(first, int) or not isinstance(last, int) or
                first != cursor or not first <= last <= end):
            raise ValueError('文字行源内容范围存在遗漏或重复')
        display = source[first:last].rstrip('\r\n').replace('\t', '    ')
        if line.get('text', '') != display:
            raise ValueError('文字测量快照与源内容不一致')
        cursor = last
    if cursor != end:
        raise ValueError('文字行内容范围不完整')


def _validate_lines(measured):
    library = get_library()
    for line in measured.get('lines', []):
        if not line.get('text'):
            continue
        key = line.get('font_key', measured.get('font_key'))
        if key not in library.faces:
            raise ValueError('测量使用的安全字体不可用，未擅自替换字体')
        font = library.export_font_for_key(key)
        size = float(line.get('font_size', measured['font_size']))
        x = float(line.get('x', 0))
        baseline = float(line['baseline'])
        if not all(math.isfinite(v) for v in (size, x, baseline)) or size <= 0:
            raise ValueError('文字测量坐标无效')
        if (x < -.05 or x + font.text_length(line['text'], fontsize=size) > measured['width'] + .05 or
                baseline - font.ascender * size < -.05 or
                baseline - font.descender * size > measured['height'] + .05):
            raise ValueError('文字测量结果溢出内容块，未裁掉文字')
        _color(line.get('color', measured.get('color')))


def _walk_blocks(blocks):
    for block in blocks:
        yield block
        yield from _walk_blocks(block.get('content',{}).get('children',[]))


def _validate_composite(block, measured):
    children={c['id']:c for c in block['content'].get('children',[])}
    values=measured.get('children',[])
    if not children or Counter(v['block_id'] for v in values)!=Counter(children.keys()):
        raise ValueError('组合块子块遗漏或重复')
    scale = float(measured.get('uniform_scale', 1.))
    if not math.isfinite(scale) or not 0 < scale <= 1.:
        raise ValueError('组合块等比缩放系数无效')
    expected_decoration = dict(block['content'].get('decorations', {}))
    if 'border_width_pt' in expected_decoration:
        expected_decoration['border_width_pt'] *= scale
    actual_decoration = measured.get('decorations', {})
    expected_border = float(expected_decoration.get('border_width_pt', 0))
    actual_border = float(actual_decoration.get('border_width_pt', 0))
    if (measured.get('background') != block['content'].get('background', {}) or
            {k: v for k, v in actual_decoration.items() if k != 'border_width_pt'} !=
            {k: v for k, v in expected_decoration.items() if k != 'border_width_pt'} or
            not math.isclose(actual_border, expected_border, rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError('组合块背景测量与源内容不一致')
    _color(measured.get('background',{}).get('fill','#ffffff'))
    decoration=measured.get('decorations',{})
    _color(decoration.get('border_color','#000000'))
    border=float(decoration.get('border_width_pt',0))
    if not math.isfinite(border) or border<0:
        raise ValueError('组合块边框无效')
    previous=[]
    for value in values:
        child=children[value['block_id']];m=value['measurement']
        if child['kind']!='text':raise ValueError('组合块子块类型无效')
        if not math.isclose(float(m.get('uniform_scale', 1.)), scale, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('组合块父子必须采用相同的等比缩放系数')
        x,y=value['x'],value['y'];w,h=m['width'],m['height']
        if (not all(math.isfinite(v) for v in (x,y,w,h)) or min(x,y)<0 or min(w,h)<=0 or
                x+w>measured['width']+.05 or y+h>measured['height']+.05):
            raise ValueError('组合块子块越界')
        for a,b,c,d in previous:
            if max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))>.05:
                raise ValueError('组合块子块发生重叠')
        previous.append((x,y,x+w,y+h))
        _validate_lines(m)
        _validate_source_lines(m.get('lines',[]),child['content'].get('text',''))


def _validate_region_frames(project, layout, page_by_index):
    """Verify isolation boundaries for the source-preserving layout mode."""
    if layout.get('target_structure') != 'source-preserving-regions':
        return None
    source_ids = [p.get('id') for p in project.get('source_pages', [])]
    target_ids = [p.get('source_page_id') for p in page_by_index.values()]
    if (None in source_ids or len(source_ids) != len(set(source_ids)) or
            Counter(target_ids) != Counter(source_ids)):
        raise ValueError('保留源版面的源页与目标页必须一一对应')
    frames, owners = {}, {}
    block_ids = {b['id'] for b in project.get('blocks', [])}
    for frame in layout.get('frames', []):
        identifier = frame.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in frames:
            raise ValueError('区域边界 ID 必须唯一且非空')
        page = page_by_index.get(frame.get('page_index'))
        box = frame.get('bbox', [])
        if (page is None or len(box) != 4 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)):
            raise ValueError('区域边界坐标或目标页无效')
        x0, y0, x1, y1 = box
        if (min(x0, y0) < -.02 or x1 <= x0 or y1 <= y0 or
                x1 > page['width']+.02 or y1 > page['height']+.02):
            raise ValueError('区域边界越出目标页面')
        if (frame.get('source_page_id') is not None and
                frame['source_page_id'] != page['source_page_id']):
            raise ValueError('区域边界的源页归属与目标页不一致')
        for member in frame.get('block_ids', []):
            if member not in block_ids or member in owners:
                raise ValueError('区域成员必须存在且只有一个可见归属')
            owners[member] = identifier
        frames[identifier] = frame
    return frames


def _preserved_owned_overlap(first, second, by_id):
    """Allow an existing transparent ownership mask only under one transform."""
    from .regional_layout import image_source_bbox, owned_image_pair
    if not first.get('frame_id') or first.get('frame_id') != second.get('frame_id'):
        return False
    if not owned_image_pair(by_id[first['block_id']], by_id[second['block_id']]):
        return False
    transforms = []
    for placement in (first, second):
        value = placement.get('source_transform', [])
        if (not isinstance(value, (list, tuple)) or len(value) != 6 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
            return False
        a, b, c, d, tx, ty = map(float, value)
        if (not 0 < a <= 1. or abs(b) > 1e-9 or abs(c) > 1e-9 or
                not math.isclose(a, d, rel_tol=1e-9, abs_tol=1e-9)):
            return False
        source = image_source_bbox(by_id[placement['block_id']])
        if (len(source) != 4 or
                not all(isinstance(v, (int, float)) and math.isfinite(v) for v in source)):
            return False
        expected = [a*source[0]+tx, d*source[1]+ty, a*source[2]+tx, d*source[3]+ty]
        if any(abs(actual-wanted) > .05 for actual, wanted in zip(placement['bbox'], expected)):
            return False
        transforms.append(value)
    return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(*transforms))


def _owned_alpha_overlap(first, second, by_id, root):
    """Check the actual rendered assets, including every intersecting pixel."""
    from PIL import Image
    a, b = first['bbox'], second['bbox']
    intersection = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    transparent = []
    for placement in (first, second):
        block = by_id[placement['block_id']]
        name = placement['measurement'].get('asset') or block['content']['asset']
        with Image.open(_asset(root, name)) as image:
            if list(image.size) != block['content'].get('pixel_size'):
                return False
            x0, y0, x1, y1 = placement['bbox']
            sx, sy = image.width/(x1-x0), image.height/(y1-y0)
            left = max(0, math.floor((intersection[0]-x0)*sx+1e-7))
            top = max(0, math.floor((intersection[1]-y0)*sy+1e-7))
            right = min(image.width, math.ceil((intersection[2]-x0)*sx-1e-7))
            bottom = min(image.height, math.ceil((intersection[3]-y0)*sy-1e-7))
            if right <= left or bottom <= top:
                return False
            alpha = image.convert('RGBA').getchannel('A').crop((left, top, right, bottom))
            transparent.append(alpha.getextrema()[1] == 0)
    return any(transparent)


def _validate(project, root=None):
    blocks = project.get('blocks', [])
    by_id = {b['id']: b for b in blocks}
    if len(by_id) != len(blocks):
        raise ValueError('重建块 ID 重复')
    all_ids=[b['id'] for b in _walk_blocks(blocks)]
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError('组合块与独立块 ID 重复')
    layout = project['layout_result']
    pages = layout.get('pages', [])
    if not pages:
        raise ValueError('自适应排版没有生成目标页面')
    page_by_index = {p['index']: p for p in pages}
    if set(page_by_index) != set(range(len(pages))):
        raise ValueError('目标页面索引必须连续且唯一')
    frames = _validate_region_frames(project, layout, page_by_index)
    seen = set()
    text_ranges = {}
    table_cells = {}
    image_counts = Counter()
    page_boxes = {}
    for p in layout.get('placements', []):
        if p['fragment_id'] in seen or p['block_id'] not in by_id:
            raise ValueError('重建片段 ID 重复或来源块不存在')
        seen.add(p['fragment_id'])
        page = page_by_index.get(p['page_index'])
        if page is None:
            raise ValueError('重建片段的目标页不存在')
        box = p['bbox']
        if len(box) != 4 or not all(math.isfinite(float(n)) for n in box):
            raise ValueError('重建片段坐标无效')
        x0, y0, x1, y1 = box
        if min(x0, y0) < -.02 or x1 <= x0 or y1 <= y0 or x1 > page['width'] + .02 or y1 > page['height'] + .02:
            raise ValueError('重建片段越出目标页面')
        measured = p['measurement']
        if abs(measured['width'] - (x1 - x0)) > .05 or abs(measured['height'] - (y1 - y0)) > .05:
            raise ValueError('排版坐标与测量快照不一致')
        block = by_id[p['block_id']]
        if frames is not None:
            frame = frames.get(p.get('frame_id'))
            if frame is None or frame['page_index'] != p['page_index']:
                raise ValueError('重建片段的所属区域不存在或跨越目标页')
            if block['source'].get('page_id') != page['source_page_id']:
                raise ValueError('保留源版面的内容块不得移到其他源页')
            fx0, fy0, fx1, fy1 = frame['bbox']
            if x0 < fx0-.02 or y0 < fy0-.02 or x1 > fx1+.02 or y1 > fy1+.02:
                raise ValueError('重建片段越出所属区域边界')
            if 'block_ids' in frame and p['block_id'] not in frame['block_ids']:
                raise ValueError('重建片段不属于当前区域')
        for previous in page_boxes.setdefault(p['page_index'], []):
            previous_box = previous['bbox']
            overlap = max(0., min(x1, previous_box[2]) - max(x0, previous_box[0])) * max(0., min(y1, previous_box[3]) - max(y0, previous_box[1]))
            if overlap > .05:
                if not (frames is not None and _preserved_owned_overlap(p, previous, by_id)):
                    raise ValueError('自适应重建的内容块发生重叠')
                if root is not None and not _owned_alpha_overlap(p, previous, by_id, root):
                    raise ValueError('重叠区域的图片透明像素与归属证据不一致')
        page_boxes[p['page_index']].append(p)
        if block['kind'] == 'text':
            _validate_lines(measured)
            content = block['content'].get('text', '')
            r = p.get('content_range', {})
            if not (0 <= r.get('start', -1) <= r.get('end', -1) <= len(content)):
                raise ValueError('文字片段内容范围无效')
            _validate_source_lines(measured.get('lines', []), content, r['start'], r['end'])
            text_ranges.setdefault(block['id'], []).append((r['start'], r['end']))
        elif block['kind'] == 'table':
            cells = {c['id']: c for c in block['content']['cells']}
            if len(cells) != len(block['content']['cells']):
                raise ValueError('源表格单元格 ID 重复')
            r = p.get('content_range', {})
            first, last = r.get('row_start'), r.get('row_end')
            if (not isinstance(first, int) or not isinstance(last, int) or
                    not 0 <= first < last <= block['content']['row_count']):
                raise ValueError('表格片段行范围无效')
            repeated, seen_cells = False, set()
            for cell in measured.get('cells', []):
                if cell['id'] in seen_cells:
                    raise ValueError('同一表格片段的单元格 ID 重复')
                seen_cells.add(cell['id'])
                if (min(cell['x'], cell['y']) < -.02 or cell['width'] <= 0 or cell['height'] <= 0 or
                        cell['x'] + cell['width'] > measured['width'] + .05 or
                        cell['y'] + cell['height'] > measured['height'] + .05):
                    raise ValueError('单元格测量结果溢出表格片段')
                _validate_lines(cell)
                source = cells.get(cell['id'])
                if source is None:
                    raise ValueError('表格片段包含未知单元格')
                if cell.get('text', '') != source.get('text', ''):
                    raise ValueError('单元格测量快照与源内容不一致')
                _validate_source_lines(cell.get('lines', []), source.get('text', ''))
                if any(cell.get(k, 1) != source.get(k, 1) for k in ('row', 'column', 'rowspan', 'colspan')):
                    raise ValueError('表格测量快照改变了单元格结构')
                if cell.get('repeat_header', False):
                    repeated = True
                    if (not p.get('repeat_header') or not source.get('header') or not cell.get('header') or
                            first == 0 or source['row'] + source.get('rowspan', 1) > first):
                        raise ValueError('重复表头标记只能用于续表中的源表头')
                else:
                    if not first <= source['row'] < source['row'] + source.get('rowspan', 1) <= last:
                        raise ValueError('普通单元格不属于当前表格片段行范围')
                    table_cells.setdefault(block['id'], Counter())[cell['id']] += 1
            if bool(p.get('repeat_header')) != repeated:
                raise ValueError('表格片段与单元格的重复表头标记不一致')
        elif block['kind'] == 'composite':
            _validate_composite(block,measured)
            image_counts[block['id']] += 1
        elif block['kind'] == 'image':
            image_counts[block['id']] += 1
        elif block['kind'] != 'image':
            raise ValueError('不支持的重建块类型')
    # Every logical block must have at least one concrete placement.
    placed = {p['block_id'] for p in layout.get('placements', [])}
    if set(by_id) != placed:
        raise ValueError('自适应排版存在未放置的内容块')
    for block in blocks:
        if block['kind'] == 'text':
            cursor = 0
            for start, end in sorted(text_ranges.get(block['id'], [])):
                if start != cursor:
                    raise ValueError('跨页文字内容范围存在遗漏或重复')
                cursor = end
            if cursor != len(block['content'].get('text', '')):
                raise ValueError('跨页文字内容范围不完整')
        elif block['kind'] == 'table':
            wanted = Counter(c['id'] for c in block['content']['cells'])
            if table_cells.get(block['id'], Counter()) != wanted:
                raise ValueError('跨页表格单元格存在遗漏或重复')
        elif image_counts[block['id']] != 1:
            raise ValueError('图片块发生重复放置')
    return by_id, pages, layout


def _line_text(page, lines, measured, origin, expected):
    library = get_library()
    for line in lines:
        text = line.get('text', '')
        if not text:
            continue
        key = line.get('font_key', measured.get('font_key'))
        if key not in library.faces:
            raise ValueError('测量使用的安全字体不可用，未擅自替换字体')
        font = library.export_font_for_key(key)
        size = float(line.get('font_size', measured['font_size']))
        writer = fitz.TextWriter(page.rect)
        writer.append((origin[0] + line.get('x', 0), origin[1] + line['baseline']),
                      text, font=font, fontsize=size)
        writer.write_text(page, color=_color(line.get('color', measured.get('color'))), overlay=True)
        expected.update(c for c in text if not c.isspace())


def _render_pdf(project, root, destination):
    by_id, pages, layout = _validate(project)
    placements = layout['placements']
    expected = []
    with fitz.open() as pdf:
        for spec in sorted(pages, key=lambda p: p['index']):
            page = pdf.new_page(width=spec['width'], height=spec['height'])
            inventory = Counter()
            for placement in (p for p in placements if p['page_index'] == spec['index']):
                block = by_id[placement['block_id']]
                m = placement['measurement']
                x, y, _, _ = placement['bbox']
                if block['kind'] == 'text':
                    _line_text(page, m.get('lines', []), m, (x, y), inventory)
                elif block['kind'] == 'image':
                    asset = m.get('asset') or block['content']['asset']
                    page.insert_image(fitz.Rect(placement['bbox']), filename=str(_asset(root, asset)), keep_proportion=True)
                elif block['kind'] == 'composite':
                    decorations=m.get('decorations',{})
                    width=float(decorations.get('border_width_pt',0))
                    inset=width/2
                    rect=fitz.Rect(x+inset,y+inset,x+m['width']-inset,y+m['height']-inset)
                    page.draw_rect(rect,fill=_color(m.get('background',{}).get('fill','#ffffff')),
                                   color=_color(decorations.get('border_color','#000000')) if width else None,
                                   width=width,overlay=True)
                    for child in m['children']:
                        cm=child['measurement']
                        _line_text(page,cm['lines'],cm,(x+child['x'],y+child['y']),inventory)
                else:
                    style = block.get('style', {})
                    border_width = max(0., float(m.get('border_width', style.get('border_width_pt', .5))))
                    border = _color(m.get('border_color', style.get('border_color', '#000000')))
                    for cell in m.get('cells', []):
                        rect = fitz.Rect(x + cell['x'], y + cell['y'],
                                         x + cell['x'] + cell['width'], y + cell['y'] + cell['height'])
                        fill = _color(style.get('header_fill', '#f1f3f5')) if cell.get('header') else None
                        if border_width or fill:
                            page.draw_rect(rect, color=border if border_width else None, fill=fill,
                                           width=border_width, overlay=True)
                        _line_text(page, cell.get('lines', []), cell,
                                   (x + cell['x'], y + cell['y']), inventory)
            expected.append(inventory)
        pdf.set_metadata({'title': '自适应结构化重建', 'creator': 'pdftransfer feature four',
                          'subject': 'V3 content, measured blocks, adaptive pagination; review required'})
        # The project stays portable when only the PDF is downloaded: use the
        # exact relative paths retained in its JSON as attachment names.
        for name in sorted(_resource_names(project)):
            asset = _asset(root, name)
            pdf.embfile_add(name, asset.read_bytes(), filename=name, desc='Project-relative reconstruction resource')
        for block in project['blocks']:
            if block['source'].get('label') == 'table' and block['source'].get('reference_asset'):
                asset = _asset(root, block['source']['reference_asset'])
                name = re.sub(r'[^\w.-]', '_', block['id']) + '-reference' + asset.suffix.lower()
                if name not in pdf.embfile_names():
                    pdf.embfile_add(name, asset.read_bytes(), filename=name, desc='Original table reference')
        return save_verified(pdf, destination, expected), expected


def _finalize_pdf(destination, expected, project_json, diagnostics_json, final_checks):
    """Freeze metadata once, then check the actual final payload without resubsetting."""
    with fitz.open(destination) as pdf:
        pdf.embfile_add('rebuild-project.json', project_json.encode('utf-8'),
                        filename='rebuild-project.json', desc='Source blocks, relations and measured target placements')
        pdf.embfile_add('conversion-report.json', diagnostics_json.encode('utf-8'),
                        filename='conversion-report.json', desc='Processing diagnostics, not recognition accuracy')
        payload = pdf.tobytes(deflate=True, garbage=3)
    if validate_pdf(payload, expected) != final_checks:
        raise ValueError('最终 PDF 元数据封装后的文字检查结果不一致')
    destination.write_bytes(payload)
    if validate_pdf(destination.read_bytes(), expected) != final_checks:
        raise ValueError('最终 PDF 重新打开后的文字检查结果不一致')


def _css_style(measured):
    key = measured.get('font_key')
    face = get_library().faces.get(key)
    family = face.family if face else 'sans-serif'
    family = re.sub(r'[^\w .,-]', '', family)
    _color(measured.get('color', '#000000'))
    return (f'font-family:"{family}",sans-serif;font-size:{measured.get("font_size",11):.3f}pt;'
            f'color:{measured.get("color","#000000")};'
            f'font-weight:{"bold" if face and face.bold else "normal"};'
            f'font-style:{"italic" if face and face.italic else "normal"};')


def _html_lines(lines, measured):
    result = []
    library = get_library()
    for line in lines:
        font = library.export_font_for_key(line.get('font_key', measured['font_key']))
        size = line.get('font_size', measured['font_size'])
        top = line['baseline'] - font.ascender * size
        height = (font.ascender - font.descender) * size
        style = (_css_style(dict(measured, **{k: line[k] for k in ('font_size','font_key','color') if k in line})) +
                 f'position:absolute;left:{line.get("x",0):.3f}pt;top:{top:.3f}pt;'
                 f'height:{height:.3f}pt;line-height:{height:.3f}pt;white-space:pre;')
        result.append(f'<span class="measured-line" style="{html.escape(style,quote=True)}">'
                      f'{html.escape(line.get("text",""))}</span>')
    return ''.join(result)


def _table_html(block, measured=None):
    """Real table topology; measured geometry belongs to the current fragment."""
    if measured is None:
        cells = block['content']['cells']
        rows = sorted({c['row'] for c in cells})
        columns = []
    else:
        cells = measured.get('cells', [])
        rows = sorted({c['row'] for c in cells}, key=lambda row: min(c['y'] for c in cells if c['row'] == row))
        columns = measured.get('column_widths', [])
    style = block.get('style', {})
    border_width = max(0., float((measured or {}).get('border_width', style.get('border_width_pt', .5))))
    border_color = (measured or {}).get('border_color', style.get('border_color', '#000000'))
    _color(border_color)
    border = f'{border_width:.3f}pt solid {border_color}' if border_width else 'none'
    geometry_style = (f' style="display:block;position:relative;width:{measured["width"]:.3f}pt;'
                      f'height:{measured["height"]:.3f}pt;border:0;"' if measured is not None else '')
    result = [f'<table class="rebuild-table"{geometry_style}>']
    if columns:
        result.append('<colgroup style="display:none">' + ''.join(f'<col style="width:{float(w):.3f}pt">' for w in columns) + '</colgroup>')
    result.append('<tbody style="display:contents">' if measured is not None else '<tbody>')
    for row in rows:
        result.append('<tr style="display:contents">' if measured is not None else '<tr>')
        for cell in sorted((c for c in cells if c['row'] == row), key=lambda c: c['column']):
            tag = 'th' if cell.get('header') else 'td'
            attrs = (f' rowspan="{int(cell.get("rowspan",1))}" colspan="{int(cell.get("colspan",1))}"'
                     f' data-cell-id="{html.escape(str(cell["id"]),quote=True)}"')
            if cell.get('repeat_header'):
                attrs += ' data-repeat-header="true"'
            if measured is None:
                body = html.escape(cell.get('text', '')).replace('\n', '<br>')
                cell_style = f'border:{border};'
            else:
                body = _html_lines(cell.get('lines', []), cell)
                cell_style = (f'display:block;position:absolute;left:{cell["x"]:.3f}pt;top:{cell["y"]:.3f}pt;'
                              f'width:{cell["width"]:.3f}pt;height:{cell["height"]:.3f}pt;padding:0;border:0;')
                if border_width:
                    cell_style += f'box-shadow:inset 0 0 0 {border_width:.3f}pt {border_color};'
                body = f'<div style="position:relative;width:{cell["width"]:.3f}pt;height:{cell["height"]:.3f}pt">{body}</div>'
            if cell.get('header'):
                fill = block.get('style', {}).get('header_fill', '#f1f3f5')
                _color(fill)
                cell_style += f'background:{fill};'
            result.append(f'<{tag}{attrs} style="{html.escape(cell_style,quote=True)}">{body}</{tag}>')
        result.append('</tr>')
    result.append('</tbody></table>')
    return ''.join(result)


def _composite_html(block,measured=None,semantic=False):
    content=block['content'];decoration_source = content if semantic else measured
    d=decoration_source.get('decorations',{})
    fill=decoration_source.get('background',{}).get('fill','#ffffff')
    border=d.get('border_color','#000000')
    _color(fill);_color(border)
    style=f'background:{fill};border:{float(d.get("border_width_pt",0)):.3f}pt solid {border};box-sizing:border-box;'
    if semantic:
        style+=f'padding:{float(content.get("padding_pt",4)):.3f}pt;'
        body=''.join(f'<p data-block-id="{html.escape(c["id"],quote=True)}" style="text-align:{c.get("style",{}).get("align","left")};color:{c.get("style",{}).get("color","#000000")}">{html.escape(c["content"].get("text","")).replace(chr(10),"<br>")}</p>' for c in content['children'])
    else:
        # An inset shadow has no CSS box-model offset, matching PDF geometry.
        style+=f'border:0;box-shadow:inset 0 0 0 {float(d.get("border_width_pt",0)):.3f}pt {border};position:relative;width:{measured["width"]:.3f}pt;height:{measured["height"]:.3f}pt;'
        body=''.join(f'<div data-block-id="{html.escape(c["block_id"],quote=True)}" style="position:absolute;left:{c["x"]:.3f}pt;top:{c["y"]:.3f}pt">{_html_lines(c["measurement"]["lines"],c["measurement"])}</div>' for c in measured['children'])
    return f'<div class="composite-container" style="{style}">{body}</div>'


def _html_document(project):
    by_id, pages, layout = _validate(project)
    body = []
    for page in sorted(pages, key=lambda p: p['index']):
        body.append(f'<section class="page" data-page="{page["index"]}" '
                    f'style="width:{page["width"]:.3f}pt;height:{page["height"]:.3f}pt">')
        for p in (p for p in layout['placements'] if p['page_index'] == page['index']):
            block, m = by_id[p['block_id']], p['measurement']
            x0,y0,x1,y1 = p['bbox']
            style = f'left:{x0:.3f}pt;top:{y0:.3f}pt;width:{x1-x0:.3f}pt;height:{y1-y0:.3f}pt;'
            if block['kind'] == 'text':
                inner = _html_lines(m.get('lines', []), m)
            elif block['kind'] == 'table':
                inner = _table_html(block, m)
            elif block['kind'] == 'composite':
                inner = _composite_html(block,m)
            else:
                asset = m.get('asset') or block['content']['asset']
                from urllib.parse import quote
                inner = f'<img src="../{quote(asset,safe="/")}" alt="{html.escape(block["source"].get("label","image"),quote=True)}">'
            body.append(f'<article class="block {block["kind"]}" data-block-id="{html.escape(block["id"],quote=True)}" '
                        f'data-fragment-id="{html.escape(p["fragment_id"],quote=True)}" style="{style}" contenteditable="true">{inner}</article>')
        body.append('</section>')
    css = ('body{background:#e8ecef;margin:20px}.page{position:relative;background:white;margin:18px auto;}'
           '.block{position:absolute;margin:0;padding:0}.block img{display:block;width:100%;height:100%;object-fit:contain}'
           '.rebuild-table{border-collapse:collapse;table-layout:fixed;border-spacing:0;width:100%;}'
           '.rebuild-table td,.rebuild-table th{border:.5pt solid #000;vertical-align:top;box-sizing:border-box;}'
           '.rebuild-table th{background:#f1f3f5;font-weight:normal;}'
           '@media print{body{background:white;margin:0}.page{margin:0;break-after:page}.page:last-child{break-after:auto}}')
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>自适应结构化重建</title><style>' + css + '</style><body>' + ''.join(body) + '</body></html>'


def _markdown_document(project):
    by_id = {b['id']: b for b in project['blocks']}
    order = project['layout_result'].get('resolved_order') or project['relations']['source_order']
    output = []
    for identifier in order:
        block = by_id[identifier]
        label = block['source'].get('label', '')
        if block['kind'] == 'text':
            value = html.escape(block['content'].get('text', '')).replace('\n', '<br>')
            tag = 'h1' if label == 'doc_title' else 'h2' if label in {'paragraph_title','reference'} else 'p'
            output.append(f'<{tag} data-block-id="{html.escape(identifier,quote=True)}">{value}</{tag}>')
        elif block['kind'] == 'table':
            output.append(_table_html(block))
        elif block['kind'] == 'composite':
            output.append(_composite_html(block,semantic=True))
        else:
            from urllib.parse import quote
            output.append(f'![{html.escape(label)}](<../{quote(block["content"]["asset"],safe="/")}>)')
    return '\n\n'.join(output) + '\n'


def _resource_names(project):
    resources = set()
    for page in project.get('source_pages', []):
        if page.get('reference_asset'):
            resources.add(page['reference_asset'])
    for block in _walk_blocks(project['blocks']):
        if block['kind'] == 'image':
            resources.add(block['content']['asset'])
        if block.get('source', {}).get('reference_asset'):
            resources.add(block['source']['reference_asset'])
        for source in block.get('source',{}).get('child_sources',[]):
            if source.get('reference_asset'):resources.add(source['reference_asset'])
    return resources


@tracked('rebuild.adaptive_export', feature=lambda v: 'rebuild.' + v.get('format', 'pdf'))
def export_project(project, output_dir, format='pdf'):
    """Return the requested result and export diagnostics for a measured IR."""
    if format not in {'pdf','html','md'}:
        raise ValueError('自适应重建仅支持 PDF、HTML 或 Markdown')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _validate(project, root=root)
    resources = {name: _asset(root, name) for name in _resource_names(project)}
    diagnostics = dict(schema='feature4-export/1', format=format,
                       strategy='v3-two-stage-adaptive-rebuild', status='needs-review',
                       sourcePageCount=len(project.get('source_pages', [])),
                       pageCount=len(project['layout_result']['pages']),
                       textBlockCount=sum(b['kind']=='text' for b in project['blocks']),
                       tableCount=sum(b['kind']=='table' for b in project['blocks']),
                       imageCount=sum(b['kind']=='image' for b in project['blocks']),
                       compositeCount=sum(b['kind']=='composite' for b in project['blocks']),
                       orderChanges=project['layout_result'].get('order_changes', []),
                       limitations=['OCR and model boundaries require review.',
                                    'Role templates are selected styles, not recovered original typography.',
                                    'PDF text/glyph validation does not certify WPS editing or OCR accuracy.'])
    diagnostics['editableCharacterCount'] = sum(
        len(_nonspace(b['content'].get('text', ''))) if b['kind']=='text' else
        sum(len(_nonspace(c.get('text', ''))) for c in b['content'].get('cells', [])) if b['kind']=='table' else 0
        for b in _walk_blocks(project['blocks']))
    diagnostics['repeatedHeaderCharacterCount'] = sum(
        len(_nonspace(c.get('text', ''))) for p in project['layout_result']['placements']
        for c in p['measurement'].get('cells', []) if c.get('repeat_header'))
    if format == 'pdf':
        destination = root/'rebuilt-text.pdf'
        diagnostics['pdfSafety'], expected = _render_pdf(project, root, destination)
        # This expected result is checked against both the packaged bytes and
        # the reopened file before this export can return successfully.
        diagnostics['pdfSafety']['finalPackaging'] = dict(diagnostics['pdfSafety']['reopened'])
    project.setdefault('report', dict(schema='feature4-conversion/1', status='needs-review'))['export'] = diagnostics
    project_json, diagnostics_json = _json(project), _json(diagnostics)
    if format == 'pdf':
        _finalize_pdf(destination, expected, project_json, diagnostics_json, diagnostics['pdfSafety']['finalPackaging'])
    else:
        tree = root/f'adaptive-{format}'
        (tree/'summary').mkdir(parents=True, exist_ok=True)
        (tree/'model').mkdir(parents=True, exist_ok=True)
        for name, source in resources.items():
            target = tree/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (tree/'project.json').write_bytes(project_json.encode('utf-8'))
        (tree/'model'/'project.json').write_bytes(project_json.encode('utf-8'))
        (tree/'summary'/f'summary.{format}').write_text(
            _html_document(project) if format=='html' else _markdown_document(project), encoding='utf-8')
        (tree/'conversion-report.json').write_bytes(diagnostics_json.encode('utf-8'))
        destination = root/f'rebuilt-{format}.zip'
        with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
            for asset in sorted(tree.rglob('*')):
                if asset.is_file():
                    archive.write(asset,asset.relative_to(tree).as_posix())
    (root/'rebuild-project.json').write_bytes(project_json.encode('utf-8'))
    (root/'conversion-report.json').write_bytes(diagnostics_json.encode('utf-8'))
    return destination, diagnostics
