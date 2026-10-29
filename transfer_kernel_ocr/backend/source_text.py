"""Conservative physical rows for source-backed text, not inferred typography.

Fragments remain separate drawing operations, so a TOC page number does not
become guessed spaces inside a paragraph.  Source line breaks, indentation and
row gaps are physical evidence.  Role templates only select the font; they do
not invent new internal alignment or remove the source block's spare space.
"""
from __future__ import annotations

import math
from statistics import median

from .font_library import get_library


VERSION = 'source-lines-v1'
_EPS = 1e-6


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        b = list(map(float, value))
    except (TypeError, ValueError):
        return None
    return b if all(math.isfinite(v) for v in b) and b[2] > b[0] and b[3] > b[1] else None


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _cluster(records):
    """Group a baseline before sorting x; a one-pixel y error is not order."""
    rows = []
    for index, record, b in sorted(records, key=lambda v: (v[2][3], v[2][0], v[0])):
        h = b[3]-b[1]
        choices = []
        # Bound comparisons on fragmented pages and retain a fixed row anchor
        # to prevent small baseline errors chaining two adjacent rows together.
        for n, row in enumerate(rows[-24:]):
            delta = abs(b[3]-row['anchor'])
            overlap = max(0., min(b[3], row['bbox_px'][3])-max(b[1], row['bbox_px'][1]))
            if delta <= max(h, row['height'])*.35 and overlap >= min(h, row['height'])*.45:
                choices.append((delta, n, row))
        if choices:
            row = min(choices, key=lambda v: v[:2])[2]
            row['records'].append((index, record, b))
            row['bbox_px'] = _union([row['bbox_px'], b])
        else:
            rows.append(dict(records=[(index, record, b)], bbox_px=b,
                             anchor=b[3], height=h))
    return sorted(rows, key=lambda r: (r['anchor'], r['bbox_px'][0]))


def build_text_content(lines, block_id):
    """Return canonical text plus independently positioned physical fragments.

    Every record appears once. Newlines encode physical fragment boundaries;
    semantic paragraphs are a secondary, conservative view for reading output.
    Invalid geometry retains the words but disables the positioned path.
    """
    records, invalid = [], []
    for index, record in enumerate(lines):
        text = str(record.get('text', ''))
        if not text.strip():
            continue
        b = _box(record.get('bbox_px'))
        if b is None or '\n' in text or '\r' in text:
            invalid.append(text)
        else:
            records.append((index, record, b))
    if invalid:
        text = '\n'.join(str(r.get('text', '')) for r in lines if str(r.get('text', '')).strip())
        return dict(text=text, paragraphs=[dict(id=block_id+'-paragraph-0', text=text)])
    rows = _cluster(records)
    fragments, serialized_rows, chunks, semantic = [], [], [], []
    cursor = 0
    typical = median([b[3]-b[1] for _, _, b in records]) if records else 1.
    paragraph, previous = [], None
    for row_index, row in enumerate(rows):
        if previous is not None and row['bbox_px'][1]-previous[3] > typical*.9:
            semantic.append(' '.join(paragraph))
            paragraph = []
        row_text, indices = [], []
        for _, record, b in sorted(row['records'], key=lambda v: (v[2][0], v[0])):
            value = str(record['text'])
            # The final newline is removed below, along with its source range.
            chunk = value+'\n'
            indices.append(len(fragments))
            fragments.append(dict(text=value, start=cursor, end=cursor+len(chunk),
                                  bbox_px=list(b), row=row_index,
                                  line_id=record.get('line_id')))
            chunks.append(chunk)
            cursor += len(chunk)
            row_text.append(value)
        serialized_rows.append(dict(bbox_px=list(row['bbox_px']), fragment_indices=indices))
        paragraph.append(' '.join(row_text))
        previous = row['bbox_px']
    if paragraph:
        semantic.append(' '.join(paragraph))
    text = ''.join(chunks)
    if fragments:
        text = text[:-1]
        fragments[-1]['end'] -= 1
    return dict(text=text,
                paragraphs=[dict(id=f'{block_id}-paragraph-{i}', text=t) for i, t in enumerate(semantic)],
                source_layout=dict(version=VERSION, coordinate_space='page-pixels',
                                   fragments=fragments, rows=serialized_rows))


def _valid_fragments(content):
    layout = content.get('source_layout', {})
    if layout.get('version') != VERSION or layout.get('coordinate_space') != 'page-pixels':
        return None
    fragments = layout.get('fragments')
    if not isinstance(fragments, list) or not fragments:
        return None
    text, cursor, result = str(content.get('text', '')), 0, []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            return None
        first, last, b = fragment.get('start'), fragment.get('end'), _box(fragment.get('bbox_px'))
        if (not isinstance(first, int) or not isinstance(last, int) or first != cursor or
                not first <= last <= len(text) or b is None):
            return None
        value = text[first:last].rstrip('\r\n')
        if '\n' in value or '\r' in value or value != fragment.get('text'):
            return None
        result.append((fragment, b, value.replace('\t', '    ')))
        cursor = last
    return result if cursor == len(text) else None


def measure_source_text(content, width, style, source, fallback, *, preserve_envelope=True):
    """Measure source rows; return None when an edited/old snapshot lacks evidence.

    Real font metrics may require extra vertical space. A modest isotropic
    font adjustment is shared by all rows; a fragment that still cannot fit
    wraps locally instead of becoming tiny. Source gaps and indentation remain.
    Compact mode releases only unused trailing envelope whitespace, not rows.
    """
    fragments = _valid_fragments(content)
    pixels, points = _box(source.get('bbox_px')), _box(source.get('bbox_pt'))
    if fragments is None or pixels is None or points is None:
        return None
    width = float(width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError('源文字布局宽度无效')
    source_width = points[2]-points[0]
    sx = width/(pixels[2]-pixels[0])
    sy = (points[3]-points[1])/(pixels[3]-pixels[1]) * width/source_width
    source_height = (points[3]-points[1])*width/source_width
    text = str(content.get('text', ''))
    library = get_library()
    face = library.resolve(text, style.get('family', 'sans-serif'), bool(style.get('bold')),
                           bool(style.get('italic')), safe_only=True)
    if face is None:
        # Use the established missing-font error and compatibility behaviour.
        return fallback(text, width, style)
    font = library.export_font_for_key(face.key)
    requested = float(style.get('font_size_pt', 11))
    if not math.isfinite(requested) or requested <= 0:
        raise ValueError('源文字布局字号无效')
    metric_height = font.ascender-font.descender
    rows = {}
    for fragment, b, display in fragments:
        row = fragment.get('row')
        if not isinstance(row, int) or row < 0:
            return None
        rows.setdefault(row, []).append((fragment, b, display))
    lines, previous_bottom, min_width, adaptations = [], 0., 0., []
    # Source row IDs must follow canonical ranges; no arbitrary reordering of
    # the content is introduced by measurement.
    if [f.get('row') for f, _, _ in fragments] != sorted(f.get('row') for f, _, _ in fragments):
        return None
    prepared, required_sizes = [], []
    for row_index, row in sorted(rows.items()):
        ordered = sorted(row, key=lambda v: v[1][0])
        if [v[0]['start'] for v in ordered] != [v[0]['start'] for v in row]:
            return None
        prepared_row = []
        for index, (fragment, b, display) in enumerate(ordered):
            x = max(0., (b[0]-pixels[0])*sx)
            right_limit = min(width, max(0., (ordered[index+1][1][0]-pixels[0])*sx)) if index+1 < len(ordered) else width
            align = style.get('align')
            if align in {'center', 'right'}:
                # Source alignment has an explicit glyph-box anchor. A font
                # wider than that observed line cannot preserve its center by
                # silently borrowing space from a neighboring fragment.
                right_limit = min(right_limit, (b[2]-pixels[0])*sx)
            available = right_limit-x
            if available <= _EPS:
                return None
            unit_advance = font.text_length(display, fontsize=1)
            required_sizes.append(available/unit_advance if unit_advance > 0 else requested)
            top = max(0., (b[1]-pixels[1])*sy)
            prepared_row.append((fragment, b, display, x, available, top))
        prepared.append((row_index, prepared_row))
    minimum_size = min(requested, max(7., requested*.85))
    size = max(minimum_size, min(requested, *required_sizes))
    if size < requested-1e-5:
        adaptations.append(dict(reason='source-common-font-width-fit', font_size=size))
    for row_index, prepared_row in prepared:
        row_values = []
        for fragment, b, display, x, available, top in prepared_row:
            advance = font.text_length(display, fontsize=size)
            if advance > available+_EPS:
                # Keep every canonical character and range, including the
                # fragment's trailing physical newline. Only this fragment
                # reflows; sibling rows retain their source anchors.
                fragment_text = text[fragment['start']:fragment['end']]
                wrapped = fallback(fragment_text, available, dict(style, font_size_pt=size, align='left'))
                wrapped_font = library.export_font_for_key(wrapped['font_key'])
                adaptations.append(dict(start=fragment['start'], reason='source-fragment-wrap',
                                        added_lines=len(wrapped['lines'])-1))
                for line in wrapped['lines']:
                    value = dict(line)
                    value.update(start=fragment['start']+line['start'], end=fragment['start']+line['end'],
                                 x=x+line['x'], source_top=top+line['baseline']-wrapped_font.ascender*size,
                                 font_key=wrapped['font_key'], font_size=size,
                                 height=(wrapped_font.ascender-wrapped_font.descender)*size)
                    value.pop('baseline', None)
                    row_values.append(value)
            else:
                if style.get('align') == 'center':
                    anchor = ((b[0]+b[2])/2-pixels[0])*sx
                    x = min(x+available-advance, max(x, anchor-advance/2))
                elif style.get('align') == 'right':
                    anchor = (b[2]-pixels[0])*sx
                    x = min(x+available-advance, max(x, anchor-advance))
                row_values.append(dict(text=display, start=fragment['start'], end=fragment['end'],
                                       x=x, source_top=top, font_size=size,
                                       height=metric_height*size))
            min_width = max(min_width, max(font.char_lengths(display, fontsize=size), default=0.))
        row_top = min(v['source_top'] for v in row_values)
        shift = max(0., previous_bottom-row_top)
        if shift > _EPS:
            adaptations.append(dict(row=row_index, reason='source-row-font-metrics-growth', shift_pt=shift))
        for value in row_values:
            top = value.pop('source_top')+shift
            line_font = library.export_font_for_key(value.get('font_key', face.key))
            value['baseline'] = top+line_font.ascender*value['font_size']
            previous_bottom = max(previous_bottom, top+value['height'])
            lines.append(value)
    # The canonical text is emitted once even for same-baseline fragments.
    lines.sort(key=lambda v: v['start'])
    occupied_height = max(v['baseline']-library.export_font_for_key(v.get('font_key', face.key)).descender*v['font_size']
                          for v in lines)
    height = max(source_height, occupied_height) if preserve_envelope else occupied_height
    return dict(width=width, height=height, font_key=face.key, font_size=size,
                line_step=size*max(float(style.get('line_height', 1.25)), metric_height),
                color=style.get('color', '#000000'), lines=lines, min_width=min_width,
                breaks=[v['end'] for v in lines], internal_layout=VERSION,
                source_envelope_preserved=bool(preserve_envelope),
                source_envelope_height=source_height, occupied_height=occupied_height,
                internal_adjustments=adaptations)
