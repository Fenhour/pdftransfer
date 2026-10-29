"""Measured cell text flow for model HTML values without physical OCR lines.

Physical OCR line boxes use the existing bounded visual font fitter. A table
HTML fallback has a cell box, rather than a text ink box: fitting its height as
one large line would enlarge or squash the text. Wrap only those fallback
values using safe font metrics, then reduce the actual font size if needed.
"""
from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
import math

from .font_library import get_library


def is_cell_flow(element):
    return (bool(element.get('tableId') and element.get('textOrigin') == 'table-html') or
            element.get('textOrigin') == 'layout-block')


def flow_cell_text(element):
    """Return a safe font and bounded lines, without changing recognized text.

    Text advances are measured once and reused by at most twelve size
    trials. No font raster matching, extra OCR, horizontal stretching or
    truncation is performed. The result follows current edits and geometry.
    """
    text = element.get('text', '')
    library = get_library()
    bold = str(element.get('fontWeight', 'normal')) in {'bold', '600', '700', '800', '900'}
    italic = element.get('fontStyle') in {'italic', 'oblique'}
    face = library.resolve(text, element.get('resolvedFontFamily') or element.get('fontFamily', ''),
                           bold, italic, safe_only=True)
    if face is None:
        raise ValueError('没有覆盖表格文字的安全字体，未截断或栅格化单元格内容')
    font = library.export_font_for_key(face.key)
    width, height = float(element['width']), float(element['height'])
    if min(width, height) <= 0:
        raise ValueError('表格文字可用区域无效')
    # Each paragraph is measured once at 1 pt; all later size trials reuse it.
    paragraphs = []
    widest_char=0.
    for paragraph in text.split('\n'):
        weights = font.char_lengths(paragraph, fontsize=1)
        widest_char=max(widest_char,max(weights,default=0.))
        paragraphs.append((paragraph, [0., *accumulate(weights)]))
    line_height = max(float(element.get('lineHeight', 1.2)), font.ascender-font.descender)

    def lines_at(size):
        lines = []
        for paragraph, prefix in paragraphs:
            if not paragraph:
                lines.append('')
                continue
            start = 0
            while start < len(paragraph):
                end = max(start+1, bisect_right(prefix, prefix[start]+width/size+1e-8)-1)
                end = min(len(paragraph), end)
                if end < len(paragraph):
                    # Prefer a real word boundary. A long token/CJK sequence
                    # can still break at a character instead of overflowing.
                    boundary = max(paragraph.rfind(' ', start, end), paragraph.rfind('\t', start, end))
                    if boundary > start:
                        end = boundary+1
                lines.append(paragraph[start:end])
                start = end
        return lines

    if widest_char>width:
        raise ValueError('表格文字区域窄于最小可用字形，请扩大单元格或校对识别结果')
    requested = max(1., min(1000., float(element.get('fontSize', 12.)),
                            width/max(.001,widest_char)))
    lines = lines_at(requested)
    size = requested
    if len(lines)*line_height*size > height+1e-6:
        low, high = min(1., requested), requested
        if len(lines_at(low))*line_height*low > height+1e-6:
            raise ValueError('表格文字无法在单元格范围内完整排版，请扩大单元格或校对识别结果')
        for _ in range(12):
            middle = (low+high)/2
            if len(lines_at(middle))*line_height*middle <= height:
                low = middle
            else:
                high = middle
        size = low
        lines = lines_at(size)
    step = line_height*size
    # The metric line box remains inside the cell. Baseline is relative to
    # the element, shared by HTML and PDF paths.
    baseline = font.ascender*size
    result = dict(lines=lines, fontSize=size, lineStep=step, baseline=baseline,
                  family=face.family, fontKey=face.key, approximate=True,
                  method='safe-metric-cell-wrap', shrunk=size < requested-.01)
    if not all(math.isfinite(result[key]) for key in ('fontSize', 'lineStep', 'baseline')):
        raise ValueError('无效的表格文字排版尺寸')
    return font, result
