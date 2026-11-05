"""Locate measured font reductions without confusing OCR estimates with truth."""
from .font_library import get_library


def font_size_audit(project):
    blocks = {}

    def index(items):
        for block in items:
            blocks[block['id']] = block
            index(block.get('content', {}).get('children', []))

    index(project['blocks'])
    pages = {p['id']: p for p in project.get('source_pages', [])}
    rows = []
    library = get_library()

    def collect(block, measurement, x, y, page_index, cell_content=None):
        source = block.get('source', {})
        page = pages.get(source.get('page_id'), {})
        content = cell_content or block.get('content', {})
        fragments = {f.get('line_id'): f for f in content.get('source_layout', {}).get('fragments', [])}
        originals = {f.get('line_id'): f for f in source.get('ocr_lines', [])}
        sx, sy = page.get('pixel_to_point', [1., 1.])
        for line_index, line in enumerate(measurement.get('lines', [])):
            identity = line.get('line_id')
            fragment = fragments.get(identity, {})
            final = float(line.get('font_size', measurement.get('font_size', 11.)))
            initial = float(fragment.get('measurement_style', {}).get('font_size_pt',
                            (cell_content or {}).get('style', {}).get('font_size_pt',
                            block.get('style', {}).get('font_size_pt', final))))
            font = library.export_font_for_key(line.get('font_key', measurement.get('font_key')))
            left = x + line['x']
            top = y + line['baseline'] - font.ascender * final
            width = line.get('advance_width', font.text_length(line['text'], fontsize=final))
            raw = originals.get(identity, fragment).get('bbox_px')
            original_box = [raw[0]*sx, raw[1]*sy, raw[2]*sx, raw[3]*sy] if raw else None
            original_basis = 'original-ocr-line' if raw else 'unavailable'
            if original_box is None and cell_content and cell_content.get('source_bbox'):
                original_box = list(cell_content['source_bbox'])
                original_basis = 'source-cell-envelope-not-individual-line'
            paragraph = next((p for p in measurement.get('paragraph_adjustments', [])
                              if p.get('paragraph_id') == line.get('paragraph_id')), {})
            reasons = []
            if paragraph.get('font_scale', 1.) < 1.-1e-8:
                reasons.append('paragraph-font-scaling')
            if measurement.get('uniform_scale', 1.) < 1.-1e-8:
                reasons.append('block-or-region-uniform-scaling')
            if any(a.get('type') == 'residual-collision-size-normalization' and a.get('line_id') == identity
                   for a in measurement.get('internal_adjustments', [])):
                reasons.append('residual-collision-size-normalization')
            shrunk = final < initial-1e-7
            if shrunk and not reasons:
                reasons.append('measured-size-reduction')
            rows.append(dict(page_number=page_index+1, source_page_number=page.get('physical_page_index', page_index)+1,
                             block_id=block['id'], line_id=identity, cell_id=(cell_content or {}).get('id'),
                             measured_line_index=line_index, text=line['text'],
                             original_bbox_pt=original_box,
                             original_bbox_basis=original_basis,
                             output_bbox_pt=[left, top, left+width, top+line['height']],
                             initial_font_size_pt=initial, final_font_size_pt=final,
                             font_scale=final/initial, shrunk=shrunk, reasons=reasons,
                             paragraph_font_scale=paragraph.get('font_scale', 1.),
                             uniform_scale=measurement.get('uniform_scale', 1.),
                             initial_size_basis='reconstruction-style-estimate-not-source-font-truth'))
        for cell in measurement.get('cells', []):
            original = next((c for c in block.get('content', {}).get('cells', []) if c['id']==cell['id']), {})
            collect(block, cell, x+cell['x'], y+cell['y'], page_index, original)
        for child in measurement.get('children', []):
            collect(blocks.get(child.get('block_id', child.get('id')), block), child['measurement'],
                    x+child['x'], y+child['y'], page_index)

    for placement in project['layout_result']['placements']:
        collect(blocks[placement['block_id']], placement['measurement'],
                *placement['bbox'][:2], placement['page_index'])
    return dict(schema='font-size-audit/1', coordinate_space='output-page-points-top-left',
                comparison='initial-reconstruction-style-vs-final-measured-size',
                line_count=len(rows), shrunk_line_count=sum(r['shrunk'] for r in rows), lines=rows)
