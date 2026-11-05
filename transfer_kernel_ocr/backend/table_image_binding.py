"""Map contained raster objects through source and measured table cells."""
from __future__ import annotations
from copy import deepcopy


def bind_table_images(blocks, placements):
    from .regional_layout import image_source_bbox, _signature
    by_id = {b['id']: b for b in blocks}
    tables = [p for p in placements if by_id[p['block_id']]['kind'] == 'table']
    for placement in placements:
        block = by_id[placement['block_id']]
        if block['kind'] != 'image':
            continue
        source = image_source_bbox(block)
        area = (source[2]-source[0])*(source[3]-source[1])
        candidates = []
        for table in tables:
            parent = by_id[table['block_id']]
            if (table['page_index'] != placement['page_index'] or
                    parent['source'].get('page_id') != block['source'].get('page_id')):
                continue
            measured = {c['id']: c for c in table['measurement'].get('cells', [])}
            overlaps = []
            for cell in parent['content'].get('cells', []):
                rect = cell.get('source_bbox')
                if not rect or cell['id'] not in measured:
                    continue
                intersection = (max(0., min(source[2], rect[2])-max(source[0], rect[0])) *
                                max(0., min(source[3], rect[3])-max(source[1], rect[1])))
                coverage = intersection/area
                if coverage > .01:
                    overlaps.append((coverage, cell, measured[cell['id']]))
                if coverage >= .8:
                    candidates.append((coverage, table, cell, measured[cell['id']]))
            # Some detections contain several vertically stacked formulae.
            # Bind the intact raster to the contiguous cell range, never pick
            # just the largest-overlap row and discard its other rows.
            if (not any(c[1] is table for c in candidates) and len(overlaps) > 1 and
                    sum(c[0] for c in overlaps) >= .95):
                cells = [c[1] for c in overlaps]
                same_column = len({(c['column'], c.get('colspan', 1)) for c in cells}) == 1
                ordered = sorted(cells, key=lambda c: c['row'])
                contiguous = all(a['row']+a.get('rowspan', 1) == b['row'] for a,b in zip(ordered, ordered[1:]))
                if same_column and contiguous:
                    def envelope(rects):
                        return [min(r[0] for r in rects), min(r[1] for r in rects),
                                max(r[2] for r in rects), max(r[3] for r in rects)]
                    rect = envelope([c['source_bbox'] for c in cells])
                    target_rect = envelope([[m['x'],m['y'],m['x']+m['width'],m['y']+m['height']]
                                            for _,_,m in overlaps])
                    cell = dict(ordered[0], id=':'.join(c['id'] for c in ordered), source_bbox=rect,
                                bound_cell_ids=[c['id'] for c in ordered])
                    target = dict(x=target_rect[0], y=target_rect[1],
                                  width=target_rect[2]-target_rect[0], height=target_rect[3]-target_rect[1])
                    candidates.append((sum(c[0] for c in overlaps), table, cell, target))
        # A cell must be established by geometry, not by the formula label.
        # Spanning or ambiguous objects retain their existing placement.
        if len(candidates) != 1:
            if candidates:
                placement['table_cell_binding'] = dict(status='ambiguous-source-cell')
            continue
        coverage, table, cell, target = candidates[0]
        rect = cell['source_bbox']
        sx = target['width']/(rect[2]-rect[0])
        sy = target['height']/(rect[3]-rect[1])
        factor = min(1., sx, sy)
        # The cell centre and local normalised anchor follow row/column changes;
        # a single scale preserves the raster's aspect ratio.
        cx = ((source[0]+source[2])/2-rect[0])/(rect[2]-rect[0])
        cy = ((source[1]+source[3])/2-rect[1])/(rect[3]-rect[1])
        width, height = (source[2]-source[0])*factor, (source[3]-source[1])*factor
        x = table['bbox'][0]+target['x']+cx*target['width']-width/2
        y = table['bbox'][1]+target['y']+cy*target['height']-height/2
        previous = placement['measurement'].get('uniform_scale', 1.)
        measurement = deepcopy(placement['measurement'])
        for key in ('width', 'height', 'available_width', 'min_width'):
            if key in measurement:
                measurement[key] *= factor/previous
        measurement.update(width=width, height=height, uniform_scale=factor)
        measurement['signature'] = _signature({k:v for k,v in measurement.items()
                                               if k not in {'signature', 'scaling_result_signature'}})
        measurement['scaling_result_signature'] = measurement['signature']
        placement['measurement'] = measurement
        placement.update(bbox=[x, y, x+width, y+height],
                         source_transform=[factor, 0., 0., factor, x-source[0]*factor, y-source[1]*factor],
                         frame_id=table['frame_id'])
        binding = dict(status='bound', table_id=table['block_id'], cell_id=cell['id'],
                       cell_ids=cell.get('bound_cell_ids', [cell['id']]),
                       row=cell['row'], column=cell['column'], source_coverage=coverage,
                       source_cell_bbox=list(rect), normalized_center=[cx, cy],
                       target_cell_bbox=[table['bbox'][0]+target['x'], table['bbox'][1]+target['y'],
                                         table['bbox'][0]+target['x']+target['width'],
                                         table['bbox'][1]+target['y']+target['height']],
                       scale_policy='uniform-at-normalized-cell-anchor')
        if len(binding['cell_ids']) > 1:
            binding['limitation'] = 'intact-multi-cell-raster-retains-internal-row-spacing'
        block['source']['table_cell_binding'] = dict(binding)
        placement['table_cell_binding'] = binding
