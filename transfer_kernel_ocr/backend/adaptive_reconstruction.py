"""Feature four's canvas contract: V3 evidence -> typed blocks -> two-level flow.

The fixed scene editor and reading converter deliberately do not use this module.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image, ImageOps
import pymupdf as fitz

from .options import selected_pages
from .progress import report as progress
from .usage import tracked, span
from .stage_timing import capture_timings, timed_step, timing_summary

PLAN = 'feature4-text-linkage-2026-11-04'
MAX_PIXELS = 40_000_000
RASTER_LABELS = {'image', 'chart', 'seal', 'formula', 'unknown'}
HEADINGS = {'doc_title', 'paragraph_title', 'title', 'reference'}


def recognize_page(rgb, *, pixel_to_point=(.5, .5)):
    from .recognition import shared_structure
    return shared_structure().recognize_rebuild(rgb, pixel_to_point=pixel_to_point)


def _area(b):
    return max(0., b[2]-b[0])*max(0., b[3]-b[1])


def _overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))


def _bbox(value, width, height):
    try:
        p = np.asarray(value, dtype=float)
        if p.shape != (4,):
            p = np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        b = [max(0., min(width, float(p[0]))), max(0., min(height, float(p[1]))),
             max(0., min(width, float(p[2]))), max(0., min(height, float(p[3])))]
        return b if all(math.isfinite(v) for v in b) and _area(b) > .1 else None
    except (TypeError, ValueError, IndexError):
        return None


def _save_image(rgb, bbox, directory, name, *, exclusions=(), mask=None):
    x0, y0 = math.floor(bbox[0]), math.floor(bbox[1])
    x1, y1 = math.ceil(bbox[2]), math.ceil(bbox[3])
    image = Image.fromarray(rgb[y0:y1, x0:x1]).convert('RGBA')
    if exclusions or mask is not None:
        alpha = np.full((y1-y0, x1-x0), 255, dtype=np.uint8)
        if mask is not None:
            alpha[~mask[y0:y1, x0:x1]] = 0
        for other in exclusions:
            a, b = max(x0, math.floor(other[0])), max(y0, math.floor(other[1]))
            c, d = min(x1, math.ceil(other[2])), min(y1, math.ceil(other[3]))
            if c > a and d > b:
                alpha[b-y0:d-y0, a-x0:c-x0] = 0
        image.putalpha(Image.fromarray(alpha))
    relative = f'assets/{name}.png'
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return relative, list(image.size)


def _page_background(rgb, boxes, page, directory):
    """Keep pixels outside detected content at page level, never as obstacles."""
    h, w = rgb.shape[:2]
    covered = np.zeros((h, w), dtype=bool)
    for detection in boxes:
        b = detection['coordinate']
        covered[max(0, math.floor(b[1])):min(h, math.ceil(b[3])),
                max(0, math.floor(b[0])):min(w, math.ceil(b[2]))] = True
    mask = ~covered
    foreground = mask & (rgb.min(axis=2) < 255)
    page.pop('background', None)
    if not foreground.any():
        return
    asset, pixels = _save_image(rgb, [0, 0, w, h], directory,
                               page['id']+'-background', mask=mask)
    page['background'] = dict(asset=asset, pixel_size=pixels,
        bbox_pt=[0., 0., page['width_pt'], page['height_pt']],
        coordinate_space='source-page-points', policy='outside-detection-background-v1',
        foreground_pixel_count=int(foreground.sum()), participates_in_layout=False)


def _records(raw, boxes):
    ocr = raw.get('overall_ocr_res') or {}
    texts = ocr.get('rec_texts') or []
    coords = ocr.get('rec_boxes') or ocr.get('rec_polys') or []
    owners = ocr.get('block_ids') or []
    scores = ocr.get('rec_scores') or []
    line_ids = ocr.get('line_ids') or []
    labels = ocr.get('labels') or []
    styles = ocr.get('line_styles') or []
    records = defaultdict(list)
    for i, (text, coords_i) in enumerate(zip(texts, coords)):
        b = _bbox(coords_i, raw['width'], raw['height'])
        if not b or not str(text).strip():
            continue
        owner = owners[i] if i < len(owners) else None
        if owner is None:
            candidates = [(_overlap(b, v['coordinate'])/_area(b), -_area(v['coordinate']), v['id'])
                          for v in boxes if str(v.get('label', '')).lower() not in RASTER_LABELS]
            best = max(candidates, default=(0, 0, None))
            owner = best[2] if best[0] >= .5 else None
        if owner:
            record = dict(text=str(text), bbox_px=b,
                                       score=float(scores[i]) if i < len(scores) else None,
                                       line_id=str(line_ids[i]) if i < len(line_ids) else f'ocr-{i}',
                                       label=labels[i] if i < len(labels) else None,
                                       evidence_index=i)
            if i < len(styles) and isinstance(styles[i],dict):
                evidence=styles[i]
                if evidence.get('trusted') is True:
                    record['measurement_style']={k:evidence[k] for k in
                        ('font_size_pt','family','bold','italic') if k in evidence}
                    record['style_evidence']=copy.deepcopy(evidence)
            records[owner].append(record)
    return records


def _text_content(lines, block_id):
    from .source_text import build_text_content
    return build_text_content(lines, block_id)


def _style(label, lines, sy):
    height = statistics.median([r['bbox_px'][3]-r['bbox_px'][1] for r in lines]) if lines else 0
    estimate = height*sy*.78 if height else 11.
    size = max(9., min(14., estimate))
    if label == 'doc_title':
        size = max(16., min(24., estimate))
    elif label in HEADINGS:
        size = max(12., min(18., estimate))
    elif label in {'footnote', 'figure_title', 'header', 'footer', 'number', 'formula_number'}:
        size = max(8., min(11., estimate))
    return dict(font_size_pt=round(size, 2), family='sans-serif', bold=label in HEADINGS,
                italic=False, color='#000000', align='unknown', line_height=1.3, origin='role-template',
                observed_features=None, size_evidence='OCR-height-estimate-clamped-to-role-template')


@timed_step('05','typed_blocks')
def blocks_from_page(raw, rgb, page, directory, *, four_layer=False):
    """Preserve identity and raw evidence while adapting all branches alike."""
    w, h = page['width_px'], page['height_px']
    if (raw.get('width'), raw.get('height')) != (w, h):
        raise ValueError('功能四模型尺寸与页图不一致，拒绝猜测坐标')
    sx, sy = page['width_pt']/w, page['height_pt']/h
    boxes = []
    for i, value in enumerate((raw.get('layout_det_res') or {}).get('boxes') or []):
        b = _bbox(value.get('coordinate'), w, h)
        if b:
            boxes.append(dict(value, coordinate=b, id=str(value.get('id', f'layout-{i}'))))
    if len({b['id'] for b in boxes}) != len(boxes):
        raise ValueError('功能四版面检测 ID 重复，无法建立可靠对应关系')
    from .source_containment import VERSION, build_containment, block_metadata, conflict_is_internal
    existing = raw.get('source_containment', {})
    containment = (copy.deepcopy(existing) if existing.get('version') == VERSION and
                   existing.get('tolerance_px') == [1./sx, 1./sy] and
                   existing.get('source_rects') == {b['id']:b['coordinate'] for b in boxes}
                   else build_containment(boxes, pixel_to_point=(sx, sy)))
    nodes = containment['nodes']
    page['source_containment'] = containment
    for detection in boxes:
        detection['containment'] = copy.deepcopy(nodes[detection['id']])
    records = _records(raw, boxes)
    original_records = copy.deepcopy(records)
    if four_layer:
        # 05I is a complete raster block. Explicitly prevent even stale or
        # differently assigned OCR results inside it from becoming prose.
        images = [v for v in boxes if str(v.get('label','')).lower() in RASTER_LABELS]
        suppressed_lines = []
        for owner, lines in list(records.items()):
            kept = []
            for line in lines:
                image = next((v for v in images if v['id'] != owner and
                              _overlap(line['bbox_px'],v['coordinate']) >= .95*_area(line['bbox_px'])),None)
                if image:
                    suppressed_lines.append(dict(line_id=line.get('line_id'),block_id=owner,
                        image_id=image['id'],reason='05I-image-owned-text-not-reconstructed'))
                else:
                    kept.append(line)
            records[owner] = kept
        page['image_text_exclusions'] = suppressed_lines
        from .four_layer_text import deduplicate_records, measure_record
        indexed = [dict(line,_owner=owner) for owner,lines in records.items()
                   if not any(v['id']==owner for v in images) for line in lines]
        styles = {owner:_style(next(v['label'] for v in boxes if v['id']==owner),lines,sy)
                  for owner,lines in records.items()}
        # Alpha uses source intersection in pt, divided by the existing rebuilt
        # character-area estimate. Prefix advances still use the measured font.
        candidates = {i for i,a in enumerate(indexed) for j,b in enumerate(indexed)
                      if i != j and _overlap(a['bbox_px'],b['bbox_px']) > 0}
        indexed = [measure_record(line,styles[line['_owner']],sx,sy) if i in candidates else line
                   for i,line in enumerate(indexed)]
        cleaned,checks=deduplicate_records(indexed)
        for owner in {line['_owner'] for line in indexed}:
            records[owner]=[]
        for line in cleaned:
            owner=line.pop('_owner'); records[owner].append(line)
        page['source_overlap_checks']=checks
    tables = {str(t.get('block_id')): t for t in raw.get('table_res_list') or []}
    suppressed = {str(d.get('block_id')) for d in raw.get('diagnostics', [])
                  if isinstance(d, dict) and d.get('reason') == 'fully-owned-by-another-block'}
    conflict_reasons = {'ocr-line-crosses-layout-blocks','ocr-line-crosses-visual-block',
                        'ocr-line-unmatched-block','ocr-line-recognition-rejected','empty-ocr-line-crop'}
    conflicts = {str(identifier) for d in raw.get('diagnostics', []) if isinstance(d, dict) and
                 d.get('reason') in conflict_reasons
                 for identifier in [d.get('block_id'), *d.get('candidate_ids', [])] if identifier
                 and not conflict_is_internal(d, str(identifier), nodes)}
    page['non_rendering_containers'] = []
    empty_parents = []
    blocks = []
    # These pure parsers validate actual HTML topology and preserve empty cells.
    from .table_structure import _table_structure
    for i, detection in enumerate(boxes):
        local_id, b = detection['id'], detection['coordinate']
        if local_id in suppressed:
            continue
        label = str(detection.get('label', 'unknown')).lower()
        identifier = f'{page["id"]}-{local_id}'
        crop, pixels = _save_image(rgb, b, directory, identifier)
        ownership = copy.deepcopy((raw.get('block_ownership') or {}).get(local_id, {}))
        exclusions = ownership.get('excluded_rects', [])
        lines = records.get(local_id, [])
        if four_layer and lines and not any(r.get('text','').strip() for r in lines):
            page.setdefault('suppressed_duplicate_text_blocks',[]).append(local_id)
            continue
        block = dict(id=identifier, kind='text',
                     source=dict(page_id=page['id'], layout_detection_id=local_id, label=label,
                                 bbox_px=b, bbox_pt=[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
                                 containment=block_metadata(nodes[local_id], page['id']),
                                 score=detection.get('score'), reference_asset=crop,
                                 crop_to_page=[[1, 0, math.floor(b[0])], [0, 1, math.floor(b[1])], [0, 0, 1]],
                                 ownership=ownership,
                                 ocr_lines=copy.deepcopy(original_records.get(local_id,lines))),
                     content={}, style=_style(label, lines, sy), local_layout={},
                     quality=dict(state='needs_review', issues=[]))
        if local_id in conflicts and label not in RASTER_LABELS:
            block['kind'] = 'image'
            block['quality'].update(state='image_fallback', issues=['ambiguous-ocr-line-ownership'])
        elif label == 'table':
            result = tables.get(local_id)
            excluded_ids = ownership.get('excluded_block_ids', [])
            internal_exclusions = bool(excluded_ids) and len(excluded_ids) == len(exclusions) and all(
                i in nodes[local_id]['contained_ids'] for i in excluded_ids)
            parsed = _table_structure(result, dict(id=identifier, bbox=b, text=''), sx, sy, w, h) if result and (not exclusions or internal_exclusions) else None
            if parsed:
                if result.get('reconstruction_evidence'):
                    block['source']['table_reconstruction'] = dict(
                        evidence=copy.deepcopy(result['reconstruction_evidence']),
                        crop_origin=copy.deepcopy(result.get('crop_origin')),
                        raw_model_pred_html=result.get('raw_model_pred_html'),
                        raw_model_cell_box_list=copy.deepcopy(result.get('raw_model_cell_box_list')))
                    block['quality']['issues'].append('table-geometry-reconciled-from-row-bands')
                cells = []
                for cell in parsed['cells']:
                    cells.append(dict(id=cell['id'], row=cell['row'], column=cell['column'],
                                      rowspan=cell['rowspan'], colspan=cell['colspan'],
                                      text=cell.get('htmlText', ''), header=cell.get('header', False),
                                      source_bbox=cell['bbox']))
                block.update(kind='table', content=dict(row_count=len(parsed['rows'])-1,
                    column_count=len(parsed['columns'])-1, cells=cells,
                    column_widths=[z-a for a,z in zip(parsed['columns'],parsed['columns'][1:])],
                    column_ratios=[(z-a)/(b[2]*sx-b[0]*sx) for a, z in zip(parsed['columns'], parsed['columns'][1:])],
                    reference_asset=crop, pred_html=result.get('pred_html', '')))
                block['quality']['issues'].append('model-table-topology-needs-review')
                if internal_exclusions:
                    block['quality']['issues'].append('contained-content-kept-as-separate-source-blocks')
                from .source_table import enrich_table_content
                enrich_table_content(block['content'], block['source'], parsed)
                if parsed['geometryStatus'] == 'estimated-from-html-topology':
                    block['quality']['issues'].append('source-cell-geometry-estimated')
            else:
                block['quality'].update(state='image_fallback', issues=[
                    'overlapping-table-preserved-as-image' if exclusions else 'unreliable-table-structure'])
                block['kind'] = 'image'
        elif label in RASTER_LABELS:
            block['kind'] = 'image'
            if label != 'image':
                block['quality']['issues'].append('unsupported-specialized-content-preserved-as-image')
        else:
            from .source_text import build_text_content
            block['content'] = dict(build_text_content(lines,identifier,four_layer=four_layer,deduplicate=False),role=label)
            if four_layer:
                block['content']['source_overlap_checks']=[c for c in page['source_overlap_checks']
                    if c['a'] in {r.get('line_id') for r in lines} or c['b'] in {r.get('line_id') for r in lines}]
            if four_layer and not lines and any(v['block_id']==local_id for v in page['image_text_exclusions']):
                page.setdefault('suppressed_image_text_blocks',[]).append(local_id)
                continue
            if not block['content']['text'].strip():
                if nodes[local_id]['contained_ids']:
                    # Decide after all children have been built. An empty
                    # geometric container is not a failed text recognition.
                    empty_parents.append(block)
                    continue
                exclusions = exclusions or [v['coordinate'] for v in boxes if v['id'] != local_id and
                              (str(v.get('label', '')).lower() in RASTER_LABELS | {'table'} or
                               _area(v['coordinate']) < _area(b))]
                crop, pixels = _save_image(rgb, b, directory, identifier+'-fallback', exclusions=exclusions)
                if any(_overlap(b, other) >= _area(b)*.99 for other in exclusions):
                    block['quality']['issues'].append('fully-covered-region-not-emitted')
                    continue
                block.update(kind='image')
                block['quality'].update(state='image_fallback', issues=['no-reliable-ocr-text'])
        if block['kind'] == 'image':
            if exclusions:
                crop, pixels = _save_image(rgb, b, directory, identifier+'-owned', exclusions=exclusions)
            block['content'] = dict(asset=crop, pixel_size=pixels, aspect_ratio=pixels[0]/pixels[1],
                transcript=copy.deepcopy(lines), transcript_rendering='metadata-only',
                transcript_coordinate_space='page-pixels')
        blocks.append(block)
    visible = {b['id'] for b in blocks}
    for block in sorted(empty_parents, key=lambda b: _area(b['source']['bbox_px'])):
        children = [i for i in block['source']['containment']['contained_ids'] if i in visible]
        if children:
            page['non_rendering_containers'].append(dict(
                id=block['id'], source=copy.deepcopy(block['source']),
                visible_contained_ids=children, participates_in_layout=False,
                reason='source-container-content-owned-by-children'))
        else:
            b = block['source']['bbox_px']
            crop, pixels = _save_image(rgb, b, directory, block['id']+'-fallback')
            block.update(kind='image', content=dict(asset=crop, pixel_size=pixels,
                aspect_ratio=pixels[0]/pixels[1], transcript=[], transcript_rendering='metadata-only',
                transcript_coordinate_space='page-pixels'))
            block['quality'].update(state='image_fallback', issues=['no-reliable-ocr-text'])
            blocks.append(block)
            visible.add(block['id'])
    page['detections'] = dict(layout=boxes, regions=(raw.get('region_det_res') or {}).get('boxes') or [])
    if blocks:
        from .source_geometry import organize_source
        def make_text(lines,identifier):
            from .source_text import build_text_content
            return build_text_content(lines,identifier,four_layer=four_layer)
        blocks = organize_source(blocks,rgb,page,raw,directory,_save_image,make_text,
                                 preserve_image_blocks=four_layer)
        coverage = boxes + [dict(coordinate=b['source']['bbox_px']) for b in blocks
                            if b['source'].get('label') == 'container']
        coverage.extend(dict(coordinate=line['bbox_px']) for lines in original_records.values() for line in lines)
        _page_background(rgb, coverage, page, directory)
    if not blocks and not four_layer:
        crop, pixels = _save_image(rgb, [0, 0, w, h], directory, page['id']+'-unclassified')
        blocks.append(dict(id=page['id']+'-unclassified', kind='image',
            source=dict(page_id=page['id'], layout_detection_id=None, label='unknown', bbox_px=[0,0,w,h],
                        bbox_pt=[0,0,page['width_pt'],page['height_pt']], reference_asset=crop,
                        score=None, ocr_lines=[], ownership={}, crop_to_page=[[1,0,0],[0,1,0],[0,0,1]]),
            content=dict(asset=crop,pixel_size=pixels,aspect_ratio=w/h), style=_style('image',[],sy),
            local_layout={}, quality=dict(state='image_fallback',issues=['no-layout-detections'])))
    if not blocks and four_layer:
        _page_background(rgb,[],page,directory)
        page['recognition_incomplete'] = 'no-layout-detections'
    page['detections'] = dict(layout=boxes, regions=(raw.get('region_det_res') or {}).get('boxes') or [])
    page['suppressed_detection_ids'] = sorted(suppressed)
    page['recognition'] = dict(service=raw.get('_service', {}), policy=raw.get('rebuild_policy'),
                               ocr_pipeline=raw.get('ocr_pipeline', {}),
                               diagnostics=raw.get('diagnostics', []))
    return blocks


def derive_relations(project):
    """07: finalize identities, spatial evidence and the reading-order graph."""
    from .spatial_relations import derive_spatial_relations
    return derive_spatial_relations(project)


def _input_pages(paths, spec):
    selections = []
    total = 0
    for path in paths:
        if path.suffix.lower() == '.pdf':
            with fitz.open(path) as pdf:
                if not pdf.is_pdf or pdf.needs_pass:
                    raise ValueError('重建需要未加密的 PDF 文件')
                selection = selected_pages(spec, len(pdf))
        elif path.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.webp','.bmp'}:
            with Image.open(path) as image:
                selection = selected_pages(spec, getattr(image,'n_frames',1))
        else:
            raise ValueError('重建仅支持 PDF 和常见图片格式')
        total += len(selection)
        if total > 1000:
            raise ValueError('每次重建最多1000页/帧')
        selections.append(selection)
    if not total:
        raise ValueError('没有选择可重建页面')
    return selections, total


@tracked('rebuild.adaptive_convert', feature=lambda v: 'rebuild.'+v.get('format','pdf'))
@capture_timings
def convert_adaptive(paths, output_dir, document_id, *, format='pdf', pages='', layout_options=None, on_page=None):
    if format not in {'pdf','html','md'}:
        raise ValueError('结构化重建仅支持 PDF、HTML 或 Markdown')
    paths, directory = [Path(p) for p in paths], Path(output_dir)
    selections, total = _input_pages(paths,pages)
    directory.mkdir(parents=True,exist_ok=True)
    project = dict(schema='feature4-rebuild/1', id=document_id, plan=PLAN,
                   source_pages=[], blocks=[], relations={}, layout_result={},
                   layout_options={**dict(four_layer=True,empty_bands=False,region_expand_ratio=1.05,
                                         text_adaptation_version='20261104'),
                                   **(layout_options or {})},
                   limitations=['OCR and model topology require review.',
                                'Style values are role templates, not recovered original typography.',
                                'Source geometry is preserved where feasible; local growth stays within isolated frames.',
                                'Region membership and reading-order evidence still require review.'])
    from .adaptive_export import export_project, _html_document
    from .region_recovery import layout_with_region_recovery
    completed = []
    ordinal, started = 0, time.perf_counter()
    for file_index, (path, selection) in enumerate(zip(paths,selections)):
        for physical in selection:
            ordinal += 1
            progress(5+65*(ordinal-1)/total,'V3分类与文字/表格解析',current=ordinal,total=total)
            if path.suffix.lower() == '.pdf':
                with fitz.open(path) as pdf:
                    page = pdf[physical]
                    width_pt,height_pt = page.rect.width,page.rect.height
                    scale = min(2., math.sqrt(MAX_PIXELS/(width_pt*height_pt))*.999)
                    pix = page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False,colorspace=fitz.csRGB)
                    rgb = np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3).copy()
            else:
                with Image.open(path) as image:
                    image.seek(physical)
                    if image.width*image.height > MAX_PIXELS:
                        raise ValueError('图片超过4000万像素限制，请先降低分辨率')
                    rgb = np.asarray(ImageOps.exif_transpose(image.copy()).convert('RGB')).copy()
                height,width = rgb.shape[:2]
                width_pt,height_pt = width/2,height/2
            h,w = rgb.shape[:2]
            source = dict(id=f'source-{ordinal}',file_index=file_index,physical_page_index=physical,
                          width_px=w,height_px=h,width_pt=float(width_pt),height_pt=float(height_pt),
                          pixel_to_point=[width_pt/w,height_pt/h])
            source['reference_asset'] = _save_image(rgb,[0,0,w,h],directory,source['id'])[0]
            with span('rebuild.v3_branches'):
                raw = (recognize_page(rgb) if source['pixel_to_point']==[.5,.5] else
                       recognize_page(rgb,pixel_to_point=source['pixel_to_point']))
            with span('rebuild.typed_blocks'):
                project['blocks'].extend(blocks_from_page(raw,rgb,source,directory,
                                                        four_layer=True))
            project['source_pages'].append(source)
            if on_page is not None:
                # Layout each page exactly once. No cross-page reflow is permitted
                # by the source-preserving Region policy.
                local = {**project, 'source_pages':[source],
                         'blocks':[b for b in project['blocks'] if b['source']['page_id']==source['id']],
                         'relations':{}, 'layout_result':{}}
                local.pop('region_fallbacks', None)
                with span('rebuild.relations'):
                    local['relations'] = derive_relations(local)
                with span('rebuild.adaptive_layout'):
                    local['layout_result'] = layout_with_region_recovery(local, directory)
                preview = directory/'previews'/f'page-{ordinal}.html'
                preview.parent.mkdir(exist_ok=True)
                temporary = preview.with_suffix('.tmp')
                temporary.write_text(_html_document(local), encoding='utf-8')
                temporary.replace(preview)
                for record in local.get('region_fallbacks',[]):
                    record['selected_page_number'] = ordinal
                completed.append(local)
                on_page(dict(number=ordinal,total=total,file_index=file_index,
                             layout_needs_review=any(i.get('action')=='export-current-geometry-and-continue'
                                 for i in local['layout_result'].get('issues',[])),
                             physical_page=physical+1,preview=preview.relative_to(directory).as_posix()))
    if on_page is not None:
        project['blocks'] = [b for page in completed for b in page['blocks']]
        project['relations'] = _merge_page_records([p['relations'] for p in completed])
        layouts = []
        offset = 0
        order_offset = 0
        for page in completed:
            layout = page['layout_result']
            for record in layout['pages']:
                record['index'] += offset
            for record in layout['placements'] + layout.get('frames',[]):
                record['page_index'] += offset
            for change in layout.get('order_changes',[]):
                change['source_index'] += order_offset
                change['target_index'] += order_offset
            order_offset += len(layout['resolved_order'])
            offset += len(layout['pages'])
            layouts.append(layout)
        project['layout_result'] = _merge_page_records(layouts)
        project['region_fallbacks'] = [r for p in completed for r in p.get('region_fallbacks',[])]
    else:
        with span('rebuild.relations'):
            project['relations'] = derive_relations(project)
        progress(72,'两层自适应排版','保留源位置，在区域内检查碰撞与等比适配')
        with span('rebuild.adaptive_layout'):
            project['layout_result'] = layout_with_region_recovery(project, directory)
    if project.get('region_fallbacks'):
        if on_page is not None:
            (directory/'region-overflow.json').write_text(json.dumps(dict(schema='region-overflow/1',
                document_id=document_id,output_dir=str(directory.resolve()),regions=project['region_fallbacks']),
                ensure_ascii=False,indent=2),encoding='utf-8')
        project['limitations'].append('Overflow regions retain original pixels; their content is not editable reconstruction. See region-overflow.json.')
    report = dict(schema='feature4-conversion/1',plan=PLAN,format=format,status='needs-review',
                  layoutElapsedSeconds=round(time.perf_counter()-started,6),
                  sourcePageCount=total,pageCount=len(project['layout_result']['pages']),
                  independentTextLines=sum(len(b['source'].get('ocr_lines',[])) for b in project['blocks'] if b['kind']=='text'),
                  tableCount=sum(b['kind']=='table' for b in project['blocks']),
                  imageCount=sum(b['kind']=='image' for b in project['blocks']),
                  backgroundPageCount=sum(bool(p.get('background')) for p in project['source_pages']),
                  compositeCount=sum(b['kind']=='composite' for b in project['blocks']),
                  blockCount=len(project['blocks']),orderChanges=project['layout_result'].get('order_changes',[]),
                  recognitionBackend='v3',strategy='typed-blocks-two-level-adaptive',
                  relationsVersion=project['relations'].get('algorithm_version'),
                  layoutVersion=project['layout_result'].get('algorithm_version'),
                  regionAdjustments=project['layout_result'].get('region_adjustments',[]),
                  layoutOptions=project['layout_options'],
                  stageTimings=timing_summary(),
                  paragraphAdjustments=paragraph_adjustments(project['layout_result']),
                  regionFallbacks=project.get('region_fallbacks', []),
                  unresolvedLayouts=[i for i in project['layout_result'].get('issues',[])
                                     if i.get('action')=='export-current-geometry-and-continue'],
                  limitations=project['limitations'])
    if any(b['quality']['state']=='image_fallback' for b in project['blocks']):
        report['status']='partial'
    if report['unresolvedLayouts']:
        report['status']='layout-needs-review'
    project['report']=report
    progress(85,'导出重建文档','写入真实文字、表格与图像资源')
    with span('rebuild.adaptive_export'):
        destination, diagnostics = export_project(project,directory,format=format)
    (directory/'conversion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    progress(100,'转换完成')
    return destination,report


def paragraph_adjustments(layout):
    """Include descendants and later Region scaling in searchable evidence."""
    records=[]
    def walk(identifier,measurement,frame_id):
        for state in measurement.get('paragraph_adjustments',[]):
            row=copy.deepcopy(state)
            factor=measurement.get('uniform_scale',1.)
            row.update(block_id=identifier,frame_id=frame_id,region_scale=factor,
                       paragraph_region_scaled=factor<1.-1e-8,
                       paragraph_any_scaled=bool(row.get('paragraph_font_scaled')) or factor<1.-1e-8)
            records.append(row)
        for child in measurement.get('children',[]):
            walk(child['block_id'],child['measurement'],frame_id)
    for p in layout.get('placements',[]):
        walk(p['block_id'],p['measurement'],p.get('frame_id'))
    return records


def _merge_page_records(records):
    """Merge independent source-page evidence without changing IDs or geometry."""
    result = {}
    for record in records:
        for key,value in record.items():
            if isinstance(value,list):
                result.setdefault(key,[]).extend(value)
            elif isinstance(value,dict):
                result.setdefault(key,{}).update(value)
            elif key not in result:
                result[key] = value
            elif result[key] != value:
                raise ValueError(f'不兼容的逐页排版元数据: {key}')
    return result
