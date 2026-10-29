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

PLAN = 'feature4-v3-canvas-2026-10-28-internal-structure'
MAX_PIXELS = 40_000_000
RASTER_LABELS = {'image', 'chart', 'seal', 'formula', 'unknown'}
HEADINGS = {'doc_title', 'paragraph_title', 'title', 'reference'}


def recognize_page(rgb):
    from .recognition import shared_structure
    return shared_structure().recognize_rebuild(rgb)


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


def _unclassified_blocks(rgb, boxes, page, directory):
    """Keep visible undetected content as review images, never invented OCR.

    This is a conservative pixel coverage check, not another classifier.
    Two pixels around detected crops absorb antialiasing at their edges.
    Thin adjacent table borders stay in the source reference instead of
    becoming standalone flow images. Unclassified text remains visible.
    """
    import cv2
    h, w = rgb.shape[:2]
    covered = np.zeros((h, w), dtype=np.uint8)
    for detection in boxes:
        b = detection['coordinate']
        covered[max(0, math.floor(b[1])-2):min(h, math.ceil(b[3])+2),
                max(0, math.floor(b[0])-2):min(w, math.ceil(b[2])+2)] = 1
    foreground = (rgb.min(axis=2) < 245) & ~covered.astype(bool)
    # Group nearby strokes into local review regions; ignore isolated specks.
    merged = cv2.dilate(foreground.astype(np.uint8), np.ones((7, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    regions = []
    for index in range(1, count):
        x, y, width, height, _ = stats[index]
        local = (labels[y:y+height, x:x+width] == index) & foreground[y:y+height, x:x+width]
        if int(local.sum()) >= 8:
            yy,xx = np.nonzero(local)
            xx,yy = xx+x,yy+y
            # Require a genuinely thin stroke, not just a narrow dilation box.
            fw,fh = int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1)
            if min(fw,fh) <= 12 and max(fw,fh) >= max(40,8*min(fw,fh)):
                owner = None
                for detection in boxes:
                    if detection.get('label') != 'table':
                        continue
                    a,b,c,d = detection['coordinate']
                    if (fh>fw and fh < (d-b)*.25) or (fw>=fh and fw < (c-a)*.25):
                        continue
                    near_vertical = ((np.abs(xx-a)<=12)|(np.abs(xx-c)<=12)) & (yy>=b-2) & (yy<=d+2)
                    near_horizontal = ((np.abs(yy-b)<=12)|(np.abs(yy-d)<=12)) & (xx>=a-2) & (xx<=c+2)
                    if np.all(near_vertical|near_horizontal):
                        owner=detection['id']
                        break
                if owner is not None:
                    page.setdefault('coverage_diagnostics',[]).append(dict(
                        reason='table-edge-decoration-retained-in-reference',block_id=owner,
                        bbox_px=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
                        pixel_count=int(local.sum())))
                    continue
            regions.append((int(index), [int(x), int(y), int(x+width), int(y+height)]))
    # Bound pathological noisy scans without silently dropping the remainder.
    groups = [[v] for v in regions[:63]]
    if len(regions) > 63:
        groups.append(regions[63:])
    blocks = []
    sx, sy = page['width_pt']/w, page['height_pt']/h
    for index, group in enumerate(groups):
        ids = [v[0] for v in group]
        b = [min(v[1][0] for v in group), min(v[1][1] for v in group),
             max(v[1][2] for v in group), max(v[1][3] for v in group)]
        mask = np.isin(labels, ids) & ~covered.astype(bool)
        identifier = f'{page["id"]}-unclassified-{index}'
        reference, _ = _save_image(rgb, b, directory, identifier+'-reference')
        asset, pixels = _save_image(rgb, b, directory, identifier, mask=mask)
        blocks.append(dict(id=identifier, kind='image',
            source=dict(page_id=page['id'], layout_detection_id=None, label='unknown',
                bbox_px=b, bbox_pt=[b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy], reference_asset=reference,
                score=None, ocr_lines=[], ownership={},
                crop_to_page=[[1,0,b[0]],[0,1,b[1]],[0,0,1]],
                coverage_evidence='uncovered-nonwhite-pixels; threshold=245; edge-tolerance=2px'),
            content=dict(asset=asset,pixel_size=pixels,aspect_ratio=pixels[0]/pixels[1]),
            style=_style('image',[],sy), local_layout={},
            quality=dict(state='image_fallback',issues=['unclassified-foreground-preserved-as-image'])))
    return blocks


def _records(raw, boxes):
    ocr = raw.get('overall_ocr_res') or {}
    texts = ocr.get('rec_texts') or []
    coords = ocr.get('rec_boxes') or ocr.get('rec_polys') or []
    owners = ocr.get('block_ids') or []
    scores = ocr.get('rec_scores') or []
    line_ids = ocr.get('line_ids') or []
    labels = ocr.get('labels') or []
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
            records[owner].append(dict(text=str(text), bbox_px=b,
                                       score=float(scores[i]) if i < len(scores) else None,
                                       line_id=str(line_ids[i]) if i < len(line_ids) else f'ocr-{i}',
                                       label=labels[i] if i < len(labels) else None,
                                       evidence_index=i))
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


def blocks_from_page(raw, rgb, page, directory):
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
    records = _records(raw, boxes)
    tables = {str(t.get('block_id')): t for t in raw.get('table_res_list') or []}
    suppressed = {str(d.get('block_id')) for d in raw.get('diagnostics', [])
                  if isinstance(d, dict) and d.get('reason') == 'fully-owned-by-another-block'}
    conflict_reasons = {'ocr-line-crosses-layout-blocks','ocr-line-crosses-visual-block',
                        'ocr-line-unmatched-block','ocr-line-recognition-rejected','empty-ocr-line-crop'}
    conflicts = {str(identifier) for d in raw.get('diagnostics', []) if isinstance(d, dict) and
                 d.get('reason') in conflict_reasons
                 for identifier in [d.get('block_id'), *d.get('candidate_ids', [])] if identifier}
    blocks = []
    # These pure parsers validate actual HTML topology and preserve empty cells.
    from .structure_parser import _table_structure
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
        block = dict(id=identifier, kind='text',
                     source=dict(page_id=page['id'], layout_detection_id=local_id, label=label,
                                 bbox_px=b, bbox_pt=[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
                                 score=detection.get('score'), reference_asset=crop,
                                 crop_to_page=[[1, 0, math.floor(b[0])], [0, 1, math.floor(b[1])], [0, 0, 1]],
                                 ownership=ownership,
                                 ocr_lines=copy.deepcopy(lines)),
                     content={}, style=_style(label, lines, sy), local_layout={},
                     quality=dict(state='needs_review', issues=[]))
        if local_id in conflicts and label not in RASTER_LABELS:
            block['kind'] = 'image'
            block['quality'].update(state='image_fallback', issues=['ambiguous-ocr-line-ownership'])
        elif label == 'table':
            result = tables.get(local_id)
            parsed = _table_structure(result, dict(id=identifier, bbox=b, text=''), sx, sy, w, h) if result and not exclusions else None
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
            block['content'] = dict(_text_content(lines, identifier), role=label)
            if not block['content']['text'].strip():
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
    page['detections'] = dict(layout=boxes, regions=(raw.get('region_det_res') or {}).get('boxes') or [])
    if blocks:
        from .source_geometry import organize_source
        blocks = organize_source(blocks,rgb,page,raw,directory,_save_image,_text_content)
        coverage = boxes + [dict(coordinate=b['source']['bbox_px']) for b in blocks
                            if b['source'].get('label') == 'container']
        blocks.extend(_unclassified_blocks(rgb, coverage, page, directory))
    if not blocks:
        crop, pixels = _save_image(rgb, [0, 0, w, h], directory, page['id']+'-unclassified')
        blocks.append(dict(id=page['id']+'-unclassified', kind='image',
            source=dict(page_id=page['id'], layout_detection_id=None, label='unknown', bbox_px=[0,0,w,h],
                        bbox_pt=[0,0,page['width_pt'],page['height_pt']], reference_asset=crop,
                        score=None, ocr_lines=[], ownership={}, crop_to_page=[[1,0,0],[0,1,0],[0,0,1]]),
            content=dict(asset=crop,pixel_size=pixels,aspect_ratio=w/h), style=_style('image',[],sy),
            local_layout={}, quality=dict(state='image_fallback',issues=['no-layout-detections'])))
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
def convert_adaptive(paths, output_dir, document_id, *, format='pdf', pages=''):
    if format not in {'pdf','html','md'}:
        raise ValueError('结构化重建仅支持 PDF、HTML 或 Markdown')
    paths, directory = [Path(p) for p in paths], Path(output_dir)
    selections, total = _input_pages(paths,pages)
    directory.mkdir(parents=True,exist_ok=True)
    project = dict(schema='feature4-rebuild/1', id=document_id, plan=PLAN,
                   source_pages=[], blocks=[], relations={}, layout_result={},
                   limitations=['OCR and model topology require review.',
                                'Style values are role templates, not recovered original typography.',
                                'Source geometry is preserved where feasible; local growth stays within isolated frames.',
                                'Region membership and reading-order evidence still require review.'])
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
                raw = recognize_page(rgb)
            with span('rebuild.typed_blocks'):
                project['blocks'].extend(blocks_from_page(raw,rgb,source,directory))
            project['source_pages'].append(source)
    with span('rebuild.relations'):
        project['relations'] = derive_relations(project)
    progress(72,'两层自适应排版','保留源位置，在区域内检查碰撞与等比适配')
    from .adaptive_layout import layout_document
    from .adaptive_export import export_project
    with span('rebuild.adaptive_layout'):
        project['layout_result'] = layout_document(project)
    report = dict(schema='feature4-conversion/1',plan=PLAN,format=format,status='needs-review',
                  layoutElapsedSeconds=round(time.perf_counter()-started,6),
                  sourcePageCount=total,pageCount=len(project['layout_result']['pages']),
                  independentTextLines=sum(len(b['source'].get('ocr_lines',[])) for b in project['blocks'] if b['kind']=='text'),
                  tableCount=sum(b['kind']=='table' for b in project['blocks']),
                  imageCount=sum(b['kind']=='image' for b in project['blocks']),
                  compositeCount=sum(b['kind']=='composite' for b in project['blocks']),
                  blockCount=len(project['blocks']),orderChanges=project['layout_result'].get('order_changes',[]),
                  recognitionBackend='v3',strategy='typed-blocks-two-level-adaptive',
                  relationsVersion=project['relations'].get('algorithm_version'),
                  layoutVersion=project['layout_result'].get('algorithm_version'),
                  regionAdjustments=project['layout_result'].get('region_adjustments',[]),
                  limitations=project['limitations'])
    if any(b['quality']['state']=='image_fallback' for b in project['blocks']):
        report['status']='partial'
    project['report']=report
    progress(85,'导出重建文档','写入真实文字、表格与图像资源')
    with span('rebuild.adaptive_export'):
        destination, diagnostics = export_project(project,directory,format=format)
    (directory/'conversion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    progress(100,'转换完成')
    return destination,report
