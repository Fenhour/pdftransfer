"""Feature-four orchestration over the installed V3 models, not new weights.

Full-page text detection is associated to layout IDs before batched recognition.
All published coordinates use the original input image's pixel space.
"""
from __future__ import annotations

import math

from pdftransfer_usage import span

POLICY = 'feature4-text-v12-containment-1pt-20261105'
TEXT_LABELS = {'text', 'paragraph_title', 'doc_title', 'number', 'abstract',
               'content', 'figure_title', 'reference', 'reference_content',
               'footnote', 'header', 'footer', 'algorithm', 'formula_number',
               'aside_text', 'title', 'section_title', 'caption', 'page_number'}


def _first(results):
    results = iter(results)
    try:
        value = next(results, None)
    finally:
        close = getattr(results, 'close', None)
        if callable(close):
            close()
    if value is None:
        raise ValueError('V3 component returned no result')
    return value


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()
                if k not in {'input_img', 'output_img', 'input_path'}}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if hasattr(value, 'item'):
        return value.item()
    return value


def _unwrap(pipeline, attribute):
    pipeline = getattr(pipeline, 'paddlex_pipeline', pipeline)
    if hasattr(pipeline, attribute):
        return pipeline
    pipeline = getattr(pipeline, '_pipeline', pipeline)
    if not hasattr(pipeline, attribute):
        raise ValueError('Installed V3 component does not expose required models')
    return pipeline


def _box(value, width, height):
    try:
        x0, y0, x1, y1 = map(float, value)
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return None
        result = [max(0., min(width, x0)), max(0., min(height, y0)),
                  max(0., min(width, x1)), max(0., min(height, y1))]
        return result if result[2] > result[0] and result[3] > result[1] else None
    except (TypeError, ValueError):
        return None


def _area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def _intersection(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def _detections(result, width, height, prefix):
    records = []
    for index, item in enumerate(result.get('boxes', [])):
        box = _box(item.get('coordinate'), width, height)
        if box:
            records.append(dict(id=f'{prefix}-{index}', cls_id=item.get('cls_id'),
                                label=str(item.get('label', 'unknown')).lower(),
                                score=float(item.get('score', 0.)), coordinate=box))
    return records


def _rank(block):
    label = block['label']
    return (2 if label in TEXT_LABELS else 1 if label == 'table' else 0,
            _area(block['coordinate']), block['id'])


def _mask_crop(image, block, blockers):
    import numpy as np
    h, w = image.shape[:2]
    b = block['coordinate']
    x0, y0 = max(0, math.floor(b[0])), max(0, math.floor(b[1]))
    x1, y1 = min(w, math.ceil(b[2])), min(h, math.ceil(b[3]))
    crop = image[y0:y1, x0:x1].copy()
    ownership = np.ones(crop.shape[:2], dtype=bool)
    excluded_ids, excluded_rects = [], []
    for other in blockers:
        ob = other['coordinate']
        if _intersection(b, ob) <= 0:
            continue
        left, top = max(x0, math.floor(ob[0]))-x0, max(y0, math.floor(ob[1]))-y0
        right, bottom = min(x1, math.ceil(ob[2]))-x0, min(y1, math.ceil(ob[3]))-y0
        if right <= left or bottom <= top:
            continue
        ownership[top:bottom, left:right] = False
        excluded_ids.append(other['id'])
        excluded_rects.append([left+x0, top+y0, right+x0, bottom+y0])
    crop[~ownership] = 255
    evidence = dict(excluded_block_ids=excluded_ids, excluded_rects=excluded_rects,
                    owned_pixel_count=int(ownership.sum()), total_pixel_count=int(ownership.size))
    return crop, x0, y0, ownership, evidence


def _ocr_records(raw, block_id, dx, dy, ownership, diagnostics):
    import numpy as np
    texts = raw.get('rec_texts', [])
    polys = raw.get('rec_polys')
    boxes = raw.get('rec_boxes')
    scores = raw.get('rec_scores', [])
    records = []
    for index, text in enumerate(texts):
        if not str(text).strip():
            continue
        if polys is not None and index < len(polys):
            # The table pipeline still needs its original crop-local OCRResult.
            poly = np.array(polys[index], dtype=float, copy=True)
        elif boxes is not None and index < len(boxes):
            x0, y0, x1, y1 = boxes[index]
            poly = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], dtype=float)
        else:
            diagnostics.append(dict(block_id=block_id, reason='ocr-text-without-position', line=index))
            continue
        if poly.shape != (4, 2) or not np.isfinite(poly).all():
            diagnostics.append(dict(block_id=block_id, reason='invalid-ocr-polygon', line=index))
            continue
        clipped = poly.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, ownership.shape[1])
        clipped[:, 1] = np.clip(clipped[:, 1], 0, ownership.shape[0])
        if not np.array_equal(clipped, poly):
            diagnostics.append(dict(block_id=block_id, reason='ocr-polygon-clipped-to-crop', line=index))
        poly = clipped
        x0,y0 = poly.min(axis=0)
        x1,y1 = poly.max(axis=0)
        # Do not publish a hallucinated OCR box lying mostly on excluded art.
        xa,ya = max(0,math.floor(x0)),max(0,math.floor(y0))
        xb,yb = min(ownership.shape[1],math.ceil(x1)),min(ownership.shape[0],math.ceil(y1))
        if xb <= xa or yb <= ya or ownership[ya:yb,xa:xb].mean() < .5:
            diagnostics.append(dict(block_id=block_id, reason='ocr-excluded-pixels', line=index))
            continue
        poly += [dx,dy]
        score = float(scores[index]) if index < len(scores) else None
        records.append(dict(text=str(text), polygon=poly.tolist(),
                            box=[float(x0+dx),float(y0+dy),float(x1+dx),float(y1+dy)],
                            score=score, block_id=block_id))
    return records


def recognize_rebuild(rgb, model, *, region_unclip_ratio=None, cached_evidence=None,
                      pixel_to_point=(.5, .5)):
    import numpy as np
    pipeline = _unwrap(model, 'layout_det_model')
    bgr = np.asarray(rgb)[:, :, ::-1].copy()
    height,width = bgr.shape[:2]
    diagnostics = []
    with span('v3.rebuild.layout', backend='native'):
        layout = (cached_evidence['layout_det_res']['boxes'] if cached_evidence is not None else
                  _detections(_first(pipeline.layout_det_model(bgr)), width,height,'layout'))
        region_model = getattr(pipeline, 'region_detection_model', None)
        if region_model is None:
            raise ValueError('Feature four requires PP-DocBlockLayout region detection')
        region_options = dict(layout_nms=True,layout_merge_bboxes_mode='small')
        if region_unclip_ratio is not None:
            if not math.isfinite(float(region_unclip_ratio)) or not 1. <= float(region_unclip_ratio) <= 1.25:
                raise ValueError('region_unclip_ratio must be between 1.0 and 1.25')
            region_options['layout_unclip_ratio'] = float(region_unclip_ratio)
        regions = (cached_evidence['region_det_res']['boxes'] if cached_evidence is not None else
                   _detections(_first(region_model(bgr,**region_options)),width,height,'region'))
    if not layout:
        diagnostics.append(dict(reason='no-layout-detections'))
    from source_containment import build_containment, conflict_is_internal
    containment = build_containment(layout, pixel_to_point=pixel_to_point)
    nodes = containment['nodes']
    ordered = sorted(layout,key=_rank)
    tables, block_ownership = [], {}
    for index,block in enumerate(ordered):
        blockers = ordered[:index]
        crop,dx,dy,ownership,evidence = _mask_crop(bgr,block,blockers)
        block_ownership[block['id']] = evidence
        overlaps = evidence['excluded_block_ids']
        if overlaps:
            diagnostics.append(dict(block_id=block['id'],reason='overlapping-source-blocks',excluded_ids=overlaps))
        if not ownership.any():
            diagnostics.append(dict(block_id=block['id'],reason='fully-owned-by-another-block'))
            continue
    from rebuild_ocr import recognize_blocks
    with span('v3.rebuild.ocr', backend='native', policy=POLICY):
        options = {}
        if cached_evidence is not None:
            overall=cached_evidence['overall_ocr_res']
            options['cached_ocr']=dict(dt_polys=cached_evidence['ocr_pipeline']['dt_polys'],
                records=[dict(text=t,score=s,polygon=p) for t,s,p in zip(
                    overall['rec_texts'],overall['rec_scores'],overall['rec_polys'])])
        ocr_records, local_results, ocr_info = recognize_blocks(
            bgr, pipeline, ordered, block_ownership, diagnostics, **options)
    conflict_reasons = {'ocr-line-crosses-layout-blocks', 'ocr-line-crosses-visual-block',
                        'ocr-line-unmatched-block', 'ocr-line-recognition-rejected',
                        'empty-ocr-line-crop'}
    conflicts = {identifier for item in diagnostics if item.get('reason') in conflict_reasons
                 for identifier in [item.get('block_id'), *item.get('candidate_ids', [])] if identifier
                 and not conflict_is_internal(item, identifier, nodes)}
    for index,block in enumerate(ordered):
        if block['id'] not in local_results:
            continue
        crop,dx,dy,ownership,evidence = _mask_crop(bgr,block,ordered[:index])
        overlaps = evidence['excluded_block_ids']
        raw = local_results[block['id']]
        if not raw['rec_texts']:
            diagnostics.append(dict(block_id=block['id'],reason='empty-ocr-result'))
        if block['label'] == 'table':
            internal_overlaps = bool(overlaps) and all(i in nodes[block['id']]['contained_ids'] for i in overlaps)
            if not raw['rec_texts'] and not internal_overlaps:
                # A valid empty grid is not evidence that source cells were
                # blank. Preserve the table rather than erase missed text.
                diagnostics.append(dict(block_id=block['id'],reason='table-empty-ocr-image-fallback'))
                continue
            if block['id'] in conflicts:
                diagnostics.append(dict(block_id=block['id'],reason='table-ocr-conflict-image-fallback'))
                continue
            if overlaps and not internal_overlaps:
                # An occluded table is not a trustworthy complete structure.
                # Preserve its owned pixels downstream instead of publishing
                # a successful grid inferred from a partly whitened source.
                diagnostics.append(dict(block_id=block['id'],reason='table-overlap-image-fallback',
                                        excluded_ids=overlaps,
                                        excluded_pixel_count=evidence['total_pixel_count']-evidence['owned_pixel_count']))
                continue
            if internal_overlaps:
                # Internal detections are legitimate content, not occlusion.
                # Use intact pixels for topology; OCR ownership stays unique.
                crop,dx,dy,_,_ = _mask_crop(bgr,block,[v for v in ordered[:index]
                    if v['id'] not in nodes[block['id']]['contained_ids']])
                diagnostics.append(dict(block_id=block['id'],reason='table-contained-blocks-accepted',
                                        contained_ids=overlaps))
            table_pipeline = getattr(pipeline,'table_recognition_pipeline',None)
            if table_pipeline is None:
                diagnostics.append(dict(block_id=block['id'],reason='table-recognition-disabled'))
                continue
            try:
                with span('v3.rebuild.table', backend='native', blockId=block['id']):
                    result = _first(table_pipeline(crop,
                              use_doc_orientation_classify=False,use_doc_unwarping=False,
                              use_layout_detection=False,use_ocr_model=False,
                              overall_ocr_res=raw,use_table_orientation_classify=True,
                              use_e2e_wired_table_rec_model=False,use_e2e_wireless_table_rec_model=False,
                              use_ocr_results_with_table_cells=True))
                for value in result.get('table_res_list', []):
                    table = _plain(value)
                    # Keep the native V3 topology. The additional OCR-content /
                    # cell-overlap gate and row-band repair are paused pending
                    # review; downstream structural parsing still applies.
                    table['validation_policy'] = 'native-v3-specialized-validation-paused'
                    diagnostics.append(dict(block_id=block['id'],
                        reason='table-native-result-specialized-validation-paused'))
                    cells = []
                    for box in table.get('cell_box_list',[]):
                        array = np.asarray(box,dtype=float)
                        if array.shape == (4,):
                            array += [dx,dy,dx,dy]
                        elif array.ndim == 1 and array.size == 8:
                            array = (array.reshape(4,2)+[dx,dy]).reshape(8)
                        elif array.ndim == 2 and array.shape[1] == 2:
                            array += [dx,dy]
                        else:
                            diagnostics.append(dict(block_id=block['id'],reason='invalid-table-cell-box'))
                            continue
                        cells.append(array.tolist())
                    table.update(block_id=block['id'],cell_box_list=cells,
                                 coordinate=block['coordinate'],coordinate_space='page-pixels',
                                 ocr_policy='reuse-page-det-batched-rec; rotated/cross-cell conflicts may rerun locally')
                    # table_ocr_pred remains a local diagnostic, not page OCR.
                    table['table_ocr_coordinate_space']='table-local-pixels'
                    table['table_ocr_orientation']='pipeline-normalized'
                    table['crop_origin']=[dx,dy]
                    tables.append(table)
            except (ValueError, IndexError, AssertionError) as error:
                diagnostics.append(dict(block_id=block['id'],reason='table-structure-unavailable',
                                        error_type=type(error).__name__))
    return dict(width=width,height=height,input_path=None,rebuild_policy=POLICY,
                evidence_reuse=cached_evidence.get('reuse_provenance') if cached_evidence is not None else None,
                layout_det_res=dict(boxes=layout),region_det_res=dict(boxes=regions,unclip_ratio=region_unclip_ratio),
                overall_ocr_res=dict(rec_texts=[r['text'] for r in ocr_records],
                     rec_boxes=[r['box'] for r in ocr_records],rec_polys=[r['polygon'] for r in ocr_records],
                     rec_scores=[r['score'] for r in ocr_records],block_ids=[r['block_id'] for r in ocr_records],
                     line_ids=[r['line_id'] for r in ocr_records],labels=[r['label'] for r in ocr_records]),
                table_res_list=tables,parsing_res_list=[],block_ownership=block_ownership,
                source_containment=containment,
                ocr_pipeline=ocr_info,diagnostics=diagnostics)
