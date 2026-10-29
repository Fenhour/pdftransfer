"""One page detection pass, unique block ownership, and bounded line batches.

The existing V3 detector, crop operator, recognizer and OCRResult are reused.
Nothing here changes installed third-party code or permanently changes the
shared recognizer's batch configuration. Public evidence uses page pixels;
the OCRResults passed to table recognition use masked block-local pixels.
"""
from __future__ import annotations

import math

import numpy as np

from pdftransfer_usage import span


OCR_LABELS = {'text', 'paragraph_title', 'doc_title', 'number', 'abstract',
              'content', 'figure_title', 'reference', 'reference_content',
              'footnote', 'header', 'footer', 'algorithm', 'formula_number',
              'aside_text', 'title', 'section_title', 'caption', 'page_number',
              'table', 'image'}
MAX_BATCH_SIZE = 8
MAX_PADDED_PIXELS = 614400
MAX_PADDING_RATIO = 1.5
MIN_OWNER_COVERAGE = .5
MIN_COMPLETE_COVERAGE = .98
MAX_OWNER_EDGE_TOLERANCE = 12.


def _ocr_result(value):
    # Keep importing this transport module independent of Paddle loading.
    from paddlex.inference.pipelines.ocr.result import OCRResult
    return OCRResult(value)


def _ocr_pipeline(pipeline):
    value = getattr(pipeline, 'general_ocr_pipeline', pipeline)
    if not hasattr(value, 'text_det_model'):
        value = getattr(value, '_pipeline', value)
    if not all(hasattr(value, key) for key in ('text_det_model', 'text_rec_model', '_crop_by_polys')):
        raise ValueError('Installed V3 OCR pipeline does not expose det/crop/rec components')
    return value


def _consume(iterator):
    iterator = iter(iterator)
    try:
        return list(iterator)
    finally:
        close = getattr(iterator, 'close', None)
        if callable(close):
            close()


def _crop_bounds(box, width, height):
    return (max(0, math.floor(box[0])), max(0, math.floor(box[1])),
            min(width, math.ceil(box[2])), min(height, math.ceil(box[3])))


def _masked_crop(image, block, evidence, visual_rectangles, padding=0):
    height, width = image.shape[:2]
    a,b,c,d = block['coordinate']
    x0, y0, x1, y1 = _crop_bounds([a-padding,b-padding,c+padding,d+padding], width, height)
    crop = image[y0:y1, x0:x1].copy()
    for rect in list(evidence.get('excluded_rects', []))+visual_rectangles:
        a, b, c, d = _crop_bounds(rect, width, height)
        a, b, c, d = max(x0, a), max(y0, b), min(x1, c), min(y1, d)
        if c > a and d > b:
            crop[b-y0:d-y0, a-x0:c-x0] = 255
    return crop, x0, y0


def _rec_dimensions(recognizer):
    transforms = getattr(recognizer, 'pre_tfs', {})
    resize = transforms.get('ReisizeNorm') if isinstance(transforms, dict) else None
    shape = getattr(resize, 'rec_image_shape', [3, 48, 320])
    height, minimum_width = int(shape[1]), int(shape[2])
    maximum_width = int(getattr(resize, 'max_imgW', 3200))
    if height < 1 or minimum_width < 1 or maximum_width < minimum_width:
        raise ValueError('Invalid installed recognizer normalization dimensions')
    return height, minimum_width, maximum_width


def _batches(lines, dimensions):
    height, minimum_width, maximum_width = dimensions
    ordered = sorted(lines, key=lambda value: (value['image'].shape[1]/value['image'].shape[0], value['line_id']))
    batch, widths = [], []
    for value in ordered:
        width = min(maximum_width, max(minimum_width,
                    math.ceil(height*value['image'].shape[1]/value['image'].shape[0])))
        proposed = widths+[width]
        padded = len(proposed)*max(proposed)*height
        ratio = len(proposed)*max(proposed)/sum(proposed)
        if batch and (len(proposed) > MAX_BATCH_SIZE or padded > MAX_PADDED_PIXELS or ratio > MAX_PADDING_RATIO):
            yield batch, widths
            batch, widths = [], []
        batch.append(value)
        widths.append(width)
    if batch:
        yield batch, widths


def _empty_result(crop, det_params, threshold):
    return dict(input_path=None, page_index=None,
                doc_preprocessor_res={'output_img': crop},
                dt_polys=np.empty((0, 4, 2), dtype=np.float32),
                model_settings=dict(use_doc_preprocessor=False, use_textline_orientation=False),
                text_det_params=det_params.copy(), text_type='general',
                text_rec_score_thresh=threshold, return_word_box=False,
                rec_texts=[], rec_scores=[], rec_polys=np.empty((0, 4, 2), dtype=np.float32),
                rec_boxes=np.empty((0, 4), dtype=np.float32), vis_fonts=[],
                textline_orientation_angles=[])


def recognize_blocks(bgr, pipeline, ordered, block_ownership, diagnostics):
    """Return page records, real per-block OCRResults and JSON-safe audit data."""
    from shapely.geometry import Polygon, box as rectangle
    from shapely.ops import unary_union

    ocr = _ocr_pipeline(pipeline)
    height, width = bgr.shape[:2]
    # Match the existing full-page OCR scale policy, without the former
    # per-block min-side upscaling that amplified tiny crops.
    det_params = dict(limit_side_len=736, limit_type='min', max_side_limit=4000,
                      thresh=float(getattr(ocr, 'text_det_thresh', .3)),
                      box_thresh=float(getattr(ocr, 'text_det_box_thresh', .6)),
                      unclip_ratio=float(getattr(ocr, 'text_det_unclip_ratio', 1.5)))
    threshold = float(getattr(ocr, 'text_rec_score_thresh', 0.))
    with span('v3.rebuild.text_det', backend='native'):
        detections = _consume(ocr.text_det_model(bgr, **det_params))
    if len(detections) != 1:
        raise ValueError('Full-page text detector must return exactly one result')
    raw_polys = detections[0].get('dt_polys')
    raw_polys = [] if raw_polys is None else raw_polys
    page_rect = rectangle(0, 0, width, height)
    visual_blocks = [block for block in ordered
                     if block['label'] == 'image' or str(block['label']).lower() not in OCR_LABELS]
    visual = [(block['id'], rectangle(*block['coordinate'])) for block in visual_blocks]
    candidates, local_values, origins = [], {}, {}
    for priority, block in enumerate(ordered):
        if str(block['label']).lower() not in OCR_LABELS:
            continue
        evidence = block_ownership.get(block['id'], {})
        # Images own their internal transcripts; smaller earlier visual blocks
        # keep priority. Prose never takes pixels from an enclosing picture.
        blockers = [v['coordinate'] for v in visual_blocks if v['id'] != block['id'] and
                    (block['label'] != 'image' or ordered.index(v) < priority)]
        excluded = [rectangle(*rect) for rect in list(evidence.get('excluded_rects', []))+blockers]
        owned = rectangle(*block['coordinate']).intersection(page_rect)
        if excluded:
            owned = owned.difference(unary_union(excluded))
        if owned.is_empty:
            continue
        padding = MAX_OWNER_EDGE_TOLERANCE if block['label'] not in {'table','image'} else 0
        crop, x0, y0 = _masked_crop(bgr, block, evidence, blockers, padding)
        if crop.size == 0:
            continue
        candidates.append((priority, block, owned))
        local_values[block['id']] = _empty_result(crop, det_params, threshold)
        origins[block['id']] = (x0, y0)

    info = dict(policy='full-page-det-owned-line-batches-v2-image-transcripts', coordinate_space='page-pixels',
                det_params=det_params, inputDetCount=len(raw_polys), acceptedLineCount=0,
                skippedVisualLineCount=0, unmatchedLineCount=0, recognitionBatches=[],
                line_assignments=[], dt_polys=[],
                batch_limits=dict(max_batch_size=MAX_BATCH_SIZE, max_padded_pixels=MAX_PADDED_PIXELS,
                                  max_padding_ratio=MAX_PADDING_RATIO))
    valid = []
    for index, value in enumerate(raw_polys):
        poly = np.array(value, dtype=np.float32, copy=True)
        if poly.shape != (4, 2) or not np.isfinite(poly).all():
            diagnostics.append(dict(reason='invalid-ocr-polygon', detection_index=index))
            continue
        poly[:, 0] = np.clip(poly[:, 0], 0, width)
        poly[:, 1] = np.clip(poly[:, 1], 0, height)
        geometry = Polygon(poly)
        if not geometry.is_valid or geometry.area <= .1:
            diagnostics.append(dict(reason='invalid-ocr-polygon', detection_index=index))
            continue
        valid.append((poly, geometry, index))
    valid.sort(key=lambda value: (float(value[0][:, 1].min()), float(value[0][:, 0].min()),
                                 tuple(float(v) for v in value[0].ravel()), value[2]))
    pending = []
    for index, (poly, geometry, detection_index) in enumerate(valid):
        line_id = f'line-{index}'
        bounds = [float(v) for v in geometry.bounds]
        assignment = dict(line_id=line_id, detection_index=detection_index,
                          block_id=None, label=None, polygon=poly.tolist(), box=bounds,
                          state='unmatched')
        info['line_assignments'].append(assignment)
        info['dt_polys'].append(poly.tolist())
        overlaps = [(geometry.intersection(owned).area/geometry.area, priority, block)
                    for priority, block, owned in candidates if geometry.intersects(owned)]
        overlaps.sort(key=lambda value: (-value[0], value[1], value[2]['id']))
        candidate_ids = [value[2]['id'] for value in overlaps if value[0] > 1e-6]
        assignment['candidate_ids'] = candidate_ids
        if overlaps and overlaps[0][0] >= MIN_OWNER_COVERAGE:
            coverage, _, owner = overlaps[0]
            assignment.update(block_id=owner['id'], label=owner['label'], owner_coverage=float(coverage))
        else:
            owner = None
        visual_ids = [identifier for identifier, shape in visual
                      if geometry.intersection(shape).area > 1e-6]
        if owner is not None and owner['label'] == 'image' and coverage >= MIN_COMPLETE_COVERAGE:
            # Unique masked ownership is already established; enclosing visual
            # detections must not veto a smaller picture's own internal line.
            visual_ids = []
        if visual_ids:
            assignment['state'] = 'visual-conflict'
            assignment['excluded_ids'] = visual_ids
            info['skippedVisualLineCount'] += 1
            diagnostics.append(dict(reason='ocr-line-crosses-visual-block', line_id=line_id,
                                    block_id=assignment['block_id'], excluded_ids=visual_ids,
                                    candidate_ids=candidate_ids,
                                    polygon=poly.tolist(), box=bounds))
            continue
        if owner is None:
            info['unmatchedLineCount'] += 1
            diagnostics.append(dict(reason='ocr-line-unmatched-block', line_id=line_id,
                                    candidate_ids=candidate_ids, polygon=poly.tolist(), box=bounds))
            continue
        tolerated = False
        if coverage < MIN_COMPLETE_COVERAGE and owner['label'] not in {'table','image'} and len(candidate_ids) == 1:
            # Layout boxes describe a paragraph, while det polygons include
            # their own border expansion. Small edge differences alone are
            # not ambiguity between blocks. Do not relax visual exclusions
            # or genuine multi-block ownership.
            tolerance = min(MAX_OWNER_EDGE_TOLERANCE, max(2., (bounds[3]-bounds[1])*.3))
            owned = next(shape for _, block, shape in candidates if block['id'] == owner['id'])
            tolerated = geometry.difference(owned.buffer(tolerance, join_style=2)).area/geometry.area <= .02
            if tolerated:
                assignment.update(assignment_method='bounded-layout-edge-tolerance', edge_tolerance_px=tolerance)
        if coverage < MIN_COMPLETE_COVERAGE and not tolerated:
            assignment['state'] = 'partial-owner-conflict'
            diagnostics.append(dict(reason='ocr-line-crosses-layout-blocks', line_id=line_id,
                                    block_id=owner['id'], candidate_ids=candidate_ids,
                                    owner_coverage=float(coverage), polygon=poly.tolist(), box=bounds))
            continue
        local = local_values[owner['id']]
        x0, y0 = origins[owner['id']]
        local_poly = poly-np.array([x0, y0], dtype=np.float32)
        # The crop operator's perspective border is replicated. Bound the
        # polygon first so neither another block nor a visual pixel leaks in.
        local_poly[:, 0] = np.clip(local_poly[:, 0], 0, local['doc_preprocessor_res']['output_img'].shape[1])
        local_poly[:, 1] = np.clip(local_poly[:, 1], 0, local['doc_preprocessor_res']['output_img'].shape[0])
        crops = _consume(ocr._crop_by_polys(local['doc_preprocessor_res']['output_img'], [local_poly]))
        if len(crops) != 1 or np.asarray(crops[0]).ndim != 3 or np.asarray(crops[0]).size == 0:
            assignment['state'] = 'empty-crop'
            diagnostics.append(dict(reason='empty-ocr-line-crop', line_id=line_id, block_id=owner['id']))
            continue
        local['dt_polys'] = np.concatenate((local['dt_polys'], local_poly[None]), axis=0)
        pending.append(dict(line_id=line_id, block_id=owner['id'], label=owner['label'],
                            polygon=poly, local_polygon=local_poly, box=bounds,
                            image=np.asarray(crops[0]), assignment=assignment))
        assignment['state'] = 'queued'

    recognizer = ocr.text_rec_model
    dimensions = _rec_dimensions(recognizer)
    sampler = getattr(recognizer, 'batch_sampler', None)
    previous_batch_size = getattr(sampler, 'batch_size', None)
    records = []
    try:
        for batch, widths in _batches(pending, dimensions):
            normalized_width = max(widths)
            batch_info = dict(line_ids=[v['line_id'] for v in batch], batch_size=len(batch),
                              normalized_width=normalized_width,
                              padded_pixels=len(batch)*dimensions[0]*normalized_width,
                              padding_ratio=round(len(batch)*normalized_width/sum(widths), 6))
            info['recognitionBatches'].append(batch_info)
            with span('v3.rebuild.text_rec_batch', backend='native', lineCount=len(batch),
                      normalizedWidth=normalized_width, paddedPixels=batch_info['padded_pixels'],
                      paddingRatio=batch_info['padding_ratio']):
                results = _consume(recognizer([v['image'] for v in batch],
                                             batch_size=len(batch), return_word_box=False))
            if len(results) != len(batch):
                raise ValueError('Text recognizer result count does not match scheduled line batch')
            for value, result in zip(batch, results):
                text, score = result.get('rec_text'), float(result.get('rec_score', 0.))
                if not isinstance(text, str) or not text.strip() or not math.isfinite(score) or score < threshold:
                    value['assignment']['state'] = 'recognition-rejected'
                    diagnostics.append(dict(reason='ocr-line-recognition-rejected', line_id=value['line_id'],
                                            block_id=value['block_id']))
                    continue
                value['assignment']['state'] = 'recognized'
                records.append(dict(text=text, polygon=value['polygon'].tolist(), box=value['box'],
                                    score=score, block_id=value['block_id'], label=value['label'],
                                    line_id=value['line_id']))
                local = local_values[value['block_id']]
                local.setdefault('_recognized', []).append((value, text, score, result.get('vis_font')))
    finally:
        if sampler is not None and previous_batch_size is not None:
            sampler.batch_size = previous_batch_size

    local_results = {}
    line_order = {v['line_id']: i for i, v in enumerate(info['line_assignments'])}
    for identifier, local in local_values.items():
        values = sorted(local.pop('_recognized', []), key=lambda value: line_order[value[0]['line_id']])
        local['rec_texts'] = [v[1] for v in values]
        local['rec_scores'] = np.asarray([v[2] for v in values], dtype=np.float32)
        local['rec_polys'] = np.asarray([v[0]['local_polygon'] for v in values], dtype=np.float32).reshape(-1, 4, 2)
        local['rec_boxes'] = np.asarray([[*poly.min(axis=0), *poly.max(axis=0)]
                                        for poly in local['rec_polys']], dtype=np.float32).reshape(-1, 4)
        local['vis_fonts'] = [v[3] for v in values if v[3] is not None]
        local['line_ids'] = [v[0]['line_id'] for v in values]
        local['block_id'] = identifier
        local['textline_orientation_angles'] = [-1]*len(local['dt_polys'])
        local_results[identifier] = _ocr_result(local)
    records.sort(key=lambda value: line_order[value['line_id']])
    info['acceptedLineCount'] = len(records)
    info['matchedLineCount'] = len(pending)
    return records, local_results, info
