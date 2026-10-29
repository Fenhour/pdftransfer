"""Conservative raster-region ownership for reconstruction on a blank page.

This is a bounded, local geometry heuristic, not a semantic figure detector.
OCR boxes are evidence, never proposed image boundaries. Crops preserve source
RGB; safe text exclusions use alpha, without inpainting or replacement pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .scan_layout import partition_text, horizontal_rules


def _overlap(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _touches_support(quad, support, margin=0):
    """Check polygon contact locally, without allocating a page per OCR line."""
    quad = np.round(np.asarray(quad)).astype(np.int32)
    margin = max(0, int(margin))
    x0, y0 = np.maximum(0, quad.min(axis=0)-margin)
    x1, y1 = np.minimum([support.shape[1], support.shape[0]], quad.max(axis=0)+margin+1)
    if x1 <= x0 or y1 <= y0:
        return False
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillConvexPoly(mask, quad-[x0,y0], 255)
    if margin:
        mask = cv2.dilate(mask, np.ones((2*margin+1, 2*margin+1), np.uint8))
    return bool(np.any((mask > 0) & (support[y0:y1,x0:x1] > 0)))


def foreground(rgb):
    """Return analysis ink and estimated paper; never modify the source RGB."""
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    paper = np.median(border, axis=0)
    ink = (np.max(np.abs(rgb.astype(np.float32)-paper), axis=2) > 22).astype(np.uint8)*255
    if min(rgb.shape[:2]) >= 31:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        ink |= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 31, 12)
    return ink, '#'+''.join(f'{int(c):02x}' for c in paper)


def _merge_regions(boxes, typical, barriers):
    """Enclosing frames absorb their contents, including disconnected labels.

    Only nearby pieces are grouped without an enclosing frame. Alignment is
    not evidence that arbitrarily distant objects belong to the same figure.
    Independent text in the intervening corridor prevents proximity merges.
    """
    boxes = [list(map(float, b)) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(boxes):
            for j in range(i+1, len(boxes)):
                b = boxes[j]
                dx = max(0., max(a[0], b[0])-min(a[2], b[2]))
                dy = max(0., max(a[1], b[1])-min(a[3], b[3]))
                # Bound proximity by local line height, not page dimensions.
                # A frame can still absorb disconnected contents by overlap.
                nearby = typical > 0 and ((dy == 0 and dx <= typical*2.5) or
                                           (dx == 0 and dy <= typical*2.5))
                joined = _union(a, b)
                corridor_text = any(_overlap(t, joined) and not _overlap(t, a) and not _overlap(t, b)
                                    for t in barriers)
                if _overlap(a, b) or nearby and not corridor_text:
                    boxes[i] = joined
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return boxes


def image_regions(rgb, candidates, scale, excluded_boxes=(), *, ignored_mask=None):
    """Return independently rebuildable text and intact, reviewable image crops.

    Coordinates are top-left PDF points. Every candidate has exactly one
    visible owner: independent text or a retained image. The source remains
    immutable. Real artwork contact retains text in the image; a rectangular
    empty corner can exclude safely separated prose with an alpha clip.
    """
    independent, retained = partition_text(rgb, candidates, scale, contour_support=True,
                                           ignored_mask=ignored_mask)
    ratio = min(1., 1600/max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=ratio, fy=ratio) if ratio < 1 else rgb
    unit = scale*ratio
    ink, paper = foreground(small)
    if ignored_mask is not None:
        ignored = cv2.resize(np.asarray(ignored_mask, dtype=np.uint8), (small.shape[1], small.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        ink[ignored > 0] = 0
    residual = ink.copy()
    heights = [max(3., c['element']['height']*unit) for c in candidates]
    typical = float(np.median(heights)) if heights else max(8., 10*unit)
    # Only independent text is excluded from the analysis mask. Retained text
    # and non-text marks participate together in determining an image boundary.
    text_mask = np.zeros_like(residual)
    for c in independent:
        quad = np.round(np.asarray(c['quad'])*unit).astype(np.int32)
        cv2.fillConvexPoly(text_mask, quad, 255)
    residual[text_mask > 0] = 0
    for line in retained:
        quad = np.round(np.asarray(line['polygon'])*unit).astype(np.int32)
        cv2.fillConvexPoly(residual, quad, 255)
    for box in excluded_boxes:
        x0, y0, x1, y1 = np.asarray(box)*unit
        pad = max(2, round(unit))
        residual[max(0,int(y0)-pad):min(len(residual),int(y1)+pad+1),
                 max(0,int(x0)-pad):min(residual.shape[1],int(x1)+pad+1)] = 0
    # Filled external contours protect both actual ink and enclosed figure
    # interiors. An open shape's rectangular empty corner remains empty.
    protected = np.zeros_like(residual)
    # A dotted/solid horizontal leader can sit next to prose without owning
    # that prose. Keep its source pixels but omit it from figure support.
    graphic_residual = residual.copy()
    graphic_residual[horizontal_rules(residual, typical) > 0] = 0
    contours, _ = cv2.findContours(graphic_residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(protected, contours, -1, 255, cv2.FILLED)
    protected = cv2.dilate(protected, np.ones((3, 3), np.uint8))
    kernel = max(3, min(25, round(typical*.6)))
    joined = cv2.dilate(residual, np.ones((kernel, kernel), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    # Too many disconnected candidates are not evidence of hundreds of figures.
    uncertain_page = count > 180
    if uncertain_page:
        boxes = [[0., 0., float(rgb.shape[1])/scale, float(rgb.shape[0])/scale]]
    else:
        # Dilation connects components; its full synthetic margin is not
        # source content. First recover each group's undilated support.
        boxes = []
        for label, (x, y, w, h, _) in enumerate(stats[1:count], 1):
            ys, xs = np.nonzero((labels[y:y+h, x:x+w] == label) &
                                (residual[y:y+h, x:x+w] > 0))
            if len(xs):
                boxes.append([(x+xs.min())/unit, (y+ys.min())/unit,
                              (x+xs.max()+1)/unit, (y+ys.max()+1)/unit])
        barriers = [c['bbox'] for c in independent]
        boxes = _merge_regions(boxes, typical/unit, barriers)
        # Keep a small source-edge allowance for antialiasing and downsampling
        # quantization. This is not the much larger connectivity margin, and
        # touching prose still requires support evidence below, not just bbox.
        edge_pad = 2/unit
        boxes = [[b[0]-edge_pad, b[1]-edge_pad, b[2]+edge_pad, b[3]+edge_pad] for b in boxes]
        boxes = _merge_regions(boxes, 0, [])
    # Enclosing boxes can include disconnected prose even if the underlying
    # connected-component mask does not. Resolve ownership before any export.
    pending = list(independent)
    # Detached short labels immediately beside map symbols are not proven
    # prose. Retain them conservatively; only a clear gap permits alpha clips.
    contacts = {id(c): _touches_support(np.asarray(c['quad'])*unit, protected,
                                       round(min(typical, c['element']['height']*unit)*.35))
                for c in pending}
    ambiguous_labels = {}
    figure_evidence = {}
    for c in pending:
        # Short labels beside a leader inside an actual figure remain
        # conservative. A standalone dotted rule beside prose is not a figure.
        value = ''.join(c['element'].get('text','').split())
        if 0 < len(value) <= 4:
            margin = round(min(typical,c['element']['height']*unit)*.35)
            ambiguous_labels[id(c)] = _touches_support(np.asarray(c['quad'])*unit,residual,margin)
    changed = True
    while changed:
        changed = False
        for c in pending[:]:
            for i, box in enumerate(boxes):
                if _overlap(box, c['bbox']):
                    label_contact = False
                    if ambiguous_labels.get(id(c),False):
                        key=tuple(box)
                        if key not in figure_evidence:
                            sx0,sy0,sx1,sy1=np.round(np.asarray(box)*unit).astype(int)
                            patch=protected[max(0,sy0):min(len(protected),sy1),max(0,sx0):min(protected.shape[1],sx1)]
                            figure_evidence[key]=bool((box[3]-box[1])*unit>typical*3 and patch.size and
                                                      np.count_nonzero(patch)>typical*typical*2)
                        label_contact=figure_evidence[key]
                    if not uncertain_page and not contacts[id(c)] and not label_contact:
                        # The bounding rectangle is only a crop envelope. It
                        # cannot by itself turn independent prose into a label.
                        continue
                    boxes[i] = _union(box, c['bbox'])
                    e = c['element']
                    retained.append(dict(textId=e['id'], text=e['text'], bbox=c['bbox'], polygon=e['polygon'],
                                         confidence=e.get('confidence'), reason='image-boundary-overlap'))
                    pending.remove(c)
                    changed = True
                    break
        if changed:
            boxes = _merge_regions(boxes, 0, [])
    regions = []
    for box in sorted(boxes, key=lambda b: (b[1], b[0])):
        # Quantize outwards once; metadata describes the actual source crop.
        x0 = max(0, int(np.floor(box[0]*scale)))
        y0 = max(0, int(np.floor(box[1]*scale)))
        x1 = min(rgb.shape[1], int(np.ceil(box[2]*scale)))
        y1 = min(rgb.shape[0], int(np.ceil(box[3]*scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        actual = [x0/scale, y0/scale, x1/scale, y1/scale]
        lines = [line for line in retained if _overlap(actual, line['bbox'])]
        exclusions = [dict(textId=c['element']['id'], polygon=c['quad'], bbox=c['bbox'])
                      for c in pending if _overlap(actual, c['bbox'])]
        regions.append(dict(bbox=actual, pixelBox=[x0, y0, x1, y1], retainedText=lines,
                            excludedText=exclusions,
                            contentType='mixed' if lines else 'image', reviewRequired=True,
                            boundary=dict(method='foreground-barriers-v2', exact=False,
                                          clip='alpha-text-exclusion' if exclusions else 'rectangle',
                                          reason='too-many-components' if uncertain_page else 'geometry-inferred')))
    return pending, regions, paper


def region_pixels(rgb, region, scale, *, excluded_mask=None):
    """Keep source RGB unchanged; alpha excludes only separated OCR polygons.

    The page reference remains the complete original image. No replacement
    paper, inpainting, or estimated graphic pixels are introduced into a crop.
    """
    x0, y0, x1, y1 = region['pixelBox']
    crop = rgb[y0:y1, x0:x1]
    rules = np.asarray(excluded_mask)[y0:y1, x0:x1] if excluded_mask is not None else None
    if not region.get('excludedText') and (rules is None or not np.any(rules)):
        return crop
    alpha = np.full(crop.shape[:2], 255, np.uint8)
    if rules is not None:
        alpha[rules > 0] = 0
    for line in region.get('excludedText', []):
        polygon = np.round(np.asarray(line['polygon'])*scale-[x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(alpha, polygon, 0)
    return np.dstack((crop, alpha))
