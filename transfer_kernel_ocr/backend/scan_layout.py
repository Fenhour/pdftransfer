"""Conservative text extraction; artwork is retained, never reconstructed."""
from __future__ import annotations

import cv2
import numpy as np


def horizontal_rules(ink, typical):
    """Actual pixels of thin rules / repeated leaders, not nearby letter ink.

    Used only as evidence that a line does not own surrounding prose. The
    original pixels remain in image crops; no answer marks are guessed.
    """
    thin = np.zeros_like(ink)
    count, labels, parts, _ = cv2.connectedComponentsWithStats(ink, 8)
    for label, (x, y, w, h, _) in enumerate(parts[1:count], 1):
        stroke = h <= max(2., typical*.22) and w >= typical*2 and w >= h*4
        dot = h <= max(2., typical*.18) and w <= max(3., typical*.4)
        if stroke or dot:
            patch = thin[y:y+h, x:x+w]
            patch[labels[y:y+h, x:x+w] == label] = 255
    kernel = np.ones((1, max(3, round(typical*.45) | 1)), np.uint8)
    connected = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, kernel)
    long = cv2.morphologyEx(connected, cv2.MORPH_OPEN,
                          np.ones((1, max(8, round(typical*2))), np.uint8))
    return cv2.bitwise_and(thin, cv2.dilate(long, np.ones((1, 3), np.uint8)))


def partition_text(rgb, candidates, scale, *, contour_support=False, ignored_mask=None):
    """Return independent text and text retained inside the page image.

    This is a geometry/contrast heuristic, not a semantic layout model. Large
    non-text components (including frames around labels) protect the original
    composite. Uncertain text remains in pixels rather than damaging a figure.
    Analysis is bounded independently of the source page's pixel count.
    """
    if not candidates:
        return [], []
    ratio = min(1., 1600 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=ratio, fy=ratio) if ratio < 1 else rgb
    unit = scale * ratio
    height, width = small.shape[:2]
    border = np.concatenate((small[0], small[-1], small[:, 0], small[:, -1]))
    paper = np.median(border, axis=0)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    ink = (np.max(np.abs(small.astype(np.float32)-paper), axis=2) > 28).astype(np.uint8)*255
    if min(height, width) >= 31:
        ink |= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    if ignored_mask is not None:
        # Rebuilt decorative rules are separate vector objects. Ignore only
        # their strokes during analysis, never the content of their container.
        ignored = cv2.resize(np.asarray(ignored_mask, dtype=np.uint8), (width, height),
                             interpolation=cv2.INTER_NEAREST)
        ink[ignored > 0] = 0
    text_mask = np.zeros((height, width), np.uint8)
    heights = []
    for candidate in candidates:
        quad = np.round(np.asarray(candidate['quad'])*unit).astype(np.int32)
        cv2.fillConvexPoly(text_mask, quad, 255)
        heights.append(max(3., candidate['element']['height']*unit))
    typical = float(np.median(heights))
    # OCR may recognize a drop cap both on its own and as the first character
    # of a complete line. Replacing both would print overlapping glyphs.
    boxes=np.asarray([candidate['bbox'] for candidate in candidates])*unit
    areas=np.maximum(1.,(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]))
    overlaps=np.zeros(len(candidates),dtype=bool)
    for index,box in enumerate(boxes):
        intersection=np.maximum(0.,np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]))
        duplicate=intersection.prod(axis=1)/np.minimum(areas[index],areas)>.45
        duplicate[index]=False
        overlaps[index]=duplicate.any()
    residual = ink.copy()
    residual[cv2.dilate(text_mask, np.ones((3, 3), np.uint8)) > 0] = 0
    rules=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(3,round(typical*2)),1),np.uint8))
    rules|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(3,round(typical*4))),np.uint8))
    if contour_support:
        rules |= horizontal_rules(ink, typical)
    boundary_residual=residual.copy()
    boundary_residual[rules>0]=0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(residual, connectivity=8)
    graphics = []
    graphic_support = np.zeros_like(residual) if contour_support else None
    for component,(x, y, w, h, area) in enumerate(stats[1:count],1):
        # A thin rule alone is not evidence that neighboring prose is a figure.
        if min(w, h) < typical*.45 or area < max(20, typical*typical*.12):
            continue
        if (w > typical*3 and h > typical*2) or (h > typical*4 and w > typical*1.5):
            # A decorative book-note border may enclose ordinary prose. Keep
            # its pixels, allowing the many text lines in an empty interior to
            # become text. Substantial interior artwork remains protected.
            margin=max(3,round(min(typical*2,min(w,h)*.12)))
            enclosed=sum(x<box[0] and y<box[1] and box[2]<x+w and box[3]<y+h for box in boxes)
            interior=residual[y+margin:y+h-margin,x+margin:x+w-margin]
            if enclosed>=6 and area/(w*h)<.12 and interior.size and np.count_nonzero(interior)/interior.size<.025:
                continue
            # A plain frame around prose is not itself a text/figure composite.
            patch=residual[y:y+h,x:x+w]
            strip=max(2,round(typical*.18))
            frame=(np.count_nonzero(np.any(patch[:strip],axis=0))>w*.8 and
                   np.count_nonzero(np.any(patch[-strip:],axis=0))>w*.8 and
                   np.count_nonzero(np.any(patch[:,:strip],axis=1))>h*.8 and
                   np.count_nonzero(np.any(patch[:,-strip:],axis=1))>h*.8)
            if frame and area/(w*h)<.15:
                interior_art=any(i!=component and ix>x+strip and iy>y+strip and ix+iw<x+w-strip and iy+ih<y+h-strip
                                 and min(iw,ih)>typical*.7 and ink_area>typical*typical*.15
                                 for i,(ix,iy,iw,ih,ink_area) in enumerate(stats[1:count],1))
                if not interior_art:
                    continue
            graphics.append((x, y, x+w, y+h))
            if contour_support:
                # Distinguish actual support from padding outside a component.
                # Candidates inside a graphic envelope remain conservative:
                # whitespace alone cannot prove that a map label is prose.
                component_mask = (labels[y:y+h, x:x+w] == component).astype(np.uint8)
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(graphic_support[y:y+h, x:x+w], contours, -1, 255, cv2.FILLED)
    if contour_support:
        graphic_support = cv2.dilate(graphic_support, np.ones((3, 3), np.uint8))
    independent, retained = [], []
    for index,candidate in enumerate(candidates):
        element = candidate['element']
        x0, y0, x1, y1 = np.asarray(candidate['bbox'])*unit
        pad = max(2, typical*.35)
        a, b = max(0, int(x0-pad)), max(0, int(y0-pad))
        c, d = min(width, int(x1+pad)+1), min(height, int(y1+pad)+1)
        area = max(1., (x1-x0)*(y1-y0))
        inside_graphic = any(max(0, min(x1, gx1+pad)-max(x0, gx0-pad))*
                             max(0, min(y1, gy1+pad)-max(y0, gy0-pad))/area > .15
                             for gx0, gy0, gx1, gy1 in graphics)
        if contour_support:
            qa, qb = max(0, int(x0)), max(0, int(y0))
            qc, qd = min(width, int(np.ceil(x1))+1), min(height, int(np.ceil(y1))+1)
            polygon = np.round(np.asarray(candidate['quad'])*unit-[qa, qb]).astype(np.int32)
            polygon_mask = np.zeros((max(0, qd-qb), max(0, qc-qa)), np.uint8)
            if polygon_mask.size:
                cv2.fillConvexPoly(polygon_mask, polygon, 255)
            in_envelope = any(max(0, min(x1,gx1)-max(x0,gx0))*
                              max(0, min(y1,gy1)-max(y0,gy0))/area > .15
                              for gx0,gy0,gx1,gy1 in graphics)
            inside_graphic = in_envelope or bool(polygon_mask.size and np.any(
                (polygon_mask > 0) & (graphic_support[qb:qd, qa:qc] > 0)))
        ring = residual[b:d, a:c]
        if contour_support:
            # Artwork just outside this text polygon is not its background.
            # Actual contact was checked above; keep texture/noise evidence.
            ring = ring.copy()
            ring[graphic_support[b:d, a:c] > 0] = 0
        crowded = bool(ring.size and np.count_nonzero(ring)/ring.size > .12)
        # A partly missed glyph immediately outside the OCR box must not be
        # left beside a new (possibly misrecognized) replacement character.
        gy0,gy1=max(0,int(y0+typical*.15)),min(height,int(y1-typical*.15)+1)
        guards=[boundary_residual[gy0:gy1,max(0,int(x0-typical*.5)):max(0,int(x0))],
                boundary_residual[gy0:gy1,min(width,int(x1)+1):min(width,int(x1+typical*.5)+1)]]
        boundary_ink=any(guard.size and np.count_nonzero(guard)>max(8,guard.size*.10) for guard in guards)
        # Colored/photographic backgrounds and low confidence are kept intact.
        crop = small[b:d, a:c]
        # Neighboring recognized lines are not background texture. Exclude all
        # OCR polygons before measuring paper, especially in tight paragraphs.
        paper_mask = text_mask[b:d,a:c] == 0
        if contour_support:
            paper_mask &= graphic_support[b:d,a:c] == 0
        paper_pixels=crop[paper_mask] if crop.size else np.empty((0,3))
        edge = paper_pixels if len(paper_pixels) else border
        spread = float(np.percentile(np.max(np.abs(edge.astype(float)-np.median(edge, axis=0)), axis=1), 85))
        reason = ('overlapping-text' if overlaps[index] else 'graphic-overlap' if inside_graphic else 'partial-text-boundary' if boundary_ink
                  else 'complex-background' if crowded or spread > 35
                  else 'uncertain-ocr' if element.get('confidence', 0) < .8 else '')
        if reason:
            retained.append({'textId': element.get('id'), 'text': element['text'], 'bbox': candidate['bbox'],
                             'polygon': element['polygon'], 'confidence': element.get('confidence'), 'reason': reason})
        else:
            independent.append(candidate)
    return independent, retained
