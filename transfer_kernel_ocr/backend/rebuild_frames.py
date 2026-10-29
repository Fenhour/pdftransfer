"""Lightweight frame geometry, independent of OCR transcription confidence.

A rectangle may decorate prose without owning the text inside it.  These
heuristics accept only complete, simple frames with horizontal text evidence
and no substantial unexplained artwork.  They never modify source pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

from .rebuild_layout import foreground, _overlap
from .rebuild_tables import _bands


def frame_rule_mask(shape, frames, scale, *, margin=0):
    """Return only frame strokes for analysis/alpha exclusion, not interiors.

    ``shape`` is the destination image shape, and ``scale`` maps PDF points to
    that image's pixels.  The caller retains the complete original reference.
    A margin can compensate for analysis resampling; it must not hide a whole
    rectangular region or be used to paint replacement RGB into the source.
    """
    mask = np.zeros(tuple(shape[:2]), np.uint8)
    for frame in frames:
        support = frame.get('ruleSupport')
        if support is not None:
            # Final frames carry observed source support. A search margin is
            # useful during detection but is not evidence that neighboring
            # pixels belong to a rule, especially where text touches a frame.
            for observed in support['patches']:
                x0, y0, x1, y1 = observed['pixelBox']
                patch = np.zeros((y1-y0, x1-x0), np.uint8)
                flat = patch.ravel()
                for start, length in observed['runs']:
                    flat[start:start+length] = 255
                box = np.round(np.asarray([x0, y0, x1, y1])*scale/support['scale']).astype(int)
                a, b, c, d = box
                if c <= a or d <= b:
                    continue
                patch = cv2.resize(patch, (c-a, d-b), interpolation=cv2.INTER_NEAREST)
                left, top, right, bottom = max(0, a), max(0, b), min(mask.shape[1], c), min(mask.shape[0], d)
                if right > left and bottom > top:
                    mask[top:bottom, left:right] |= patch[top-b:bottom-b, left-a:right-a]
            continue
        for segment in frame.get('ruleSegments', ()):
            start = tuple(np.round(np.asarray(segment['start'])*scale).astype(int))
            end = tuple(np.round(np.asarray(segment['end'])*scale).astype(int))
            thickness = max(1, round(segment['width']*scale)) + max(0, int(margin))*2
            cv2.line(mask, start, end, 255, thickness, cv2.LINE_8)
    return mask


def _source_rule_support(rgb, segments, candidates, scale, typical, paper):
    """Observe thin continuous source strokes; protect ambiguous text contact.

    Only a narrow edge band is examined at source resolution. Long-axis
    opening removes glyph protrusions from rule evidence. Where remaining
    ink meets that evidence, keep the local source intersection as well: a
    scan cannot tell which object originally owned an overlapping pixel.
    The compact runs contain mask geometry, never RGB or OCR transcription.
    """
    patches = []
    length = max(13, round(typical*scale*2) | 1)
    for segment in segments:
        points = np.asarray([segment['start'], segment['end']])*scale
        allowance = segment['width']*scale/2+max(2, scale)
        x0, y0 = np.maximum(0, np.floor(points.min(axis=0)-allowance).astype(int))
        x1, y1 = np.minimum([rgb.shape[1], rgb.shape[0]], np.ceil(points.max(axis=0)+allowance+1).astype(int))
        crop = rgb[y0:y1, x0:x1]
        ink = (np.max(np.abs(crop.astype(np.float32)-paper), axis=2) > 3).astype(np.uint8)*255
        origin = np.asarray([x0, y0])/scale
        local = dict(start=(np.asarray(segment['start'])-origin).tolist(),
                     end=(np.asarray(segment['end'])-origin).tolist(), width=segment['width'])
        band = frame_rule_mask(crop.shape, [dict(ruleSegments=[local])], scale, margin=max(1, round(scale/2)))
        horizontal = abs(local['end'][0]-local['start'][0]) >= abs(local['end'][1]-local['start'][1])
        kernel = np.ones((1, length) if horizontal else (length, 1), np.uint8)
        observed = cv2.morphologyEx(ink & band, cv2.MORPH_OPEN, kernel)
        unknown = ink & ~observed
        # A perpendicular side at either corner is still frame evidence.
        # Restrict contact protection to the side's open interior instead.
        corner = max(2, round(segment['width']*scale*2))
        if horizontal:
            unknown[:, :corner] = unknown[:, -corner:] = 0
        else:
            unknown[:corner] = unknown[-corner:] = 0
        # A letter joined to a same-color frame must keep its intersection
        # too, rather than trusting a geometric strip to split the objects.
        protection = max(1, round(typical*scale*.55))
        safe = observed.copy()
        contact = unknown & cv2.dilate(observed, np.ones((3, 3), np.uint8))
        if horizontal:
            positions = np.any(contact, axis=0).astype(np.uint8)[None, :]
            protected = cv2.dilate(positions, np.ones((1, protection*2+1), np.uint8))[0] > 0
            safe[:, protected] = 0
        else:
            positions = np.any(contact, axis=1).astype(np.uint8)[:, None]
            protected = cv2.dilate(positions, np.ones((protection*2+1, 1), np.uint8))[:, 0] > 0
            safe[protected] = 0
        for candidate in candidates:
            quad = np.round(np.asarray(candidate['quad'])*scale-[x0, y0]).astype(np.int32)
            cv2.fillConvexPoly(safe, quad, 0)
        occupied = np.flatnonzero(safe.ravel())
        if len(occupied):
            starts = np.r_[0, np.flatnonzero(np.diff(occupied) != 1)+1]
            ends = np.r_[starts[1:], len(occupied)]
            runs = [[int(occupied[a]), int(b-a)] for a, b in zip(starts, ends)]
        else:
            runs = []
        patches.append(dict(pixelBox=[int(x0), int(y0), int(x1), int(y1)], runs=runs))
    return dict(scale=scale, patches=patches)


def _largest_gap(present):
    missing = np.flatnonzero(~present)
    if not len(missing):
        return 0
    groups = np.split(missing, np.flatnonzero(np.diff(missing) != 1)+1)
    return max(map(len, groups))


def _complete_edge(patch, axis, typical):
    support = np.any(patch > 0, axis=axis)
    return (support.size and np.mean(support) >= .97 and
            _largest_gap(support) <= max(2, round(typical*.2)))


def _text_rows(boxes, typical):
    rows = []
    for box in sorted(boxes, key=lambda b: (b[1]+b[3], b[0])):
        center = (box[1]+box[3])/2
        match = next((r for r in rows if abs(r['center']-center) <= typical*.45), None)
        if match is None:
            rows.append(dict(center=center, left=box[0], right=box[2],
                             top=box[1], bottom=box[3]))
        else:
            match.update(left=min(match['left'], box[0]), right=max(match['right'], box[2]),
                         top=min(match['top'], box[1]), bottom=max(match['bottom'], box[3]))
    return rows


def _prose_evidence(boxes, typical, frame_width):
    rows = _text_rows(boxes, typical)
    if not rows:
        return []
    widths = np.asarray([r['right']-r['left'] for r in rows])
    # A small single-line note is supported only if its transcription occupies
    # a real line, rather than a scatter of isolated labels in a diagram.
    if len(rows) <= 2:
        return rows if np.max(widths) >= max(typical*4, frame_width*.18) else []
    left = np.asarray([r['left'] for r in rows])
    right = np.asarray([r['right'] for r in rows])
    aligned = min(np.percentile(left, 80)-np.percentile(left, 20),
                  np.percentile(right, 80)-np.percentile(right, 20)) <= frame_width*.18
    long_lines = np.count_nonzero(widths >= max(typical*3, frame_width*.25))
    return rows if aligned and long_lines >= max(2, len(rows)*.5) else []


def _interior_is_text(ink, coverage, rules, typical):
    """Allow missed glyphs/lines, reject large or textured graphic support."""
    covered = cv2.dilate(coverage, np.ones((3, 3), np.uint8))
    residual = ink.copy()
    residual[(covered > 0) | (rules > 0)] = 0
    # Dense non-text backgrounds cannot be explained away by the frame.
    if np.count_nonzero(residual) > residual.size*.10:
        return False, 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
    significant = []
    for x, y, w, h, area in stats[1:count]:
        if area <= max(3, typical*.15):
            continue
        # An unrecognized word can be long, but it still has text-line height.
        # Closed figures, connected map roads and photographic patches do not.
        if (h > typical*1.5 or w > typical*12 or area > typical*typical*5 or
                w > typical*3 and h < typical*.20):
            return False, count-1
        significant.append([x, y, x+w, y+h])
    if not significant:
        return True, count-1
    residual_rows = _text_rows(significant, typical)
    # Significant unexplained support must have glyph-scale height, rather
    # than many thin map/grid strokes or arbitrary dots across the interior.
    for row in residual_rows:
        if row['bottom']-row['top'] > typical*1.5:
            return False, count-1
    return True, count-1


def text_frames(rgb, candidates, scale, *, excluded_boxes=()):
    """Detect simple prose frames separately from grids and OCR accuracy.

    Returned records retain the existing frame/table coordinates and style
    fields.  ``candidateIndices`` describes containment, *not* permission to
    force uncertain OCR into editable text.  All candidates still require the
    normal text/image ownership checks.  ``ruleSegments`` permits removing
    only the frame strokes from an analysis mask, preserving unknown content.
    """
    if not candidates or scale <= 0:
        return []
    ratio = min(1., 1600/max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=ratio, fy=ratio) if ratio < 1 else rgb
    unit = scale*ratio
    ink, paper_color = foreground(small)
    paper = np.asarray([int(paper_color[i:i+2], 16) for i in (1, 3, 5)], dtype=np.float32)
    horizontal_candidates = [c for c in candidates
                             if abs(c['element'].get('rotation', 0)) <= 4]
    heights = [c['element']['height']*unit for c in horizontal_candidates]
    if not heights:
        return []
    typical = max(6., float(np.median(heights)))
    length = max(20, round(typical*2))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal | vertical, 8)
    frames = []
    for x, y, w, h, _ in stats[1:count]:
        if min(w, h) < typical*3:
            continue
        xs = _bands(np.count_nonzero(vertical[y:y+h, x:x+w], axis=0), h*.90)
        ys = _bands(np.count_nonzero(horizontal[y:y+h, x:x+w], axis=1), w*.90)
        if len(xs) != 2 or len(ys) != 2:
            continue
        left, right = [x+p for p, _ in xs]
        top, bottom = [y+p for p, _ in ys]
        widths = [v for _, v in xs+ys]
        if max(widths) > typical*.45:
            continue
        stroke = max(1, round(max(widths)/2)+1)
        # Check all four source edges, including the longest gap.  Projection
        # totals alone can accept a rectangle with a missing side segment.
        edge_patches = [ink[max(0, round(top)-stroke):round(top)+stroke+1, round(left):round(right)+1],
                        ink[max(0, round(bottom)-stroke):round(bottom)+stroke+1, round(left):round(right)+1],
                        ink[round(top):round(bottom)+1, max(0, round(left)-stroke):round(left)+stroke+1],
                        ink[round(top):round(bottom)+1, max(0, round(right)-stroke):round(right)+stroke+1]]
        if not all(_complete_edge(p, 0 if i < 2 else 1, typical) for i, p in enumerate(edge_patches)):
            continue
        box = [max(0., (x-2)/unit), max(0., (y-2)/unit),
               min(small.shape[1]/unit, (x+w+2)/unit), min(small.shape[0]/unit, (y+h+2)/unit)]
        if any(_overlap(box, excluded) > 0 for excluded in excluded_boxes):
            continue
        assigned, text_boxes = [], []
        coverage = np.zeros((h, w), np.uint8)
        for index, candidate in enumerate(candidates):
            b = candidate['bbox']
            if not _overlap(box, b):
                continue
            # A watermark or another known line can cross the outline.  Its
            # pixels are text evidence even if the line belongs outside this
            # frame; containment and rendering ownership remain separate.
            quad = np.round(np.asarray(candidate['quad'])*unit-[x, y]).astype(np.int32)
            cv2.fillConvexPoly(coverage, quad, 255)
            area = max(1e-6, (b[2]-b[0])*(b[3]-b[1]))
            if _overlap(box, b)/area < .8:
                continue
            assigned.append(index)
            if abs(candidate['element'].get('rotation', 0)) <= 4:
                text_boxes.append(np.asarray(b)*unit-[x, y, x, y])
        rows = _prose_evidence(text_boxes, typical, w)
        if not rows:
            continue
        # Ignore only the outer four strokes.  Internal grid lines and shapes
        # remain evidence against treating a table/map as a prose frame.
        segments = [dict(start=[left/unit, top/unit], end=[right/unit, top/unit], width=ys[0][1]/unit),
                    dict(start=[left/unit, bottom/unit], end=[right/unit, bottom/unit], width=ys[1][1]/unit),
                    dict(start=[left/unit, top/unit], end=[left/unit, bottom/unit], width=xs[0][1]/unit),
                    dict(start=[right/unit, top/unit], end=[right/unit, bottom/unit], width=xs[1][1]/unit)]
        provisional = dict(ruleSegments=segments)
        rules = frame_rule_mask(small.shape, [provisional], unit, margin=1)[y:y+h, x:x+w]
        is_text, unknown_components = _interior_is_text(ink[y:y+h, x:x+w], coverage, rules, typical)
        if not is_text:
            continue
        colors = small[y:y+h, x:x+w][(rules > 0) & (ink[y:y+h, x:x+w] > 0)]
        color = '#'+''.join(f'{int(v):02x}' for v in np.median(colors, axis=0))
        cols, row_coordinates = [left/unit, right/unit], [top/unit, bottom/unit]
        frames.append(dict(id=f'frame-{len(frames)+1}', bbox=box, columns=cols, rows=row_coordinates,
                           cells=[dict(row=0, column=0, rowspan=1, colspan=1,
                                       bbox=[cols[0], row_coordinates[0], cols[1], row_coordinates[1]],
                                       textIds=[candidates[i]['element']['id'] for i in assigned])],
                           candidateIndices=assigned, contentType='decoration', reviewRequired=True,
                           structureStatus='heuristic-prose-frame', templateId='decoration.rect.v1',
                           observedStyle=dict(borderColor=color, borderWidth=float(np.median(widths))/unit),
                           ruleSegments=segments,
                           ruleSupport=_source_rule_support(rgb, segments, candidates, scale, typical/unit, paper),
                           frameEvidence=dict(method='complete-frame-text-interior-v1', textRows=len(rows),
                                              unexplainedComponents=unknown_components,
                                              ocrConfidenceRequired=False)))
    return frames
