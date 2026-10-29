"""Recognize only demonstrably simple ruled grids; preserve other tables as art."""
from __future__ import annotations

import cv2
import numpy as np

from .rebuild_layout import foreground, _overlap


def _bands(projection, threshold):
    indices = np.flatnonzero(projection >= threshold)
    groups = []
    for value in indices:
        if groups and value == groups[-1][-1]+1:
            groups[-1].append(int(value))
        else:
            groups.append([int(value)])
    return [(float(np.mean(g)), len(g)) for g in groups]


def ruled_tables(rgb, candidates, scale, *, include_frames=False):
    """Return conservative grid candidates and their OCR candidate indices.

    No merged-cell inference, chart-data recovery or neural layout claims.
    Rules must span the grid; every text box must fit exactly one cell, and
    substantial unexplained ink disqualifies the grid rather than disappearing.
    """
    ratio = min(1.,1600/max(rgb.shape[:2]))
    small = cv2.resize(rgb,None,fx=ratio,fy=ratio) if ratio < 1 else rgb
    unit = scale*ratio
    ink, _ = foreground(small)
    heights = [c['element']['height']*unit for c in candidates]
    typical = max(6.,float(np.median(heights))) if heights else 10*unit
    length = max(20,round(typical*2))
    horizontal = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,length),np.uint8))
    vertical = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((length,1),np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal|vertical,8)
    tables = []
    for x,y,w,h,_ in stats[1:count]:
        if min(w,h) < typical*3:
            continue
        xs = _bands(np.count_nonzero(vertical[y:y+h,x:x+w],axis=0),h*.88)
        ys = _bands(np.count_nonzero(horizontal[y:y+h,x:x+w],axis=1),w*.88)
        frame=len(xs)==len(ys)==2
        if not (2 <= len(xs) <= 35 and 2 <= len(ys) <= 100) or frame and not include_frames:
            continue
        if not frame and min(len(xs),len(ys))<3:
            continue
        cols = [(x+p)/unit for p,_ in xs]
        rows = [(y+p)/unit for p,_ in ys]
        if min(np.diff(cols)) < 5 or min(np.diff(rows)) < 5:
            continue
        box = [max(0.,(x-2)/unit),max(0.,(y-2)/unit),
               min(small.shape[1]/unit,(x+w+2)/unit),min(small.shape[0]/unit,(y+h+2)/unit)]
        cells = [dict(row=r,column=c,rowspan=1,colspan=1,bbox=[cols[c],rows[r],cols[c+1],rows[r+1]],textIds=[])
                 for r in range(len(rows)-1) for c in range(len(cols)-1)]
        assigned = []
        invalid = False
        coverage = np.zeros_like(ink[y:y+h,x:x+w])
        for index, candidate in enumerate(candidates):
            b = candidate['bbox']
            if not _overlap(box,b):
                continue
            if abs(candidate['element'].get('rotation',0)) > 2 or candidate['element'].get('confidence',0) < .8:
                invalid = True
                break
            matches = [cell for cell in cells if b[0] >= cell['bbox'][0] and b[1] >= cell['bbox'][1]
                       and b[2] <= cell['bbox'][2] and b[3] <= cell['bbox'][3]]
            if len(matches) != 1:
                invalid = True
                break
            matches[0]['textIds'].append(candidate['element']['id'])
            assigned.append(index)
            quad = np.round(np.asarray(candidate['quad'])*unit-[x,y]).astype(np.int32)
            cv2.fillConvexPoly(coverage,quad,255)
        occupied = [c for c in cells if c['textIds']]
        if invalid or not occupied:
            continue
        if not frame and (len(occupied)<2 or len({c['row'] for c in occupied})<2 or len({c['column'] for c in occupied})<2):
            continue
        rules = (horizontal|vertical)[y:y+h,x:x+w]
        covered = cv2.dilate(rules|coverage,np.ones((3,3),np.uint8))
        unexplained = (ink[y:y+h,x:x+w]>0) & (covered==0)
        if np.count_nonzero(unexplained) > max(5,w*h*.0005):
            continue
        colors = small[y:y+h,x:x+w][rules>0]
        color = '#'+''.join(f'{int(v):02x}' for v in np.median(colors,axis=0))
        tables.append(dict(id=f'{"frame" if frame else "table"}-{len(tables)+1}',bbox=box,columns=cols,rows=rows,cells=cells,
                           candidateIndices=assigned,contentType='decoration' if frame else 'table',reviewRequired=True,
                           structureStatus='heuristic-simple-frame' if frame else 'heuristic-simple-grid',
                           observedStyle=dict(borderColor=color,borderWidth=float(np.median([v for _,v in xs+ys]))/unit),
                           templateId='decoration.rect.v1' if frame else 'table.ruled-grid.v1'))
    return tables
