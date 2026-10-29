"""Small parameterized style registry; observations never become ground truth."""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np

TEMPLATES = {
    'text.body.v1': {'type':'text','role':'body'},
    'text.heading.v1': {'type':'text','role':'heading'},
    'table.ruled-grid.v1': {'type':'table','mergedCells':False},
    'image.original.v1': {'type':'image','replaceContent':False},
    'decoration.rect.v1': {'type':'decoration','shape':'rect'},
}


def observe_styles(page):
    text = [e for e in page['elements'] if e['type']=='text']
    typical = float(np.median([e.get('fontSize',12) for e in text])) if text else 12.
    for e in text:
        e['observedStyle'] = {k:e.get(k) for k in ('fontFamily','fontSize','fontWeight','fontStyle','color','lineHeight')}
        heading = not e.get('tableId') and e.get('fontSize',12) > typical*1.3 and len(e['text']) < 160
        e['contentType'] = 'heading' if heading else 'text'
        e['templateId'] = 'text.heading.v1' if heading else 'text.body.v1'
        e['styleEvidence'] = 'native' if e.get('source')=='native' else 'estimated-from-scan'


def resolve_styles(document):
    """Record actual chosen faces, including local font runs and safe fallbacks."""
    counts = Counter()
    for page in document['pages']:
        for e in page['elements']:
            if e['type']=='text':
                e['resolvedStyle'] = {k:e.get(k) for k in ('fontFamily','fontSize','fontWeight','fontStyle','color','lineHeight')}
                e['resolvedStyle']['runs'] = [dict(fontKey=r['fontKey'],family=r.get('family'),
                                                     start=r.get('start'),end=r.get('end')) for r in e.get('fontRuns',[])]
                e['styleMapping'] = dict(templateId=e.get('templateId','text.body.v1'),version=1,
                                         approximate=e.get('source')!='native',
                                         fallbackReasons=copy.deepcopy(e.get('fontFallbacks',[])))
            counts[e.get('templateId','image.original.v1')] += 1
    return dict(counts)
