"""Reuse only hash-matched recognition evidence, never an old target layout."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


LEGACY_POLICIES = ('feature4-four-layer-v6-20261102',
                   'feature4-four-layer-v7-native-tables-20261102',
                   'feature4-text-v8-reverse-owner-20261103',
                   'feature4-text-v9-bounded-linkage-20261104',
                   'feature4-text-v10-owner80-20261104',
                   'feature4-text-v11-source-containment-20261105')


def read_evidence(data, config, build, roots, *, width, height):
    identity={k:v for k,v in config.items() if k != 'results_dir'}
    # Paths/model options must match, rather than trusting externally modified
    # source code or renaming an arbitrary JSON file to the new cache key.
    signature=json.dumps(identity,sort_keys=True,ensure_ascii=False).encode('utf8')
    for policy in LEGACY_POLICIES:
        key=hashlib.sha256(build.encode()+signature+policy.encode()+data).hexdigest()
        for root in roots:
            path=Path(root)/(key+'.json')
            if not path.is_file():continue
            try:
                result=json.loads(path.read_text(encoding='utf8'))
                if (result.get('rebuild_policy') != policy or result.get('width') != width or
                        result.get('height') != height):continue
                for name in ('layout_det_res','region_det_res'):
                    boxes=result[name]['boxes']
                    if not isinstance(boxes,list):raise ValueError('boxes')
                    ids=set()
                    for box in boxes:
                        rect=box['coordinate']
                        if box['id'] in ids:raise ValueError('duplicate detection')
                        ids.add(box['id'])
                        if (len(rect)!=4 or not all(math.isfinite(v) for v in rect) or
                            not 0 <= rect[0] < rect[2] <= width or not 0 <= rect[1] < rect[3] <= height):
                            raise ValueError('detection bounds')
                polygons=result['ocr_pipeline']['dt_polys']
                overall=result['overall_ocr_res']
                texts,scores,polys=overall['rec_texts'],overall['rec_scores'],overall['rec_polys']
                if not len(texts)==len(scores)==len(polys):raise ValueError('record counts')
                for poly in list(polygons)+list(polys):
                    if (len(poly)!=4 or any(len(point)!=2 for point in poly) or
                        any(not math.isfinite(v) for point in poly for v in point)):
                        raise ValueError('polygon')
                    if any(not 0 <= x <= width or not 0 <= y <= height for x,y in poly):
                        raise ValueError('polygon bounds')
                if any(not isinstance(t,str) for t in texts) or any(not math.isfinite(s) or not 0<=s<=1 for s in scores):
                    raise ValueError('recognition')
                # Strip old ownership, tables, fallback diagnoses and project
                # state. The current pipeline recomputes these from evidence.
                answer={name:deepcopy(result[name]) for name in ('layout_det_res','region_det_res',
                    'overall_ocr_res','ocr_pipeline')}
                answer['reuse_provenance']=dict(source_cache=str(path.resolve()),source_policy=policy,
                    source_cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    input_sha256=hashlib.sha256(data).hexdigest(),identity='exact-runtime-config-and-input-key')
                return answer
            except (OSError,ValueError,TypeError,KeyError):
                continue
    return None
