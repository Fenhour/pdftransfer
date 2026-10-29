"""Bounded font fitting for typed reconstruction, independent of legacy matching.

Region/block membership is supplied by layout. Samples choose a safe face before
its glyph metrics are used; words share a visual line scale and baseline unless
their own width, height and alignment support a real size difference. Scores and
budgets describe fitting work, not OCR or font identification accuracy.
"""
from __future__ import annotations

from .usage import tracked

from collections import Counter, OrderedDict, defaultdict
import math
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .font_library import (_distance, _ink, _load_appearance, _normal, _tight,
                           get_library, ink_metrics, valid_runs)
from .progress import progress_span, report


MAX_FACES = 12
MAX_PAGE_SAMPLES = 12
MAX_BLOCK_SAMPLES = 2
MAX_SAMPLE_WORDS = 3
MAX_PAGE_STYLE_WORDS = 96
MAX_LINE_STYLE_WORDS = 6
MAX_STYLE_FACES = 4
MAX_TEMPLATE_CACHE = 256
_STYLE = ('fontFamily', 'fontWeight', 'fontStyle', 'fontSize')
_COMMON = ('timesnewroman', 'arial', 'calibri', 'cambria', 'georgia',
           'couriernew', 'dejavuserif', 'dejavusans', 'liberationserif',
           'liberationsans', 'simsun', 'microsoftyahei', 'simhei', 'noto')


def _safe(face):
    return face.outline == 'truetype' and not face.variable


def _needed(text):
    return frozenset(ord(c) for c in text if not c.isspace())


def _shortlist(lib, text, family, cache):
    """Balance families and normal/bold/italic faces without rasterizing a catalog."""
    needed = _needed(text)
    key = (needed, _normal(family))
    shortlists = cache.setdefault('shortlists', OrderedDict())
    if key in shortlists:
        shortlists.move_to_end(key)
        return [lib.faces[k] for k in shortlists[key] if k in lib.faces]
    faces = [f for f in lib.faces.values() if _safe(f) and needed <= f.coverage]
    wanted = _normal(family)
    aliases = {'sansserif': 'arial', 'serif': 'timesnewroman', 'monospace': 'couriernew'}
    wanted = aliases.get(wanted, wanted)
    faces.sort(key=lambda f: (
        _normal(f.family) != wanted,
        next((i for i, name in enumerate(_COMMON) if name == _normal(f.family)),
             len(_COMMON)+next((i for i,name in enumerate(_COMMON) if name in _normal(f.family)),len(_COMMON))),
        any(s in f.style.lower() for s in ('narrow', 'condensed', 'expanded', 'black', 'light')),
        f.bold, f.italic, f.family, f.style))
    families = defaultdict(list)
    for face in faces:
        families[face.family].append(face)
    # Keep serif and sans alternatives. Ordinary proportional text gets a
    # second family of its hinted generic, rather than spending a third of
    # the fixed budget on monospaced variants regardless of the input style.
    chosen = []
    generic_hint = faces[0].generic if faces else ''
    generics = (generic_hint,'serif','sans-serif')
    for generic in generics:
        candidate = next((f.family for f in faces if f.generic == generic), None)
        if candidate and candidate not in chosen:
            chosen.append(candidate)
    for face in faces:
        if len(chosen)>=3:
            break
        if face.generic==generic_hint and face.family not in chosen:
            chosen.append(face.family)
    for face in faces:
        if len(chosen) >= 3:
            break
        if face.family not in chosen:
            chosen.append(face.family)
    result = []
    for name in chosen:
        variants = {}
        for face in families[name]:
            variants.setdefault((face.bold, face.italic), face)
        result.extend(variants.values())
    result = result[:MAX_FACES]
    shortlists[key] = [f.key for f in result]
    while len(shortlists) > 128:
        shortlists.popitem(last=False)
    return result


def _template(lib, face, text, cache):
    templates = cache.setdefault('templates', OrderedDict())
    key = (face.key, text)
    if key in templates:
        templates.move_to_end(key)
        return templates[key]
    font = lib.pil_font(face.key, 48)
    x0, y0, x1, y1 = font.getbbox(text)
    if x1 <= x0 or y1 <= y0 or x1-x0 > 16000:
        return None
    image = Image.new('L', (x1-x0+4, y1-y0+4))
    ImageDraw.Draw(image).text((2-x0, 2-y0), text, font=font, fill=255)
    ink = _tight(np.asarray(image))
    value = ink[0] if ink else None
    if value is not None:
        if value.shape[1] > 2048:
            value = cv2.resize(value,(2048,max(1,round(value.shape[0]*2048/value.shape[1]))))
        templates[key] = value
        while len(templates) > MAX_TEMPLATE_CACHE:
            templates.popitem(last=False)
    return value


def _score(lib, face, text, mask, cache, stats, counter):
    stats[counter] += 1
    template = _template(lib, face, text, cache)
    return _distance(mask, template) if template is not None else 2.


def _clean_evidence(mask):
    """Ignore thin long rules for fitting; original appearance pixels are untouched."""
    cleaned = mask.copy()
    count, labels, parts, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    for label in range(1, count):
        _, _, w, h, _ = parts[label]
        if w >= mask.shape[1]*.60 and w > h*15 and h <= max(2, mask.shape[0]*.13):
            cleaned[labels == label] = 0
    # Dense OCR quads can include a few descenders from the preceding line or
    # ascenders from the following line. Their floating components must not
    # increase this line's height or become false word gaps. Keep components
    # attached to the dominant horizontal body; this is fitting evidence only.
    rows = np.count_nonzero(cleaned,axis=1)
    if cleaned.shape[1] >= cleaned.shape[0]*3 and rows.max(initial=0) >= 12:
        dense = rows >= max(3,float(rows.max())*.35)
        starts = np.flatnonzero(dense & ~np.r_[False,dense[:-1]])
        ends = np.flatnonzero(dense & ~np.r_[dense[1:],False])+1
        if len(starts):
            top,bottom = max(zip(starts,ends),key=lambda b:int(rows[b[0]:b[1]].sum()))
            count,labels,parts,_ = cv2.connectedComponentsWithStats(cleaned,8)
            selected = set(np.unique(labels[top:bottom]))-{0}
            if selected:
                kept = np.where(np.isin(labels,list(selected)),255,0).astype(np.uint8)
                # Floating accents/dots inside the established line envelope
                # can contribute, but cannot expand it toward adjacent lines.
                support = _tight(kept)
                if support:
                    _,(_,y,_,h) = support
                    for label in range(1,count):
                        if label in selected:
                            continue
                        _,sy,_,sh,_ = parts[label]
                        if sy >= y-1 and sy+sh <= y+h+1:
                            kept[labels==label] = 255
                    if np.count_nonzero(kept) >= np.count_nonzero(cleaned)*.60:
                        cleaned = kept
    tight = _tight(cleaned)
    return tight if tight else (mask, (0, 0, mask.shape[1], mask.shape[0]))


def _word_groups(mask):
    """Observed ink groups, before any association with OCR words."""
    # Median component height is stable when an underline inflated the OCR box.
    count, _, parts, _ = cv2.connectedComponentsWithStats(mask, 8)
    heights = [int(p[3]) for p in parts[1:count] if p[4] >= 3 and p[3] > 2]
    height = float(np.median(heights)) if heights else mask.shape[0]
    kernel = np.ones((1, max(3, round(height*.22) | 1)), np.uint8)
    columns = np.any(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), axis=0)
    starts = np.flatnonzero(columns & ~np.r_[False, columns[:-1]])
    ends = np.flatnonzero(columns & ~np.r_[columns[1:], False])+1
    return list(zip(starts.tolist(), ends.tolist()))


def _regions(mask, text):
    """Use exact whitespace alignment, otherwise keep the intact line; no beam search."""
    words = list(re.finditer(r'\S+\s*', text))
    if len(words) < 2 or len(words) > 80 or '\n' in text:
        return [(0, len(text), 0, mask.shape[1])]
    groups = _word_groups(mask)
    # Punctuation and wide letter pairs can add small gaps inside a word. Merge
    # those first by the observed gap ordering; this has no text/template beam.
    if len(groups) > len(words):
        gaps = [groups[i+1][0]-groups[i][1] for i in range(len(groups)-1)]
        cuts = set(sorted(range(len(gaps)),key=lambda i:gaps[i],reverse=True)[:len(words)-1])
        joined, start = [], groups[0][0]
        for i,(_,end) in enumerate(groups):
            if i == len(groups)-1 or i in cuts:
                joined.append((start,end))
                if i+1 < len(groups):
                    start = groups[i+1][0]
        groups = joined
    if len(groups) != len(words) or any(not np.any(mask[:,a:b]) for a,b in groups):
        return [(0, len(text), 0, mask.shape[1])]
    return [(word.start(), word.end(), a, b) for word, (a, b) in zip(words, groups)]


def _pieces(evidence, lib=None):
    mask, element = evidence['mask'], evidence['element']
    result = []
    for start, end, x0, x1 in _regions(mask, element['text']):
        tight = _tight(mask[:,x0:x1])
        if tight:
            target, (x, y, w, h) = tight
            result.append(dict(start=start, end=end, text=element['text'][start:end],
                               mask=target, box=(x0+x, y, w, h)))
    if len(result) > 1 and lib is not None:
        face = lib.resolve(element['text'],element.get('fontFamily',''),safe_only=True)
        if face is None:
            result = []
        else:
            scales = []
            for piece in result:
                value = piece['text'].strip()
                metrics = ink_metrics(face.key,value)
                expected = max(1.,metrics[2]-metrics[0])
                piece['_widthScale'] = piece['box'][2]/expected
                if len(re.sub(r'\W','',value)) >= 3:
                    scales.append((piece['_widthScale'],min(20,len(value))))
            reference = _weighted_median(scales) if scales else 1.
            # Equal group counts are not evidence of correct alignment. One
            # missed OCR space shifts every following word, and floating ink
            # can otherwise assign an entire word to a one-pixel fragment.
            valid = all((.55 < p['_widthScale']/reference < 1.65) if len(re.sub(r'\W','',p['text'])) >= 3
                        else (.25 < p['_widthScale']/reference < 2.7) for p in result)
            if not valid:
                result = []
            for piece in result:
                piece.pop('_widthScale',None)
    if not result:
        result = [dict(start=0,end=len(element['text']),text=element['text'],mask=mask,
                       box=(0,0,mask.shape[1],mask.shape[0]))]
    return result


def _select_indices(count, maximum):
    if count <= maximum:
        return list(range(count))
    return sorted(set(round(i*(count-1)/(maximum-1)) for i in range(maximum))) if maximum > 1 else [0]


def _style_pieces(evidence, lib):
    """Recover only geometrically verified edge words for style sampling.

    A missing interior OCR space can invalidate all word positions. The first
    and last explicit OCR words can still correspond to the outer ink groups;
    checking their aspect ratios avoids stretching an unaligned whole line to
    choose a family. These samples never split or rewrite the output text.
    """
    pieces = evidence['pieces']
    if len(pieces) > 1:
        return pieces
    element,mask = evidence['element'],evidence['mask']
    words = list(re.finditer(r'\S+',element['text']))
    groups = _word_groups(mask)
    # Connected dots, thin capitals and kerning can leave extra intra-word
    # groups. Join only short gaps relative to this cleaned line's ink height;
    # the subsequent aspect check must still support each explicit edge word.
    joined = []
    for start,end in groups:
        if joined and start-joined[-1][1] <= max(2,round(mask.shape[0]*.22)):
            joined[-1] = (joined[-1][0],end)
        else:
            joined.append((start,end))
    groups = joined
    if len(words)<2 or len(groups)<2:
        return pieces
    face = lib.resolve(element['text'],element.get('fontFamily',''),safe_only=True)
    if face is None:
        return pieces
    result = []
    for word,(x0,x1) in ((words[0],groups[0]),(words[-1],groups[-1])):
        text = word.group()
        if not 3<=len(re.sub(r'\W','',text))<=18:
            continue
        ink = _tight(mask[:,x0:x1])
        if not ink:
            continue
        target,(x,y,w,h) = ink
        metrics = ink_metrics(face.key,text)
        ratio = (w/max(1,h))/((metrics[2]-metrics[0])/max(1,metrics[3]-metrics[1]))
        if .72 < ratio < 1.38:
            result.append(dict(text=text,mask=target,box=(x0+x,y,w,h),
                               start=word.start(),end=word.end()))
    return result or pieces


def _group_key(element):
    # Without layout membership each line is its own group, never a whole page.
    region = element.get('regionId', element.get('tableId', element.get('frameId', element['id'])))
    block = element.get('blockId', element['id'])
    height = max(1., element['height'])
    cased = ''.join(c for c in element['text'] if c.isalpha() and c.lower()!=c.upper())
    capitals = len(cased)>=4 and cased.isupper()
    return str(region), str(block), round(math.log(height)/math.log(1.35)), capitals


def _choose_face(lib, samples, fallback, faces, cache, stats):
    scores = Counter()
    weights = Counter()
    for sample in samples:
        source = sample.get('stylePieces',sample['pieces'])
        pieces = [p for p in source if len(re.sub(r'\W', '', p['text'])) >= 3]
        if not pieces:
            # A couple of digits plus leaders/dashes is insufficient evidence
            # for a different family, bold or italic face.
            continue
        for index in _select_indices(len(pieces), MAX_SAMPLE_WORDS):
            piece = pieces[index]
            text = piece['text'].strip()
            if not text:
                continue
            aligned = len(sample['pieces'])>1 or source is not sample['pieces']
            n = min(16, max(1, len(text))) if aligned else min(6,len(text))*.20
            for face in faces:
                if _needed(text) <= face.coverage:
                    distance = _score(lib,face,text,piece['mask'],cache,stats,'faceComparisons')
                    scores[face.key] += n*distance
                    weights[face.key] += n
    if not scores:
        return fallback
    def rank(face):
        # Near ties at low scan resolution are weaker than the supplied style
        # hint. Genuine serif / bold / italic differences can exceed these
        # small priors; unreliable whole-line spacing cannot dominate a group.
        prior = (.065*(face.generic!=fallback.generic)+
                 .025*(face.bold!=fallback.bold)+.025*(face.italic!=fallback.italic))
        return scores[face.key]/weights[face.key]+prior if weights[face.key] else 2.
    return min(faces,key=rank)


def _weighted_median(values):
    values = sorted(values)
    half = sum(weight for _, weight in values)/2
    consumed = 0
    for value, weight in values:
        consumed += weight
        if consumed >= half:
            return value
    return values[-1][0]


def _draft(lib, evidence, face, cache, stats, review_limit):
    element = evidence['element']
    pieces = evidence['pieces']
    # Local review is limited to this safe family's normal/bold/italic variants.
    variants = [f for f in lib.faces.values() if _safe(f) and f.family == face.family]
    by_style = {(face.bold, face.italic): face}
    for variant in variants:
        by_style.setdefault((variant.bold, variant.italic), variant)
    variants = list(by_style.values())[:MAX_STYLE_FACES]
    eligible = [i for i,p in enumerate(pieces) if 3 <= len(re.sub(r'\W', '', p['text'])) <= 40]
    reviewed = set(_select_indices(len(eligible), min(review_limit, MAX_LINE_STYLE_WORDS))) if review_limit else set()
    reviewed = {eligible[i] for i in reviewed}
    px, py = element['width']/evidence['image'].width, element['height']/evidence['image'].height
    bx, by = evidence['offset']
    result = []
    for index, piece in enumerate(pieces):
        text = piece['text']
        chosen = face
        score = None
        if index in reviewed:
            scores = {f.key: _score(lib, f, text.strip(), piece['mask'], cache, stats, 'wordComparisons')
                      for f in variants if _needed(text) <= f.coverage}
            if scores:
                best = min(scores, key=scores.get)
                score = scores[best]
                if best != face.key and scores.get(face.key, 2.)-scores[best] >= .06:
                    chosen = lib.faces[best]
            stats['styleWordsReviewed'] += 1
        x,y,w,h = piece['box']
        metrics = ink_metrics(chosen.key, text.strip() or text)
        gh, gw = max(1, metrics[3]-metrics[1]), max(1, metrics[2]-metrics[0])
        height_size = h*py/gh*64
        width_size = w*px/gw*64
        # Width comes from the intact word/line, while OCR crop height may
        # contain neighboring ink. Validated word geometry supports a genuine
        # size change; a tall crop alone never does.
        size = width_size
        alnum = re.sub(r'[^A-Za-z0-9]','',text)
        if len(pieces)==1 and len(alnum)>=8 and width_size>height_size*1.18:
            # An intact line can contain source spaces / tracking omitted by
            # OCR. Enlarging every glyph to consume that width would create
            # tall, horizontally squeezed text. The cleaned ink still gives
            # direct evidence of the visible character height.
            size = height_size
            stats['intactLineSizeAdjustments'] += 1
        if (0<len(alnum)<=4 and re.search(r'[^\w\s]',text) and
                not re.search(r'[^\x00-\x7f]',text) and width_size>height_size*1.30):
            # OCR may omit a dash or leader dots while their source pixels
            # remain in the crop. Their width must not enlarge the digits.
            size = height_size
            stats['shortLabelSizeAdjustments'] += 1
        size = max(1., min(1000., size))
        result.append(dict(text=text, start=piece['start'], end=piece['end'],
                           x=(bx+x)*px, y=(by+y)*py, width=w*px, height=h*py,
                           fontKey=chosen.key, fontFamily=chosen.family,
                           fontWeight='bold' if chosen.bold else 'normal',
                           fontStyle='italic' if chosen.italic else 'normal', fontSize=size,
                           _baseline=(by+y+h)*py-metrics[3]/64*size,
                           _heightSize=height_size,
                           _widthSize=width_size, _letters=len(re.sub(r'\W', '', text)),
                           _score=score))
    return result


def _fit_line(items, stats):
    """Fit one physical line, also when OCR supplied several text fragments."""
    all_runs = [(e, r) for e, runs in items for r in runs]
    reliable = [(e,r) for e,r in all_runs if r['_letters'] >= 3 and
                .68 < r['_heightSize']/r['fontSize'] < 1.5]
    anchors = reliable or [(e,r) for e,r in all_runs if r['_letters'] >= 2] or all_runs
    shared = _weighted_median([(r['fontSize'], min(24, max(1,r['_letters']))) for _,r in anchors])
    baseline = float(np.median([e['y']+r['_baseline'] for e,r in anchors]))
    for e, run in all_runs:
        ratio = run['fontSize']/shared
        strong_size = (run['_letters'] >= 3 and (ratio < .78 or ratio > 1.25) and
                       .80 < run['_heightSize']/run['fontSize'] < 1.25 and
                       abs(e['y']+run['_baseline']-baseline) < shared*.25)
        if strong_size:
            stats['supportedSizeDifferences'] += 1
        else:
            if ratio < .78 or ratio > 1.25:
                stats['sizeOutliersAdjusted'] += 1
            run['fontSize'] = shared
        run['baseline'] = baseline-e['y']
        for key in ('_baseline', '_widthSize', '_heightSize', '_letters', '_score'):
            run.pop(key)


def _block_scale_priors(drafts, stats):
    """Missing OCR spaces can enlarge whole-line width estimates.

    Only intact fallback lines borrow scale from at least two well-aligned
    lines in the same layout block and safe family, with matching ink height.
    Explicit multiword size changes stay under their own geometric evidence.
    """
    groups = defaultdict(list)
    for e,runs,face in drafts:
        key = (str(e.get('regionId',e['id'])),str(e.get('blockId',e['id'])),face.family)
        height = max(r['y']+r['height'] for r in runs)-min(r['y'] for r in runs)
        good = [r for r in runs if r['_letters'] >= 3 and .75 < r['_heightSize']/r['fontSize'] < 1.4]
        if good:
            size = _weighted_median([(r['fontSize'],min(20,r['_letters'])) for r in good])
            groups[key].append((e,runs,height,size,len(good)>=3))
    for group in groups.values():
        anchors = [(height,size) for _,runs,height,size,reliable in group if reliable and len(runs)>1]
        if len(anchors) < 2:
            continue
        observed = float(np.median([h for h,_ in anchors]))
        reference = float(np.median([size for _,size in anchors]))
        for e,runs,height,_,_ in group:
            if len(runs)!=1 or runs[0]['_letters']<20 or not .86 < height/observed < 1.16:
                continue
            run = runs[0]
            prior = reference*height/observed
            if not .92 < run['fontSize']/prior < 1.08:
                metrics = ink_metrics(run['fontKey'],run['text'].strip())
                run['fontSize'] = prior
                run['_baseline'] = run['y']+run['height']-metrics[3]/64*prior
                stats['blockScaleAdjustments'] += 1


@tracked('fonts.rebuild_match')
def match_rebuild_fonts(document, output_dir, *, cache=None):
    """Populate the existing valid_runs contract using a measured per-page budget.

    ``cache`` may be shared across pages of one conversion. It contains bounded
    template/shortlist caches, never source rasters or edits to OCR characters.
    Missing-font coverage is left for the existing safe-export gate to report.
    """
    lib = get_library()
    cache = {} if cache is None else cache
    stats = dict(templateCount=len(lib.faces), matchedLines=0, sampleCount=0,
                 faceComparisons=0, wordComparisons=0, styleWordsReviewed=0,
                 supportedSizeDifferences=0, sizeOutliersAdjusted=0,
                 blockScaleAdjustments=0, shortLabelSizeAdjustments=0, intactLineSizeAdjustments=0,
                 skippedLines=0, unsampledBlocks=0, unreviewedStyleLines=0, families=Counter(),
                 method='bounded-safe-block-samples', approximate=True,
                 budget=dict(faces=MAX_FACES, samplesPerPage=MAX_PAGE_SAMPLES,
                             samplesPerBlock=MAX_BLOCK_SAMPLES, wordsPerSample=MAX_SAMPLE_WORDS,
                             styleWordsPerPage=MAX_PAGE_STYLE_WORDS, styleFaces=MAX_STYLE_FACES))
    output_dir = Path(output_dir)
    pages = document['pages']
    for page_index, page in enumerate(pages):
        with progress_span(100*page_index/max(1,len(pages)), 100*(page_index+1)/max(1,len(pages))):
            report(0, '采样重建文字样式')
            groups = defaultdict(list)
            for element in page['elements']:
                if element.get('type') != 'text' or element.get('source') != 'ocr' or not element.get('text','').strip():
                    continue
                if valid_runs(element) and all(_safe(lib.faces[r['fontKey']]) for r in element['fontRuns']):
                    stats['matchedLines'] += 1
                    stats['families'][element.get('fontFamily','')] += 1
                    continue
                info = element.get('fontMatch')
                if (not element.get('appearanceSrc') or
                        isinstance(info,dict) and info.get('text',element['text']) != element['text']):
                    stats['skippedLines'] += 1
                    continue
                image = _load_appearance(element, document['id'], output_dir)
                ink = _ink(image) if image else None
                if not ink:
                    stats['skippedLines'] += 1
                    continue
                mask, (bx,by,_,_) = ink
                mask, (x,y,_,_) = _clean_evidence(mask)
                evidence = dict(element=element, image=image, mask=mask, offset=(bx+x,by+y))
                evidence['pieces'] = _pieces(evidence,lib)
                evidence['stylePieces'] = _style_pieces(evidence,lib)
                groups[_group_key(element)].append(evidence)
            sample_budget = MAX_PAGE_SAMPLES
            style_budget = MAX_PAGE_STYLE_WORDS
            remaining_lines = sum(map(len, groups.values()))
            drafts = []
            for group_index, (_, lines) in enumerate(groups.items()):
                report(10+85*group_index/max(1,len(groups)), '拟合重建字体与行尺度',
                       current=group_index+1,total=len(groups))
                # High-information lines get the small sampling budget first.
                ranked = sorted(lines,key=lambda l:(-len(re.sub(r'\W','',l['element']['text'])),l['element']['y']))
                samples = ranked[:min(MAX_BLOCK_SAMPLES,sample_budget)]
                if not samples:
                    stats['unsampledBlocks'] += 1
                sample_budget -= len(samples)
                stats['sampleCount'] += len(samples)
                seed = ranked[0]['element']
                faces = _shortlist(lib, seed['text'], seed.get('fontFamily',''), cache)
                if not faces:
                    stats['skippedLines'] += len(lines)
                    remaining_lines -= len(lines)
                    continue
                fallback = lib.resolve(seed['text'],seed.get('fontFamily',''),
                                       str(seed.get('fontWeight')) in {'bold','600','700','800','900'},
                                       seed.get('fontStyle') in {'italic','oblique'},safe_only=True)
                fallback = fallback or min(faces,key=lambda f:(f.bold,f.italic))
                face = _choose_face(lib,samples,fallback,faces,cache,stats)
                for line in lines:
                    remaining_lines -= 1
                    e = line['element']
                    chosen = face
                    if not _needed(e['text']) <= chosen.coverage:
                        alternatives = _shortlist(lib,e['text'],face.family,cache)
                        if not alternatives:
                            stats['skippedLines'] += 1
                            continue
                        chosen = min(alternatives,key=lambda f:(f.family != face.family,f.bold!=face.bold,f.italic!=face.italic))
                    # Spread the fixed local-review budget across the page,
                    # instead of spending it all on the first few lines.
                    count = min(style_budget, MAX_LINE_STYLE_WORDS,
                                max(1, style_budget//max(1, remaining_lines+1)))
                    before = stats['styleWordsReviewed']
                    runs = _draft(lib,line,chosen,cache,stats,count)
                    if stats['styleWordsReviewed'] == before:
                        stats['unreviewedStyleLines'] += 1
                    style_budget -= stats['styleWordsReviewed']-before
                    if runs and ''.join(r['text'] for r in runs) == e['text']:
                        drafts.append((e,runs,chosen))
            line_groups = defaultdict(list)
            _block_scale_priors(drafts,stats)
            for e,runs,face in drafts:
                key = (str(e.get('regionId',e.get('tableId',e['id']))),
                       str(e.get('blockId',e['id'])), str(e.get('lineId',e['id'])))
                line_groups[key].append((e,runs))
            for items in line_groups.values():
                _fit_line(items,stats)
            for e,runs,face in drafts:
                e['fontRuns'] = runs
                e['fontFamily'] = face.family
                e['fontWeight'] = 'bold' if face.bold else 'normal'
                e['fontStyle'] = 'italic' if face.italic else 'normal'
                e['fontSize'] = _weighted_median([(r['fontSize'],max(1,len(r['text'].strip()))) for r in runs])
                e['fontMatch'] = dict(key=face.key, family=face.family, style=face.style, source=face.source,
                                      method='bounded-safe-block-samples',approximate=True,text=e['text'],
                                      geometry=[e['width'],e['height']],inputStyle=[e.get(k) for k in _STYLE])
                stats['matchedLines'] += 1
                stats['families'][face.family] += 1
            report(100, '重建文字样式拟合完成')
    stats['families'] = dict(stats['families'])
    return stats
