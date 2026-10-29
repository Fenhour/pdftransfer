"""Small local accuracy benchmark; independent annotations, not extraction counts."""
import re
import unicodedata


def normalize_text(text, blank_runs=False):
    """NFKC + remove layout whitespace; retain case, punctuation and numbers."""
    text = unicodedata.normalize('NFKC', text or '')
    if blank_runs:
        # One placeholder is one character, not a seven-character markup token.
        text = re.sub(r'[.·]{3,}', '\ufffc', text)
    return re.sub(r'\s+', '', text)


def edit_distance(reference, hypothesis):
    """Levenshtein distance, O(min(n, m)) memory. CER may exceed one."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def text_score(reference, hypothesis, blank_runs=False):
    ref = normalize_text(reference, blank_runs)
    hyp = normalize_text(hypothesis, blank_runs)
    errors = edit_distance(ref, hyp)
    return {'reference_chars': len(ref), 'predicted_chars': len(hyp),
            'errors': errors, 'cer': errors / len(ref) if ref else None,
            'exact': ref == hyp}


def bbox_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_boxes(references, predictions, threshold=0.5):
    """Maximum-cardinality one-to-one matching, IoU-preferred traversal.

    This is not globally maximum-weight matching; detection counts do not depend
    on greedy processing order. Return (reference index, prediction index, IoU).
    """
    edges = [sorted([(j, bbox_iou(a, b)) for j, b in enumerate(predictions)
                     if bbox_iou(a, b) >= threshold], key=lambda x: (-x[1], x[0]))
             for a in references]
    owners = {}

    def assign(i, seen):
        for j, _ in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(references)):
        assign(i, set())
    return sorted((i, j, bbox_iou(references[i], predictions[j]))
                  for j, i in owners.items())


def score_tables(references, predictions, threshold=0.5):
    matches = match_boxes([r['bbox'] for r in references],
                          [b.bbox for b in predictions], threshold)
    details = []
    by_reference = {i: (j, iou) for i, j, iou in matches}
    for i, ref in enumerate(references):
        detail = {'id': ref['id'], 'matched': i in by_reference,
                  'shape_exact': False, 'cells_correct': 0,
                  'cells_total': sum(map(len, ref.get('rows', []))),
                  'blank_cells_correct': 0,
                  'blank_cells_total': sum(v == '' for row in ref.get('rows', []) for v in row)}
        if i in by_reference:
            j, iou = by_reference[i]
            pred = predictions[j]
            shape = [len(pred.rows), max(map(len, pred.rows), default=0)]
            detail.update({'prediction_id': pred.id, 'iou': iou, 'shape': shape,
                           'shape_exact': shape == ref['shape'], 'source': pred.source})
            for r, row in enumerate(ref.get('rows', [])):
                for c, text in enumerate(row):
                    present = r < len(pred.rows) and c < len(pred.rows[r])
                    # None is a covered merged slot, not an empty anchor cell.
                    value = pred.rows[r][c] if present else None
                    correct = present and ((text is None and value is None) or
                        (text is not None and value is not None and
                         normalize_text(text) == normalize_text(value)))
                    detail['cells_correct'] += int(correct)
                    detail['blank_cells_correct'] += int(correct and text == '')
            if 'spans' in ref:
                grid = pred.meta.get('grid_cells')
                spans = None if grid is None else sorted(
                    [cell['row'], cell['col'], cell.get('rowspan', 1), cell.get('colspan', 1)]
                    for cell in grid if cell.get('rowspan', 1) > 1 or cell.get('colspan', 1) > 1)
                detail.update({'spans_exact': detail['shape_exact'] and spans == sorted(ref['spans']),
                               'predicted_spans': spans})
        elif 'spans' in ref:
            detail['spans_exact'] = False
        details.append(detail)
    unmatched = [b.id for j, b in enumerate(predictions) if j not in {m[1] for m in matches}]
    return {'true_positive': len(matches), 'false_positive': len(unmatched),
            'false_negative': len(references) - len(matches),
            'false_positive_ids': unmatched, 'details': details}


def score_ocr_region(region, blocks):
    """Use the extractor's existing order, never sort by the reference text."""
    x0, y0, x1, y1 = region['bbox']
    selected = [b for b in blocks if b.source == 'ocr' and b.kind != 'image'
                and x0 <= (b.bbox[0] + b.bbox[2]) / 2 <= x1
                and y0 <= (b.bbox[1] + b.bbox[3]) / 2 <= y1]
    # Include OCR table cells if the region is classified as a table.
    hypothesis = '\n'.join('\n'.join(' '.join(v or '' for v in row) for row in b.rows)
                           if b.kind == 'table' else b.text for b in selected)
    return {'id': region['id'], 'bbox': region['bbox'],
            'block_ids': [b.id for b in selected], 'hypothesis': hypothesis,
            'strict': text_score(region['text'], hypothesis) if region.get('strict', True) else None,
            'blank_normalized': text_score(region['text'], hypothesis, True)}
