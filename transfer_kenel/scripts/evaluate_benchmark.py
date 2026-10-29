"""Run private, reviewed PDF annotations against the current local extractor."""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdf_extractor.benchmark import score_ocr_region, score_tables
from pdf_extractor.engine import extract
from pdf_extractor.model import Options
from pdf_extractor.paths import INPUT_PDF, OUTPUT


def source_path(root, name):
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'PDF 路径必须位于输入目录内：{name}')
    return resolved


def validate_manifest(data, root):
    if data.get('schema_version') != 1 or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('基准需要 schema_version=1 和非空 cases。')
    ids, page_keys = set(), set()
    for case in data['cases']:
        if not isinstance(case.get('id'), str) or not case['id'] or case['id'] in ids:
            raise ValueError('案例 id 为空或重复。')
        ids.add(case['id'])
        source_path(root, case['file'])
        if not re_full_sha(case.get('sha256', '')):
            raise ValueError(f"{case['id']} 缺少 PDF SHA-256。")
        if type(case.get('page')) is not int or case['page'] < 1:
            raise ValueError('page 必须是从 1 开始的物理页码。')
        page_key = (case['file'], case['page'])
        if page_key in page_keys:
            raise ValueError('同一 PDF 页面只能有一个案例，避免重复计数。')
        page_keys.add(page_key)
        if case.get('status') not in ('draft', 'reviewed'):
            raise ValueError('status 必须为 draft 或 reviewed。')
        if case['status'] == 'reviewed' and not all(case.get(k) for k in ('reviewed_by', 'reviewed_at')):
            raise ValueError('reviewed 标注需要 reviewed_by/reviewed_at。')
        size = case.get('page_size', [])
        if len(size) != 2 or any(not finite_number(v) or v <= 0 for v in size):
            raise ValueError('page_size 必须为正数 [width,height]（PDF point）。')
        for key in ('tables', 'ocr_regions'):
            if key in case and not isinstance(case[key], list):
                raise ValueError(f'{key} 必须为列表。')
        task_ids = set()
        for task in case.get('tables', []) + case.get('ocr_regions', []):
            if not isinstance(task.get('id'), str) or not task['id'] or task['id'] in task_ids:
                raise ValueError('页面内任务 id 为空或重复。')
            task_ids.add(task['id'])
            box = task.get('bbox', [])
            if len(box) != 4 or any(not finite_number(v) for v in box) or not (
                    0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
                raise ValueError('bbox 必须位于未旋转页面 point 坐标内。')
        if 'tables' in case and case.get('table_coverage') != 'page':
            raise ValueError('表格任务必须标完整页（包括空列表负样例），否则无法统计误报。')
        if 'tables' not in case and not case.get('ocr_regions'):
            raise ValueError('案例必须有表格或 OCR 任务。')
        for table in case.get('tables', []):
            shape = table.get('shape', [])
            if len(shape) != 2 or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError('表格 shape 必须为正整数 [rows,cols]。')
            if 'rows' in table and (len(table['rows']) != shape[0] or any(
                    len(row) != shape[1] or any(v is not None and not isinstance(v, str) for v in row)
                    for row in table['rows'])):
                raise ValueError('表格 rows 与 shape 不一致；单元格为字符串或 null。')
            occupied, covered = set(), set()
            for span in table.get('spans', []):
                if len(span) != 4 or any(type(v) is not int for v in span):
                    raise ValueError('span 必须为 [row,col,rowspan,colspan]。')
                r, c, rs, cs = span
                if r < 0 or c < 0 or rs < 1 or cs < 1 or rs * cs <= 1 or r + rs > shape[0] or c + cs > shape[1]:
                    raise ValueError('span 超出表格或不是合并单元格。')
                slots = {(i, j) for i in range(r, r + rs) for j in range(c, c + cs)}
                if occupied & slots:
                    raise ValueError('合并区域重叠。')
                occupied |= slots
                covered |= slots - {(r, c)}
                if 'rows' in table and any(table['rows'][i][j] is not None for i, j in slots - {(r, c)}):
                    raise ValueError('合并覆盖位置必须为 null，不是空字符串。')
            if 'rows' in table and any((v is None) != ((r, c) in covered)
                    for r, row in enumerate(table['rows']) for c, v in enumerate(row)):
                raise ValueError('null 只用于已声明合并区域的覆盖位置。')
        for region in case.get('ocr_regions', []):
            if 'strict' in region and type(region['strict']) is not bool:
                raise ValueError('strict 必须为布尔值。')
            if not isinstance(region.get('text'), str) or not region['text'].strip():
                raise ValueError('OCR 真值不能为空。')
    return data


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def re_full_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def aggregate(cases):
    table_reports = [c['tables'] for c in cases if 'tables' in c and c['state'] == 'complete']
    tp, fp, fn = (sum(r[k] for r in table_reports) for k in ('true_positive', 'false_positive', 'false_negative'))
    details = [d for r in table_reports for d in r['details']]
    ocr = [r for c in cases if c['state'] == 'complete' for r in c.get('ocr_regions', [])]
    def ratio(a, b):
        return a / b if b else None
    summary = {'cases_complete': sum(c['state'] == 'complete' for c in cases),
               'cases_error': sum(c['state'] == 'error' for c in cases),
               'tables': {'true_positive': tp, 'false_positive': fp, 'false_negative': fn,
                          'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
                          'shape_exact': sum(d['shape_exact'] for d in details), 'reference_tables': len(details),
                          'cells_correct': sum(d['cells_correct'] for d in details),
                          'cells_total': sum(d['cells_total'] for d in details),
                          'blank_cells_correct': sum(d['blank_cells_correct'] for d in details),
                          'blank_cells_total': sum(d['blank_cells_total'] for d in details),
                          'spans_exact': sum(d.get('spans_exact', False) for d in details),
                          'spans_annotated': sum('spans_exact' in d for d in details)}, 'ocr': {}}
    for mode in ('strict', 'blank_normalized'):
        scored = [r[mode] for r in ocr if r[mode] is not None]
        chars = sum(r['reference_chars'] for r in scored)
        errors = sum(r['errors'] for r in scored)
        summary['ocr'][mode] = {'regions': len(scored), 'reference_chars': chars,
                                'errors': errors, 'cer': ratio(errors, chars)}
    return summary


def markdown(report):
    def fmt(value):
        return 'N/A' if value is None else f'{value:.2%}'
    summary = report['summary']
    t = summary['tables']
    lines = ['# 本地提取基准', '', f"时间：{report['generated_at']}", '',
             f"标注：{report['reference_status']}；草稿 {report['draft_cases']} 页；跳过 {len(report['skipped'])} 页。",
             '**草稿分数仅用于定位问题，不是已人工审核的准确率。存在运行错误时，汇总只覆盖成功页，不得作为完整基线。**', '',
             '## 表格（完整标注页；IoU ≥ 0.5）', '',
             f"TP / FP / FN：{t['true_positive']} / {t['false_positive']} / {t['false_negative']}；precision {fmt(t['precision'])}，recall {fmt(t['recall'])}。",
             f"行列全对：{t['shape_exact']}/{t['reference_tables']}；单元格文字全对：{t['cells_correct']}/{t['cells_total']}；空单元格：{t['blank_cells_correct']}/{t['blank_cells_total']}；合并关系：{t['spans_exact']}/{t['spans_annotated']}。", '',
             '## 扫描 OCR（选定区域，保留转换器输出顺序）', '',
             '| 案例 / 区域 | CER：严格 | CER：答题点线归一 |', '| --- | ---: | ---: |']
    for case in report['cases']:
        for r in case.get('ocr_regions', []):
            lines.append(f"| {case['id']} / {r['id']} | {fmt(r['strict']['cer'] if r['strict'] is not None else None)} | {fmt(r['blank_normalized']['cer'])} |")
    lines += ['', f"微平均严格 CER：{fmt(summary['ocr']['strict']['cer'])}；答题点线归一 CER：{fmt(summary['ocr']['blank_normalized']['cer'])}。", '',
              '## 逐页诊断', '']
    for case in report['cases']:
        lines += [f"### {case['id']}（物理第 {case['page']} 页，{case['status']}）", '', f"状态：{case['state']}。"]
        if case.get('error'):
            lines.append(f"错误：{case['error']}")
        for d in case.get('tables', {}).get('details', []):
            lines.append(f"- 表格 {d['id']}：匹配 {d['matched']}；IoU {d.get('iou', 0):.3f}；行列正确 {d['shape_exact']}；单元格 {d['cells_correct']}/{d['cells_total']}。")
        if 'tables' in case:
            lines.append(f"- 未匹配表格候选：{case['tables']['false_positive_ids']}")
        lines.extend(f'- 提示：{w}' for w in case.get('warnings', []))
        lines.append('')
    lines += ['## 计分口径', '',
              '- 文本做 Unicode NFKC 和空白移除；保留大小写、数字、标点。严格 CER 包括点线长度错误；点线字符数未确定的区域 strict=false，不参与严格 CER。',
              '- 第二种 CER 额外把连续至少三个 . / · 归一为单字符点线占位（U+FFFC）；不抹掉小数点。不是语义准确率。',
              '- OCR 以 block 中心是否落入标注区域选择，输出顺序错误会反映在 CER；没有逐行定位/单独阅读顺序准确率。',
              '- 表格按最大匹配数一对一匹配；IoU 优先遍历，不保证总 IoU 最大。漏检计入行列分母，误检单独计 FP。',
              '- 未标注 rows 的表格不计单元格文字；未标注 spans 的不计合并。缺少 grid_cells 元数据不算合并正确。',
              '- PNG、完整转录、PDF 校验和与诊断 JSON 均为本地私有产物，不应提交或公开。', '']
    return '\n'.join(lines)


def run_benchmark(data, root, output, include_drafts=False, dpi=180):
    import pymupdf as fitz
    selected = [c for c in data['cases'] if include_drafts or c['status'] == 'reviewed']
    if not selected:
        raise ValueError('没有已复核案例；目视草稿需显式使用 --include-drafts，分数不是审核准确率。')
    if output.exists():
        raise ValueError(f'输出目录已存在，不覆盖旧基准：{output}')
    output.mkdir(parents=True)
    report = {'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'reference_status': 'provisional' if any(c['status'] == 'draft' for c in selected) else 'reviewed',
              'draft_cases': sum(c['status'] == 'draft' for c in selected),
              'skipped': [c['id'] for c in data['cases'] if c not in selected],
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'dependencies': {name: importlib.metadata.version(name) for name in
                                               ('PyMuPDF', 'rapidocr-onnxruntime', 'onnxruntime')}},
              'cases': []}
    # Group pages: LocalOCR/model initialization happens once per document.
    for name in dict.fromkeys(c['file'] for c in selected):
        group = [c for c in selected if c['file'] == name]
        records = [{'id': c['id'], 'file': name, 'page': c['page'], 'status': c['status'],
                    'sha256': c['sha256'], 'state': 'error'} for c in group]
        options = Options(pages=','.join(str(c['page']) for c in group), ocr='auto', dpi=dpi)
        start = time.perf_counter()
        try:
            source = source_path(root, name)
            digest = sha256(source)
            if any(c['sha256'] != digest for c in group):
                raise ValueError('PDF 校验和不同，必须重新复核标注，不能静默计分。')
            with fitz.open(source) as doc:
                for c in group:
                    if c['page'] > len(doc):
                        raise ValueError('标注物理页码超出 PDF。')
                    page = doc[c['page'] - 1]
                    page.set_rotation(0)
                    if any(abs(a - b) > .1 for a, b in zip(c['page_size'], [page.rect.width, page.rect.height])):
                        raise ValueError('页面尺寸与标注不同。')
            work = output / f'document_{len(report["cases"]):03d}'
            work.mkdir()
            result = extract(source, work, options)
            (work / 'result.json').write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
            for c, record in zip(group, records):
                blocks = [b for b in result.blocks if b.page == c['page']]
                record.update(state='complete', warnings=result.warnings)
                if 'tables' in c:
                    record['tables'] = score_tables(c['tables'], [b for b in blocks if b.kind == 'table'])
                record['ocr_regions'] = [score_ocr_region(r, blocks) for r in c.get('ocr_regions', [])]
                # Engine catches OCR exceptions; an empty hypothesis is not a valid accuracy measurement.
                images = [b for b in blocks if b.kind == 'image']
                if c.get('ocr_regions') and (any(b.meta.get('ocr_status') == 'failed' for b in images)
                                             or any('OCR 不可用' in w for w in result.warnings)):
                    record.update(state='error', error='OCR 环境/执行失败；不计入准确率汇总。')
        except Exception as exc:
            for record in records:
                record.update(state='error', error=f'{type(exc).__name__}: {exc}')
        for record in records:
            record.update(options=asdict(options), document_seconds=round(time.perf_counter() - start, 3))
            print(f"{record['id']}: {record['state']}")
        report['cases'].extend(records)
    report['summary'] = aggregate(report['cases'])
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'report.md').write_text(markdown(report), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=INPUT_PDF)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--include-drafts', action='store_true')
    parser.add_argument('--dpi', type=int, choices=range(72, 401), default=180, metavar='72-400')
    args = parser.parse_args()
    output = args.output or OUTPUT / ('benchmark_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S'))
    try:
        data = validate_manifest(json.loads(args.manifest.read_text(encoding='utf-8-sig')), args.input.resolve())
        report = run_benchmark(data, args.input.resolve(), output.resolve(), args.include_drafts, args.dpi)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"报告：{output.resolve() / 'report.md'}")
    return int(report['summary']['cases_error'] > 0)


if __name__ == '__main__':
    sys.exit(main())
