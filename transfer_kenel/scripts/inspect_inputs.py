"""Describe local input coverage; diagnostics are not accuracy benchmarks."""
import argparse
from datetime import date
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf as fitz
from pdf_extractor.engine import _native_text, _scan_page
from pdf_extractor.paths import INPUT_PDF, OUTPUT
from pdf_extractor.tables import native_tables


def inspect(root, detect_tables=False):
    records = []
    for source in sorted(root.rglob('*.pdf')):
        role = 'OCR 测试' if 'ocr' in source.relative_to(root).parts else '论文类型分析'
        with fitz.open(source) as doc:
            text_pages, scan_pages, image_pages, tables, formula_pages = [], [], [], [], []
            first_text = ''
            warnings = []
            for page in doc:
                text = page.get_text()
                if not first_text:
                    first_text = ' '.join(text.split())[:120]
                native = _native_text(page, [])
                if text.strip(): text_pages.append(page.number+1)
                if _scan_page(page, native, []): scan_pages.append(page.number+1)
                if page.get_image_info(): image_pages.append(page.number+1)
                if role == '论文类型分析' and detect_tables:
                    try:
                        for table in native_tables(page, 'auto', warnings):
                            tables.append(f'{page.number+1}页:{len(table.rows)}×{len(table.rows[0])}')
                    except Exception as exc:
                        warnings.append(f'{page.number+1}页检测失败:{type(exc).__name__}')
                if re.search(r'[∑∫√≤≥±]|(?:\\alpha|\\frac)', text): formula_pages.append(page.number+1)
            records.append({'file': source.relative_to(root).as_posix(), 'role': role, 'pages':len(doc),
                            'text':len(text_pages), 'scan':len(scan_pages), 'images':len(image_pages),
                            'tables':tables, 'formula':formula_pages, 'first':first_text, 'warnings':warnings})
    return records


def report(records, root, include_tables):
    lines = ['# 输入样例覆盖清单', '', f'生成日期：{date.today().isoformat()}。输入：`{root}`。', '',
             '此报告是结构诊断，不是提取准确率报告：有原生文字不代表整页不是扫描件，检测到表格不代表行列正确；公式标记只是候选。', '',
             '| 文件 | 用途 | 页数 | 有原生文字页 | 扫描页候选 | 有位图页 | 表格候选 |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for r in records:
        tables = ', '.join(r['tables']) if include_tables else '未检查'
        lines.append(f"| `{r['file']}` | {r['role']} | {r['pages']} | {r['text']} | {r['scan']} | {r['images']} | {tables or '无'} |")
    lines += ['', '## 诊断详情', '']
    for r in records:
        lines += [f"### {r['file']}", '', f"- 公式/数学符号候选页：{', '.join(map(str,r['formula'])) or '未找到'}。"]
        if r['role']=='论文类型分析':lines.append(f"- 首页原生文字片段：{r['first']}")
        if r['warnings']:lines.append('- 需要校对：'+'；'.join(dict.fromkeys(r['warnings'])))
        lines.append('')
    lines += ['## 样例范围', '', '普通论文用于整理内容类型；OCR 子目录用于扫描识别改进，不混为同一个准确率统计。',
              '当前扫描件以英文考试书为主，不能代表中文、手写、倾斜、复杂表格或各种图标。后续需要逐页标注文字、位置和结构真值，再报告准确率。', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_PDF)
    parser.add_argument('--output', type=Path, default=OUTPUT/'input_inventory.md')
    parser.add_argument('--tables', action='store_true', help='检查普通论文的原生表格候选；不运行 OCR')
    args=parser.parse_args()
    root=args.input.expanduser().resolve()
    if not root.is_dir(): parser.error(f'输入目录不存在：{root}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    records=inspect(root,args.tables)
    args.output.write_text(report(records,root,args.tables),encoding='utf8')
    print(f'已分析 {len(records)} 份 PDF；报告：{args.output.resolve()}')


if __name__=='__main__':main()
