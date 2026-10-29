"""Page-at-a-time conversion for scan books larger than the editing canvas."""

from .usage import tracked
import json
from pathlib import Path
import shutil

import pymupdf as fitz

from .progress import progress_span, report as update_progress

from .exporter import append_text_pages
from .options import selected_pages
from .parser import parse_document
from .font_library import match_document_fonts
from .pdf_safety import character_inventory, prepare_fonts, save_verified

MAX_BOOK_PAGES=1000


@tracked('text_pdf.convert', feature='text_pdf')
def convert_text_pdf(paths, output_dir: Path, document_id: str, *, pages=''):
    selections=[]
    for path in paths:
        with fitz.open(path) as pdf:
            if not pdf.is_pdf or pdf.needs_pass:
                raise ValueError('整本转换需要未加密的 PDF 文件')
            selections.append(selected_pages(pages,len(pdf)))
    total=sum(len(selection) for selection in selections)
    if not 1<=total<=MAX_BOOK_PAGES:
        raise ValueError(f'每次整本转换支持 1–{MAX_BOOK_PAGES} 页，请通过页码分批转换')
    destination=output_dir/'converted-text.pdf'
    report={'pageCount':total,'independentTextLines':0,'retainedMixedLines':0,'pages':[]}
    scratch=output_dir/'page-work'
    expected=[]
    ordinal=0
    with fitz.open() as combined:
        for path,selection in zip(paths,selections):
            for physical in selection:
                ordinal+=1
                scratch.mkdir(parents=True,exist_ok=False)
                try:
                    with progress_span(85*(ordinal-1)/total,85*ordinal/total,page=(ordinal,total)):
                        with progress_span(0,45):
                            scene=parse_document([path],scratch,document_id,path.stem,ocr=True,pages=str(physical+1),mode='text_pdf')
                        with progress_span(45,75):
                            fonts=match_document_fonts(scene,scratch)
                        with progress_span(75,82):
                            prepare_fonts(scene)
                        with progress_span(82,100):
                            count=append_text_pages(combined,scene,scratch)
                        expected.extend(character_inventory(page) for page in scene['pages'])
                        retained=sum(len(e.get('retainedText',[])) for p in scene['pages'] for e in p['elements'])
                        report['independentTextLines']+=count
                        report['retainedMixedLines']+=retained
                        report['pages'].append({'file':path.name,'sourcePage':physical+1,
                                                'independentTextLines':count,'retainedMixedLines':retained,'fonts':fonts})
                finally:
                    # This directory is created exclusively for this conversion
                    # and contains only the current page's generated assets.
                    shutil.rmtree(scratch)
        if not report['independentTextLines']:
            raise ValueError('未找到可独立转换的文字，原始图文已保留；请检查 OCR 或选择其他页面')
        combined.set_metadata({'title':'扫描 PDF 转文本与图片','creator':'PDF OCR Parser',
                               'subject':'Visible text and preserved images; page-at-a-time conversion'})
        combined.embfile_add('conversion-report.json',json.dumps(report,ensure_ascii=False).encode(),
                             filename='conversion-report.json',desc='Conversion diagnostics, not accuracy measurements')
        with progress_span(85,100):
            report['pdfSafety']=save_verified(combined,destination,expected)
    update_progress(100,'转换完成')
    return destination,report
