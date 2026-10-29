"""Page-bounded typed reconstruction, separate from legacy scan inpainting."""
from __future__ import annotations

from .usage import tracked

from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time
import zipfile

import pymupdf as fitz
from PIL import Image

from .progress import progress_span, report as update_progress

from .exporter import append_text_pages, read_asset
from .rebuild_fonts import match_rebuild_fonts
from .options import selected_pages
from .parser import parse_document
from .pdf_safety import character_inventory, prepare_fonts, save_verified
from .rebuild_export import export_page, finalize_html
from .style_templates import resolve_styles
from .text_pdf import MAX_BOOK_PAGES


@tracked('rebuild.convert', feature=lambda v: 'rebuild.' + v.get('format', 'pdf'))
def convert_reconstruction(paths,output_dir:Path,document_id:str,*,format='pdf',pages='',recognition_backend=None):
    if recognition_backend == 'v3':
        from .adaptive_reconstruction import convert_adaptive
        return convert_adaptive(paths, output_dir, document_id, format=format, pages=pages)
    if format not in {'pdf','html','md'}:
        raise ValueError('结构化重建仅支持 PDF、HTML 或 Markdown')
    paths=[Path(path) for path in paths]
    selections=[]
    for value in paths:
        path=Path(value)
        if path.suffix.lower()=='.pdf':
            with fitz.open(path) as pdf:
                if not pdf.is_pdf or pdf.needs_pass:
                    raise ValueError('重建需要未加密的 PDF 文件')
                selections.append(selected_pages(pages,len(pdf)))
        elif path.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.webp','.bmp'}:
            with Image.open(path) as image:
                selections.append(selected_pages(pages,getattr(image,'n_frames',1)))
        else:
            raise ValueError('重建仅支持 PDF、PNG、JPEG、TIFF、WebP 或 BMP')
    total=sum(map(len,selections))
    if not 1 <= total <= MAX_BOOK_PAGES:
        raise ValueError(f'每次重建支持 1–{MAX_BOOK_PAGES} 页，请通过页码分批处理')
    output_dir.mkdir(parents=True,exist_ok=True)
    scratch=output_dir/'rebuild-page-work'
    tree=output_dir/f'rebuilt-{format}'
    destination=output_dir/('rebuilt-text.pdf' if format=='pdf' else f'rebuilt-{format}.zip')
    report=dict(schema='reconstruction/1',format=format,pageCount=total,independentTextLines=0,
                retainedMixedLines=0,tableCount=0,imageCount=0,pages=[],strategy='blank-page-typed-reconstruction',
                status='needs-review',boundaryPolicy='foreground-barriers-v2; source RGB; optional alpha text and frame-rule exclusions; no inpainting',
                limitations=['Geometry boundaries and table classification require review.',
                             'Only simple complete ruled grids are reconstructed; merged cells are not inferred.',
                             'Complex artwork remains raster; local font templates are approximate.'])
    expected=[]
    font_cache={}
    started=time.perf_counter()
    with fitz.open() as combined, ExitStack() as native_resources:
        ordinal=0
        for path,selection in zip(paths,selections):
            native_resources.close()
            if recognition_backend=='v3' and path.suffix.lower()=='.pdf':
                from .structure_parser import native_table_session
                native_resources.enter_context(native_table_session(path,selection))
            for physical in selection:
                ordinal+=1
                scratch.mkdir(parents=True,exist_ok=False)
                try:
                    with progress_span(85*(ordinal-1)/total,85*ordinal/total,page=(ordinal,total)):
                        tick=time.perf_counter()
                        with progress_span(0,45):
                            recognition_options=({'recognition_backend':recognition_backend}
                                                 if recognition_backend is not None else {})
                            scene=parse_document([path],scratch,document_id,path.stem,ocr=True,
                                                 pages=str(physical+1),mode='rebuild',**recognition_options)
                        recognition=scene['pages'][0].get('recognitionBackend','legacy')
                        report['recognitionBackend']=recognition
                        if recognition in {'v3','pp-structurev3','native+v3'}:
                            report['boundaryPolicy']='Native PDF text/tables reused; V3 semantic layout on image regions; single text ownership; original figure/table references; no inpainting'
                            report['limitations']=[
                                'Model layout boundaries, OCR text and merged-cell topology require review.',
                                'Unsupported or inconsistent table structures remain original image regions.',
                                'Font templates approximate scan appearance; cell fallback text wraps using safe font metrics.']
                        parsed=time.perf_counter()
                        with progress_span(45,75):
                            fonts=match_rebuild_fonts(scene,scratch,cache=font_cache)
                        if format=='pdf':
                            with progress_span(75,82):
                                prepare_fonts(scene)
                        templates=resolve_styles(scene)
                        matched=time.perf_counter()
                        page=scene['pages'][0]
                        count=sum(e['type']=='text' for e in page['elements'])
                        retained=sum(len(e.get('retainedText',[])) for e in page['elements'])
                        if format=='pdf':
                            with progress_span(82,100):
                                append_text_pages(combined,scene,scratch)
                            expected.append(character_inventory(page))
                            # PDF content streams do not preserve generic text
                            # boxes. Retain our own coordinates and membership
                            # as small, inspectable per-page evidence instead.
                            layout=dict(schema='reconstruction-page/1',sourcePage=physical+1,
                                        width=page['width'],height=page['height'],
                                        textLayout=page.get('textLayout'),recognition=page.get('recognition',[]),
                                        elements=[{k:v for k,v in e.items() if k not in
                                                   {'src','appearanceSrc','originalAppearanceSrc'}} for e in page['elements']])
                            layout_name=f'page-{ordinal:04d}-layout.json'
                            combined.embfile_add(layout_name,json.dumps(layout,ensure_ascii=False).encode(),
                                                 filename=layout_name,desc='OCR ownership, positioned lines and fitted styles; not editor text boxes')
                            references=[]
                            for table in page.get('tables',[]):
                                name=f'page-{ordinal:04d}-{table["id"]}-reference.png'
                                combined.embfile_add(name,read_asset(table['referenceSrc'],scratch,scene['id']),
                                                     filename=name,desc='Original table pixels for structure review')
                                references.append(dict(tableId=table['id'],attachment=name,bbox=table['referenceBbox'],
                                                       rows=table['rows'],columns=table['columns'],cells=table['cells']))
                            adapter=dict(tableReferences=references,layoutAttachment=layout_name)
                        else:
                            adapter=export_page(scene,scratch,tree,ordinal,format)
                        report['independentTextLines']+=count
                        report['retainedMixedLines']+=retained
                        report['tableCount']+=len(page.get('tables',[]))
                        images=sum(e['type']=='image' for e in page['elements'])
                        report['imageCount']+=images
                        report['pages'].append(dict(file=path.name,sourcePage=physical+1,independentTextLines=count,
                                                    retainedMixedLines=retained,tableCount=len(page.get('tables',[])),
                                                    imageCount=images,fonts=fonts,templates=templates,adapter=adapter,
                                                    recognitionBackend=recognition,
                                                    modelDispatch=page.get('modelDispatch',{}),
                                                    textLayout=page.get('textLayout',{}).get('stats',{}),
                                                    unassignedCandidates=sum(not c.get('ownerId') for c in page.get('recognition',[])),
                                                    imageBoundaries=[dict(bbox=e.get('bbox'),boundary=e.get('boundary'))
                                                                     for e in page['elements'] if e['type']=='image'],
                                                    warnings=page['warnings'],timings=dict(parse=parsed-tick,
                                                        fontMapping=matched-parsed,export=time.perf_counter()-matched)))
                finally:
                    # Only the exclusively-created per-page scratch tree.
                    shutil.rmtree(scratch)
        if report['retainedMixedLines']:
            report['status']='partial'
        report['elapsedBeforeFinalization']=time.perf_counter()-started
        if format=='pdf':
            if not report['independentTextLines']:
                raise ValueError('未找到可独立重建的文字；图文不能可靠分离，未冒充可编辑文本 PDF')
            combined.set_metadata(dict(title='扫描文档结构化重建',creator='PDF OCR Parser',
                                       subject='Visible text, ruled tables, original image regions; no scan inpainting'))
            combined.embfile_add('conversion-report.json',json.dumps(report,ensure_ascii=False).encode(),
                                 filename='conversion-report.json',desc='Review diagnostics, not accuracy claims')
            with progress_span(85,100):
                report['pdfSafety']=save_verified(combined,destination,expected)
        else:
            update_progress(85,'生成重建文档','正在汇总页面并打包资源')
            if format=='html':
                finalize_html(tree)
            (tree/'conversion-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
                assets=[asset for asset in sorted(tree.rglob('*')) if asset.is_file()]
                for asset_index,asset in enumerate(assets):
                    update_progress(85+15*asset_index/max(1,len(assets)),'打包重建文档',current=asset_index+1,total=len(assets))
                    if asset.is_file():
                        archive.write(asset,asset.relative_to(tree).as_posix())
    (output_dir/'conversion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    update_progress(100,'转换完成')
    return destination,report
