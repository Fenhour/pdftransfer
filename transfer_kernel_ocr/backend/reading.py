"""Bridge the existing local reading extractor into the browser workflow."""

from .usage import tracked
from pathlib import Path
import sys
import zipfile
from .progress import report


@tracked('reading.package', feature=lambda v: 'reading.' + v.get('format', 'html'))
def convert_reading(paths, output_dir: Path, *, format='html', ocr=True, pages='', recognition_backend=None):
    if format not in {'html','md'}:
        raise ValueError('阅读转换仅支持 HTML 或 Markdown')
    if any(path.suffix.lower()!='.pdf' for path in paths):
        raise ValueError('HTML / Markdown 转换仅接受 PDF 文件')
    if recognition_backend is None:
        from .recognition import selected_backend
        recognition_backend=selected_backend()
    module_root=Path(__file__).resolve().parents[2]/'transfer_kenel'
    if not (module_root/'pdf_extractor/engine.py').is_file():
        raise ValueError('找不到阅读型提取器，请保留同级 transfer_kenel 目录')
    if str(module_root) not in sys.path:
        sys.path.insert(0,str(module_root))
    from pdf_extractor.engine import run
    from pdf_extractor.model import Options
    work=output_dir/'reading'
    warnings=[]
    for index,path in enumerate(paths):
        def on_progress(done,total,message):
            report(85*(index+done/max(1,total))/len(paths),'转换阅读文档',message,current=done,total=total)
        _,result=run(path,work,Options(pages=pages,ocr='auto' if ocr else 'off',
                                    output_format=format,recognition_backend=recognition_backend),progress=on_progress)
        warnings.extend(result.warnings)
    root=work/f'output_{format}'
    destination=output_dir/f'converted-{format}.zip'
    assets=[path for path in sorted(root.rglob('*')) if path.is_file()]
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for index,path in enumerate(assets):
            report(85+15*index/max(1,len(assets)),'打包阅读结果',current=index+1,total=len(assets))
            archive.write(path,str(Path(root.name)/path.relative_to(root)))
    report(100,'阅读文档已生成')
    return destination,warnings
