"""Isolate diagnostic single-model runs from the production V3 worker."""
import json
import os
from pathlib import Path
import subprocess
import threading

from .models import validate_document
from .options import selected_pages
from .progress import report

PROJECT = Path(__file__).resolve().parents[2]
DETECTORS = {'det', 'layout', 'region'}
_RUNNING = threading.Lock()


def run_detection(paths, output_dir, detector, pages=''):
    if detector not in DETECTORS:
        raise ValueError('请选择文字 det、plus-L 类别框或 BlockLayout 大区域框')
    import pymupdf as fitz
    total=0
    for path in paths:
        if path.suffix.lower()!='.pdf':
            raise ValueError('单步检测目前只接受 PDF')
        with fitz.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError('单步检测需要未加密 PDF')
            total+=len(selected_pages(pages,len(pdf)))
    if not 1<=total<=100:
        raise ValueError('单步检测每次支持1–100页')
    python=PROJECT/'services/structure_v3/.venv/Scripts/python.exe'
    if not python.is_file():
        raise ValueError('未找到本机 V3 环境，请先运行 setup-native.ps1')
    if not _RUNNING.acquire(blocking=False):
        raise ValueError('已有单步检测正在运行，请稍后重试')
    try:
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)
        destination=output_dir/'detection'
        report(12,'单模型检测','加载所选本机模型并检测页面',current=0,total=total)
        command=[str(python),'-X','utf8',str(PROJECT/'services/tmp/detection_preview.py'),
                 *[str(p) for p in paths],'--detector',detector,'--pages',pages,
                 '--output',str(destination)]
        with (output_dir/'detection-process.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,
                    timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),
                    env={**os.environ,'PYTHONIOENCODING':'utf-8'})
            except subprocess.TimeoutExpired as exc:
                raise ValueError('单步检测超过30分钟，已停止本次独立检测进程') from exc
        if result.returncode:
            raise ValueError('单步检测失败，请查看本次文档目录的 detection-process.log')
        report(90,'生成检测图层','正在载入原页、边框和模型类别标签',current=total,total=total)
        scene=json.loads((destination/'scene-document.json').read_text(encoding='utf-8'))
        return validate_document(scene)
    finally:
        _RUNNING.release()
