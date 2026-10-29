"""Recognition policy at the application boundary; never an implicit fallback."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def selected_backend():
    config = json.loads((Path(__file__).resolve().parents[1] / 'config.json').read_text(encoding='utf-8-sig'))
    backend = os.environ.get('PDFTRANSFER_RECOGNITION', config.get('recognition_backend', 'v3'))
    if backend not in {'v3', 'legacy'}:
        raise ValueError('recognition_backend 必须为 v3 或 legacy')
    return backend


def shared_structure():
    root = str(Path(__file__).resolve().parents[2] / 'transfer_kenel')
    if root not in sys.path:
        sys.path.insert(0, root)
    from pdf_extractor import structure
    return structure


def status():
    if selected_backend() == 'legacy':
        from .parser import ocr_status
        return dict(ocr_status(), backend='legacy')
    info = shared_structure().service_status()
    return dict(available=info['available'], backend='v3', engine='PP-StructureV3 / CPU',
                languages=['中文', 'English'], local=True, loaded=info.get('loaded', False),
                message=('本地 V3 可用；原生页直接提取，扫描页分析版面、文字和表格'
                         if info['available'] else '本机 V3 环境未就绪；原生页可直接提取，请配置本机模型环境'),
                service=info)
