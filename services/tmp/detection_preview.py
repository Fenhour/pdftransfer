"""Run ONE local detector and export labeled boxes + editable PDF/scene.

Use the V3 virtual environment. Directory inputs resolve numbered split PDFs;
PDF file inputs use physical page numbers (1-based). No rec or other models.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'services/structure_v3'))
MODELS = {'det': 'PP-OCRv5_server_det', 'layout': 'PP-DocLayout_plus-L',
          'region': 'PP-DocBlockLayout'}
CHINESE = dict(text='文本块', image='图片', table='表格', region='大区域', Region='大区域',
    paragraph_title='段落标题', doc_title='文档标题', number='页码', abstract='摘要',
    content='目录', figure_title='图题', figure_table_chart_title='图表标题',
    formula='公式', reference='参考文献标题', footnote='脚注', header='页眉',
    algorithm='算法', footer='页脚', seal='印章', chart='图表',
    formula_number='公式编号', aside_text='侧栏文字', reference_content='参考文献',
    text_line='文字行')
COLORS = ['#d32f2f', '#1565c0', '#008060', '#8e24aa', '#b45309', '#007c91']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def select_sources(paths, spec):
    sys.path.insert(0, str(ROOT / 'transfer_kernel_ocr'))
    from backend.options import selected_pages
    import pymupdf as fitz
    chosen = []
    for path in paths:
        if path.is_dir():
            if not spec:
                raise ValueError('Directory input requires explicit numbered pages')
            for index in selected_pages(spec, 10000):
                number = index+1
                matches = list(path.glob(f'*_{number}.pdf'))
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one *_{number}.pdf')
                with fitz.open(matches[0]) as pdf:
                    if len(pdf) != 1 or pdf.needs_pass:
                        raise ValueError('Split input must be a single unencrypted page')
                chosen.append((matches[0], 0, number))
        else:
            with fitz.open(path) as pdf:
                if not pdf.is_pdf or pdf.needs_pass:
                    raise ValueError('Detection preview requires unencrypted PDFs')
                chosen.extend((path, i, i+1) for i in selected_pages(spec, len(pdf)))
    if not 1 <= len(chosen) <= 100:
        raise ValueError('Detection preview supports 1-100 pages')
    return chosen


def model_parameters(kind):
    if kind == 'det':
        return dict(limit_side_len=736, limit_type='min', max_side_limit=4000,
                    thresh=.3, box_thresh=.6, unclip_ratio=1.5)
    # Read the installed V3 pipeline's actual submodel parameters without
    # constructing its OCR/table/other models.
    import paddlex
    import yaml
    config_path = Path(paddlex.__file__).parent / 'configs/pipelines/PP-StructureV3.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    module = config['SubModules']['LayoutDetection' if kind == 'layout' else 'RegionDetection']
    if module['model_name'] != MODELS[kind]:
        raise ValueError('Installed V3 detector configuration changed')
    return {k: module[k] for k in ('threshold', 'layout_nms', 'layout_unclip_ratio',
                                   'layout_merge_bboxes_mode') if k in module}


def detections(result, kind):
    if kind == 'det':
        polygons = result.get('dt_polys')
        polygons = [] if polygons is None else polygons
        scores = result.get('dt_scores')
        return [dict(label='text_line', cls_id=None, score=None if scores is None else float(scores[i]),
                     polygon=plain(poly)) for i, poly in enumerate(polygons)]
    found = []
    for box in result.get('boxes', []):
        x, y, z, t = map(float, box['coordinate'])
        found.append(dict(label=str(box['label']), cls_id=plain(box.get('cls_id')),
            score=plain(box.get('score')), coordinate=[x,y,z,t],
            polygon=[[x,y],[z,y],[z,t],[x,t]]))
    return found


def run(paths, spec, kind, output):
    if kind not in MODELS:
        raise ValueError('Unknown detector')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite output: {output}')
    from native_worker import load_config
    from runtime_threads import configure_cpu_environment
    config = load_config()
    threads = configure_cpu_environment(config)
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
    # Reuse the editor's PDF library when the V3 environment lacks it.
    sys.path.append(str(ROOT / 'transfer_kernel_ocr/.venv/Lib/site-packages'))
    import pymupdf as fitz
    import numpy as np
    chosen = select_sources(paths, spec)
    if any(p.resolve().is_relative_to(output.resolve()) for p, _, _ in chosen):
        raise ValueError('Output must not contain input files')
    from model_profile import ModelProfiler
    model_dir = ModelProfiler(config['model_dir'], model_paths=config.get('model_paths')).resolve_model_dir(MODELS[kind])
    for filename in ('inference.json','inference.pdiparams','inference.yml'):
        if not (model_dir/filename).is_file():
            raise FileNotFoundError(model_dir/filename)
    import socket
    def offline(*a, **kw):
        raise RuntimeError('Network disabled for local detector preview')
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.create_connection = offline
    from paddleocr import TextDetection, LayoutDetection
    params = model_parameters(kind)
    started = time.perf_counter()
    wrapper = TextDetection if kind == 'det' else LayoutDetection
    detector = wrapper(model_name=MODELS[kind], model_dir=str(model_dir),
        device='cpu', cpu_threads=threads, enable_mkldnn=config.get('enable_mkldnn', True),
        enable_hpi=False, **{k:v for k,v in params.items() if k!='max_side_limit'})
    report = dict(model=MODELS[kind], detector=kind, parameters=plain(params),
        cpuThreads=threads, initializationSeconds=time.perf_counter()-started,
        weightsSha256=digest(model_dir/'inference.pdiparams'), pages=[])
    scene = dict(schemaVersion=1, id=f'{kind}-preview', name=f'{MODELS[kind]} 检测预览',
        pages=[], warnings=['仅单模型检测结果；类别为模型预测，尚未人工校对。'], detectionModel=MODELS[kind])
    output.mkdir(parents=True)
    pdf = fitz.open()
    label_font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
    if not label_font.is_file():
        raise FileNotFoundError('Chinese label font missing: Microsoft YaHei')
    cjk = fitz.Font(fontfile=str(label_font))
    for ordinal, (path, index, number) in enumerate(chosen, 1):
        before = digest(path)
        with fitz.open(path) as original:
            page = original[index]
            w,h = page.rect.width,page.rect.height
            scale = min(2., math.sqrt(40_000_000/(w*h))*.999)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale,scale), alpha=False, colorspace=fitz.csRGB)
            rgb = np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,3).copy()
            started = time.perf_counter()
            results = list(detector.predict(rgb[:,:,::-1].copy(), **params))
            seconds = time.perf_counter()-started
            if len(results) != 1:
                raise ValueError('Detector must return one result per page')
            boxes = detections(results[0], kind)
            png = pix.tobytes('png')
            data_url = 'data:image/png;base64,'+base64.b64encode(png).decode('ascii')
            target = pdf.new_page(width=w,height=h)
            target.insert_image(target.rect,stream=png)
            target.insert_font(fontname='label',fontfile=str(label_font))
            elements = [dict(id=f'p{ordinal}-bg',type='background',name=f'原页 {number}',
                x=0,y=0,width=w,height=h,zIndex=0,locked=True,src=data_url)]
            tags = []
            for i, box in enumerate(boxes):
                points = np.asarray(box['polygon'],dtype=float)
                if points.shape != (4,2) or not np.isfinite(points).all():
                    raise ValueError('Invalid detection polygon')
                points = points*[w/pix.width,h/pix.height]
                x,y = points.min(axis=0);z,t = points.max(axis=0)
                bw,bh = max(.01,float(z-x)),max(.01,float(t-y))
                label = box['label']
                color = '#ff0000' if kind=='det' else COLORS[int(hashlib.sha256(label.encode()).hexdigest()[:4],16)%len(COLORS)]
                stroke = tuple(int(color[k:k+2],16)/255 for k in (1,3,5))
                target.draw_polyline([fitz.Point(*p) for p in points],closePath=True,color=stroke,width=.8)
                coords = ' '.join(f'{px-x:.5f},{py-y:.5f}' for px,py in points)
                svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {bw} {bh}">'
                       f'<polygon points="{coords}" fill="none" stroke="{color}" stroke-width="0.8"/></svg>')
                caption = f'{i+1} {CHINESE.get(label,label)} / {label}'
                elements.append(dict(id=f'p{ordinal}-box-{i}',type='vector',name=caption,
                    x=float(x),y=float(y),width=bw,height=bh,zIndex=i+1,svg=svg,fill='none',
                    stroke=color,strokeWidth=.8,detectorOutput=box))
                if kind != 'det':
                    if len(label)>100:
                        raise ValueError('Unexpected detector label')
                    font_size=6.5
                    tw=min(w-4,cjk.text_length(caption,fontsize=font_size)+5)
                    tx=max(1,min(float(x),w-tw-1));ty=max(1,min(float(y)-11,h-12))
                    rect=fitz.Rect(tx,ty,tx+tw,ty+10)
                    for attempt in range(12):
                        if not any(rect.intersects(previous) for previous in tags):break
                        ty=min(h-11,ty+11)
                        rect=fitz.Rect(tx,ty,tx+tw,ty+10)
                    tags.append(rect)
                    target.draw_rect(rect,color=None,fill=(1,1,1))
                    target.insert_text((tx+2,ty+7.5),caption,fontname='label',fontsize=font_size,color=stroke)
                    elements.append(dict(id=f'p{ordinal}-label-{i}',type='text',name=caption,
                        x=tx,y=ty,width=tw,height=10,zIndex=len(boxes)+i+1,text=caption,
                        fontFamily='Microsoft YaHei',fontSize=font_size,lineHeight=1.15,
                        color=color,fill='#ffffff',detectionLabel=label))
            scene['pages'].append(dict(id=f'page-{ordinal}',index=ordinal-1,width=w,height=h,
                sourceSrc=data_url,elements=elements,sourcePhysicalPage=number))
            report['pages'].append(dict(sourceFile=path.name,sourceSha256=before,physicalPage=number,
                pixelSize=[pix.width,pix.height],pageSizePt=[w,h],seconds=seconds,
                boxCount=len(boxes),classes=dict(Counter(b['label'] for b in boxes)),boxes=boxes))
        if digest(path)!=before:raise RuntimeError('Source changed')
        print(f'{MODELS[kind]} page {number}: {len(boxes)} boxes ({seconds:.3f}s)',flush=True)
    from backend.models import validate_document
    scene=validate_document(scene)
    encoded=json.dumps(scene,ensure_ascii=False,allow_nan=False).encode('utf-8')
    pdf.embfile_add('scene-document.json',encoded,filename='scene-document.json')
    pdf.embfile_add('detection-results.json',json.dumps(report,ensure_ascii=False).encode('utf-8'))
    pdf.save(output/'detection-preview.pdf',deflate=True)
    pdf.close()
    (output/'scene-document.json').write_bytes(encoded)
    (output/'detection-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with fitz.open(output/'detection-preview.pdf') as check:
        assert len(check)==len(chosen)
        assert json.loads(check.embfile_get('scene-document.json'))==scene
    return scene,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',type=Path,nargs='+')
    parser.add_argument('--detector',choices=MODELS,required=True)
    parser.add_argument('--pages',default='8,9')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run([p.resolve() for p in args.inputs],args.pages,args.detector,args.output.resolve())


if __name__=='__main__':
    main()
