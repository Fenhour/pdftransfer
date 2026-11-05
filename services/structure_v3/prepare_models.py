"""Prepare official inference models at install time; runtime stays offline."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[2]
MODELS=('PP-DocLayout_plus-L','PP-DocBlockLayout','PP-OCRv5_server_det','PP-OCRv5_server_rec',
        'PP-LCNet_x1_0_table_cls','SLANeXt_wired','SLANet_plus','RT-DETR-L_wired_table_cell_det',
        'RT-DETR-L_wireless_table_cell_det','PP-LCNet_x1_0_doc_ori','PP-LCNet_x1_0_textline_ori')
FILES=('inference.json','inference.pdiparams','inference.yml')

def ready(directory):
    return all((directory/name).is_file() and (directory/name).stat().st_size>0 for name in FILES)

def validate(directory):
    config=json.loads((Path(__file__).with_name('native_config.json')).read_text(encoding='utf8'))
    missing=[name for name in MODELS if not ready(directory/config.get('model_paths',{}).get(name,name))]
    if missing: raise ValueError('Missing/incomplete local models: '+', '.join(missing))
    return len(MODELS)

def prepare(directory):
    directory=directory.resolve()
    if os.name=='nt' and not str(directory).isascii():
        raise ValueError('Use an ASCII-only model directory on Windows.')
    directory.mkdir(parents=True,exist_ok=True)
    # This manager belongs to the exact pinned PaddleX runtime. It downloads
    # official model packages without constructing or running an inference model.
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    from paddlex.inference.utils.official_models import official_models
    for name in MODELS:
        target=directory/name
        if ready(target):
            print('Already prepared: '+name,flush=True)
            continue
        if target.exists():
            raise FileExistsError(f'Incomplete model directory: {target}. Move it aside and retry.')
        print('Preparing official model: '+name,flush=True)
        source=Path(official_models[name])
        if not ready(source): raise ValueError('Official download is incomplete: '+name)
        with tempfile.TemporaryDirectory(prefix='.download-',dir=directory) as temporary:
            staged=Path(temporary)/name
            shutil.copytree(source,staged)
            staged.rename(target)
    from prepare_cpu_rec import prepare as prepare_rec
    from prepare_cpu_det import prepare as prepare_det
    config=json.loads(Path(__file__).with_name('native_config.json').read_text(encoding='utf8'))
    prepare_rec(directory/'PP-OCRv5_server_rec',directory/config['model_paths']['PP-OCRv5_server_rec'])
    prepare_det(directory/'PP-OCRv5_server_det',directory/config['model_paths']['PP-OCRv5_server_det'])
    return validate(directory)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=ROOT/'V3/models')
    parser.add_argument('--check',action='store_true',help='Validate local files without downloading or inference')
    options=parser.parse_args()
    count=validate(options.model_dir) if options.check else prepare(options.model_dir)
    print(f'{count} models are ready. Model inference runs locally.',flush=True)

if __name__=='__main__':
    main()
