"""Resident local PP-StructureV3 process, using a JSON-lines stdio protocol.

This process has no listening socket and never requires Docker.  Its stdout
file descriptor is reserved for protocol replies; native and Python diagnostic
output is redirected to stderr before importing Paddle / PaddleX.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

from model_profile import LocalModelUnavailable, ModelProfiler, collect_profiles
from pdftransfer_usage import span
from runtime_threads import configure_cpu_environment, cpu_threads


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = 'native-structure-3.7.0-paddlex-3.7.2-paddle-3.2.2-profile-1'
MAX_BYTES = 48 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_LINE = (MAX_BYTES * 4 // 3) + 16384
DEFAULTS = dict(cpu_threads=6, device='cpu', enable_mkldnn=True,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                use_textline_orientation=False, use_formula_recognition=False,
                use_chart_recognition=False, use_seal_recognition=False)
PIPELINE_KEYS = set(DEFAULTS) | {'use_table_recognition', 'use_region_detection'}


def _path(value, base=HERE):
    value = Path(value)
    return (value if value.is_absolute() else base / value).resolve()


def load_config(path=None):
    """Load paths and options without importing Paddle or mutating user HOME."""
    selected = _path(path or os.environ.get('PDFTRANSFER_V3_CONFIG') or HERE / 'native_config.json')
    if selected.is_file():
        supplied = json.loads(selected.read_text(encoding='utf-8'))
        if not isinstance(supplied, dict):
            raise ValueError('Native V3 configuration must be a JSON object')
    else:
        supplied = {}
    config = {**DEFAULTS, 'cache_version': '1', **supplied}
    project_models = ROOT / 'V3' / 'models'
    legacy_models = ROOT / 'models' / 'structure_v3'
    model_dir = os.environ.get('PDFTRANSFER_V3_MODEL_DIR') or supplied.get('model_dir') or (
        project_models if project_models.is_dir() else
        legacy_models if legacy_models.is_dir() else
        (Path(os.environ['PDFTRANSFER_PADDLE_ROOT']) / 'cache/.paddlex/official_models'
         if os.environ.get('PDFTRANSFER_PADDLE_ROOT') else Path.home() / '.paddlex/official_models'))
    config['model_dir'] = str(_path(model_dir, selected.parent))
    config['results_dir'] = str(_path(os.environ.get('PDFTRANSFER_V3_RESULTS') or
                              config.get('results_dir', ROOT / 'tmp/v3-native-results'), selected.parent))
    cpu_threads(config)
    if config['device'] != 'cpu':
        raise ValueError('This native runtime currently supports CPU; GPU resource statistics are not implemented')
    for key in ('pipeline_options', 'predict_options', 'model_paths'):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(f'{key} must be a JSON object')
    if not isinstance(config['cache_version'], (str, int)) or isinstance(config['cache_version'], bool):
        raise ValueError('cache_version must be a string or an integer')
    return config


class NativeRuntime:
    def __init__(self, config=None):
        self.config = dict(config or load_config())
        self.pipeline = None
        self.profiler = None
        self.counters = dict(requests=0, cacheHits=0, inferences=0, failures=0)
        identity = {k: v for k, v in self.config.items() if k != 'results_dir'}
        self.cache_signature = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode('utf-8')
        self.runtime_id = hashlib.sha256(BUILD.encode() + self.cache_signature).hexdigest()[:16]
        self.pipeline_options = {key: self.config[key] for key in PIPELINE_KEYS if key in self.config}
        self.pipeline_options.update(self.config.get('pipeline_options', {}))
        self.pipeline_options['device'] = self.config['device']
        self.runtime_metadata = dict(runtimeId=self.runtime_id,
                                     cpuThreads=self.pipeline_options['cpu_threads'],
                                     mkldnn=bool(self.pipeline_options['enable_mkldnn']))

    def health(self):
        return dict(status='ok', service='pdftransfer-structure-v3-native', engine=BUILD,
                    transport='stdio', loaded=self.pipeline is not None,
                    modelDirReady=Path(self.config['model_dir']).is_dir(),
                    **self.runtime_metadata, device=self.config['device'],
                    acceleration='mkldnn' if self.runtime_metadata['mkldnn'] else 'plain',
                    **self.counters)

    def _pipeline(self):
        if self.pipeline is None:
            with span('v3.pipeline_initialize', backend='native', device=self.config['device'],
                      **self.runtime_metadata):
                # These only affect this child process, not the caller's environment.
                configure_cpu_environment(self.config)
                os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
                from paddleocr import PPStructureV3
                self.profiler = ModelProfiler(self.config['model_dir'], device=self.config['device'],
                                              model_paths=self.config.get('model_paths'),
                                              runtime_id=self.runtime_id,
                                              cpu_threads=self.runtime_metadata['cpuThreads'],
                                              mkldnn=self.runtime_metadata['mkldnn']).install()
                # CPU-only is deliberate: CPU seconds are not GPU utilization / FLOPs.
                try:
                    self.pipeline = PPStructureV3(**self.pipeline_options)
                except BaseException:
                    self.profiler.uninstall()
                    self.profiler = None
                    raise
        return self.pipeline

    def recognize(self, data, trace=None, *, rebuild=False):
        if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
            raise ValueError('PNG input must be between 1 byte and 48 MiB')
        if not data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('PNG input is required')
        self.counters['requests'] += 1
        with collect_profiles() as profiles, span('v3.native_worker', remote=trace, backend='native',
                                                **self.runtime_metadata) as usage:
            started = time.perf_counter()
            try:
                answer = self._recognize(data, rebuild=rebuild)
                answer['seconds'] = round(time.perf_counter() - started, 6)
                answer['modelProfile'] = profiles
                answer.update(self.runtime_metadata)
                usage.metadata['cacheHit'] = answer['cacheHit']
                self.counters['cacheHits' if answer['cacheHit'] else 'inferences'] += 1
                return answer
            except BaseException:
                self.counters['failures'] += 1
                raise

    def _recognize(self, data, *, rebuild=False):
        from rebuild_pipeline import POLICY
        policy = POLICY.encode() if rebuild else b''
        key = hashlib.sha256(BUILD.encode() + self.cache_signature + policy + data).hexdigest()
        target = Path(self.config['results_dir']) / (key + '.json')
        if target.is_file():
            try:
                with span('v3.cache_read', backend='native'):
                    cached = json.loads(target.read_text(encoding='utf-8'))
                if (not isinstance(cached, dict) or 'parsing_res_list' not in cached
                        or (rebuild and cached.get('rebuild_policy') != POLICY)):
                    raise ValueError('Invalid cached schema')
                return dict(result=cached, cacheHit=True, engine=BUILD)
            except (ValueError, OSError):
                pass
        import numpy as np
        from PIL import Image
        with span('v3.decode_image', backend='native'), Image.open(io.BytesIO(data)) as source:
            width, height = source.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise ValueError('Image exceeds 40 million pixels; reduce render resolution')
            rgb = np.asarray(source.convert('RGB'))
        model = self._pipeline()
        if rebuild:
            from rebuild_pipeline import recognize_rebuild
            with span('v3.rebuild.predict', backend='native'):
                result = recognize_rebuild(rgb, model)
            self._publish(target, result)
            return dict(result=result, cacheHit=False, engine=BUILD)
        options = dict(use_table_orientation_classify=True,
                       use_e2e_wireless_table_rec_model=False, markdown_ignore_labels=[])
        options.update(self.config.get('predict_options', {}))
        with span('v3.predict', backend='native'):
            predictions = iter(model.predict_iter(rgb[:, :, ::-1].copy(), **options))
            try:
                first = next(predictions, None)
            finally:
                close = getattr(predictions, 'close', None)
                if callable(close):
                    close()
        if first is None:
            raise ValueError('PP-StructureV3 returned no page result')
        with span('v3.result_serialize', backend='native'):
            payload = first.json
            if isinstance(payload, str):
                payload = json.loads(payload)
            result = payload.get('res', payload)
            if not isinstance(result, dict) or 'parsing_res_list' not in result:
                raise ValueError('Unexpected PP-StructureV3 response schema')
            result['width'], result['height'], result['input_path'] = width, height, None
        self._publish(target, result)
        return dict(result=result, cacheHit=False, engine=BUILD)

    def _publish(self, target, result):
        # Cache publication failures are not inference failures.
        try:
            with span('v3.cache_write', backend='native'):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.' + uuid.uuid4().hex + '.tmp')
                try:
                    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            pass

    def close(self):
        if self.pipeline is not None:
            pipeline = getattr(self.pipeline, 'paddlex_pipeline', self.pipeline)
            close = getattr(pipeline, 'close', None)
            if callable(close):
                close()
        if self.profiler is not None:
            self.profiler.uninstall()


def handle_request(runtime, request):
    if not isinstance(request, dict):
        raise ValueError('Protocol request must be a JSON object')
    operation = request.get('op')
    if operation == 'health':
        return dict(id=request.get('id'), ok=True, **runtime.health())
    if operation == 'shutdown':
        return dict(id=request.get('id'), ok=True, shutdown=True)
    if operation not in {'recognize', 'recognize_rebuild'}:
        raise ValueError('Unsupported native V3 operation')
    image = request.get('pngBase64')
    if not isinstance(image, str) or len(image) > MAX_BYTES * 4 // 3 + 8:
        raise ValueError('Invalid PNG payload size')
    try:
        data = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError('Invalid base64 PNG payload') from None
    trace = request.get('trace')
    if trace is not None and not isinstance(trace, dict):
        raise ValueError('Invalid trace metadata')
    if operation == 'recognize_rebuild':
        return dict(id=request.get('id'), ok=True, **runtime.recognize(data, trace, rebuild=True))
    return dict(id=request.get('id'), ok=True, **runtime.recognize(data, trace))


def serve(runtime, input_stream, output_stream):
    """The reply never includes exception strings that may expose local paths."""
    while True:
        line = input_stream.readline(MAX_LINE + 1)
        if not line:
            break
        request = None
        try:
            if len(line) > MAX_LINE:
                raise ValueError('Native V3 request is too large')
            request = json.loads(line)
            reply = handle_request(runtime, request)
        except Exception as error:
            reply = dict(id=request.get('id') if isinstance(request, dict) else None,
                         ok=False, errorType=type(error).__name__,
                         message=str(error) if isinstance(error, LocalModelUnavailable) else
                         'Local V3 failed; check the local runtime, model configuration and usage log')
        output_stream.write(json.dumps(reply, ensure_ascii=False, allow_nan=False) + '\n')
        output_stream.flush()
        if reply.get('shutdown') or len(line) > MAX_LINE:
            break


def main():
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', newline='\n', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    os.environ.setdefault('PDFTRANSFER_LOG_ROLE', 'v3-native')
    runtime = None
    try:
        runtime = NativeRuntime()
        serve(runtime, sys.stdin, protocol)
    finally:
        if runtime is not None:
            runtime.close()
        protocol.close()


if __name__ == '__main__':
    main()
