"""Native stdio contract, local configuration, cache, and result preservation."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from pdf_extractor import usage
from native_worker import NativeRuntime, handle_request, load_config, serve
from runtime_threads import configure_cpu_environment


class NativeWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / 'models'
        self.models.mkdir()
        self.path = self.root / 'native_config.json'
        self.path.write_text(json.dumps(dict(model_dir='models', results_dir='results', cpu_threads=2)), encoding='utf-8')
        environment = patch.dict(os.environ, {'PDFTRANSFER_LOG_DIR': str(self.root / 'logs')})
        environment.start()
        self.addCleanup(environment.stop)
        self.config = load_config(self.path)

    def png(self):
        output = io.BytesIO()
        Image.new('RGB', (16, 12), (1, 2, 3)).save(output, format='PNG')
        return output.getvalue()

    def test_configuration_is_read_only_and_model_root_can_migrate(self):
        previous_home = os.environ.get('HOME')
        self.assertEqual(Path(self.config['model_dir']), self.models)
        with patch.dict(os.environ, {'PDFTRANSFER_V3_MODEL_DIR': str(self.root / 'moved')}):
            self.assertEqual(Path(load_config(self.path)['model_dir']), self.root / 'moved')
        self.assertEqual(os.environ.get('HOME'), previous_home)

    def test_explicit_model_options_and_cpu_validation(self):
        self.path.write_text(json.dumps(dict(cpu_threads=0)), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'cpu_threads'):
            load_config(self.path)
        self.path.write_text(json.dumps(dict(device='gpu:0')), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'GPU resource statistics'):
            load_config(self.path)

    def test_thread_configuration_is_bounded_and_direct_worker_uses_same_environment(self):
        environment={'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        self.assertEqual(configure_cpu_environment(dict(cpu_threads=6),environment),6)
        self.assertEqual(environment,{'OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6'})
        for value in (True,0,257,'6'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                configure_cpu_environment(dict(cpu_threads=6,pipeline_options=dict(cpu_threads=value)),{})

    def test_project_v3_models_are_default_without_overriding_explicit_paths(self):
        project_models = self.root / 'V3' / 'models'
        project_models.mkdir(parents=True)
        with patch('native_worker.ROOT', self.root):
            self.assertEqual(Path(load_config(self.path)['model_dir']), self.models)
            self.path.write_text('{}', encoding='utf-8')
            self.assertEqual(Path(load_config(self.path)['model_dir']), project_models)
            with patch.dict(os.environ, {'PDFTRANSFER_V3_MODEL_DIR': str(self.root / 'override')}):
                self.assertEqual(Path(load_config(self.path)['model_dir']), self.root / 'override')

    def test_runtime_identifier_is_stable_and_cache_version_invalidates_results(self):
        first = NativeRuntime(self.config)
        reordered = NativeRuntime(dict(reversed(list(self.config.items()))))
        alternate_results = NativeRuntime(dict(self.config, results_dir=str(self.root / 'another-results')))
        changed = NativeRuntime(dict(self.config, cache_version='weights-replaced-2'))
        self.assertRegex(first.runtime_id, r'^[0-9a-f]{16}$')
        self.assertEqual(first.runtime_id, reordered.runtime_id)
        self.assertEqual(first.runtime_id, alternate_results.runtime_id)
        self.assertNotEqual(first.runtime_id, changed.runtime_id)
        self.assertNotEqual(first.cache_signature, changed.cache_signature)
        self.assertEqual(self.config['cache_version'], '1')

    def test_effective_nested_pipeline_options_are_reported(self):
        runtime = NativeRuntime(dict(self.config, pipeline_options=dict(cpu_threads=4, enable_mkldnn=False)))
        status = runtime.health()
        self.assertEqual(status['cpuThreads'], 4)
        self.assertFalse(status['mkldnn'])
        self.assertEqual(status['acceleration'], 'plain')
        self.assertEqual(status['runtimeId'], runtime.runtime_id)

    def test_cache_version_requires_a_scalar_identifier(self):
        self.path.write_text(json.dumps(dict(cache_version=['invalid'])), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'cache_version'):
            load_config(self.path)

    def test_health_does_not_import_paddle_or_initialize_pipeline(self):
        runtime = NativeRuntime(self.config)
        with patch.dict(sys.modules, {'paddleocr': None}):
            reply = handle_request(runtime, dict(id='health', op='health'))
        self.assertTrue(reply['ok'])
        self.assertTrue(reply['modelDirReady'])
        self.assertFalse(reply['loaded'])
        self.assertEqual(reply['transport'], 'stdio')

    def test_predict_preserves_v3_json_and_second_request_uses_cache(self):
        runtime = NativeRuntime(self.config)
        calls = []
        class Pipeline:
            def predict_iter(self, rgb, **kwargs):
                calls.append((rgb[0, 0].tolist(), kwargs))
                yield SimpleNamespace(json=dict(res=dict(parsing_res_list=[dict(block_content='synthetic')],
                                                          overall_ocr_res=dict(rec_texts=['synthetic']))))
        runtime.pipeline = Pipeline()
        request = dict(id=1, op='recognize', pngBase64=base64.b64encode(self.png()).decode(),
                       trace=dict(traceId='a' * 32, parentId='b' * 32, feature='reading.md'))
        first = handle_request(runtime, request)
        second = handle_request(runtime, request)
        self.assertFalse(first['cacheHit'])
        self.assertTrue(second['cacheHit'])
        self.assertEqual(first['result'], second['result'])
        self.assertEqual(first['result']['overall_ocr_res']['rec_texts'], ['synthetic'])
        self.assertEqual(first['result']['width'], 16)
        self.assertEqual(first['result']['height'], 12)
        self.assertIsNone(first['result']['input_path'])
        self.assertEqual(second['modelProfile'], [])
        self.assertEqual(first['runtimeId'], runtime.runtime_id)
        self.assertEqual(first['cpuThreads'], 2)
        self.assertTrue(first['mkldnn'])
        self.assertEqual(calls[0][0], [3, 2, 1])
        self.assertEqual(len(calls), 1)
        self.assertEqual(runtime.health()['cacheHits'], 1)

    def test_cache_version_prevents_reusing_identical_image_with_old_weights_version(self):
        calls = []
        class Pipeline:
            def predict_iter(self, image, **kwargs):
                calls.append(1)
                yield SimpleNamespace(json={'res': {'parsing_res_list': []}})
        first = NativeRuntime(self.config)
        first.pipeline = Pipeline()
        first.recognize(self.png())
        changed = NativeRuntime(dict(self.config, cache_version='2'))
        changed.pipeline = Pipeline()
        result = changed.recognize(self.png())
        self.assertFalse(result['cacheHit'])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list(Path(self.config['results_dir']).glob('*.json'))), 2)

    def test_invalid_cache_runs_inference_again(self):
        runtime = NativeRuntime(self.config)
        class Pipeline:
            def predict_iter(self, image, **kwargs):
                yield SimpleNamespace(json={'res': {'parsing_res_list': []}})
        runtime.pipeline = Pipeline()
        first = runtime.recognize(self.png())
        target = next(Path(self.config['results_dir']).glob('*.json'))
        target.write_text('[]', encoding='utf-8')
        self.assertFalse(runtime.recognize(self.png())['cacheHit'])

    def test_bad_images_and_protocol_requests_fail_explicitly(self):
        runtime = NativeRuntime(self.config)
        for request in ([], dict(op='other'), dict(op='recognize', pngBase64='not base64'),
                        dict(op='recognize', pngBase64=base64.b64encode(b'not PNG').decode())):
            with self.subTest(request=request), self.assertRaises(ValueError):
                handle_request(runtime, request)

    def test_protocol_errors_exclude_exception_content_and_continue(self):
        class Runtime:
            def recognize(self, data, trace):
                raise RuntimeError('private document and C:/private/input.pdf')
        request = dict(id='bad', op='recognize', pngBase64=base64.b64encode(self.png()).decode())
        incoming = io.StringIO(json.dumps(request) + '\n' + json.dumps(dict(id='done', op='shutdown')) + '\n')
        outgoing = io.StringIO()
        serve(Runtime(), incoming, outgoing)
        replies = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertFalse(replies[0]['ok'])
        self.assertEqual(replies[0]['errorType'], 'RuntimeError')
        self.assertNotIn('private', outgoing.getvalue())
        self.assertTrue(replies[1]['shutdown'])

    def test_worker_process_can_start_and_exit_without_docker_or_paddle(self):
        worker = Path(__file__).resolve().parents[2] / 'services/structure_v3/native_worker.py'
        requests = '\n'.join(json.dumps(dict(id=index, op=operation))
                             for index, operation in enumerate(('health', 'shutdown'))) + '\n'
        environment = dict(os.environ, PDFTRANSFER_V3_CONFIG=str(self.path))
        completed = subprocess.run([sys.executable, '-X', 'utf8', '-u', str(worker)], input=requests,
                                   encoding='utf-8', capture_output=True, timeout=15, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        replies = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(replies), 2)
        self.assertTrue(replies[0]['ok'])
        self.assertEqual(replies[0]['transport'], 'stdio')
        self.assertTrue(replies[1]['shutdown'])

