"""Resident child lifecycle and default local V3 routing, without Paddle."""
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from pdf_extractor import native_client, structure
from pdf_extractor.usage import span


FAKE_WORKER = r'''
import json, os, sys, time
count = 0
print('third-party startup diagnostic', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    identifier, operation = request['id'], request.get('op')
    if operation == 'shutdown':
        print(json.dumps(dict(id=identifier, ok=True)), flush=True)
        break
    if operation == 'timeout':
        time.sleep(60)
    if operation == 'wrong-id':
        print(json.dumps(dict(id='stale-reply', ok=True)), flush=True)
        print(json.dumps(dict(id=identifier, ok=True)), flush=True)
        continue
    if operation == 'exit':
        sys.exit(7)
    if operation == 'fail':
        print(json.dumps(dict(id=identifier, ok=False, errorType='RuntimeError', message='synthetic-model-failure')), flush=True)
        continue
    count += 1
    reply = dict(id=identifier, ok=True, pid=os.getpid(), count=count,
                 omp=os.environ.get('OMP_NUM_THREADS'),mkl=os.environ.get('MKL_NUM_THREADS'),
                 result=dict(width=16, height=12, parsing_res_list=[]),
                 engine='fake-native', seconds=0.01, cacheHit=False, modelProfile=[])
    print(json.dumps(reply), flush=True)
'''


class NativeClientTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'native_worker.py').write_text(FAKE_WORKER, encoding='utf-8')
        service = patch.object(native_client, 'SERVICE', self.root)
        service.start()
        self.addCleanup(service.stop)
        environment = patch.dict(os.environ, {'PDFTRANSFER_V3_PYTHON': sys.executable,
                                              'PDFTRANSFER_LOG_DIR': str(self.root / 'logs')})
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('PDFTRANSFER_V3_TRANSPORT', None)
        os.environ.pop('PDFTRANSFER_STRUCTURE_URL', None)
        self.client = native_client.NativeClient()
        # Registered last: close child before patches and temporary files expire.
        self.addCleanup(self.client.close)

    def test_resident_process_reused_and_non_json_startup_noise_ignored(self):
        first = self.client.request(dict(op='recognize'), timeout=5)
        child = self.client.process
        second = self.client.request(dict(op='recognize'), timeout=5)
        self.assertEqual(first['pid'], second['pid'])
        self.assertIs(child, self.client.process)
        self.assertEqual([first['count'], second['count']], [1, 2])
        self.assertTrue(self.client.loaded)
        self.client.close()
        self.assertIsNotNone(child.poll())
        self.assertIsNone(self.client.process)
        self.assertFalse(self.client.loaded)

    def test_child_thread_environment_matches_effective_config_not_parent(self):
        config=dict(cpu_threads=6,pipeline_options=dict(cpu_threads=3))
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}), \
             patch('native_worker.load_config',return_value=config):
            reply=self.client.request(dict(op='recognize'),timeout=5)
            self.assertEqual((reply['omp'],reply['mkl']),('3','3'))
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')

    def test_native_failure_terminates_child_and_does_not_fallback(self):
        with patch.object(structure, '_http_recognize', side_effect=AssertionError('unexpected HTTP')):
            with self.assertRaisesRegex(ValueError, 'synthetic-model-failure'):
                self.client.request(dict(op='fail'), timeout=5)
        self.assertIsNone(self.client.process)
        self.assertFalse(self.client.loaded)
        self.assertEqual(self.client.request(dict(op='recognize'), timeout=5)['count'], 1)

    def test_timeout_terminates_old_process_and_next_request_is_clean(self):
        self.client.request(dict(op='recognize'), timeout=5)
        old = self.client.process
        with self.assertRaisesRegex(ValueError, '超时.*终止'):
            self.client.request(dict(op='timeout'), timeout=.05)
        self.assertIsNotNone(old.poll())
        self.assertIsNone(self.client.process)
        reply = self.client.request(dict(op='recognize'), timeout=5)
        self.assertNotEqual(reply['pid'], old.pid)
        self.assertEqual(reply['count'], 1)

    def test_wrong_response_id_is_rejected_and_extra_old_reply_cannot_leak(self):
        self.client.request(dict(op='recognize'), timeout=5)
        old = self.client.process
        with self.assertRaisesRegex(ValueError, '不匹配的请求编号'):
            self.client.request(dict(op='wrong-id'), timeout=5)
        self.assertIsNotNone(old.poll())
        next_reply = self.client.request(dict(op='recognize'), timeout=5)
        self.assertNotEqual(next_reply['pid'], old.pid)
        self.assertEqual(next_reply['count'], 1)

    def test_unexpected_child_exit_has_no_automatic_backend_fallback(self):
        with self.assertRaisesRegex(ValueError, '进程退出.*不会自动改用 Docker 或旧 OCR'):
            self.client.request(dict(op='exit'), timeout=5)
        self.assertIsNone(self.client.process)

    def test_missing_python_fails_before_process_launch(self):
        with patch.dict(os.environ, {'PDFTRANSFER_V3_PYTHON': str(self.root / 'absent.exe')}), \
                patch.object(native_client.subprocess, 'Popen', side_effect=AssertionError('unexpected spawn')):
            with self.assertRaisesRegex(ValueError, '环境未安装.*无需 Docker'):
                self.client.request(dict(op='recognize'))

    def test_default_native_recognition_uses_png_stdio_and_preserves_profile(self):
        captured = []
        profile = [dict(model='synthetic-model', phase='predict', wallSeconds=.2, processCpuSeconds=.3)]
        def request(message, **kwargs):
            captured.append(message)
            self.assertTrue(base64.b64decode(message['pngBase64']).startswith(b'\x89PNG\r\n\x1a\n'))
            return dict(result=dict(width=16, height=12, parsing_res_list=[]), engine='test-native',
                        seconds=.3, cacheHit=False, modelProfile=profile)
        with patch.object(native_client.client, 'request', side_effect=request), \
                patch.object(structure, 'build_opener', side_effect=AssertionError('unexpected HTTP')):
            with span('reading.request', feature='reading.md') as usage:
                result = structure.recognize_page(Image.new('RGB', (16, 12), 'white'))
        self.assertEqual(structure.transport(), 'native')
        self.assertEqual(captured[0]['op'], 'recognize')
        self.assertEqual(captured[0]['trace']['traceId'], usage.state['traceId'])
        self.assertEqual(captured[0]['trace']['feature'], 'reading.md')
        self.assertEqual(result['_service']['transport'], 'native')
        self.assertEqual(result['_service']['modelProfile'], profile)

    def test_native_error_at_shared_entry_never_calls_http(self):
        with patch.object(native_client.client, 'request', side_effect=ValueError('native failed')), \
                patch.object(structure, '_http_recognize', side_effect=AssertionError('unexpected fallback')):
            with self.assertRaisesRegex(ValueError, 'native failed'):
                structure.recognize_page(Image.new('RGB', (16, 12)))

    def test_default_native_status_does_not_open_http(self):
        with patch.object(native_client.client, 'status', return_value=dict(available=True, transport='native')) as status, \
                patch.object(structure, 'build_opener', side_effect=AssertionError('unexpected HTTP')):
            self.assertEqual(structure.service_status(), dict(available=True, transport='native'))
        status.assert_called_once_with()

    def test_direct_client_import_can_load_status_without_logging_bridge(self):
        environment = dict(os.environ, PYTHONPATH=str(native_client.ROOT / 'transfer_kenel'))
        result = native_client.subprocess.run(
            [sys.executable, '-X', 'utf8', '-c',
             'from pdf_extractor.native_client import NativeClient; '
             'assert "modelDirReady" in NativeClient().status()'],
            cwd=self.root, env=environment, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_status_metadata_probe_is_cached_and_never_starts_model_worker(self):
        config = dict(model_dir=str(self.root), cpu_threads=2, device='cpu')
        probe = SimpleNamespace(returncode=0, stdout=json.dumps(dict(paddle=True, paddleocr=True, paddlex=True)))
        with patch.object(native_client.subprocess, 'run', return_value=probe) as run, \
                patch('native_worker.load_config', return_value=config), \
                patch.object(native_client.subprocess, 'Popen', side_effect=AssertionError('unexpected model process')):
            first, second = self.client.status(), self.client.status()
        self.assertTrue(first['available'])
        self.assertEqual(first, second)
        self.assertFalse(first['loaded'])
        self.assertEqual(run.call_count, 1)

    def test_status_reports_missing_models_without_model_initialization(self):
        probe = SimpleNamespace(returncode=0, stdout=json.dumps(dict(paddle=True, paddleocr=True, paddlex=True)))
        with patch.object(native_client.subprocess, 'run', return_value=probe), \
                patch('native_worker.load_config', return_value=dict(model_dir=str(self.root / 'absent'),
                                                                    cpu_threads=2, device='cpu')):
            status = self.client.status()
        self.assertFalse(status['available'])
        self.assertFalse(status['modelDirReady'])
        self.assertIn('模型目录不存在', status['error'])

    def test_explicit_old_url_selects_http_and_native_override_wins(self):
        with patch.dict(os.environ, {'PDFTRANSFER_STRUCTURE_URL': 'http://127.0.0.1:8790'}):
            self.assertEqual(structure.transport(), 'http')
            expected = dict(width=16, height=12, parsing_res_list=[])
            with patch.object(structure, '_http_recognize', return_value=expected) as http, \
                    patch.object(native_client.client, 'request', side_effect=AssertionError('unexpected native')):
                self.assertIs(structure.recognize_page(Image.new('RGB', (16, 12))), expected)
            self.assertEqual(http.call_count, 1)
            with patch.dict(os.environ, {'PDFTRANSFER_V3_TRANSPORT': 'native'}):
                self.assertEqual(structure.transport(), 'native')

    def test_invalid_transport_is_explicit_configuration_error(self):
        with patch.dict(os.environ, {'PDFTRANSFER_V3_TRANSPORT': 'remote'}):
            with self.assertRaisesRegex(ValueError, '仅支持 native 或 http'):
                structure.transport()

