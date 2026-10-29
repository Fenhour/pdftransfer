"""The app must not send documents off-machine or silently accept bad geometry."""
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from PIL import Image
from pdf_extractor.structure import endpoint, recognize_page, service_status


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({'service': 'pdftransfer-structure-v3', 'status': 'ok'}).encode())

            def do_POST(self):
                cls.received = self.rfile.read(int(self.headers['Content-Length']))
                cls.received_headers = dict(self.headers)
                self.send_response(cls.status)
                self.end_headers()
                self.wfile.write(json.dumps(cls.body).encode())

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        type(self).status = 200
        type(self).body = {'result': {'width': 16, 'height': 12, 'parsing_res_list': []},
                           'cacheHit': True, 'seconds': .1, 'engine': 'test'}
        self.env = patch.dict(os.environ, {'PDFTRANSFER_STRUCTURE_URL': f'http://127.0.0.1:{self.server.server_port}'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_local_png_contract_and_cache_metadata(self):
        self.assertTrue(service_status()['available'])
        result = recognize_page(Image.new('RGB', (16, 12), 'white'))
        self.assertTrue(self.received.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(result['_service']['cacheHit'], True)
        self.assertEqual((result['width'], result['height']), (16, 12))

    def test_usage_trace_is_forwarded_to_worker(self):
        from pdf_extractor.usage import span
        with span('reading.request', feature='reading.md') as root:
            recognize_page(Image.new('RGB', (16, 12), 'white'))
        self.assertEqual(self.received_headers['X-Pdftransfer-Trace'], root.state['traceId'])
        self.assertEqual(self.received_headers['X-Pdftransfer-Feature'], 'reading.md')
        self.assertNotEqual(self.received_headers['X-Pdftransfer-Parent'], root.state['spanId'])

    def test_remote_and_path_urls_rejected_before_connection(self):
        for url in ('https://127.0.0.1:1', 'http://example.com', 'http://127.0.0.1/data',
                    'http://user:pass@localhost:1', 'http://localhost:1?token=x'):
            with self.subTest(url=url), patch.dict(os.environ, {'PDFTRANSFER_STRUCTURE_URL': url}):
                with self.assertRaisesRegex(ValueError, '本机 HTTP'):
                    endpoint()

    def test_geometry_mismatch_is_an_error(self):
        type(self).body['result']['width'] = 999
        with self.assertRaisesRegex(ValueError, '页面尺寸'):
            recognize_page(Image.new('RGB', (16, 12)))

    def test_worker_failure_is_not_legacy_fallback(self):
        type(self).status = 503
        type(self).body = {'error': 'model unavailable'}
        with self.assertRaisesRegex(ValueError, 'model unavailable.*没有改用旧'):
            recognize_page(Image.new('RGB', (16, 12)))
