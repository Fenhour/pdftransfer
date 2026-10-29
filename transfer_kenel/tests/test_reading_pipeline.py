"""Synthetic truth for native reading, independent tables and span styling."""
import hashlib
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import threading
import unittest
from unittest.mock import patch
import uuid

import pymupdf as fitz
from PIL import Image

from pdf_extractor import engine
from pdf_extractor.export import html_block, markdown
from pdf_extractor.model import Options
from pdf_extractor.tables import native_tables
from pdf_extractor.usage import span


class ReadingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('tmp') / ('reading-pipeline-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, True)
        environment = patch.dict(os.environ, {'PDFTRANSFER_LOG_DIR': str(self.root / 'log')})
        environment.start()
        self.addCleanup(environment.stop)

    def source(self, pages=1, picture=False):
        path = self.root / 'native.pdf'
        with fitz.open() as document:
            for index in range(pages):
                page = document.new_page(width=400, height=600)
                page.insert_text((30, 70), f'Native body before table on page {index+1}.', fontsize=11)
                for x in (30, 180, 330):
                    page.draw_line((x, 120), (x, 220))
                for y in (120, 170, 220):
                    page.draw_line((30, y), (330, y))
                for point, text in (((45, 150), 'Name'), ((195, 150), 'Value'),
                                    ((45, 200), f'Cell{index+1}'), ((195, 200), '42')):
                    page.insert_text(point, text, fontsize=11)
                page.insert_text((30, 260), 'Native body after table.', fontsize=11)
                if picture:
                    buffer = io.BytesIO()
                    Image.new('RGB', (300, 200), 'white').save(buffer, format='PNG')
                    page.insert_image(fitz.Rect(30, 300, 330, 500), stream=buffer.getvalue())
            document.save(path)
        return path

    def test_native_span_truth_preserves_color_size_font_bold_italic_and_positions(self):
        with fitz.open() as document:
            page = document.new_page()
            page.insert_text((40, 80), 'Red', fontname='hebi', fontsize=16, color=(1, 0, 0))
            page.insert_text((75, 80), '<safe>', fontname='helv', fontsize=11, color=(0, 0, 1))
            blocks = engine._native_text(page, [])
        red = next(run for block in blocks for line in block.meta['text_lines'] for run in line['runs']
                   if 'Red' in run['text'])
        self.assertEqual(red['color'], '#ff0000')
        self.assertEqual(red['size'], 16)
        self.assertTrue(red['bold'])
        self.assertTrue(red['italic'])
        self.assertIn('BoldOblique', red['font'])
        self.assertEqual(red['bbox'][0], 40)
        for render in (html_block, markdown):
            output = '\n'.join(render(block) for block in blocks)
            self.assertIn('font-weight:bold', output)
            self.assertIn('font-style:italic', output)
            self.assertIn('font-size:16.00pt', output)
            self.assertIn('color:#ff0000', output)
            self.assertIn('data-pdf-font=', output)
            self.assertIn('data-bbox=', output)
            self.assertIn('&lt;safe&gt;', output)
            self.assertNotIn('<safe>', output)

    def test_native_pictures_are_not_ocr_in_auto_mode_for_either_backend(self):
        source = self.source(picture=True)
        for backend in ('legacy', 'v3'):
            with self.subTest(backend=backend), \
                    patch('pdf_extractor.engine.LocalOCR.recognize', side_effect=AssertionError('picture must stay a picture')):
                _, result = engine.run(source, self.root / backend,
                                       Options(ocr='auto', dpi=72, recognition_backend=backend, output_format='md'))
            self.assertFalse(any(block.source == 'ocr' for block in result.blocks))
            picture = next(block for block in result.blocks if block.kind == 'image')
            self.assertEqual(picture.bbox, [30, 300, 330, 500])
            self.assertEqual(len([block for block in result.blocks if block.kind == 'table']), 1)

    def test_short_native_label_does_not_trigger_automatic_ocr(self):
        path = self.root / 'label.pdf'
        with fitz.open() as document:
            page = document.new_page(width=200, height=200)
            page.insert_text((30, 80), 'OK', fontsize=12)
            document.save(path)
        with patch('pdf_extractor.engine.LocalOCR.recognize', side_effect=AssertionError('native label is reliable')):
            _, result = engine.run(path, self.root / 'label', Options(ocr='auto', output_format='html'))
        self.assertEqual([block.text for block in result.blocks], ['OK'])

    def test_short_top_native_title_and_small_picture_never_call_shared_v3(self):
        path = self.root / 'top-title-picture.pdf'
        buffer = io.BytesIO()
        Image.new('RGB', (120, 80), 'white').save(buffer, format='PNG')
        with fitz.open() as document:
            page = document.new_page(width=300, height=220)
            page.insert_text((20, 30), 'Native title', fontsize=12)
            page.insert_image(fitz.Rect(100, 80, 220, 160), stream=buffer.getvalue())
            native = engine._native_text(page, [])
            self.assertLess(native[0].bbox[1], page.rect.height * .08)
            self.assertFalse(engine._structure_scan_page(page, native, []))
            document.save(path)
        with patch('pdf_extractor.structure.recognize_page', side_effect=AssertionError('native title and picture must not call V3')) as model, \
                patch('pdf_extractor.engine.LocalOCR.recognize', side_effect=AssertionError('native picture must not call legacy OCR')) as legacy:
            _, result = engine.run(path, self.root / 'top-title-picture',
                                   Options(ocr='auto', dpi=72, recognition_backend='v3', output_format='html'))
        model.assert_not_called()
        legacy.assert_not_called()
        self.assertEqual([block.text for block in result.blocks if block.kind != 'image'], ['Native title'])
        self.assertEqual(next(block.bbox for block in result.blocks if block.kind == 'image'), [100, 80, 220, 160])

    def test_short_top_watermark_on_full_page_scan_still_requires_recognition(self):
        buffer = io.BytesIO()
        Image.new('RGB', (300, 220), 'white').save(buffer, format='PNG')
        with fitz.open() as document:
            page = document.new_page(width=300, height=220)
            page.insert_image(page.rect, stream=buffer.getvalue())
            page.insert_text((20, 30), 'Native mark', fontsize=12)
            self.assertTrue(engine._structure_scan_page(page, engine._native_text(page, []), []))

    def test_all_pages_native_work_precedes_table_join_and_cells_are_not_duplicated(self):
        source = self.source(pages=3)
        expected = {}
        with fitz.open(source) as document:
            for index in range(len(document)):
                expected[index] = native_tables(document[index], 'auto', [])
        seen = set()
        original = engine._native_text
        def native(page, tables):
            seen.add(page.number)
            return original(page, tables)
        class ReadyQueue:
            def __init__(self, *args):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def take(inner, index, cancel=None):
                self.assertEqual(seen, {0, 1, 2}, 'table wait blocked a later native page')
                return expected[index], []
        with patch.object(engine, '_TableQueue', ReadyQueue), patch.object(engine, '_native_text', side_effect=native):
            _, result = engine.run(source, self.root / 'pipeline', Options(ocr='off', output_format='html'))
        for index in range(3):
            page_blocks = [block for block in result.blocks if block.page == index+1]
            tables = [block for block in page_blocks if block.kind == 'table']
            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0].rows, [['Name', 'Value'], [f'Cell{index+1}', '42']])
            ordinary = ' '.join(block.text for block in page_blocks if block.kind != 'table')
            self.assertNotIn('Cell', ordinary)
            self.assertNotIn('Name', ordinary)
            before = next(i for i, block in enumerate(page_blocks) if 'before table' in block.text)
            after = next(i for i, block in enumerate(page_blocks) if 'after table' in block.text)
            self.assertLess(before, page_blocks.index(tables[0]))
            self.assertLess(page_blocks.index(tables[0]), after)

    def test_worker_cpu_log_is_separate_and_trace_matches_host_with_source_unchanged(self):
        source = self.source(pages=5)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with span('test.reading', feature='reading.html') as operation:
            _, result = engine.run(source, self.root / 'worker', Options(ocr='off', output_format='html'))
        self.assertEqual(len([block for block in result.blocks if block.kind == 'table']), 5)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        records = [json.loads(line) for path in (self.root / 'log').glob('usage-*.jsonl')
                   for line in path.read_text(encoding='utf-8').splitlines()]
        workers = [record for record in records if record['module'] == 'reading.table_worker']
        self.assertEqual(len(workers), 5)
        self.assertEqual({record['metadata']['page'] for record in workers}, {1, 2, 3, 4, 5})
        self.assertTrue(all(record['role'] == 'table-worker' and record['pid'] != os.getpid() for record in workers))
        self.assertTrue(all(record['traceId'] == operation.state['traceId'] for record in workers))
        self.assertTrue(all(record['processCpuSeconds'] >= 0 for record in workers))

    def test_table_failure_retains_ordinary_text_and_original_visual_reference(self):
        class FailedQueue:
            def __init__(self, *args):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def take(self, *args):
                return [], ['原生表格独立进程异常，表格未确认恢复。']
        with patch.object(engine, '_TableQueue', FailedQueue):
            destination, result = engine.run(self.source(), self.root / 'failed', Options(ocr='off', output_format='html'))
        self.assertIn('Native body after table.', ' '.join(block.text for block in result.blocks))
        reference = next(block for block in result.blocks if block.meta.get('table_recovery') == 'unresolved')
        self.assertTrue((destination / 'graph' / Path(reference.asset).name).is_file())
        self.assertTrue(any('未确认恢复' in warning for warning in result.warnings))

    def test_cancellation_during_queue_wait_cleans_worker_and_unpublished_output(self):
        source = self.source(pages=5)
        cancelled = threading.Event()
        def progress(done, total, message):
            if '汇合独立表格队列' in message:
                cancelled.set()
        with self.assertRaises(engine.Cancelled):
            engine.run(source, self.root / 'cancelled', Options(ocr='off'), progress=progress, cancel=cancelled)
        self.assertFalse(any(child.name == 'pdftransfer-native-tables' and child.is_alive()
                             for child in multiprocessing.active_children()))
        self.assertEqual(list((self.root / 'cancelled').iterdir()), [])


if __name__ == '__main__':
    unittest.main()
