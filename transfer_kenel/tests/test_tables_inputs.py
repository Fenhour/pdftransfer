"""Table spans, sparse ruled structure, previews and scan-watermark regression."""
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

import pymupdf as fitz

from pdf_extractor.engine import _native_text, _scan_page, run
from pdf_extractor.export import html_block, markdown
from pdf_extractor.model import Options
from pdf_extractor.paths import INPUT_PDF, INPUT_OCR, PROJECT_ROOT
from pdf_extractor.tables import native_tables


class TableInputTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('tmp') / ('tables-inputs-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        assert self.root.resolve().is_relative_to(Path('tmp').resolve())
        shutil.rmtree(self.root)

    def test_input_paths_independent_of_cwd(self):
        self.assertEqual(INPUT_PDF, PROJECT_ROOT / 'input' / 'pdf')
        self.assertEqual(INPUT_OCR, INPUT_PDF / 'ocr')

    def test_merged_header_and_vertical_cell_geometry(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        # Header covers 2 columns; first body cell covers 2 rows.
        for y in (120, 150, 240):
            p.draw_line((50, y), (450, y))
        p.draw_line((170, 195), (450, 195))
        for x in (50, 170, 450):
            p.draw_line((x, 120), (x, 240))
        p.draw_line((310, 150), (310, 240))
        for point, text in [((60, 140), 'Name'), ((180, 140), 'Combined values'),
                            ((60, 180), 'Merged label'), ((180, 180), 'A'),
                            ((320, 180), 'B'), ((180, 225), 'C'), ((320, 225), 'D')]:
            p.insert_text(point, text, fontsize=10)
        tables = native_tables(p, 'lines_strict', [])
        self.assertEqual(len(tables), 1)
        b = tables[0]
        self.assertEqual(len(b.rows), 3)
        cells = b.meta['grid_cells']
        self.assertEqual(next(c for c in cells if c['text']=='Combined values')['colspan'], 2)
        self.assertEqual(next(c for c in cells if c['text']=='Merged label')['rowspan'], 2)
        rendered = html_block(b)
        self.assertIn('colspan="2"', rendered)
        self.assertIn('rowspan="2"', rendered)
        self.assertEqual(rendered.count('Merged label'), 1)
        self.assertIn('<colgroup>', rendered)
        self.assertIn('<table', markdown(b))
        self.assertNotIn('列 1', markdown(b))
        self.assertIn('C', rendered)
        doc.close()

    def test_captioned_three_rule_table_recovers_rows_and_columns(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        p.insert_text((50, 105), 'Table 1: Methods and results.', fontsize=10)
        for y in (120, 145, 215):
            p.draw_line((50, y), (500, y))
        p.draw_line((220, 120), (220, 215))
        for y, values in [(136, ['Method', 'First result', 'Second result']),
                          (161, ['Baseline', '10', '20']), (181, ['Our method', '30', '40']),
                          (201, ['Alternative', '50', '60'])]:
            for x, value in zip((60, 260, 395), values):
                p.insert_text((x, y), value, fontsize=10)
        b = native_tables(p, 'auto', [])[0]
        self.assertEqual(b.rows, [['Method', 'First result', 'Second result'],
                                 ['Baseline', '10', '20'], ['Our method', '30', '40'],
                                 ['Alternative', '50', '60']])
        self.assertTrue(b.meta['structure_inferred'])
        cells = b.meta['grid_cells']
        self.assertFalse(cells[-1]['borders']['left'])
        self.assertTrue(cells[-1]['borders']['bottom'])
        doc.close()

    def test_chart_rules_without_table_caption_are_not_sparse_tables(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        for y in (120, 145, 215):
            p.draw_line((50, y), (500, y))
        for y in (136, 165, 195):
            p.insert_text((60, y), 'Chart axis', fontsize=10)
            p.insert_text((260, y), '10', fontsize=10)
        self.assertFalse(native_tables(p, 'auto', []))
        doc.close()

    def test_native_table_preview_links_in_both_formats(self):
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        p.insert_text((50, 80), 'Body text before a native table.', fontsize=11)
        for x in (50, 220, 390):
            p.draw_line((x, 120), (x, 220))
        for y in (120, 170, 220):
            p.draw_line((50, y), (390, y))
        for point, text in [((60, 150), 'Name'), ((230, 150), 'Value'),
                            ((60, 200), 'Cell'), ((230, 200), '42')]:
            p.insert_text(point, text, fontsize=11)
        source = self.root / 'native.pdf'
        doc.save(source)
        doc.close()
        dest, result = run(source, self.root / 'output', Options(ocr='off'))
        table = next(b for b in result.blocks if b.kind=='table')
        self.assertIn('preview_asset', table.meta)
        for fmt in ('md','html'):
            tree = dest.parent.parent / f'output_{fmt}' / dest.name
            self.assertTrue((tree / 'graph' / Path(table.meta['preview_asset']).name).is_file())
            content = (tree / 'table' / f'table.{fmt}').read_text(encoding='utf8')
            self.assertIn('原表截图', content)
            self.assertIn('../graph/', content)

    def test_scan_watermark_does_not_disable_ocr(self):
        from PIL import Image
        tile = self.root / 'scan.png'
        Image.new('RGB', (600,800), 'white').save(tile)
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        p.insert_image(fitz.Rect(40,80,340,480), filename=str(tile))
        p.insert_text((50, 30), 'Local sample watermark with more than thirty characters.', fontsize=9)
        self.assertTrue(_scan_page(p, _native_text(p, []), []))
        source = self.root / 'scan.pdf'
        doc.save(source)
        doc.close()
        with patch('pdf_extractor.engine.LocalOCR.recognize', return_value=[
                {'bbox': [100,200,700,240], 'text': 'CONTENT FROM SCAN', 'confidence': .96}]) as recognize:
            _, result = run(source, self.root / 'output')
        recognize.assert_called_once()
        self.assertTrue(next(b for b in result.blocks if b.kind=='image').meta['scan_reference'])
        self.assertIn('CONTENT FROM SCAN', ' '.join(b.text for b in result.blocks))

    def test_local_ocr_runs_without_network(self):
        from PIL import Image, ImageDraw, ImageFont
        from pdf_extractor.ocr import LocalOCR
        font = Path('C:/Windows/Fonts/arial.ttf')
        if not font.exists():
            self.skipTest('OCR fixture font unavailable')
        image = Image.new('RGB', (800, 140), 'white')
        ImageDraw.Draw(image).text((30,40), 'LOCAL OCR TEST', fill='black', font=ImageFont.truetype(str(font), 42))
        path = self.root / 'offline.png'
        image.save(path)
        with patch('socket.socket', side_effect=AssertionError('Network is disabled')), \
                patch('socket.create_connection', side_effect=AssertionError('Network is disabled')):
            lines = LocalOCR().recognize(path)
        self.assertIn('LOCAL OCR TEST', ' '.join(line['text'] for line in lines).upper())

    def test_native_body_and_small_raster_do_not_become_scan(self):
        from PIL import Image
        tile = self.root / 'small.png'
        Image.new('RGB', (100,100), 'white').save(tile)
        doc = fitz.open()
        p = doc.new_page(width=600, height=800)
        p.insert_image(fitz.Rect(50,120,150,220), filename=str(tile))
        p.insert_text((50,300), 'This is a native text paragraph with a small inline diagram.', fontsize=11)
        self.assertFalse(_scan_page(p, _native_text(p, []), []))
        doc.close()


if __name__ == '__main__':
    unittest.main()
