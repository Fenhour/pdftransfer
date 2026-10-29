"""Regression cases for list structure, quoted prompts and positioned OCR."""
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
import uuid
from unittest.mock import patch

import pymupdf as fitz

from pdf_extractor.engine import _native_text, run
from pdf_extractor.export import html_block, markdown
from pdf_extractor.model import Block, Options


def fake_page(rows):
    lines = []
    for i, (text, mono) in enumerate(rows):
        y = 120 + i * 14
        box = [50, y, 450, y + 10]
        lines.append({"bbox": box, "dir": (1, 0), "spans": [
            {"text": text, "bbox": box, "origin": (50, y + 8),
             "size": 10, "flags": 8 if mono else 0}]})
    raw = {"type": 0, "bbox": [50, 120, 450, 120 + len(rows) * 14], "lines": lines}
    return SimpleNamespace(number=0, rect=fitz.Rect(0, 0, 600, 800),
                           get_text=lambda *args, **kwargs: {"blocks": [raw]})


class TextRegressionTests(unittest.TestCase):
    def test_numbered_items_keep_their_own_paragraphs(self):
        page = fake_page([("(1) First directive", False), ("continued here;", False),
                          ("(2) Agent state;", False), ("(3) Previous tasks;", False),
                          ("(4) Additional context.", False)])
        blocks = _native_text(page, [])
        self.assertEqual([b.text for b in blocks], [
            "(1) First directive continued here;", "(2) Agent state;",
            "(3) Previous tasks;", "(4) Additional context."])
        self.assertEqual(sum(markdown(b).count("(2)") for b in blocks), 1)

    def test_mixed_font_quote_is_one_prose_item(self):
        page = fake_page([
            ('(1) Guidelines for code generation, such as “Your function will be reused', False),
            ('for building more complex functions. Therefore, you should make', True),
            ('it generic and reusable.”;', True),
            ('(2) Control primitive APIs and relevant skills;', False)])
        blocks = _native_text(page, [])
        self.assertEqual(len(blocks), 2)
        self.assertFalse(any(b.meta.get("code") for b in blocks))
        self.assertIn('“Your function will be reused for building', blocks[0].text)
        self.assertTrue(blocks[0].text.endswith('it generic and reusable.”;'))
        self.assertNotIn('```', markdown(blocks[0]))
        self.assertNotIn('<pre>', html_block(blocks[0]))

    def test_complete_monospaced_prompt_quote_is_prose(self):
        page = fake_page([('(1) Directives, such as', False),
                          ('“My ultimate goal is to discover things', True),
                          ('and reuse skills.”;', True), ('(2) State;', False)])
        blocks = _native_text(page, [])
        self.assertEqual(len(blocks), 2)
        self.assertFalse(blocks[0].meta.get('code'))
        self.assertIn('such as “My ultimate goal', blocks[0].text)

    def test_quote_end_does_not_reclassify_real_code(self):
        page = fake_page([('Quoted guidance: “Be reusable', False),
                          ('and generic.”', True),
                          ('const xs = [];', True), ('xs.push(data[0]);', True)])
        blocks = _native_text(page, [])
        self.assertEqual(len(blocks), 2)
        self.assertFalse(blocks[0].meta.get('code'))
        self.assertTrue(blocks[1].meta.get('code'))
        self.assertEqual(blocks[1].text, 'const xs = [];\nxs.push(data[0]);')

    def test_ascii_quote_and_fullwidth_list_markers(self):
        page = fake_page([('（1） Guidance: "Your function is reusable', False),
                          ('and generic.";', True), ('（2） State;', False)])
        blocks = _native_text(page, [])
        self.assertEqual(len(blocks), 2)
        self.assertFalse(blocks[0].meta.get('code'))

    def test_positioned_ocr_duplicate_tolerates_quote_errors(self):
        from pdf_extractor.ocr import duplicates_native
        native = Block('text', 1, [300, 90, 445, 125], 'grouped text', meta={
            'text_lines': [{'bbox': [310, 95, 440, 103],
                            'text': 'throw new Error(`No item named ${name}`);'},
                           {'bbox': [310, 105, 390, 113], 'text': 'No item named acacia_axe'}]})
        bad = "throw new Error('No item named ${name});"
        self.assertTrue(duplicates_native([309, 95, 441, 103], bad, [native]))
        # Identical text at a genuinely different position is not a duplicate.
        self.assertFalse(duplicates_native([309, 115, 441, 123], bad, [native]))
        self.assertFalse(duplicates_native([309, 95, 441, 103], 'Only visible in raster', [native]))

    def test_short_label_and_ocr_spelling_variants_are_not_globally_deleted(self):
        from pdf_extractor.ocr import duplicates_native
        native = [Block('text', 1, [10, 10, 50, 20], 'Save')]
        self.assertTrue(duplicates_native([10, 10, 50, 20], 'Save', native))
        self.assertFalse(duplicates_native([10, 10, 50, 20], 'Safe', native))
        self.assertFalse(duplicates_native([10, 30, 50, 40], 'Save', native))
        code = [Block('text', 1, [10, 60, 250, 72], 'await craftItem(bot, "acacia_axe", 1);')]
        self.assertTrue(duplicates_native([10, 60, 250, 72], 'await craftltem(bot, "acacia_axe", 1);', code))

    def test_figure_transcript_orders_columns_before_rows(self):
        from pdf_extractor.engine import _figure_text
        asset = 'assets/image.png'
        def label(text, box):
            return Block('ocr_text', 1, box, text, source='ocr',
                         meta={'image': asset, 'figure_label': True})
        caption = Block('text', 1, [10, 50, 216, 60], 'Figure 1: Caption', meta={'caption_for': asset})
        labels = [label('right second', [116, 30, 216, 40]),
                  label('left second', [10, 30, 110, 40]),
                  label('right first', [116, 10, 216, 20]),
                  label('left first', [10, 10, 110, 20])]
        self.assertEqual([b.text for b in _figure_text(labels, 'columns')],
                         ['left first', 'left second', 'right first', 'right second'])
        self.assertEqual([b.text for b in _figure_text(labels + [caption], 'columns')],
                         ['left first', 'left second', 'right first', 'right second', 'Figure 1: Caption'])
        self.assertEqual([b.text for b in _figure_text(labels, 'rows')],
                         ['left first', 'right first', 'left second', 'right second'])

    def test_native_transcript_preserves_lines_and_code_indent(self):
        from pdf_extractor.engine import _figure_text
        block = Block('text', 1, [10, 10, 200, 45], 'flattened', meta={
            'image': 'assets/image.png', 'figure_label': True, 'code': True, 'font_size': 10,
            'text_lines': [{'text': 'if (ready) {', 'bbox': [10, 10, 180, 20]},
                           {'text': 'run();', 'bbox': [30, 25, 120, 35]},
                           {'text': '}', 'bbox': [10, 35, 15, 45]}]})
        lines = _figure_text([block], 'columns')
        self.assertEqual([b.text for b in lines], ['if (ready) {', '    run();', '}'])
        self.assertEqual(lines[1].bbox, [30, 25, 120, 35])

    def test_default_page_order_keeps_original_coordinate_precision(self):
        from pdf_extractor.layout import reading_order
        left = Block('text', 1, [10, 10.04, 100, 20], 'left')
        right = Block('text', 1, [20, 10.03, 110, 20], 'right')
        self.assertEqual(reading_order([left, right]), [right, left])
        self.assertEqual(reading_order([left, right], 'rows'), [left, right])

    def test_distant_same_baseline_native_labels_stay_separate(self):
        from pdf_extractor.text import visual_lines
        page = fake_page([('GPT-4', False), ('GPT-4', False)])
        data = page.get_text()
        raw = data['blocks'][0]
        for line, box in zip(raw['lines'], ([100, 120, 125, 130], [300, 120, 325, 130])):
            line['bbox'] = box
            line['spans'][0]['bbox'] = box
            line['spans'][0]['origin'] = (box[0], 128)
        self.assertEqual(len(visual_lines(raw)), 2)
        native = _native_text(page, [])
        self.assertEqual([b.text for b in native], ['GPT-4', 'GPT-4'])
        self.assertEqual([b.bbox for b in native], [[100, 120, 125, 130], [300, 120, 325, 130]])

    def test_diff_prefix_is_not_a_column_and_stays_on_its_row(self):
        from pdf_extractor.engine import _figure_text
        def row(text, box):
            return Block('ocr_text', 1, box, text, source='ocr',
                         meta={'figure_label': True, 'image': 'assets/image.png'})
        labels = [row('first statement', [20, 10, 160, 20]),
                  row('second statement', [30, 30, 160, 40]),
                  row('third statement', [30, 50, 160, 60]),
                  row('+', [15, 34, 18, 37])]
        self.assertEqual([b.text for b in _figure_text(labels, 'columns')],
                         ['first statement', '+ second statement', 'third statement'])

    def test_duplicate_filter_runs_after_pixel_to_page_mapping(self):
        from PIL import Image
        root = Path('tmp') / ('positioned-ocr-' + uuid.uuid4().hex)
        root.mkdir(parents=True)
        try:
            tile = root / 'raster.png'
            Image.new('RGB', (100, 100), 'white').save(tile)
            source = root / 'fixture.pdf'
            region = fitz.Rect(100, 100, 500, 300)
            with fitz.open() as doc:
                p = doc.new_page(width=600, height=800)
                p.insert_text((50, 80), 'Body text establishes a native-text document.', fontsize=11)
                p.insert_image(fitz.Rect(100, 100, 200, 200), filename=str(tile))
                p.insert_text((300, 130), 'throw new Error(`No item named ${name}`);', fontsize=6)
                native = next(b for b in _native_text(p, []) if 'throw' in b.text)
                box = native.bbox
                doc.save(source)
            def recognize(path):
                with Image.open(path) as image:
                    width, height = image.size
                def pixels(b):
                    return [(b[0]-region.x0)*width/region.width, (b[1]-region.y0)*height/region.height,
                            (b[2]-region.x0)*width/region.width, (b[3]-region.y0)*height/region.height]
                return [{'bbox': pixels(box), 'text': "throw new Error('No item named ${name});", 'confidence': .97},
                        {'bbox': pixels([300, 220, 420, 230]), 'text': 'RASTER ONLY NOTE', 'confidence': .93}]
            with patch('pdf_extractor.engine.figure_regions', return_value=[(region, None)]), \
                    patch('pdf_extractor.engine.LocalOCR.recognize', side_effect=recognize):
                destination, result = run(source, root / 'output', Options(ocr='all'))
            ocr = [b for b in result.blocks if b.source == 'ocr']
            self.assertEqual([b.text for b in ocr], ['RASTER ONLY NOTE'])
            self.assertEqual(ocr[0].bbox, [300, 220, 420, 230])
            self.assertAlmostEqual(ocr[0].confidence, .93)
            graph = (destination / 'graph' / 'graph.html').read_text(encoding='utf8')
            self.assertEqual(graph.count('throw new Error'), 1)
            self.assertIn('RASTER ONLY NOTE', graph)
        finally:
            assert root.resolve().is_relative_to(Path('tmp').resolve())
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
