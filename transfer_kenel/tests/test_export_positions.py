"""Synthetic export regressions; no private PDFs or OCR backend required."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote

from pdf_extractor.export import attachments, export_all, html_block, markdown, top_blocks
from pdf_extractor.model import Block, Result


class Markup(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.tags = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def find(self, tag=None, css=None):
        return [attrs for name, attrs in self.tags
                if (tag is None or name == tag)
                and (css is None or css in attrs.get('class', '').split())]


class ExportPositionTests(unittest.TestCase):
    def setUp(self):
        # Nonzero crop origin and a raster size unrelated to PDF units catch
        # accidental normalization against page or bitmap dimensions.
        self.image = Block('image', 3, [100, 200, 500, 400], id='figure-3',
                           asset='assets/crop & sample.png',
                           meta={'width': 1600, 'height': 800, 'caption': 'A < B'})
        self.native = Block('text', 3, [140, 220, 260, 240], 'native <label>', id='native-1',
                            meta={'image': self.image.asset, 'figure_label': True})
        self.ocr = Block('ocr_text', 3, [300, 300, 420, 330], 'OCR symbols: [] ` &',
                         id='ocr-1', source='ocr', confidence=.93,
                         meta={'image': self.image.asset, 'figure_label': True})

    def test_absolute_and_relative_coordinates_match_boxes_at_any_scale(self):
        parsed = Markup(html_block(self.image, children=[self.native, self.ocr]))
        rows = parsed.find(css='transcript-row')
        boxes = parsed.find(css='image-text-box')
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(boxes), 2)
        for row, box, child in zip(rows, boxes, (self.native, self.ocr)):
            self.assertEqual(row['data-page'], '3')
            self.assertEqual(json.loads(row['data-bbox']), child.bbox)
            self.assertEqual(row['data-block-id'], child.id)
            self.assertEqual(row['data-line'], box['data-line'])
            self.assertEqual(row['data-relative-bbox'], box['data-relative-bbox'])
            left, top, right, bottom = json.loads(row['data-relative-bbox'])
            style = dict(re.findall(r'(left|top|width|height):([\d.-]+)%', box['style']))
            for name, expected in zip(('left', 'top', 'width', 'height'),
                                      (left, top, right-left, bottom-top)):
                self.assertAlmostEqual(float(style[name]), expected)
            for width in (1600, 700, 256):
                scale = width / 400
                self.assertAlmostEqual(left / 100 * width, (child.bbox[0]-100) * scale)
                self.assertAlmostEqual(top / 100 * (width/2), (child.bbox[1]-200) * scale)
                self.assertAlmostEqual((right-left) / 100 * width,
                                       (child.bbox[2]-child.bbox[0]) * scale)
        self.assertEqual(json.loads(rows[0]['data-relative-bbox']), [10, 10, 40, 20])
        self.assertIn('hidden', parsed.find(css='image-text-layer')[0])
        self.assertEqual(parsed.find(css='toggle-image-text')[0]['aria-pressed'], 'false')
        self.assertEqual(parsed.find(css='figure-text')[0]['contenteditable'], 'false')
        self.assertTrue(all(a['contenteditable'] == 'true' for a in parsed.find(css='transcript-text')))

    def test_native_line_geometry_is_not_replaced_by_paragraph_bounds(self):
        self.native.text = 'first second'
        self.native.meta['text_lines'] = [
            {'text': 'first', 'bbox': [140, 220, 180, 230]},
            {'text': 'second', 'bbox': [160, 240, 220, 250]},
        ]
        before = copy.deepcopy(self.native)
        rendered = html_block(self.image, children=[self.native])
        rows = Markup(rendered).find(css='transcript-row')
        self.assertEqual([json.loads(row['data-bbox']) for row in rows],
                         [line['bbox'] for line in self.native.meta['text_lines']])
        self.assertEqual([row['data-line'] for row in rows], ['1', '2'])
        self.assertEqual(self.native, before)

    def test_scan_attachment_uses_asset_and_page_without_removing_body(self):
        self.image.meta['scan_reference'] = True
        self.ocr.meta['figure_label'] = False
        wrong_page = copy.deepcopy(self.ocr)
        wrong_page.page = 4
        wrong_asset = copy.deepcopy(self.ocr)
        wrong_asset.meta['image'] = 'assets/other.png'
        native_body = Block('text', 3, [140, 270, 260, 280], 'native body',
                            meta={'image': self.image.asset})
        table = Block('table', 3, self.ocr.bbox, rows=[['x']], source='ocr',
                      meta={'image': self.image.asset})
        caption = copy.deepcopy(self.native)
        caption.meta['caption_for'] = self.image.asset
        result = Result('synthetic.pdf', 4, [3],
                        [self.image, self.native, self.ocr, wrong_page, wrong_asset, native_body, table, caption])
        self.assertEqual(attachments(result, self.image), [self.native, self.ocr])
        self.assertEqual(top_blocks(result).count(self.ocr), 1)
        self.assertNotIn(self.native, top_blocks(result))
        self.assertNotIn(caption, top_blocks(result))
        self.image.asset = ''
        self.assertEqual(attachments(result, self.image), [])

    def test_figure_and_ordinary_ocr_blocks_keep_provenance(self):
        for kind in ('ocr_text', 'text', 'heading'):
            with self.subTest(kind=kind):
                block = copy.deepcopy(self.ocr)
                block.kind = kind
                article = Markup(html_block(block)).find('article')[0]
                self.assertEqual(article['data-page'], '3')
                self.assertEqual(json.loads(article['data-bbox']), block.bbox)
                self.assertEqual(article['data-source'], 'ocr')

    def test_markdown_retains_image_and_readable_coordinate_appendix(self):
        value = markdown(self.image, children=[self.native, self.ocr])
        self.assertIn('![第 3 页图片](<../graph/crop%20%26%20sample.png>)', value)
        self.assertIn('PDF bbox [140.0,220.0,260.0,240.0]', value)
        self.assertIn('原图 bbox (%) [10, 10, 40, 20]', value)
        rows = Markup(value).find('tr')[1:]
        self.assertEqual(len(rows), 2)
        self.assertEqual(json.loads(rows[0]['data-relative-bbox']), [10, 10, 40, 20])
        self.assertIn('native &lt;label&gt;', value)
        self.assertIn('OCR symbols: [] ` &amp;', value)

    def test_text_and_identifiers_cannot_inject_html(self):
        payload = '</span><img src=x onerror="alert(1)"><script>alert(2)</script>'
        self.ocr.text = payload
        self.ocr.id = '" onmouseover="alert(3)'
        self.image.id = self.ocr.id
        self.image.meta['caption'] = payload
        for value in (html_block(self.image, children=[self.ocr]),
                      markdown(self.image, children=[self.ocr])):
            parsed = Markup(value)
            self.assertEqual(parsed.find('script'), [])
            self.assertFalse(any(key.startswith('on') for _, attrs in parsed.tags for key in attrs))
            self.assertNotIn(payload, value)
            self.assertEqual(parsed.find(css='transcript-row')[0]['data-block-id']
                             if parsed.find(css='transcript-row') else parsed.find('tr')[1]['data-block-id'],
                             self.ocr.id)
        # The transparent selection layer is hydrated safely at load time, so
        # new exports store each line's text once, including code punctuation.
        clean = html_block(self.image, children=[self.native])
        self.assertEqual(clean.count('native &lt;label&gt;'), 1)
        self.assertIn('preserveAspectRatio="none"', clean)

    def test_unusable_geometry_retains_transcript_without_bogus_boxes(self):
        for bbox in ([100, 200, 100, 400], [0, 0, float('nan'), 40], [], None):
            with self.subTest(bbox=bbox):
                self.image.bbox = bbox
                value = html_block(self.image, children=[self.ocr])
                parsed = Markup(value)
                self.assertEqual(parsed.find(css='image-text-box'), [])
                self.assertEqual(json.loads(parsed.find(css='transcript-row')[0]['data-bbox']), self.ocr.bbox)
                self.assertIn('disabled', parsed.find(css='locate-text')[0])
                self.assertNotIn('NaN', value)
                self.assertIn('图内位置不可用', value)

    def test_partial_boxes_keep_unclipped_coordinates_but_outside_boxes_do_not_overlay(self):
        self.native.bbox = [80, 190, 140, 220]
        self.ocr.bbox = [600, 450, 700, 500]
        parsed = Markup(html_block(self.image, children=[self.native, self.ocr]))
        rows = parsed.find(css='transcript-row')
        self.assertEqual(json.loads(rows[0]['data-relative-bbox']), [-5, -5, 10, 10])
        self.assertEqual(len(parsed.find(css='image-text-box')), 1)
        self.assertIn('left:-5.00000000%', parsed.find(css='image-text-box')[0]['style'])
        self.assertIn('disabled', parsed.find(css='locate-text')[1])

    def test_export_tree_preserves_paths_editing_and_scan_body(self):
        self.image.meta['scan_reference'] = True
        self.ocr.meta['figure_label'] = False
        result = Result('synthetic.pdf', 3, [3], [self.image, self.native, self.ocr])
        original = copy.deepcopy(result)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / 'assets'
            assets.mkdir()
            (assets / Path(self.image.asset).name).write_bytes(b'synthetic image resource')
            export_all(result, root / 'export', assets, 'synthetic')
            for fmt in ('html', 'md'):
                base = root / 'export' / fmt
                for folder in ('graph', 'summary', 'ocr'):
                    value = (base / folder / f'{folder}.{fmt}').read_text(encoding='utf8')
                    parsed = Markup(value)
                    if fmt == 'html':
                        self.assertEqual(len([a for a in parsed.find('button') if a.get('id') == 'save']), 1)
                        self.assertEqual(parsed.find('main')[0]['contenteditable'], 'true')
                        self.assertIn('cloneNode(true)', value)
                        self.assertIn('.textContent=value', value)
                        self.assertNotIn('innerHTML', value)
                        self.assertNotIn('<script src=', value)
                        for img in parsed.find('img'):
                            self.assertTrue((base / folder / unquote(img['src'])).is_file())
                        if folder == 'summary':
                            body = [a for a in parsed.find('article') if a.get('id') == self.ocr.id]
                            self.assertEqual(len(body), 1)
                        if folder == 'ocr':
                            self.assertEqual(json.loads(parsed.find('article')[0]['data-bbox']), self.ocr.bbox)
                    elif folder == 'ocr':
                        anchor = parsed.find('a')[1]  # page anchor, then source block
                        self.assertEqual(json.loads(anchor['data-bbox']), self.ocr.bbox)
                self.assertEqual((base / 'graph' / Path(self.image.asset).name).read_bytes(),
                                 b'synthetic image resource')
        self.assertEqual(result, original)


if __name__ == '__main__':
    unittest.main()
