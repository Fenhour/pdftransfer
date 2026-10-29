"""Original-page HTML positions use one DOM and PDF point coordinates."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from pdf_extractor.export import _render_document, SCRIPT
from pdf_extractor.model import Block, Result


class Markup(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.tags = []
        self.feed(value)
    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
    def find(self, tag=None, css=None):
        return [attrs for name, attrs in self.tags if (tag is None or name == tag)
                and (css is None or css in attrs.get('class', '').split())]


class PositionedReadingTests(unittest.TestCase):
    def result(self):
        text = Block('text', 1, [50, 60, 250, 110], 'Native red text', id='body',
                     meta={'native_style': True, 'text_lines': [
                         {'text': 'Native red text', 'bbox': [55, 65, 180, 85], 'runs': [
                             {'text': 'Native red text', 'bbox': [55, 65, 180, 85], 'font': 'Arial',
                              'size': 14, 'color': '#ff0000', 'bold': True, 'italic': True}]}]})
        image = Block('image', 1, [100, 200, 300, 350], id='picture', asset='assets/picture.png')
        table = Block('table', 1, [40, 400, 360, 500], id='table', rows=[['A', 'B'], ['C', 'D']])
        return Result('synthetic.pdf', 1, [1], [text, image, table], page_sizes={1: [400, 600]})

    def render(self, result, fmt='html', summary=True, blocks=None):
        return _render_document(result, result.blocks if blocks is None else blocks, fmt,
                                'Synthetic', '../graph/', 'synthetic', summary)

    def test_summary_defaults_to_source_positions_and_classification_remains_reading(self):
        result = self.result()
        parsed = Markup(self.render(result))
        self.assertEqual(parsed.find('body')[0]['data-view'], 'positioned')
        button = next(attrs for attrs in parsed.find('button') if attrs.get('id') == 'toggle-page-view')
        self.assertEqual(button['aria-pressed'], 'true')
        self.assertEqual(len(parsed.find(css='source-position')), 3)
        self.assertEqual(len(parsed.find(css='page-surface')), 1)
        self.assertEqual(json.loads(parsed.find(css='page-canvas')[0]['data-page-size']), [400, 600])
        self.assertEqual(self.render(result).count('Native red text'), 1)
        category = Markup(self.render(result, summary=False, blocks=[result.blocks[0]]))
        self.assertEqual(category.find('body')[0]['data-view'], 'reading')
        self.assertEqual(category.find(css='source-position'), [])
        self.assertFalse(any(attrs.get('id') == 'toggle-page-view' for attrs in category.find('button')))

    def test_image_text_table_and_run_coordinates_scale_from_source_page(self):
        result = self.result()
        parsed = Markup(self.render(result))
        positioned = {attrs['id']: attrs for attrs in parsed.find(css='source-position')}
        for block in result.blocks:
            attrs = positioned[block.id]
            self.assertEqual(json.loads(attrs['data-bbox']), block.bbox)
            values = {name: float(number) for name, number in re.findall(r'--box-(left|top|width|height):([\d.-]+)pt', attrs['style'])}
            for name, expected in zip(('left', 'top', 'width', 'height'),
                                      (block.bbox[0], block.bbox[1], block.bbox[2]-block.bbox[0], block.bbox[3]-block.bbox[1])):
                self.assertEqual(values[name], expected)
            for display_width in (1600, 700, 256):
                scale = display_width / 400
                self.assertAlmostEqual(values['left'] * scale, block.bbox[0] / 400 * display_width)
                self.assertAlmostEqual(values['top'] * scale, block.bbox[1] / 600 * (display_width * 600 / 400))
        run = parsed.find(css='native-run')[0]
        values = {name: float(number) for name, number in re.findall(r'--run-(left|top|width|height):([\d.-]+)pt', run['style'])}
        self.assertEqual(values, {'left': 5, 'top': 5, 'width': 125, 'height': 20})
        self.assertIn('font-size:14.00pt', run['style'])
        self.assertIn('font-style:italic', run['style'])
        self.assertIn('color:#ff0000', run['style'])

    def test_caption_uses_source_coordinate_and_single_stored_text(self):
        result = self.result()
        picture = result.blocks[1]
        picture.meta['caption'] = 'One caption'
        caption = Block('text', 1, [110, 360, 280, 380], 'One caption', id='caption',
                        meta={'caption_for': picture.asset})
        result.blocks.append(caption)
        value = self.render(result, blocks=result.blocks[:-1])
        self.assertEqual(value.count('One caption'), 1)
        attrs = Markup(value).find(css='source-caption')[0]
        self.assertEqual(json.loads(attrs['data-bbox']), caption.bbox)
        self.assertIn('--caption-left:10.00000000pt', attrs['style'])
        self.assertIn('--caption-top:160.00000000pt', attrs['style'])

    def test_md_retains_sizes_and_all_block_coordinates_without_claiming_fixed_layout(self):
        result = self.result()
        value = self.render(result, fmt='md')
        parsed = Markup(value)
        self.assertIn('阅读器不保证原页绝对排版', value)
        anchors = {attrs['id']: attrs for attrs in parsed.find('a')}
        self.assertEqual(anchors['page-1']['data-coordinate-unit'], 'pt')
        self.assertEqual(json.loads(anchors['page-1']['data-page-size']), [400, 600])
        for block in result.blocks:
            self.assertEqual(json.loads(anchors[block.id]['data-bbox']), block.bbox)
            self.assertEqual(json.loads(anchors[block.id]['data-page-size']), [400, 600])

    def test_missing_geometry_and_scanned_pages_keep_reading_view(self):
        result = self.result()
        result.blocks[0].source = 'ocr'
        value = self.render(result)
        self.assertEqual(Markup(value).find('body')[0]['data-view'], 'reading')
        self.assertEqual(Markup(value).find(css='source-position'), [])
        result = self.result()
        result.page_sizes = {1: [float('nan'), 600]}
        value = self.render(result)
        self.assertNotIn('NaN', value)
        self.assertEqual(Markup(value).find('body')[0]['data-view'], 'reading')

    def test_toggle_reuses_dom_and_resizes_at_source_scale(self):
        node = shutil.which('node')
        if not node:
            bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
            node = str(bundled) if bundled.is_file() else None
        if not node:
            self.skipTest('Node runtime unavailable for UI behavior check')
        prefix = SCRIPT.split('function setLayer', 1)[0]
        setup = """
const vm=require('node:vm');
let click, resize; const values={};
const button={setAttribute(k,v){values[k]=v;},addEventListener(k,f){click=f;},textContent:''};
const canvas={clientWidth:600,dataset:{pageWidth:'400'},style:{setProperty(k,v){values[k]=v;}}};
const body={dataset:{view:'positioned'}};
const document={body,querySelector(){return {};},getElementById(){return button;},querySelectorAll(){return [canvas];}};
class ResizeObserver{constructor(callback){resize=callback;}observe(){}}
const context={document,ResizeObserver};vm.createContext(context);
"""
        checks = """
const initial=values['--page-scale'];click();const reading={...values,view:body.dataset.view,label:button.textContent};
canvas.clientWidth=300;resize();click();
process.stdout.write(JSON.stringify({initial,reading,final:{...values,view:body.dataset.view,label:button.textContent}}));
"""
        command = setup + '\nvm.runInContext(' + json.dumps(prefix) + ',context);\n' + checks
        run = subprocess.run([node, '-e', command], capture_output=True, text=True, encoding='utf-8', check=True)
        state = json.loads(run.stdout)
        self.assertAlmostEqual(float(state['initial']), 1.125)
        self.assertEqual(state['reading']['view'], 'reading')
        self.assertEqual(state['reading']['aria-pressed'], 'false')
        self.assertEqual(state['final']['view'], 'positioned')
        self.assertEqual(state['final']['aria-pressed'], 'true')
        self.assertAlmostEqual(float(state['final']['--page-scale']), .5625)
        self.assertNotIn('innerHTML', prefix)


if __name__ == '__main__':
    unittest.main()
