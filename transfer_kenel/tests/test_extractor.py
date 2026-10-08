from pathlib import Path
import shutil
import threading
import unittest
import uuid
from unittest.mock import patch

import pymupdf as fitz

from pdf_extractor.engine import Cancelled, run
from pdf_extractor.layout import classify, reading_order
from pdf_extractor.model import Block, Options, parse_pages


class ExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("tmp") / ("tests-" + uuid.uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.source = cls.root / "sample.pdf"
        doc = fitz.open()
        for n in range(2):
            p = doc.new_page(width=600, height=800)
            p.insert_text((50, 30), "REPEATED REPORT HEADER", fontsize=9)
            p.insert_text((50, 95), f"{n+1} Introduction", fontsize=22)
            p.insert_text((50, 140), "Body paragraph before the table.", fontsize=11)
            p.insert_text((50, 160), "More body content to establish the normal font size.", fontsize=11)
            for x in (50, 220, 390):
                p.draw_line((x, 210), (x, 310))
            for y in (210, 260, 310):
                p.draw_line((50, y), (390, y))
            for point, text in [((60, 240), "Name"), ((230, 240), "Value"), ((60, 290), "UniqueCell"), ((230, 290), "42")]:
                p.insert_text(point, text, fontsize=11)
            p.insert_text((50, 350), "Body paragraph after the table.", fontsize=11)
            p.insert_text((295, 775), str(n + 1), fontsize=10)
        doc.set_toc([[1, "Introduction", 1], [1, "Introduction", 2]])
        doc.save(cls.source)
        doc.close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root)

    def test_page_selection_validation(self):
        self.assertEqual(parse_pages("3,1-2,2", 3), [0, 1, 2])
        for value in ("0", "2-1", "4", "1,", "1-a", "-1"):
            with self.assertRaises(ValueError):
                parse_pages(value, 3)

    def test_native_classification_order_and_export(self):
        destination, result = run(self.source, self.root / "output", Options(ocr="off"))
        self.assertEqual(len([b for b in result.blocks if b.kind == "page_number"]), 2)
        self.assertEqual(len([b for b in result.blocks if b.kind == "header_footer"]), 2)
        tables = [b for b in result.blocks if b.kind == "table"]
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0].rows, [["Name", "Value"], ["UniqueCell", "42"]])
        self.assertFalse(any("UniqueCell" in b.text for b in result.blocks if b.kind == "text"))
        first = [b for b in result.blocks if b.page == 1]
        before = next(i for i, b in enumerate(first) if "before the table" in b.text)
        after = next(i for i, b in enumerate(first) if "after the table" in b.text)
        self.assertLess(before, first.index(tables[0]))
        self.assertLess(first.index(tables[0]), after)
        self.assertTrue(all(b.level == 1 for b in result.blocks if b.kind == "heading"))
        output = destination.parent.parent
        for fmt in ('md', 'html'):
            base = output / f'output_{fmt}' / destination.name
            for name in (f"summary/summary.{fmt}", f"text/text.{fmt}", f"heading/heading.{fmt}", f"page_number/page_number.{fmt}", f"table/table.{fmt}"):
                self.assertTrue((base / name).is_file(), name)
            self.assertFalse((base / 'graph').exists())
            self.assertEqual((output / f'output_{fmt}' / 'source' / f'{destination.name}.pdf').read_bytes(), self.source.read_bytes())
            self.assertFalse(any(p.suffix in ('.txt', '.json', '.csv') for p in base.rglob('*')))

    def test_columns_and_spanning_title(self):
        title = Block("heading", 1, [40, 20, 560, 50], "Title")
        left1 = Block("text", 1, [40, 100, 270, 140], "L1")
        left2 = Block("text", 1, [40, 145, 270, 180], "L2")
        right1 = Block("text", 1, [320, 100, 560, 140], "R1")
        right2 = Block("text", 1, [320, 145, 560, 180], "R2")
        self.assertEqual([b.text for b in reading_order([right2, left1, title, left2, right1])], ["Title", "L1", "L2", "R1", "R2"])

    def test_appendix_outline_and_side_margin(self):
        heading = Block("text", 1, [50, 100, 300, 125], "B.4.2\nMap Traversal", meta={"font_size": 10})
        side = Block("text", 1, [10, 200, 30, 450], "Document identifier", meta={"font_size": 28, "vertical": True})
        body = Block("text", 1, [50, 150, 500, 600], "Body text " * 20, meta={"font_size": 10})
        blocks = [heading, side, body]
        classify(blocks, {1: (600, 800)}, [{"page": 1, "title": "Map Traversal", "level": 3}])
        self.assertEqual((heading.kind, heading.level), ("heading", 3))
        self.assertEqual(side.kind, "header_footer")
        self.assertEqual(reading_order(blocks)[0], heading)

    def test_selected_pages_and_no_overwrite(self):
        first, result = run(self.source, self.root / "selected", Options(pages="2", ocr="off"))
        second, _ = run(self.source, self.root / "selected", Options(pages="2", ocr="off"))
        self.assertNotEqual(first, second)
        self.assertEqual(result.selected_pages, [2])
        self.assertTrue(all(b.page == 2 for b in result.blocks))

    def test_password_and_cancellation(self):
        protected = self.root / "locked.pdf"
        with fitz.open(self.source) as doc:
            doc.save(protected, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
        with self.assertRaisesRegex(ValueError, "密码"):
            run(protected, self.root / "locked", Options(ocr="off"))
        _, result = run(protected, self.root / "locked", Options(password="secret", ocr="off"))
        self.assertEqual(result.total_pages, 2)
        event = threading.Event()
        event.set()
        with self.assertRaises(Cancelled):
            run(self.source, self.root / "cancel", cancel=event)
        self.assertEqual(list((self.root / "cancel").iterdir()), [])

    def test_html_escaping(self):
        from pdf_extractor.export import html_block
        rendered = html_block(Block("text", 1, [0, 0, 1, 1], '<script>alert("x")</script>'))
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_punctuation_and_code_layout(self):
        from pdf_extractor.engine import _native_text
        from pdf_extractor.export import markdown, md_text
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        # Mixed-font pieces on the same baseline must remain a single sentence.
        page.insert_text((50, 100), "Read [1, 2]", fontsize=11)
        page.insert_text((115, 100), "and {values} < 3.", fontsize=11, fontname="hebo")
        page.insert_text((50, 150), 'const xs = [];', fontsize=10, fontname="cour")
        page.insert_text((70, 170), 'xs.push(data[0]);', fontsize=10, fontname="cour")
        page.insert_text((50, 190), 'return xs;', fontsize=10, fontname="cour")
        blocks = _native_text(page, [])
        code = [b for b in blocks if b.meta.get('code')]
        self.assertEqual(len(code), 1)
        self.assertIn('const xs = [];', code[0].text)
        self.assertIn('\n    xs.push(data[0]);', code[0].text)
        self.assertIn('[]', markdown(code[0]))
        self.assertIn('[1, 2]', md_text('Read [1, 2]'))
        self.assertEqual(sum(b.text.count('[') for b in blocks), 3)
        self.assertIn('```text', markdown(code[0]))
        doc.close()

    def test_composite_figure_and_relative_links(self):
        from PIL import Image
        from pdf_extractor.export import top_blocks
        from html.parser import HTMLParser
        from urllib.parse import unquote
        image = self.root / 'tile.png'
        Image.new('RGB', (100, 100), '#237c88').save(image)
        source = self.root / 'figure #1.pdf'
        with fitz.open() as doc:
            p = doc.new_page(width=600, height=800)
            p.insert_text((50, 80), 'Body paragraph before the composite figure.', fontsize=11)
            p.draw_rect(fitz.Rect(50, 120, 500, 320))
            p.insert_image(fitz.Rect(70, 150, 170, 250), filename=str(image))
            p.insert_image(fitz.Rect(350, 150, 450, 250), filename=str(image))
            p.insert_text((190, 200), 'Figure label [A]', fontsize=11)
            p.insert_text((50, 345), 'Figure 1: A complete composite diagram.', fontsize=11)
            p.insert_text((50, 390), 'Body paragraph after the figure.', fontsize=11)
            doc.save(source)
        destination, result = run(source, self.root / 'figures', Options(ocr='off'))
        images = [b for b in result.blocks if b.kind == 'image']
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].meta['caption'], 'Figure 1: A complete composite diagram.')
        self.assertFalse(any(b.text == 'Figure label [A]' for b in top_blocks(result)))
        self.assertTrue(any(b.text == 'Figure label [A]' for b in result.blocks))
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []
                self.ids = set()
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if 'id' in attrs:
                    self.ids.add(attrs['id'])
                if tag in ('a', 'img'):
                    self.links.append(attrs.get('href') or attrs.get('src'))
        for file in destination.rglob('*.html'):
            parser = Links()
            parser.feed(file.read_text(encoding='utf-8'))
            for link in parser.links:
                if not link:
                    continue
                if link.startswith('#'):
                    self.assertIn(link[1:], parser.ids)
                else:
                    self.assertTrue((file.parent / unquote(link)).is_file(), link)

    def test_scan_ocr_and_grid(self):
        # A real raster-only PDF verifies model loading, text recognition, and grid recovery.
        from PIL import Image, ImageDraw, ImageFont
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        font_path = Path("C:/Windows/Fonts/arial.ttf")
        if not font_path.exists():
            self.skipTest("OCR fixture font is unavailable")
        font = ImageFont.truetype(str(font_path), 34)
        draw.text((100, 100), "SCANNED DOCUMENT", font=font, fill="black")
        draw.text((100, 190), "The extraction test is working.", font=font, fill="black")
        chinese_font = Path("C:/Windows/Fonts/msyh.ttc")
        if chinese_font.exists():
            draw.text((100, 270), "中文内容提取测试", font=ImageFont.truetype(str(chinese_font), 34), fill="black")
        for x in (100, 550, 1000):
            draw.line((x, 350, x, 650), fill="black", width=4)
        for y in (350, 500, 650):
            draw.line((100, y, 1000, y), fill="black", width=4)
        for xy, text in [((130, 390), "Product"), ((580, 390), "Count"), ((130, 540), "Apples"), ((580, 540), "42")]:
            draw.text(xy, text, font=font, fill="black")
        draw.text((510, 1510), "Page 1", font=font, fill="black")
        image_path = self.root / "scan.png"
        image.save(image_path)
        source = self.root / "scan.pdf"
        with fitz.open() as doc:
            page = doc.new_page(width=600, height=800)
            page.insert_image(page.rect, filename=str(image_path))
            doc.save(source)
        destination, result = run(source, self.root / "scan-output", Options(ocr="auto"))
        ocr_text = " ".join(b.text for b in result.blocks if b.source == "ocr")
        self.assertIn("SCANNED", ocr_text.upper())
        if chinese_font.exists():
            self.assertIn("中文", ocr_text)
        grids = [b for b in result.blocks if b.kind == "table"]
        self.assertEqual(len(grids), 1)
        self.assertTrue(any("Apples" in str(row) for row in grids[0].rows))
        self.assertFalse(any("Apples" in b.text for b in result.blocks if b.kind == "ocr_text"))
        self.assertEqual([b.text for b in result.blocks if b.kind == "page_number"], ["Page 1"])
        self.assertTrue(all((destination / 'graph' / Path(b.asset).name).exists() for b in result.blocks if b.asset))
        # Missing/broken OCR must produce an explicit warning and still export images.
        with patch("pdf_extractor.ocr.LocalOCR.recognize", side_effect=RuntimeError("test unavailable")):
            _, degraded = run(source, self.root / "degraded")
        self.assertTrue(any("OCR 不可用" in w for w in degraded.warnings))


if __name__ == "__main__":
    unittest.main()
