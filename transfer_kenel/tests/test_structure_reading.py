"""Routing/export contracts with synthetic V3 results; no accuracy claim."""
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import pymupdf as fitz

from pdf_extractor.engine import run
from pdf_extractor.model import Options
from pdf_extractor.structure_reading import parse_table


class StructureReadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def source(self, name="source.pdf", *, scanned=False, mixed=False):
        path = self.root / name
        with fitz.open() as doc:
            if not scanned or mixed:
                page = doc.new_page(width=400, height=600)
                page.insert_text((35, 100), "Native editable paragraph with reliable text extracted directly.", fontsize=10)
            if scanned:
                buffer = io.BytesIO()
                Image.new("RGB", (400, 600), "white").save(buffer, format="PNG")
                page = doc.new_page(width=400, height=600)
                page.insert_image(page.rect, stream=buffer.getvalue())
            doc.save(path)
        return path

    def test_selected_formats_create_only_requested_tree_and_keep_unique_names(self):
        source = self.source()
        original = source.read_bytes()
        for fmt in ("html", "md"):
            with self.subTest(fmt=fmt):
                output = self.root / fmt
                options = Options(ocr="off", output_format=fmt)
                first, _ = run(source, output, options)
                second, _ = run(source, output, options)
                self.assertNotEqual(first, second)
                self.assertEqual({p.name for p in output.iterdir()}, {f"output_{fmt}"})
                self.assertTrue((first / "summary" / f"summary.{fmt}").is_file())
                self.assertEqual((first.parent / "source" / f"{first.name}.pdf").read_bytes(), original)
                self.assertFalse(list(first.rglob("*.md" if fmt == "html" else "*.html")))
        self.assertEqual(source.read_bytes(), original)

    def test_native_v3_mode_never_calls_models(self):
        recognize = Mock(side_effect=AssertionError("native pages must not call OCR"))
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}), \
                patch("pdf_extractor.engine.LocalOCR.recognize", side_effect=AssertionError("no legacy OCR")) as legacy:
            _, result = run(self.source(), self.root / "native",
                            Options(output_format="html", recognition_backend="v3"))
            short = self.root / "short.pdf"
            with fitz.open() as doc:
                doc.new_page().insert_text((30, 50), "Short title")
                doc.save(short)
            _, short_result = run(short, self.root / "native-short",
                                  Options(output_format="html", recognition_backend="v3"))
        legacy.assert_not_called()
        recognize.assert_not_called()
        self.assertIn("Native editable", " ".join(b.text for b in result.blocks))
        self.assertTrue(all(b.source == "native" for b in result.blocks))
        self.assertIn("Short title", " ".join(b.text for b in short_result.blocks))

    def test_mixed_document_only_scans_raster_page_and_retains_table_spans_references(self):
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "doc_title", "block_bbox": [20, 20, 360, 50], "block_content": "Recovered title"},
            {"block_label": "table", "block_bbox": [20, 100, 380, 220],
             "block_content": '<table><tr><th colspan="2">Header</th></tr><tr><td>A</td><td>&lt;script&gt;</td></tr></table>'},
            {"block_label": "image", "block_bbox": [20, 260, 200, 400], "block_content": ""}],
            "overall_ocr_res": {"rec_texts": ["Recovered title", "Header", "diagram label", "missed by layout"],
                                "rec_scores": [.99, .98, .97, .96],
                                "rec_boxes": [[20, 20, 360, 50], [30, 110, 200, 140],
                                              [40, 280, 160, 300], [20, 450, 200, 470]]}}
        recognize = Mock(return_value=raw)
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}):
            destination, result = run(self.source(scanned=True, mixed=True), self.root / "mixed",
                                      Options(dpi=72, output_format="html", recognition_backend="v3"))
        recognize.assert_called_once()
        tables = [b for b in result.blocks if b.kind == "table"]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].rows, [["Header", None], ["A", "<script>"]])
        self.assertEqual(tables[0].meta["grid_cells"][0]["colspan"], 2)
        self.assertTrue((destination / "graph" / Path(tables[0].meta["preview_asset"]).name).is_file())
        scan_blocks = [b for b in result.blocks if b.page == 2]
        self.assertEqual(scan_blocks[0].kind, "heading")
        self.assertTrue(scan_blocks[0].meta["style_approximate"])
        self.assertEqual([b.text for b in scan_blocks if b.meta.get("figure_label")], ["diagram label"])
        self.assertEqual([b.text for b in scan_blocks if b.meta.get("needs_review")], ["missed by layout"])
        summary = (destination / "summary/summary.html").read_text(encoding="utf-8")
        self.assertIn('colspan="2"', summary)
        self.assertIn("&lt;script&gt;", summary)
        self.assertIn("原页参考", summary)
        self.assertTrue((destination / "graph/page_0002_reference.png").is_file())
        self.assertFalse((destination.parent.parent / "output_md").exists())

    def test_malformed_table_is_retained_as_image_and_reported(self):
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "table", "block_bbox": [20, 100, 380, 220],
             "block_content": "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>"}]}
        recognize = Mock(return_value=raw)
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}):
            destination, result = run(self.source(scanned=True), self.root / "fallback",
                                      Options(dpi=72, output_format="md", recognition_backend="v3"))
        self.assertFalse(any(b.kind == "table" for b in result.blocks))
        self.assertTrue(any(b.meta.get("table_recovery") == "unresolved" for b in result.blocks))
        self.assertTrue(any("未可靠恢复" in message for message in result.warnings))
        self.assertTrue(list((destination / "graph").glob("*.png")))

    def test_model_failure_is_explicit_and_never_silently_changes_backend(self):
        recognize = Mock(side_effect=RuntimeError("local V3 service unavailable"))
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}):
            with self.assertRaisesRegex(RuntimeError, "V3 service unavailable"):
                run(self.source(scanned=True), self.root / "failed",
                    Options(dpi=72, output_format="html", recognition_backend="v3"))
        self.assertEqual(list((self.root / "failed").iterdir()), [])

    def test_native_page_with_large_picture_keeps_picture_without_automatic_ocr(self):
        source = self.root / "same-page.pdf"
        buffer = io.BytesIO()
        Image.new("RGB", (360, 300), "white").save(buffer, format="PNG")
        native_text = "Reliable native text keeps symbols [A], {B}, and spaces."
        with fitz.open() as doc:
            page = doc.new_page(width=400, height=600)
            page.insert_text((20, 60), native_text, fontsize=10)
            page.insert_image(fitz.Rect(20, 200, 380, 500), stream=buffer.getvalue())
            doc.save(source)
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "text", "block_bbox": [15, 40, 385, 70],
             "block_content": "Unreliable OCR text without original symbols"},
            {"block_label": "text", "block_bbox": [30, 250, 350, 290],
             "block_content": "Scanned paragraph inside the raster."}],
            "overall_ocr_res": {"rec_texts": ["Unreliable OCR text without original symbols", "Scanned paragraph inside the raster."],
                                "rec_boxes": [[20, 48, 300, 63], [30, 250, 350, 270]], "rec_scores": [.99, .99]}}
        recognize = Mock(return_value=raw)
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}):
            _, result = run(source, self.root / "same-page-output",
                            Options(dpi=72, output_format="html", recognition_backend="v3"))
        recognize.assert_not_called()
        self.assertTrue(any(b.text == native_text and b.source == "native" for b in result.blocks))
        self.assertFalse(any(b.source == "ocr" for b in result.blocks))
        self.assertTrue(any(b.kind == "image" and b.bbox == [20, 200, 380, 500] for b in result.blocks))
        self.assertFalse(any("Unreliable OCR" in b.text for b in result.blocks))

    def test_v3_all_on_native_page_uses_v3_for_small_image_and_preserves_native_text(self):
        source = self.root / "native-small-image.pdf"
        buffer = io.BytesIO()
        Image.new("RGB", (60, 60), "white").save(buffer, format="PNG")
        native_text = "Native source with exact symbols [A] and {B} stays unchanged."
        with fitz.open() as doc:
            page = doc.new_page(width=400, height=600)
            page.insert_text((20, 60), native_text, fontsize=10)
            page.insert_image(fitz.Rect(20, 200, 80, 260), stream=buffer.getvalue())
            doc.save(source)
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "text", "block_bbox": [15, 40, 385, 70],
             "block_content": "OCR differs from exact native characters"},
            {"block_label": "image", "block_bbox": [20, 200, 80, 260], "block_content": ""}],
            "overall_ocr_res": {"rec_texts": ["OCR differs from exact native characters", "small image label"],
                                "rec_boxes": [[20, 48, 300, 63], [25, 210, 75, 225]],
                                "rec_scores": [.99, .99]}}
        recognize = Mock(return_value=raw)
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=recognize)}), \
                patch("pdf_extractor.engine.LocalOCR.recognize", side_effect=AssertionError("no legacy OCR")) as legacy:
            _, result = run(source, self.root / "all-output",
                            Options(ocr="all", dpi=72, output_format="html", recognition_backend="v3"))
        recognize.assert_called_once()
        legacy.assert_not_called()
        self.assertTrue(any(b.text == native_text and b.source == "native" for b in result.blocks))
        self.assertEqual([b.text for b in result.blocks if b.meta.get("figure_label")], ["small image label"])

    def test_mixed_native_and_scanned_lines_in_one_semantic_block_never_drop_scanned_text(self):
        source = self.root / "one-block-mixed.pdf"
        buffer = io.BytesIO()
        Image.new("RGB", (360, 300), "white").save(buffer, format="PNG")
        native_text = "Long native paragraph with precise symbols [A], {B} and exact punctuation."
        scanned_text = "SCAN TAIL"
        with fitz.open() as doc:
            page = doc.new_page(width=400, height=600)
            page.insert_text((20, 60), native_text, fontsize=9)
            page.insert_image(fitz.Rect(20, 100, 380, 400), stream=buffer.getvalue())
            doc.save(source)
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "text", "block_bbox": [15, 40, 385, 145],
             "block_content": native_text + "\n" + scanned_text}],
            "overall_ocr_res": {"rec_texts": ["Imprecise OCR of the long native paragraph", scanned_text],
                                "rec_boxes": [[20, 48, 335, 63], [20, 110, 160, 130]], "rec_scores": [.98, .99]}}
        # The old 70% block-length rule would choose only native_text here.
        self.assertGreater(len(native_text), len(raw["parsing_res_list"][0]["block_content"]) * .7)
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=Mock(return_value=raw))}):
            _, result = run(source, self.root / "one-block-output",
                            Options(ocr="all", dpi=72, output_format="html", recognition_backend="v3"))
        text_blocks = [b for b in result.blocks if b.kind == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0].text, native_text + "\n" + scanned_text)
        self.assertEqual(text_blocks[0].meta["text_sources"], ["native", "ocr"])
        self.assertNotIn("Imprecise OCR", text_blocks[0].text)

    def test_missing_ocr_line_geometry_keeps_both_native_and_model_evidence(self):
        native_text = "Native editable paragraph with reliable text extracted directly."
        model_text = "Model paragraph may additionally contain scanned words."
        raw = {"width": 400, "height": 600, "parsing_res_list": [
            {"block_label": "text", "block_bbox": [20, 80, 390, 160], "block_content": model_text}]}
        with patch.dict(sys.modules, {"pdf_extractor.structure": SimpleNamespace(recognize_page=Mock(return_value=raw))}):
            _, result = run(self.source(), self.root / "missing-lines",
                            Options(ocr="all", dpi=72, output_format="html", recognition_backend="v3"))
        content = "\n".join(b.text for b in result.blocks)
        self.assertIn(native_text, content)
        self.assertIn(model_text, content)
        self.assertTrue(any("缺少 OCR 行坐标" in warning for warning in result.warnings))

    def test_table_parser_rejects_overlapping_and_unbounded_spans(self):
        for markup in ("<table><tr><td rowspan='999999'>x</td></tr></table>",
                       "<table><tr><td rowspan='2'>A</td><td>B</td></tr><tr><td colspan='2'>C</td></tr></table>"):
            with self.subTest(markup=markup), self.assertRaises(ValueError):
                parse_table(markup)


if __name__ == "__main__":
    unittest.main()
