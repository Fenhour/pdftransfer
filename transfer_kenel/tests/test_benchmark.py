"""Benchmark tests use generated PDFs/strings, never private source documents."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf as fitz

from pdf_extractor.benchmark import (bbox_iou, edit_distance, match_boxes,
                                     normalize_text, score_ocr_region, score_tables, text_score)
from pdf_extractor.model import Block, Result
from scripts.evaluate_benchmark import (aggregate, markdown, run_benchmark, sha256,
                                        source_path, validate_manifest)


class BenchmarkMetricTests(unittest.TestCase):
    def test_text_normalization_preserves_case_numbers_and_punctuation(self):
        self.assertEqual(normalize_text('Ａ \n B １'), 'AB1')
        self.assertNotEqual(normalize_text('x = 1'), normalize_text('X = 2'))
        self.assertNotEqual(normalize_text('x=1;'), normalize_text('x=1'))

    def test_edit_distance_and_cer_can_exceed_one(self):
        self.assertEqual(edit_distance('kitten', 'sitting'), 3)
        self.assertEqual(edit_distance('', 'abc'), 3)
        self.assertEqual(text_score('a', 'abc')['cer'], 2)
        self.assertIsNone(text_score('', 'abc')['cer'])
        self.assertFalse(text_score('', 'abc')['exact'])

    def test_blank_run_metric_does_not_remove_decimal_points(self):
        self.assertEqual(text_score('26 .......... £50', '26 ... £50', True)['cer'], 0)
        self.assertGreater(text_score('26 .......... £50', '26 ... £50')['cer'], 0)
        self.assertGreater(text_score('9.30', '930', True)['cer'], 0)
        self.assertEqual(normalize_text('a……b', True), 'a\ufffcb')

    def test_bbox_iou(self):
        self.assertEqual(bbox_iou([0, 0, 2, 2], [0, 0, 2, 2]), 1)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)
        self.assertEqual(bbox_iou([0, 0, 0, 0], [0, 0, 0, 0]), 0)

    def test_matching_is_one_to_one_and_not_greedy(self):
        refs = [[0, 0, 20, 10], [0, 0, 10, 10]]
        predictions = [[0, 0, 10, 10], [10, 0, 20, 10]]
        matches = match_boxes(refs, predictions)
        self.assertEqual([(i, j) for i, j, _ in matches], [(0, 1), (1, 0)])
        self.assertEqual(len(match_boxes([refs[0], refs[0]], [refs[0]])), 1)

    def test_tables_count_false_positives_and_misses(self):
        refs = [{'id': 'r', 'bbox': [0, 0, 20, 20], 'shape': [2, 2]}]
        pred = Block('table', 1, [50, 50, 80, 80], rows=[['x']], id='false')
        score = score_tables(refs, [pred])
        self.assertEqual((score['true_positive'], score['false_positive'], score['false_negative']), (0, 1, 1))
        self.assertFalse(score['details'][0]['shape_exact'])
        self.assertEqual(score_tables([], [pred])['false_positive'], 1)

    def test_empty_slots_are_not_merged_slots_or_missing_slots(self):
        ref = {'id': 't', 'bbox': [0, 0, 20, 20], 'shape': [1, 3], 'rows': [['', None, 'x']]}
        pred = Block('table', 1, ref['bbox'], rows=[[None, '', 'x']])
        score = score_tables([ref], [pred])['details'][0]
        self.assertEqual(score['cells_correct'], 1)
        self.assertEqual(score['blank_cells_correct'], 0)
        pred.rows = [['']]
        score = score_tables([ref], [pred])['details'][0]
        self.assertEqual(score['cells_correct'], 1)
        self.assertEqual(score['cells_total'], 3)

    def test_merge_metadata_must_be_present(self):
        ref = {'id': 't', 'bbox': [0, 0, 20, 20], 'shape': [1, 2], 'spans': [[0, 0, 1, 2]]}
        pred = Block('table', 1, ref['bbox'], rows=[['title', None]])
        self.assertFalse(score_tables([ref], [pred])['details'][0]['spans_exact'])
        pred.meta['grid_cells'] = [{'row': 0, 'col': 0, 'rowspan': 1, 'colspan': 2}]
        self.assertTrue(score_tables([ref], [pred])['details'][0]['spans_exact'])

    def test_ocr_filters_native_watermark_but_retains_existing_order(self):
        region = {'id': 'body', 'bbox': [0, 0, 30, 30], 'text': 'AB'}
        blocks = [Block('ocr_text', 1, [1, 1, 10, 10], 'B', id='b', source='ocr'),
                  Block('text', 1, [1, 1, 10, 10], 'watermark'),
                  Block('ocr_text', 1, [2, 12, 10, 20], 'A', id='a', source='ocr'),
                  Block('ocr_text', 1, [40, 40, 50, 50], 'outside', source='ocr')]
        score = score_ocr_region(region, blocks)
        self.assertEqual(score['block_ids'], ['b', 'a'])
        self.assertGreater(score['strict']['cer'], 0)

    def test_ocr_table_content_and_unscored_strict_reference(self):
        region = {'id': 'body', 'bbox': [0, 0, 30, 30], 'text': 'A ...', 'strict': False}
        table = Block('table', 1, [0, 0, 20, 20], rows=[['A', '......']], source='ocr')
        score = score_ocr_region(region, [table])
        self.assertIsNone(score['strict'])
        self.assertEqual(score['blank_normalized']['cer'], 0)

    def test_aggregate_uses_micro_average_and_excludes_errors(self):
        regions = [{'strict': text_score('a', 'b'), 'blank_normalized': text_score('a', 'b')},
                   {'strict': text_score('abcdefghi', 'abcdefghi'),
                    'blank_normalized': text_score('abcdefghi', 'abcdefghi')}]
        cases = [{'state': 'complete', 'ocr_regions': regions},
                 {'state': 'error', 'ocr_regions': regions}]
        summary = aggregate(cases)
        self.assertEqual(summary['cases_error'], 1)
        self.assertEqual(summary['ocr']['strict']['cer'], .1)
        self.assertEqual(summary['ocr']['strict']['regions'], 2)


class BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pdf = self.root / 'synthetic.pdf'
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 30), 'No tables on this synthetic page.')
        doc.save(self.pdf)
        doc.close()
        self.data = {'schema_version': 1, 'cases': [dict(
            id='negative', file='synthetic.pdf', sha256=sha256(self.pdf), page=1,
            page_size=[200, 200], status='draft', table_coverage='page', tables=[])]}

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_validates_physical_page_and_full_table_coverage(self):
        validate_manifest(self.data, self.root)
        for key, value in [('page', 0), ('page', True), ('sha256', 'bad'),
                           ('table_coverage', 'region'), ('page_size', [float('nan'), 200])]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.data)
                data['cases'][0][key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(data, self.root)

    def test_manifest_rejects_duplicates_and_reviewed_without_reviewer(self):
        data = copy.deepcopy(self.data)
        data['cases'].append(copy.deepcopy(data['cases'][0]))
        with self.assertRaises(ValueError):
            validate_manifest(data, self.root)
        data['cases'][1]['id'] = 'new-id'
        with self.assertRaises(ValueError):
            validate_manifest(data, self.root)
        self.data['cases'][0]['status'] = 'reviewed'
        with self.assertRaises(ValueError):
            validate_manifest(self.data, self.root)

    def test_manifest_rejects_invalid_cells_spans_and_bbox(self):
        case = self.data['cases'][0]
        case['tables'] = [{'id': 'table', 'bbox': [0, 0, 50, 50], 'shape': [1, 2],
                           'rows': [['x', None]], 'spans': [[0, 0, 1, 2]]}]
        validate_manifest(self.data, self.root)
        for key, value in [('rows', [['x']]), ('spans', [[0, 1, 1, 2]]),
                           ('spans', [[0, 0, 1, 2], [0, 0, 1, 2]]),
                           ('bbox', [0, 0, 500, 50]), ('rows', [['x', '']])]:
            with self.subTest(key=key):
                data = copy.deepcopy(self.data)
                data['cases'][0]['tables'][0][key] = value
                with self.assertRaises(ValueError):
                    validate_manifest(data, self.root)

    def test_source_path_rejects_escaping_and_absolute_path(self):
        with self.assertRaises(ValueError):
            source_path(self.root, '../outside.pdf')
        with self.assertRaises(ValueError):
            source_path(self.root, str(self.pdf))

    def test_reviewed_runner_skips_drafts_and_reports_reviewed_status(self):
        reviewed = self.data['cases'][0]
        reviewed.update(status='reviewed', reviewed_by='Synthetic test reviewer', reviewed_at='2026-10-09')
        draft = copy.deepcopy(reviewed)
        draft.update(id='draft', file='other.pdf', status='draft')
        self.data['cases'].append(draft)
        validate_manifest(self.data, self.root)
        report = run_benchmark(self.data, self.root, self.root / 'reviewed')
        self.assertEqual(report['reference_status'], 'reviewed')
        self.assertEqual(report['draft_cases'], 0)
        self.assertEqual(report['skipped'], ['draft'])

    def test_manifest_rejects_ambiguous_strict_flag_and_undeclared_null(self):
        data = copy.deepcopy(self.data)
        data['cases'][0]['ocr_regions'] = [dict(id='body', bbox=[0, 0, 10, 10], text='x', strict='false')]
        with self.assertRaises(ValueError):
            validate_manifest(data, self.root)
        data = copy.deepcopy(self.data)
        data['cases'][0]['tables'] = [dict(id='t', bbox=[0, 0, 10, 10], shape=[1, 1], rows=[[None]])]
        with self.assertRaises(ValueError):
            validate_manifest(data, self.root)

    def test_drafts_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, '没有已复核'):
            run_benchmark(self.data, self.root, self.root / 'result')
        self.assertFalse((self.root / 'result').exists())

    def test_runner_extracts_synthetic_pdf_and_never_overwrites(self):
        output = self.root / 'result'
        original = self.pdf.read_bytes()
        report = run_benchmark(self.data, self.root, output, include_drafts=True)
        self.assertEqual(report['summary']['cases_complete'], 1)
        self.assertEqual(report['reference_status'], 'provisional')
        self.assertEqual(report['summary']['tables']['false_positive'], 0)
        self.assertEqual(self.pdf.read_bytes(), original)
        saved = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['draft_cases'], 1)
        self.assertIn('不是已人工审核', markdown(report))
        with self.assertRaisesRegex(ValueError, '不覆盖'):
            run_benchmark(self.data, self.root, output, include_drafts=True)

    def test_wrong_sha_or_missing_pdf_is_input_error_not_accuracy_failure(self):
        self.data['cases'][0]['sha256'] = '0' * 64
        report = run_benchmark(self.data, self.root, self.root / 'mismatch', True)
        self.assertEqual(report['summary']['cases_error'], 1)
        self.assertIn('校验和不同', report['cases'][0]['error'])
        self.pdf.unlink()
        report = run_benchmark(self.data, self.root, self.root / 'missing', True)
        self.assertEqual(report['summary']['cases_complete'], 0)
        self.assertIsNone(report['summary']['tables']['recall'])

    def test_caught_ocr_environment_failure_not_counted_as_character_errors(self):
        self.data['cases'][0].pop('tables')
        self.data['cases'][0]['ocr_regions'] = [dict(id='body', bbox=[0, 0, 100, 100], text='text')]
        result = Result(str(self.pdf), 1, [1], warnings=['OCR 不可用: missing model'])
        result.blocks = [Block('image', 1, [0, 0, 200, 200], meta={'ocr_status': 'failed'})]
        with patch('scripts.evaluate_benchmark.extract', return_value=result):
            report = run_benchmark(self.data, self.root, self.root / 'ocr-error', True)
        self.assertEqual(report['summary']['cases_error'], 1)
        self.assertIn('OCR 环境/执行失败', report['cases'][0]['error'])
        self.assertEqual(report['summary']['ocr']['strict']['regions'], 0)
        self.assertIsNone(report['summary']['ocr']['strict']['cer'])


if __name__ == '__main__':
    unittest.main()
