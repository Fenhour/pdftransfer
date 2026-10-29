"""The native model hook times executed batches, not generator construction."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pdf_extractor import usage
from model_profile import ModelProfiler, collect_profiles


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def process(self, batch):
        self.calls += 1
        return list(batch)

    def __call__(self, batches):
        for batch in batches:
            yield self.process(batch)


class FakePipeline:
    def create_model(self, config, **kwargs):
        self.received = config
        return FakePredictor()


class ModelProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'model-A').mkdir()
        environment = patch.dict(os.environ, {'PDFTRANSFER_LOG_DIR': str(self.root / 'logs')})
        environment.start()
        self.addCleanup(environment.stop)
        self.profiler = ModelProfiler(self.root).install(FakePipeline)
        self.addCleanup(self.profiler.uninstall)
        self.config = dict(model_name='model-A', module_name='text_recognition')

    def test_factory_records_initialization_and_keeps_original_predictor_type(self):
        with collect_profiles() as rows, usage.span('convert', feature='reading.md'):
            owner = FakePipeline()
            predictor = owner.create_model(self.config)
        self.assertIsInstance(predictor, FakePredictor)
        self.assertEqual(owner.received['model_dir'], str((self.root / 'model-A').resolve()))
        self.assertNotIn('model_dir', self.config)
        self.assertEqual([row['phase'] for row in rows], ['initialize'])
        self.assertEqual(rows[0]['model'], 'model-A')
        records = [json.loads(line) for path in (self.root / 'logs').glob('*.jsonl')
                   for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(records[0]['traceId'], records[-1]['traceId'])
        self.assertEqual(records[0]['parentId'], records[-1]['spanId'])

    def test_lazy_prediction_records_each_actual_batch_and_excludes_consumer_pause(self):
        predictor = FakePipeline().create_model(self.config)
        with collect_profiles() as rows:
            iterator = predictor([[1, 2], [3]])
            self.assertEqual(rows, [])
            self.assertEqual(next(iterator), [1, 2])
            time.sleep(.04)
            self.assertEqual(next(iterator), [3])
        self.assertEqual(predictor.calls, 2)
        self.assertEqual([row['batchSize'] for row in rows], [2, 1])
        self.assertTrue(all(row['phase'] == 'predict' for row in rows))
        self.assertLess(sum(row['wallSeconds'] for row in rows), .035)
        for row in rows:
            self.assertGreaterEqual(row['wallSeconds'], 0)
            self.assertGreaterEqual(row['processCpuSeconds'], 0)
            self.assertGreaterEqual(row['threadCpuSeconds'], 0)

    def test_missing_model_rejected_before_factory_without_download(self):
        with self.assertRaisesRegex(FileNotFoundError, 'automatic download is disabled'):
            FakePipeline().create_model(dict(model_name='absent', module_name='layout'))

    def test_actual_runner_settings_are_separate_from_requested_switches(self):
        predictor=FakePredictor()
        predictor.runner=SimpleNamespace(_config=dict(run_mode='paddle',cpu_threads=6,enable_new_ir=False))
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6'}):
            self.profiler.instrument(predictor,self.config,'ActualPipeline')
        with collect_profiles() as rows:
            predictor.process([1])
        self.assertEqual(rows[0]['runnerRunMode'],'paddle')
        self.assertEqual(rows[0]['runnerCpuThreads'],6)
        self.assertFalse(rows[0]['runnerNewIr'])
        self.assertEqual(rows[0]['ompNumThreads'],'6')

    def test_failure_is_attributed_to_actual_model_batch(self):
        predictor = FakePipeline().create_model(self.config)
        with collect_profiles() as rows:
            with self.assertRaises(TypeError):
                predictor.process(None)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'error')
        self.assertEqual(rows[0]['branch'], 'text_recognition')

    def test_lazy_future_model_initialization_remains_instrumented(self):
        with collect_profiles() as first:
            FakePipeline().create_model(self.config)
        with collect_profiles() as second:
            predictor = FakePipeline().create_model(self.config)
            predictor.process([9])
        self.assertEqual([row['phase'] for row in first], ['initialize'])
        self.assertEqual([row['phase'] for row in second], ['initialize', 'predict'])

    def test_hook_is_idempotent_and_restores_factory(self):
        original = self.profiler._original
        self.assertIs(self.profiler.install(FakePipeline), self.profiler)
        self.profiler.uninstall()
        self.assertIs(FakePipeline.create_model, original)

    def test_effective_runtime_configuration_labels_each_model_phase(self):
        self.profiler.uninstall()
        self.profiler = ModelProfiler(self.root, runtime_id='1234567890abcdef',
                                      cpu_threads=3, mkldnn=False).install(FakePipeline)
        self.addCleanup(self.profiler.uninstall)
        with collect_profiles() as rows:
            predictor = FakePipeline().create_model(self.config)
            predictor.process([1, 2])
        self.assertEqual([row['phase'] for row in rows], ['initialize', 'predict'])
        for row in rows:
            self.assertEqual(row['runtimeId'], '1234567890abcdef')
            self.assertEqual(row['cpuThreads'], 3)
            self.assertFalse(row['mkldnn'])
        logged = [json.loads(line) for path in (self.root / 'logs').glob('*.jsonl')
                  for line in path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(logged), 2)
        for row in logged:
            self.assertEqual(row['metadata']['runtimeId'], '1234567890abcdef')
            self.assertEqual(row['metadata']['cpuThreads'], 3)
            self.assertFalse(row['metadata']['mkldnn'])

