import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pdf_extractor import usage  # Loads the same shared module used by the worker.
import pdftransfer_usage as log


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        environment = patch.dict(os.environ, {'PDFTRANSFER_LOG_DIR': str(self.directory)})
        environment.start()
        self.addCleanup(environment.stop)

    def rows(self):
        return [json.loads(line) for path in self.directory.glob('*.jsonl*')
                for line in path.read_text(encoding='utf-8').splitlines()]

    def test_nested_exception_privacy_and_cpu(self):
        with log.span('convert', feature='reading.md', filename='private.pdf', password='secret') as root:
            with self.assertRaises(ValueError):
                with log.span('extract', format='md'):
                    raise ValueError('private document contents')
        child, parent = self.rows()
        self.assertEqual(child['parentId'], parent['spanId'])
        self.assertEqual(child['traceId'], root.state['traceId'])
        self.assertEqual(child['feature'], 'reading.md')
        self.assertEqual(child['status'], 'error')
        self.assertEqual(parent['status'], 'ok')
        self.assertEqual(child['errorType'], 'ValueError')
        self.assertNotIn('private', json.dumps(self.rows()))
        self.assertNotIn('secret', json.dumps(self.rows()))
        for row in self.rows():
            self.assertGreaterEqual(row['wallSeconds'], 0)
            self.assertGreaterEqual(row['processCpuSeconds'], 0)
            self.assertGreaterEqual(row['threadCpuSeconds'], 0)
            self.assertAlmostEqual(row['processCpuPercent'], 100 * row['processCpuSeconds'] / row['wallSeconds'])
        self.assertEqual(log.trace_headers(), {})

    def test_effective_runner_and_batch_shape_are_logged_without_document_data(self):
        with log.span('v3.model.predict.ocr', runnerRunMode='mkldnn',runnerCpuThreads=6,
                      runnerNewIr=True,ompNumThreads='6',mklNumThreads='6',
                      normalizedWidth=640,paddedPixels=30000,paddingRatio=1.2,
                      text='private document'):
            pass
        metadata=self.rows()[0]['metadata']
        self.assertEqual(metadata['runnerCpuThreads'],6)
        self.assertEqual(metadata['ompNumThreads'],'6')
        self.assertEqual(metadata['normalizedWidth'],640)
        self.assertNotIn('private',json.dumps(self.rows()))

    def test_threads_have_isolated_traces(self):
        def run(index):
            with log.span('thread', feature=f'feature.{index}'):
                with log.span('child'):
                    return log.trace_headers()['X-Pdftransfer-Trace']
        with ThreadPoolExecutor(max_workers=4) as pool:
            ids = list(pool.map(run, range(12)))
        self.assertEqual(len(set(ids)), 12)
        self.assertEqual(len(self.rows()), 24)
        for row in self.rows():
            self.assertEqual(sum(other['traceId'] == row['traceId'] for other in self.rows()), 2)

    def test_async_context_and_cancellation(self):
        async def job(index):
            with log.span('async', feature=f'feature.{index}'):
                await asyncio.sleep(0)
                return log.trace_headers()['X-Pdftransfer-Trace']
        async def main():
            ids = await asyncio.gather(job(1), job(2))
            self.assertNotEqual(*ids)
            with self.assertRaises(asyncio.CancelledError):
                with log.span('cancel'):
                    raise asyncio.CancelledError()
        asyncio.run(main())
        self.assertEqual(self.rows()[-1]['status'], 'error')

    def test_logging_failure_does_not_change_result(self):
        with patch.object(log, '_write', wraps=log._write), patch.object(Path, 'mkdir', side_effect=OSError('full')):
            @log.tracked('safe')
            def operation():
                return 42
            self.assertEqual(operation(), 42)
        self.assertEqual(self.rows(), [])

    def test_rotation_retention_and_partial_records(self):
        stale = self.directory / 'usage-2000-01-01-host-123.jsonl'
        stale.write_text('old', encoding='utf-8')
        os.utime(stale, (1, 1))
        with patch.object(log, 'MAX_BYTES', 1):
            for index in range(8):
                with log.span('rotate', feature='reading.html'):
                    pass
        self.assertFalse(stale.exists())
        self.assertEqual(len(list(self.directory.glob('*.jsonl*'))), 4)
        with next(self.directory.glob('*.jsonl')).open('a') as stream:
            stream.write('[]\n')
            stream.write('{incomplete')
        rows = log.summarize(self.directory)
        self.assertEqual(rows[0]['count'], 4)
        self.assertEqual(rows[0]['feature'], 'reading.html')
        self.assertEqual(log.summarize(self.directory, date='2000-01-01'), [])

    def test_remote_trace_and_summary_filter(self):
        remote = dict(traceId='a'*32, parentId='b'*32, feature='rebuild.pdf')
        with log.span('v3.worker', remote=remote) as worker:
            with log.span('v3.predict'):
                pass
            worker.metadata['cacheHit'] = False
        rows = self.rows()
        self.assertTrue(all(row['traceId'] == 'a'*32 for row in rows))
        self.assertEqual(rows[-1]['parentId'], 'b'*32)
        self.assertEqual(len(log.summarize(self.directory, trace='a'*32)), 2)
        self.assertEqual(log.summarize(self.directory, trace='unknown'), [])
        with log.span('worker', remote=dict(traceId='../../private', feature='bad/path')):
            pass
        self.assertNotIn('private', json.dumps(self.rows()))

    def test_models_keep_runtime_settings_separate_and_count_batch_items(self):
        for runtime, size in [('a1', 2), ('a1', 3), ('b2', 4)]:
            with log.span('v3.model.predict.ocr', feature='rebuild.md',
                          model='OCR', phase='predict', runtimeId=runtime,
                          cpuThreads=6, mkldnn=True, batchSize=size):
                pass
        with log.span('v3.predict', feature='rebuild.md'):
            pass
        groups = log.summarize(self.directory, models_only=True)
        self.assertEqual(len(groups), 2)
        self.assertEqual([(row['runtimeId'], row['count'], row['batchItemCount'])
                          for row in groups], [('a1', 2, 5), ('b2', 1, 4)])
        self.assertTrue(all(row['cpuThreads'] == 6 and row['mkldnn'] for row in groups))
        self.assertTrue(all(row['model'] == 'OCR' for row in groups))

