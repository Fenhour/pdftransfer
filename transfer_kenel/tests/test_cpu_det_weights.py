"""Det identity, traceability and combined det/rec cache selection."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services/structure_v3'))
from prepare_cpu_det import prepare
from prepare_cpu_rec import DET_MODEL, MODEL, MANIFEST, prepare as prepare_rec
from model_profile import ModelProfiler
from native_worker import NativeRuntime


class CPUDetWeightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / DET_MODEL
        self.source.mkdir()
        (self.source / 'inference.yml').write_text(f'Global:\n  model_name: {DET_MODEL}\n')
        (self.source / 'inference.json').write_text('{}')
        (self.source / 'inference.pdiparams').write_bytes(
            struct.pack('<IQII', 0, 0, 0, 4) + b'\x08\x05\x10\x02' + struct.pack('<II', 1, 0x00800000))

    def test_det_preparation_preserves_original_and_reuses_only_intact_identity(self):
        original = (self.source / 'inference.pdiparams').read_bytes()
        result = prepare(self.source)
        self.assertEqual(result['model'], DET_MODEL)
        self.assertEqual(result['policy'], 'det-fp32-zero-subnormals-v1')
        self.assertEqual(result['zeroedSubnormals'], 1)
        output = Path(result['directory'])
        self.assertTrue(output.name.startswith(DET_MODEL + '_cpu_fp32_zero_subnormals_v1_'))
        self.assertEqual((self.source / 'inference.pdiparams').read_bytes(), original)
        self.assertEqual((output / 'inference.pdiparams').read_bytes()[-8:], struct.pack('<II', 0, 0x00800000))
        self.assertTrue(prepare(self.source)['reused'])
        report = json.loads((output / MANIFEST).read_text())
        report['model'] = MODEL
        (output / MANIFEST).write_text(json.dumps(report))
        with self.assertRaises(FileExistsError):
            prepare(self.source)

    def test_rec_and_det_metadata_cannot_be_interchanged(self):
        with self.assertRaises(ValueError):
            prepare_rec(self.source)
        (self.source / 'inference.yml').write_text(f'Global:\n  model_name: {MODEL}\n')
        with self.assertRaises(ValueError):
            prepare(self.source)

    def test_combined_mapping_preserves_rec_and_changes_runtime_cache(self):
        det = Path(prepare(self.source)['directory'])
        rec = self.root / 'rec-derived'
        rec.mkdir()
        config = dict(model_dir=str(self.root), results_dir=str(self.root / 'results'),
                      cpu_threads=6, device='cpu', enable_mkldnn=True, model_paths={MODEL: rec.name})
        before = NativeRuntime(config)
        paths = dict(config['model_paths'], **{DET_MODEL: det.name})
        after = NativeRuntime(dict(config, model_paths=paths))
        profiler = ModelProfiler(self.root, model_paths=paths)
        self.assertEqual(profiler.resolve_model_dir(DET_MODEL), det)
        self.assertEqual(profiler.resolve_model_dir(MODEL), rec)
        self.assertNotEqual(before.runtime_id, after.runtime_id)
        self.assertNotEqual(before.cache_signature, after.cache_signature)
