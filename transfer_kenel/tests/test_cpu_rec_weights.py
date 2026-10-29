"""Parameter integrity and explicit rec path/cache isolation; no Paddle needed."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'services/structure_v3'))
from prepare_cpu_rec import MANIFEST, normalize_parameters, prepare
from model_profile import ModelProfiler
from native_worker import NativeRuntime


def tensor(bits, descriptor=b'\x08\x05\x10\x08'):
    return struct.pack('<IQII', 0, 0, 0, len(descriptor)) + descriptor + struct.pack('<' + 'I' * len(bits), *bits)


class CPURecWeightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original'
        self.source.mkdir()
        self.output = self.root / 'derived'
        # +0, -0, smallest +subnormal, largest -subnormal, smallest normal,
        # ordinary negative, infinity and NaN: only entries 2/3 may change.
        self.bits = [0, 0x80000000, 1, 0x807fffff, 0x00800000, 0xbf800000, 0x7f800000, 0x7fc00001]
        (self.source / 'inference.pdiparams').write_bytes(tensor(self.bits))
        (self.source / 'inference.json').write_bytes(b'{"unchanged": true}\n')
        (self.source / 'inference.yml').write_text('Global:\n  model_name: PP-OCRv5_server_rec\n')

    def test_only_subnormal_parameter_bits_change_in_multiple_tensors(self):
        original = tensor(self.bits) + tensor(self.bits)
        changed, stats = normalize_parameters(original)
        self.assertEqual(len(changed), len(original))
        self.assertEqual(sum(t['subnormals'] for t in stats), 4)
        expected = self.bits.copy()
        expected[2:4] = [0, 0]
        self.assertEqual(changed, tensor(expected) + tensor(expected))

    def test_unknown_or_incomplete_serializations_fail(self):
        examples = [b'', tensor(self.bits)[:-1], tensor(self.bits) + b'\x00',
                    tensor(self.bits, b'\x08\x04\x10\x08'), tensor(self.bits, b'\x08\x05\x10\x00'),
                    tensor(self.bits, b'\x08\x05\x10\x80'), b'\x01' + tensor(self.bits)[1:]]
        for value in examples:
            with self.subTest(size=len(value)), self.assertRaises(ValueError):
                normalize_parameters(value)

    def test_preparation_is_traceable_reusable_and_preserves_original(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        result = prepare(self.source, self.output)
        self.assertFalse(result['reused'])
        self.assertEqual(result['zeroedSubnormals'], 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertEqual((self.output / 'inference.json').read_bytes(), before['inference.json'])
        self.assertEqual((self.output / 'inference.yml').read_bytes(), before['inference.yml'])
        self.assertEqual(json.loads((self.output / MANIFEST).read_text())['policy'], result['policy'])
        self.assertTrue(prepare(self.source, self.output)['reused'])

    def test_corrupt_output_or_changed_source_is_never_overwritten(self):
        prepare(self.source, self.output)
        (self.output / 'inference.pdiparams').write_bytes(b'corrupted')
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.output)
        other = self.root / 'other'
        prepare(self.source, other)
        (self.source / 'inference.json').write_text('{}')
        with self.assertRaises(FileExistsError):
            prepare(self.source, other)

    def test_overlap_and_unrelated_model_are_rejected(self):
        for target in [self.source, self.source / 'derived', self.root]:
            with self.assertRaises(ValueError):
                prepare(self.source, target)
        (self.source / 'inference.yml').write_text('Global:\n  model_name: other\n')
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_manifest_and_malformed_model_metadata_are_rejected(self):
        prepare(self.source, self.output)
        for malformed in [[], {'derivedHashes': []}, {'policy': 'other'}]:
            (self.output / MANIFEST).write_text(json.dumps(malformed))
            with self.assertRaises(FileExistsError):
                prepare(self.source, self.output)
        (self.source / 'inference.yml').write_text('Global: []\n')
        with self.assertRaises(ValueError):
            prepare(self.source, self.root / 'another')

    def test_existing_model_override_selects_rec_only_and_separates_result_cache(self):
        model_name = 'PP-OCRv5_server_rec'
        prepare(self.source, self.output)
        profiler = ModelProfiler(self.root, model_paths={model_name: self.output.name})
        self.assertEqual(profiler.resolve_model_dir(model_name), self.output.resolve())
        (self.root / 'other-model').mkdir()
        self.assertEqual(profiler.resolve_model_dir('other-model'), (self.root / 'other-model').resolve())
        config = dict(model_dir=str(self.root), results_dir=str(self.root / 'results'),
                      device='cpu', cpu_threads=6, enable_mkldnn=True)
        original = NativeRuntime(config)
        optimized = NativeRuntime(dict(config, model_paths={model_name: self.output.name}))
        self.assertNotEqual(original.cache_signature, optimized.cache_signature)
        self.assertNotEqual(original.runtime_id, optimized.runtime_id)


if __name__ == '__main__':
    unittest.main()
