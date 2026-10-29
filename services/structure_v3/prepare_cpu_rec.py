"""Prepare a traceable FP32 OCR copy with subnormal parameters flushed to zero.

This is an offline, one-time operation. It never overwrites source weights or
changes process floating point modes. Only the verified Paddle LoD-free FP32
parameter format is accepted; unknown formats fail before publishing output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

import numpy as np


POLICY = 'rec-fp32-zero-subnormals-v1'
MODEL = 'PP-OCRv5_server_rec'
DET_MODEL = 'PP-OCRv5_server_det'
FILES = ('inference.json', 'inference.pdiparams', 'inference.yml')
MANIFEST = 'cpu-optimization.json'
ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _varint(data, offset):
    result = shift = 0
    while offset < len(data) and shift < 70:
        value = data[offset]
        offset += 1
        result |= (value & 127) << shift
        if value < 128:
            return result, offset
        shift += 7
    raise ValueError('Invalid Paddle tensor descriptor varint')


def normalize_parameters(original):
    """Change only exponent-zero, nonzero-mantissa FP32 parameters to +0."""
    cleaned = bytearray(original)
    offset = 0
    tensors = []
    while offset < len(original):
        if len(original) - offset < 20:
            raise ValueError('Truncated Paddle tensor header')
        version, lod, tensor_version, length = struct.unpack_from('<IQII', original, offset)
        offset += 20
        if version != 0 or lod != 0 or tensor_version != 0:
            raise ValueError('Only version-zero, LoD-free Paddle tensors are supported')
        if not 0 < length <= 1024 or offset + length > len(original):
            raise ValueError('Invalid Paddle tensor descriptor length')
        descriptor = original[offset:offset + length]
        offset += length
        index = 0
        dtype = None
        dimensions = []
        while index < len(descriptor):
            key, index = _varint(descriptor, index)
            value, index = _varint(descriptor, index)
            if key == 8 and dtype is None:
                dtype = value
            elif key == 16 and value > 0:
                dimensions.append(value)
            else:
                raise ValueError('Unsupported Paddle tensor descriptor field')
        if dtype != 5 or not dimensions:
            raise ValueError('Only FP32 parameter tensors are supported')
        count = math.prod(dimensions)
        end = offset + count * 4
        if end > len(original):
            raise ValueError('Truncated Paddle FP32 tensor data')
        values = np.frombuffer(cleaned, dtype='<u4', count=count, offset=offset)
        mask = ((values & 0x7f800000) == 0) & ((values & 0x007fffff) != 0)
        changed = int(mask.sum())
        values[mask] = 0
        tensors.append(dict(index=len(tensors), dimensions=dimensions, count=count,
                            subnormals=changed, dataOffset=offset, dataEnd=end))
        offset = end
    if not tensors:
        raise ValueError('Empty Paddle parameter file')
    return bytes(cleaned), tensors


def prepare(source, output=None, *, model=MODEL):
    if model not in (MODEL, DET_MODEL):
        raise ValueError('Only the verified server rec/det models are supported')
    kind = 'rec' if model == MODEL else 'det'
    policy = f'{kind}-fp32-zero-subnormals-v1'
    source = Path(source).resolve()
    contents = {name: (source / name).read_bytes() for name in FILES}
    # Require the requested identity; rec and det cannot be interchanged.
    import yaml
    metadata = yaml.safe_load(contents['inference.yml'])
    global_metadata = metadata.get('Global') if isinstance(metadata, dict) else None
    if not isinstance(global_metadata, dict) or global_metadata.get('model_name') != model:
        raise ValueError(f'Expected model metadata for {model}')
    original_hashes = {name: digest(data) for name, data in contents.items()}
    output = Path(output).resolve() if output else source.with_name(
        model + '_cpu_fp32_zero_subnormals_v1_' + original_hashes['inference.pdiparams'][:12])
    if (output == source or output.is_relative_to(source) or source.is_relative_to(output)):
        raise ValueError('Source and derived model directories must be separate siblings')
    if output.exists():
        # Repeated setup is cheap; refuse stale, modified or unrelated directories.
        try:
            report = json.loads((output / MANIFEST).read_text(encoding='utf-8'))
            intact = (isinstance(report, dict) and report.get('policy') == policy
                      and report.get('model') == model
                      and report.get('sourceHashes') == original_hashes
                      and isinstance(report.get('derivedHashes'), dict)
                      and set(report['derivedHashes']) == set(FILES)
                      and all(digest((output / name).read_bytes()) == value
                              for name, value in report['derivedHashes'].items()))
        except (OSError, ValueError, KeyError, TypeError):
            intact = False
        if not intact:
            raise FileExistsError('Derived model exists but its manifest or file hashes differ; use a new output directory')
        return dict(report, directory=str(output), reused=True)
    normalized, tensors = normalize_parameters(contents['inference.pdiparams'])
    derived = dict(contents, **{'inference.pdiparams': normalized})
    report = dict(schema=f'pdftransfer.cpu-{kind}-optimization/1', policy=policy,
                  model=model, preparedAt=datetime.now(timezone.utc).isoformat(),
                  sourceHashes=original_hashes,
                  derivedHashes={name: digest(data) for name, data in derived.items()},
                  fp32Count=sum(t['count'] for t in tensors),
                  zeroedSubnormals=sum(t['subnormals'] for t in tensors),
                  absoluteThresholdExclusive='0x1p-126', tensors=tensors)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish all three files plus lineage together; no partially built model path.
    with tempfile.TemporaryDirectory(prefix=f'.cpu-{kind}-', dir=output.parent) as temporary:
        staging = Path(temporary) / 'model'
        staging.mkdir()
        for name, data in derived.items():
            (staging / name).write_bytes(data)
        (staging / MANIFEST).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        if any(digest((source / name).read_bytes()) != value for name, value in original_hashes.items()):
            raise RuntimeError('Source model changed during preparation; no derived model published')
        if output.exists():
            raise FileExistsError('Derived output appeared during preparation; refusing overwrite')
        staging.rename(output)
    return dict(report, directory=str(output), reused=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'V3/models' / MODEL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
