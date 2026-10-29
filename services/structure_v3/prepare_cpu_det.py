"""Prepare the verified server det model; originals are preserved, rec is separate."""
import argparse
import json
from pathlib import Path

from prepare_cpu_rec import DET_MODEL, ROOT, prepare as _prepare


def prepare(source, output=None):
    return _prepare(source, output, model=DET_MODEL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'V3/models' / DET_MODEL)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    main()
