"""Standard-library usage spans shared by both applications and the V3 worker.

Process CPU deltas include concurrent threads, but exclude child processes.
Nested spans are inclusive: never sum them as independent request costs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

_current = ContextVar('pdftransfer_usage', default=None)
_lock = threading.RLock()
_cleaned = set()
_SAFE = re.compile(r'^[a-zA-Z0-9_.:-]{1,96}$')
_META = {'format', 'backend', 'ocr', 'mode', 'page', 'pages', 'cacheHit', 'httpStatus',
         'model', 'branch', 'pipeline', 'batchSize', 'phase', 'device',
         'runtimeId', 'cpuThreads', 'mkldnn', 'runnerRunMode', 'runnerCpuThreads',
         'runnerNewIr', 'ompNumThreads', 'mklNumThreads', 'normalizedWidth',
         'paddedPixels', 'paddingRatio'}
MAX_BYTES = 5 * 1024 * 1024
BACKUPS = 3
RETENTION_DAYS = 14


def log_dir():
    return Path(os.environ.get('PDFTRANSFER_LOG_DIR') or Path(__file__).resolve().parents[2] / 'log')


def _token(value, fallback):
    return value if isinstance(value, str) and _SAFE.fullmatch(value) else fallback


def _metadata(values):
    # Explicit allowlist; never serialize arguments, paths, document text or errors.
    return {k: v for k, v in values.items() if k in _META and
            (v is None or isinstance(v, (bool, int, float)) or
             isinstance(v, str) and len(v) <= 64 and re.fullmatch(r'[\w.,: -]*', v))}


def trace_headers():
    state = _current.get()
    if not state:
        return {}
    return {'X-Pdftransfer-Trace': state['traceId'], 'X-Pdftransfer-Parent': state['spanId'],
            'X-Pdftransfer-Feature': state['feature']}


def _write(record):
    try:
        directory = log_dir()
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        role = _token(os.environ.get('PDFTRANSFER_LOG_ROLE'), 'host')
        target = directory / f'usage-{day}-{role}-{os.getpid()}.jsonl'
        data = json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n'
        with _lock:
            directory.mkdir(parents=True, exist_ok=True)
            cleanup_key = (str(directory), day)
            if cleanup_key not in _cleaned:
                cutoff = time.time() - RETENTION_DAYS * 86400
                for path in directory.glob('usage-*.jsonl*'):
                    try:
                        if path.is_file() and path.stat().st_mtime < cutoff:
                            path.unlink()
                    except OSError:
                        pass
                _cleaned.add(cleanup_key)
            if target.exists() and target.stat().st_size + len(data.encode('utf-8')) > MAX_BYTES:
                for index in range(BACKUPS, 0, -1):
                    previous = target if index == 1 else target.with_name(target.name + f'.{index-1}')
                    if previous.exists():
                        previous.replace(target.with_name(target.name + f'.{index}'))
            with target.open('a', encoding='utf-8') as stream:
                stream.write(data)
    except Exception:
        # Diagnostics must never break conversion, including full/unwritable disks.
        pass


class span:
    def __init__(self, module, *, feature=None, remote=None, **metadata):
        self.module = module
        self.feature = feature
        self.remote = remote or {}
        self.metadata = _metadata(metadata)

    def __enter__(self):
        parent = _current.get()
        self.state = dict(traceId=parent['traceId'] if parent else
                          _token(self.remote.get('traceId'), uuid.uuid4().hex),
                          spanId=uuid.uuid4().hex,
                          parentId=parent['spanId'] if parent else _token(self.remote.get('parentId'), None),
                          feature=_token(self.feature, None) or (parent['feature'] if parent else
                                  _token(self.remote.get('feature'), 'standalone')))
        self.binding = _current.set(self.state)
        self.started = datetime.now(timezone.utc).isoformat()
        self.wall, self.cpu, self.thread = time.perf_counter(), time.process_time(), time.thread_time()
        self.status = 'ok'
        return self

    def __exit__(self, kind, error, traceback):
        wall = max(0., time.perf_counter() - self.wall)
        cpu = max(0., time.process_time() - self.cpu)
        thread = max(0., time.thread_time() - self.thread)
        _current.reset(self.binding)
        record = dict(schema='pdftransfer.usage/1', **self.state, module=self.module,
                      startedAt=self.started, pid=os.getpid(), threadId=threading.get_ident(),
                      role=_token(os.environ.get('PDFTRANSFER_LOG_ROLE'), 'host'),
                      status='error' if kind else self.status,
                      wallSeconds=wall, processCpuSeconds=cpu, threadCpuSeconds=thread,
                      processCpuPercent=100 * cpu / wall if wall else 0.,
                      threadCpuPercent=100 * thread / wall if wall else 0.,
                      cpuScope='process-delta-includes-concurrent-threads-excludes-children',
                      inclusive=True, metadata=self.metadata)
        if kind:
            record['errorType'] = kind.__name__
        _write(record)
        return False


def tracked(module, *, feature=None):
    """Instrument synchronous business functions without changing their signature."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            values = signature.bind_partial(*args, **kwargs).arguments
            metadata = {k: v for k, v in values.items() if k in _META}
            page = values.get('page')
            if page is not None and not isinstance(page, (int, str)):
                metadata['page'] = getattr(page, 'number', -1) + 1
            elif 'index' in values and isinstance(values['index'], int):
                metadata['page'] = values['index'] + 1
            if 'recognition_backend' in values:
                metadata['backend'] = values['recognition_backend']
            options = values.get('options')
            if options is not None:
                metadata.update(format=getattr(options, 'output_format', None),
                                backend=getattr(options, 'recognition_backend', None),
                                ocr=getattr(options, 'ocr', None))
            selected = feature(values) if callable(feature) else feature
            with span(module, feature=selected, **metadata):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def summarize(directory=None, *, date=None, trace=None, models_only=False):
    groups = defaultdict(lambda: dict(count=0, errors=0, wallSeconds=0., processCpuSeconds=0., threadCpuSeconds=0.))
    for path in sorted(Path(directory or log_dir()).glob('usage-*.jsonl*')):
        if date and not path.name.startswith('usage-' + date + '-'):
            continue
        try:
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get('schema') != 'pdftransfer.usage/1' or trace and row.get('traceId') != trace:
                            continue
                        if models_only and not row.get('module', '').startswith('v3.model.'):
                            continue
                        metadata = row.get('metadata', {})
                        # Keep different native runtime settings separate when
                        # comparing model costs; historical rows remain valid.
                        runtime = tuple(str(metadata.get(field, '')) for field in
                                        ('runtimeId', 'cpuThreads', 'mkldnn'))
                        key = (row['feature'], row['module'], row['role'], *runtime)
                        group = groups[key]
                        for field in ('model', 'branch', 'pipeline', 'phase', 'device',
                                      'runtimeId', 'cpuThreads', 'mkldnn'):
                            if field in metadata:
                                group[field] = metadata[field]
                        batch_size = metadata.get('batchSize')
                        if type(batch_size) is int and batch_size >= 0:
                            group['batchItemCount'] = group.get('batchItemCount', 0) + batch_size
                            group['batchesWithSize'] = group.get('batchesWithSize', 0) + 1
                        group['count'] += 1
                        group['errors'] += row['status'] != 'ok'
                        for field in ('wallSeconds', 'processCpuSeconds', 'threadCpuSeconds'):
                            group[field] += row[field]
                    except (ValueError, KeyError, TypeError):
                        continue  # A process killed during append may leave an incomplete line.
        except OSError:
            continue
    result = []
    for (feature, module, role, *_runtime), group in sorted(groups.items()):
        wall = group['wallSeconds']
        result.append(dict(feature=feature, module=module, role=role, **group,
                           meanWallSeconds=wall / group['count'],
                           meanProcessCpuPercent=100 * group['processCpuSeconds'] / wall if wall else 0.))
    return result


def main():
    parser = argparse.ArgumentParser(description='按功能、模块、进程角色汇总使用日志（嵌套耗时不可相加）')
    parser.add_argument('--date', help='UTC 日期，例如 2026-10-26')
    parser.add_argument('--trace', help='仅统计一次请求的 traceId')
    parser.add_argument('--json', action='store_true', help='输出 JSON，支持重定向保存')
    parser.add_argument('--models', action='store_true', help='仅汇总逐模型初始化与实际处理批次')
    parser.add_argument('--output', type=Path, help='将 JSON 汇总保存为 UTF-8 文件')
    args = parser.parse_args()
    rows = summarize(date=args.date, trace=args.trace, models_only=args.models)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print('feature / module / role | count errors | total(s) mean(s) | CPU(s) CPU%(1 core)')
        for row in rows:
            if row.get('runtimeId'):
                print(f"runtime={row['runtimeId']} cpu_threads={row.get('cpuThreads')} "
                      f"mkldnn={row.get('mkldnn')} batch_items={row.get('batchItemCount', 0)}")
            print(f"{row['feature']} / {row['module']} / {row['role']} | "
                  f"{row['count']} {row['errors']} | {row['wallSeconds']:.3f} {row['meanWallSeconds']:.3f} | "
                  f"{row['processCpuSeconds']:.3f} {row['meanProcessCpuPercent']:.1f}")


if __name__ == '__main__':
    main()
