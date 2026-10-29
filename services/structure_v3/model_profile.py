"""Measure the real PaddleX model batch executor, without editing PaddleX files.

PaddleX 3.7.2 predictor ``__call__``/``apply`` return lazy iterators.  Its
``process`` method performs each batch's preprocessing, runner inference and
postprocessing synchronously.  Wrapping that method therefore measures work
when the iterator is consumed and excludes time spent in downstream consumers.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import os
import re
import time

from pdftransfer_usage import span


_records = ContextVar('pdftransfer_model_records', default=None)


class LocalModelUnavailable(FileNotFoundError):
    """Safe to return to callers: contains a model identifier, never a path."""


def _label(value):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(value))[:96] or 'unknown'


@contextmanager
def collect_profiles():
    """Collect only this request's executed phases; cache hits remain empty."""
    rows = []
    token = _records.set(rows)
    try:
        yield rows
    finally:
        _records.reset(token)


@contextmanager
def _measure(phase, *, model, branch, pipeline, device, batch_size=None, runtime=None):
    metadata = dict(model=model, branch=branch, pipeline=pipeline,
                    phase=phase, device=device)
    metadata.update(runtime or {})
    if batch_size is not None:
        metadata['batchSize'] = batch_size
    module = '.'.join(('v3.model', phase, pipeline, branch, model))
    with span(module, **metadata):
        started = time.perf_counter(), time.process_time(), time.thread_time()
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'error'
            raise
        finally:
            rows = _records.get()
            if rows is not None:
                wall, cpu, thread = (max(0., clock() - initial) for clock, initial in
                    zip((time.perf_counter, time.process_time, time.thread_time), started))
                rows.append(dict(**metadata, status=status, wallSeconds=wall,
                                 processCpuSeconds=cpu, threadCpuSeconds=thread,
                                 processCpuPercent=100 * cpu / wall if wall else 0.,
                                 threadCpuPercent=100 * thread / wall if wall else 0.,
                                 measurementScope='model-batch-preprocess-inference-postprocess'
                                 if phase == 'predict' else 'model-construction'))


def _batch_size(batch):
    try:
        return len(getattr(batch, 'instances', batch))
    except (TypeError, AttributeError):
        return None


class ModelProfiler:
    """Install one factory hook in the dedicated, serialized native worker.

    Predictors retain their original identity and type.  The hook also remains
    active for lazily created table orientation / OCR models.  No model is
    disabled and no configured local model directory is downloaded implicitly.
    """
    def __init__(self, model_dir, *, device='cpu', model_paths=None,
                 runtime_id=None, cpu_threads=None, mkldnn=None):
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        self.model_paths = dict(model_paths or {})
        self.runtime = {key: value for key, value in
                        (('runtimeId', runtime_id), ('cpuThreads', cpu_threads), ('mkldnn', mkldnn))
                        if value is not None}
        self._base = self._original = self._wrapper = None

    def resolve_model_dir(self, model_name, configured=None):
        selected = self.model_paths.get(model_name) or configured
        candidate = Path(selected) if selected else self.model_dir / model_name
        if not candidate.is_absolute():
            candidate = self.model_dir / candidate
        if not candidate.is_dir():
            # The message deliberately omits paths and input document content.
            raise LocalModelUnavailable(f'Local model unavailable: {_label(model_name)}; automatic download is disabled')
        return candidate.resolve()

    def instrument(self, predictor, config, pipeline):
        if getattr(predictor, '_pdftransfer_profiled', False):
            return predictor
        execute = getattr(predictor, 'process', None)
        if not callable(execute):
            raise RuntimeError('Unsupported PaddleX predictor: synchronous process executor is required')
        model = _label(config.get('model_name', getattr(predictor, 'model_name', 'unknown')))
        branch = _label(config.get('module_name', 'unknown'))
        pipeline = _label(pipeline)
        # These come from the constructed runner rather than the requested
        # top-level switches (individual models can disable oneDNN).
        actual = getattr(getattr(predictor, 'runner', None), '_config', {}) or {}
        if not isinstance(actual, dict):
            actual = {}
        runtime = dict(self.runtime, ompNumThreads=os.environ.get('OMP_NUM_THREADS'),
                       mklNumThreads=os.environ.get('MKL_NUM_THREADS'))
        for source, target in (('run_mode','runnerRunMode'), ('cpu_threads','runnerCpuThreads'),
                               ('enable_new_ir','runnerNewIr')):
            if isinstance(actual.get(source), (str, int, bool)):
                runtime[target] = actual[source]

        @wraps(execute)
        def process(*args, **kwargs):
            batch = args[0] if args else kwargs.get('batch_data')
            with _measure('predict', model=model, branch=branch, pipeline=pipeline,
                          device=self.device, batch_size=_batch_size(batch), runtime=runtime):
                return execute(*args, **kwargs)

        predictor.process = process
        predictor._pdftransfer_profiled = True
        return predictor

    def install(self, base_class=None):
        if self._base is not None:
            return self
        if base_class is None:
            from paddlex.inference.pipelines.base import BasePipeline
            base_class = BasePipeline
        original = base_class.create_model
        if getattr(original, '_pdftransfer_profile_hook', False):
            raise RuntimeError('A PaddleX model profiler is already installed')

        @wraps(original)
        def create_model(owner, config, **kwargs):
            config = dict(config)
            model = _label(config.get('model_name', 'unknown'))
            branch = _label(config.get('module_name', 'unknown'))
            pipeline = _label(type(owner).__name__.lstrip('_'))
            # Supply every path explicitly, including lazily initialized models.
            # Do this before calling PaddleX's factory to forbid network fallback.
            config['model_dir'] = str(self.resolve_model_dir(config.get('model_name', ''), config.get('model_dir')))
            with _measure('initialize', model=model, branch=branch,
                          pipeline=pipeline, device=self.device, runtime=self.runtime):
                predictor = original(owner, config, **kwargs)
            return self.instrument(predictor, config, pipeline)

        create_model._pdftransfer_profile_hook = True
        self._base, self._original, self._wrapper = base_class, original, create_model
        base_class.create_model = create_model
        return self

    def uninstall(self):
        if self._base is not None and self._base.create_model is self._wrapper:
            self._base.create_model = self._original
        self._base = self._original = self._wrapper = None

