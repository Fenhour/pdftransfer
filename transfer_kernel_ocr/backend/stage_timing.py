"""Request-local reconstruction timings, mirrored to existing usage spans."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

from .usage import span

_session = ContextVar('rebuild_stage_timing', default=None)
_stack = ContextVar('rebuild_stage_stack', default=())
LABELS = {'05':'块构建与内部排版', '06':'文字包络与块适配',
          '07':'块间碰撞调整', '08':'Region边界适配', 'O1':'纯空带调整'}


def capture_timings(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        binding=_session.set([])
        stack_binding=_stack.set(())
        try:
            return function(*args, **kwargs)
        finally:
            _stack.reset(stack_binding)
            _session.reset(binding)
    return wrapped


@contextmanager
def stage_span(step, phase, **metadata):
    record=dict(step=step,phase=phase,**metadata,status='ok')
    parent_stack=_stack.get()
    record['_child_wall']=record['_child_cpu']=0.
    binding=_stack.set(parent_stack+(record,))
    wall,cpu=time.perf_counter(),time.process_time()
    try:
        log_phase=(f"{phase}-a{metadata['attempt']}-r{metadata['iteration']}"
                   if 'attempt' in metadata and 'iteration' in metadata else phase)
        with span('rebuild.step'+step+'.'+phase,phase=log_phase):
            yield record
    except BaseException:
        record['status']='error'
        raise
    finally:
        record['wallSeconds']=max(0.,time.perf_counter()-wall)
        record['processCpuSeconds']=max(0.,time.process_time()-cpu)
        record['selfWallSeconds']=max(0.,record['wallSeconds']-record.pop('_child_wall'))
        record['selfProcessCpuSeconds']=max(0.,record['processCpuSeconds']-record.pop('_child_cpu'))
        _stack.reset(binding)
        if parent_stack:
            parent_stack[-1]['_child_wall']+=record['wallSeconds']
            parent_stack[-1]['_child_cpu']+=record['processCpuSeconds']
        records=_session.get()
        if records is not None:
            records.append(record)


def timed_step(step,phase):
    def decorate(function):
        @wraps(function)
        def wrapped(*args,**kwargs):
            with stage_span(step,phase):
                return function(*args,**kwargs)
        return wrapped
    return decorate


def timing_summary():
    events=[dict(r) for r in (_session.get() or [])]
    stages=[]
    for step,label in LABELS.items():
        rows=[r for r in events if r['step']==step]
        stages.append(dict(step=step,label=label,calls=len(rows),
            wallSeconds=sum(r['selfWallSeconds'] for r in rows),
            inclusiveWallSeconds=sum(r['wallSeconds'] for r in rows),
            processCpuSeconds=sum(r['selfProcessCpuSeconds'] for r in rows),
            errorCount=sum(r['status']=='error' for r in rows),
            status='recorded' if rows else 'not-executed'))
    return dict(schema='rebuild-stage-timing/1',unit='seconds',stages=stages,events=events,
                scope='all-layout-attempts-including-failed-region-retries-before-export',
                aggregation='wallSeconds-excludes-nested-stage-time; inclusiveWallSeconds-must-not-be-added')
