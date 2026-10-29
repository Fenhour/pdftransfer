"""Bounded, process-local progress for synchronous local conversion requests."""
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
import uuid

_task = ContextVar('operation_progress', default=None)
_bounds = ContextVar('operation_progress_bounds', default=(0., 100.))
_page = ContextVar('operation_progress_page', default=None)


class ProgressStore:
    def __init__(self, limit=256, ttl=3600):
        self.limit, self.ttl = limit, ttl
        self._items = {}
        self._lock = threading.Lock()

    def create(self):
        with self._lock:
            now = time.monotonic()
            expired = [key for key, item in self._items.items()
                       if item['status'] != 'running' and now-item['_updated'] > self.ttl]
            for key in expired:
                del self._items[key]
            if len(self._items) >= self.limit:
                removable = [key for key, item in self._items.items() if item['status'] in {'completed', 'failed'}]
                if not removable:
                    raise ValueError('进度任务较多，请稍后重试')
                del self._items[removable[0]]
            key = uuid.uuid4().hex
            self._items[key] = dict(id=key, status='waiting', percent=0., stage='等待请求',
                                    message='', current=None, total=None, _updated=now)
            return key

    def get(self, key):
        with self._lock:
            item = self._items.get(key)
            return {k: v for k, v in item.items() if not k.startswith('_')} if item else None

    def begin(self, key):
        with self._lock:
            item = self._items.get(key)
            if item is None or item['status'] != 'waiting':
                raise ValueError('进度任务不存在或已被使用，请重新发起操作')
            item.update(status='running', stage='接收请求', _updated=time.monotonic())

    def update(self, key, percent, stage, message='', current=None, total=None):
        with self._lock:
            item = self._items.get(key)
            if item and item['status'] == 'running':
                item.update(percent=round(max(item['percent'], min(100., percent)), 2), stage=stage,
                            message=message, current=current, total=total, _updated=time.monotonic())

    def finish(self, key, success):
        with self._lock:
            item = self._items.get(key)
            if item and item['status'] == 'running':
                item.update(status='completed' if success else 'failed',
                            stage='结果已生成' if success else '处理失败', _updated=time.monotonic())
                if success:
                    item['percent'] = 100.


store = ProgressStore()


@contextmanager
def bind_progress(key):
    token = _task.set(key)
    bounds = _bounds.set((0., 100.))
    page = _page.set(None)
    try:
        yield
    finally:
        _page.reset(page)
        _bounds.reset(bounds)
        _task.reset(token)


@contextmanager
def progress_span(start, end, page=None):
    low, high = _bounds.get()
    token = _bounds.set((low+(high-low)*start/100, low+(high-low)*end/100))
    page_token = _page.set(page) if page is not None else None
    try:
        yield
    finally:
        _bounds.reset(token)
        if page_token is not None:
            _page.reset(page_token)


def report(percent, stage, message='', current=None, total=None):
    key = _task.get()
    if key:
        low, high = _bounds.get()
        page = _page.get()
        if page:
            message = f'第 {page[0]} / {page[1]} 页；'+message
        store.update(key, low+(high-low)*max(0., min(100., percent))/100,
                     stage, message, current, total)
